"""Conversational RAG pipeline for NG12 guideline Q&A."""

import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from app.models import ChatMessage, ChatResponse, Citation
from app.rag import query_guidelines_text

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
SIMILARITY_THRESHOLD = 1.2  # ChromaDB cosine distance: 0 = identical, 2 = opposite

_sessions: dict[str, list[ChatMessage]] = {}

SYSTEM_PROMPT = """You are an NG12 Clinical Knowledge Assistant. Your sole purpose is to
answer questions about the NICE NG12 guidelines ("Suspected cancer: recognition and referral")
using ONLY the retrieved guideline passages provided below.

## CRITICAL OUTPUT RULES:
- Your ENTIRE response must be a single valid JSON object. No text before or after the JSON.
- Do NOT repeat the retrieved passages verbatim. SYNTHESIZE and SUMMARIZE the information
  in your own words, organized clearly for the reader.
- When the user asks to "summarize", provide a concise, well-structured summary with
  key points — do NOT copy-paste the raw guideline text.

## Handling Greetings & Non-Clinical Messages:
If the user sends a greeting or non-clinical message, respond with:
{"answer": "Hi there! I'm the NG12 Clinical Knowledge Assistant. I can help you with questions about the NICE NG12 guidelines on suspected cancer recognition and referral.", "citations": []}

## Strict Rules:
1. ONLY use information from the RETRIEVED CONTEXT below. Never use your own knowledge.
2. ALWAYS include citations for every clinical statement — use the chunk_id, page number,
   and a short excerpt (1-2 sentences max) from the retrieved passage.
3. NEVER invent or guess:
   - Age thresholds (e.g., "refer if over 40") unless the retrieved text explicitly states them.
   - Investigation intervals or timelines not found in the retrieved text.
   - Referral criteria not present in the retrieved text.
4. NEVER reference documents other than NG12.
5. If the retrieved context does not contain enough information to answer the question,
   you MUST say: "I couldn't find clear support in the NG12 guidelines for that question."
   and return an empty citations list.
6. When the user asks a follow-up, use the conversation history for context but still
   ground your answer in the retrieved guideline passages.

## Output Format:
Respond with ONLY this JSON (no markdown, no extra text):
{
  "answer": "<your synthesized, well-structured answer — NOT a verbatim copy of the passages>",
  "citations": [
    {
      "source": "NG12 PDF",
      "page": <page number>,
      "chunk_id": "<chunk identifier>",
      "excerpt": "<1-2 sentence excerpt from the guideline>"
    }
  ]
}
"""

LOW_EVIDENCE_ANSWER = (
    "I couldn't find support in the NG12 text for that question. "
    "The retrieved guideline passages did not contain relevant information. "
    "Please try rephrasing your question or ask about specific cancer types, "
    "symptoms, or referral criteria covered by the NG12 guidelines."
)

GREETING_EXACT = {
    "hi", "hii", "hiii", "hey", "hello", "howdy", "sup", "yo",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "bye", "goodbye", "see you",
    "what's up", "whats up", "how are you", "who are you",
    "what can you do", "help",
}

GREETING_WORDS = {"hi", "hii", "hiii", "hey", "hello", "howdy", "sup", "yo", "bye", "goodbye"}

GREETING_RESPONSE = (
    "Hi there! I'm the NG12 Clinical Knowledge Assistant. I can help you with "
    "questions about the NICE NG12 guidelines on suspected cancer recognition "
    "and referral. Feel free to ask about symptoms, referral criteria, "
    "investigations, or any topic covered by the guidelines."
)

DISCLAIMER_PHRASES = [
    "couldn't find", "could not find", "not found",
    "no relevant", "no clear support", "unclear in ng12",
    "not contain relevant",
]


def _build_llm():
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.1,
    )


def _format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant guideline passages were retrieved."
    parts = []
    for c in chunks:
        parts.append(
            f"[Chunk {c['chunk_id']} | Page {c['page']} | Distance {c['distance']:.3f}]\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _has_good_evidence(chunks: list[dict]) -> bool:
    if not chunks:
        return False
    return min(c["distance"] for c in chunks) < SIMILARITY_THRESHOLD


def _is_greeting(text: str) -> bool:
    cleaned = text.strip().lower().rstrip("!?.,")
    if cleaned in GREETING_EXACT:
        return True
    # Short messages containing a greeting word (e.g. "hii how are u")
    if len(cleaned) < 60 and set(cleaned.split()) & GREETING_WORDS:
        return True
    return False


def _is_disclaimer(text: str) -> bool:
    lower = text.lower()
    return any(p in lower for p in DISCLAIMER_PHRASES)


def _parse_json(raw_text: str) -> dict:
    """Extract JSON from the LLM response, handling markdown fences and mixed text."""
    text = raw_text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        first = raw_text.find("{")
        last = raw_text.rfind("}")
        if first != -1 and last > first:
            return json.loads(raw_text[first:last + 1])
        raise


def _citations_from_chunks(chunks: list[dict], limit: int = 3) -> list[Citation]:
    return [
        Citation(
            source="NG12 PDF",
            page=c["page"],
            chunk_id=c["chunk_id"],
            excerpt=c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
        )
        for c in chunks[:limit]
    ]


def chat_with_guidelines(session_id: str, message: str, top_k: int = 5) -> ChatResponse:
    if session_id not in _sessions:
        _sessions[session_id] = []

    history = _sessions[session_id]
    history.append(ChatMessage(role="user", content=message))

    # Handle greetings without hitting RAG
    if _is_greeting(message):
        history.append(ChatMessage(role="assistant", content=GREETING_RESPONSE, citations=[]))
        return ChatResponse(session_id=session_id, answer=GREETING_RESPONSE, citations=[])

    # Retrieve relevant guideline chunks
    retrieved_chunks = query_guidelines_text(message, top_k=top_k)

    # Bail early if evidence is too weak
    if not _has_good_evidence(retrieved_chunks):
        history.append(ChatMessage(role="assistant", content=LOW_EVIDENCE_ANSWER, citations=[]))
        return ChatResponse(session_id=session_id, answer=LOW_EVIDENCE_ANSWER, citations=[])

    # Build the message list for Gemini
    messages = [("system", SYSTEM_PROMPT)]

    for msg in history[:-1][-20:]:
        messages.append((msg.role, msg.content))

    messages.append(("user", (
        f"RETRIEVED CONTEXT:\n{_format_context(retrieved_chunks)}\n\n"
        f"USER QUESTION:\n{message}"
    )))

    # Call LLM
    llm = _build_llm()
    raw_text = llm.invoke(messages).content

    # Parse the JSON response
    try:
        parsed = _parse_json(raw_text)
        answer = parsed.get("answer", raw_text)
        citations = [
            Citation(
                source=c.get("source", "NG12 PDF"),
                page=c.get("page", 0),
                chunk_id=c.get("chunk_id", "unknown"),
                excerpt=c.get("excerpt", ""),
            )
            for c in parsed.get("citations", [])
        ]

        if _is_disclaimer(answer):
            citations = []
        elif not citations and _has_good_evidence(retrieved_chunks):
            citations = _citations_from_chunks(retrieved_chunks)

    except (json.JSONDecodeError, KeyError):
        answer = raw_text
        if _is_disclaimer(answer):
            citations = []
        else:
            citations = _citations_from_chunks(retrieved_chunks)

    history.append(ChatMessage(role="assistant", content=answer, citations=citations))
    return ChatResponse(session_id=session_id, answer=answer, citations=citations)


def get_history(session_id: str) -> list[ChatMessage]:
    return _sessions.get(session_id, [])


def clear_session(session_id: str) -> bool:
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False
