# Chat Prompt Engineering Strategy (Part 2)

This document explains the system prompt design, grounding strategy, and guardrails for the **conversational NG12 chat agent**.

For the Part 1 risk assessment prompt strategy, see [PROMPTS.md](PROMPTS.md).

## Overview

The chat agent answers free-text questions about the NICE NG12 guidelines using RAG. Unlike the Part 1 agent (which uses tool calling), the chat agent follows a simpler pipeline: retrieve chunks, inject them as context into the prompt, and have the LLM synthesize an answer with citations.

## System Prompt Design

### Role Definition

The chat agent is defined as an **NG12 Clinical Knowledge Assistant** — distinct from the Part 1 risk assessor. Its sole purpose is answering questions about the NG12 guidelines using retrieved evidence. This narrow framing prevents the model from drifting into general medical advice.

### Critical Output Rules

The prompt enforces two key behaviors:

1. **JSON-only output** — the entire response must be a single valid JSON object with no text before or after. This prevents mixed text+JSON responses that break parsing.
2. **Synthesize, don't copy** — the prompt explicitly instructs the model to summarize and restructure retrieved passages in its own words, rather than repeating them verbatim. This was added to fix an issue where answers would just dump raw guideline text back to the user.

### Grounding Rules

The prompt enforces strict evidence-based answers:

- **Only use retrieved context** — the model must never use its own medical knowledge
- **Mandatory citations** — every clinical statement must include a citation with page number, chunk ID, and a 1-2 sentence excerpt
- **No inventing** — the model must never guess age thresholds, investigation timelines, or referral criteria not present in the retrieved text
- **No external references** — only NG12 content is allowed
- **Acknowledge gaps** — if the retrieved context doesn't contain enough information, the model must say "I couldn't find clear support in the NG12 guidelines for that question" and return an empty citations list

## Guardrails

### Greeting Detection

Before hitting the RAG pipeline, messages are checked for greetings ("hi", "hello", "how are you", etc.) using two methods:

1. **Exact match** against a set of known greeting patterns (e.g., "hi", "good morning", "thanks")
2. **Fuzzy match** for short messages (< 60 characters) containing a greeting word (e.g., "hii how are u" matches because it contains "hii")

Greetings return a canned response with no citations and skip RAG entirely, saving API cost and avoiding irrelevant chunk retrieval.

### Low-Evidence Gate

ChromaDB always returns results, even for nonsensical queries. To prevent the LLM from hallucinating answers from irrelevant chunks, a **similarity threshold** (cosine distance > 1.2) gates the pipeline:

- If the best-matching chunk has a distance above 1.2, the system returns a canned "couldn't find relevant info" disclaimer
- The LLM is never called with weak evidence — this prevents it from confabulating an answer

### Disclaimer Detection

After the LLM responds, the system checks if the answer contains disclaimer phrases like "couldn't find", "no relevant", or "no clear support":

- If the answer is a disclaimer, any citations the LLM may have attached are stripped (don't cite sources for an "I don't know" answer)
- This runs on both successfully parsed JSON responses and fallback raw-text responses

### Citation Enforcement

If the LLM returns a clinical answer but forgets to include citations (and good evidence was retrieved), the system automatically derives citations from the top 3 retrieved chunks. This ensures every substantive answer includes at least one citation.

## JSON Parsing

LLMs don't always return clean JSON. The parsing layer handles three scenarios:

1. **Markdown fences** — strips `` ```json ... ``` `` wrappers
2. **Mixed text + JSON** — if the LLM outputs plain text followed by a JSON object, the parser finds the first `{` and last `}` and extracts the JSON between them
3. **Complete failure** — if JSON parsing fails entirely, the raw text is used as the answer, with citations auto-derived from retrieved chunks

## Conversation Memory

### Storage

Session history is stored in an in-memory dictionary: `session_id -> list[ChatMessage]`. Each message stores `role`, `content`, and `citations`.

### Context Window Management

The last 20 messages (approximately 10 user-assistant exchanges) are injected into the LLM prompt as prior conversation turns. This keeps the context window manageable while supporting multi-turn follow-ups like:

- "What about if the patient is under 40?" — the model uses prior context to understand what condition is being discussed
- "Can you quote the specific criteria?" — the model can reference what was discussed earlier

### Session Lifecycle

- Sessions are created automatically on the first message (no explicit creation needed)
- `GET /chat/{session_id}/history` retrieves the full conversation
- `DELETE /chat/{session_id}` clears the session
- Sessions are lost when the server restarts (in-memory only)

## RAG Pipeline Reuse

The chat agent reuses the **same vector store and embedding model** from Part 1:

- Same ChromaDB collection (`ng12_guidelines`)
- Same embedding model (`models/gemini-embedding-001` via `langchain-google-genai`)
- A shared `_query()` helper in `rag.py` is used by both `query_guidelines()` (Part 1, symptom-list input) and `query_guidelines_text()` (Part 2, free-text input)
- No re-downloading or re-embedding per chat request

## Temperature

Set to **0.1** — same as Part 1. Clinical guideline Q&A requires consistent, reproducible answers. Low temperature minimizes creative elaboration that could introduce inaccuracies.
