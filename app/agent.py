"""Clinical Decision Support Agent using LangGraph + Gemini."""

import json
import os

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from app.models import AssessResponse, Citation
from app.rag import query_guidelines
from app.tools import get_patient

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

SYSTEM_PROMPT = """You are a Clinical Decision Support Agent specializing in cancer risk assessment
using the NG12 guidelines ("Suspected cancer: recognition and referral").

Your task is to assess a patient's cancer risk based on their clinical data and the NG12 guidelines.

## Process:
1. First, retrieve the patient's data using the get_patient_data tool.
2. Then, search the NG12 guidelines for relevant sections using the search_guidelines tool,
   based on the patient's symptoms.
3. Analyze the patient's data against the retrieved guideline criteria.
4. Determine the appropriate risk level and recommendation.

## Risk Levels:
- "Urgent Referral (2-week wait)": Patient meets NG12 criteria for urgent suspected cancer referral.
- "Urgent Investigation": Patient meets criteria for urgent investigation (e.g. imaging, blood tests).
- "Non-Urgent Referral": Symptoms warrant further investigation but do not meet urgent criteria.
- "Low Risk - Routine Follow-up": Symptoms are present but do not meet NG12 thresholds for referral.

## Important Rules:
- ONLY base your assessment on the retrieved NG12 guideline text. Do not invent criteria.
- Always cite the specific guideline passages that support your assessment.
- Consider patient age, gender, smoking history, symptom duration, and symptom combination.
- If the guidelines do not clearly address the patient's presentation, state this explicitly.

## Output Format:
You MUST respond with valid JSON in exactly this format:
{
  "risk_level": "<one of the risk levels above>",
  "assessment": "<detailed clinical reasoning explaining why this risk level was assigned>",
  "citations": [
    {
      "source": "NG12 PDF",
      "page": <page number>,
      "chunk_id": "<chunk identifier>",
      "excerpt": "<relevant text excerpt from the guideline>"
    }
  ]
}
"""


@tool
def get_patient_data(patient_id: str) -> str:
    """Retrieve patient clinical data by their patient ID."""
    patient = get_patient(patient_id)
    return json.dumps(patient.model_dump(), indent=2)


@tool
def search_guidelines(symptoms: list[str]) -> str:
    """Search the NG12 guidelines for sections relevant to the given symptoms."""
    chunks = query_guidelines(symptoms, top_k=8)
    return json.dumps(chunks, indent=2)


def _build_agent():
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.1,
    )
    return create_react_agent(llm, [get_patient_data, search_guidelines], prompt=SYSTEM_PROMPT)


def _parse_json(raw_text: str) -> dict:
    text = raw_text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


def assess_patient(patient_id: str) -> AssessResponse:
    agent = _build_agent()

    result = agent.invoke({"messages": [("user", (
        f"Assess the cancer risk for patient {patient_id}. "
        "Retrieve their data, search the NG12 guidelines for their symptoms, "
        "and provide a risk assessment with citations."
    ))]})

    final_text = result["messages"][-1].content

    try:
        parsed = _parse_json(final_text)
        patient = get_patient(patient_id)

        citations = [
            Citation(
                source=c.get("source", "NG12 PDF"),
                page=c.get("page", 0),
                chunk_id=c.get("chunk_id", "unknown"),
                excerpt=c.get("excerpt", ""),
            )
            for c in parsed.get("citations", [])
        ]

        return AssessResponse(
            patient_id=patient_id,
            patient_name=patient.name,
            risk_level=parsed.get("risk_level", "Unknown"),
            assessment=parsed.get("assessment", final_text),
            citations=citations,
        )

    except (json.JSONDecodeError, KeyError):
        patient = get_patient(patient_id)
        return AssessResponse(
            patient_id=patient_id,
            patient_name=patient.name,
            risk_level="Assessment Error",
            assessment=f"Agent returned non-JSON response: {final_text}",
            citations=[],
        )
