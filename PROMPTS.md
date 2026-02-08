# Prompt Engineering Strategy

This document explains the system prompt design for the **Part 1: NG12 Cancer Risk Assessor** agent.

For the Part 2 chat prompt strategy, see [CHAT_PROMPTS.md](CHAT_PROMPTS.md).

## Overview

The agent receives a Patient ID, retrieves their clinical data, searches the NG12 guidelines via RAG, and produces a risk assessment with citations. The entire reasoning flow is driven by the system prompt.

## System Prompt Design

### Role Definition

The agent is framed as a **Clinical Decision Support Agent** specializing in NG12 cancer risk assessment. This narrow role prevents the model from drifting into general medical advice or acting outside its scope.

### Reasoning Chain

The prompt defines a strict 4-step process:

1. **Retrieve patient data** using the `get_patient_data` tool
2. **Search NG12 guidelines** using the `search_guidelines` tool based on the patient's symptoms
3. **Analyze** the patient's clinical data against the retrieved guideline criteria
4. **Determine** the risk level and produce a structured JSON assessment

This chain-of-thought (COT) approach ensures the agent gathers all evidence before reasoning, which prevents hallucinated clinical answers.

### Risk Level

Four explicit risk levels are defined in the prompt so the agent maps its assessment to a consistent vocabulary:

| Risk Level | When to Use |
_____________________________
| Urgent Referral (2-week wait) | Patient meets NG12 criteria for urgent suspected cancer referral |
| Urgent Investigation | Patient meets criteria for urgent investigation (imaging, blood tests) |
| Non-Urgent Referral | Symptoms warrant follow-up but don't meet urgent criteria |
| Low Risk - Routine Follow-up | Symptoms present but below NG12 referral thresholds |

### Grounding Rules

The prompt enforces strict grounding to prevent hallucination:

- **Only use retrieved guideline text** - the agent must never invent criteria from its own knowledge
- **Always cite specific passages** - every clinical statement needs a page number, chunk ID and excerpt
- **Consider all patient factors** - age, gender, smoking history, symptom duration, and symptom combinations
- **Acknowledge uncertainty** - if the guidelines don't clearly address the patient's presentation, the agent must say so explicitly

### Output Format

The prompt requires strict JSON output:

```json
{
  "risk_level": "Urgent Referral (2-week wait)",
  "assessment": "Clinical reasoning explaining the risk level...",
  "citations": [
    {
      "source": "NG12 PDF",
      "page": 23,
      "chunk_id": "ng12_p023_c0042",
      "excerpt": "Relevant guideline text..."
    }
  ]
}
```

JSON parsing in `agent.py` handles both clean JSON and markdown-wrapped responses (` ```json ... ``` `), with a fallback error response if parsing fails entirely.

## Tool Definitions

The agent has two LangChain tools, invoked via Gemini function calling:

### `get_patient_data(patient_id: str)`

- Fetches structured patient data (age, gender, symptoms, smoking history, symptom duration) from `patients.json`
- Defined as a tool so the agent explicitly requests patient data as its first action

### `search_guidelines(symptoms: list[str])`

- Performs a RAG query against the ChromaDB vector store using the patient's symptoms
- Returns the top 8 most relevant guideline chunks with page numbers and text
- The agent decides which symptoms to search for, allowing it to reformulate queries or search for related terms

## Agent Architecture

The agent uses **LangGraph's `create_react_agent`** — a ReAct (Reason + Act) loop where the model alternates between reasoning and tool calls until it produces a final text response. This allows the agent to make multiple tool calls if needed (e.g., retrieving patient data first then searching guidelines).

## Temperature

Set to **0.1** for near-deterministic output.
