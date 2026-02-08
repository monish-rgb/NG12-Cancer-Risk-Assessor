# NG12 Cancer Risk Assessor

A Clinical Decision Support system that uses the NG12 guidelines to assess patient cancer risk and answer clinical questions. Built with Google Gemini, LangGraph, FastAPI, ChromaDB, and Streamlit.

## Architecture

```
User (Streamlit UI)
  |
FastAPI Service
  |
  +-- Part 1: Risk Assessment
  |     Patient ID -> Gemini ReAct Agent -> [get_patient_data + search_guidelines] -> Risk Assessment + Citations

  +-- Part 2: Chat
        Question -> RAG Retrieval -> Gemini LLM -> Synthesized Answer + Citations
  |
  +-- Shared RAG Pipeline
        ChromaDB (ng12_guidelines collection) <- Gemini Embeddings <- NG12 PDF
```

### Part 1: Risk Assessment Flow

1. User submits a Patient ID via Streamlit UI or API
2. LangGraph ReAct agent calls `get_patient_data` tool to fetch patient records from `patients.json`
3. Agent calls `search_guidelines` tool to search the NG12 vector store for relevant guideline sections
4. Agent synthesizes patient data + guidelines to determine risk level
5. Returns JSON with risk level, clinical assessment, and guideline citations

### Part 2: Chat Flow

1. User asks a free-text question about NG12 guidelines
2. Question is embedded and matched against the ChromaDB vector store (top 5 chunks)
3. Evidence quality is checked weak matches trigger a "not found" disclaimer without calling the LLM
4. The quality is checked between the user prompt and the NG12 guidelines using cosine similarity using this the LLM answered are grounded (guardrails)
5. Retrieved chunks + conversation history are sent to Gemini for synthesis
6. Returns a grounded answer with citations from the guideline text

Both parts share the same vector store and embedding model — no re-ingestion per request.

## Tech Stack

- **LLM**: Google gemini-2.0-flash-lite-001 (via `langchain-google-genai`) as gemini-1.5 is not available in vertexAI or GoogleAI Studio
- **Agent Framework**: LangGraph ReAct agent (Part 1), direct LLM calls (Part 2)
- **Embeddings**: `models/gemini-embedding-001` (via `langchain-google-genai`)
- **Vector DB**: ChromaDB
- **PDF Parsing**: PyMuPDF
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Streamlit
- **Containerization**: Docker + Docker Compose

## Quick Start

### Prerequisites

- Python 3.11+
- Google AI Studio API key ([get one here](https://aistudio.google.com/apikey))
- The NG12 guidelines PDF (already included in `data/`)

### 1. Setup Environment

```bash
cd "NG12 Cancer Risk Assessor"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# or: .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

### 2. Build Vector Store

```
you can just run the application the pdf is ingested automatically and embeddings are created in vectorstore
So, just start the FastAPI backend and the Streamlit frontend the first query will trigger auto-ingestion if the vectorstore does not exist yet. The only thing is that the first request will be a little slower since it needs to parse the PDF, subsequent requests are faster
```

This parses the NG12 PDF, generates embeddings via Gemini, and stores them in ChromaDB under `vectorstore/`. If the vector store already exists, it skips ingestion.

The vector store is also auto-built on first API request if it does not exist.

### 3. Run the Application

```bash
# Terminal 1: Start FastAPI backend first
uvicorn app.main:app --reload --port 8000

# Terminal 2: Then start Streamlit frontend
streamlit run ui/streamlit_app.py
```

- API: <http://localhost:8000>
- API Docs: <http://localhost:8000/docs>
- UI: <http://localhost:8501>

The Streamlit UI has two tabs:

- **Risk Assessment** — select a patient, click "Assess Cancer Risk"
- **NG12 Chat** — ask free-text questions about the NG12 guidelines

### 4. Run with Docker

```bash
# Build and start both services
docker-compose up --build
```

Both tabs are available at <http://localhost:8501>.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Health check |
| GET | `/patients` | List all patient IDs |
| GET | `/patients/{id}` | Get patient details |
| POST | `/assess` | Assess cancer risk for a patient |
| POST | `/chat` | Chat with NG12 guidelines |
| GET | `/chat/{session_id}/history` | Get chat conversation history |
| DELETE | `/chat/{session_id}` | Clear a chat session |

### Risk Assessment Example

```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "PT-101"}'
```

```json
{
  "patient_id": "PT-101",
  "patient_name": "John Doe",
  "risk_level": "Urgent Referral (2-week wait)",
  "assessment": "Based on NG12 guidelines, a 55-year-old male current smoker presenting with unexplained hemoptysis meets the criteria for urgent referral...",
  "citations": [
    {
      "source": "NG12 PDF",
      "page": 23,
      "chunk_id": "ng12_p023_c0042",
      "excerpt": "Refer people using a suspected cancer pathway referral..."
    }
  ]
}
```

### Chat Example

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-123", "message": "What symptoms trigger an urgent referral for lung cancer?"}'
```

```json
{
  "session_id": "test-123",
  "answer": "According to NG12, an urgent referral for suspected lung cancer should be made if...",
  "citations": [
    {
      "source": "NG12 PDF",
      "page": 23,
      "chunk_id": "ng12_p023_c0042",
      "excerpt": "Refer people using a suspected cancer pathway referral..."
    }
  ]
}
```

## Prompt Engineering

Detailed documentation of the prompt strategies:

- **[PROMPTS.md](PROMPTS.md)** — Part 1 system prompt design (risk assessment agent, tool definitions, reasoning chain)
- **[CHAT_PROMPTS.md](CHAT_PROMPTS.md)** — Part 2 chat prompt design (grounding strategy, guardrails, citation enforcement, conversation memory)

## Running Tests

```bash
# Run all tests (some require GOOGLE_API_KEY and vectorstore)
pytest tests/ -v

# Run only unit tests (no API key needed)
pytest tests/test_tools.py -v

# Run inside Docker
docker-compose exec api pytest tests/ -v
```

## Project Structure

```
├── app/
│   ├── main.py            # FastAPI routes
│   ├── agent.py           # LangGraph ReAct agent (Part 1)
│   ├── chat.py            # Conversational RAG pipeline (Part 2)
│   ├── rag.py             # Shared RAG pipeline (ChromaDB queries + embeddings)
│   ├── tools.py           # Patient data lookup
│   └── models.py          # Pydantic request/response schemas
├── ingestion/
│   └── ingest_pdf.py      # PDF parsing + embedding + ChromaDB indexing
├── data/
│   ├── patients.json      # Simulated patient database
│   └── *.pdf              # NG12 guidelines PDF
├── vectorstore/           # ChromaDB
├── tests/                 # Pytest test suite
├── ui/
│   └── streamlit_app.py   # Streamlit frontend
├── PROMPTS.md             # Prompt engineering docs (Part 1)
├── CHAT_PROMPTS.md        # Chat prompt & grounding docs (Part 2)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                   # GOOGLE_API_KEY (not committed)
```

## Further system improvements

**With more time:** I wil add a proper clinical validation layer (compare outputs against expert annotated cases, track false positive/negative rates for urgent referrals) and add a human-in-the-loop review step before any risk level is shown as final.
**Re-Ranking:** After retrieving top-k chunks from ChromaDB I would add a cross-encoder re-ranker (e.g flashrank) to re-score and filter chunks before passing them to the LLM.
**My Suggestions:** gemini-2.0-flash-lite-001 is optimized for speed and cost but is the least capable Gemini model now available. For clinical use, gemini-2.0-flash would give better reasoning at moderate cost increase
