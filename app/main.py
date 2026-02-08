"""FastAPI service for NG12 Cancer Risk Assessor."""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent import assess_patient
from app.chat import chat_with_guidelines, get_history, clear_session
from app.models import AssessRequest, AssessResponse, ChatRequest, ChatResponse
from app.tools import get_patient, list_patient_ids

app = FastAPI(
    title="NG12 Cancer Risk Assessor",
    description="Clinical Decision Support Agent using NICE NG12 guidelines and Google Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "ng12-cancer-risk-assessor"}


@app.get("/patients")
def get_patients():
    return {"patient_ids": list_patient_ids()}


@app.get("/patients/{patient_id}")
def get_patient_detail(patient_id: str):
    try:
        patient = get_patient(patient_id)
        return patient.model_dump()
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")


@app.post("/assess", response_model=AssessResponse)
def assess(request: AssessRequest):
    try:
        get_patient(request.patient_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Patient {request.patient_id} not found")

    try:
        return assess_patient(request.patient_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        return chat_with_guidelines(
            session_id=request.session_id,
            message=request.message,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.get("/chat/{session_id}/history")
def chat_history(session_id: str):
    history = get_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"session_id": session_id, "messages": [m.model_dump() for m in history]}


@app.delete("/chat/{session_id}")
def delete_chat(session_id: str):
    deleted = clear_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return {"status": "deleted", "session_id": session_id}
