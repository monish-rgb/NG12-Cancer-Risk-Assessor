"""Pydantic models for request/response schemas."""

from pydantic import BaseModel


class AssessRequest(BaseModel):
    patient_id: str


class Citation(BaseModel):
    source: str = "NG12 PDF"
    page: int
    chunk_id: str
    excerpt: str


class AssessResponse(BaseModel):
    patient_id: str
    patient_name: str
    risk_level: str
    assessment: str
    citations: list[Citation] = []


class PatientInfo(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    smoking_history: str
    symptoms: list[str]
    symptom_duration_days: int


class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: int = 5


class ChatMessage(BaseModel):
    role: str
    content: str
    citations: list[Citation] = []


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[Citation] = []
