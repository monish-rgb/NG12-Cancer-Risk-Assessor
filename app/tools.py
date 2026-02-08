"""Patient data lookup — simulates a database query."""

import json
import os

from app.models import PatientInfo

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "patients.json")

_patients_cache: dict[str, dict] | None = None


def _load_patients() -> dict[str, dict]:
    global _patients_cache
    if _patients_cache is None:
        with open(os.path.abspath(DATA_PATH), "r") as f:
            patients_list = json.load(f)
        _patients_cache = {p["patient_id"]: p for p in patients_list}
    return _patients_cache


def get_patient(patient_id: str) -> PatientInfo:
    patients = _load_patients()
    if patient_id not in patients:
        raise KeyError(f"Patient {patient_id} not found")
    return PatientInfo(**patients[patient_id])


def list_patient_ids() -> list[str]:
    return sorted(_load_patients().keys())
