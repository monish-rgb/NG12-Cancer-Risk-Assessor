"""Integration tests for FastAPI endpoints."""

import os

import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestPatientsEndpoint:
    def test_list_patients(self, client):
        response = client.get("/patients")
        assert response.status_code == 200
        data = response.json()
        assert "patient_ids" in data
        assert len(data["patient_ids"]) == 10

    def test_get_patient_detail(self, client):
        response = client.get("/patients/PT-101")
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "PT-101"
        assert data["name"] == "John Doe"

    def test_get_invalid_patient(self, client):
        response = client.get("/patients/PT-999")
        assert response.status_code == 404


class TestAssessEndpoint:
    def test_invalid_patient_returns_404(self, client):
        response = client.post("/assess", json={"patient_id": "PT-999"})
        assert response.status_code == 404

    def test_missing_patient_id_returns_422(self, client):
        response = client.post("/assess", json={})
        assert response.status_code == 422

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    )
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "vectorstore")),
        reason="Vector store not built",
    )
    def test_assess_valid_patient(self, client):
        response = client.post("/assess", json={"patient_id": "PT-101"})
        assert response.status_code == 200
        data = response.json()
        assert data["patient_id"] == "PT-101"
        assert data["patient_name"] == "John Doe"
        assert "risk_level" in data
        assert "assessment" in data
        assert "citations" in data
        assert isinstance(data["citations"], list)

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    )
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "vectorstore")),
        reason="Vector store not built",
    )
    def test_assess_citations_have_required_fields(self, client):
        response = client.post("/assess", json={"patient_id": "PT-110"})
        assert response.status_code == 200
        data = response.json()
        for citation in data["citations"]:
            assert "source" in citation
            assert "page" in citation
            assert "chunk_id" in citation
            assert "excerpt" in citation
