"""Tests for patient lookup tool."""

import pytest

from app.tools import get_patient, list_patient_ids


class TestListPatientIds:
    def test_returns_all_patients(self):
        ids = list_patient_ids()
        assert len(ids) == 10

    def test_ids_are_sorted(self):
        ids = list_patient_ids()
        assert ids == sorted(ids)

    def test_expected_ids_present(self):
        ids = list_patient_ids()
        for i in range(101, 111):
            assert f"PT-{i}" in ids


class TestGetPatient:
    def test_valid_patient(self):
        patient = get_patient("PT-101")
        assert patient.patient_id == "PT-101"
        assert patient.name == "John Doe"
        assert patient.age == 55
        assert patient.gender == "Male"
        assert "unexplained hemoptysis" in patient.symptoms

    def test_all_patients_loadable(self):
        ids = list_patient_ids()
        for pid in ids:
            patient = get_patient(pid)
            assert patient.patient_id == pid
            assert patient.name
            assert patient.age > 0
            assert len(patient.symptoms) > 0

    def test_invalid_patient_raises(self):
        with pytest.raises(KeyError):
            get_patient("PT-999")

    def test_patient_fields_populated(self):
        patient = get_patient("PT-105")
        assert patient.name == "Michael Chang"
        assert patient.age == 65
        assert patient.smoking_history == "Ex-Smoker"
        assert "iron-deficiency anaemia" in patient.symptoms
        assert patient.symptom_duration_days == 60
