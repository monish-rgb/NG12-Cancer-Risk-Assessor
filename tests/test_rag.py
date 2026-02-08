"""Tests for RAG pipeline.

These tests require the vector store to be built first (run ingest_pdf.py).
They also require GOOGLE_API_KEY to be set for embedding queries.
"""

import os

import pytest

# Skip all tests in this module if vectorstore doesn't exist or API key not set
pytestmark = [
    pytest.mark.skipif(
        not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "vectorstore")),
        reason="Vector store not built. Run: python -m ingestion.ingest_pdf",
    ),
    pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set",
    ),
]


class TestQueryGuidelines:
    def test_returns_results(self):
        from app.rag import query_guidelines
        results = query_guidelines(["hemoptysis"], top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_result_has_required_fields(self):
        from app.rag import query_guidelines
        results = query_guidelines(["cough"], top_k=1)
        assert len(results) == 1
        chunk = results[0]
        assert "chunk_id" in chunk
        assert "page" in chunk
        assert "text" in chunk
        assert "distance" in chunk

    def test_page_is_positive_int(self):
        from app.rag import query_guidelines
        results = query_guidelines(["breast lump"], top_k=3)
        for chunk in results:
            assert isinstance(chunk["page"], int)
            assert chunk["page"] > 0

    def test_chunk_id_format(self):
        from app.rag import query_guidelines
        results = query_guidelines(["haematuria"], top_k=3)
        for chunk in results:
            assert chunk["chunk_id"].startswith("ng12_p")

    def test_different_symptoms_return_different_results(self):
        from app.rag import query_guidelines
        lung_results = query_guidelines(["hemoptysis", "cough"], top_k=3)
        breast_results = query_guidelines(["breast lump"], top_k=3)

        lung_ids = {r["chunk_id"] for r in lung_results}
        breast_ids = {r["chunk_id"] for r in breast_results}
        # At least some results should differ
        assert lung_ids != breast_ids
