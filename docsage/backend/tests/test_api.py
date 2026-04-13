"""
backend/tests/test_api.py — Integration tests for FastAPI endpoints.
Run with: pytest tests/test_api.py -v
"""
import pytest
from httpx import AsyncClient
from unittest.mock import patch, MagicMock
from api.main import app


@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_pipeline():
    """Mock the DocSage pipeline to avoid loading ML models."""
    with patch("core.pipeline.DocSagePipeline.get") as mock_get:
        pipeline = MagicMock()

        # Mock session creation
        pipeline.create_session.return_value = "test-session-id"
        pipeline.get_session_history.return_value = []

        # Mock QA answer
        from core.pipeline import PipelineResponse
        pipeline.answer.return_value = PipelineResponse(
            answer="BERT achieves 90.1 F1 on SQuAD 2.0.",
            confidence=0.87,
            is_impossible=False,
            sources=[{
                "doc_id": "doc_123",
                "page": 3,
                "section": "Results",
                "snippet": "BERT achieves 90.1 F1 on SQuAD 2.0.",
            }],
            adversarial_risk="low",
            answer_type="extractive",
            latency_breakdown={"retrieval_ms": 45.2, "reading_ms": 120.3},
            total_latency_ms=180.5,
            session_id="test-session-id",
        )

        mock_get.return_value = pipeline
        yield pipeline


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestSessionEndpoints:
    @pytest.mark.asyncio
    async def test_create_session(self, client, mock_pipeline):
        response = await client.post("/api/v1/sessions/", json={"doc_ids": []})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    @pytest.mark.asyncio
    async def test_get_history_empty(self, client, mock_pipeline):
        response = await client.get("/api/v1/sessions/test-session/history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.asyncio
    async def test_clear_history(self, client, mock_pipeline):
        response = await client.delete("/api/v1/sessions/test-session/history")
        assert response.status_code == 200


class TestQAEndpoint:
    @pytest.mark.asyncio
    async def test_ask_question_success(self, client, mock_pipeline):
        response = await client.post("/api/v1/qa/ask", json={
            "question": "What is BERT?",
            "session_id": "test-session",
        })
        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "confidence" in data
        assert "confidence_label" in data
        assert data["confidence_label"] in ("high", "medium", "low", "uncertain")
        assert "sources" in data
        assert "adversarial_risk" in data
        assert "total_latency_ms" in data
        assert 0.0 <= data["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_ask_question_with_doc_filter(self, client, mock_pipeline):
        response = await client.post("/api/v1/qa/ask", json={
            "question": "What datasets were used?",
            "session_id": "test-session",
            "doc_ids": ["doc_abc", "doc_xyz"],
        })
        assert response.status_code == 200
        # Verify pipeline was called with doc_ids
        mock_pipeline.answer.assert_called_once()
        call_kwargs = mock_pipeline.answer.call_args.kwargs
        assert call_kwargs.get("doc_ids") == ["doc_abc", "doc_xyz"]

    @pytest.mark.asyncio
    async def test_ask_question_short_rejected(self, client, mock_pipeline):
        response = await client.post("/api/v1/qa/ask", json={
            "question": "Hi",  # Too short (< 3 chars... but "Hi" is 2)
            "session_id": "test-session",
        })
        # Pydantic validation: min_length=3 for question
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_confidence_labels_correct(self, client, mock_pipeline):
        """Test confidence label mapping."""
        from api.routers.qa import _confidence_label
        assert _confidence_label(0.90) == "high"
        assert _confidence_label(0.65) == "medium"
        assert _confidence_label(0.35) == "low"
        assert _confidence_label(0.10) == "uncertain"


class TestDocumentEndpoints:
    @pytest.mark.asyncio
    async def test_list_documents_empty(self, client):
        response = await client.get("/api/v1/documents/")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, client):
        response = await client.get("/api/v1/documents/nonexistent-id")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_nonexistent_document(self, client):
        response = await client.delete("/api/v1/documents/nonexistent-id")
        assert response.status_code == 404