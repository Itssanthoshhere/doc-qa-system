"""
api/routers/qa.py — Question answering endpoint.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.pipeline import DocSagePipeline
from core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class QARequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    session_id: str = Field(..., description="Session ID for conversation tracking")
    doc_ids: Optional[list[str]] = Field(
        None,
        description="Restrict search to these documents. If None, searches all.",
    )


class SourceEvidence(BaseModel):
    doc_id: str
    page: Optional[int]
    section: Optional[str]
    snippet: str


class QAResponse(BaseModel):
    answer: str
    confidence: float
    confidence_label: str   # "high" | "medium" | "low" | "uncertain"
    is_impossible: bool
    sources: list[SourceEvidence]
    adversarial_risk: str
    answer_type: str
    latency_breakdown: dict[str, float]
    total_latency_ms: float
    session_id: str


def _confidence_label(conf: float) -> str:
    if conf >= 0.80:
        return "high"
    elif conf >= 0.50:
        return "medium"
    elif conf >= 0.25:
        return "low"
    else:
        return "uncertain"


@router.post("/ask", response_model=QAResponse)
async def ask_question(request: QARequest):
    """
    Ask a question against indexed documents.

    - Multi-turn conversation tracked via session_id
    - Hybrid retrieval: FAISS + BM25 + Knowledge Graph
    - Cross-encoder reranking + BERT reading
    - Adversarial robustness scoring
    - Returns calibrated confidence + source citations
    """
    try:
        pipeline = DocSagePipeline.get()
        response = pipeline.answer(
            question=request.question,
            session_id=request.session_id,
            doc_ids=request.doc_ids,
        )

        sources = [
            SourceEvidence(
                doc_id=s["doc_id"],
                page=s.get("page"),
                section=s.get("section"),
                snippet=s.get("snippet", "")[:300],
            )
            for s in response.sources
        ]

        return QAResponse(
            answer=response.answer,
            confidence=round(response.confidence, 4),
            confidence_label=_confidence_label(response.confidence),
            is_impossible=response.is_impossible,
            sources=sources,
            adversarial_risk=response.adversarial_risk,
            answer_type=response.answer_type,
            latency_breakdown={k: round(v, 1) for k, v in response.latency_breakdown.items()},
            total_latency_ms=round(response.total_latency_ms, 1),
            session_id=response.session_id,
        )

    except Exception as e:
        logger.error("qa_endpoint_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"QA failed: {str(e)}")