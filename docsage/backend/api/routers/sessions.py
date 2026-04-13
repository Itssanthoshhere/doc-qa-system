"""
api/routers/sessions.py — Conversation session management.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.pipeline import DocSagePipeline
from core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class SessionCreateRequest(BaseModel):
    doc_ids: Optional[list[str]] = None


class SessionResponse(BaseModel):
    session_id: str
    doc_ids: list[str]
    turn_count: int


class HistoryTurn(BaseModel):
    question: str
    answer: str
    confidence: float


@router.post("/", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new conversation session, optionally pinning it to specific documents."""
    pipeline = DocSagePipeline.get()
    session_id = pipeline.create_session(request.doc_ids)
    return SessionResponse(session_id=session_id, doc_ids=request.doc_ids or [], turn_count=0)


@router.get("/{session_id}/history", response_model=list[HistoryTurn])
async def get_history(session_id: str):
    """Retrieve conversation history for a session."""
    pipeline = DocSagePipeline.get()
    history = pipeline.get_session_history(session_id)
    return [HistoryTurn(**h) for h in history]


@router.delete("/{session_id}/history")
async def clear_history(session_id: str):
    """Clear conversation history (reset context) without deleting the session."""
    pipeline = DocSagePipeline.get()
    pipeline.clear_session(session_id)
    return {"cleared": session_id}