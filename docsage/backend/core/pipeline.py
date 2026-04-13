"""
core/pipeline.py — DocSage QA Pipeline Orchestrator

The central engine that wires together:
  Document ingestion → Chunking → Indexing
  Query → Retrieval → Reranking → Reading → Adversarial filter → Response

Also manages:
  - Session-level conversation history (multi-turn CoQA)
  - Per-document domain adaptation hooks
  - Latency tracking across pipeline stages
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.config import settings
from core.logging import get_logger
from utils.document_parser import DocumentParser
from utils.chunker import HierarchicalChunker
from utils.retriever import HybridRetriever
from models.reader import BERTReader, QAAnswer
from models.reranker import CrossEncoderReranker, AdversarialFilter

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    question: str
    answer: str
    confidence: float
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class Session:
    session_id: str
    doc_ids: list[str] = field(default_factory=list)
    history: list[ConversationTurn] = field(default_factory=list)

    def add_turn(self, question: str, answer: QAAnswer) -> None:
        self.history.append(ConversationTurn(
            question=question,
            answer=answer.answer,
            confidence=answer.confidence,
        ))
        if len(self.history) > settings.max_history_turns:
            self.history = self.history[-settings.max_history_turns:]

    def history_as_list(self) -> list[dict]:
        return [
            {"question": t.question, "answer": t.answer}
            for t in self.history
        ]


@dataclass
class PipelineResponse:
    """Full response from the QA pipeline with all metadata."""
    answer: str
    confidence: float
    is_impossible: bool
    sources: list[dict]
    adversarial_risk: str         # "low" | "medium" | "high"
    answer_type: str
    latency_breakdown: dict[str, float]
    total_latency_ms: float
    session_id: str
    turn_id: Optional[str] = None


class DocSagePipeline:
    """
    Singleton pipeline that:
    1. Loads all models lazily on first use
    2. Manages the document index (add/remove)
    3. Handles full QA pipeline including conversation state
    4. Exposes indexing + QA as clean methods for the API layer
    """

    _instance: Optional[DocSagePipeline] = None

    def __init__(self):
        settings.create_dirs()
        self.parser = DocumentParser()
        self.chunker = HierarchicalChunker()
        self.retriever = HybridRetriever()
        self._reader: Optional[BERTReader] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._adversarial_filter: Optional[AdversarialFilter] = None
        self._sessions: dict[str, Session] = {}
        logger.info("pipeline_initialized")

    @classmethod
    def get(cls) -> DocSagePipeline:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ─── Lazy Model Loading ──────────────────────────────────────────────────

    @property
    def reader(self) -> BERTReader:
        if self._reader is None:
            self._reader = BERTReader()
        return self._reader

    @property
    def reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    @property
    def adversarial_filter(self) -> AdversarialFilter:
        if self._adversarial_filter is None:
            self._adversarial_filter = AdversarialFilter()
        return self._adversarial_filter

    # ─── Document Management ─────────────────────────────────────────────────

    def ingest_document(self, file_path: Path, doc_id: Optional[str] = None) -> str:
        """
        Full ingestion pipeline for a document:
        Parse → Chunk → Index
        Returns the doc_id.
        """
        doc_id = doc_id or str(uuid.uuid4())
        t0 = time.perf_counter()

        logger.info("ingesting_document", path=str(file_path), doc_id=doc_id)

        # Step 1: Parse
        t1 = time.perf_counter()
        parsed = self.parser.parse(file_path, doc_id)
        parse_ms = (time.perf_counter() - t1) * 1000

        # Step 2: Chunk
        t2 = time.perf_counter()
        chunks = self.chunker.chunk_document(parsed)
        chunk_ms = (time.perf_counter() - t2) * 1000

        # Step 3: Index
        t3 = time.perf_counter()
        self.retriever.add_chunks(chunks)
        index_ms = (time.perf_counter() - t3) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "ingestion_complete",
            doc_id=doc_id,
            sections=len(parsed.sections),
            chunks=len(chunks),
            parse_ms=round(parse_ms),
            chunk_ms=round(chunk_ms),
            index_ms=round(index_ms),
            total_ms=round(total_ms),
        )

        return doc_id

    def remove_document(self, doc_id: str) -> int:
        """Remove a document and all its chunks from the index."""
        return self.retriever.remove_doc(doc_id)

    # ─── Question Answering ──────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        session_id: str,
        doc_ids: Optional[list[str]] = None,
    ) -> PipelineResponse:
        """
        Full QA pipeline for a question in a session context.

        Pipeline stages:
        0. Adversarial screening of the question
        1. Hybrid retrieval (BM25 + FAISS + KG)
        2. Cross-encoder reranking
        3. BERT reader (multi-passage extraction)
        4. Confidence calibration with adversarial penalty
        5. Conversation history update
        """
        t_total = time.perf_counter()
        latency: dict[str, float] = {}

        # 0. Adversarial filter
        t0 = time.perf_counter()
        adv_score = self.adversarial_filter.score(question)
        latency["adversarial_ms"] = (time.perf_counter() - t0) * 1000

        if adv_score.is_suspicious:
            logger.warning("suspicious_query", risk=adv_score.risk_level, fraction=adv_score.replaced_fraction)

        # 1. Retrieve
        t1 = time.perf_counter()
        session = self._get_or_create_session(session_id)
        retrieved = self.retriever.retrieve(
            query=question,
            doc_ids=doc_ids or session.doc_ids or None,
        )
        latency["retrieval_ms"] = (time.perf_counter() - t1) * 1000

        # 2. Rerank
        t2 = time.perf_counter()
        reranked_chunks = self.reranker.rerank(query=question, passages=retrieved)
        latency["reranking_ms"] = (time.perf_counter() - t2) * 1000

        # 3. Read
        t3 = time.perf_counter()
        qa_answer = self.reader.answer(
            question=question,
            passages=reranked_chunks,
            history=session.history_as_list(),
        )
        latency["reading_ms"] = (time.perf_counter() - t3) * 1000

        # 4. Apply adversarial penalty
        if adv_score.is_suspicious:
            qa_answer.confidence = max(
                0.0, qa_answer.confidence - adv_score.penalty
            )

        # 5. Update session
        session.add_turn(question, qa_answer)

        total_ms = (time.perf_counter() - t_total) * 1000
        latency["total_ms"] = total_ms

        logger.info(
            "qa_complete",
            session_id=session_id,
            confidence=round(qa_answer.confidence, 3),
            latency_ms=round(total_ms),
            adversarial_risk=adv_score.risk_level,
        )

        return PipelineResponse(
            answer=qa_answer.answer,
            confidence=qa_answer.confidence,
            is_impossible=qa_answer.is_impossible,
            sources=qa_answer.sources,
            adversarial_risk=adv_score.risk_level,
            answer_type=qa_answer.answer_type,
            latency_breakdown=latency,
            total_latency_ms=total_ms,
            session_id=session_id,
        )

    # ─── Session Management ──────────────────────────────────────────────────

    def create_session(self, doc_ids: Optional[list[str]] = None) -> str:
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = Session(
            session_id=session_id,
            doc_ids=doc_ids or [],
        )
        return session_id

    def get_session_history(self, session_id: str) -> list[dict]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [
            {"question": t.question, "answer": t.answer, "confidence": t.confidence}
            for t in session.history
        ]

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].history.clear()

    def _get_or_create_session(self, session_id: str) -> Session:
        if session_id not in self._sessions:
            self._sessions[session_id] = Session(session_id=session_id)
        return self._sessions[session_id]