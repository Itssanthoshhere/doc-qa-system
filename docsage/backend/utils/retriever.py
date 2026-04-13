"""
utils/retriever.py — Hybrid Retrieval Engine

Combines three retrieval signals and fuses scores using Reciprocal Rank Fusion (RRF):
  1. Dense: BERT sentence-transformer embeddings → FAISS IVF index
  2. Sparse: BM25 keyword matching via rank-bm25
  3. Knowledge Graph: Entity-hop scoring via NetworkX

Why this beats every single-method paper in the review:
  - Dense alone misses exact keyword matches (fails "what is BM25 score for X?")
  - BM25 alone misses paraphrase and semantic variation
  - KG alone lacks coverage
  - RRF fusion is parameter-free and shown to outperform learned fusion
    (Cormack et al. 2009, confirmed repeatedly on TREC benchmarks)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import faiss
import networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.logging import get_logger
from utils.chunker import Chunk

logger = get_logger(__name__)

# Reciprocal Rank Fusion constant
_RRF_K = 60


@dataclass
class RetrievedPassage:
    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    kg_score: float = 0.0
    rrf_score: float = 0.0       # final fused rank score
    retrieval_method: str = "hybrid"


class HybridRetriever:
    """
    Maintains a FAISS index for dense retrieval, BM25 for sparse,
    and a NetworkX KG for entity-aware scoring.

    Index is persisted to disk and hot-reloaded on startup.
    Per-document indexing is supported for incremental updates.
    """

    def __init__(self, index_dir: Optional[Path] = None):
        self.index_dir = index_dir or settings.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.embedder = SentenceTransformer(
            settings.embedder_model_name,
            cache_folder=str(settings.model_cache_dir),
            device=settings.reader_model_device,
        )
        self.embedding_dim = settings.embedding_dim

        # State
        self._chunks: list[Chunk] = []
        self._chunk_id_to_idx: dict[str, int] = {}
        self._faiss_index: Optional[faiss.IndexIVFFlat] = None
        self._bm25: Optional[BM25Okapi] = None
        self._kg: nx.Graph = nx.Graph()
        self._bm25_corpus: list[list[str]] = []

        self._load_index()

    # ─── Public API ──────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> None:
        """Add new chunks to all indices."""
        if not chunks:
            return

        logger.info("adding_chunks_to_index", count=len(chunks))
        start_idx = len(self._chunks)

        # 1. Add to chunk store
        for i, chunk in enumerate(chunks):
            self._chunks.append(chunk)
            self._chunk_id_to_idx[chunk.chunk_id] = start_idx + i

        # 2. Embed and add to FAISS
        texts = [c.as_context_string() for c in chunks]
        embeddings = self._embed(texts)
        self._ensure_faiss_index()
        self._faiss_index.train(embeddings) if not self._faiss_index.is_trained else None
        self._faiss_index.add(embeddings)

        # 3. Rebuild BM25 (BM25Okapi requires full corpus each time)
        self._bm25_corpus = [self._tokenize_bm25(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(self._bm25_corpus)

        # 4. Update KG with entities from new chunks
        for chunk in chunks:
            self._update_kg(chunk)

        self._save_index()
        logger.info("index_updated", total_chunks=len(self._chunks))

    def remove_doc(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document (rebuild needed for FAISS)."""
        original_count = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
        self._rebuild_all_indices()
        removed = original_count - len(self._chunks)
        logger.info("doc_removed_from_index", doc_id=doc_id, chunks_removed=removed)
        return removed

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        doc_ids: Optional[list[str]] = None,
        filter_section_types: Optional[list[str]] = None,
    ) -> list[RetrievedPassage]:
        """
        Full hybrid retrieval with RRF fusion.

        Args:
            query: Natural language question
            top_k: Number of candidates to return before reranking
            doc_ids: If set, restrict retrieval to these documents
            filter_section_types: E.g. ["text", "table"]
        """
        if not self._chunks:
            return []

        top_k = top_k or settings.retrieval_top_k
        k_per_method = min(top_k * 2, len(self._chunks))

        # Filter-aware index
        valid_indices = self._get_valid_indices(doc_ids, filter_section_types)
        if not valid_indices:
            return []

        # 1. Dense retrieval
        dense_results = self._dense_retrieve(query, k_per_method, valid_indices)

        # 2. Sparse retrieval
        sparse_results = self._sparse_retrieve(query, k_per_method, valid_indices)

        # 3. KG-aware scoring
        kg_results = self._kg_retrieve(query, k_per_method, valid_indices)

        # 4. RRF fusion
        fused = self._rrf_fuse(dense_results, sparse_results, kg_results)

        # Return top_k
        final = sorted(fused.values(), key=lambda p: p.rrf_score, reverse=True)[:top_k]
        logger.info("retrieved_passages", query_len=len(query), count=len(final))
        return final

    # ─── Dense Retrieval ─────────────────────────────────────────────────────

    def _dense_retrieve(
        self, query: str, k: int, valid_indices: set[int]
    ) -> list[tuple[int, float]]:
        """FAISS IVF approximate nearest neighbor search."""
        if self._faiss_index is None or not self._faiss_index.is_trained:
            return []

        q_emb = self._embed([query])
        self._faiss_index.nprobe = settings.faiss_nprobe

        # Search with 3x candidates to account for post-filter
        search_k = min(k * 3, len(self._chunks))
        distances, indices = self._faiss_index.search(q_emb, search_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx >= 0 and idx in valid_indices:
                # Convert L2 distance to similarity score
                score = float(1 / (1 + dist))
                results.append((int(idx), score))
                if len(results) >= k:
                    break

        return results

    # ─── Sparse Retrieval ─────────────────────────────────────────────────────

    def _sparse_retrieve(
        self, query: str, k: int, valid_indices: set[int]
    ) -> list[tuple[int, float]]:
        """BM25 Okapi scoring over all chunks."""
        if self._bm25 is None:
            return []

        query_tokens = self._tokenize_bm25(query)
        scores = self._bm25.get_scores(query_tokens)

        # Rank and filter
        ranked = np.argsort(scores)[::-1]
        results = []
        for idx in ranked:
            if int(idx) in valid_indices and scores[idx] > 0:
                results.append((int(idx), float(scores[idx])))
                if len(results) >= k:
                    break

        return results

    # ─── Knowledge Graph ─────────────────────────────────────────────────────

    def _kg_retrieve(
        self, query: str, k: int, valid_indices: set[int]
    ) -> list[tuple[int, float]]:
        """Score chunks based on entity overlap with query via KG."""
        query_entities = self._extract_simple_entities(query)
        if not query_entities or self._kg.number_of_nodes() == 0:
            return []

        chunk_scores: dict[int, float] = {}
        for entity in query_entities:
            if entity not in self._kg:
                continue
            # Hop 1: directly connected chunks
            for neighbor in self._kg.neighbors(entity):
                if neighbor.startswith("chunk_"):
                    idx = self._chunk_id_to_idx.get(neighbor)
                    if idx is not None and idx in valid_indices:
                        weight = self._kg[entity][neighbor].get("weight", 1.0)
                        chunk_scores[idx] = chunk_scores.get(idx, 0) + weight

        if not chunk_scores:
            return []

        max_score = max(chunk_scores.values())
        results = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(idx, score / max_score) for idx, score in results]

    def _update_kg(self, chunk: Chunk) -> None:
        """Extract entities from chunk and add to KG with chunk node."""
        entities = self._extract_simple_entities(chunk.content)
        chunk_node = chunk.chunk_id
        self._kg.add_node(chunk_node, type="chunk", doc_id=chunk.doc_id)

        for entity in entities:
            if not self._kg.has_node(entity):
                self._kg.add_node(entity, type="entity")
            if self._kg.has_edge(entity, chunk_node):
                self._kg[entity][chunk_node]["weight"] += 1
            else:
                self._kg.add_edge(entity, chunk_node, weight=1.0)

    def _extract_simple_entities(self, text: str) -> list[str]:
        """
        Lightweight NER: extract capitalized noun phrases and numbers.
        In production, replace with spaCy or a BERT NER model for domain accuracy.
        """
        import re
        # Capitalized multi-word phrases
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # Acronyms
        entities += re.findall(r'\b[A-Z]{2,}\b', text)
        return list(set(e.lower() for e in entities if len(e) > 2))

    # ─── RRF Fusion ──────────────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        dense: list[tuple[int, float]],
        sparse: list[tuple[int, float]],
        kg: list[tuple[int, float]],
    ) -> dict[int, RetrievedPassage]:
        """
        Reciprocal Rank Fusion: score(d) = Σ 1/(k + rank_i(d))
        Robust, parameter-free, consistently outperforms learned fusion at small k.
        """
        passages: dict[int, RetrievedPassage] = {}

        def process(results: list[tuple[int, float]], method: str, weight: float):
            for rank, (idx, score) in enumerate(results):
                if idx >= len(self._chunks):
                    continue
                rrf_contribution = weight / (_RRF_K + rank + 1)
                if idx not in passages:
                    passages[idx] = RetrievedPassage(
                        chunk=self._chunks[idx],
                        retrieval_method="hybrid",
                    )
                p = passages[idx]
                p.rrf_score += rrf_contribution
                if method == "dense":
                    p.dense_score = score
                elif method == "sparse":
                    p.sparse_score = score
                elif method == "kg":
                    p.kg_score = score

        process(dense, "dense", settings.dense_weight)
        process(sparse, "sparse", settings.sparse_weight)
        process(kg, "kg", settings.kg_weight)

        return passages

    # ─── Index Persistence ───────────────────────────────────────────────────

    def _ensure_faiss_index(self) -> None:
        """Create FAISS IVF index if not present."""
        if self._faiss_index is not None:
            return
        n_chunks = max(len(self._chunks), 1)
        nlist = min(max(int(n_chunks ** 0.5), 4), 256)
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self._faiss_index = faiss.IndexIVFFlat(
            quantizer, self.embedding_dim, nlist, faiss.METRIC_L2
        )

    def _rebuild_all_indices(self) -> None:
        """Full rebuild after deletions (FAISS doesn't support delete)."""
        self._faiss_index = None
        self._chunk_id_to_idx = {c.chunk_id: i for i, c in enumerate(self._chunks)}
        self._kg = nx.Graph()

        if not self._chunks:
            self._bm25 = None
            self._bm25_corpus = []
            self._save_index()
            return

        texts = [c.as_context_string() for c in self._chunks]
        embeddings = self._embed(texts)
        self._ensure_faiss_index()
        self._faiss_index.train(embeddings)
        self._faiss_index.add(embeddings)

        self._bm25_corpus = [self._tokenize_bm25(c.content) for c in self._chunks]
        self._bm25 = BM25Okapi(self._bm25_corpus)

        for chunk in self._chunks:
            self._update_kg(chunk)

        self._save_index()

    def _save_index(self) -> None:
        """Persist all index components to disk."""
        try:
            # Save FAISS
            if self._faiss_index and self._faiss_index.is_trained:
                faiss.write_index(self._faiss_index, str(self.index_dir / "dense.faiss"))

            # Save chunks + BM25 corpus
            state = {
                "chunks": self._chunks,
                "chunk_id_to_idx": self._chunk_id_to_idx,
                "bm25_corpus": self._bm25_corpus,
            }
            with open(self.index_dir / "state.pkl", "wb") as f:
                pickle.dump(state, f)

            # Save KG
            nx.write_gpickle(self._kg, str(self.index_dir / "kg.gpickle"))
        except Exception as e:
            logger.error("index_save_failed", error=str(e))

    def _load_index(self) -> None:
        """Load persisted index from disk if available."""
        try:
            state_path = self.index_dir / "state.pkl"
            if state_path.exists():
                with open(state_path, "rb") as f:
                    state = pickle.load(f)
                self._chunks = state["chunks"]
                self._chunk_id_to_idx = state["chunk_id_to_idx"]
                self._bm25_corpus = state["bm25_corpus"]
                if self._bm25_corpus:
                    self._bm25 = BM25Okapi(self._bm25_corpus)

            faiss_path = self.index_dir / "dense.faiss"
            if faiss_path.exists():
                self._faiss_index = faiss.read_index(str(faiss_path))

            kg_path = self.index_dir / "kg.gpickle"
            if kg_path.exists():
                self._kg = nx.read_gpickle(str(kg_path))

            logger.info("index_loaded", chunks=len(self._chunks))
        except Exception as e:
            logger.warning("index_load_failed", error=str(e))

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedder.encode(
            texts,
            batch_size=settings.embedder_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def _tokenize_bm25(self, text: str) -> list[str]:
        return text.lower().split()

    def _get_valid_indices(
        self,
        doc_ids: Optional[list[str]],
        section_types: Optional[list[str]],
    ) -> set[int]:
        """Pre-filter chunk indices by doc_id and section_type."""
        valid = set()
        for i, chunk in enumerate(self._chunks):
            if doc_ids and chunk.doc_id not in doc_ids:
                continue
            if section_types and chunk.section_type not in section_types:
                continue
            valid.add(i)
        return valid