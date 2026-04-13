"""
models/reranker.py — Cross-encoder reranker + adversarial robustness filter.

Two components:

1. CrossEncoderReranker:
   Uses a cross-encoder (jointly encodes query + passage) for fine-grained
   relevance scoring. Much more accurate than bi-encoder similarity alone,
   at the cost of running inference per (query, passage) pair.
   Implements the ColBERT-style passage scoring from the literature review.

2. AdversarialFilter:
   Detects potentially adversarially-crafted inputs (TextFooler-style attacks)
   by running ELECTRA discriminator to flag token-level anomalies.
   Reduces confidence for flagged inputs — inspired by BERT-based QA under
   Adversarial Attacks paper (paper #20) but applied proactively at inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForPreTraining,
)
from core.config import settings
from core.logging import get_logger
from utils.retriever import RetrievedPassage
from utils.chunker import Chunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for passage relevance scoring.

    Unlike bi-encoders that embed query and passage separately,
    cross-encoders see BOTH at once → much richer interaction modeling.

    Trade-off: O(k) inference passes instead of O(1) + ANN search.
    We apply this only on top-k candidates (k=20) from hybrid retrieval,
    then return top-r (r=5) for the reader. This is the standard pipeline.
    """

    def __init__(self):
        logger.info("loading_reranker", model=settings.reranker_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            settings.reranker_model_name,
            cache_dir=str(settings.model_cache_dir),
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            settings.reranker_model_name,
            cache_dir=str(settings.model_cache_dir),
        )
        self.device = torch.device(settings.reader_model_device)
        self.model.to(self.device)
        self.model.eval()

    def rerank(
        self,
        query: str,
        passages: list[RetrievedPassage],
        top_k: int = None,
    ) -> list[Chunk]:
        """
        Score each passage against the query and return top_k chunks.

        Returns Chunk objects (not RetrievedPassage) for clean handoff to reader.
        """
        if not passages:
            return []

        top_k = top_k or settings.rerank_top_k

        # Score all passages in batch
        pairs = [(query, p.chunk.as_context_string()) for p in passages]
        scores = self._score_pairs(pairs)

        # Attach reranker scores and sort
        scored = sorted(
            zip(scores, passages),
            key=lambda x: x[0],
            reverse=True,
        )

        # Update rrf_score with cross-encoder score (weighted blend)
        result_chunks = []
        for rank, (ce_score, passage) in enumerate(scored[:top_k]):
            # Blend: 70% cross-encoder, 30% hybrid RRF
            final_score = 0.7 * ce_score + 0.3 * passage.rrf_score
            passage.rrf_score = final_score  # update in-place for logging
            result_chunks.append(passage.chunk)

        logger.info("reranked", query_len=len(query), from_k=len(passages), to_k=len(result_chunks))
        return result_chunks

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Batch score query-passage pairs."""
        if not pairs:
            return []

        encodings = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits.squeeze(-1)
            # If binary classification model (relevance/non-relevance)
            if logits.dim() > 1:
                scores = torch.softmax(logits, dim=-1)[:, 1]
            else:
                scores = torch.sigmoid(logits)

        return scores.cpu().numpy().tolist()


class AdversarialFilter:
    """
    Proactive adversarial detection using ELECTRA discriminator.

    ELECTRA is trained to distinguish real tokens from replaced (corrupted) ones.
    TextFooler-style attacks work by replacing words with synonyms — exactly what
    ELECTRA is trained to detect.

    How it works:
    1. Run ELECTRA discriminator on the input text
    2. Count the fraction of tokens flagged as "replaced"
    3. If above threshold → flag as potentially adversarial
    4. Reduce answer confidence by adversarial_score_penalty

    This gives robustness without requiring adversarial training data,
    improving on paper #20 which only evaluates post-hoc robustness.
    """

    def __init__(self):
        if not settings.enable_adversarial_filter:
            self.enabled = False
            return

        logger.info("loading_adversarial_filter", model=settings.robustness_model_name)
        self.enabled = True
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.robustness_model_name,
                cache_dir=str(settings.model_cache_dir),
            )
            self.model = AutoModelForPreTraining.from_pretrained(
                settings.robustness_model_name,
                cache_dir=str(settings.model_cache_dir),
            )
            self.device = torch.device(settings.reader_model_device)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning("adversarial_filter_load_failed", error=str(e))
            self.enabled = False

    def score(self, text: str) -> "AdversarialScore":
        """
        Returns an AdversarialScore indicating how suspicious the input is.
        """
        if not self.enabled:
            return AdversarialScore(replaced_fraction=0.0, is_suspicious=False, penalty=0.0)

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # ELECTRA discriminator outputs per-token replaced logits
            if hasattr(outputs, "logits"):
                probs = torch.sigmoid(outputs.logits.squeeze(0))
                replaced_mask = (probs > 0.5).float()
                # Ignore [CLS] and [SEP] tokens
                replaced_fraction = float(replaced_mask[1:-1].mean().item())
            else:
                replaced_fraction = 0.0

            is_suspicious = replaced_fraction > (1 - settings.robustness_threshold)
            penalty = settings.adversarial_score_penalty if is_suspicious else 0.0

            return AdversarialScore(
                replaced_fraction=replaced_fraction,
                is_suspicious=is_suspicious,
                penalty=penalty,
            )

        except Exception as e:
            logger.warning("adversarial_scoring_failed", error=str(e))
            return AdversarialScore(replaced_fraction=0.0, is_suspicious=False, penalty=0.0)


@dataclass
class AdversarialScore:
    replaced_fraction: float   # 0.0 = all natural, 1.0 = all replaced
    is_suspicious: bool
    penalty: float             # confidence reduction to apply

    @property
    def risk_level(self) -> str:
        if self.replaced_fraction < 0.1:
            return "low"
        elif self.replaced_fraction < 0.3:
            return "medium"
        else:
            return "high"