
"""Faithfulness scoring primitives for RAG outputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol, Sequence

import numpy as np


class EmbeddingModel(Protocol):
    """Protocol representing the minimum embedding model contract used by the scorer."""

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
    ) -> Any:
        """Return an embedding tensor/array for the provided texts."""


@dataclass(frozen=True)
class SentenceScore:
    """Holds the strongest context match score for one response sentence."""

    sentence: str
    max_similarity: float


@dataclass(frozen=True)
class FaithfulnessResult:
    """Represents an aggregated faithfulness score and supporting evidence."""

    score: float
    verdict: str
    threshold: float
    sentence_scores: list[SentenceScore]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into a JSON-friendly dictionary."""

        return {
            "score": self.score,
            "verdict": self.verdict,
            "threshold": self.threshold,
            "sentence_scores": [
                {
                    "sentence": item.sentence,
                    "max_similarity": item.max_similarity,
                }
                for item in self.sentence_scores
            ],
        }


class HallucinationScorer:
    """Computes context-grounded faithfulness score from embedding similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.6,
        model: EmbeddingModel | None = None,
    ) -> None:
        """Initialize the scorer with an embedding model and decision threshold."""

        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be between 0 and 1")

        self._model = model if model is not None else _load_default_model(model_name)
        self._threshold = threshold
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Return the model name used by this scorer."""

        return self._model_name

    @property
    def threshold(self) -> float:
        """Return the active faithfulness threshold."""

        return self._threshold

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Expose underlying embedding backend for controlled reuse."""

        return self._model

    def faithfulness_score(
        self,
        context: str,
        response: str,
        threshold: float | None = None,
    ) -> FaithfulnessResult:
        """Score how well a response is grounded in the given context."""

        active_threshold = self._threshold if threshold is None else float(threshold)
        if active_threshold < 0.0 or active_threshold > 1.0:
            raise ValueError("threshold must be between 0 and 1")

        context_sentences = _split_sentences(context)
        response_sentences = _split_sentences(response)

        if not response_sentences:
            return FaithfulnessResult(
                score=0.0,
                verdict="hallucinated",
                threshold=active_threshold,
                sentence_scores=[],
            )

        if not context_sentences:
            empty_scores = [SentenceScore(sentence=s, max_similarity=0.0) for s in response_sentences]
            return FaithfulnessResult(
                score=0.0,
                verdict="hallucinated",
                threshold=active_threshold,
                sentence_scores=empty_scores,
            )

        context_embeddings = _to_2d_array(
            self._model.encode(context_sentences, normalize_embeddings=True)
        )
        response_embeddings = _to_2d_array(
            self._model.encode(response_sentences, normalize_embeddings=True)
        )

        similarity_matrix = _cosine_similarity_matrix(response_embeddings, context_embeddings)
        max_similarities = np.max(similarity_matrix, axis=1)

        sentence_scores = [
            SentenceScore(sentence=sentence, max_similarity=float(similarity))
            for sentence, similarity in zip(response_sentences, max_similarities, strict=True)
        ]

        score = float(np.clip(np.mean(max_similarities), 0.0, 1.0))
        verdict = "faithful" if score >= active_threshold else "hallucinated"

        return FaithfulnessResult(
            score=score,
            verdict=verdict,
            threshold=active_threshold,
            sentence_scores=sentence_scores,
        )

    def batch_faithfulness_scores(
        self,
        pairs: Sequence[tuple[str, str]],
        threshold: float | None = None,
    ) -> list[FaithfulnessResult]:
        """Score multiple context-response pairs in order using shared model backend."""

        return [
            self.faithfulness_score(context=context, response=response, threshold=threshold)
            for context, response in pairs
        ]


def _load_default_model(model_name: str) -> EmbeddingModel:
    """Load the default sentence-transformers embedding model lazily."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _split_sentences(text: str) -> list[str]:
    """Split text into non-empty sentences while preserving readable segments."""

    candidate_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in candidate_sentences if sentence.strip()]


def _to_2d_array(value: Any) -> np.ndarray:
    """Convert a model output into a strict two-dimensional float array."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("embedding output must be one or two dimensional")
    return array


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix between two embedding matrices."""

    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return np.matmul(a_norm, b_norm.T)
