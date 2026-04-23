
"""Unit tests for faithfulness scoring behavior."""

import numpy as np

from hallucination_lens.scorer import HallucinationScorer


class FakeEmbeddingModel:
    """Tiny deterministic embedding backend for fast tests."""

    def encode(self, texts, normalize_embeddings=True):
        """Return deterministic 3D vectors keyed by topic words."""

        vectors = []
        for text in texts:
            lower = text.lower()
            vector = np.array([0.1, 0.1, 0.1], dtype=np.float32)
            if "paris" in lower or "france" in lower:
                vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif "berlin" in lower or "germany" in lower:
                vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif "ocean" in lower:
                vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            vectors.append(vector)

        return np.vstack(vectors)


def test_faithful_response_scores_high():
    """Response aligned with context should score above threshold."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    result = scorer.faithfulness_score(
        context="Paris is the capital of France.",
        response="Paris is in France.",
    )

    assert result.verdict == "faithful"
    assert result.score >= 0.9
    assert len(result.sentence_scores) == 1


def test_hallucinated_response_scores_low():
    """Response unrelated to context should score below threshold."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    result = scorer.faithfulness_score(
        context="Paris is the capital of France.",
        response="Berlin is in Germany.",
    )

    assert result.verdict == "hallucinated"
    assert result.score < 0.6


def test_empty_context_returns_zero_score():
    """Empty context should always produce a hallucinated verdict."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    result = scorer.faithfulness_score(
        context="",
        response="Paris is in France.",
    )

    assert result.score == 0.0
    assert result.verdict == "hallucinated"
    assert result.sentence_scores[0].max_similarity == 0.0


def test_threshold_changes_verdict():
    """Higher thresholds should tighten the faithful decision boundary."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.95)
    result = scorer.faithfulness_score(
        context="Paris is the capital of France.",
        response="Paris is in France. The ocean is blue.",
    )

    assert result.score < 0.95
    assert result.verdict == "hallucinated"


def test_batch_scoring_preserves_input_order_and_returns_results():
    """Batch scoring should return one result per pair in original order."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    results = scorer.batch_faithfulness_scores(
        pairs=[
            ("Paris is the capital of France.", "Paris is in France."),
            ("Paris is the capital of France.", "Berlin is in Germany."),
        ]
    )

    assert len(results) == 2
    assert results[0].verdict == "faithful"
    assert results[1].verdict == "hallucinated"


def test_threshold_override_applies_without_mutating_default_threshold():
    """Per-call threshold override should not mutate scorer default threshold."""

    scorer = HallucinationScorer(model=FakeEmbeddingModel(), threshold=0.6)
    result = scorer.faithfulness_score(
        context="Paris is the capital of France.",
        response="Paris is in France. The ocean is blue.",
        threshold=0.95,
    )

    assert result.threshold == 0.95
    assert scorer.threshold == 0.6
