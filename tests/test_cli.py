
"""Unit tests for CLI behavior and output contract."""

import json
from pathlib import Path

from hallucination_lens import cli
from hallucination_lens.scorer import FaithfulnessResult, SentenceScore


class StubScorer:
    """Scorer stub that avoids loading real embedding models during CLI tests."""

    def __init__(self, model_name, threshold):
        """Store config for assertions if needed."""

        self.model_name = model_name
        self.threshold = threshold

    def faithfulness_score(self, context, response):
        """Return deterministic result payload for CLI contract testing."""

        return FaithfulnessResult(
            score=0.88,
            verdict="faithful",
            threshold=self.threshold,
            sentence_scores=[SentenceScore(sentence=response, max_similarity=0.88)],
        )

    def batch_faithfulness_scores(self, pairs, threshold=None):
        """Return deterministic per-item results for CLI batch-mode testing."""

        active_threshold = self.threshold if threshold is None else threshold
        return [
            FaithfulnessResult(
                score=0.88,
                verdict="faithful",
                threshold=active_threshold,
                sentence_scores=[SentenceScore(sentence=response, max_similarity=0.88)],
            )
            for _, response in pairs
        ]


def test_cli_success_returns_json(monkeypatch, capsys):
    """CLI should emit confidence JSON when valid inputs are provided."""

    monkeypatch.setattr(cli, "HallucinationScorer", StubScorer)

    exit_code = cli.main(
        [
            "--context",
            "Paris is the capital of France.",
            "--response",
            "Paris is in France.",
            "--pretty",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["verdict"] == "faithful"
    assert payload["confidence"] == 0.88


def test_cli_invalid_threshold_returns_error(capsys):
    """CLI should return non-zero exit code and error JSON for bad thresholds."""

    exit_code = cli.main(
        [
            "--context",
            "Paris is the capital of France.",
            "--response",
            "Paris is in France.",
            "--threshold",
            "2",
        ]
    )

    assert exit_code == 1
    error_output = capsys.readouterr().err
    payload = json.loads(error_output)
    assert "threshold" in payload["error"]


def test_cli_batch_mode_outputs_results(monkeypatch, tmp_path, capsys):
    """CLI batch mode should emit aggregate JSON and per-item results."""

    monkeypatch.setattr(cli, "HallucinationScorer", StubScorer)

    batch_file = tmp_path / "batch.json"
    batch_file.write_text(
        json.dumps(
            [
                {
                    "context": "Paris is the capital of France.",
                    "response": "Paris is in France.",
                },
                {
                    "context": "Paris is the capital of France.",
                    "response": "Paris is in France.",
                },
            ]
        ),
        encoding="utf-8",
    )

    exit_code = cli.main(["--batch-file", str(Path(batch_file)), "--pretty"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["item_count"] == 2
    assert len(payload["results"]) == 2
