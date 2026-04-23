
"""Command line interface for hallucination-lens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .scorer import HallucinationScorer
from .validators import validate_text_argument, validate_threshold


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""

    parser = argparse.ArgumentParser(
        prog="hallucination-lens",
        description="Score how faithful an LLM response is to the provided context.",
    )
    parser.add_argument("--context", help="Source context text")
    parser.add_argument("--response", help="Model response text")
    parser.add_argument(
        "--batch-file",
        help="Path to JSON file with [{\"context\": ..., \"response\": ...}]",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Faithfulness threshold between 0 and 1 (default: 0.6)",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model identifier",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output",
    )
    return parser


def _load_batch_pairs(file_path: str) -> list[tuple[str, str]]:
    """Load and validate batch scoring context-response pairs from a JSON file."""

    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("batch-file must contain a non-empty JSON array")

    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"batch item at index {index} must be an object")
        context = validate_text_argument("context", str(item.get("context", "")))
        response = validate_text_argument("response", str(item.get("response", "")))
        pairs.append((context, response))

    return pairs


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI application and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        threshold = validate_threshold(args.threshold)

        scorer = HallucinationScorer(model_name=args.model_name, threshold=threshold)

        if args.batch_file:
            pairs = _load_batch_pairs(args.batch_file)
            results = scorer.batch_faithfulness_scores(pairs=pairs, threshold=threshold)

            verdict_counts = {"faithful": 0, "hallucinated": 0}
            serialized = []
            for index, item in enumerate(results):
                verdict_counts[item.verdict] = verdict_counts.get(item.verdict, 0) + 1
                serialized.append(
                    {
                        "index": index,
                        "score": round(item.score, 6),
                        "verdict": item.verdict,
                        "sentence_scores": [
                            {
                                "sentence": sentence.sentence,
                                "max_similarity": round(sentence.max_similarity, 6),
                            }
                            for sentence in item.sentence_scores
                        ],
                    }
                )

            payload = {
                "item_count": len(serialized),
                "average_score": round(sum(item["score"] for item in serialized) / len(serialized), 6),
                "threshold": threshold,
                "model_name": scorer.model_name,
                "verdict_counts": verdict_counts,
                "results": serialized,
            }
        else:
            if not args.context or not args.response:
                raise ValueError("--context and --response are required unless --batch-file is provided")

            context = validate_text_argument("context", args.context)
            response = validate_text_argument("response", args.response)
            result = scorer.faithfulness_score(context=context, response=response)

            payload = {
                "confidence": round(result.score, 6),
                "verdict": result.verdict,
                "threshold": result.threshold,
                "model_name": scorer.model_name,
                "sentence_scores": [
                    {
                        "sentence": item.sentence,
                        "max_similarity": round(item.max_similarity, 6),
                    }
                    for item in result.sentence_scores
                ],
            }

        indent = 2 if args.pretty else None
        print(json.dumps(payload, indent=indent))
        return 0
    except Exception as exc:  # pragma: no cover - defensive top-level CLI boundary
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
