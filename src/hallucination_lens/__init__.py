
"""Public exports for hallucination_lens."""

from .scorer import FaithfulnessResult, HallucinationScorer, SentenceScore

__version__ = "0.2.0"

__all__ = ["FaithfulnessResult", "HallucinationScorer", "SentenceScore", "__version__"]
