"""Semantic scorer: cosine similarity between hypothesis and ground truth embeddings."""

from __future__ import annotations

import math
import time
from typing import Callable

from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task


class SemanticScorer(BaseScorer):
    """Scores using cosine similarity between embeddings of hypothesis and ground truth.

    Zero LLM cost but misses semantic nuance.
    Best for: open-ended answers with known ground truth.
    """

    def __init__(
        self,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        super().__init__(name="semantic", description="Embedding cosine similarity to ground truth")
        self._embed = embedding_fn or self._default_embedding

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        start = time.monotonic_ns()

        ground_truth_str = (
            " ".join(str(item) for item in task.ground_truth)
            if isinstance(task.ground_truth, list)
            else str(task.ground_truth)
        )

        hyp_embedding = self._embed(hypothesis)
        gt_embedding = self._embed(ground_truth_str)

        similarity = self._cosine_similarity(hyp_embedding, gt_embedding)
        # Clamp to [0, 1] — cosine similarity can be negative
        clamped_score = max(0.0, min(1.0, similarity))

        elapsed_ms = int((time.monotonic_ns() - start) / 1_000_000)

        return ScorerResult(
            score=clamped_score,
            explanation=f"Cosine similarity: {similarity:.4f}",
            confidence=0.7,
            latency_ms=elapsed_ms,
            cost_usd=0.0,
        )

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError(f"Embedding dimensions must match: {len(a)} vs {len(b)}")
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _default_embedding(text: str) -> list[float]:
        """Simple character-frequency embedding as a placeholder.

        Replace with a real embedding model (e.g., all-MiniLM-L6-v2) for production.
        """
        # 26-dimensional: frequency of each letter a-z
        text_lower = text.lower()
        total = max(len(text_lower), 1)
        return [text_lower.count(chr(ord("a") + i)) / total for i in range(26)]
