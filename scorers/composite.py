"""Composite scorer: weighted combination of multiple scorers."""

from __future__ import annotations

import time

from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task


class CompositeScorer(BaseScorer):
    """Weighted combination of multiple scorers.

    Example: composite = 0.6 * rule_based + 0.4 * semantic.
    """

    def __init__(self, scorers: list[tuple[BaseScorer, float]]) -> None:
        if not scorers:
            raise ValueError("CompositeScorer requires at least one (scorer, weight) pair")

        total_weight = sum(w for _, w in scorers)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        names = "+".join(f"{s.name}({w})" for s, w in scorers)
        super().__init__(name=f"composite_{names}", description="Weighted combination of scorers")
        self.scorers = scorers

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        start = time.monotonic_ns()

        weighted_score = 0.0
        weighted_confidence = 0.0
        total_cost = 0.0
        explanations: list[str] = []

        for scorer, weight in self.scorers:
            result = scorer.score(hypothesis, task, iteration)
            weighted_score += result.score * weight
            weighted_confidence += result.confidence * weight
            total_cost += result.cost_usd
            explanations.append(f"{scorer.name}({weight}): {result.score:.2f} — {result.explanation}")

        elapsed_ms = int((time.monotonic_ns() - start) / 1_000_000)

        return ScorerResult(
            score=max(0.0, min(1.0, weighted_score)),
            explanation=" | ".join(explanations),
            confidence=max(0.0, min(1.0, weighted_confidence)),
            latency_ms=elapsed_ms,
            cost_usd=total_cost,
        )
