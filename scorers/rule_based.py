"""Rule-based scorer: pattern matching, regex, threshold rules."""

from __future__ import annotations

import re
import time

from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task


class RuleBasedScorer(BaseScorer):
    """Scores hypotheses using regex patterns and threshold rules.

    Zero cost, deterministic, fast. Best for structured extraction and factual queries.
    """

    def __init__(self, rules: list[str] | None = None) -> None:
        super().__init__(name="rule_based", description="Pattern matching and threshold rules")
        self.rules = rules or []

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        start = time.monotonic_ns()

        ground_truth = task.ground_truth

        if isinstance(ground_truth, list):
            score = self._score_list(hypothesis, ground_truth)
            explanation = f"Matched {int(score * len(ground_truth))}/{len(ground_truth)} expected items"
        elif isinstance(ground_truth, str):
            score = self._score_string(hypothesis, ground_truth)
            explanation = "Exact match" if score == 1.0 else "Partial or no match"
        elif isinstance(ground_truth, (int, float)):
            score = self._score_numeric(hypothesis, ground_truth)
            explanation = f"Numeric comparison against target {ground_truth}"
        else:
            score = 0.0
            explanation = "Unsupported ground_truth type for rule-based scoring"

        elapsed_ms = int((time.monotonic_ns() - start) / 1_000_000)

        return ScorerResult(
            score=score,
            explanation=explanation,
            confidence=1.0,
            latency_ms=elapsed_ms,
            cost_usd=0.0,
        )

    def _score_list(self, hypothesis: str, expected: list) -> float:
        if not expected:
            return 1.0
        found = sum(1 for item in expected if str(item).lower() in hypothesis.lower())
        return found / len(expected)

    def _score_string(self, hypothesis: str, expected: str) -> float:
        if expected.lower() == hypothesis.strip().lower():
            return 1.0
        if expected.lower() in hypothesis.lower():
            return 0.5
        return 0.0

    def _score_numeric(self, hypothesis: str, expected: int | float) -> float:
        numbers = re.findall(r"-?\d+\.?\d*", hypothesis)
        if not numbers:
            return 0.0
        closest = min(numbers, key=lambda n: abs(float(n) - expected))
        diff = abs(float(closest) - expected)
        if diff == 0:
            return 1.0
        return max(0.0, 1.0 - diff / (abs(expected) + 1))
