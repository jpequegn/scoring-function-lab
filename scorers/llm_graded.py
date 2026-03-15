"""LLM-graded scorer: asks an LLM to grade the hypothesis."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task


class LLMGradedScorer(BaseScorer):
    """Scores hypotheses by sending them to an LLM for evaluation.

    The LLM returns structured JSON: {"score": 0.7, "rationale": "..."}.
    Cost: ~150 tokens per scoring call.
    Best for: subjective quality, creative tasks, summarization.
    """

    def __init__(
        self,
        prompt: str = "Rate the quality of this response from 0.0 to 1.0",
        llm_call: Callable[[str], dict[str, Any]] | None = None,
        cost_per_call: float = 0.003,
    ) -> None:
        super().__init__(name="llm_graded", description="LLM-based quality grading")
        self.prompt = prompt
        self._llm_call = llm_call or self._default_llm_call
        self.cost_per_call = cost_per_call

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        start = time.monotonic_ns()

        grading_prompt = (
            f"{self.prompt}\n\n"
            f"Task: {task.description}\n"
            f"Ground truth: {task.ground_truth}\n"
            f"Hypothesis: {hypothesis}\n\n"
            "Respond with JSON: {\"score\": <0.0-1.0>, \"rationale\": \"<explanation>\"}"
        )

        result = self._llm_call(grading_prompt)
        llm_score = float(result.get("score", 0.0))
        rationale = result.get("rationale", "No rationale provided")

        llm_score = max(0.0, min(1.0, llm_score))
        elapsed_ms = int((time.monotonic_ns() - start) / 1_000_000)

        return ScorerResult(
            score=llm_score,
            explanation=rationale,
            confidence=0.8,
            latency_ms=elapsed_ms,
            cost_usd=self.cost_per_call,
        )

    @staticmethod
    def _default_llm_call(prompt: str) -> dict[str, Any]:
        """Placeholder LLM call. Replace with actual API integration."""
        return {"score": 0.5, "rationale": "Default placeholder — no LLM configured"}
