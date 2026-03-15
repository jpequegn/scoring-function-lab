"""Adaptive scorer: switches strategies mid-loop based on convergence signal quality."""

from __future__ import annotations

import time
from enum import Enum

from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task


class FailureMode(Enum):
    """Detected convergence failure modes."""

    NONE = "none"
    STALLING = "stalling"          # 2+ flat scores — scorer can't differentiate
    OSCILLATING = "oscillating"    # score going up-down-up — signal too noisy
    OVER_EXPLORING = "exploring"   # many tiny positive deltas — converging too slowly


class AdaptiveScorer(BaseScorer):
    """Monitors its own convergence signal and switches strategies mid-loop.

    Starts with the cheapest scorer and promotes to more expensive ones
    when it detects the current strategy is failing:

    - Stalling (2+ flat scores) → switch to a scorer with more judgment
    - Oscillating (up-down-up) → switch to a scorer with more stability
    - Over-exploring (tiny deltas) → switch to a scorer with sharper signal

    The adaptive scorer gets cheap iterations cheap and only pays for
    expensive scoring when it matters.
    """

    STALL_WINDOW = 2       # consecutive flat scores to detect stalling
    OSCILLATION_WINDOW = 3 # minimum scores to detect oscillation
    MIN_DELTA = 0.02       # deltas below this are "tiny"
    EXPLORE_WINDOW = 3     # consecutive tiny-delta iterations to detect over-exploration

    def __init__(self, scorers: list[BaseScorer]) -> None:
        if len(scorers) < 2:
            raise ValueError("AdaptiveScorer requires at least 2 scorers to switch between")

        super().__init__(
            name="adaptive",
            description="Self-optimizing scorer that switches strategies based on convergence signal",
        )
        self.scorers = scorers
        self._active_idx = 0
        self._score_history: list[float] = []
        self._switch_log: list[tuple[int, str, FailureMode]] = []

    @property
    def active_scorer(self) -> BaseScorer:
        return self.scorers[self._active_idx]

    @property
    def switch_log(self) -> list[tuple[int, str, FailureMode]]:
        """Log of (iteration, scorer_name, reason) for each switch."""
        return list(self._switch_log)

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        start = time.monotonic_ns()

        # Detect failure mode and switch if needed
        failure = self._detect_failure_mode()
        if failure != FailureMode.NONE:
            self._switch(iteration, failure)

        # Score with active scorer
        result = self.active_scorer.score(hypothesis, task, iteration)
        self._score_history.append(result.score)

        elapsed_ms = int((time.monotonic_ns() - start) / 1_000_000)

        return ScorerResult(
            score=result.score,
            explanation=f"[{self.active_scorer.name}] {result.explanation}",
            confidence=result.confidence,
            latency_ms=elapsed_ms,
            cost_usd=result.cost_usd,
        )

    def _detect_failure_mode(self) -> FailureMode:
        h = self._score_history
        if len(h) < 2:
            return FailureMode.NONE

        # Check stalling: last N scores are identical (or nearly)
        if len(h) >= self.STALL_WINDOW:
            recent = h[-self.STALL_WINDOW:]
            if all(abs(recent[i] - recent[i - 1]) < 1e-6 for i in range(1, len(recent))):
                return FailureMode.STALLING

        # Check oscillation: alternating up-down pattern
        if len(h) >= self.OSCILLATION_WINDOW:
            deltas = [h[i] - h[i - 1] for i in range(-self.OSCILLATION_WINDOW + 1, 0)]
            sign_changes = sum(
                1 for i in range(1, len(deltas))
                if deltas[i] * deltas[i - 1] < 0  # opposite signs
            )
            if sign_changes >= len(deltas) - 1:
                return FailureMode.OSCILLATING

        # Check over-exploration: many tiny positive deltas
        if len(h) >= self.EXPLORE_WINDOW:
            recent_deltas = [h[i] - h[i - 1] for i in range(-self.EXPLORE_WINDOW + 1, 0)]
            if all(0 < d < self.MIN_DELTA for d in recent_deltas):
                return FailureMode.OVER_EXPLORING

        return FailureMode.NONE

    def _switch(self, iteration: int, reason: FailureMode) -> None:
        """Advance to the next scorer in the chain."""
        next_idx = self._active_idx + 1
        if next_idx >= len(self.scorers):
            return  # already at last scorer, can't switch further

        self._active_idx = next_idx
        self._switch_log.append((iteration, self.active_scorer.name, reason))
        # Reset history after switch so the new scorer gets a fresh signal window
        self._score_history.clear()
