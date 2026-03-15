"""Scorer interface and result type for the scoring function lab."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from scorelab.task import Task


@dataclass
class ScorerResult:
    """Result of scoring a hypothesis.

    Attributes:
        score: Numeric score from 0.0 (worst) to 1.0 (best).
        explanation: Human-readable rationale for the score.
        confidence: How confident the scorer is in this score (0.0–1.0).
        latency_ms: Time taken to compute the score in milliseconds.
        cost_usd: Monetary cost of this scoring call (e.g. LLM API cost).
    """

    score: float
    explanation: str
    confidence: float
    latency_ms: int
    cost_usd: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must be non-negative, got {self.latency_ms}")
        if self.cost_usd < 0:
            raise ValueError(f"cost_usd must be non-negative, got {self.cost_usd}")


class BaseScorer(ABC):
    """Abstract base class for all scorers.

    Subclasses must implement `score()` and set `name` and `description`.
    """

    name: str
    description: str

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        """Score a hypothesis against a task.

        Args:
            hypothesis: The agent's current answer/attempt.
            task: The task being solved.
            iteration: Current iteration number (1-indexed).

        Returns:
            A ScorerResult with score, explanation, confidence, latency, and cost.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
