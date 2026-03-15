"""Agent loop with pluggable scorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task


@dataclass
class Iteration:
    """One iteration of the agent loop.

    Attributes:
        number: 1-indexed iteration number.
        hypothesis: What the agent tried this iteration.
        score: Score for this iteration (0.0–1.0).
        score_delta: Improvement over previous iteration's score.
        tokens_used: Tokens consumed this iteration.
        cost_usd: Cost of this iteration.
        converged: Whether this iteration met the success threshold.
        scorer_result: Full ScorerResult from the scorer.
    """

    number: int
    hypothesis: str
    score: float
    score_delta: float
    tokens_used: int
    cost_usd: float
    converged: bool
    scorer_result: ScorerResult


@dataclass
class LoopResult:
    """Result of a complete agent loop run.

    Attributes:
        task: The task that was run.
        scorer_name: Name of the scorer used.
        iterations: Full iteration history.
        converged: Whether the loop converged (met success threshold).
        termination_reason: Why the loop stopped.
    """

    task: Task
    scorer_name: str
    iterations: list[Iteration]
    converged: bool
    termination_reason: str

    @property
    def final_score(self) -> float:
        return self.iterations[-1].score if self.iterations else 0.0

    @property
    def total_iterations(self) -> int:
        return len(self.iterations)

    @property
    def total_tokens(self) -> int:
        return sum(it.tokens_used for it in self.iterations)

    @property
    def total_cost_usd(self) -> float:
        return sum(it.cost_usd for it in self.iterations)

    @property
    def score_trajectory(self) -> list[float]:
        return [it.score for it in self.iterations]


# Type for the agent function: takes (task prompt, iteration, previous hypothesis) → (hypothesis, tokens_used)
AgentFn = Callable[[str, int, str | None], tuple[str, int]]


STUCK_THRESHOLD = 3  # consecutive non-improving iterations before giving up


class AgentLoop:
    """Runs an agent loop with a pluggable scorer.

    The agent function generates hypotheses; the scorer evaluates them.
    The loop continues until convergence, stall, or max iterations.
    """

    def __init__(
        self,
        scorer: BaseScorer,
        task: Task,
        agent_fn: AgentFn,
    ) -> None:
        self.scorer = scorer
        self.task = task
        self.agent_fn = agent_fn
        self.iterations: list[Iteration] = []

    def step(self) -> Iteration:
        """Execute one agent iteration: generate hypothesis, score, record."""
        iteration_num = len(self.iterations) + 1
        prev_hypothesis = self.iterations[-1].hypothesis if self.iterations else None
        prev_score = self.iterations[-1].score if self.iterations else 0.0

        prompt = self.task.to_prompt()
        hypothesis, tokens_used = self.agent_fn(prompt, iteration_num, prev_hypothesis)

        scorer_result = self.scorer.score(hypothesis, self.task, iteration_num)

        iteration = Iteration(
            number=iteration_num,
            hypothesis=hypothesis,
            score=scorer_result.score,
            score_delta=scorer_result.score - prev_score,
            tokens_used=tokens_used,
            cost_usd=scorer_result.cost_usd,
            converged=scorer_result.score >= self.task.success_threshold,
            scorer_result=scorer_result,
        )

        self.iterations.append(iteration)
        return iteration

    def run(self) -> LoopResult:
        """Execute the full loop until termination. Returns LoopResult."""
        stuck_count = 0

        while len(self.iterations) < self.task.max_iterations:
            iteration = self.step()

            # Convergence: score meets threshold
            if iteration.converged:
                return self._result("converged")

            # Stuck detection: no improvement for N consecutive iterations
            if iteration.score_delta <= 0:
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count >= STUCK_THRESHOLD:
                return self._result("stuck")

        return self._result("max_iterations")

    def _result(self, reason: str) -> LoopResult:
        return LoopResult(
            task=self.task,
            scorer_name=self.scorer.name,
            iterations=list(self.iterations),
            converged=reason == "converged",
            termination_reason=reason,
        )
