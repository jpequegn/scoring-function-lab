"""Convergence metrics for scoring function experiments."""

from __future__ import annotations

from dataclasses import dataclass

from scorelab.loop import LoopResult


@dataclass
class ConvergenceMetrics:
    """Metrics computed from a completed agent loop run.

    Attributes:
        iterations_to_converge: Number of iterations to converge, or None if didn't converge.
        final_score: Final score achieved.
        total_tokens: Total tokens consumed across all iterations.
        total_cost_usd: Total monetary cost across all iterations.
        churn_index: total_iterations / depth_of_win (lower is better).
        score_trajectory: Score at each iteration.
        stuck_episodes: Number of times score didn't improve between consecutive iterations.
        cost_per_score_point: total_cost / final_score — the unified comparison metric.
        efficiency_score: sum(positive deltas) / total_iterations.
    """

    iterations_to_converge: int | None
    final_score: float
    total_tokens: int
    total_cost_usd: float
    churn_index: float
    score_trajectory: list[float]
    stuck_episodes: int
    cost_per_score_point: float
    efficiency_score: float


def compute_metrics(result: LoopResult) -> ConvergenceMetrics:
    """Compute convergence metrics from a LoopResult."""
    trajectory = result.score_trajectory
    total_iterations = result.total_iterations

    # iterations_to_converge
    iterations_to_converge: int | None = None
    if result.converged:
        iterations_to_converge = total_iterations

    # stuck_episodes: count iterations where score didn't improve
    stuck_episodes = 0
    for i in range(1, len(trajectory)):
        if trajectory[i] <= trajectory[i - 1]:
            stuck_episodes += 1

    # churn_index: total_iterations / depth_of_win
    # depth_of_win = max score improvement from any single iteration
    depth_of_win = 0.0
    if len(trajectory) >= 1:
        # First iteration's "improvement" is its score itself (from 0)
        depth_of_win = trajectory[0]
        for i in range(1, len(trajectory)):
            delta = trajectory[i] - trajectory[i - 1]
            if delta > depth_of_win:
                depth_of_win = delta

    churn_index = total_iterations / depth_of_win if depth_of_win > 0 else float("inf")

    # cost_per_score_point
    final_score = result.final_score
    total_cost = result.total_cost_usd
    cost_per_score_point = total_cost / final_score if final_score > 0 else float("inf")

    # efficiency_score: sum of positive deltas / total_iterations
    positive_deltas = trajectory[0] if trajectory else 0.0  # first iteration contributes its score
    for i in range(1, len(trajectory)):
        delta = trajectory[i] - trajectory[i - 1]
        if delta > 0:
            positive_deltas += delta
    efficiency_score = positive_deltas / total_iterations if total_iterations > 0 else 0.0

    return ConvergenceMetrics(
        iterations_to_converge=iterations_to_converge,
        final_score=final_score,
        total_tokens=result.total_tokens,
        total_cost_usd=total_cost,
        churn_index=churn_index,
        score_trajectory=trajectory,
        stuck_episodes=stuck_episodes,
        cost_per_score_point=cost_per_score_point,
        efficiency_score=efficiency_score,
    )
