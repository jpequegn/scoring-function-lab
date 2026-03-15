"""Tests for convergence metrics computation."""

import math

from scorelab.loop import AgentLoop, LoopResult
from scorelab.metrics import ConvergenceMetrics, compute_metrics
from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task, TaskType


def _make_task(**overrides) -> Task:
    defaults = {
        "name": "test-task",
        "description": "A test task",
        "input": {"data": "hello"},
        "ground_truth": "hello",
        "max_iterations": 10,
        "success_threshold": 0.85,
        "task_type": TaskType.EXTRACTION,
    }
    defaults.update(overrides)
    return Task(**defaults)


class FixedScorer(BaseScorer):
    def __init__(self, scores: list[float]) -> None:
        super().__init__(name="fixed", description="Fixed scores")
        self._scores = scores
        self._i = 0

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        idx = min(self._i, len(self._scores) - 1)
        s = self._scores[idx]
        self._i += 1
        return ScorerResult(score=s, explanation="fixed", confidence=1.0, latency_ms=0, cost_usd=0.002)


def _agent(prompt: str, iteration: int, prev: str | None) -> tuple[str, int]:
    return f"h{iteration}", 100


def _run_loop(scores: list[float], threshold: float = 0.85, max_iter: int = 10) -> LoopResult:
    task = _make_task(success_threshold=threshold, max_iterations=max_iter)
    scorer = FixedScorer(scores)
    loop = AgentLoop(scorer=scorer, task=task, agent_fn=_agent)
    return loop.run()


class TestConvergenceMetrics:
    def test_converged_run(self):
        result = _run_loop([0.3, 0.6, 0.9], threshold=0.85)
        m = compute_metrics(result)
        assert m.iterations_to_converge == 3
        assert m.final_score == 0.9
        assert m.total_tokens == 300
        assert abs(m.total_cost_usd - 0.006) < 0.0001
        assert m.score_trajectory == [0.3, 0.6, 0.9]

    def test_non_converged_run(self):
        result = _run_loop([0.1, 0.2, 0.3], threshold=0.99, max_iter=3)
        m = compute_metrics(result)
        assert m.iterations_to_converge is None
        assert m.final_score == 0.3

    def test_stuck_episodes(self):
        result = _run_loop([0.3, 0.3, 0.5, 0.5, 0.5, 0.5], threshold=0.99, max_iter=20)
        m = compute_metrics(result)
        # Trajectory: 0.3, 0.3, 0.5, 0.5, 0.5 (stuck detected at 3 non-improving after iter 3)
        assert m.stuck_episodes > 0

    def test_cost_per_score_point(self):
        result = _run_loop([0.5], threshold=0.5)
        m = compute_metrics(result)
        # cost = 0.002, score = 0.5 → CPS = 0.004
        assert abs(m.cost_per_score_point - 0.004) < 0.0001

    def test_cost_per_score_point_zero_score(self):
        result = _run_loop([0.0, 0.0, 0.0], threshold=0.99, max_iter=3)
        m = compute_metrics(result)
        assert m.cost_per_score_point == float("inf")

    def test_churn_index(self):
        result = _run_loop([0.3, 0.6, 0.9], threshold=0.85)
        m = compute_metrics(result)
        # depth_of_win = max(0.3, 0.3, 0.3) = 0.3, churn = 3 / 0.3 = 10
        assert abs(m.churn_index - 10.0) < 0.01

    def test_churn_index_no_progress(self):
        result = _run_loop([0.0, 0.0, 0.0], threshold=0.99, max_iter=3)
        m = compute_metrics(result)
        assert m.churn_index == float("inf")

    def test_efficiency_score(self):
        result = _run_loop([0.3, 0.6, 0.9], threshold=0.85)
        m = compute_metrics(result)
        # positive deltas: 0.3 + 0.3 + 0.3 = 0.9, efficiency = 0.9 / 3 = 0.3
        assert abs(m.efficiency_score - 0.3) < 0.01

    def test_efficiency_with_regression(self):
        result = _run_loop([0.5, 0.3, 0.7, 0.9], threshold=0.85)
        m = compute_metrics(result)
        # positive deltas: 0.5 (first) + 0.4 (0.3→0.7) + 0.2 (0.7→0.9) = 1.1
        # 0.3→0.7 is not right — the delta from 0.3 to 0.7 is 0.4
        # efficiency = 1.1 / 4 = 0.275
        assert abs(m.efficiency_score - 0.275) < 0.01

    def test_single_iteration_converge(self):
        result = _run_loop([0.95], threshold=0.9)
        m = compute_metrics(result)
        assert m.iterations_to_converge == 1
        assert m.stuck_episodes == 0
        assert m.efficiency_score == 0.95
