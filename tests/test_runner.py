"""Tests for the ExperimentRunner."""

from scorelab.runner import ExperimentRunner, ComparisonReport, MatrixReport, ScorerReport
from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task, TaskType


def _make_task(name: str = "test-task", threshold: float = 0.85, max_iter: int = 5) -> Task:
    return Task(
        name=name,
        description="A test task",
        input={"data": "hello"},
        ground_truth="hello",
        max_iterations=max_iter,
        success_threshold=threshold,
        task_type=TaskType.EXTRACTION,
    )


class FixedScorer(BaseScorer):
    def __init__(self, name: str, scores: list[float], cost: float = 0.001) -> None:
        super().__init__(name=name, description=f"Fixed scorer {name}")
        self._scores = scores
        self._cost = cost
        self._i = 0

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        idx = min(self._i, len(self._scores) - 1)
        s = self._scores[idx]
        self._i += 1
        return ScorerResult(score=s, explanation="fixed", confidence=1.0, latency_ms=0, cost_usd=self._cost)


def _agent(prompt: str, iteration: int, prev: str | None) -> tuple[str, int]:
    return f"h{iteration}", 100


class TestExperimentRunner:
    def test_compare_returns_comparison_report(self):
        task = _make_task(threshold=0.5)
        fast = FixedScorer("fast", [0.9], cost=0.001)
        slow = FixedScorer("slow", [0.2, 0.4, 0.6], cost=0.002)
        runner = ExperimentRunner(agent_fn=_agent)
        report = runner.compare(task, [fast, slow], runs=1)
        assert isinstance(report, ComparisonReport)
        assert len(report.scorer_reports) == 2
        assert report.task.name == "test-task"

    def test_compare_picks_winner_by_cps(self):
        task = _make_task(threshold=0.5)
        # "cheap" converges in 1 iter at cost 0.001 → CPS = 0.001/0.9
        cheap = FixedScorer("cheap", [0.9], cost=0.001)
        # "expensive" converges in 1 iter at cost 0.01 → CPS = 0.01/0.9
        expensive = FixedScorer("expensive", [0.9], cost=0.01)
        runner = ExperimentRunner(agent_fn=_agent)
        report = runner.compare(task, [cheap, expensive], runs=1)
        assert report.winner == "cheap"

    def test_compare_multiple_runs(self):
        task = _make_task(threshold=0.5)
        scorer = FixedScorer("s1", [0.9], cost=0.001)
        runner = ExperimentRunner(agent_fn=_agent)
        report = runner.compare(task, [scorer], runs=3)
        sr = report.scorer_reports[0]
        assert len(sr.runs) == 3
        assert len(sr.metrics) == 3
        assert sr.convergence_rate == 1.0

    def test_scorer_report_averages(self):
        task = _make_task(threshold=0.5)
        scorer = FixedScorer("s1", [0.9], cost=0.002)
        runner = ExperimentRunner(agent_fn=_agent)
        report = runner.compare(task, [scorer], runs=2)
        sr = report.scorer_reports[0]
        assert sr.avg_final_score == 0.9
        assert abs(sr.avg_cost_usd - 0.002) < 0.0001
        assert sr.avg_iterations == 1.0

    def test_convergence_rate(self):
        task = _make_task(threshold=0.95, max_iter=2)
        # Score 0.5 never meets 0.95 threshold
        scorer = FixedScorer("never", [0.5], cost=0.001)
        runner = ExperimentRunner(agent_fn=_agent)
        report = runner.compare(task, [scorer], runs=3)
        sr = report.scorer_reports[0]
        assert sr.convergence_rate == 0.0

    def test_run_matrix(self):
        t1 = _make_task(name="task-a", threshold=0.5)
        t2 = _make_task(name="task-b", threshold=0.5)
        s1 = FixedScorer("s1", [0.9], cost=0.001)
        s2 = FixedScorer("s2", [0.9], cost=0.005)
        runner = ExperimentRunner(agent_fn=_agent)
        matrix = runner.run_matrix([t1, t2], [s1, s2], runs=1)
        assert isinstance(matrix, MatrixReport)
        assert len(matrix.comparisons) == 2
        assert "task-a" in matrix.task_winners
        assert "task-b" in matrix.task_winners

    def test_run_matrix_winners(self):
        t1 = _make_task(name="task-a", threshold=0.5)
        cheap = FixedScorer("cheap", [0.9], cost=0.001)
        pricey = FixedScorer("pricey", [0.9], cost=0.01)
        runner = ExperimentRunner(agent_fn=_agent)
        matrix = runner.run_matrix([t1], [cheap, pricey], runs=1)
        assert matrix.task_winners["task-a"] == "cheap"

    def test_reports_sorted_by_scorer_name(self):
        task = _make_task(threshold=0.5)
        b = FixedScorer("beta", [0.9])
        a = FixedScorer("alpha", [0.9])
        runner = ExperimentRunner(agent_fn=_agent)
        report = runner.compare(task, [b, a], runs=1)
        names = [sr.scorer_name for sr in report.scorer_reports]
        assert names == ["alpha", "beta"]
