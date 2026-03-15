"""Tests for the Rich terminal renderer."""

from scorelab.loop import AgentLoop, Iteration, LoopResult
from scorelab.renderer import render_iteration, render_loop_result, render_comparison, render_matrix
from scorelab.runner import ComparisonReport, MatrixReport, ExperimentRunner
from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task, TaskType


def _make_task(name: str = "test-task", threshold: float = 0.85) -> Task:
    return Task(
        name=name,
        description="A test task",
        input={"data": "hello"},
        ground_truth="hello",
        max_iterations=10,
        success_threshold=threshold,
        task_type=TaskType.EXTRACTION,
    )


class FixedScorer(BaseScorer):
    def __init__(self, name: str = "fixed", scores: list[float] | None = None, cost: float = 0.001) -> None:
        super().__init__(name=name, description=f"Fixed scorer {name}")
        self._scores = scores or [0.5]
        self._cost = cost
        self._i = 0

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        idx = min(self._i, len(self._scores) - 1)
        s = self._scores[idx]
        self._i += 1
        return ScorerResult(score=s, explanation="fixed", confidence=1.0, latency_ms=0, cost_usd=self._cost)


def _agent(prompt: str, iteration: int, prev: str | None) -> tuple[str, int]:
    return f"hypothesis_{iteration}", 100


def _run(scores: list[float], threshold: float = 0.5) -> LoopResult:
    task = _make_task(threshold=threshold)
    scorer = FixedScorer(scores=scores)
    loop = AgentLoop(scorer=scorer, task=task, agent_fn=_agent)
    return loop.run()


class TestRenderIteration:
    def test_contains_task_name(self):
        result = _run([0.9])
        it = result.iterations[0]
        output = render_iteration(it, "my-task", "my-scorer", 10)
        assert "my-task" in output

    def test_contains_scorer_name(self):
        result = _run([0.9])
        it = result.iterations[0]
        output = render_iteration(it, "t", "rule_based", 10)
        assert "rule_based" in output

    def test_contains_score(self):
        result = _run([0.75])
        it = result.iterations[0]
        output = render_iteration(it, "t", "s", 10)
        assert "0.75" in output

    def test_contains_hypothesis(self):
        result = _run([0.9])
        it = result.iterations[0]
        output = render_iteration(it, "t", "s", 10)
        assert "hypothesis" in output


class TestRenderLoopResult:
    def test_contains_termination_reason(self):
        result = _run([0.9])
        output = render_loop_result(result)
        assert "converged" in output

    def test_contains_trajectory(self):
        result = _run([0.3, 0.6, 0.9], threshold=0.95)
        output = render_loop_result(result)
        assert "0.30" in output
        assert "0.90" in output

    def test_contains_cost(self):
        result = _run([0.9])
        output = render_loop_result(result)
        assert "$" in output


class TestRenderComparison:
    def test_contains_winner(self):
        task = _make_task(threshold=0.5)
        runner = ExperimentRunner(agent_fn=_agent)
        cheap = FixedScorer("cheap", [0.9], cost=0.001)
        pricey = FixedScorer("pricey", [0.9], cost=0.01)
        report = runner.compare(task, [cheap, pricey], runs=1)
        output = render_comparison(report)
        assert "cheap" in output
        assert "Winner" in output

    def test_contains_all_scorers(self):
        task = _make_task(threshold=0.5)
        runner = ExperimentRunner(agent_fn=_agent)
        s1 = FixedScorer("alpha", [0.9])
        s2 = FixedScorer("beta", [0.9])
        report = runner.compare(task, [s1, s2], runs=1)
        output = render_comparison(report)
        assert "alpha" in output
        assert "beta" in output

    def test_contains_task_name(self):
        task = _make_task(name="extract-startups", threshold=0.5)
        runner = ExperimentRunner(agent_fn=_agent)
        s = FixedScorer("s1", [0.9])
        report = runner.compare(task, [s], runs=1)
        output = render_comparison(report)
        assert "extract-startups" in output


class TestRenderMatrix:
    def test_contains_all_tasks(self):
        t1 = _make_task(name="task-a", threshold=0.5)
        t2 = _make_task(name="task-b", threshold=0.5)
        s = FixedScorer("s1", [0.9])
        runner = ExperimentRunner(agent_fn=_agent)
        matrix = runner.run_matrix([t1, t2], [s], runs=1)
        output = render_matrix(matrix)
        assert "task-a" in output
        assert "task-b" in output

    def test_contains_winners(self):
        t1 = _make_task(name="task-a", threshold=0.5)
        s = FixedScorer("best-scorer", [0.9])
        runner = ExperimentRunner(agent_fn=_agent)
        matrix = runner.run_matrix([t1], [s], runs=1)
        output = render_matrix(matrix)
        assert "best-scorer" in output
