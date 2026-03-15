"""Tests for AgentLoop, Iteration, and LoopResult."""

from scorelab.loop import AgentLoop, Iteration, LoopResult, STUCK_THRESHOLD
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
    """Returns a fixed sequence of scores for testing."""

    def __init__(self, scores: list[float]) -> None:
        super().__init__(name="fixed", description="Fixed score sequence")
        self._scores = scores
        self._call_count = 0

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        idx = min(self._call_count, len(self._scores) - 1)
        s = self._scores[idx]
        self._call_count += 1
        return ScorerResult(score=s, explanation="fixed", confidence=1.0, latency_ms=0, cost_usd=0.001)


def _dummy_agent(prompt: str, iteration: int, prev: str | None) -> tuple[str, int]:
    return f"hypothesis_{iteration}", 100


class TestIteration:
    def test_fields(self):
        sr = ScorerResult(score=0.5, explanation="ok", confidence=1.0, latency_ms=1, cost_usd=0.0)
        it = Iteration(number=1, hypothesis="h", score=0.5, score_delta=0.5, tokens_used=100, cost_usd=0.0, converged=False, scorer_result=sr)
        assert it.number == 1
        assert it.hypothesis == "h"
        assert it.score == 0.5
        assert it.score_delta == 0.5
        assert not it.converged


class TestLoopResult:
    def test_properties(self):
        task = _make_task()
        sr = ScorerResult(score=0.9, explanation="ok", confidence=1.0, latency_ms=0, cost_usd=0.002)
        iterations = [
            Iteration(number=1, hypothesis="h1", score=0.5, score_delta=0.5, tokens_used=100, cost_usd=0.002, converged=False, scorer_result=sr),
            Iteration(number=2, hypothesis="h2", score=0.9, score_delta=0.4, tokens_used=120, cost_usd=0.002, converged=True, scorer_result=sr),
        ]
        result = LoopResult(task=task, scorer_name="test", iterations=iterations, converged=True, termination_reason="converged")
        assert result.final_score == 0.9
        assert result.total_iterations == 2
        assert result.total_tokens == 220
        assert result.total_cost_usd == 0.004
        assert result.score_trajectory == [0.5, 0.9]

    def test_empty_iterations(self):
        task = _make_task()
        result = LoopResult(task=task, scorer_name="test", iterations=[], converged=False, termination_reason="max_iterations")
        assert result.final_score == 0.0
        assert result.total_iterations == 0
        assert result.total_tokens == 0


class TestAgentLoop:
    def test_converges_when_threshold_met(self):
        task = _make_task(success_threshold=0.8)
        scorer = FixedScorer([0.3, 0.6, 0.9])
        loop = AgentLoop(scorer=scorer, task=task, agent_fn=_dummy_agent)
        result = loop.run()
        assert result.converged
        assert result.termination_reason == "converged"
        assert result.total_iterations == 3
        assert result.final_score == 0.9

    def test_stops_when_stuck(self):
        task = _make_task(success_threshold=0.9, max_iterations=20)
        # Score improves once then stays flat
        scores = [0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        scorer = FixedScorer(scores)
        loop = AgentLoop(scorer=scorer, task=task, agent_fn=_dummy_agent)
        result = loop.run()
        assert not result.converged
        assert result.termination_reason == "stuck"
        # 1 improving + STUCK_THRESHOLD non-improving = 1 + 3 = 4 (plus initial)
        assert result.total_iterations == 2 + STUCK_THRESHOLD

    def test_stops_at_max_iterations(self):
        task = _make_task(success_threshold=0.99, max_iterations=5)
        # Slowly improving but never reaching threshold
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        scorer = FixedScorer(scores)
        loop = AgentLoop(scorer=scorer, task=task, agent_fn=_dummy_agent)
        result = loop.run()
        assert not result.converged
        assert result.termination_reason == "max_iterations"
        assert result.total_iterations == 5

    def test_step_increments_iteration(self):
        task = _make_task()
        scorer = FixedScorer([0.5, 0.6])
        loop = AgentLoop(scorer=scorer, task=task, agent_fn=_dummy_agent)
        it1 = loop.step()
        assert it1.number == 1
        assert it1.hypothesis == "hypothesis_1"
        it2 = loop.step()
        assert it2.number == 2
        assert abs(it2.score_delta - 0.1) < 0.001

    def test_score_delta_computed_correctly(self):
        task = _make_task()
        scorer = FixedScorer([0.3, 0.7, 0.5])
        loop = AgentLoop(scorer=scorer, task=task, agent_fn=_dummy_agent)
        loop.step()
        it2 = loop.step()
        assert abs(it2.score_delta - 0.4) < 0.001
        it3 = loop.step()
        assert abs(it3.score_delta - (-0.2)) < 0.001

    def test_cost_accumulated(self):
        task = _make_task(success_threshold=0.99, max_iterations=3)
        scorer = FixedScorer([0.1, 0.2, 0.3])
        loop = AgentLoop(scorer=scorer, task=task, agent_fn=_dummy_agent)
        result = loop.run()
        assert abs(result.total_cost_usd - 0.003) < 0.0001

    def test_tokens_accumulated(self):
        task = _make_task(success_threshold=0.99, max_iterations=3)
        scorer = FixedScorer([0.1, 0.2, 0.3])
        loop = AgentLoop(scorer=scorer, task=task, agent_fn=_dummy_agent)
        result = loop.run()
        assert result.total_tokens == 300  # 3 iterations × 100 tokens

    def test_converges_on_first_iteration(self):
        task = _make_task(success_threshold=0.5)
        scorer = FixedScorer([0.9])
        loop = AgentLoop(scorer=scorer, task=task, agent_fn=_dummy_agent)
        result = loop.run()
        assert result.converged
        assert result.total_iterations == 1

    def test_stuck_resets_on_improvement(self):
        task = _make_task(success_threshold=0.99, max_iterations=20)
        # 2 flat, then improve, then 2 flat, then improve — never hits STUCK_THRESHOLD
        scores = [0.3, 0.3, 0.3, 0.5, 0.5, 0.5, 0.7, 0.7, 0.7, 0.8]
        scorer = FixedScorer(scores)
        loop = AgentLoop(scorer=scorer, task=task, agent_fn=_dummy_agent)
        result = loop.run()
        # First non-improving streak: iterations 2,3 (delta<=0), then iteration 4 improves → reset
        # Second: iterations 5,6, then 7 improves → reset
        # Third: iterations 8,9,10 → stuck after 3 non-improving
        assert result.termination_reason == "stuck"
