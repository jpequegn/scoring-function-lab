"""Tests for the AdaptiveScorer."""

import pytest

from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task, TaskType
from scorers.adaptive import AdaptiveScorer, FailureMode


def _make_task() -> Task:
    return Task(
        name="test-task",
        description="A test task",
        input={"data": "hello"},
        ground_truth="hello",
        task_type=TaskType.EXTRACTION,
    )


class ProgrammableScorer(BaseScorer):
    """Returns a programmed sequence of scores."""

    def __init__(self, name: str, scores: list[float], cost: float = 0.0) -> None:
        super().__init__(name=name, description=f"Programmable {name}")
        self._scores = scores
        self._cost = cost
        self._i = 0

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        idx = min(self._i, len(self._scores) - 1)
        s = self._scores[idx]
        self._i += 1
        return ScorerResult(score=s, explanation=f"{self.name} scored", confidence=1.0, latency_ms=0, cost_usd=self._cost)


class TestAdaptiveScorerInit:
    def test_requires_at_least_two_scorers(self):
        with pytest.raises(ValueError, match="at least 2"):
            AdaptiveScorer([ProgrammableScorer("only", [0.5])])

    def test_starts_with_first_scorer(self):
        s1 = ProgrammableScorer("cheap", [0.5])
        s2 = ProgrammableScorer("expensive", [0.9])
        adaptive = AdaptiveScorer([s1, s2])
        assert adaptive.active_scorer.name == "cheap"

    def test_name_and_description(self):
        s1 = ProgrammableScorer("a", [0.5])
        s2 = ProgrammableScorer("b", [0.5])
        adaptive = AdaptiveScorer([s1, s2])
        assert adaptive.name == "adaptive"
        assert "switch" in adaptive.description.lower()


class TestStallingDetection:
    def test_switches_on_flat_scores(self):
        s1 = ProgrammableScorer("cheap", [0.5, 0.5, 0.5], cost=0.0)
        s2 = ProgrammableScorer("smart", [0.8], cost=0.003)
        adaptive = AdaptiveScorer([s1, s2])
        task = _make_task()

        # First two calls build up flat history
        adaptive.score("h1", task, 1)
        adaptive.score("h2", task, 2)
        # Third call should detect stalling and switch
        r3 = adaptive.score("h3", task, 3)
        assert adaptive.active_scorer.name == "smart"
        assert len(adaptive.switch_log) == 1
        assert adaptive.switch_log[0][2] == FailureMode.STALLING

    def test_no_switch_on_improving_scores(self):
        s1 = ProgrammableScorer("cheap", [0.3, 0.5, 0.7])
        s2 = ProgrammableScorer("smart", [0.9])
        adaptive = AdaptiveScorer([s1, s2])
        task = _make_task()

        for i in range(3):
            adaptive.score(f"h{i}", task, i + 1)
        assert adaptive.active_scorer.name == "cheap"
        assert len(adaptive.switch_log) == 0


class TestOscillationDetection:
    def test_switches_on_oscillating_scores(self):
        # up-down-up pattern
        s1 = ProgrammableScorer("noisy", [0.3, 0.6, 0.4, 0.7], cost=0.0)
        s2 = ProgrammableScorer("stable", [0.8], cost=0.001)
        adaptive = AdaptiveScorer([s1, s2])
        task = _make_task()

        for i in range(4):
            adaptive.score(f"h{i}", task, i + 1)

        # After 3+ scores with alternating deltas, should switch
        assert adaptive.active_scorer.name == "stable"
        assert any(log[2] == FailureMode.OSCILLATING for log in adaptive.switch_log)


class TestOverExplorationDetection:
    def test_switches_on_tiny_deltas(self):
        # Scores improving but barely
        scores = [0.5, 0.505, 0.509, 0.512]
        s1 = ProgrammableScorer("slow", scores, cost=0.0)
        s2 = ProgrammableScorer("sharp", [0.9], cost=0.002)
        adaptive = AdaptiveScorer([s1, s2])
        task = _make_task()

        for i in range(4):
            adaptive.score(f"h{i}", task, i + 1)

        assert adaptive.active_scorer.name == "sharp"
        assert any(log[2] == FailureMode.OVER_EXPLORING for log in adaptive.switch_log)


class TestSwitchChaining:
    def test_can_switch_through_multiple_scorers(self):
        s1 = ProgrammableScorer("tier1", [0.3, 0.3, 0.3])
        s2 = ProgrammableScorer("tier2", [0.5, 0.5, 0.5])
        s3 = ProgrammableScorer("tier3", [0.9])
        adaptive = AdaptiveScorer([s1, s2, s3])
        task = _make_task()

        # Stall on tier1 → switch to tier2
        for i in range(3):
            adaptive.score(f"h{i}", task, i + 1)
        assert adaptive.active_scorer.name == "tier2"

        # Stall on tier2 → switch to tier3
        for i in range(3, 6):
            adaptive.score(f"h{i}", task, i + 1)
        assert adaptive.active_scorer.name == "tier3"
        assert len(adaptive.switch_log) == 2

    def test_stays_at_last_scorer_when_exhausted(self):
        s1 = ProgrammableScorer("a", [0.3, 0.3, 0.3])
        s2 = ProgrammableScorer("b", [0.5, 0.5, 0.5])
        adaptive = AdaptiveScorer([s1, s2])
        task = _make_task()

        # Stall through both
        for i in range(6):
            adaptive.score(f"h{i}", task, i + 1)

        assert adaptive.active_scorer.name == "b"
        # Only 1 switch (a→b), can't go further
        assert len(adaptive.switch_log) == 1


class TestScoringBehavior:
    def test_explanation_includes_active_scorer_name(self):
        s1 = ProgrammableScorer("cheap", [0.5])
        s2 = ProgrammableScorer("smart", [0.9])
        adaptive = AdaptiveScorer([s1, s2])
        task = _make_task()

        r = adaptive.score("h", task, 1)
        assert "cheap" in r.explanation

    def test_cost_comes_from_active_scorer(self):
        s1 = ProgrammableScorer("free", [0.5], cost=0.0)
        s2 = ProgrammableScorer("paid", [0.9], cost=0.005)
        adaptive = AdaptiveScorer([s1, s2])
        task = _make_task()

        r1 = adaptive.score("h1", task, 1)
        assert r1.cost_usd == 0.0

    def test_switch_log_records_iteration(self):
        s1 = ProgrammableScorer("a", [0.5, 0.5, 0.5])
        s2 = ProgrammableScorer("b", [0.9])
        adaptive = AdaptiveScorer([s1, s2])
        task = _make_task()

        for i in range(3):
            adaptive.score(f"h{i}", task, i + 1)

        log = adaptive.switch_log
        assert log[0][0] == 3  # switched at iteration 3
        assert log[0][1] == "b"  # switched to scorer b


class TestIntegrationWithAgentLoop:
    def test_adaptive_in_agent_loop(self):
        from scorelab.loop import AgentLoop

        s1 = ProgrammableScorer("cheap", [0.3, 0.3, 0.3, 0.3, 0.3], cost=0.0)
        s2 = ProgrammableScorer("smart", [0.6, 0.9], cost=0.003)
        adaptive = AdaptiveScorer([s1, s2])
        task = _make_task()
        task.success_threshold = 0.85

        def agent(prompt: str, iteration: int, prev: str | None) -> tuple[str, int]:
            return f"h{iteration}", 100

        loop = AgentLoop(scorer=adaptive, task=task, agent_fn=agent)
        result = loop.run()

        assert result.converged
        assert len(adaptive.switch_log) >= 1
        # Started cheap, switched to smart, converged — total cost lower than all-smart
        cheap_iterations = adaptive.switch_log[0][0] - 1  # iterations before first switch
        assert cheap_iterations >= 1  # at least 1 cheap iteration saved cost
