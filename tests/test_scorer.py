"""Tests for the BaseScorer interface and ScorerResult."""

import pytest

from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task, TaskType


class DummyScorer(BaseScorer):
    """Concrete scorer for testing the abstract interface."""

    def __init__(self) -> None:
        super().__init__(name="dummy", description="A test scorer")

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        return ScorerResult(
            score=0.5,
            explanation="dummy score",
            confidence=1.0,
            latency_ms=0,
            cost_usd=0.0,
        )


def _make_task() -> Task:
    return Task(
        name="test-task",
        description="A test task",
        input={"data": "hello"},
        ground_truth="hello",
        task_type=TaskType.EXTRACTION,
    )


class TestScorerResult:
    def test_valid_result(self):
        r = ScorerResult(score=0.75, explanation="good", confidence=0.9, latency_ms=10, cost_usd=0.001)
        assert r.score == 0.75
        assert r.explanation == "good"
        assert r.confidence == 0.9
        assert r.latency_ms == 10
        assert r.cost_usd == 0.001

    def test_boundary_values(self):
        ScorerResult(score=0.0, explanation="", confidence=0.0, latency_ms=0, cost_usd=0.0)
        ScorerResult(score=1.0, explanation="", confidence=1.0, latency_ms=0, cost_usd=0.0)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="score must be between"):
            ScorerResult(score=-0.1, explanation="", confidence=0.5, latency_ms=0, cost_usd=0.0)

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError, match="score must be between"):
            ScorerResult(score=1.1, explanation="", confidence=0.5, latency_ms=0, cost_usd=0.0)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence must be between"):
            ScorerResult(score=0.5, explanation="", confidence=1.5, latency_ms=0, cost_usd=0.0)

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError, match="latency_ms must be non-negative"):
            ScorerResult(score=0.5, explanation="", confidence=0.5, latency_ms=-1, cost_usd=0.0)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="cost_usd must be non-negative"):
            ScorerResult(score=0.5, explanation="", confidence=0.5, latency_ms=0, cost_usd=-0.01)


class TestBaseScorer:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseScorer(name="x", description="y")  # type: ignore[abstract]

    def test_concrete_scorer_has_name_and_description(self):
        s = DummyScorer()
        assert s.name == "dummy"
        assert s.description == "A test scorer"

    def test_score_returns_scorer_result(self):
        s = DummyScorer()
        result = s.score("test hypothesis", _make_task(), iteration=1)
        assert isinstance(result, ScorerResult)
        assert result.score == 0.5

    def test_repr(self):
        s = DummyScorer()
        assert repr(s) == "DummyScorer(name='dummy')"
