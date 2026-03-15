"""Tests for the five scorer implementations."""

import pytest

from scorelab.scorer import ScorerResult
from scorelab.task import Task, TaskType
from scorers.composite import CompositeScorer
from scorers.llm_graded import LLMGradedScorer
from scorers.rule_based import RuleBasedScorer
from scorers.semantic import SemanticScorer
from scorers.test_runner import TestRunnerScorer


def _extraction_task() -> Task:
    return Task(
        name="extract-startups",
        description="Extract all startup names mentioned",
        input={"transcript": "Anthropic and OpenAI are leading AI companies."},
        ground_truth=["Anthropic", "OpenAI", "Mistral"],
        task_type=TaskType.EXTRACTION,
    )


def _string_task() -> Task:
    return Task(
        name="answer-question",
        description="Answer the question",
        input={"question": "What is 2+2?"},
        ground_truth="4",
        task_type=TaskType.QUERY_ANSWER,
    )


def _numeric_task() -> Task:
    return Task(
        name="compute-complexity",
        description="Compute cyclomatic complexity",
        input={"code": "def foo(): pass"},
        ground_truth=5,
        task_type=TaskType.REFACTOR,
    )


# --- RuleBasedScorer ---


class TestRuleBasedScorer:
    def test_list_ground_truth_full_match(self):
        task = _extraction_task()
        scorer = RuleBasedScorer()
        result = scorer.score("Anthropic, OpenAI, and Mistral are here", task, 1)
        assert result.score == 1.0
        assert result.cost_usd == 0.0
        assert result.confidence == 1.0

    def test_list_ground_truth_partial_match(self):
        task = _extraction_task()
        scorer = RuleBasedScorer()
        result = scorer.score("Anthropic is great", task, 1)
        assert abs(result.score - 1 / 3) < 0.01

    def test_list_ground_truth_no_match(self):
        task = _extraction_task()
        scorer = RuleBasedScorer()
        result = scorer.score("No companies mentioned", task, 1)
        assert result.score == 0.0

    def test_string_exact_match(self):
        task = _string_task()
        scorer = RuleBasedScorer()
        result = scorer.score("4", task, 1)
        assert result.score == 1.0

    def test_string_partial_match(self):
        task = _string_task()
        scorer = RuleBasedScorer()
        result = scorer.score("The answer is 4 obviously", task, 1)
        assert result.score == 0.5

    def test_string_no_match(self):
        task = _string_task()
        scorer = RuleBasedScorer()
        result = scorer.score("I don't know", task, 1)
        assert result.score == 0.0

    def test_numeric_exact_match(self):
        task = _numeric_task()
        scorer = RuleBasedScorer()
        result = scorer.score("The complexity is 5", task, 1)
        assert result.score == 1.0

    def test_numeric_close_match(self):
        task = _numeric_task()
        scorer = RuleBasedScorer()
        result = scorer.score("The complexity is 6", task, 1)
        assert 0.0 < result.score < 1.0

    def test_numeric_no_numbers(self):
        task = _numeric_task()
        scorer = RuleBasedScorer()
        result = scorer.score("No numbers here", task, 1)
        assert result.score == 0.0

    def test_returns_scorer_result(self):
        scorer = RuleBasedScorer()
        result = scorer.score("test", _extraction_task(), 1)
        assert isinstance(result, ScorerResult)


# --- LLMGradedScorer ---


class TestLLMGradedScorer:
    def test_default_placeholder(self):
        scorer = LLMGradedScorer()
        result = scorer.score("test hypothesis", _extraction_task(), 1)
        assert result.score == 0.5
        assert result.cost_usd == 0.003

    def test_custom_llm_call(self):
        def mock_llm(prompt: str) -> dict:
            return {"score": 0.9, "rationale": "Excellent extraction"}

        scorer = LLMGradedScorer(llm_call=mock_llm, cost_per_call=0.005)
        result = scorer.score("Anthropic, OpenAI, Mistral", _extraction_task(), 1)
        assert result.score == 0.9
        assert result.explanation == "Excellent extraction"
        assert result.cost_usd == 0.005

    def test_clamps_score_above_one(self):
        def bad_llm(prompt: str) -> dict:
            return {"score": 1.5, "rationale": "Over-scored"}

        scorer = LLMGradedScorer(llm_call=bad_llm)
        result = scorer.score("test", _extraction_task(), 1)
        assert result.score == 1.0

    def test_clamps_score_below_zero(self):
        def bad_llm(prompt: str) -> dict:
            return {"score": -0.5, "rationale": "Under-scored"}

        scorer = LLMGradedScorer(llm_call=bad_llm)
        result = scorer.score("test", _extraction_task(), 1)
        assert result.score == 0.0

    def test_confidence_is_0_8(self):
        scorer = LLMGradedScorer()
        result = scorer.score("test", _extraction_task(), 1)
        assert result.confidence == 0.8


# --- TestRunnerScorer ---


class TestTestRunnerScorer:
    def test_parse_pytest_all_passed(self):
        output = "3 passed in 0.01s"
        passed, total = TestRunnerScorer._parse_pytest_output(output)
        assert passed == 3
        assert total == 3

    def test_parse_pytest_some_failed(self):
        output = "2 passed, 1 failed in 0.05s"
        passed, total = TestRunnerScorer._parse_pytest_output(output)
        assert passed == 2
        assert total == 3

    def test_parse_pytest_with_errors(self):
        output = "1 passed, 1 failed, 1 error in 0.10s"
        passed, total = TestRunnerScorer._parse_pytest_output(output)
        assert passed == 1
        assert total == 3

    def test_parse_pytest_no_results(self):
        passed, total = TestRunnerScorer._parse_pytest_output("")
        assert passed == 0
        assert total == 0

    def test_name_and_description(self):
        scorer = TestRunnerScorer()
        assert scorer.name == "test_runner"
        assert "test" in scorer.description.lower()


# --- SemanticScorer ---


class TestSemanticScorer:
    def test_identical_text_scores_high(self):
        task = _extraction_task()
        scorer = SemanticScorer()
        gt_str = " ".join(str(i) for i in task.ground_truth)
        result = scorer.score(gt_str, task, 1)
        assert result.score > 0.99

    def test_similar_text_scores_moderate(self):
        task = _extraction_task()
        scorer = SemanticScorer()
        result = scorer.score("Anthropic and OpenAI", task, 1)
        assert 0.0 < result.score < 1.0

    def test_zero_cost(self):
        scorer = SemanticScorer()
        result = scorer.score("test", _extraction_task(), 1)
        assert result.cost_usd == 0.0

    def test_custom_embedding_fn(self):
        def mock_embed(text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

        scorer = SemanticScorer(embedding_fn=mock_embed)
        result = scorer.score("anything", _string_task(), 1)
        assert result.score == 1.0  # identical embeddings

    def test_cosine_similarity_orthogonal(self):
        sim = SemanticScorer._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 0.001

    def test_cosine_similarity_identical(self):
        sim = SemanticScorer._cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(sim - 1.0) < 0.001

    def test_cosine_similarity_dimension_mismatch(self):
        with pytest.raises(ValueError, match="dimensions must match"):
            SemanticScorer._cosine_similarity([1.0], [1.0, 2.0])


# --- CompositeScorer ---


class TestCompositeScorer:
    def test_weighted_combination(self):
        rule = RuleBasedScorer()
        semantic = SemanticScorer()
        composite = CompositeScorer([(rule, 0.6), (semantic, 0.4)])
        task = _extraction_task()
        result = composite.score("Anthropic, OpenAI, Mistral", task, 1)
        assert 0.0 <= result.score <= 1.0
        assert isinstance(result, ScorerResult)

    def test_aggregates_cost(self):
        def mock_llm(prompt: str) -> dict:
            return {"score": 0.8, "rationale": "Good"}

        rule = RuleBasedScorer()
        llm = LLMGradedScorer(llm_call=mock_llm, cost_per_call=0.01)
        composite = CompositeScorer([(rule, 0.5), (llm, 0.5)])
        result = composite.score("test", _extraction_task(), 1)
        assert result.cost_usd == 0.01  # 0.0 from rule + 0.01 from llm

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            CompositeScorer([(RuleBasedScorer(), 0.3), (SemanticScorer(), 0.3)])

    def test_empty_scorers_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CompositeScorer([])

    def test_explanation_includes_all_scorers(self):
        composite = CompositeScorer([(RuleBasedScorer(), 0.6), (SemanticScorer(), 0.4)])
        result = composite.score("test", _extraction_task(), 1)
        assert "rule_based" in result.explanation
        assert "semantic" in result.explanation
