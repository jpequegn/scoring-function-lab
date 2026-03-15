"""Tests for the Task dataclass and TaskType enum."""

from scorelab.task import Task, TaskType


class TestTaskType:
    def test_all_five_types_exist(self):
        assert len(TaskType) == 5
        expected = {"extraction", "code_fix", "summarization", "query_answer", "refactor"}
        assert {t.value for t in TaskType} == expected

    def test_enum_values(self):
        assert TaskType.EXTRACTION.value == "extraction"
        assert TaskType.CODE_FIX.value == "code_fix"
        assert TaskType.SUMMARIZATION.value == "summarization"
        assert TaskType.QUERY_ANSWER.value == "query_answer"
        assert TaskType.REFACTOR.value == "refactor"


class TestTask:
    def _make_task(self, **overrides) -> Task:
        defaults = {
            "name": "extract-startups",
            "description": "Extract all startup names from the transcript",
            "input": {"transcript": "Anthropic and OpenAI are leading."},
            "ground_truth": ["Anthropic", "OpenAI"],
        }
        defaults.update(overrides)
        return Task(**defaults)

    def test_defaults(self):
        task = self._make_task()
        assert task.max_iterations == 10
        assert task.success_threshold == 0.85
        assert task.task_type == TaskType.EXTRACTION

    def test_custom_values(self):
        task = self._make_task(
            max_iterations=20,
            success_threshold=0.9,
            task_type=TaskType.CODE_FIX,
        )
        assert task.max_iterations == 20
        assert task.success_threshold == 0.9
        assert task.task_type == TaskType.CODE_FIX

    def test_to_prompt_contains_name(self):
        task = self._make_task()
        prompt = task.to_prompt()
        assert "extract-startups" in prompt

    def test_to_prompt_contains_description(self):
        task = self._make_task()
        prompt = task.to_prompt()
        assert "Extract all startup names" in prompt

    def test_to_prompt_contains_inputs(self):
        task = self._make_task()
        prompt = task.to_prompt()
        assert "transcript" in prompt
        assert "Anthropic and OpenAI" in prompt

    def test_to_prompt_contains_constraints(self):
        task = self._make_task(task_type=TaskType.SUMMARIZATION)
        prompt = task.to_prompt()
        assert "summarization" in prompt
        assert "10" in prompt
        assert "0.85" in prompt

    def test_to_prompt_with_multiple_inputs(self):
        task = self._make_task(input={"code": "def foo(): pass", "tests": "test_foo()"})
        prompt = task.to_prompt()
        assert "code" in prompt
        assert "tests" in prompt

    def test_ground_truth_can_be_any_type(self):
        # List
        t1 = self._make_task(ground_truth=["a", "b"])
        assert t1.ground_truth == ["a", "b"]

        # String
        t2 = self._make_task(ground_truth="tests pass")
        assert t2.ground_truth == "tests pass"

        # Number
        t3 = self._make_task(ground_truth=5)
        assert t3.ground_truth == 5

        # Dict
        t4 = self._make_task(ground_truth={"metric": "complexity", "target": 3})
        assert t4.ground_truth["target"] == 3
