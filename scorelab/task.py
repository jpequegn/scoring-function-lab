"""Task definition for the scoring function lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskType(Enum):
    """Five task types, each testing different scoring challenges."""

    EXTRACTION = "extraction"
    CODE_FIX = "code_fix"
    SUMMARIZATION = "summarization"
    QUERY_ANSWER = "query_answer"
    REFACTOR = "refactor"


@dataclass
class Task:
    """A task definition for the agent loop.

    Attributes:
        name: Short identifier for the task.
        description: Natural language task description.
        input: Task inputs (transcript, code, etc.).
        ground_truth: Expected answer (for scoring).
        max_iterations: Maximum agent iterations before timeout.
        success_threshold: Score required to consider "done" (0.0–1.0).
        task_type: The category of task.
    """

    name: str
    description: str
    input: dict[str, Any]
    ground_truth: Any
    max_iterations: int = 10
    success_threshold: float = 0.85
    task_type: TaskType = TaskType.EXTRACTION

    def to_prompt(self) -> str:
        """Format the task as an agent prompt."""
        lines = [
            f"# Task: {self.name}",
            "",
            self.description,
            "",
            "## Inputs",
        ]

        for key, value in self.input.items():
            lines.append(f"- **{key}**: {value}")

        lines.append("")
        lines.append(f"## Constraints")
        lines.append(f"- Task type: {self.task_type.value}")
        lines.append(f"- Maximum iterations: {self.max_iterations}")
        lines.append(f"- Success threshold: {self.success_threshold}")

        return "\n".join(lines)
