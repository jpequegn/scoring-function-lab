"""Test runner scorer: executes tests and scores by fraction passing."""

from __future__ import annotations

import subprocess
import time

from scorelab.scorer import BaseScorer, ScorerResult
from scorelab.task import Task


class TestRunnerScorer(BaseScorer):
    """Scores by running a test command and computing fraction of tests passing.

    Binary per test, continuous overall. Best for code_fix and refactor tasks.
    """

    def __init__(self, test_command: str = "python3 -m pytest -x -q") -> None:
        super().__init__(name="test_runner", description="Run tests, score = fraction passing")
        self.test_command = test_command

    def score(self, hypothesis: str, task: Task, iteration: int) -> ScorerResult:
        start = time.monotonic_ns()

        test_dir = task.input.get("test_dir", "tests/")
        command = f"{self.test_command} {test_dir}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            passed, total = self._parse_pytest_output(result.stdout + result.stderr)
            test_score = passed / total if total > 0 else 0.0
            explanation = f"{passed}/{total} tests passed"
        except subprocess.TimeoutExpired:
            test_score = 0.0
            explanation = "Test execution timed out after 60s"
        except Exception as e:
            test_score = 0.0
            explanation = f"Test execution failed: {e}"

        elapsed_ms = int((time.monotonic_ns() - start) / 1_000_000)

        return ScorerResult(
            score=test_score,
            explanation=explanation,
            confidence=1.0,
            latency_ms=elapsed_ms,
            cost_usd=0.0,
        )

    @staticmethod
    def _parse_pytest_output(output: str) -> tuple[int, int]:
        """Parse pytest output to extract passed/total counts."""
        import re

        # Match "X passed" and "X failed" patterns
        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)
        error_match = re.search(r"(\d+) error", output)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        errors = int(error_match.group(1)) if error_match else 0

        total = passed + failed + errors
        return passed, total
