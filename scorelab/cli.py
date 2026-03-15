"""CLI for the scoring function lab."""

from __future__ import annotations

import os
from typing import Any

import click
from rich.console import Console

from scorelab.loop import AgentFn, AgentLoop
from scorelab.metrics import compute_metrics
from scorelab.renderer import render_comparison, render_iteration, render_loop_result, render_matrix
from scorelab.runner import ExperimentRunner
from scorelab.scorer import BaseScorer
from scorelab.task import Task, TaskType
from scorers.composite import CompositeScorer
from scorers.llm_graded import LLMGradedScorer
from scorers.rule_based import RuleBasedScorer
from scorers.semantic import SemanticScorer
from scorers.test_runner import TestRunnerScorer

console = Console()

# --- Registry ---

SAMPLE_TASKS: dict[str, Task] = {
    "extraction": Task(
        name="extract-startups",
        description="Extract all startup names mentioned in this transcript",
        input={"transcript": "Anthropic and OpenAI are leading AI companies. Mistral and Cohere are also notable."},
        ground_truth=["Anthropic", "OpenAI", "Mistral", "Cohere"],
        task_type=TaskType.EXTRACTION,
    ),
    "code_fix": Task(
        name="fix-bug",
        description="Fix the off-by-one error in this function",
        input={"code": "def get_last(items): return items[len(items)]", "tests": "assert get_last([1,2,3]) == 3"},
        ground_truth="tests pass",
        task_type=TaskType.CODE_FIX,
    ),
    "summarization": Task(
        name="summarize-episode",
        description="Summarize this podcast episode in under 50 words",
        input={"transcript": "Today we discussed how scoring functions drive agent behavior. The key insight is that the same task with different scoring produces completely different convergence patterns."},
        ground_truth="Scoring functions are the most important design decision in agent loops, dramatically affecting convergence behavior.",
        task_type=TaskType.SUMMARIZATION,
    ),
    "query_answer": Task(
        name="answer-question",
        description="Answer the factual question based on the provided context",
        input={"context": "The capital of France is Paris. It has a population of 2.1 million.", "question": "What is the capital of France?"},
        ground_truth="Paris",
        task_type=TaskType.QUERY_ANSWER,
    ),
    "refactor": Task(
        name="reduce-complexity",
        description="Refactor to reduce cyclomatic complexity to 3 or below",
        input={"code": "def f(x):\n  if x > 0:\n    if x > 10:\n      return 'big'\n    else:\n      return 'small'\n  else:\n    return 'negative'"},
        ground_truth=3,
        task_type=TaskType.REFACTOR,
    ),
}


def _get_scorer(name: str) -> BaseScorer:
    scorers: dict[str, BaseScorer] = {
        "rule_based": RuleBasedScorer(),
        "llm_graded": LLMGradedScorer(),
        "test_runner": TestRunnerScorer(),
        "semantic": SemanticScorer(),
        "composite": CompositeScorer([(RuleBasedScorer(), 0.6), (SemanticScorer(), 0.4)]),
    }
    if name not in scorers:
        raise click.BadParameter(f"Unknown scorer: {name}. Available: {', '.join(scorers)}")
    return scorers[name]


def _get_all_scorers() -> list[BaseScorer]:
    return [_get_scorer(name) for name in ["rule_based", "llm_graded", "semantic", "composite"]]


def _dummy_agent(prompt: str, iteration: int, prev: str | None) -> tuple[str, int]:
    """Placeholder agent that returns progressively better hypotheses."""
    if prev:
        return f"{prev} [refined iteration {iteration}]", 150
    return f"Initial attempt at solving the task (iteration {iteration})", 200


# --- CLI ---

TASK_NAMES = list(SAMPLE_TASKS.keys())
SCORER_NAMES = ["rule_based", "llm_graded", "test_runner", "semantic", "composite"]


@click.group()
def cli() -> None:
    """Scoring Function Lab — test and compare agent scoring functions."""


@cli.command()
@click.option("--task", "task_name", type=click.Choice(TASK_NAMES), required=True, help="Task to run")
@click.option("--scorer", "scorer_name", type=click.Choice(SCORER_NAMES), required=True, help="Scorer to use")
@click.option("--iterations", default=10, help="Max iterations")
def run(task_name: str, scorer_name: str, iterations: int) -> None:
    """Run a task with a specific scorer."""
    task = SAMPLE_TASKS[task_name]
    task.max_iterations = iterations
    scorer = _get_scorer(scorer_name)

    loop = AgentLoop(scorer=scorer, task=task, agent_fn=_dummy_agent)

    for _ in range(task.max_iterations):
        it = loop.step()
        render_iteration(it, task.name, scorer.name, task.max_iterations, console=console)
        console.print()
        if it.converged:
            break

    result = loop._result("converged" if loop.iterations[-1].converged else "max_iterations")
    render_loop_result(result, console=console)


@cli.command()
@click.option("--task", "task_name", type=click.Choice(TASK_NAMES), required=True, help="Task to compare scorers on")
@click.option("--runs", default=3, help="Number of runs per scorer")
def compare(task_name: str, runs: int) -> None:
    """Compare all scorers on a task."""
    task = SAMPLE_TASKS[task_name]
    scorers = _get_all_scorers()
    runner = ExperimentRunner(agent_fn=_dummy_agent)
    report = runner.compare(task, scorers, runs=runs)
    render_comparison(report, console=console)


@cli.command()
@click.option("--tasks", "task_filter", default="all", help="Comma-separated task names or 'all'")
@click.option("--scorers", "scorer_filter", default="all", help="Comma-separated scorer names or 'all'")
@click.option("--runs", default=3, help="Number of runs per scorer per task")
def matrix(task_filter: str, scorer_filter: str, runs: int) -> None:
    """Run full task x scorer matrix."""
    if task_filter == "all":
        tasks = list(SAMPLE_TASKS.values())
    else:
        names = [n.strip() for n in task_filter.split(",")]
        tasks = [SAMPLE_TASKS[n] for n in names if n in SAMPLE_TASKS]

    if scorer_filter == "all":
        scorers = _get_all_scorers()
    else:
        names = [n.strip() for n in scorer_filter.split(",")]
        scorers = [_get_scorer(n) for n in names]

    runner = ExperimentRunner(agent_fn=_dummy_agent)
    report = runner.run_matrix(tasks, scorers, runs=runs)
    render_matrix(report, console=console)


@cli.command()
def taxonomy() -> None:
    """Print the scoring taxonomy."""
    taxonomy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SCORING_TAXONOMY.md")
    if os.path.exists(taxonomy_path):
        with open(taxonomy_path) as f:
            console.print(f.read())
    else:
        console.print("[yellow]SCORING_TAXONOMY.md not found. Run 'score matrix' first to generate data.[/yellow]")


@cli.command()
@click.option("--task", "task_name", type=click.Choice(TASK_NAMES), default=None, help="Filter by task")
@click.option("--limit", default=20, help="Number of results to show")
def history(task_name: str | None, limit: int) -> None:
    """View experiment history (placeholder — requires store.py)."""
    console.print("[yellow]History requires the SQLite store (Phase 9). Not yet implemented.[/yellow]")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
