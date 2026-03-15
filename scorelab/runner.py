"""Experiment runner for multi-scorer comparison."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from scorelab.loop import AgentLoop, AgentFn, LoopResult
from scorelab.metrics import ConvergenceMetrics, compute_metrics
from scorelab.scorer import BaseScorer
from scorelab.task import Task


@dataclass
class ScorerReport:
    """Aggregated results for one scorer across multiple runs.

    Attributes:
        scorer_name: Name of the scorer.
        runs: Individual LoopResults from each run.
        metrics: ConvergenceMetrics for each run.
        avg_iterations: Average iterations across runs.
        avg_final_score: Average final score across runs.
        avg_cost_usd: Average total cost across runs.
        avg_cost_per_score_point: Average CPS across runs.
        convergence_rate: Fraction of runs that converged.
    """

    scorer_name: str
    runs: list[LoopResult]
    metrics: list[ConvergenceMetrics]
    avg_iterations: float
    avg_final_score: float
    avg_cost_usd: float
    avg_cost_per_score_point: float
    convergence_rate: float


@dataclass
class ComparisonReport:
    """Comparison of multiple scorers on the same task.

    Attributes:
        task: The task that was evaluated.
        scorer_reports: Results for each scorer.
        winner: Name of the scorer with lowest avg cost_per_score_point.
    """

    task: Task
    scorer_reports: list[ScorerReport]
    winner: str


@dataclass
class MatrixReport:
    """Full factorial: every task x every scorer.

    Attributes:
        comparisons: One ComparisonReport per task.
        task_winners: Mapping of task name to winning scorer name.
    """

    comparisons: list[ComparisonReport]
    task_winners: dict[str, str]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _avg_cps(metrics_list: list[ConvergenceMetrics]) -> float:
    """Average CPS, treating inf values as excluded."""
    finite = [m.cost_per_score_point for m in metrics_list if m.cost_per_score_point != float("inf")]
    return _avg(finite) if finite else float("inf")


class ExperimentRunner:
    """Runs experiments comparing multiple scorers on tasks."""

    def __init__(self, agent_fn: AgentFn, max_workers: int = 4) -> None:
        self.agent_fn = agent_fn
        self.max_workers = max_workers

    def _single_run(self, scorer: BaseScorer, task: Task) -> LoopResult:
        loop = AgentLoop(scorer=scorer, task=task, agent_fn=self.agent_fn)
        return loop.run()

    def _build_scorer_report(self, scorer: BaseScorer, task: Task, runs: int) -> ScorerReport:
        results: list[LoopResult] = []
        for _ in range(runs):
            results.append(self._single_run(scorer, task))

        metrics_list = [compute_metrics(r) for r in results]

        converged_count = sum(1 for r in results if r.converged)

        return ScorerReport(
            scorer_name=scorer.name,
            runs=results,
            metrics=metrics_list,
            avg_iterations=_avg([float(m.iterations_to_converge or r.total_iterations) for m, r in zip(metrics_list, results)]),
            avg_final_score=_avg([m.final_score for m in metrics_list]),
            avg_cost_usd=_avg([m.total_cost_usd for m in metrics_list]),
            avg_cost_per_score_point=_avg_cps(metrics_list),
            convergence_rate=converged_count / len(results) if results else 0.0,
        )

    def compare(self, task: Task, scorers: list[BaseScorer], runs: int = 3) -> ComparisonReport:
        """Run the same task N times against each scorer. Return comparison."""
        reports: list[ScorerReport] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._build_scorer_report, scorer, task, runs): scorer
                for scorer in scorers
            }
            for future in as_completed(futures):
                reports.append(future.result())

        # Sort by scorer name for deterministic output
        reports.sort(key=lambda r: r.scorer_name)

        # Winner = lowest avg CPS
        winner = min(reports, key=lambda r: r.avg_cost_per_score_point)

        return ComparisonReport(task=task, scorer_reports=reports, winner=winner.scorer_name)

    def run_matrix(self, tasks: list[Task], scorers: list[BaseScorer], runs: int = 3) -> MatrixReport:
        """Full factorial: every task x every scorer. Build taxonomy."""
        comparisons: list[ComparisonReport] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self.compare, task, scorers, runs): task
                for task in tasks
            }
            for future in as_completed(futures):
                comparisons.append(future.result())

        comparisons.sort(key=lambda c: c.task.name)
        task_winners = {c.task.name: c.winner for c in comparisons}

        return MatrixReport(comparisons=comparisons, task_winners=task_winners)
