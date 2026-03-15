"""Rich terminal renderer for live loop view and comparison tables."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.table import Table
from rich.text import Text

from scorelab.loop import Iteration, LoopResult
from scorelab.metrics import ConvergenceMetrics, compute_metrics
from scorelab.runner import ComparisonReport, MatrixReport


def render_iteration(iteration: Iteration, task_name: str, scorer_name: str, max_iterations: int, console: Console | None = None) -> str:
    """Render a single iteration's live view. Returns the rendered string."""
    buf = StringIO()
    con = console or Console(file=buf, force_terminal=True, width=72)

    score_bar_filled = int(iteration.score * 10)
    score_bar = "\u2588" * score_bar_filled + "\u2591" * (10 - score_bar_filled)

    delta_sign = "+" if iteration.score_delta >= 0 else ""
    status = "converged!" if iteration.converged else "converging..."

    con.print(f"Task: {task_name}  |  Scorer: {scorer_name}  |  Iteration {iteration.number}/{max_iterations}")
    con.print("\u2500" * 72)
    hypothesis_preview = iteration.hypothesis[:60] + "..." if len(iteration.hypothesis) > 60 else iteration.hypothesis
    con.print(f"Hypothesis: {hypothesis_preview}")
    con.print(f"Score: {iteration.score:.2f} ({delta_sign}{iteration.score_delta:.2f}) {score_bar}  {status}")
    con.print(f"Tokens this iter: {iteration.tokens_used}  |  Cost: ${iteration.cost_usd:.4f}")

    if console is None:
        return buf.getvalue()
    return ""


def render_loop_result(result: LoopResult, console: Console | None = None) -> str:
    """Render final loop result summary. Returns the rendered string."""
    buf = StringIO()
    con = console or Console(file=buf, force_terminal=True, width=72)

    metrics = compute_metrics(result)
    trajectory_str = " \u2192 ".join(f"{s:.2f}" for s in metrics.score_trajectory)

    con.print()
    con.print(f"[bold]Result:[/bold] {result.termination_reason} after {result.total_iterations} iterations")
    con.print(f"Score Trajectory: {trajectory_str}")
    con.print(f"Final Score: {metrics.final_score:.2f}  |  Total Tokens: {metrics.total_tokens}  |  Cost: ${metrics.total_cost_usd:.4f}")
    con.print(f"CPS: ${metrics.cost_per_score_point:.4f}  |  Churn Index: {metrics.churn_index:.1f}  |  Efficiency: {metrics.efficiency_score:.3f}")

    if console is None:
        return buf.getvalue()
    return ""


def render_comparison(report: ComparisonReport, console: Console | None = None) -> str:
    """Render a comparison table of scorers. Returns the rendered string."""
    buf = StringIO()
    con = console or Console(file=buf, force_terminal=True, width=100)

    table = Table(title=f"Comparison: {report.task.name}")
    table.add_column("Scorer", style="cyan")
    table.add_column("Iters", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Final", justify="right")
    table.add_column("CPS", justify="right")
    table.add_column("Conv%", justify="right")
    table.add_column("", justify="center")

    for sr in report.scorer_reports:
        is_winner = sr.scorer_name == report.winner
        style = "bold green" if is_winner else ""
        marker = "\u2605" if is_winner else ""

        cps_str = f"${sr.avg_cost_per_score_point:.4f}" if sr.avg_cost_per_score_point != float("inf") else "inf"

        table.add_row(
            sr.scorer_name,
            f"{sr.avg_iterations:.1f}",
            f"{int(sr.avg_iterations * 100)}",  # approx tokens from avg iters
            f"${sr.avg_cost_usd:.4f}",
            f"{sr.avg_final_score:.2f}",
            cps_str,
            f"{sr.convergence_rate * 100:.0f}%",
            marker,
            style=style,
        )

    con.print(table)
    con.print(f"\n[bold]Winner (cost-per-score-point):[/bold] {report.winner}")

    if console is None:
        return buf.getvalue()
    return ""


def render_matrix(matrix: MatrixReport, console: Console | None = None) -> str:
    """Render a matrix report showing winners per task. Returns the rendered string."""
    buf = StringIO()
    con = console or Console(file=buf, force_terminal=True, width=100)

    table = Table(title="Scoring Matrix: Winners by Task")
    table.add_column("Task", style="cyan")
    table.add_column("Winner", style="bold green")

    for task_name, winner in sorted(matrix.task_winners.items()):
        table.add_row(task_name, winner)

    con.print(table)

    # Also render each comparison
    for comparison in matrix.comparisons:
        con.print()
        render_comparison(comparison, console=con)

    if console is None:
        return buf.getvalue()
    return ""
