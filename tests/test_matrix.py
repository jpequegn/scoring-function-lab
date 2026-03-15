"""Tests for the matrix runner and taxonomy generation."""

import os

from run_matrix import TASKS, SCORERS, generate_taxonomy, _dummy_agent
from scorelab.runner import ExperimentRunner


class TestMatrixRun:
    def test_matrix_produces_results_for_all_tasks(self):
        runner = ExperimentRunner(agent_fn=_dummy_agent)
        matrix = runner.run_matrix(TASKS, SCORERS, runs=1)
        assert len(matrix.comparisons) == len(TASKS)
        assert len(matrix.task_winners) == len(TASKS)

    def test_each_comparison_has_all_scorers(self):
        runner = ExperimentRunner(agent_fn=_dummy_agent)
        matrix = runner.run_matrix(TASKS, SCORERS, runs=1)
        for comparison in matrix.comparisons:
            assert len(comparison.scorer_reports) == len(SCORERS)

    def test_winner_exists_for_each_task(self):
        runner = ExperimentRunner(agent_fn=_dummy_agent)
        matrix = runner.run_matrix(TASKS, SCORERS, runs=1)
        for task in TASKS:
            assert task.name in matrix.task_winners
            assert matrix.task_winners[task.name]  # non-empty


class TestTaxonomyGeneration:
    def test_taxonomy_contains_all_task_types(self):
        results = {
            "extraction": {"rule_based": {"avg_iterations": 3.0, "avg_final_score": 0.5, "avg_cost_usd": 0.0, "avg_cps": 0.0, "convergence_rate": 0.0}},
            "code_fix": {"rule_based": {"avg_iterations": 3.0, "avg_final_score": 0.5, "avg_cost_usd": 0.0, "avg_cps": 0.0, "convergence_rate": 0.0}},
        }
        winners = {"extract-startups": "rule_based", "fix-bug": "rule_based"}
        taxonomy = generate_taxonomy(results, winners)
        assert "extraction" in taxonomy
        assert "code_fix" in taxonomy

    def test_taxonomy_has_key_sections(self):
        results = {
            "extraction": {"rule_based": {"avg_iterations": 3.0, "avg_final_score": 0.5, "avg_cost_usd": 0.0, "avg_cps": 0.0, "convergence_rate": 0.0}},
        }
        winners = {"extract-startups": "rule_based"}
        taxonomy = generate_taxonomy(results, winners)
        assert "# Scoring Function Taxonomy" in taxonomy
        assert "## Key Patterns" in taxonomy
        assert "Cost Per Score Point" in taxonomy
        assert "## Methodology" in taxonomy

    def test_taxonomy_file_exists_after_generation(self):
        assert os.path.exists("SCORING_TAXONOMY.md")

    def test_taxonomy_file_not_empty(self):
        with open("SCORING_TAXONOMY.md") as f:
            content = f.read()
        assert len(content) > 500
        assert "Scoring Function Taxonomy" in content
