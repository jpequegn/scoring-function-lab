"""Tests for the CLI."""

from click.testing import CliRunner

from scorelab.cli import cli


runner = CliRunner()


class TestCLIRun:
    def test_run_extraction_rule_based(self):
        result = runner.invoke(cli, ["run", "--task", "extraction", "--scorer", "rule_based", "--iterations", "3"])
        assert result.exit_code == 0
        assert "extract-startups" in result.output

    def test_run_invalid_task(self):
        result = runner.invoke(cli, ["run", "--task", "nonexistent", "--scorer", "rule_based"])
        assert result.exit_code != 0

    def test_run_invalid_scorer(self):
        result = runner.invoke(cli, ["run", "--task", "extraction", "--scorer", "nonexistent"])
        assert result.exit_code != 0


class TestCLICompare:
    def test_compare_extraction(self):
        result = runner.invoke(cli, ["compare", "--task", "extraction", "--runs", "1"])
        assert result.exit_code == 0
        assert "Winner" in result.output

    def test_compare_query_answer(self):
        result = runner.invoke(cli, ["compare", "--task", "query_answer", "--runs", "1"])
        assert result.exit_code == 0


class TestCLIMatrix:
    def test_matrix_all(self):
        result = runner.invoke(cli, ["matrix", "--tasks", "all", "--scorers", "all", "--runs", "1"])
        assert result.exit_code == 0
        assert "Winner" in result.output

    def test_matrix_specific_tasks(self):
        result = runner.invoke(cli, ["matrix", "--tasks", "extraction,summarization", "--runs", "1"])
        assert result.exit_code == 0


class TestCLITaxonomy:
    def test_taxonomy_prints_content(self):
        result = runner.invoke(cli, ["taxonomy"])
        assert result.exit_code == 0
        assert "Scoring Function Taxonomy" in result.output or "not found" in result.output


class TestCLIHistory:
    def test_history_placeholder(self):
        result = runner.invoke(cli, ["history"])
        assert result.exit_code == 0
        assert "store" in result.output.lower() or "not yet" in result.output.lower()
