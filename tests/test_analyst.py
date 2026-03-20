"""Tests for the Analyst agent."""

import json
from unittest.mock import MagicMock, patch

import pytest

from auto_scientist.agents.analyst import _format_success_criteria, run_analyst
from auto_scientist.config import SuccessCriterion


class TestFormatSuccessCriteria:
    """Tests for the pure helper function."""

    def test_empty_list(self):
        assert _format_success_criteria([]) == "(no success criteria defined)"

    def test_target_min_only(self):
        sc = SuccessCriterion(name="acc", description="accuracy", metric_key="acc", target_min=0.9)
        result = _format_success_criteria([sc])
        assert ">= 0.9" in result
        assert "REQUIRED" in result

    def test_target_max_only(self):
        sc = SuccessCriterion(name="loss", description="loss", metric_key="loss", target_max=0.1)
        result = _format_success_criteria([sc])
        assert "<= 0.1" in result

    def test_both_bounds(self):
        sc = SuccessCriterion(
            name="f1", description="f1 score", metric_key="f1",
            target_min=0.8, target_max=1.0,
        )
        result = _format_success_criteria([sc])
        assert "[0.8, 1.0]" in result

    def test_optional_label(self):
        sc = SuccessCriterion(
            name="extra", description="extra metric", metric_key="extra",
            required=False,
        )
        result = _format_success_criteria([sc])
        assert "optional" in result

    def test_multiple_criteria_numbered(self):
        criteria = [
            SuccessCriterion(name="a", description="da", metric_key="a"),
            SuccessCriterion(name="b", description="db", metric_key="b"),
        ]
        result = _format_success_criteria(criteria)
        assert result.startswith("1.")
        assert "2." in result


class TestRunAnalyst:
    """Tests for the agent runner, mocking the claude_code_sdk query."""

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_returns_parsed_json(self, mock_query, tmp_path):
        analysis = {
            "success_score": 75,
            "criteria_results": [],
            "key_metrics": {"rmse": 0.5},
            "improvements": ["better"],
            "regressions": [],
            "observations": ["noted"],
            "iteration_criteria_results": [],
        }

        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("rmse: 0.5")
        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        result = await run_analyst(
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
        )

        assert result["success_score"] == 75
        assert result["key_metrics"]["rmse"] == 0.5

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_handles_markdown_fenced_json(self, mock_query, tmp_path):
        analysis = {
            "success_score": 50, "criteria_results": [], "key_metrics": {},
            "improvements": [], "regressions": [], "observations": [],
            "iteration_criteria_results": [],
        }
        fenced = f"```json\n{json.dumps(analysis)}\n```"

        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = fenced

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        result = await run_analyst(
            results_path=results_path, plot_paths=[], notebook_path=notebook_path,
        )
        assert result["success_score"] == 50

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_raises_on_empty_output(self, mock_query, tmp_path):
        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        with pytest.raises(RuntimeError, match="returned no output"):
            await run_analyst(
                results_path=results_path, plot_paths=[], notebook_path=notebook_path,
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_accepts_data_dir_param(self, mock_query, tmp_path):
        """run_analyst should accept a data_dir parameter."""
        analysis = {
            "success_score": None,
            "criteria_results": [],
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": ["2 CSV files found"],
            "iteration_criteria_results": [],
            "domain_knowledge": "Environmental sensor data",
            "data_summary": {"files": [], "total_rows": 0},
        }

        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "sample.csv").write_text("x,y\n1,2\n")
        notebook_path = tmp_path / "notebook.md"

        result = await run_analyst(
            results_path=None,
            plot_paths=[],
            notebook_path=notebook_path,
            data_dir=data_dir,
        )
        assert result["success_score"] is None
        assert result["domain_knowledge"] == "Environmental sensor data"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_data_dir_in_prompt(self, mock_query, tmp_path):
        """When data_dir is provided, it should appear in the prompt."""
        analysis = {
            "success_score": None, "criteria_results": [], "key_metrics": {},
            "improvements": [], "regressions": [], "observations": [],
            "iteration_criteria_results": [],
        }

        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            yield result_msg

        mock_query.side_effect = fake_query

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "sample.csv").write_text("x,y\n1,2\n")
        notebook_path = tmp_path / "notebook.md"

        await run_analyst(
            results_path=None,
            plot_paths=[],
            notebook_path=notebook_path,
            data_dir=data_dir,
        )
        assert str(data_dir) in captured_prompt["prompt"]

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_iteration0_output_shape(self, mock_query, tmp_path):
        """Iteration 0 output: success_score null, optional domain_knowledge/data_summary."""
        analysis = {
            "success_score": None,
            "criteria_results": [],
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": ["200 rows, 2 float columns"],
            "iteration_criteria_results": [],
            "domain_knowledge": "This dataset contains sensor readings",
            "data_summary": {
                "files": [{"name": "data.csv", "rows": 200, "columns": ["x", "y"]}],
                "total_rows": 200,
            },
        }

        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "data.csv").write_text("x,y\n" + "\n".join(f"{i},{i*2}" for i in range(200)))
        notebook_path = tmp_path / "notebook.md"

        result = await run_analyst(
            results_path=None,
            plot_paths=[],
            notebook_path=notebook_path,
            data_dir=data_dir,
        )
        assert result["success_score"] is None
        assert result["domain_knowledge"] == "This dataset contains sensor readings"
        assert result["data_summary"]["total_rows"] == 200

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_missing_results_file_uses_fallback(self, mock_query, tmp_path):
        analysis = {
            "success_score": 0, "criteria_results": [], "key_metrics": {},
            "improvements": [], "regressions": [], "observations": [],
            "iteration_criteria_results": [],
        }

        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "nonexistent.txt"
        notebook_path = tmp_path / "notebook.md"

        result = await run_analyst(
            results_path=results_path, plot_paths=[], notebook_path=notebook_path,
        )
        assert result["success_score"] == 0

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_populates_message_buffer(self, mock_query, tmp_path):
        analysis = {
            "success_score": 50, "criteria_results": [], "key_metrics": {},
            "improvements": [], "regressions": [], "observations": [],
            "iteration_criteria_results": [],
        }

        from auto_scientist.agents.analyst import AssistantMessage, ResultMessage, TextBlock

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Analyzing the data..."
        assistant_msg.content = [text_block]

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        buf: list[str] = []
        await run_analyst(
            results_path=results_path, plot_paths=[], notebook_path=notebook_path,
            message_buffer=buf,
        )
        assert len(buf) == 1
        assert "Analyzing the data..." in buf[0]

    def test_no_targets_no_target_text(self):
        sc = SuccessCriterion(name="x", description="d", metric_key="x")
        result = _format_success_criteria([sc])
        assert ">=" not in result
        assert "<=" not in result

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_fallback_to_assistant_text(self, mock_query, tmp_path):
        analysis = {
            "success_score": 42, "criteria_results": [], "key_metrics": {},
            "improvements": [], "regressions": [], "observations": [],
            "iteration_criteria_results": [],
        }

        from auto_scientist.agents.analyst import AssistantMessage, ResultMessage, TextBlock

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = json.dumps(analysis)
        assistant_msg.content = [text_block]

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        result = await run_analyst(
            results_path=results_path, plot_paths=[], notebook_path=notebook_path,
        )
        assert result["success_score"] == 42

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_iteration0_uses_data_dir(self, mock_query, tmp_path):
        analysis = {
            "success_score": None, "criteria_results": [], "key_metrics": {},
            "improvements": [], "regressions": [], "observations": [],
            "iteration_criteria_results": [],
        }

        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            yield result_msg

        mock_query.side_effect = fake_query

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "sample.csv").write_text("x,y\n1,2\n")
        notebook_path = tmp_path / "notebook.md"

        await run_analyst(
            results_path=None, plot_paths=[], notebook_path=notebook_path,
            data_dir=data_dir,
        )
        assert "<data_directory>" in captured_prompt["prompt"]

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_options_configuration(self, mock_query, tmp_path):
        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps({
            "success_score": 50, "criteria_results": [], "key_metrics": {},
            "improvements": [], "regressions": [], "observations": [],
            "iteration_criteria_results": [],
        })

        captured_opts = {}

        async def fake_query(**kwargs):
            captured_opts.update(kwargs)
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        await run_analyst(
            results_path=results_path, plot_paths=[], notebook_path=notebook_path,
        )
        opts = captured_opts["options"]
        assert opts.allowed_tools == ["Read", "Glob"]
        assert opts.max_turns == 30
