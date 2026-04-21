"""Tests for the Analyst agent."""

import json
from unittest.mock import MagicMock, patch

import pytest
from auto_core.sdk_utils import OutputValidationError

from auto_scientist.agents.analyst import run_analyst


class TestRunAnalyst:
    """Tests for the agent runner, mocking the claude_code_sdk query."""

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_returns_parsed_json(self, mock_query, tmp_path):
        analysis = {
            "key_metrics": [{"name": "rmse", "value": 0.5}],
            "improvements": ["better"],
            "regressions": [],
            "observations": ["noted"],
        }

        from claude_code_sdk import ResultMessage

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

        assert result["key_metrics"] == [{"name": "rmse", "value": 0.5}]

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_handles_markdown_fenced_json(self, mock_query, tmp_path):
        analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": [],
        }
        fenced = f"```json\n{json.dumps(analysis)}\n```"

        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = fenced

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        result = await run_analyst(
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_raises_on_empty_output(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

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
                results_path=results_path,
                plot_paths=[],
                notebook_path=notebook_path,
            )

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_accepts_data_dir_param(self, mock_query, tmp_path):
        """run_analyst should accept a data_dir parameter."""
        analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": ["2 CSV files found"],
            "domain_knowledge": "Environmental sensor data",
            "data_summary": "Files: sample.csv (2 rows, 2 columns: x, y)",
        }

        from claude_code_sdk import ResultMessage

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
        assert result["domain_knowledge"] == "Environmental sensor data"

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_data_dir_in_prompt(self, mock_query, tmp_path):
        """When data_dir is provided, it should appear in the prompt."""
        analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": [],
        }

        from claude_code_sdk import ResultMessage

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
    @patch("auto_core.sdk_backend.claude_query")
    async def test_iteration0_output_shape(self, mock_query, tmp_path):
        """Iteration 0 output: success_score null, optional domain_knowledge/data_summary."""
        analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": ["200 rows, 2 float columns"],
            "domain_knowledge": "This dataset contains sensor readings",
            "data_summary": "Files: data.csv (200 rows, columns: x, y). Total: 200 rows.",
        }

        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "data.csv").write_text("x,y\n" + "\n".join(f"{i},{i * 2}" for i in range(200)))
        notebook_path = tmp_path / "notebook.md"

        result = await run_analyst(
            results_path=None,
            plot_paths=[],
            notebook_path=notebook_path,
            data_dir=data_dir,
        )
        assert result["domain_knowledge"] == "This dataset contains sensor readings"
        assert "200 rows" in result["data_summary"]

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_missing_results_file_uses_fallback(self, mock_query, tmp_path):
        analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": [],
        }

        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "nonexistent.txt"
        notebook_path = tmp_path / "notebook.md"

        result = await run_analyst(
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
        )
        assert "success_score" not in result

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_populates_message_buffer(self, mock_query, tmp_path):
        analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": [],
        }

        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

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
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
            message_buffer=buf,
        )
        assert len(buf) == 1
        assert "Analyzing the data..." in buf[0]

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_fallback_to_assistant_text(self, mock_query, tmp_path):
        analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": [],
        }

        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

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
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
        )
        assert "success_score" not in result

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_iteration0_uses_data_dir(self, mock_query, tmp_path):
        analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": [],
        }

        from claude_code_sdk import ResultMessage

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
        assert "<data_directory>" in captured_prompt["prompt"]

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_options_configuration(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(
            {
                "key_metrics": [],
                "improvements": [],
                "regressions": [],
                "observations": [],
            }
        )

        captured_opts = {}

        async def fake_query(**kwargs):
            captured_opts.update(kwargs)
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        await run_analyst(
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
        )
        opts = captured_opts["options"]
        assert opts.allowed_tools == [
            "Read",
            "Glob",
            "mcp__notebook__read_notebook",
        ]
        assert opts.max_turns == 30
        assert "notebook" in opts.mcp_servers


class TestAnalystRetry:
    """Tests for the retry-on-validation-failure behavior."""

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_retry_on_invalid_json_then_succeed(self, mock_query, tmp_path):
        """First attempt returns invalid JSON, second returns valid."""
        valid_analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": ["ok"],
        }

        from claude_code_sdk import ResultMessage

        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            msg = MagicMock(spec=ResultMessage)
            if call_count == 1:
                msg.result = "not valid json at all"
            else:
                msg.result = json.dumps(valid_analysis)
            yield msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        result = await run_analyst(
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
        )
        assert result["observations"] == ["ok"]
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_retry_on_schema_violation_then_succeed(self, mock_query, tmp_path):
        """First attempt returns JSON missing required fields, second is valid."""
        valid_analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": [],
        }

        from claude_code_sdk import ResultMessage

        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            msg = MagicMock(spec=ResultMessage)
            if call_count == 1:
                msg.result = json.dumps({"only_hypothesis": "test"})
            else:
                msg.result = json.dumps(valid_analysis)
            yield msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        await run_analyst(
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
        )
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_exhausts_retries_raises(self, mock_query, tmp_path):
        """All attempts return invalid output; raises OutputValidationError."""
        from claude_code_sdk import ResultMessage

        async def fake_query(**kwargs):
            msg = MagicMock(spec=ResultMessage)
            msg.result = "always bad json"
            yield msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        with pytest.raises(OutputValidationError, match="Analyst"):
            await run_analyst(
                results_path=results_path,
                plot_paths=[],
                notebook_path=notebook_path,
            )

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_correction_hint_in_retry_prompt(self, mock_query, tmp_path):
        """On retry, the prompt should include a validation_error correction hint."""
        valid_analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": [],
        }

        from claude_code_sdk import ResultMessage

        captured_prompts = []

        async def fake_query(**kwargs):
            captured_prompts.append(kwargs.get("prompt", ""))
            msg = MagicMock(spec=ResultMessage)
            if len(captured_prompts) == 1:
                msg.result = "bad json"
            else:
                msg.result = json.dumps(valid_analysis)
            yield msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        await run_analyst(
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
        )
        assert len(captured_prompts) == 2
        assert "<validation_error>" in captured_prompts[1]


class TestTimeoutContext:
    """Tests for timeout_context parameter in run_analyst."""

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_timeout_context_prepends_to_prompt(self, mock_query, tmp_path):
        """When timeout_context is provided, the prompt should contain timeout info."""
        analysis = {
            "key_metrics": [{"name": "timeout_minutes", "value": 120}],
            "improvements": [],
            "regressions": [],
            "observations": ["script timed out after 120 minutes"],
        }

        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        await run_analyst(
            results_path=None,
            plot_paths=[],
            notebook_path=notebook_path,
            timeout_context={
                "timeout_minutes": 120,
                "hypothesis": "Test heavy computation",
            },
        )
        assert "<timeout_info>" in captured_prompt["prompt"]
        assert "TIMED OUT after 120 minutes" in captured_prompt["prompt"]
        assert "Test heavy computation" in captured_prompt["prompt"]

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_timeout_context_with_partial_results(self, mock_query, tmp_path):
        """When timeout_context is provided with existing results file, indicate partial."""
        analysis = {
            "key_metrics": [{"name": "timeout_minutes", "value": 60}],
            "improvements": [],
            "regressions": [],
            "observations": ["partial results"],
        }

        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("partial output here")
        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        await run_analyst(
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
            timeout_context={
                "timeout_minutes": 60,
                "hypothesis": "Test hypothesis",
            },
        )
        assert "Partial results available: yes" in captured_prompt["prompt"]

    @pytest.mark.asyncio
    @patch("auto_core.sdk_backend.claude_query")
    async def test_no_timeout_context_omits_block(self, mock_query, tmp_path):
        """Without timeout_context, prompt should not contain timeout info."""
        analysis = {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": [],
        }

        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        await run_analyst(
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
        )
        assert "<timeout_info>" not in captured_prompt["prompt"]


class TestAnalystPromptBuilder:
    def test_gpt_prompt_uses_exact_read_tool_name_and_tool_use_guidance(self):
        from auto_scientist.prompts.analyst import build_analyst_system

        system = build_analyst_system("gpt")

        assert "Tool calls are allowed before the final JSON response." in system
        assert "with the `Read` tool" in system
        assert "`Glob` only when you need to verify file presence" in system

    def test_gpt_prompt_keeps_double_recap_and_three_examples(self):
        from auto_scientist.prompts.analyst import build_analyst_system

        system = build_analyst_system("gpt")

        assert system.count("<recap>") == 2
        assert system.count("<example>") == 3
        assert "script crashed: ZeroDivisionError at line 142" not in system
