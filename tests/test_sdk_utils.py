"""Tests for SDK utility functions (validation, collect_text, safe_query)."""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from auto_scientist.schemas import AnalystOutput, ScientistPlanOutput
from auto_scientist.sdk_backend import SDKMessage, SDKOptions, _tolerant_parse_message
from auto_scientist.sdk_utils import (
    OutputValidationError,
    collect_text_from_query,
    safe_query,
    validate_json_output,
    validate_report_structure,
)


class TestTolerantParseMessage:
    def test_known_type_passes_through(self):
        msg = MagicMock()
        with patch(
            "auto_scientist.sdk_backend._original_parse_message",
            return_value=msg,
        ):
            result = _tolerant_parse_message({"type": "assistant"})
        assert result is msg

    def test_unknown_type_returns_none(self):
        from claude_code_sdk._errors import MessageParseError

        with patch(
            "auto_scientist.sdk_backend._original_parse_message",
            side_effect=MessageParseError("Unknown message type: rate_limit_event"),
        ):
            result = _tolerant_parse_message({"type": "rate_limit_event"})
        assert result is None

    def test_non_unknown_parse_error_reraises(self):
        from claude_code_sdk._errors import MessageParseError

        with (
            patch(
                "auto_scientist.sdk_backend._original_parse_message",
                side_effect=MessageParseError("Malformed JSON payload"),
            ),
            pytest.raises(MessageParseError, match="Malformed JSON payload"),
        ):
            _tolerant_parse_message({"type": "bad"})

    def test_logs_skipped_type(self, caplog):
        from claude_code_sdk._errors import MessageParseError

        with (
            patch(
                "auto_scientist.sdk_backend._original_parse_message",
                side_effect=MessageParseError("Unknown message type: foo_event"),
            ),
            caplog.at_level(logging.DEBUG, logger="auto_scientist.sdk_backend"),
        ):
            _tolerant_parse_message({"type": "foo_event"})

        assert "foo_event" in caplog.text


class TestSafeQuery:
    @pytest.mark.asyncio
    async def test_yields_non_none_messages(self):
        msg1 = SDKMessage(type="assistant", text="hello")
        msg2 = SDKMessage(type="result", result="done")

        class FakeBackend:
            async def query(self, prompt, options):
                yield msg1
                yield msg2

        opts = SDKOptions(system_prompt="test", allowed_tools=[], max_turns=5)
        results = [m async for m in safe_query("hi", opts, FakeBackend())]
        assert results == [msg1, msg2]

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self):
        class FakeBackend:
            async def query(self, prompt, options):
                return
                yield

        opts = SDKOptions(system_prompt="test", allowed_tools=[], max_turns=5)
        results = [m async for m in safe_query("hi", opts, FakeBackend())]
        assert results == []


# ---------------------------------------------------------------------------
# OutputValidationError
# ---------------------------------------------------------------------------


class TestOutputValidationError:
    def test_stores_attributes(self):
        inner = ValueError("bad field")
        e = OutputValidationError(
            raw_output='{"bad": true}',
            validation_error=inner,
            agent_name="Analyst",
        )
        assert e.raw_output == '{"bad": true}'
        assert e.validation_error is inner
        assert e.agent_name == "Analyst"

    def test_correction_prompt_contains_error(self):
        inner = ValueError("missing field 'observations'")
        e = OutputValidationError(
            raw_output='{"bad": true}',
            validation_error=inner,
            agent_name="Analyst",
        )
        prompt = e.correction_prompt()
        assert "<validation_error>" in prompt
        assert "missing field" in prompt
        assert '{"bad": true}' in prompt

    def test_correction_prompt_truncates_long_output(self):
        long_output = "x" * 1000
        e = OutputValidationError(
            raw_output=long_output,
            validation_error=ValueError("err"),
            agent_name="Test",
        )
        prompt = e.correction_prompt()
        assert len(prompt) < len(long_output) + 500  # correction text + truncated output


# ---------------------------------------------------------------------------
# validate_json_output
# ---------------------------------------------------------------------------


class TestValidateJsonOutput:
    def test_valid_json_valid_schema(self):
        data = {
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": ["ok"],
        }
        raw = json.dumps(data)
        result = validate_json_output(raw, AnalystOutput, "Analyst")
        assert isinstance(result, dict)
        assert result["observations"] == ["ok"]

    def test_strips_markdown_fencing(self):
        data = {
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": [],
        }
        raw = f"```json\n{json.dumps(data)}\n```"
        result = validate_json_output(raw, AnalystOutput, "Analyst")
        assert isinstance(result, dict)

    def test_invalid_json_raises(self):
        with pytest.raises(OutputValidationError) as exc_info:
            validate_json_output("not json at all", AnalystOutput, "Analyst")
        assert exc_info.value.agent_name == "Analyst"

    def test_valid_json_invalid_schema_raises(self):
        raw = json.dumps({"hypothesis": "test"})  # missing required fields
        with pytest.raises(OutputValidationError) as exc_info:
            validate_json_output(raw, ScientistPlanOutput, "Scientist")
        assert exc_info.value.agent_name == "Scientist"

    def test_extra_fields_tolerated(self):
        data = {
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": [],
            "llm_reasoning": "should be ignored",
        }
        result = validate_json_output(json.dumps(data), AnalystOutput, "Analyst")
        assert "llm_reasoning" not in result

    def test_trailing_text_after_json(self):
        """raw_decode() handles JSON followed by model commentary."""
        data = {
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": ["ok"],
        }
        raw = json.dumps(data) + "\n\nHere is my analysis of the above..."
        result = validate_json_output(raw, AnalystOutput, "Analyst")
        assert result["observations"] == ["ok"]

    def test_leading_prose_before_json(self):
        """Leading prose before JSON object is stripped."""
        data = {
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": ["ok"],
        }
        raw = "Here is my structured output:\n\n" + json.dumps(data)
        result = validate_json_output(raw, AnalystOutput, "Analyst")
        assert result["observations"] == ["ok"]

    def test_leading_prose_and_trailing_text(self):
        """Both leading prose and trailing commentary handled."""
        data = {
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": ["ok"],
        }
        raw = "My analysis:\n" + json.dumps(data) + "\n\nLet me know if you need more."
        result = validate_json_output(raw, AnalystOutput, "Analyst")
        assert result["observations"] == ["ok"]


# ---------------------------------------------------------------------------
# collect_text_from_query
# ---------------------------------------------------------------------------


class TestCollectTextFromQuery:
    @pytest.mark.asyncio
    async def test_collects_result_text(self):
        """Prefers result text over assistant text."""

        class FakeBackend:
            async def query(self, prompt, options):
                # TextBlock-like object
                text_block = MagicMock()
                text_block.text = "intermediate text"
                del text_block.name  # ensure it's treated as TextBlock, not ToolUseBlock

                yield SDKMessage(type="assistant", content_blocks=[text_block])
                yield SDKMessage(
                    type="result",
                    result='{"answer": 42}',
                    usage={"input_tokens": 10},
                )

        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)
        raw, usage = await collect_text_from_query("prompt", opts, FakeBackend())
        assert raw == '{"answer": 42}'

    @pytest.mark.asyncio
    async def test_falls_back_to_assistant_text(self):
        """When no result text, falls back to concatenated assistant text."""

        class FakeBackend:
            async def query(self, prompt, options):
                text_block = MagicMock()
                text_block.text = '{"answer": 42}'
                del text_block.name

                yield SDKMessage(type="assistant", content_blocks=[text_block])
                yield SDKMessage(type="result", result="", usage={})

        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)
        raw, usage = await collect_text_from_query("prompt", opts, FakeBackend())
        assert raw == '{"answer": 42}'

    @pytest.mark.asyncio
    async def test_empty_output_raises(self):
        class FakeBackend:
            async def query(self, prompt, options):
                yield SDKMessage(type="result", result="", usage={})

        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)
        with pytest.raises(RuntimeError, match="returned no output"):
            await collect_text_from_query("prompt", opts, FakeBackend(), agent_name="Analyst")

    @pytest.mark.asyncio
    async def test_populates_message_buffer(self):
        text_block = MagicMock()
        text_block.text = "block text"
        del text_block.name

        class FakeBackend:
            async def query(self, prompt, options):
                yield SDKMessage(type="assistant", content_blocks=[text_block])
                yield SDKMessage(type="result", result="result", usage={})

        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)
        buffer: list[str] = []
        with patch("auto_scientist.sdk_utils.append_block_to_buffer") as mock_append:
            await collect_text_from_query("prompt", opts, FakeBackend(), message_buffer=buffer)
            mock_append.assert_called_once_with(text_block, buffer)

    @pytest.mark.asyncio
    async def test_returns_usage_dict(self):
        """Usage dict from result message is returned as second element."""

        class FakeBackend:
            async def query(self, prompt, options):
                yield SDKMessage(
                    type="result",
                    result="output",
                    usage={"input_tokens": 100, "output_tokens": 50, "num_turns": 3},
                )

        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)
        _text, usage = await collect_text_from_query("prompt", opts, FakeBackend())
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["num_turns"] == 3


# Valid report with all 10 sections
_VALID_REPORT = """\
## Executive Summary
This report summarizes our investigation.

## Problem Statement and Data
We studied the dataset to discover patterns.

## Methodology
We used an iterative autonomous loop.

## Journey
v01 explored linear models. v02 switched to polynomial.

## Best Approach
Degree-4 polynomial with regularization.

## Results
Test R² = 0.964 with RMSE = 1.23.

## Key Scientific Insights
The relationship is nonlinear with a cubic component.

## Limitations
Fails for x > 100 due to polynomial divergence.

## Recommended Future Work
Try Gaussian process regression.

## Version Comparison Table
| Version | Status | Key Change | Key Metric | Prediction Outcome |
|---------|--------|------------|------------|-------------------|
| v01 | baseline | Linear model | R²=0.72 | N/A |
| v02 | best | Polynomial | R²=0.964 | CONFIRMED |
"""


class TestValidateReportStructure:
    def test_valid_report_returns_empty(self):
        issues = validate_report_structure(_VALID_REPORT)
        assert issues == []

    def test_missing_heading_reported(self):
        # Remove the "Journey" section
        journey_section = "## Journey\nv01 explored linear models. v02 switched to polynomial.\n\n"
        report = _VALID_REPORT.replace(journey_section, "")
        issues = validate_report_structure(report)
        assert any("journey" in issue.lower() for issue in issues)

    def test_empty_section_reported(self):
        # Make "Limitations" section empty
        report = _VALID_REPORT.replace(
            "## Limitations\nFails for x > 100 due to polynomial divergence.",
            "## Limitations\n",
        )
        issues = validate_report_structure(report)
        assert any("limitations" in issue.lower() for issue in issues)

    def test_missing_version_table_reported(self):
        # Remove the table from Version Comparison Table
        report = _VALID_REPORT.replace(
            "| Version | Status | Key Change | Key Metric | Prediction Outcome |\n"
            "|---------|--------|------------|------------|-------------------|\n"
            "| v01 | baseline | Linear model | R²=0.72 | N/A |\n"
            "| v02 | best | Polynomial | R²=0.964 | CONFIRMED |\n",
            "No table here.\n",
        )
        issues = validate_report_structure(report)
        assert any("table" in issue.lower() for issue in issues)

    def test_fuzzy_heading_match(self):
        # "Problem Statement and Data" should match the expected "Problem Statement"
        report = _VALID_REPORT  # Already uses "Problem Statement and Data"
        issues = validate_report_structure(report)
        assert issues == []

    def test_alternative_heading_names(self):
        # "Insights" should match "Key Scientific Insights"
        report = _VALID_REPORT.replace("## Key Scientific Insights", "## Insights")
        issues = validate_report_structure(report)
        assert issues == []
