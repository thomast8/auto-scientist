"""Tests for the auto-reviewer Findings agent."""

from unittest.mock import MagicMock, patch

import pytest
from auto_core.sdk_backend import SDKMessage
from auto_core.state import ExperimentState
from auto_reviewer.agents.findings import run_findings


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    del block.name
    return block


def _assistant_msg(text: str) -> SDKMessage:
    return SDKMessage(type="assistant", content_blocks=[_text_block(text)])


def _result_msg(session_id: str | None = None) -> SDKMessage:
    return SDKMessage(type="result", result=None, usage={}, session_id=session_id)


def _polluted_findings_report(body: str = "No open questions remain.") -> str:
    return (
        "[userMessage]\n"
        "Inspecting probe artifacts before writing findings.\n"
        "[tool]\n"
        "```markdown\n"
        "## Summary\n\n"
        "This README sample is not the findings report.\n"
        "```\n\n"
        "# README\n\n"
        "This is command output from a tool call, not the report.\n\n"
        "    # code comment from a fixture\n\n"
        "## Open Questions / Assumptions\n\n"
        f"{body} This text is long enough to pass the minimum report length "
        "check while starting at the first real report markdown heading."
    )


@pytest.mark.asyncio
@patch("auto_reviewer.agents.findings.validate_report_structure", return_value=[])
@patch("auto_reviewer.agents.findings.safe_query")
async def test_findings_strips_tool_transcript_before_section_heading(
    mock_query, _mock_validate, tmp_path
):
    """Text-channel fallback should not preserve tool transcript preamble."""

    async def fake_query(**kwargs):
        yield _assistant_msg(_polluted_findings_report())
        yield _result_msg()

    mock_query.side_effect = fake_query

    state = ExperimentState(domain="review", goal="review PR")
    notebook_path = tmp_path / "lab_notebook.xml"
    notebook_path.write_text("# Notebook")

    result = await run_findings(
        state=state,
        notebook_path=notebook_path,
        output_dir=tmp_path,
    )

    assert result.startswith("## Open Questions / Assumptions")
    assert "[userMessage]" not in result
    assert "README sample" not in result
    assert "# README" not in result


@pytest.mark.asyncio
@patch("auto_reviewer.agents.findings.validate_report_structure", return_value=[])
@patch("auto_reviewer.agents.findings.safe_query")
async def test_findings_prefers_report_file_written_during_attempt(
    mock_query, _mock_validate, tmp_path
):
    """A freshly written report.md is stripped and used as the artifact."""
    report_path = tmp_path / "report.md"

    async def fake_query(**kwargs):
        report_path.write_text(_polluted_findings_report("Disk report wins."))
        yield _assistant_msg("I wrote report.md")
        yield _result_msg()

    mock_query.side_effect = fake_query

    state = ExperimentState(domain="review", goal="review PR")
    notebook_path = tmp_path / "lab_notebook.xml"
    notebook_path.write_text("# Notebook")

    result = await run_findings(
        state=state,
        notebook_path=notebook_path,
        output_dir=tmp_path,
    )

    assert result.startswith("## Open Questions / Assumptions")
    assert "Disk report wins." in result
    assert "[userMessage]" not in result
    assert "# README" not in result


@pytest.mark.asyncio
@patch("auto_reviewer.agents.findings.validate_report_structure", return_value=[])
@patch("auto_reviewer.agents.findings.safe_query")
async def test_findings_ignores_stale_report_file(mock_query, _mock_validate, tmp_path):
    """An unchanged preexisting report.md cannot override text-channel fallback."""
    (tmp_path / "report.md").write_text(
        "## Open Questions / Assumptions\n\n"
        "Stale disk content that should not be returned by this attempt."
    )

    async def fake_query(**kwargs):
        yield _assistant_msg(_polluted_findings_report("Fresh text fallback wins."))
        yield _result_msg()

    mock_query.side_effect = fake_query

    state = ExperimentState(domain="review", goal="review PR")
    notebook_path = tmp_path / "lab_notebook.xml"
    notebook_path.write_text("# Notebook")

    result = await run_findings(
        state=state,
        notebook_path=notebook_path,
        output_dir=tmp_path,
    )

    assert "Fresh text fallback wins." in result
    assert "Stale disk content" not in result
    assert "[userMessage]" not in result
    assert "# README" not in result


@pytest.mark.asyncio
@patch("auto_reviewer.agents.findings.validate_report_structure", return_value=[])
@patch("auto_reviewer.agents.findings.safe_query")
async def test_findings_empty_report_file_uses_stripped_text_fallback(
    mock_query, _mock_validate, tmp_path
):
    """A blank report.md still leaves the text-channel fallback sanitized."""
    (tmp_path / "report.md").write_text(" \n")

    async def fake_query(**kwargs):
        yield _assistant_msg(_polluted_findings_report("Text fallback survives."))
        yield _result_msg()

    mock_query.side_effect = fake_query

    state = ExperimentState(domain="review", goal="review PR")
    notebook_path = tmp_path / "lab_notebook.xml"
    notebook_path.write_text("# Notebook")

    result = await run_findings(
        state=state,
        notebook_path=notebook_path,
        output_dir=tmp_path,
    )

    assert result.startswith("## Open Questions / Assumptions")
    assert "Text fallback survives." in result
    assert "[userMessage]" not in result
    assert "# README" not in result
