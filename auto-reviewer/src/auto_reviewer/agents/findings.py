"""Findings agent: compiles the final PR review report.

Generates a prioritized markdown report: confirmed bugs with reproducers,
refuted suspicions with reasoning, ungrounded findings, open questions, and
known limitations.

Writes report.md via the Write tool; the orchestrator round-trips it for
validation.
"""

import logging
from pathlib import Path
from typing import Any, cast

from auto_core.agents.notebook_tool import (
    NOTEBOOK_SPEC,
    build_notebook_mcp_server,
    format_notebook_toc,
)
from auto_core.notebook import parse_notebook_entries
from auto_core.retry import QueryResult, agent_retry_loop
from auto_core.retry import ValidationError as RetryValidationError
from auto_core.sdk_backend import SDKOptions, get_backend
from auto_core.sdk_utils import (
    append_block_to_buffer,
    collect_text_from_query,
    prepare_turn_budget,
    resolve_prompt_provider,
    safe_query,
    strip_report_preamble,
    validate_report_structure,
)
from auto_core.state import ExperimentState

from auto_reviewer.agents.surveyor import _format_prediction_tree
from auto_reviewer.prompts.findings import FINDINGS_USER, build_findings_system

logger = logging.getLogger(__name__)

# Minimum report length to consider valid (characters)
MIN_REPORT_LENGTH = 100
FINDINGS_REPORT_HEADINGS = (
    "summary",
    "confirmed bugs",
    "refuted suspicions",
    "ungrounded findings",
    "open questions",
    "known limitations",
)


async def run_findings(
    state: ExperimentState,
    notebook_path: Path,
    output_dir: Path,
    model: str | None = None,
    message_buffer: list[str] | None = None,
    provider: str = "openai",
    pending_abductions: str = "",
) -> str:
    """Generate the final PR review report.

    Args:
        state: Final review state with all iteration history.
        notebook_path: Path to the investigation log.
        output_dir: Review workspace directory (for reading iteration artifacts).
        model: Optional model override.
        message_buffer: Optional buffer for streaming messages.

    Returns:
        Report content as a markdown string.
    """
    notebook_entries = parse_notebook_entries(notebook_path)

    user_prompt = FINDINGS_USER.format(
        state_json=state.model_dump_json(indent=2),
        notebook_toc=format_notebook_toc(notebook_entries),
        prediction_tree=_format_prediction_tree(state),
        workspace_path=str(output_dir),
    )

    report_path = output_dir / "report.md"

    max_turns = 10
    allowed_tools = ["Read", "Glob", "Write", NOTEBOOK_SPEC.mcp_tool_name]
    mcp_servers: dict[str, Any] = {
        "notebook": build_notebook_mcp_server(notebook_path, output_dir=output_dir),
    }
    prompt_provider = resolve_prompt_provider(provider)
    report_system = build_findings_system(prompt_provider)
    budget = prepare_turn_budget(report_system, max_turns, allowed_tools, provider=provider)
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
        extra_args={},
        mcp_servers=mcp_servers,
    )

    # Shared state between query and validate closures.
    last_full_text = [""]

    async def _query(prompt_text: str, resume_session_id: str | None) -> QueryResult:
        report_state_before = _report_file_state(report_path)
        opts = options
        if resume_session_id is not None:
            retry_budget = prepare_turn_budget(report_system, max_turns, allowed_tools)
            opts = SDKOptions(
                system_prompt=retry_budget.system_prompt,
                allowed_tools=retry_budget.allowed_tools,
                max_turns=retry_budget.max_turns,
                permission_mode="acceptEdits",
                cwd=output_dir,
                model=model,
                resume=resume_session_id,
                extra_args={"setting-sources": ""},
                mcp_servers=mcp_servers,
            )

        report_parts: list[str] = []
        sid: str | None = None
        async for message in safe_query(prompt=prompt_text, options=opts, backend=backend):
            if message.type == "assistant":
                for block in message.content_blocks:
                    if message_buffer is not None:
                        append_block_to_buffer(block, message_buffer)
                    if hasattr(block, "text") and not hasattr(block, "name"):
                        report_parts.append(block.text)
            elif message.type == "result":
                sid = message.session_id
                usage = message.usage
                collect_text_from_query.last_usage = usage  # type: ignore[attr-defined]

        raw = strip_report_preamble("\n".join(report_parts), FINDINGS_REPORT_HEADINGS)
        # The agent's contract is to write report.md via the Write tool. When
        # that file was created or changed by this attempt, it (not the text
        # channel) is the artifact we validate against. An unchanged preexisting
        # file can be stale from a prior failed attempt, so keep the text
        # fallback in that case.
        report_state_after = _report_file_state(report_path)
        if report_state_after is not None and report_state_after != report_state_before:
            disk_text = report_path.read_text().strip()
            if disk_text:
                raw = strip_report_preamble(disk_text, FINDINGS_REPORT_HEADINGS)
        last_full_text[0] = raw
        return QueryResult(raw_output=raw, session_id=sid, usage={})

    def _validate(result: QueryResult) -> str:
        text = result.raw_output
        if len(text) < MIN_REPORT_LENGTH:
            raise RetryValidationError(
                "<validation_error>\n"
                "Your previous output was too short or empty. "
                "Please generate a comprehensive markdown report with headings, "
                "covering the review journey, key findings, and recommendations.\n"
                "</validation_error>"
            )
        structure_issues = validate_report_structure(text)
        if structure_issues:
            issues_list = "\n".join(f"- {issue}" for issue in structure_issues)
            raise RetryValidationError(
                f"<validation_error>\n"
                f"Your report is missing required sections or has structural issues:\n"
                f"{issues_list}\n\n"
                f"Please regenerate the report with all 10 required sections.\n"
                f"</validation_error>"
            )
        return cast(str, text)

    def _on_exhausted(result: QueryResult | None, error: Exception) -> str:
        if result is None:
            raise error
        full_text = last_full_text[0]
        if not full_text and report_path.exists():
            full_text = strip_report_preamble(
                report_path.read_text(),
                FINDINGS_REPORT_HEADINGS,
            )
        if not full_text:
            raise RuntimeError("Report generation produced no output after 3 attempts")

        remaining = validate_report_structure(full_text)
        if remaining:
            logger.warning(
                f"Returning incomplete report after 3 attempts. Remaining issues: {remaining}"
            )
            warning_header = (
                "> **WARNING: This report is incomplete.** "
                f"Missing sections: {', '.join(remaining)}\n\n"
            )
            full_text = warning_header + full_text

        return full_text

    return cast(
        str,
        await agent_retry_loop(
            query_fn=_query,
            validate_fn=_validate,
            prompt=user_prompt,
            agent_name="Findings",
            on_exhausted=_on_exhausted,
        ),
    )


def _report_file_state(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_mtime_ns, stat.st_size)
