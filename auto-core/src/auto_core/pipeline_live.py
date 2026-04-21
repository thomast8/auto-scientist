"""Bridge between the orchestrator worker thread and the Textual app.

Also contains summarizer integration helpers extracted from the Orchestrator.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from io import TextIOWrapper
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console, RenderableType
from textual.css.query import NoMatches

from auto_core.widgets import (
    AgentPanel,
    IterationContainer,
    MetricsBar,
    _format_elapsed,
    console,
)

if TYPE_CHECKING:
    from auto_scientist.app import PipelineApp
    from auto_scientist.desktop_notifier import DesktopNotifier

logger = logging.getLogger(__name__)


class PipelineLive:
    """Bridge between the orchestrator (worker thread) and the Textual app.

    In app mode (_app is set): mounts widgets via call_from_thread.
    In headless mode (_app is None): tracks state only, no rendering.
    """

    def __init__(self) -> None:
        self._panels: list[AgentPanel] = []
        self._app: PipelineApp | None = None
        self._current_iteration: IterationContainer | None = None
        self._file_console: Console | None = None
        self._file_handle: TextIOWrapper | None = None
        self._notifier: DesktopNotifier | None = None

    def attach_notifier(self, notifier: DesktopNotifier) -> None:
        """Register a DesktopNotifier for agent/iteration/run lifecycle events."""
        self._notifier = notifier

    def start(self, log_path: Path | None = None) -> None:
        """Open the optional log file."""
        if log_path:
            self._file_handle = log_path.open("a")
            self._file_console = Console(
                file=self._file_handle,
                no_color=True,
                width=120,
            )

    def stop(self) -> None:
        """Close the log file."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            self._file_console = None

    def add_panel(self, panel: AgentPanel) -> None:
        """Track a panel and mount it in the app if running."""
        self._panels.append(panel)
        if self._app is not None:
            self._app.call_from_thread(self._app._mount_panel, panel)

    def mount_restored_panel(self, panel_data: dict) -> None:
        """Mount a pre-built collapsed panel into the current iteration.

        Used when resuming from a specific agent: panels for agents that were
        loaded from disk are shown with their original stats and content.
        *panel_data* has the same shape as entries in ``mount_restored_iteration``'s
        *panels* list (name, model, style, done_summary, tokens, etc.).
        """
        panel = AgentPanel(
            name=panel_data["name"],
            model=panel_data["model"],
            style=panel_data.get("style", "cyan"),
            restored=True,
        )

        self._panels.append(panel)
        if self._app is not None:

            def _do_mount():
                self._app._mount_panel(panel)
                # Pre-set metadata so _build_footer() works
                panel.input_tokens = panel_data.get("input_tokens", 0)
                panel.output_tokens = panel_data.get("output_tokens", 0)
                panel.thinking_tokens = panel_data.get("thinking_tokens", 0)
                panel.num_turns = panel_data.get("num_turns", 0)
                for line in panel_data.get("lines", []):
                    panel.all_lines.append(line)
                    panel._write_to_richlog(line)
                panel.complete(panel_data.get("done_summary", ""))
                panel._apply_complete_dom()
                # Override _end_time AFTER complete() so saved elapsed is preserved
                panel._end_time = panel.start_time + panel_data.get("elapsed_seconds", 0)
                # Carry per-panel stats into the persistent header totals
                try:
                    bar = self._app.query_one(MetricsBar)
                except NoMatches:
                    bar = None
                if bar is not None:
                    bar.carry_over(panel)

            self._app.call_from_thread(_do_mount)

        if self._file_console is not None:
            self._file_console.print(
                f"[{panel.panel_name}] {panel_data.get('done_summary', '')} "
                f"(restored from previous run)"
            )

    def collapse_panel(
        self,
        panel: AgentPanel,
        done_summary: str = "",
    ) -> None:
        """Mark a panel as complete and accumulate stats."""
        panel.complete(done_summary)
        if self._app is not None:
            self._app.call_from_thread(
                self._app._do_panel_collapse,
                panel,
            )
        if self._file_console is not None:
            self._file_console.print(
                f"[{panel.panel_name}] {panel.done_summary} ({panel._build_footer()})"
            )
        if self._notifier is not None:
            total_tokens = panel.input_tokens + panel.output_tokens + panel.thinking_tokens
            self._notifier.agent_done(
                panel.panel_name,
                _format_elapsed(panel.elapsed),
                panel.done_summary,
                num_turns=panel.num_turns,
                total_tokens=total_tokens,
            )

    def start_iteration(self, title: int | str, *, max_iterations: int | None = None) -> None:
        """Begin an iteration container."""
        if isinstance(title, int) and max_iterations is not None:
            iter_title = f"Iteration {title}/{max_iterations}"
        elif isinstance(title, int):
            iter_title = f"Iteration {title}"
        else:
            iter_title = title
        container = IterationContainer(iter_title=iter_title)
        self._current_iteration = container
        if self._app is not None:
            self._app.call_from_thread(
                self._app._mount_iteration,
                container,
            )
        if self._file_console is not None:
            self._file_console.print(f"\n{'=' * 60}")
            self._file_console.print(iter_title)
            self._file_console.print(f"{'=' * 60}")
        if self._notifier is not None:
            self._notifier.set_iteration(iter_title)

    def end_iteration(self, subtitle: str, style: str, summary_text: str = "") -> None:
        """Finalize the iteration container with a result."""
        label = subtitle
        if self._current_iteration is not None:
            if self._app is not None:
                self._app.call_from_thread(
                    self._current_iteration.set_result,
                    subtitle,
                    style,
                    summary_text,
                )
            else:
                self._current_iteration.set_result(subtitle, style, summary_text)
            label = f"{self._current_iteration._iter_title}: {subtitle}"
        if self._file_console is not None:
            self._file_console.print(f"--- {label} ---")
        if self._notifier is not None:
            self._notifier.iteration_done(label, summary_text)

    def notify_run_complete(self, status: str, summary: str) -> None:
        """Fire a terminus notification for the whole run, if a notifier is attached."""
        if self._notifier is not None:
            self._notifier.run_complete(status, summary)

    def flush_completed(self) -> None:
        """Clear iteration state. Panels stay mounted in the DOM."""
        self._current_iteration = None

    def remove_panel(self, panel: AgentPanel) -> None:
        """Remove a panel from tracking."""
        if panel in self._panels:
            self._panels.remove(panel)
        if self._app is not None:
            self._app.call_from_thread(panel.remove)  # type: ignore[arg-type]

    def update_status(self, **kwargs) -> None:
        """Update the metrics bar fields."""
        if self._app is not None:
            self._app.call_from_thread(
                self._app._on_status_update,
                **kwargs,
            )

    def log(self, message: str) -> None:
        """Write a message to the log file only (no terminal output)."""
        if self._file_console is not None:
            self._file_console.print(message)

    def mount_banner(self, renderable: RenderableType) -> None:
        """Mount the startup banner into the banner area (above Run)."""
        if self._app is not None:
            self._app.call_from_thread(
                self._app._mount_banner,
                renderable,
            )
        else:
            console.print(renderable)
        if self._file_console is not None:
            self._file_console.print(renderable)

    def print_static(self, renderable: RenderableType) -> None:
        """Print a renderable. In app mode, mount as Static widget."""
        if self._app is not None:
            self._app.call_from_thread(
                self._app._mount_static,
                renderable,
            )
        else:
            console.print(renderable)
        if self._file_console is not None:
            self._file_console.print(renderable)

    def add_rule(self, rule: RenderableType) -> None:
        """Add a rule/separator. In app mode, mount as Static widget."""
        if self._app is not None:
            self._app.call_from_thread(self._app._mount_static, rule)
        if self._file_console is not None:
            self._file_console.print(rule)

    def has_panel(self, panel: AgentPanel) -> bool:
        """Check if a panel is tracked."""
        return panel in self._panels

    @property
    def panel_count(self) -> int:
        """Number of tracked panels that are not yet done."""
        return sum(1 for p in self._panels if not p.done)

    def mount_restored_iteration(
        self,
        title: str,
        result_text: str,
        result_style: str,
        summary: str,
        panels: list[dict],
    ) -> None:
        """Mount a pre-built collapsed iteration from saved manifest data.

        Each entry in *panels* must have keys: name, model, style,
        done_summary, input_tokens, output_tokens, num_turns, elapsed_seconds, lines.
        """
        if self._app is None:
            return

        def _do_mount():
            container = IterationContainer(iter_title=title, restored=True)
            self._app.query_one("#run-area").mount(container)
            try:
                bar = self._app.query_one(MetricsBar)
            except NoMatches:
                bar = None

            for p in panels:
                panel = AgentPanel(
                    name=p["name"],
                    model=p["model"],
                    style=p.get("style", "cyan"),
                    restored=True,
                )
                container.mount(panel)
                container.add_panel(panel)
                # Pre-set metadata so _build_footer() works
                panel.input_tokens = p.get("input_tokens", 0)
                panel.output_tokens = p.get("output_tokens", 0)
                panel.thinking_tokens = p.get("thinking_tokens", 0)
                panel.num_turns = p.get("num_turns", 0)
                # Populate saved summary lines so the panel is expandable
                for line in p.get("lines", []):
                    panel.all_lines.append(line)
                    panel._write_to_richlog(line)
                panel.complete(p.get("done_summary", ""))
                panel._apply_complete_dom()
                # Override _end_time AFTER complete() so saved elapsed is preserved
                panel._end_time = panel.start_time + p.get("elapsed_seconds", 0)
                # Carry per-panel stats into the persistent header totals
                if bar is not None:
                    bar.carry_over(panel)

            container.set_result(result_text, result_style, summary)

        self._app.call_from_thread(_do_mount)

    def wait_for_dismiss(self) -> None:
        """No-op. PipelineApp handles dismiss via key binding."""

    def refresh(self) -> None:
        """No-op. Widget refresh is handled automatically."""


# ---------------------------------------------------------------------------
# Summarizer integration helpers (extracted from Orchestrator)
# ---------------------------------------------------------------------------


async def with_summaries(
    coro_fn: Callable[..., Any],
    agent_name: str,
    message_buffer: list[str],
    panel: AgentPanel | None,
    live: PipelineLive,
    summary_model: str | None,
) -> Any:
    """Wrap an agent call in run_with_summaries if enabled.

    When a panel is provided, summaries are routed to it via
    summary_collector callback instead of being printed directly.
    """
    from auto_core.summarizer import run_with_summaries

    if not summary_model:
        result = await coro_fn(message_buffer)
        if panel is not None:
            apply_sdk_usage(panel)
        return result

    if panel is not None:
        summary_collector: list[tuple[str, str, str]] = []
        seen = 0

        async def _poll_collector():
            """Poll collector and push to panel."""
            nonlocal seen
            while True:
                await asyncio.sleep(0.5)
                new_entries = summary_collector[seen:]
                for _name, summary, label in new_entries:
                    panel.add_line(f"[{label}] {summary}")
                    live.refresh()
                seen = len(summary_collector)

        poll_task = asyncio.create_task(_poll_collector())
        try:
            result = await run_with_summaries(
                coro_fn,
                agent_name,
                summary_model,
                message_buffer,
                summary_collector=summary_collector,
            )
            apply_sdk_usage(panel)
        finally:
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task
            # Flush entries added since last poll drain
            for _name, summary, label in summary_collector[seen:]:
                panel.add_line(f"[{label}] {summary}")
            live.refresh()
        return result

    return await run_with_summaries(
        coro_fn,
        agent_name,
        summary_model,
        message_buffer,
    )


def collapse_panel(
    panel: AgentPanel, live: PipelineLive, summary_model: str | None, fallback: str = ""
) -> None:
    """Collapse a panel, preferring the summarizer's done line over a fallback."""
    if summary_model and panel.lines:
        # Summarizer populated the panel; let complete() use the last line
        live.collapse_panel(panel)
    else:
        live.collapse_panel(panel, fallback)


def apply_sdk_usage(panel: AgentPanel) -> None:
    """Read token usage from the last SDK query and apply it to a panel."""
    from auto_core.sdk_utils import collect_text_from_query

    usage = getattr(collect_text_from_query, "last_usage", {})
    if not usage:
        return
    # Claude Code SDK splits input tokens across cache buckets:
    # input_tokens (non-cached) + cache_creation + cache_read = total input
    in_tok = (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
    )
    panel.set_stats(
        input_tokens=in_tok,
        output_tokens=usage.get("output_tokens", 0),
        thinking_tokens=usage.get("thinking_tokens", 0),
        num_turns=usage.get("num_turns", 0),
    )


async def generate_iteration_summary(live: PipelineLive, summary_model: str | None) -> str:
    """Generate a combined recap from all agents' done_summaries for the current iteration."""
    if not summary_model:
        return ""
    container = live._current_iteration
    if container is None:
        return ""
    summaries = [
        (p.panel_name, p.done_summary)
        for p in getattr(container, "_panels", [])
        if p.done and p.done_summary
    ]
    if not summaries:
        return ""
    from auto_core.summarizer import summarize_iteration

    return await summarize_iteration(summaries, summary_model)
