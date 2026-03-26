"""PipelineLive: bridge between the orchestrator (worker thread) and the Textual app."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console, RenderableType

from auto_scientist.ui.styles import console
from auto_scientist.ui.widgets import AgentPanel, IterationContainer

if TYPE_CHECKING:
    from auto_scientist.ui.app import PipelineApp


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
        self._file_handle = None

    def start(self, log_path: Path | None = None) -> None:
        """Open the optional log file."""
        if log_path:
            self._file_handle = log_path.open("a")
            self._file_console = Console(
                file=self._file_handle, no_color=True, width=120,
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

    def collapse_panel(
        self, panel: AgentPanel, done_summary: str = "",
    ) -> None:
        """Mark a panel as complete and accumulate stats."""
        panel.complete(done_summary)
        if self._app is not None:
            self._app.call_from_thread(
                self._app._do_panel_collapse, panel,
            )
        if self._file_console is not None:
            self._file_console.print(
                f"[{panel.panel_name}] "
                f"{panel.done_summary} ({panel._build_footer()})"
            )

    def start_iteration(self, title: int | str) -> None:
        """Begin an iteration container."""
        iter_title = (
            f"Iteration {title}" if isinstance(title, int) else title
        )
        container = IterationContainer(iter_title=iter_title)
        self._current_iteration = container
        if self._app is not None:
            self._app.call_from_thread(
                self._app._mount_iteration, container,
            )
        if self._file_console is not None:
            self._file_console.print(f"\n{'=' * 60}")
            self._file_console.print(iter_title)
            self._file_console.print(f"{'=' * 60}")

    def end_iteration(self, subtitle: str, style: str) -> None:
        """Finalize the iteration container with a result."""
        if self._current_iteration is not None:
            if self._app is not None:
                self._app.call_from_thread(
                    self._current_iteration.set_result, subtitle, style,
                )
            else:
                self._current_iteration.set_result(subtitle, style)
        if self._file_console is not None:
            label = subtitle
            if self._current_iteration is not None:
                label = (
                    f"{self._current_iteration.border_title}: {subtitle}"
                )
            self._file_console.print(f"--- {label} ---")

    def flush_completed(self) -> None:
        """Clear iteration state. Panels stay mounted in the DOM."""
        self._current_iteration = None

    def remove_panel(self, panel: AgentPanel) -> None:
        """Remove a panel from tracking."""
        if panel in self._panels:
            self._panels.remove(panel)
        if self._app is not None:
            self._app.call_from_thread(panel.remove)

    def update_status(self, **kwargs) -> None:
        """Update the metrics bar fields."""
        if self._app is not None:
            self._app.call_from_thread(
                self._app._on_status_update, **kwargs,
            )

    def log(self, message: str) -> None:
        """Write a message to the log file only (no terminal output)."""
        if self._file_console is not None:
            self._file_console.print(message)

    def print_static(self, renderable: RenderableType) -> None:
        """Print a renderable. In app mode, mount as Static widget."""
        if self._app is not None:
            self._app.call_from_thread(
                self._app._mount_static, renderable,
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

    def wait_for_dismiss(self) -> None:
        """No-op. PipelineApp handles dismiss via key binding."""

    def refresh(self) -> None:
        """No-op. Widget refresh is handled automatically."""
