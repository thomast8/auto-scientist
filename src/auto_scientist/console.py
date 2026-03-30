"""Textual-based CLI dashboard for the auto-scientist pipeline.

Provides:
- AgentPanel: Collapsible + RichLog widget for each agent phase
- MetricsBar: Persistent metrics display (sparkline, tokens, phase)
- IterationContainer: Bordered container grouping panels per iteration
- AgentDetailScreen: Full-screen view of one agent's output
- QuitConfirmScreen: Modal confirmation dialog for quit
- PipelineCommandProvider: Command palette with navigation and control
- PipelineLive: Bridge between orchestrator (worker thread) and Textual app
- PipelineApp: Textual App with screens, command palette, and message handlers
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import subprocess
import threading
import time
from collections.abc import Callable
from functools import partial
from io import TextIOWrapper
from pathlib import Path
from typing import Literal

from rich.console import Console, RenderableType
from rich.markup import escape as rich_escape
from rich.text import Text
from textual._context import NoActiveAppError
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Label,
    RichLog,
    Static,
)
from textual.widgets._collapsible import CollapsibleTitle
from textual.worker import Worker, WorkerState

from auto_scientist.latex_to_unicode import latex_to_unicode
from auto_scientist.preferences import load_theme, save_theme

logger = logging.getLogger(__name__)


# Module-level console for one-time prints (startup banner in headless mode, etc.)
console = Console()

# Agent style palette
AGENT_STYLES = {
    "Analyst": "green",
    "Scientist": "cyan",
    "Coder": "magenta1",
    "Ingestor": "bright_red",
    "Report": "blue",
    "Critic": "yellow",
    "Debate": "yellow",
    "Assessor": "blue",
    "Results": "dim",
}

# Short descriptions shown immediately when an agent panel opens (before first summary)
AGENT_DESCRIPTIONS: dict[str, str] = {
    "Ingestor": "Preparing and canonicalizing raw data for experiment scripts...",
    "Analyst": "Analyzing experiment results and producing quantitative assessments...",
    "Scientist": "Formulating hypotheses and planning the next experiment...",
    "Critic": "Challenging the plan through critical debate...",
    "Revision": "Revising the experiment plan based on critique feedback...",
    "Coder": "Implementing the experiment plan as a Python script and running it...",
    "Report": "Generating a comprehensive summary report of all findings...",
    "Assessor": "Evaluating investigation completeness against the stated goal...",
}

# Maps orchestrator phase names to colors (matches the active agent)
PHASE_STYLES = {
    "INGESTION": "bright_red",
    "ANALYZE": "green",
    "PLAN": "cyan",
    "DEBATE": "yellow",
    "REVISE": "cyan",
    "IMPLEMENT": "magenta1",
    "REPORT": "blue",
    "ASSESS": "blue",
    "STOP_DEBATE": "yellow",
    "STOP_REVISE": "cyan",
}


def _format_elapsed(seconds: float) -> str:
    """Format seconds into a human-readable duration like '4m 32s'."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# AgentPanel widget
# ---------------------------------------------------------------------------


class AgentPanel(Widget):
    """Collapsible panel widget for a single agent phase.

    Contains a Collapsible wrapping a RichLog. The Collapsible title shows
    agent name, model, and stats. On completion, collapses to show just
    the done summary.
    """

    DEFAULT_CSS = """
    AgentPanel {
        width: 100%;
        height: auto;
        min-height: 3;
        padding: 0 0;
        margin: 0 0;
        border: round $accent;
        border-subtitle-align: right;
        border-title-align: left;
        background: $surface;
    }
    AgentPanel:hover {
        background: $surface-lighten-1;
    }
    AgentPanel Collapsible {
        border-top: none;
        padding-bottom: 0;
        padding-left: 0;
        background: transparent;
    }
    AgentPanel Collapsible Contents {
        padding: 0 0 0 1;
        background: transparent;
    }
    AgentPanel RichLog {
        height: auto;
        max-height: 20;
        overflow-y: auto;
        background: transparent;
    }
    AgentPanel .agent-description {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }
    AgentPanel CollapsibleTitle {
        width: 100%;
        display: none;
    }
    AgentPanel Collapsible:disabled CollapsibleTitle {
        opacity: 1;
    }
    """

    def __init__(
        self,
        name: str,
        model: str,
        style: str = "cyan",
        description: str = "",
        restored: bool = False,
    ) -> None:
        super().__init__()
        self._panel_name = name
        self.model = model
        self.panel_style = style
        self._is_restored = restored
        self.all_lines: list[str] = []
        self.start_time = time.monotonic()
        self.input_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        self.num_turns = 0
        self.done = False
        self.done_summary = ""
        self.error_msg = ""
        self._in_error = False
        self._end_time: float | None = None
        # Resolve description: explicit > exact lookup > prefix lookup (e.g. "Critic/X" -> "Critic")
        if not description:
            description = AGENT_DESCRIPTIONS.get(
                name, AGENT_DESCRIPTIONS.get(name.split("/")[0], "")
            )
        self._description = description

    def compose(self) -> ComposeResult:
        with Collapsible(title=self._make_title(), collapsed=False):
            if self._description:
                yield Static(self._description, classes="agent-description")
            yield RichLog(auto_scroll=True, markup=True, wrap=True)

    def on_mount(self) -> None:
        self._refresh_timer = self.set_interval(1, self._tick)
        self._dot_count = 0
        self._apply_border_color()
        restored_suffix = " (restored)" if self._is_restored else ""
        self.border_title = f"{self._panel_name} ({self.model}){restored_suffix}"
        self.border_subtitle = self._build_footer()

    # Rich markup color names that need translation for Textual CSS styles.color
    _RICH_TO_TEXTUAL_COLOR: dict[str, str] = {
        "bright_red": "ansi_bright_red",
        "magenta1": "ansi_bright_magenta",
        "yellow": "ansi_yellow",
        "green": "ansi_green",
        "cyan": "ansi_cyan",
        "blue": "ansi_blue",
    }

    def _apply_border_color(self) -> None:
        """Force the border and title colors to match the agent style.

        Without this, some Textual themes override Rich markup colors on the
        CollapsibleTitle widget, making completed panels appear grey.
        Uses red when the panel is in an error state.
        """
        color = "red" if self._in_error else self.panel_style
        css_color = self._RICH_TO_TEXTUAL_COLOR.get(color, color)
        try:
            title_widget = self.query_one(CollapsibleTitle)
            title_widget.styles.color = css_color
        except NoMatches:
            pass
        border_type: Literal["dashed", "round"] = "dashed" if self._is_restored else "round"
        self.styles.border = (border_type, css_color)
        self.styles.border_title_color = css_color
        self.styles.border_subtitle_color = css_color

    def _tick(self) -> None:
        """Update border subtitle with elapsed time and animate description dots."""
        if self.done and hasattr(self, "_refresh_timer"):
            self._refresh_timer.stop()
            return
        self.border_subtitle = self._build_footer()
        if self._description and not self.all_lines:
            self._dot_count = (self._dot_count + 1) % 4
            dots = "." * self._dot_count if self._dot_count else ""
            base = self._description.rstrip(".")
            try:
                desc_widget = self.query_one(".agent-description", Static)
                desc_widget.update(f"{base}{dots}")
            except NoMatches:
                pass

    def on_resize(self, event) -> None:
        """Re-render RichLog content at new width."""
        if not self.all_lines:
            return
        try:
            rich_log = self.query_one(RichLog)
        except NoMatches:
            return
        rich_log.clear()
        for line in self.all_lines:
            rich_log.write(Text(line), expand=True)

    def _make_title(self) -> str:
        """Build the Collapsible title string (empty while running, summary when done)."""
        return ""

    def _update_title(self) -> None:
        """Update the Collapsible title in the DOM."""
        try:
            collapsible = self.query_one(Collapsible)
        except NoMatches:
            return
        collapsible.title = self._make_title()

    @property
    def panel_name(self) -> str:
        return self._panel_name

    @property
    def lines(self) -> list[str]:
        """Backward-compatible property. Returns all_lines."""
        return self.all_lines

    def add_line(self, text: str) -> None:
        """Append a summary line. Thread-safe: routes DOM update to UI thread."""
        if self.done:
            return
        cleaned = latex_to_unicode(" ".join(text.split()))
        self.all_lines.append(cleaned)
        try:
            app = self.app
        except NoActiveAppError:
            return
        if app._thread_id == threading.get_ident():
            self._write_to_richlog(cleaned)
        else:
            app.call_from_thread(self._write_to_richlog, cleaned)

    def _write_to_richlog(self, text: str) -> None:
        """Write a line to the RichLog widget (must be called from UI thread)."""
        if len(self.all_lines) == 1:
            for desc in self.query(".agent-description"):
                desc.remove()
        try:
            rich_log = self.query_one(RichLog)
        except NoMatches:
            return
        # Auto-scroll main container if user is near the bottom
        try:
            app = self.app
            near_bottom = app._is_near_bottom()  # type: ignore[attr-defined]
        except (NoActiveAppError, NoMatches, AttributeError):
            near_bottom = False
        rich_log.write(Text(text), expand=True)
        if near_bottom:
            app._scroll_to_end()  # type: ignore[attr-defined]

    def complete(self, done_summary: str = "") -> None:
        """Mark this panel as done.

        If done_summary is empty and the panel has lines, the last line
        is used as the done summary. Sets metadata immediately (thread-safe),
        defers DOM updates to the UI thread.
        """
        if self.done:
            return
        self.done = True
        if done_summary:
            self.done_summary = latex_to_unicode(done_summary)
        elif self.all_lines:
            self.done_summary = self.all_lines[-1]
        self._end_time = time.monotonic()

    def _apply_complete_dom(self) -> None:
        """Apply completion state to DOM (must be called from UI thread)."""
        if self._in_error:
            return  # Error DOM is already applied; don't overwrite it
        for desc in self.query(".agent-description"):
            desc.remove()
        try:
            collapsible = self.query_one(Collapsible)
        except NoMatches:
            return
        summary = self.done_summary
        # Strip summarizer label prefixes like "[done] " or "[15s] "
        summary = re.sub(r"^\[\w+\]\s+", "", summary)
        # Escape Rich markup in the summary to prevent broken rendering
        collapsible.title = f"[{self.panel_style}]{rich_escape(summary)}[/]"
        self.border_subtitle = self._build_footer()
        # Show the CollapsibleTitle now that we have content to toggle
        try:
            title_widget = collapsible.query_one(CollapsibleTitle)
            title_widget.styles.display = "block"
        except NoMatches:
            pass
        # Suppress Textual's built-in scroll-on-collapse (Collapsible._watch_collapsed
        # calls self.call_after_refresh(self.scroll_visible) unconditionally)
        object.__setattr__(collapsible, "scroll_visible", lambda *a, **kw: None)
        collapsible.collapsed = True
        # Restore inherited method for user-initiated toggles
        object.__delattr__(collapsible, "scroll_visible")
        self._apply_border_color()
        if len(self.all_lines) <= 1:
            collapsible.disabled = True
            try:
                title_widget = collapsible.query_one(CollapsibleTitle)
                title_widget.collapsed_symbol = "●"
                title_widget.expanded_symbol = "●"
                title_widget._update_label()
            except NoMatches:
                pass

    def error(self, msg: str) -> None:
        """Mark this panel as errored. Thread-safe: routes DOM update to UI thread."""
        if self.done:
            return
        self.done = True
        self.error_msg = msg
        self._in_error = True
        self._end_time = time.monotonic()
        try:
            app = self.app
        except NoActiveAppError:
            return
        if app._thread_id == threading.get_ident():
            self._apply_error_dom(msg)
        else:
            app.call_from_thread(self._apply_error_dom, msg)

    def _apply_error_dom(self, msg: str) -> None:
        """Apply error state to DOM (must be called from UI thread)."""
        for desc in self.query(".agent-description"):
            desc.remove()
        try:
            collapsible = self.query_one(Collapsible)
            rich_log = self.query_one(RichLog)
        except NoMatches:
            return
        collapsible.title = f"[red][error] {rich_escape(msg)}[/red]"
        self.border_subtitle = self._build_footer()
        # Show the CollapsibleTitle for the error message
        try:
            title_widget = collapsible.query_one(CollapsibleTitle)
            title_widget.styles.display = "block"
        except NoMatches:
            pass
        rich_log.write(Text(f"[error] {msg}", style="red"))
        self._apply_border_color()

    def set_tokens(self, input_tokens: int, output_tokens: int, thinking_tokens: int = 0) -> None:
        """Set token usage metadata."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.thinking_tokens = thinking_tokens

    def set_stats(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        num_turns: int = 0,
    ) -> None:
        """Set rich usage stats from SDK ResultMessage or direct API calls."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.thinking_tokens = thinking_tokens
        self.num_turns = num_turns

    @property
    def elapsed(self) -> float:
        """Elapsed seconds since panel creation."""
        end = self._end_time if self._end_time else time.monotonic()
        return end - self.start_time

    def _build_footer(self) -> str:
        """Build the footer subtitle string."""
        parts = [_format_elapsed(self.elapsed)]
        if self.input_tokens or self.output_tokens:
            tok = f"{self.input_tokens:,} in"
            if self.thinking_tokens:
                tok += f" / {self.thinking_tokens:,} think"
            tok += f" / {self.output_tokens:,} out"
            parts.append(tok)
        if self.num_turns:
            parts.append(f"{self.num_turns} {'turn' if self.num_turns == 1 else 'turns'}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# MetricsBar widget
# ---------------------------------------------------------------------------


class MetricsBar(Widget):
    """Persistent metrics bar showing iteration, phase, scores, tokens."""

    DEFAULT_CSS = """
    MetricsBar {
        dock: top;
        height: 1;
        background: $surface;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.start_time = time.monotonic()
        self.iteration = 0
        self.phase = ""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_thinking_tokens = 0
        self.total_turns = 0
        self.scores: list[float] = []
        self.finished: bool = False
        self._end_time: float | None = None

    def on_mount(self) -> None:
        self.set_interval(1, self.refresh)

    def set_status(
        self,
        iteration: int | None = None,
        phase: str | None = None,
    ) -> None:
        """Update metrics bar fields. Only non-None values are changed."""
        if iteration is not None:
            self.iteration = iteration
        if phase is not None:
            self.phase = phase
        self.refresh()

    def finish(self) -> None:
        """Freeze the elapsed timer at the current value."""
        self.finished = True
        self._end_time = time.monotonic()

    def add_agent_stats(self, panel: AgentPanel) -> None:
        """Accumulate a completed agent's stats into the running totals."""
        self.total_input_tokens += panel.input_tokens
        self.total_output_tokens += panel.output_tokens
        self.total_thinking_tokens += panel.thinking_tokens
        self.total_turns += panel.num_turns

    def render(self) -> Text:
        end = self._end_time if self._end_time else time.monotonic()
        elapsed = _format_elapsed(end - self.start_time)
        phase_style = PHASE_STYLES.get(self.phase, "cyan")

        line = Text()
        line.append(f" Iteration {self.iteration}", style="bold")
        line.append("  ")
        line.append(self.phase, style=phase_style)
        line.append("  ", style="dim")
        line.append(elapsed, style="dim")

        total_tokens = self.total_input_tokens + self.total_output_tokens
        if total_tokens > 0:
            tokens = f"{self.total_input_tokens:,} in"
            if self.total_thinking_tokens:
                tokens += f" / {self.total_thinking_tokens:,} think"
            tokens += f" / {self.total_output_tokens:,} out"
            line.append(f" | {tokens}", style="dim")
        if self.total_turns:
            label = "turn" if self.total_turns == 1 else "turns"
            line.append(f" | {self.total_turns} {label}", style="dim")

        if self.scores:
            blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
            max_s = max(self.scores) or 1
            spark = ""
            for s in self.scores[-20:]:
                idx = min(int(s / max_s * 8), 8)
                spark += blocks[idx]
            line.append(f"  [{spark}]", style="dim")

        return line


# ---------------------------------------------------------------------------
# IterationContainer
# ---------------------------------------------------------------------------


class IterationToggle(Static, can_focus=True):
    """Clickable toggle for expanding/collapsing iteration agent panels."""

    DEFAULT_CSS = """
    IterationToggle {
        padding: 0 1;
        color: $text-muted;
    }
    IterationToggle:hover {
        background: $surface;
    }
    IterationToggle:focus {
        background: $surface;
    }
    """

    class Toggled(Message):
        """Posted when the toggle is activated."""

    BINDINGS = [Binding("enter", "activate", show=False)]

    async def _on_click(self, event) -> None:
        event.stop()
        self.post_message(self.Toggled())

    def action_activate(self) -> None:
        self.post_message(self.Toggled())


class IterationContainer(Vertical):
    """Bordered container grouping panels for one iteration."""

    SPINNER_CHARS = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    DEFAULT_CSS = """
    IterationContainer {
        height: auto;
        border: round grey;
        transition: border 300ms in_out_cubic;
    }
    IterationContainer .iteration-recap {
        padding: 0 1;
        color: $text-muted;
        text-style: italic;
    }
    """

    def __init__(self, iter_title: str, restored: bool = False) -> None:
        super().__init__()
        self._iter_title = iter_title
        self._is_restored = restored
        restored_suffix = " (restored)" if restored else ""
        self.border_title = f"{iter_title}{restored_suffix}"
        self._in_progress = True
        self._spinner_index = 0
        self._panels: list[AgentPanel] = []
        self._is_collapsed = False

    def on_mount(self) -> None:
        self._spinner_timer = self.set_interval(1 / 10, self._tick_spinner)
        if self._is_restored:
            self.styles.border = ("dashed", "grey")

    def _tick_spinner(self) -> None:
        if not self._in_progress:
            self._spinner_timer.stop()
            return
        char = self.SPINNER_CHARS[self._spinner_index % len(self.SPINNER_CHARS)]
        restored_suffix = " (restored)" if self._is_restored else ""
        self.border_title = f"{char} {self._iter_title}{restored_suffix}"
        self._spinner_index += 1

    def add_panel(self, panel: AgentPanel) -> None:
        """Register an agent panel as a child of this iteration."""
        self._panels.append(panel)

    def collapse_iteration(self, summary_text: str = "") -> None:
        """Collapse all agent panels and show an aggregated recap.

        Called once when the iteration finishes. Subsequent show/hide
        is handled by toggle_iteration().
        """
        if self._is_collapsed:
            return
        self._is_collapsed = True

        # Aggregate metrics from all panels
        total_elapsed = sum(p.elapsed for p in self._panels)
        total_in = sum(p.input_tokens for p in self._panels)
        total_out = sum(p.output_tokens for p in self._panels)
        total_think = sum(p.thinking_tokens for p in self._panels)
        total_turns = sum(p.num_turns for p in self._panels)

        # Build aggregated subtitle: "4m 32s | 1,200 in / 800 out | 5 turns"
        parts: list[str] = []
        parts.append(_format_elapsed(total_elapsed))
        if total_in or total_out:
            tok = f"{total_in:,} in"
            if total_think:
                tok += f" / {total_think:,} think"
            tok += f" / {total_out:,} out"
            parts.append(tok)
        if total_turns:
            label = "turn" if total_turns == 1 else "turns"
            parts.append(f"{total_turns} {label}")
        self.border_subtitle = " | ".join(parts)

        # Mount recap summary text (before agent panels)
        first_panel = self._panels[0] if self._panels else None
        if summary_text:
            summary_text = latex_to_unicode(summary_text)
            recap = Static(summary_text, classes="iteration-recap")
            if first_panel:
                self.mount(recap, before=first_panel)
            else:
                self.mount(recap)

        # Mount toggle widget
        n = len(self._panels)
        toggle = IterationToggle(f"▶ Show {n} agent{'s' if n != 1 else ''}")
        if first_panel:
            self.mount(toggle, before=first_panel)
        else:
            self.mount(toggle)

        # Hide all agent panels
        for panel in self._panels:
            panel.styles.display = "none"

    def toggle_iteration(self) -> None:
        """Toggle agent panel visibility between shown and hidden."""
        if self._is_collapsed:
            # Expand
            self._is_collapsed = False
            for panel in self._panels:
                panel.styles.display = "block"
            for toggle in self.query(IterationToggle):
                n = len(self._panels)
                toggle.update(f"▼ Hide {n} agent{'s' if n != 1 else ''}")
        else:
            # Collapse
            self._is_collapsed = True
            for panel in self._panels:
                panel.styles.display = "none"
            for toggle in self.query(IterationToggle):
                n = len(self._panels)
                toggle.update(f"▶ Show {n} agent{'s' if n != 1 else ''}")

    def on_iteration_toggle_toggled(self, event: IterationToggle.Toggled) -> None:
        event.stop()
        self.toggle_iteration()

    def set_result(self, text: str, style: str, summary_text: str = "") -> None:
        """Set the iteration result as border subtitle and collapse."""
        self._in_progress = False
        restored_suffix = " (restored)" if self._is_restored else ""
        self.border_title = f"{self._iter_title}{restored_suffix}"
        self.border_subtitle = ""
        valid = {"red", "green", "yellow"}
        if style in valid:
            border_type: Literal["dashed", "round"] = "dashed" if self._is_restored else "round"
            self.styles.border = (border_type, style)
        if self._panels:
            self.collapse_iteration(summary_text)


# ---------------------------------------------------------------------------
# AgentDetailScreen
# ---------------------------------------------------------------------------


class AgentDetailScreen(ModalScreen):
    """Full-screen view of one agent's complete output."""

    DEFAULT_CSS = """
    AgentDetailScreen {
        align: center middle;
    }
    AgentDetailScreen > Vertical {
        width: 90%;
        height: 90%;
        border: solid $accent;
        background: $surface;
    }
    AgentDetailScreen > Vertical > RichLog {
        height: 1fr;
    }
    AgentDetailScreen > Vertical > Static {
        height: auto;
        padding: 0 1;
        background: $primary-background;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Back", show=True),
    ]

    def __init__(
        self,
        panel_name: str,
        model: str,
        stats: str,
        lines: list[str],
    ) -> None:
        super().__init__()
        self._panel_name = panel_name
        self._model = model
        self._stats = stats
        self._lines = lines

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f"[bold]{self._panel_name}[/bold] ({self._model}) | {self._stats}")
            yield RichLog(auto_scroll=False, markup=True, wrap=True)

    def on_mount(self) -> None:
        rich_log = self.query_one(RichLog)
        for line in self._lines:
            rich_log.write(Text(line))

    async def action_dismiss(self, result: object = None) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# QuitConfirmScreen
# ---------------------------------------------------------------------------


class QuitConfirmScreen(ModalScreen[bool]):
    """Modal confirmation dialog for quitting while pipeline runs."""

    DEFAULT_CSS = """
    QuitConfirmScreen {
        align: center middle;
    }
    QuitConfirmScreen > Vertical {
        width: 50;
        height: auto;
        border: round $error;
        background: $surface;
        padding: 1 2;
    }
    QuitConfirmScreen > Vertical > Label {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }
    QuitConfirmScreen > Vertical > Static {
        width: 100%;
        text-align: center;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("y", "yes", "Yes", show=True),
        Binding("n", "no", "No", show=True),
        Binding("escape", "no", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Pipeline is still running. Quit anyway?")
            yield Static("y/n")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# PipelineCommandProvider
# ---------------------------------------------------------------------------


class PipelineCommandProvider(Provider):
    """Command palette provider for pipeline navigation and control."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        app = self.app
        if not isinstance(app, PipelineApp):
            return

        # Static commands
        commands: list[tuple[str, Callable[[], object]]] = [
            ("Expand all panels", app.action_toggle_expand),
            ("Collapse all panels", app.action_toggle_expand),
            ("Go to top", partial(app._scroll_to, "top")),
            ("Go to bottom", partial(app._scroll_to, "bottom")),
            ("Quit", app.action_quit),
        ]

        # Theme switching
        for theme_name in sorted(app.available_themes):
            commands.append(
                (
                    f"Switch theme: {theme_name}",
                    partial(app._set_theme, theme_name),
                )
            )

        # Pipeline control
        if hasattr(app._orchestrator, "pause_requested"):
            commands.append(
                (
                    "Pause after current iteration",
                    partial(app._set_orchestrator_flag, "pause_requested"),
                )
            )
        if hasattr(app._orchestrator, "skip_to_report"):
            commands.append(
                (
                    "Skip to report",
                    partial(app._set_orchestrator_flag, "skip_to_report"),
                )
            )

        # Dynamic: go to iteration N
        for container in app.query(IterationContainer):
            title = getattr(container, "_iter_title", container.border_title) or "?"
            commands.append(
                (
                    f"Go to {title}",
                    partial(app._scroll_to_widget, container),
                )
            )

        # Dynamic: view agent details
        for panel in app.query(AgentPanel):
            commands.append(
                (
                    f"View {panel.panel_name} details",
                    partial(app._open_agent_detail, panel),
                )
            )

        # Open experiment directory (macOS)
        if app._orchestrator and hasattr(app._orchestrator, "output_dir"):
            commands.append(
                (
                    "Open experiment directory",
                    partial(
                        app._open_directory,
                        app._orchestrator.output_dir,
                    ),
                )
            )

        for label, callback in commands:
            score = matcher.match(label)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(label),
                    callback,
                )


# ---------------------------------------------------------------------------
# PipelineLive bridge
# ---------------------------------------------------------------------------


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

    def start_iteration(self, title: int | str) -> None:
        """Begin an iteration container."""
        iter_title = f"Iteration {title}" if isinstance(title, int) else title
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

    def end_iteration(self, subtitle: str, style: str, summary_text: str = "") -> None:
        """Finalize the iteration container with a result."""
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
        if self._file_console is not None:
            label = subtitle
            if self._current_iteration is not None:
                label = f"{self._current_iteration._iter_title}: {subtitle}"
            self._file_console.print(f"--- {label} ---")

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

            container.set_result(result_text, result_style, summary)

        self._app.call_from_thread(_do_mount)

    def wait_for_dismiss(self) -> None:
        """No-op. PipelineApp handles dismiss via key binding."""

    def refresh(self) -> None:
        """No-op. Widget refresh is handled automatically."""


# ---------------------------------------------------------------------------
# PipelineApp
# ---------------------------------------------------------------------------


class PipelineApp(App):
    """Textual app that runs the orchestrator and displays the dashboard."""

    COMMANDS = App.COMMANDS | {PipelineCommandProvider}

    BINDINGS = [
        Binding("ctrl+o", "toggle_expand", "Expand/Collapse", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding(
            "enter",
            "open_focused_detail",
            "Detail",
            show=False,
        ),
    ]

    DEFAULT_CSS = """
    #outer-container {
        height: 1fr;
        border: round grey;
        padding: 0 1;
    }
    #banner-area {
        height: auto;
    }
    #run-area {
        height: auto;
        border: round grey;
        transition: border 300ms in_out_cubic;
    }
    #run-area > Static {
        width: 100%;
    }
    """

    def __init__(self, orchestrator) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._finished: bool = False
        self._live: PipelineLive = PipelineLive()
        self._worker_loop: asyncio.AbstractEventLoop | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield MetricsBar()
        with VerticalScroll(id="outer-container") as outer:
            outer.border_title = "Auto-Scientist"
            yield Vertical(id="banner-area")
            with Vertical(id="run-area") as run:
                run.border_title = "Run"
                pass
        yield Footer()

    def on_mount(self) -> None:
        saved_theme = load_theme()
        if saved_theme in self.available_themes:
            self.theme = saved_theme
        self.title = "Auto-Scientist"
        self._live._app = self
        self._orchestrator._live = self._live
        self.run_worker(
            self._run_pipeline,
            thread=True,
            exit_on_error=False,
        )

    def _run_pipeline(self) -> None:
        """Run the async orchestrator in its own event loop (worker thread)."""
        loop = asyncio.new_event_loop()
        self._worker_loop = loop
        try:
            loop.run_until_complete(self._orchestrator.run())
        except asyncio.CancelledError:
            pass
        finally:
            loop.close()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR):
            self._finished = True
            try:
                bar = self.query_one(MetricsBar)
            except NoMatches:
                bar = None
            if bar is not None:
                bar.finish()

            run_area = self.query_one("#run-area")
            if event.state == WorkerState.ERROR:
                run_area.styles.border = ("round", "red")
                error = event.worker.error
                error_msg = f"{type(error).__name__}: {error}" if error else "Unknown error"
                self.notify(
                    "Pipeline failed! See error below.",
                    severity="error",
                    timeout=0,
                )
                self._mount_static(
                    Text(f"\n{error_msg}", style="red"),
                )
            else:
                run_area.styles.border = ("round", "green")
                self.notify(
                    "Pipeline complete!",
                    severity="information",
                    timeout=0,
                )

    # -- Callbacks from PipelineLive (called via call_from_thread) --

    def _do_panel_collapse(self, panel: AgentPanel) -> None:
        """Apply panel DOM collapse and handle post-collapse UI (runs on UI thread)."""
        near_bottom = self._is_near_bottom()
        panel._apply_complete_dom()
        self._on_panel_collapsed(panel)
        if near_bottom:
            self._scroll_to_end()

    def _on_panel_collapsed(self, panel: AgentPanel) -> None:
        """Handle panel completion: accumulate stats, fire toast."""
        try:
            bar = self.query_one(MetricsBar)
        except NoMatches:
            bar = None
        if bar is not None:
            bar.add_agent_stats(panel)
        elapsed = _format_elapsed(panel.elapsed)
        self.notify(f"{panel.panel_name} complete ({elapsed})")

    def _on_status_update(self, **kwargs) -> None:
        """Handle status update: update MetricsBar."""
        try:
            bar = self.query_one(MetricsBar)
        except NoMatches:
            return
        bar.set_status(**kwargs)

    # -- Scroll helpers --

    def _is_near_bottom(self) -> bool:
        """Check if the scroll view is at or near the bottom."""
        scroll = self.query_one("#outer-container", VerticalScroll)
        return scroll.scroll_offset.y >= scroll.max_scroll_y - 2

    def _scroll_to_end(self) -> None:
        """Scroll to the bottom of the main scroll area."""
        self.call_after_refresh(
            self.query_one("#outer-container").scroll_end,
            animate=False,
        )

    # -- Key binding actions --

    def action_toggle_expand(self) -> None:
        """Toggle expanded state on all AgentPanel Collapsibles and IterationContainers.

        When collapsing, skip panels that are still running so their
        live output remains visible.
        """
        panels = list(self.query(AgentPanel))
        containers = list(self.query(IterationContainer))
        if not panels and not containers:
            return

        # Determine direction from first available panel
        collapsing = True
        if panels:
            try:
                first_collapsible = panels[0].query_one(Collapsible)
                collapsing = not first_collapsible.collapsed
            except NoMatches:
                pass

        was_near_bottom = self._is_near_bottom()

        # Toggle agent panels
        for panel in panels:
            if collapsing and not panel.done:
                continue
            try:
                c = panel.query_one(Collapsible)
            except NoMatches:
                continue
            # Suppress Textual's scroll_visible during batch toggle
            object.__setattr__(c, "scroll_visible", lambda *a, **kw: None)
            c.collapsed = collapsing
            object.__delattr__(c, "scroll_visible")

        # Toggle finished iteration containers
        for container in containers:
            if container._in_progress or not container._panels:
                continue
            if (collapsing and not container._is_collapsed) or (
                not collapsing and container._is_collapsed
            ):
                container.toggle_iteration()

        if was_near_bottom:
            self._scroll_to_end()

    async def action_quit(self) -> None:
        """Quit with confirmation if pipeline is still running."""
        if not self._finished:
            self.push_screen(
                QuitConfirmScreen(),
                callback=self._handle_quit_confirm,
            )
        else:
            self.exit()

    def _handle_quit_confirm(self, confirmed: bool | None) -> None:
        """Handle quit confirmation result."""
        if confirmed:
            self._force_quit()

    def _force_quit(self) -> None:
        """Cancel the pipeline and exit."""
        if self._worker_loop is not None and self._worker_loop.is_running():
            self._worker_loop.call_soon_threadsafe(self._cancel_all_tasks)
        self.exit()

    def _cancel_all_tasks(self) -> None:
        """Cancel all tasks on the worker event loop."""
        if self._worker_loop:
            for task in asyncio.all_tasks(self._worker_loop):
                task.cancel()

    def watch_theme(self, theme_name: str) -> None:
        """Persist every theme change, regardless of how it was triggered.

        Catches changes from the custom command palette AND the built-in
        Textual ThemeProvider (which otherwise bypasses persistence).
        """
        save_theme(theme_name)

    def action_open_focused_detail(self) -> None:
        """Open detail view for the currently focused AgentPanel."""
        focused = self.focused
        if focused is None:
            return
        panel = None
        widget: Widget | None = focused
        while widget is not None:
            if isinstance(widget, AgentPanel):
                panel = widget
                break
            widget = widget.parent  # type: ignore[assignment]
        if panel is not None:
            self._open_agent_detail(panel)

    # -- Command palette helpers --

    def _open_agent_detail(self, panel: AgentPanel) -> None:
        """Push the AgentDetailScreen for a panel."""
        self.push_screen(
            AgentDetailScreen(
                panel_name=panel.panel_name,
                model=panel.model,
                stats=panel._build_footer(),
                lines=list(panel.all_lines),
            )
        )

    def _scroll_to(self, direction: str) -> None:
        """Scroll the main area to top or bottom."""
        scroll = self.query_one("#outer-container", VerticalScroll)
        if direction == "top":
            scroll.scroll_home(animate=False)
        else:
            scroll.scroll_end(animate=False)

    def _scroll_to_widget(self, widget: Widget) -> None:
        """Scroll to make a widget visible."""
        widget.scroll_visible(animate=False)

    def _set_theme(self, theme_name: str) -> None:
        """Switch to a named theme."""
        self.theme = theme_name
        self.notify(f"Theme: {theme_name}")

    def _set_orchestrator_flag(self, flag_name: str) -> None:
        """Set a boolean flag on the orchestrator."""
        setattr(self._orchestrator, flag_name, True)
        self.notify(f"{flag_name.replace('_', ' ').title()} requested")

    def _open_directory(self, path: Path) -> None:
        """Open a directory in the system file manager."""
        try:
            subprocess.Popen(["open", str(path)])
        except OSError:
            self.notify(f"Directory: {path}", timeout=10)

    # -- Mount helpers (called via call_from_thread from PipelineLive) --

    def _mount_panel(self, panel: AgentPanel) -> None:
        """Mount a panel into the current iteration container or scroll."""
        near_bottom = self._is_near_bottom()
        target = self._live._current_iteration or self.query_one("#run-area")
        target.mount(panel)
        if isinstance(target, IterationContainer):
            target.add_panel(panel)
        if near_bottom:
            self._scroll_to_end()

    def _mount_iteration(self, container: IterationContainer) -> None:
        """Mount an iteration container into the run area."""
        near_bottom = self._is_near_bottom()
        self.query_one("#run-area").mount(container)
        if near_bottom:
            self._scroll_to_end()

    def _mount_banner(self, renderable: RenderableType) -> None:
        """Mount the startup banner into the banner area."""
        self.query_one("#banner-area").mount(Static(renderable))

    def _mount_static(self, renderable: RenderableType) -> None:
        """Mount a Rich renderable as a Static widget."""
        near_bottom = self._is_near_bottom()
        self.query_one("#run-area").mount(Static(renderable))
        if near_bottom:
            self._scroll_to_end()


class ShowApp(App):
    """Read-only viewer for a completed run's TUI panels."""

    BINDINGS = [
        Binding("ctrl+o", "toggle_expand", "Expand/Collapse", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("enter", "open_focused_detail", "Detail", show=False),
    ]

    DEFAULT_CSS = PipelineApp.DEFAULT_CSS

    def __init__(self, manifest_records: list, run_title: str = "Run") -> None:
        super().__init__()
        self._manifest_records = manifest_records
        self._run_title = run_title

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="outer-container") as outer:
            outer.border_title = "Auto-Scientist"
            with Vertical(id="run-area") as run:
                run.border_title = self._run_title
                pass
        yield Footer()

    def on_mount(self) -> None:
        saved_theme = load_theme()
        if saved_theme in self.available_themes:
            self.theme = saved_theme
        self.title = f"Auto-Scientist — {self._run_title}"

        run_area = self.query_one("#run-area")
        for record in self._manifest_records:
            container = IterationContainer(iter_title=record.title)
            run_area.mount(container)
            for p in record.panels:
                panel = AgentPanel(name=p.name, model=p.model, style=p.style)
                container.mount(panel)
                container.add_panel(panel)
                panel.input_tokens = p.input_tokens
                panel.output_tokens = p.output_tokens
                panel.thinking_tokens = p.thinking_tokens
                panel.num_turns = p.num_turns
                for line in p.lines:
                    panel.all_lines.append(line)
                    panel._write_to_richlog(line)
                panel.complete(p.done_summary)
                panel._apply_complete_dom()
                panel._end_time = panel.start_time + p.elapsed_seconds
            container.set_result(record.result_text, record.result_style, record.summary)

        run_area.styles.border = ("round", "green")

    def action_toggle_expand(self) -> None:
        """Toggle expanded state on all panels and iteration containers."""
        panels = list(self.query(AgentPanel))
        containers = list(self.query(IterationContainer))
        if not panels and not containers:
            return
        collapsing = True
        if panels:
            with contextlib.suppress(NoMatches):
                collapsing = not panels[0].query_one(Collapsible).collapsed
        for panel in panels:
            try:
                c = panel.query_one(Collapsible)
            except NoMatches:
                continue
            object.__setattr__(c, "scroll_visible", lambda *a, **kw: None)
            c.collapsed = collapsing
            object.__delattr__(c, "scroll_visible")
        for container in containers:
            if not container._panels:
                continue
            if (collapsing and not container._is_collapsed) or (
                not collapsing and container._is_collapsed
            ):
                container.toggle_iteration()

    async def action_quit(self) -> None:
        self.exit()

    def action_open_focused_detail(self) -> None:
        focused = self.focused
        if focused is None:
            return
        widget: Widget | None = focused
        while widget is not None:
            if isinstance(widget, AgentPanel):
                self.push_screen(
                    AgentDetailScreen(
                        panel_name=widget.panel_name,
                        model=widget.model,
                        stats=widget._build_footer(),
                        lines=list(widget.all_lines),
                    )
                )
                return
            widget = widget.parent  # type: ignore[assignment]

    def watch_theme(self, theme_name: str) -> None:
        save_theme(theme_name)
