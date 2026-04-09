"""Textual widgets and shared constants for the auto-scientist dashboard.

Provides:
- AgentPanel: Collapsible + RichLog widget for each agent phase
- MetricsBar: Persistent metrics display (sparkline, tokens, phase)
- IterationToggle: Clickable toggle for expanding/collapsing iteration panels
- IterationContainer: Bordered container grouping panels per iteration
"""

from __future__ import annotations

import re
import threading
import time
from typing import Literal

from rich.console import Console
from rich.markup import escape as rich_escape
from rich.text import Text
from textual._context import NoActiveAppError
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.message import Message
from textual.widget import Widget
from textual.widgets import (
    Collapsible,
    RichLog,
    Static,
)
from textual.widgets._collapsible import CollapsibleTitle

from auto_scientist.latex_to_unicode import latex_to_unicode

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
    "Stop Revision": "Deciding whether to uphold or withdraw the stop proposal...",
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
        self.max_iterations: int | None = None
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
        max_iterations: int | None = None,
    ) -> None:
        """Update metrics bar fields. Only non-None values are changed."""
        if iteration is not None:
            self.iteration = iteration
        if phase is not None:
            self.phase = phase
        if max_iterations is not None:
            self.max_iterations = max_iterations
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

    def carry_over(self, panel: AgentPanel) -> None:
        """Accumulate a restored panel's stats AND shift start_time backwards.

        Used on resume so the header shows totals and wall clock that include
        prior sessions' work. Shifting ``start_time`` by ``panel.elapsed``
        leaves the existing render math (``now - start_time``) producing
        ``prior_work_time + current_session_wall_time`` with no other
        changes.
        """
        self.add_agent_stats(panel)
        self.start_time -= panel.elapsed

    def render(self) -> Text:
        end = self._end_time if self._end_time else time.monotonic()
        elapsed = _format_elapsed(end - self.start_time)
        phase_style = PHASE_STYLES.get(self.phase, "cyan")

        line = Text()
        iter_label = (
            f"{self.iteration}/{self.max_iterations}"
            if self.max_iterations
            else str(self.iteration)
        )
        line.append(f" Iteration {iter_label}", style="bold")
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
