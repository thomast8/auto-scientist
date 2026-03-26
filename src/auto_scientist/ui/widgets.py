"""Core Textual widgets: AgentPanel, MetricsBar, IterationContainer."""

import threading
import time

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import (
    Collapsible,
    RichLog,
    Static,
)
from textual.widgets._collapsible import CollapsibleTitle

from auto_scientist.ui.styles import (
    AGENT_DESCRIPTIONS,
    PHASE_STYLES,
    _format_elapsed,
    _score_style,
)

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
    }
    AgentPanel:hover {
        background: $surface;
    }
    AgentPanel RichLog {
        height: auto;
        max-height: 20;
    }
    AgentPanel .agent-description {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }
    AgentPanel CollapsibleTitle {
        width: 100%;
    }
    AgentPanel Collapsible:disabled CollapsibleTitle {
        opacity: 1;
    }
    """

    def __init__(self, name: str, model: str, style: str = "cyan", description: str = "") -> None:
        super().__init__()
        self._panel_name = name
        self.model = model
        self.panel_style = style
        self.all_lines: list[str] = []
        self.start_time = time.monotonic()
        self.input_tokens = 0
        self.output_tokens = 0
        self.num_turns = 0
        self.done = False
        self.done_summary = ""
        self.error_msg = ""
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
        self._apply_title_color()

    # Rich markup color names that need translation for Textual CSS styles.color
    _RICH_TO_TEXTUAL_COLOR: dict[str, str] = {
        "bright_red": "ansi_bright_red",
        "magenta1": "ansi_bright_magenta",
    }

    def _apply_title_color(self) -> None:
        """Force the CollapsibleTitle foreground color to match the agent style.

        Without this, some Textual themes override Rich markup colors on the
        CollapsibleTitle widget, making completed panels appear grey.
        """
        try:
            title_widget = self.query_one(CollapsibleTitle)
        except NoMatches:
            return
        css_color = self._RICH_TO_TEXTUAL_COLOR.get(self.panel_style, self.panel_style)
        title_widget.styles.color = css_color

    def _tick(self) -> None:
        """Update the Collapsible title with elapsed time and animate description dots."""
        if self.done and hasattr(self, "_refresh_timer"):
            self._refresh_timer.stop()
            return
        self._update_title()
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
        """Build the Collapsible title string."""
        footer = self._build_footer()
        return f"[{self.panel_style}]{self._panel_name} ({self.model}) | {footer}[/]"

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
        cleaned = " ".join(text.split())
        self.all_lines.append(cleaned)
        try:
            app = self.app
        except Exception:
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
            near_bottom = app._is_near_bottom()
        except Exception:
            near_bottom = False
        rich_log.write(Text(text), expand=True)
        if near_bottom:
            app._scroll_to_end()

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
            self.done_summary = done_summary
        elif self.all_lines:
            self.done_summary = self.all_lines[-1]
        self._end_time = time.monotonic()

    def _apply_complete_dom(self) -> None:
        """Apply completion state to DOM (must be called from UI thread)."""
        for desc in self.query(".agent-description"):
            desc.remove()
        try:
            collapsible = self.query_one(Collapsible)
        except NoMatches:
            return
        summary = self.done_summary
        if summary.startswith("[done] "):
            summary = summary[len("[done] "):]
        collapsible.title = (
            f"[{self.panel_style}]{self._panel_name}: {summary} | {self._build_footer()}[/]"
        )
        # Suppress Textual's built-in scroll-on-collapse (Collapsible._watch_collapsed
        # calls self.call_after_refresh(self.scroll_visible) unconditionally)
        collapsible.scroll_visible = lambda *a, **kw: None
        collapsible.collapsed = True
        del collapsible.scroll_visible  # Restore inherited method for user-initiated toggles
        self._apply_title_color()
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
        self._end_time = time.monotonic()
        try:
            app = self.app
        except Exception:
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
        footer = self._build_footer()
        collapsible.title = (
            f"[{self.panel_style}]{self._panel_name}:[/]"
            f" [red][error] {msg}[/red] | {footer}"
        )
        rich_log.write(Text(f"[error] {msg}", style="red"))
        self._apply_title_color()

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Set token usage metadata."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def set_stats(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        num_turns: int = 0,
    ) -> None:
        """Set rich usage stats from SDK ResultMessage or direct API calls."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
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
            parts.append(
                f"{self.input_tokens:,} in / {self.output_tokens:,} out"
            )
        if self.num_turns:
            parts.append(
                f"{self.num_turns} "
                f"{'turn' if self.num_turns == 1 else 'turns'}"
            )
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
        self.best_version = ""
        self.best_score: int | None = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
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
        best_version: str | None = None,
        best_score: int | None = None,
    ) -> None:
        """Update metrics bar fields. Only non-None values are changed."""
        if iteration is not None:
            self.iteration = iteration
        if phase is not None:
            self.phase = phase
        if best_version is not None:
            self.best_version = best_version
        if best_score is not None:
            self.best_score = best_score
        self.refresh()

    def finish(self) -> None:
        """Freeze the elapsed timer at the current value."""
        self.finished = True
        self._end_time = time.monotonic()

    def add_agent_stats(self, panel: "AgentPanel") -> None:
        """Accumulate a completed agent's stats into the running totals."""
        self.total_input_tokens += panel.input_tokens
        self.total_output_tokens += panel.output_tokens
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
            tokens = (
                f"{self.total_input_tokens:,} in"
                f" / {self.total_output_tokens:,} out"
            )
            line.append(f" | {tokens}", style="dim")
        if self.total_turns:
            label = "turn" if self.total_turns == 1 else "turns"
            line.append(f" | {self.total_turns} {label}", style="dim")

        if self.best_score is not None:
            try:
                best_iter = int(self.best_version.lstrip("v")) + 1
            except (ValueError, AttributeError):
                best_iter = "?"
            style = _score_style(self.best_score)
            line.append(
                f"  best: iter {best_iter} ({self.best_score})",
                style=style,
            )

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


class IterationContainer(Vertical):
    """Bordered container grouping panels for one iteration."""

    DEFAULT_CSS = """
    IterationContainer {
        height: auto;
        border: solid $accent;
        transition: border 300ms in_out_cubic;
    }
    """

    def __init__(self, iter_title: str) -> None:
        super().__init__()
        self.border_title = iter_title

    def set_result(self, text: str, style: str) -> None:
        """Set the iteration result as border subtitle."""
        self.border_subtitle = text
        valid = {
            "red", "green", "yellow", "blue", "cyan", "magenta", "white",
        }
        if style in valid:
            self.styles.border = ("solid", style)
