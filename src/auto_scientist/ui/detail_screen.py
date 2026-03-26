"""Modal screens: AgentDetailScreen, QuitConfirmScreen."""

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Label,
    RichLog,
    Static,
)


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
            yield Static(
                f"[bold]{self._panel_name}[/bold] ({self._model})"
                f" | {self._stats}"
            )
            yield RichLog(auto_scroll=False, markup=True, wrap=True)

    def on_mount(self) -> None:
        rich_log = self.query_one(RichLog)
        for line in self._lines:
            rich_log.write(Text(line))

    def action_dismiss(self) -> None:
        self.app.pop_screen()


class QuitConfirmScreen(ModalScreen[bool]):
    """Modal confirmation dialog for quitting while pipeline runs."""

    DEFAULT_CSS = """
    QuitConfirmScreen {
        align: center middle;
    }
    QuitConfirmScreen > Vertical {
        width: 50;
        height: auto;
        border: solid $error;
        background: $surface;
        padding: 1 2;
    }
    QuitConfirmScreen > Vertical > Label {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }
    QuitConfirmScreen > Vertical > Center {
        height: auto;
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
            with Center():
                yield Button("Yes", variant="error", id="yes-btn")
                yield Button("No", variant="primary", id="no-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes-btn")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)
