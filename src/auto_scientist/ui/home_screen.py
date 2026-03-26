"""HomeScreen: launcher with presets, config form, and past runs."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
)

from auto_scientist.experiment_store import ExperimentStore
from auto_scientist.model_config import BUILTIN_PRESETS
from auto_scientist.ui.config_form import ConfigForm


class HomeScreen(Screen):
    """Home screen with preset launcher, config form, and past runs table."""

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    DEFAULT_CSS = """
    HomeScreen {
        layout: vertical;
    }
    HomeScreen #preset-bar {
        height: auto;
        padding: 1 2;
        dock: top;
    }
    HomeScreen #preset-bar Label {
        padding: 0 1 0 0;
    }
    HomeScreen #preset-bar Button {
        margin: 0 1 0 0;
        min-width: 10;
    }
    HomeScreen #home-scroll {
        height: 1fr;
    }
    HomeScreen #past-runs-section {
        height: auto;
        min-height: 8;
        padding: 1 2;
    }
    HomeScreen #past-runs-section Label {
        text-style: bold;
        margin: 0 0 1 0;
    }
    HomeScreen DataTable {
        height: auto;
        max-height: 15;
    }
    """

    def __init__(self, store: ExperimentStore | None = None) -> None:
        super().__init__()
        self._store = store

    def compose(self) -> ComposeResult:
        yield Header()

        with VerticalScroll(id="home-scroll"):
            # Preset quick-launch buttons
            with Horizontal(id="preset-bar"):
                yield Label("Quick start:")
                for name in BUILTIN_PRESETS:
                    yield Button(
                        name.capitalize(),
                        id=f"preset-{name}",
                        variant="primary" if name == "default" else "default",
                    )

            # Config form
            ingestion_sources = self._load_ingestion_sources()
            yield ConfigForm(ingestion_sources=ingestion_sources)

            # Past runs table
            with Vertical(id="past-runs-section"):
                yield Label("Past Runs")
                table = DataTable(id="past-runs-table")
                table.cursor_type = "row"
                yield table

        yield Footer()

    def on_mount(self) -> None:
        self.app.title = "Auto-Scientist"
        self.app.sub_title = "Home"
        self._populate_past_runs()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id.startswith("preset-"):
            preset_name = btn_id.replace("preset-", "")
            try:
                form = self.query_one(ConfigForm)
                preset_select = form.query_one("#preset-select")
                preset_select.value = preset_name
            except Exception:
                pass

    def _load_ingestion_sources(self) -> list[tuple[str, str]]:
        """Load ingestion sources from the store."""
        if self._store is None:
            return []
        try:
            sources = self._store.get_ingestion_sources()
            return [(s.id, f"{s.goal[:40]} ({s.id})") for s in sources]
        except Exception:
            return []

    def _populate_past_runs(self) -> None:
        """Fill the past runs DataTable from the store."""
        try:
            table = self.query_one("#past-runs-table", DataTable)
        except Exception:
            return

        table.add_columns("Goal", "Preset", "Iteration", "Status", "Directory")

        if self._store is None:
            return

        try:
            experiments = self._store.list_experiments()
        except Exception:
            return

        for exp in experiments:
            goal_display = exp.goal[:40] + "..." if len(exp.goal) > 40 else exp.goal
            table.add_row(
                goal_display,
                exp.preset_name or "custom",
                str(exp.iteration),
                exp.status,
                exp.id,
                key=exp.id,
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle clicking a past run row."""
        if event.row_key is None:
            return
        experiment_id = str(event.row_key.value)
        if self._store is None:
            return
        detail = self._store.get_experiment(experiment_id)
        if detail is None:
            return

        # Import here to avoid circular
        from auto_scientist.ui.experiment_screen import ExperimentScreen

        screen = ExperimentScreen(
            read_only=True,
            experiment_label=f"{detail.state.goal[:30]}",
        )
        self.app.push_screen(screen)

    def action_quit(self) -> None:
        self.app.exit()
