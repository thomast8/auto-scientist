"""Lightweight TUI launch form for experiment configuration.

A standalone Textual App with a form for data path, goal, preset, etc.
On Run, it validates inputs, builds an ExperimentConfig, and exits.
The calling CLI code then constructs the Orchestrator and PipelineApp.
"""

from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Input,
    Label,
    Rule,
    Select,
    Static,
    TextArea,
)

from auto_scientist.experiment_config import ExperimentConfig

PRESET_OPTIONS = [
    ("default (medium)", "default"),
    ("fast", "fast"),
    ("high", "high"),
    ("max", "max"),
]

# Sentinel for "no domain selected" in the domain picker
_CUSTOM = "__custom__"

DOMAIN_DIFFICULTY: dict[str, tuple[str, int]] = {
    "toy_function": ("easy", 0),
    "alien_minerals": ("medium", 1),
    "alloy_design": ("medium", 2),
    "water_treatment": ("hard", 3),
    "spo2": ("hard", 4),
}


def _discover_domains() -> list[tuple[str, str]]:
    """Find domains that have experiment.yaml files, return (label, yaml_path) pairs.

    Sorted by difficulty (easy to hard).
    """
    domains_dir = Path("domains")
    if not domains_dir.is_dir():
        return []
    results = []
    for candidate in domains_dir.iterdir():
        yaml_path = candidate / "experiment.yaml"
        if yaml_path.is_file() and candidate.name != "example_template":
            name = candidate.name.replace("_", " ").title()
            diff_info = DOMAIN_DIFFICULTY.get(candidate.name)
            if diff_info:
                label = f"{name} ({diff_info[0]})"
                sort_key = diff_info[1]
            else:
                label = name
                sort_key = 99
            results.append((sort_key, label, str(yaml_path)))
    results.sort(key=lambda x: x[0])
    return [(label, path) for _, label, path in results]


class BrowseScreen(ModalScreen[str | None]):
    """Modal file/directory browser using DirectoryTree."""

    CSS = """
    BrowseScreen {
        align: center middle;
    }
    #browse-container {
        width: 80%;
        height: 80%;
        border: round cyan;
        border-title-color: cyan;
        background: $surface;
        padding: 1;
    }
    #browse-tree {
        height: 1fr;
    }
    #browse-selected {
        height: auto;
        padding: 1 0 0 0;
        color: $text-muted;
    }
    #browse-buttons {
        height: auto;
        dock: bottom;
        padding-top: 1;
    }
    #browse-buttons Button {
        border: none;
        height: 1;
        min-height: 1;
        padding: 0 2;
        margin-right: 1;
        background: $surface-lighten-1;
        min-width: 0;
    }
    #browse-select-btn {
        color: cyan;
    }
    #browse-cancel-btn {
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("ctrl+q", "cancel", "Cancel"),
    ]

    def __init__(self, start_path: str = ".") -> None:
        super().__init__()
        self._start_path = start_path
        self._selected: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="browse-container") as container:
            container.border_title = "Browse"
            yield DirectoryTree(self._start_path, id="browse-tree")
            yield Static("", id="browse-selected")
            with Horizontal(id="browse-buttons"):
                yield Button("Select", variant="default", id="browse-select-btn")
                yield Button("Select dir", variant="default", id="browse-dir-btn")
                yield Button("Cancel", variant="default", id="browse-cancel-btn")

    @on(DirectoryTree.FileSelected)
    def _on_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self._selected = str(event.path)
        self.query_one("#browse-selected", Static).update(f"Selected: {event.path}")

    @on(DirectoryTree.DirectorySelected)
    def _on_dir_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self._selected = str(event.path)
        self.query_one("#browse-selected", Static).update(f"Selected: {event.path}")

    @on(Button.Pressed, "#browse-select-btn")
    def _on_select(self, event: Button.Pressed) -> None:
        if self._selected:
            self.dismiss(self._selected)

    @on(Button.Pressed, "#browse-dir-btn")
    def _on_select_dir(self, event: Button.Pressed) -> None:
        # Use the currently highlighted node's path (directory)
        tree = self.query_one("#browse-tree", DirectoryTree)
        if tree.cursor_node and tree.cursor_node.data:
            path = tree.cursor_node.data.path
            self.dismiss(str(path))

    @on(Button.Pressed, "#browse-cancel-btn")
    def _on_cancel(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class LaunchApp(App[ExperimentConfig | None]):
    """TUI form for configuring and launching an experiment."""

    CSS = """
    Screen {
        align: center middle;
    }
    #form-container {
        max-width: 100;
        border: round cyan;
        border-title-color: cyan;
        padding: 1 2;
    }
    .form-row {
        height: auto;
        margin-bottom: 1;
    }
    .form-label {
        width: 14;
        padding: 0 1 0 0;
        text-align: right;
        color: cyan;
        text-style: bold;
    }
    #goal-input {
        height: 4;
    }
    .short-input {
        width: 20;
    }
    #data-input, #output-dir-input {
        width: 1fr;
    }
    #domain-select {
        width: 1fr;
    }
    Input {
        border: none;
        height: 1;
        padding: 0 1;
    }
    Input:focus {
        border: none;
    }
    TextArea {
        border: none;
        padding: 0 1;
    }
    TextArea:focus {
        border: none;
    }
    Select {
        height: auto;
    }
    SelectCurrent {
        border: none;
        height: 1;
        padding: 0 1;
    }
    Select:focus SelectCurrent {
        border: none;
    }
    #browse-btn, #browse-output-btn {
        width: auto;
        min-width: 8;
        margin-left: 1;
        border: none;
        height: 1;
        min-height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }
    #browse-btn:hover, #browse-output-btn:hover {
        color: $text;
        text-style: bold;
    }
    #error-display {
        color: red;
        padding-left: 14;
        height: auto;
    }
    Rule {
        color: $text-muted;
        margin: 1 0 0 0;
    }
    """

    BINDINGS = [
        ("ctrl+r", "run", "Run"),
        ("ctrl+s", "save", "Save config"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        prefill: ExperimentConfig | None = None,
        save_path: Path | None = None,
    ) -> None:
        super().__init__()
        self._prefill = prefill
        self._save_path = save_path
        self.result_config: ExperimentConfig | None = None
        self._domain_options = _discover_domains()
        # Track the YAML path for domain-relative data resolution
        self._yaml_path: Path | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="form-container") as container:
            container.border_title = "Auto-Scientist"

            # Domain picker (only if domains are available)
            if self._domain_options:
                with Horizontal(classes="form-row"):
                    yield Label("Domain:", classes="form-label")
                    domain_choices = [
                        ("Custom", _CUSTOM),
                        *self._domain_options,
                    ]
                    yield Select(
                        domain_choices,
                        value=_CUSTOM,
                        allow_blank=False,
                        id="domain-select",
                    )

            yield Rule()

            # Data path with Browse button
            with Horizontal(classes="form-row"):
                yield Label("Data path:", classes="form-label")
                yield Input(
                    placeholder="path/to/dataset.csv or directory",
                    id="data-input",
                )
                yield Button("Browse", id="browse-btn")

            # Goal
            with Horizontal(classes="form-row"):
                yield Label("Goal:", classes="form-label")
                yield TextArea(id="goal-input")

            yield Rule()

            # Preset
            with Horizontal(classes="form-row"):
                yield Label("Preset:", classes="form-label")
                yield Select(
                    PRESET_OPTIONS,
                    value="default",
                    allow_blank=False,
                    id="preset-select",
                    classes="short-input",
                )

            # Iterations
            with Horizontal(classes="form-row"):
                yield Label("Iterations:", classes="form-label")
                yield Input(
                    value="20",
                    type="integer",
                    id="max-iterations-input",
                    classes="short-input",
                )

            # Debate
            with Horizontal(classes="form-row"):
                yield Label("Debate:", classes="form-label")
                yield Input(
                    value="1",
                    type="integer",
                    id="debate-rounds-input",
                    classes="short-input",
                )

            # Output dir with Browse button
            with Horizontal(classes="form-row"):
                yield Label("Output:", classes="form-label")
                yield Input(
                    value="experiments",
                    id="output-dir-input",
                )
                yield Button("Browse", id="browse-output-btn")

            # Error display
            yield Static("", id="error-display")

        yield Footer()

    def on_mount(self) -> None:
        if self._prefill:
            self._apply_config(self._prefill)

    def _apply_config(self, cfg: ExperimentConfig) -> None:
        """Fill the form from an ExperimentConfig."""
        self.query_one("#data-input", Input).value = cfg.data
        self.query_one("#goal-input", TextArea).text = cfg.goal
        self.query_one("#preset-select", Select).value = cfg.preset
        self.query_one("#max-iterations-input", Input).value = str(cfg.max_iterations)
        self.query_one("#debate-rounds-input", Input).value = str(cfg.debate_rounds)
        self.query_one("#output-dir-input", Input).value = cfg.output_dir

    @on(Button.Pressed, "#browse-btn")
    def _on_browse(self, event: Button.Pressed) -> None:
        """Open the file browser modal for data path."""
        self.push_screen(BrowseScreen("."), self._on_browse_data_result)

    def _on_browse_data_result(self, result: str | None) -> None:
        if result is not None:
            self.query_one("#data-input", Input).value = result

    @on(Button.Pressed, "#browse-output-btn")
    def _on_browse_output(self, event: Button.Pressed) -> None:
        """Open the file browser modal for output directory."""
        self.push_screen(BrowseScreen("."), self._on_browse_output_result)

    def _on_browse_output_result(self, result: str | None) -> None:
        if result is not None:
            self.query_one("#output-dir-input", Input).value = result

    @on(Select.Changed, "#domain-select")
    def _on_domain_changed(self, event: Select.Changed) -> None:
        """When a domain is selected, load its experiment.yaml and fill the form."""
        if event.value == _CUSTOM:
            self._yaml_path = None
            return

        yaml_path = Path(str(event.value))
        if not yaml_path.exists():
            return

        try:
            cfg = ExperimentConfig.from_yaml(yaml_path)
        except ValueError:
            return

        self._yaml_path = yaml_path
        # Resolve data path relative to the YAML file for display
        resolved_data = cfg.resolve_data_path(yaml_path.parent)
        cfg.data = str(resolved_data)
        # Set output dir to domain name
        cfg.output_dir = f"experiments/{yaml_path.parent.name}"
        self._apply_config(cfg)

    def _build_config(self) -> ExperimentConfig | None:
        """Collect form values and build an ExperimentConfig, or None on validation error."""
        from pydantic import ValidationError

        data = self.query_one("#data-input", Input).value.strip()
        goal = self.query_one("#goal-input", TextArea).text.strip()

        if not data or not goal:
            self.query_one("#error-display", Static).update(
                "Data path and goal are required."
            )
            return None

        self.query_one("#error-display", Static).update("")

        try:
            return ExperimentConfig(
                data=data,
                goal=goal,
                preset=str(self.query_one("#preset-select", Select).value),
                max_iterations=int(
                    self.query_one("#max-iterations-input", Input).value or "20"
                ),
                debate_rounds=int(
                    self.query_one("#debate-rounds-input", Input).value or "1"
                ),
                output_dir=self.query_one("#output-dir-input", Input).value
                or "experiments",
            )
        except (ValidationError, ValueError) as e:
            self.query_one("#error-display", Static).update(str(e))
            return None

    def action_run(self) -> None:
        config = self._build_config()
        if config is not None:
            self.result_config = config
            self.exit(config)

    def action_save(self) -> None:
        config = self._build_config()
        if config is None:
            return

        save_path = self._save_path or Path("experiment.yaml")
        try:
            config.to_yaml(save_path)
        except OSError as e:
            self.query_one("#error-display", Static).update(
                f"Failed to save config: {e}"
            )
            return

        self.query_one("#error-display", Static).update(
            f"Config saved to {save_path}"
        )

    def action_quit(self) -> None:
        self.exit(None)
