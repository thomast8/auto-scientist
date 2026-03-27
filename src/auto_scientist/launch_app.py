"""Lightweight TUI launch form for experiment configuration.

A standalone Textual App with a form for data path, goal, preset, etc.
On Run, it validates inputs, builds an ExperimentConfig, and exits.
The calling CLI code then constructs the Orchestrator and PipelineApp.
"""

from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TextArea,
)

from auto_scientist.experiment_config import ExperimentConfig
from auto_scientist.model_config import BUILTIN_PRESETS

PRESET_OPTIONS = [(name, name) for name in BUILTIN_PRESETS if name != "medium"]


class LaunchApp(App[ExperimentConfig | None]):
    """TUI form for configuring and launching an experiment."""

    CSS = """
    Screen {
        layout: vertical;
        overflow-y: auto;
    }
    #form-container {
        padding: 1 2;
    }
    .form-row {
        height: auto;
        margin-bottom: 1;
    }
    .form-label {
        width: 18;
        padding: 1 1 0 0;
        text-align: right;
    }
    #goal-input {
        height: 6;
    }
    .short-input {
        width: 20;
    }
    #data-input, #output-dir-input {
        width: 1fr;
    }
    .checkbox-row {
        height: auto;
        margin-bottom: 1;
        padding-left: 19;
    }
    .button-row {
        height: auto;
        margin-top: 1;
        padding-left: 19;
    }
    #error-display {
        color: red;
        padding-left: 19;
        height: auto;
    }
    Button {
        margin-right: 2;
    }
    """

    BINDINGS = [
        ("escape", "quit", "Quit"),
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

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="form-container"):
            # Data path
            with Horizontal(classes="form-row"):
                yield Label("Data path:", classes="form-label")
                yield Input(
                    placeholder="path/to/dataset.csv or directory",
                    id="data-input",
                )

            # Goal
            with Horizontal(classes="form-row"):
                yield Label("Goal:", classes="form-label")
                yield TextArea(id="goal-input")

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

            # Max iterations
            with Horizontal(classes="form-row"):
                yield Label("Max iterations:", classes="form-label")
                yield Input(
                    value="20",
                    type="integer",
                    id="max-iterations-input",
                    classes="short-input",
                )

            # Debate rounds
            with Horizontal(classes="form-row"):
                yield Label("Debate rounds:", classes="form-label")
                yield Input(
                    value="1",
                    type="integer",
                    id="debate-rounds-input",
                    classes="short-input",
                )

            # Output dir
            with Horizontal(classes="form-row"):
                yield Label("Output dir:", classes="form-label")
                yield Input(
                    value="experiments",
                    id="output-dir-input",
                )

            # Checkboxes
            with Horizontal(classes="checkbox-row"):
                yield Checkbox("Interactive", id="interactive-checkbox")
                yield Checkbox("Verbose", id="verbose-checkbox")

            # Error display
            yield Static("", id="error-display")

            # Buttons
            with Horizontal(classes="button-row"):
                yield Button("Run", variant="primary", id="run-button")
                yield Button("Save config", variant="default", id="save-button")

        yield Footer()

    def on_mount(self) -> None:
        if self._prefill:
            self.query_one("#data-input", Input).value = self._prefill.data
            self.query_one("#goal-input", TextArea).text = self._prefill.goal
            self.query_one("#preset-select", Select).value = self._prefill.preset
            self.query_one("#max-iterations-input", Input).value = str(
                self._prefill.max_iterations
            )
            self.query_one("#debate-rounds-input", Input).value = str(
                self._prefill.debate_rounds
            )
            self.query_one("#output-dir-input", Input).value = self._prefill.output_dir
            self.query_one("#interactive-checkbox", Checkbox).value = (
                self._prefill.interactive
            )
            self.query_one("#verbose-checkbox", Checkbox).value = self._prefill.verbose

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
                interactive=self.query_one("#interactive-checkbox", Checkbox).value,
                verbose=self.query_one("#verbose-checkbox", Checkbox).value,
            )
        except (ValidationError, ValueError) as e:
            self.query_one("#error-display", Static).update(str(e))
            return None

    @on(Button.Pressed, "#run-button")
    def _on_run(self, event: Button.Pressed) -> None:
        config = self._build_config()
        if config is not None:
            self.result_config = config
            self.exit(config)

    @on(Button.Pressed, "#save-button")
    def _on_save(self, event: Button.Pressed) -> None:
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
