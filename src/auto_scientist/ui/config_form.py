"""ConfigForm: preset-aware experiment configuration form widget."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    TextArea,
)

from auto_scientist.model_config import (
    BUILTIN_PRESETS,
    AgentModelConfig,
    ModelConfig,
    ReasoningConfig,
)

# Available models per provider
MODELS_BY_PROVIDER: dict[str, list[str]] = {
    "anthropic": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ],
    "openai": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "o4-mini",
        "o3",
    ],
    "google": [
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
}

PROVIDERS = ["anthropic", "openai", "google"]

REASONING_LEVELS = ["default", "off", "minimal", "low", "medium", "high", "max"]

# Agent roles that appear in the per-agent model table
AGENT_ROLES = ["analyst", "scientist", "coder", "ingestor", "report"]

PRESET_NAMES = list(BUILTIN_PRESETS.keys())


class ConfigForm(Widget):
    """Experiment configuration form with preset awareness.

    Selecting a preset fills all fields. Modifying any field marks the
    preset as "(modified)".
    """

    DEFAULT_CSS = """
    ConfigForm {
        height: auto;
        padding: 1 2;
    }
    ConfigForm .form-row {
        height: auto;
        margin: 0 0 1 0;
    }
    ConfigForm .form-row Label {
        width: 16;
        padding: 0 1 0 0;
    }
    ConfigForm Input {
        width: 1fr;
    }
    ConfigForm TextArea {
        height: 4;
        width: 1fr;
    }
    ConfigForm Select {
        width: 1fr;
    }
    ConfigForm .launch-btn {
        margin: 1 0 0 0;
        width: 100%;
    }
    ConfigForm #data-source-radio {
        height: auto;
        layout: horizontal;
    }
    """

    class LaunchRequested(Message):
        """Posted when the Launch button is pressed with valid config."""

        def __init__(
            self,
            sender: Widget,
            data_path: str,
            goal: str,
            max_iterations: int,
            debate_rounds: int,
            model_config: ModelConfig,
            ingestion_source: str | None,
        ) -> None:
            super().__init__()
            self.data_path = data_path
            self.goal = goal
            self.max_iterations = max_iterations
            self.debate_rounds = debate_rounds
            self.model_config = model_config
            self.ingestion_source = ingestion_source

    _selected_preset: reactive[str] = reactive("default")
    _modified: reactive[bool] = reactive(False)

    def __init__(self, ingestion_sources: list[tuple[str, str]] | None = None) -> None:
        super().__init__()
        # (id, label) pairs for the "Reuse from" dropdown
        self._ingestion_sources = ingestion_sources or []
        self._applying_preset = False  # Guard to suppress modified flag during preset fill

    def compose(self) -> ComposeResult:
        # Data source selection
        with Horizontal(classes="form-row"):
            yield Label("Data source")
            with RadioSet(id="data-source-radio"):
                yield RadioButton("Raw data path", value=True)
                yield RadioButton("Reuse from past run")

        with Horizontal(classes="form-row", id="raw-data-row"):
            yield Label("Data path")
            yield Input(placeholder="/path/to/data", id="data-path-input")

        if self._ingestion_sources:
            with Horizontal(classes="form-row", id="reuse-row"):
                yield Label("Reuse from")
                yield Select(
                    [(label, sid) for sid, label in self._ingestion_sources],
                    id="reuse-select",
                    prompt="Select experiment...",
                )

        with Horizontal(classes="form-row"):
            yield Label("Goal")
            yield TextArea(id="goal-input")

        with Horizontal(classes="form-row"):
            yield Label("Preset")
            yield Select(
                [(name.capitalize(), name) for name in PRESET_NAMES],
                id="preset-select",
                value="default",
            )

        with Horizontal(classes="form-row"):
            yield Label("Max iterations")
            yield Input(value="10", id="max-iter-input", type="integer")

        with Horizontal(classes="form-row"):
            yield Label("Debate rounds")
            yield Input(value="2", id="debate-rounds-input", type="integer")

        # Per-agent model overrides (collapsible)
        with Collapsible(title="Per-agent models", collapsed=True):
            for role in AGENT_ROLES:
                with Horizontal(classes="form-row"):
                    yield Label(role.capitalize())
                    yield Select(
                        [(p, p) for p in PROVIDERS],
                        id=f"{role}-provider",
                        value="anthropic",
                    )
                    yield Select(
                        [(m, m) for m in MODELS_BY_PROVIDER["anthropic"]],
                        id=f"{role}-model",
                        value="claude-sonnet-4-6",
                    )
                    yield Select(
                        [(r, r) for r in REASONING_LEVELS],
                        id=f"{role}-reasoning",
                        value="default",
                    )

        yield Button("Launch Experiment", variant="success", classes="launch-btn")

    def on_mount(self) -> None:
        self._apply_preset("default")

    def on_select_changed(self, event: Select.Changed) -> None:
        if self._applying_preset:
            return
        if event.select.id == "preset-select" and event.value != Select.BLANK:
            self._apply_preset(str(event.value))
        elif event.select.id and event.select.id.endswith("-provider"):
            role = event.select.id.replace("-provider", "")
            provider = str(event.value)
            self._update_model_options(role, provider)
            self._modified = True
        else:
            self._modified = True

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._applying_preset:
            return
        self._modified = True

    def _apply_preset(self, name: str) -> None:
        """Fill form fields from a preset."""
        if name not in BUILTIN_PRESETS:
            return
        self._applying_preset = True
        self._selected_preset = name
        mc = ModelConfig.builtin_preset(name)
        for role in AGENT_ROLES:
            resolved = mc.resolve(role)
            self._set_agent_fields(role, resolved)
        self._applying_preset = False
        # Reset after filling: use call_after_refresh to clear after any
        # queued change events have fired
        self.call_after_refresh(self._clear_modified)

    def _clear_modified(self) -> None:
        """Reset the modified flag (called after preset application settles)."""
        self._modified = False

    def _set_agent_fields(self, role: str, config: AgentModelConfig) -> None:
        """Set provider/model/reasoning fields for a role."""
        try:
            self.query_one(f"#{role}-provider", Select).value = config.provider
            self._update_model_options(role, config.provider)
            self.query_one(f"#{role}-model", Select).value = config.model
            self.query_one(f"#{role}-reasoning", Select).value = config.reasoning.level
        except Exception:
            pass

    def _update_model_options(self, role: str, provider: str) -> None:
        """Update the model dropdown options for a given provider."""
        models = MODELS_BY_PROVIDER.get(provider, [])
        try:
            model_select = self.query_one(f"#{role}-model", Select)
            model_select.set_options([(m, m) for m in models])
            if models:
                model_select.value = models[0]
        except Exception:
            pass

    def get_model_config(self) -> ModelConfig:
        """Build a ModelConfig from current form values."""
        mc = ModelConfig.builtin_preset(self._selected_preset)

        # Override per-agent settings
        for role in AGENT_ROLES:
            try:
                provider = str(self.query_one(f"#{role}-provider", Select).value)
                model = str(self.query_one(f"#{role}-model", Select).value)
                reasoning_level = str(self.query_one(f"#{role}-reasoning", Select).value)
            except Exception:
                continue

            agent_config = AgentModelConfig(
                provider=provider,
                model=model,
                reasoning=ReasoningConfig(level=reasoning_level),
            )
            setattr(mc, role, agent_config)

        if self._modified:
            mc.preset_name = f"{self._selected_preset} (modified)"
        else:
            mc.preset_name = self._selected_preset

        return mc

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if "launch-btn" not in (event.button.classes or set()):
            return

        # Gather form values
        try:
            data_path = self.query_one("#data-path-input", Input).value
            goal = self.query_one("#goal-input", TextArea).text
            max_iter_str = self.query_one("#max-iter-input", Input).value
            debate_str = self.query_one("#debate-rounds-input", Input).value
        except Exception:
            return

        max_iterations = int(max_iter_str) if max_iter_str.isdigit() else 10
        debate_rounds = int(debate_str) if debate_str.isdigit() else 2

        # Determine data source
        ingestion_source = None
        try:
            radio_set = self.query_one("#data-source-radio", RadioSet)
            if radio_set.pressed_index == 1:
                # Reuse from past run
                reuse_select = self.query_one("#reuse-select", Select)
                ingestion_source = str(reuse_select.value)
                data_path = ""  # Will be loaded from source
        except Exception:
            pass

        mc = self.get_model_config()

        self.post_message(self.LaunchRequested(
            self,
            data_path=data_path,
            goal=goal,
            max_iterations=max_iterations,
            debate_rounds=debate_rounds,
            model_config=mc,
            ingestion_source=ingestion_source,
        ))
