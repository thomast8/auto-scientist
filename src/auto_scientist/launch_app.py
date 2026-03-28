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

from auto_scientist.experiment_config import ExperimentConfig, ExperimentModelsConfig
from auto_scientist.model_config import AgentModelConfig
from auto_scientist.preferences import load_theme, save_theme

PRESET_OPTIONS = [
    ("turbo", "turbo"),
    ("fast", "fast"),
    ("default (medium)", "default"),
    ("high", "high"),
    ("max", "max"),
]

PROVIDER_OPTIONS = [
    ("anthropic", "anthropic"),
    ("openai", "openai"),
    ("google", "google"),
]

REASONING_OPTIONS = [
    ("off", "off"),
    ("minimal", "minimal"),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
    ("max", "max"),
]

# Agents that can be overridden, in display order
_AGENT_FIELDS = ["ingestor", "analyst", "scientist", "coder", "report", "summarizer"]

# SDK agents are locked to anthropic provider
_SDK_AGENTS = {"ingestor", "analyst", "scientist", "coder", "report"}

_NUM_CRITIC_SLOTS = 4

# Known models per provider (most capable first)
MODELS_BY_PROVIDER: dict[str, list[tuple[str, str]]] = {
    "anthropic": [
        ("claude-opus-4-6", "claude-opus-4-6"),
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
    ],
    "openai": [
        ("gpt-5.4", "gpt-5.4"),
        ("gpt-5.4-mini", "gpt-5.4-mini"),
        ("gpt-5.4-nano", "gpt-5.4-nano"),
        ("gpt-4.1", "gpt-4.1"),
        ("o3", "o3"),
        ("o4-mini", "o4-mini"),
    ],
    "google": [
        ("gemini-3.1-pro-preview", "gemini-3.1-pro-preview"),
        ("gemini-3-flash-preview", "gemini-3-flash-preview"),
        ("gemini-3.1-flash-lite-preview", "gemini-3.1-flash-lite-preview"),
        ("gemini-2.5-pro", "gemini-2.5-pro"),
        ("gemini-2.5-flash", "gemini-2.5-flash"),
    ],
}

# Flat list of all models across all providers
ALL_MODELS: list[tuple[str, str]] = [
    item for models in MODELS_BY_PROVIDER.values() for item in models
]

# Sentinel for "no domain selected" in the domain picker
_CUSTOM = "__custom__"

DOMAIN_DIFFICULTY: dict[str, tuple[str, int, str]] = {
    "toy_function": ("easy", 0, "fast"),
    "alien_minerals": ("medium", 1, "default"),
    "alloy_design": ("medium", 2, "default"),
    "water_treatment": ("hard", 3, "high"),
    "spo2": ("hard", 4, "high"),
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


class _PlainDirectoryTree(DirectoryTree):
    """DirectoryTree with plain text icons instead of emoji."""

    ICON_NODE = "+ "
    ICON_NODE_EXPANDED = "- "
    ICON_FILE = "  "

    COMPONENT_CLASSES = {
        "tree--cursor",
        "tree--guides",
        "tree--guides-hover",
        "tree--guides-selected",
        "tree--highlight",
        "tree--highlight-line",
    }

    DEFAULT_CSS = """
    _PlainDirectoryTree {
        color: $text;
    }
    _PlainDirectoryTree > .tree--guides {
        color: grey;
    }
    _PlainDirectoryTree > .tree--guides-hover {
        color: grey;
    }
    _PlainDirectoryTree > .tree--guides-selected {
        color: grey;
    }
    _PlainDirectoryTree > .tree--highlight-line {
        color: grey;
    }
    """


class BrowseScreen(ModalScreen[str | None]):
    """Modal file/directory browser using DirectoryTree."""

    CSS = """
    BrowseScreen {
        align: center middle;
    }
    #browse-container {
        width: 80%;
        height: 80%;
        border: round grey;
        background: $surface;
        padding: 1;
    }
    #browse-tree {
        height: 1fr;
        scrollbar-color: grey;
    }
    _PlainDirectoryTree {
        color: $text;
    }
    _PlainDirectoryTree > .tree--guides {
        color: grey;
    }
    _PlainDirectoryTree > .tree--guides-hover {
        color: grey;
    }
    _PlainDirectoryTree > .tree--guides-selected {
        color: grey;
    }
    _PlainDirectoryTree > .tree--cursor {
        background: $surface-lighten-1;
        color: $text;
        text-style: bold;
    }
    _PlainDirectoryTree:focus > .tree--cursor {
        background: $surface-lighten-2;
    }
    _PlainDirectoryTree > .tree--highlight {
        background: $surface-lighten-1;
    }
    _PlainDirectoryTree > .tree--highlight-line {
        color: grey;
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
        color: $text;
        text-style: bold;
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
            yield _PlainDirectoryTree(self._start_path, id="browse-tree")
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
        align-horizontal: center;
    }
    #outer {
        max-width: 100;
        height: 1fr;
        scrollbar-color: grey;
    }
    #banner-container {
        height: auto;
        border: solid grey;
        padding: 1 2;
    }
    #banner {
        height: auto;
        color: $text-muted;
    }
    #form-container {
        height: auto;
        border: solid grey;
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
    .model-row {
        height: auto;
        margin-bottom: 0;
    }
    .model-label {
        width: 14;
        padding: 0 1 0 0;
        text-align: right;
    }
    .model-provider {
        width: 16;
    }
    .model-name {
        width: 36;
    }
    .model-reasoning {
        width: 14;
    }
    SelectOverlay {
        width: auto;
        min-width: 16;
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

    BANNER = (
        "Autonomous scientific investigation framework.\n"
        "\n"
        "Given a dataset and a research goal, the system runs an iterative\n"
        "loop of analysis, hypothesis generation, debate, and experimentation\n"
        "until it converges on an answer. Each iteration produces a self-\n"
        "contained Python script that runs the experiment and logs results.\n"
        "\n"
        "Select a built-in domain below or choose Custom to provide your\n"
        "own dataset and goal. Adjust pipeline settings as needed."
    )

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="outer"):
            with Vertical(id="banner-container") as banner:
                banner.border_title = "Auto-Scientist"
                yield Static(self.BANNER, id="banner", markup=True)

            with Vertical(id="form-container"):
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

                yield Rule()

                # Per-agent model overrides header
                with Horizontal(classes="model-row"):
                    yield Label("", classes="model-label")
                    yield Static("[dim]Provider[/dim]", classes="model-provider", markup=True)
                    yield Static("[dim]Model[/dim]", classes="model-name", markup=True)
                    yield Static("[dim]Reasoning[/dim]", classes="model-reasoning", markup=True)

                for agent in _AGENT_FIELDS:
                    display = agent.title()
                    sdk_locked = agent in _SDK_AGENTS
                    with Horizontal(classes="model-row"):
                        yield Label(f"{display}:", classes="model-label")
                        provider_sel = Select(
                            PROVIDER_OPTIONS,
                            value="anthropic",
                            allow_blank=not sdk_locked,
                            id=f"model-{agent}-provider",
                            classes="model-provider",
                            disabled=sdk_locked,
                        )
                        yield provider_sel
                        initial_models = (
                            MODELS_BY_PROVIDER["anthropic"] if sdk_locked else ALL_MODELS
                        )
                        yield Select(
                            initial_models,
                            allow_blank=True,
                            id=f"model-{agent}-name",
                            classes="model-name",
                        )
                        yield Select(
                            REASONING_OPTIONS,
                            allow_blank=True,
                            id=f"model-{agent}-reasoning",
                            classes="model-reasoning",
                        )

                # Critic rows (any provider)
                for i in range(_NUM_CRITIC_SLOTS):
                    label = f"Critic {i + 1}:"
                    with Horizontal(classes="model-row"):
                        yield Label(label, classes="model-label")
                        yield Select(
                            PROVIDER_OPTIONS,
                            allow_blank=True,
                            id=f"model-critic-{i}-provider",
                            classes="model-provider",
                        )
                        yield Select(
                            ALL_MODELS,
                            allow_blank=True,
                            id=f"model-critic-{i}-name",
                            classes="model-name",
                        )
                        yield Select(
                            REASONING_OPTIONS,
                            allow_blank=True,
                            id=f"model-critic-{i}-reasoning",
                            classes="model-reasoning",
                        )

                # Error display
                yield Static("", id="error-display", markup=False)

        yield Footer()

    def on_mount(self) -> None:
        saved_theme = load_theme()
        if saved_theme in self.available_themes:
            self.theme = saved_theme
        if self._prefill:
            self._apply_config(self._prefill)
        else:
            # Populate model fields from the default preset
            default_cfg = ExperimentConfig(data=".", goal=".")
            self._apply_model_defaults(default_cfg)

    def watch_theme(self, theme_name: str) -> None:
        """Persist every theme change made through the launcher UI."""
        save_theme(theme_name)

    def _apply_model_defaults(self, cfg: ExperimentConfig) -> None:
        """Populate only the model/reasoning fields from the resolved preset."""
        from auto_scientist.model_config import ModelConfig

        mc = ModelConfig.from_experiment_config(cfg)

        for agent in _AGENT_FIELDS:
            agent_cfg = mc.resolve(agent) if agent != "summarizer" else mc.summarizer
            if agent_cfg is None:
                continue
            self._set_agent_model(f"model-{agent}", agent_cfg)

        for i in range(_NUM_CRITIC_SLOTS):
            if i < len(mc.critics):
                self._set_agent_model(f"model-critic-{i}", mc.critics[i])

    def _set_agent_model(self, prefix: str, cfg: AgentModelConfig) -> None:
        """Set provider, model, and reasoning for one agent row."""
        provider_sel = self.query_one(f"#{prefix}-provider", Select)
        provider_sel.value = cfg.provider

        model_sel = self.query_one(f"#{prefix}-name", Select)
        options = MODELS_BY_PROVIDER.get(cfg.provider, [])
        # Only reset options if the provider changed
        current_options = [(label, val) for label, val in options]
        model_sel.set_options(current_options)
        # Use call_after_refresh to ensure options are loaded before setting value
        self.call_after_refresh(setattr, model_sel, "value", cfg.model)

        reasoning_sel = self.query_one(f"#{prefix}-reasoning", Select)
        reasoning_sel.value = cfg.reasoning.level

    def _apply_config(self, cfg: ExperimentConfig) -> None:
        """Fill the form from an ExperimentConfig."""
        from auto_scientist.model_config import ModelConfig

        self.query_one("#data-input", Input).value = cfg.data
        self.query_one("#goal-input", TextArea).text = cfg.goal
        self.query_one("#preset-select", Select).value = cfg.preset
        self.query_one("#max-iterations-input", Input).value = str(cfg.max_iterations)
        self.query_one("#debate-rounds-input", Input).value = str(cfg.debate_rounds)
        self.query_one("#output-dir-input", Input).value = cfg.output_dir

        # Resolve the full model config (preset + any overrides)
        mc = ModelConfig.from_experiment_config(cfg)

        # Fill per-agent model fields from resolved config
        for agent in _AGENT_FIELDS:
            agent_cfg = mc.resolve(agent) if agent != "summarizer" else mc.summarizer
            if agent_cfg is None:
                continue
            self._set_agent_model(f"model-{agent}", agent_cfg)

        # Fill critic fields from resolved config
        for i in range(_NUM_CRITIC_SLOTS):
            if i < len(mc.critics):
                self._set_agent_model(f"model-critic-{i}", mc.critics[i])

    @on(Button.Pressed, "#browse-btn")
    def _on_browse(self, event: Button.Pressed) -> None:
        """Open the file browser modal for data path."""
        self.push_screen(BrowseScreen("."), self._on_browse_data_result)

    def _on_browse_data_result(self, result: str | None) -> None:
        if result is not None:
            self.query_one("#data-input", Input).value = result

    @on(Select.Changed, "#preset-select")
    def _on_preset_changed(self, event: Select.Changed) -> None:
        """When the preset changes, refresh model fields to show preset defaults."""
        if not isinstance(event.value, str):
            return
        cfg = ExperimentConfig(data=".", goal=".", preset=event.value)
        self._apply_model_defaults(cfg)

    @on(Button.Pressed, "#browse-output-btn")
    def _on_browse_output(self, event: Button.Pressed) -> None:
        """Open the file browser modal for output directory."""
        self.push_screen(BrowseScreen("."), self._on_browse_output_result)

    def _on_browse_output_result(self, result: str | None) -> None:
        if result is not None:
            self.query_one("#output-dir-input", Input).value = result

    @on(Select.Changed, "#domain-select")
    def _on_domain_changed(self, event: Select.Changed) -> None:
        """When a domain is selected, load its experiment.yaml and fill the form.

        Also sets the preset to a sensible default based on domain difficulty.
        """
        if event.value == _CUSTOM:
            self._yaml_path = None
            return

        yaml_path = Path(str(event.value))
        if not yaml_path.exists():
            self._show_error(f"Domain config not found: {yaml_path}")
            return

        try:
            cfg = ExperimentConfig.from_yaml(yaml_path)
        except (ValueError, OSError) as e:
            self._show_error(f"Failed to load {yaml_path}: {e}")
            return

        # Pick preset from difficulty mapping if the YAML uses the default preset
        domain_name = yaml_path.parent.name
        diff_info = DOMAIN_DIFFICULTY.get(domain_name)
        if diff_info and cfg.preset == "default":
            cfg.preset = diff_info[2]

        self._yaml_path = yaml_path
        # Resolve data path relative to the YAML file for display
        resolved_data = cfg.resolve_data_path(yaml_path.parent)
        cfg.data = str(resolved_data)
        # Set output dir to domain name
        cfg.output_dir = f"experiments/runs/{yaml_path.parent.name}"
        self._apply_config(cfg)

    @on(Select.Changed)
    def _on_provider_changed(self, event: Select.Changed) -> None:
        """When a provider dropdown changes, update the corresponding model dropdown."""
        sel_id = event.select.id or ""
        if not sel_id.startswith("model-") or not sel_id.endswith("-provider"):
            return
        agent = sel_id.removeprefix("model-").removesuffix("-provider")
        provider = event.value if isinstance(event.value, str) else None
        models = MODELS_BY_PROVIDER.get(provider, ALL_MODELS) if provider else ALL_MODELS
        model_sel = self.query_one(f"#model-{agent}-name", Select)
        model_sel.set_options(models)

    def _show_error(self, msg: str) -> None:
        """Display an error message in the error display area."""
        display = self.query_one("#error-display", Static)
        display.update(msg)

    def _build_config(self) -> ExperimentConfig | None:
        """Collect form values and build an ExperimentConfig, or None on validation error."""
        from pydantic import ValidationError

        data = self.query_one("#data-input", Input).value.strip()
        goal = self.query_one("#goal-input", TextArea).text.strip()

        if not data or not goal:
            self._show_error("Data path and goal are required.")
            return None

        self._show_error("")

        # Collect per-agent model overrides
        models_dict: dict[str, AgentModelConfig] = {}
        for agent in _AGENT_FIELDS:
            model_val = self.query_one(f"#model-{agent}-name", Select).value
            if not isinstance(model_val, str):
                continue
            model_name = model_val
            provider_val = self.query_one(f"#model-{agent}-provider", Select).value
            reasoning_val = self.query_one(f"#model-{agent}-reasoning", Select).value
            provider_str = provider_val if isinstance(provider_val, str) else "anthropic"
            reasoning_str = reasoning_val if isinstance(reasoning_val, str) else "off"
            try:
                models_dict[agent] = AgentModelConfig(
                    provider=provider_str,  # type: ignore[arg-type]
                    model=model_name,
                    reasoning=reasoning_str,  # type: ignore[arg-type]
                )
            except (ValidationError, ValueError) as e:
                self._show_error(f"Invalid model config for {agent}: {e}")
                return None

        # Collect critic overrides
        critics_list: list[AgentModelConfig] = []
        for i in range(_NUM_CRITIC_SLOTS):
            model_val = self.query_one(f"#model-critic-{i}-name", Select).value
            if not isinstance(model_val, str):
                continue
            provider_val = self.query_one(f"#model-critic-{i}-provider", Select).value
            reasoning_val = self.query_one(f"#model-critic-{i}-reasoning", Select).value
            provider_str = provider_val if isinstance(provider_val, str) else "anthropic"
            reasoning_str = reasoning_val if isinstance(reasoning_val, str) else "off"
            try:
                critics_list.append(
                    AgentModelConfig(
                        provider=provider_str,  # type: ignore[arg-type]
                        model=model_val,
                        reasoning=reasoning_str,  # type: ignore[arg-type]
                    )
                )
            except (ValidationError, ValueError) as e:
                self._show_error(f"Invalid model config for critic {i + 1}: {e}")
                return None

        has_overrides = bool(models_dict) or bool(critics_list)
        models = (
            ExperimentModelsConfig(
                **models_dict,
                critics=critics_list,
            )
            if has_overrides
            else None
        )

        try:
            return ExperimentConfig(
                data=data,
                goal=goal,
                preset=str(self.query_one("#preset-select", Select).value),
                max_iterations=int(self.query_one("#max-iterations-input", Input).value or "20"),
                debate_rounds=int(self.query_one("#debate-rounds-input", Input).value or "1"),
                output_dir=self.query_one("#output-dir-input", Input).value or "experiments",
                models=models,
            )
        except (ValidationError, ValueError) as e:
            self._show_error(str(e))
            return None

    def _validate_models(self, config: ExperimentConfig) -> list[str]:
        """Run provider auth, model name, and reasoning validation."""
        from auto_scientist.model_config import ModelConfig
        from auto_scientist.orchestrator import (
            _check_provider_auth,
            _validate_model_names,
            _validate_reasoning_configs,
        )

        mc = ModelConfig.from_experiment_config(config)
        errors: list[str] = []

        # Check API keys for required providers
        providers: set[str] = {"anthropic"}
        if mc.summarizer:
            providers.add(mc.summarizer.provider)
        for critic in mc.critics:
            providers.add(critic.provider)
        for provider in sorted(providers):
            err = _check_provider_auth(provider)
            if err:
                errors.append(err)

        # Check model names exist
        errors.extend(_validate_model_names(mc))

        # Check reasoning configs are valid for provider/model
        errors.extend(_validate_reasoning_configs(mc))

        return errors

    def action_run(self) -> None:
        config = self._build_config()
        if config is None:
            return

        # Validate models, reasoning, and API keys before launching
        try:
            errors = self._validate_models(config)
        except (ValueError, OSError, ImportError) as e:
            self._show_error(f"Validation failed: {e}")
            return
        if errors:
            self._show_error("\n".join(errors))
            return

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
            self._show_error(f"Failed to save config: {e}")
            return

        self._show_error("")
        self.notify(f"Config saved to {save_path}")

    async def action_quit(self) -> None:
        self.exit(None)
