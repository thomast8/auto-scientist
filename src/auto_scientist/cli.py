"""CLI entry point: run, resume, status commands."""

from pathlib import Path

import click
from dotenv import load_dotenv

from auto_scientist.console import PipelineApp, console
from auto_scientist.experiment_config import ExperimentConfig
from auto_scientist.launch_app import LaunchApp
from auto_scientist.model_config import ModelConfig
from auto_scientist.orchestrator import Orchestrator
from auto_scientist.state import ExperimentState

load_dotenv()


def _run_orchestrator(orchestrator: Orchestrator) -> None:
    """Run the orchestrator with user-friendly error handling."""
    try:
        app = PipelineApp(orchestrator)
        app.run()
    except RuntimeError as e:
        msg = str(e)
        if "Pre-flight check failed" not in msg:
            raise

        # Format pre-flight errors with guidance
        lines = msg.split("\n")
        header = lines[0]
        issues = [line.strip("- ").strip() for line in lines[1:] if line.strip()]

        parts = [f"\n{header}\n"]
        for issue in issues:
            if "not found" in issue and "openai" in issue.lower():
                parts.append(f"  - {issue}")
                parts.append(
                    "    Fix: check the model ID at https://platform.openai.com/docs/models"
                )
            elif "not found" in issue and "google" in issue.lower():
                parts.append(f"  - {issue}")
                parts.append(
                    "    Fix: check the model ID at https://ai.google.dev/gemini-api/docs/models"
                )
            elif "require provider 'anthropic'" in issue:
                parts.append(f"  - {issue}")
                parts.append(
                    "    Fix: SDK agents (analyst, scientist, coder, ingestor, report) must use "
                    "Anthropic models (claude-*). Use non-Anthropic models only for critics "
                    "and summarizer."
                )
            elif "authentication failed" in issue.lower() or "API_KEY" in issue:
                parts.append(f"  - {issue}")
                parts.append("    Fix: set the required API key in your environment or .env file")
            elif "Claude Code CLI not found" in issue:
                parts.append(f"  - {issue}")
                parts.append("    Fix: npm install -g @anthropic-ai/claude-code")
            else:
                parts.append(f"  - {issue}")

        parts.append(
            "\nSee --config or --preset options to change model assignments. "
            "Run with --help for details."
        )
        raise click.ClickException("\n".join(parts)) from None


def _resolve_source(source: str) -> tuple[Path, ExperimentState]:
    """Resolve a --from argument to (run_directory, loaded_state).

    Accepts either a run directory or a path to state.json directly.
    """
    source_path = Path(source)
    if source_path.is_dir():
        state_path = source_path / "state.json"
    else:
        state_path = source_path
        source_path = state_path.parent
    if not state_path.exists():
        raise click.UsageError(f"No state.json found at {state_path}")
    return source_path, ExperimentState.load(state_path)


def _next_output_dir(base: Path) -> Path:
    """If *base* already contains a state.json, return base_001, base_002, etc."""
    if not (base / "state.json").exists():
        return base
    seq = 1
    while True:
        candidate = base.parent / f"{base.name}_{seq:03d}"
        if not (candidate / "state.json").exists():
            return candidate
        seq += 1


def _is_yaml_config(path: str) -> bool:
    """Check if a config file path is a YAML file by extension."""
    return Path(path).suffix.lower() in (".yaml", ".yml")


def _resolve_model_config(
    config_path: str | None,
    preset: str | None,
    no_summaries: bool = False,
) -> ModelConfig:
    """Resolve ModelConfig from CLI flags (TOML configs only)."""
    if config_path and preset:
        raise click.UsageError("--config and --preset are mutually exclusive")
    if config_path:
        model_config = ModelConfig.from_toml(Path(config_path))
    elif preset:
        model_config = ModelConfig.builtin_preset(preset)
    else:
        model_config = ModelConfig.builtin_preset("default")

    if no_summaries:
        model_config.summarizer = None

    return model_config


def _run_from_experiment_config(exp_config: ExperimentConfig, data_path: Path) -> None:
    """Build Orchestrator from an ExperimentConfig and run it."""
    try:
        model_config = ModelConfig.from_experiment_config(exp_config)
    except ValueError as e:
        raise click.UsageError(str(e)) from None

    if not data_path.exists():
        raise click.UsageError(
            f"Data path does not exist: {data_path}\n"
            f"Resolved from config value: {exp_config.data!r}"
        )

    resolved_output = _next_output_dir(Path(exp_config.output_dir))
    if resolved_output != Path(exp_config.output_dir):
        console.print(
            f"Previous run detected in {exp_config.output_dir}/. Using {resolved_output}/ instead.",
            style="yellow",
        )

    state = ExperimentState(
        domain="auto",
        goal=exp_config.goal,
        phase="ingestion",
        schedule=exp_config.schedule,
        data_path=str(data_path.resolve()),
    )

    orchestrator = Orchestrator(
        state=state,
        data_path=data_path,
        output_dir=resolved_output,
        max_iterations=exp_config.max_iterations,
        model_config=model_config,
        interactive=exp_config.interactive,
        debate_rounds=exp_config.debate_rounds,
        verbose=exp_config.verbose,
    )

    _run_orchestrator(orchestrator)


@click.group(invoke_without_command=True)
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Pre-fill TUI form from experiment.yaml",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None):
    """Auto-Scientist: Autonomous scientific investigation framework."""
    if ctx.invoked_subcommand is not None:
        return

    # Bare `auto-scientist` with no subcommand: launch the TUI form
    prefill = None
    if config_path and _is_yaml_config(config_path):
        try:
            prefill = ExperimentConfig.from_yaml(Path(config_path))
        except ValueError as e:
            raise click.UsageError(str(e)) from None

    app = LaunchApp(prefill=prefill)
    result = app.run()

    if result is None or app.result_config is None:
        return

    exp_config = app.result_config
    data_path = Path(exp_config.data)

    _run_from_experiment_config(exp_config, data_path)


@cli.command()
@click.option("--data", default=None, type=click.Path(), help="Path to dataset")
@click.option("--goal", default=None, help="Problem statement / investigation goal")
@click.option("--max-iterations", default=20, help="Maximum iteration count")
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to experiment.yaml or models.toml config file",
)
@click.option(
    "--preset", default=None, help="Named preset: turbo, fast, default (medium), high, max"
)
@click.option("--no-summaries", is_flag=True, help="Disable periodic agent summaries")
@click.option("--schedule", default=None, help="Time window for execution (e.g., '22:00-06:00')")
@click.option("--interactive", is_flag=True, help="Enable interactive mode")
@click.option(
    "--debate-rounds",
    default=1,
    type=int,
    help="Number of critic-scientist debate rounds per persona (1 = single-pass, default 1)",
)
@click.option(
    "--output-dir",
    default="experiments/runs",
    type=click.Path(),
    help="Output directory for experiment runs",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show debug log messages on console (always written to debug.log).",
)
@click.pass_context
def run(
    ctx: click.Context,
    data: str | None,
    goal: str | None,
    max_iterations: int,
    config_path: str | None,
    preset: str | None,
    no_summaries: bool,
    schedule: str | None,
    interactive: bool,
    debate_rounds: int,
    output_dir: str,
    verbose: bool,
):
    """Run autonomous scientific investigation from raw data."""
    # YAML config path: load ExperimentConfig, merge CLI overrides on top
    if config_path and _is_yaml_config(config_path):
        try:
            exp_config = ExperimentConfig.from_yaml(Path(config_path))
        except ValueError as e:
            raise click.UsageError(str(e)) from None

        yaml_dir = Path(config_path).parent

        # CLI flags override YAML values (use get_parameter_source to detect explicit CLI flags)
        _cli = click.core.ParameterSource.COMMANDLINE
        if ctx.get_parameter_source("preset") == _cli and preset is not None:
            exp_config.preset = preset
        if ctx.get_parameter_source("max_iterations") == _cli:
            exp_config.max_iterations = max_iterations
        if ctx.get_parameter_source("debate_rounds") == _cli:
            exp_config.debate_rounds = debate_rounds
        if ctx.get_parameter_source("output_dir") == _cli:
            exp_config.output_dir = output_dir
        if ctx.get_parameter_source("schedule") == _cli:
            exp_config.schedule = schedule
        if ctx.get_parameter_source("interactive") == _cli:
            exp_config.interactive = interactive
        if ctx.get_parameter_source("verbose") == _cli:
            exp_config.verbose = verbose
        if ctx.get_parameter_source("no_summaries") == _cli and no_summaries:
            exp_config.summaries = False

        # Override data/goal only if explicitly provided on CLI
        data_cli_override = ctx.get_parameter_source("data") == _cli
        if data_cli_override and data is not None:
            exp_config.data = data
        if ctx.get_parameter_source("goal") == _cli and goal is not None:
            exp_config.goal = goal

        # CLI --data resolves from CWD; YAML data resolves from YAML dir
        data_path = (
            Path(data)
            if data_cli_override and data is not None
            else exp_config.resolve_data_path(yaml_dir)
        )

        _run_from_experiment_config(exp_config, data_path)
        return

    # Non-YAML path: require --data and --goal
    if not data:
        raise click.UsageError("Missing option '--data'. Required when not using a YAML config.")
    if not goal:
        raise click.UsageError("Missing option '--goal'. Required when not using a YAML config.")
    if not Path(data).exists():
        raise click.UsageError(f"Path '{data}' does not exist.")

    model_config = _resolve_model_config(config_path, preset, no_summaries)

    data_abs = str(Path(data).resolve())

    resolved_output = _next_output_dir(Path(output_dir))
    if resolved_output != Path(output_dir):
        console.print(
            f"Previous run detected in {output_dir}/. Using {resolved_output}/ instead.",
            style="yellow",
        )

    state = ExperimentState(
        domain="auto",
        goal=goal,
        phase="ingestion",
        schedule=schedule,
        data_path=data_abs,
    )

    orchestrator = Orchestrator(
        state=state,
        data_path=Path(data),
        output_dir=resolved_output,
        max_iterations=max_iterations,
        model_config=model_config,
        interactive=interactive,
        debate_rounds=debate_rounds,
        verbose=verbose,
    )

    _run_orchestrator(orchestrator)


@cli.command()
@click.option(
    "--from",
    "--state",
    "source",
    required=True,
    type=click.Path(exists=True),
    help="Path to run directory (or state.json)",
)
@click.option(
    "--fork",
    is_flag=True,
    help="Copy to a new directory before resuming (original untouched)",
)
@click.option(
    "--resume-from",
    "--at-iteration",
    "resume_from",
    default=None,
    type=click.IntRange(min=0),
    help=("Resume from this iteration (keeps all prior iterations intact). Requires --fork."),
)
@click.option("--max-iterations", default=20, type=int, help="Maximum iteration count")
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(),
    help="Output directory for forked run (default: auto-generated). Requires --fork.",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to models.toml config file (overrides saved config)",
)
@click.option(
    "--preset", default=None, help="Named preset: turbo, fast, default (medium), high, max"
)
@click.option("--no-summaries", is_flag=True, help="Disable periodic agent summaries")
@click.option(
    "--debate-rounds",
    default=1,
    type=int,
    help="Number of critic-scientist debate rounds per persona",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show debug log messages on console (always written to debug.log).",
)
def resume(
    source: str,
    fork: bool,
    resume_from: int | None,
    max_iterations: int,
    output_dir: str | None,
    config_path: str | None,
    preset: str | None,
    no_summaries: bool,
    debate_rounds: int,
    verbose: bool,
):
    """Resume a previously paused or crashed run.

    By default, resumes in-place. With --fork, copies to a new directory
    first (original untouched). With --fork --resume-from N, rewinds to
    iteration N in the copy (keeps iterations 0 through N-1).

    Examples:

      auto-scientist resume --from experiments/runs/my-run
      auto-scientist resume --from experiments/runs/my-run --fork
      auto-scientist resume --from experiments/runs/my-run --fork --resume-from 3
    """
    import shutil

    from auto_scientist.replay import rewind_run

    # Validate flag combinations
    if resume_from is not None and not fork:
        raise click.UsageError(
            "--resume-from requires --fork (rewinding in-place would destroy data)"
        )
    if output_dir is not None and not fork:
        raise click.UsageError("--output-dir requires --fork")

    src, source_state = _resolve_source(source)

    if fork:
        # Determine output directory
        if output_dir is None:
            run_dir = _next_output_dir(Path("experiments/runs") / src.name)
        else:
            run_dir = _next_output_dir(Path(output_dir))

        console.print(f"Forking {src} -> {run_dir}")
        try:
            shutil.copytree(src, run_dir)
        except (shutil.Error, OSError) as e:
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
            raise click.ClickException(f"Failed to copy {src} to {run_dir}: {e}") from None

        # Rewind handles everything: phase reset, report stripping, green
        # border, iteration bump (extend mode), artifact cleanup
        target = resume_from if resume_from is not None else source_state.iteration
        try:
            state = rewind_run(run_dir, target)
        except ValueError as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise click.UsageError(str(e)) from None
        except Exception as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise click.ClickException(f"Failed to rewind run: {e}") from None

        console.print(
            f"Resuming from iteration {state.iteration} "
            f"({len(state.versions)} prior iterations preserved)"
        )
    else:
        run_dir = src
        state = source_state

        if state.phase in ("stopped", "report"):
            raise click.UsageError(
                "This run has already completed. Use --fork to continue "
                "from a copy (the original run is preserved)."
            )

    # Resolve model config
    if config_path or preset:
        model_config = _resolve_model_config(config_path, preset, no_summaries)
    else:
        saved_mc = run_dir / "model_config.json"
        if saved_mc.exists():
            model_config = ModelConfig.model_validate_json(saved_mc.read_text())
        else:
            model_config = ModelConfig.builtin_preset("default")
        if no_summaries:
            model_config.summarizer = None

    data_path = Path(state.data_path) if state.data_path else None

    orchestrator = Orchestrator(
        state=state,
        data_path=data_path,
        output_dir=run_dir,
        max_iterations=max_iterations,
        model_config=model_config,
        debate_rounds=debate_rounds,
        verbose=verbose,
    )

    _run_orchestrator(orchestrator)


@cli.command()
@click.option(
    "--from",
    "--state",
    "source",
    required=True,
    type=click.Path(exists=True),
    help="Path to run directory (or state.json)",
)
def status(source: str):
    """Check progress of an experiment run.

    Examples:

      auto-scientist status --from experiments/runs/my-run
    """
    _, loaded_state = _resolve_source(source)
    click.echo(f"Domain:     {loaded_state.domain}")
    click.echo(f"Phase:      {loaded_state.phase}")
    click.echo(f"Iteration:  {loaded_state.iteration}")
    click.echo(f"Versions:   {len(loaded_state.versions)}")
    click.echo(f"Dead ends:  {len(loaded_state.dead_ends)}")


@cli.command()
@click.option(
    "--from",
    "--state",
    "source",
    required=True,
    type=click.Path(exists=True),
    help="Path to run directory (or state.json)",
)
def show(source: str):
    """Display a completed run in the TUI (read-only).

    Examples:

      auto-scientist show --from experiments/runs/my-run
    """
    from auto_scientist.console import ShowApp
    from auto_scientist.iteration_manifest import MANIFEST_FILENAME, load_manifest

    run_dir, _ = _resolve_source(source)
    records = load_manifest(run_dir / MANIFEST_FILENAME)
    if not records:
        raise click.ClickException(
            f"No iteration manifest found in {run_dir}. This run may predate manifest tracking."
        )
    app = ShowApp(records, run_title=run_dir.name)
    app.run()


if __name__ == "__main__":
    cli()
