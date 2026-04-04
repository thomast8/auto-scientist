"""CLI entry point: run, resume, status commands."""

import atexit
import logging
import os
import signal
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

import click
from dotenv import load_dotenv

from auto_scientist.console import PipelineApp, console
from auto_scientist.experiment_config import ExperimentConfig
from auto_scientist.launch_app import LaunchApp
from auto_scientist.model_config import ModelConfig
from auto_scientist.orchestrator import Orchestrator
from auto_scientist.state import ExperimentState

load_dotenv()

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process cleanup: kill SDK subprocesses on unexpected exit
# ---------------------------------------------------------------------------
# When auto-scientist is killed by SIGHUP (terminal closed) or SIGTERM,
# Python's default handler terminates immediately with no cleanup.  This
# orphans SDK subprocesses (claude CLI, codex app-server) that may be
# executing experiment scripts at 100 % CPU.
#
# Fix: install signal handlers that kill the process group before exiting
# and an atexit handler as a safety net for crashes / normal exit.
# ---------------------------------------------------------------------------

_cleanup_done = False


def _kill_child_processes() -> None:
    """Terminate direct child processes via ``pgrep -P``.

    Targets only our children, so it is safe to call from both signal
    handlers and atexit without killing sibling processes like git or
    pre-commit hooks that share our process group.

    Idempotent: repeated calls are no-ops after the first.
    """
    global _cleanup_done  # noqa: PLW0603 - intentional module-level flag
    if _cleanup_done:
        return
    _cleanup_done = True

    import subprocess as _sp

    # Prevent recursive delivery while we broadcast
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    pid = os.getpid()
    try:
        result = _sp.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
        )
        for line in result.stdout.strip().splitlines():
            with suppress(ProcessLookupError, PermissionError, ValueError, OSError):
                os.kill(int(line.strip()), signal.SIGTERM)
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass


def _fatal_signal_handler(signum: int, _frame: Any) -> None:
    """Handle SIGHUP / SIGTERM by killing children and exiting."""
    _kill_child_processes()
    # os._exit avoids Python shutdown machinery that can hang
    # during signal handling.  atexit handlers are intentionally
    # skipped since we already cleaned up above.
    os._exit(128 + signum)


def _install_cleanup_handlers() -> None:
    """Register signal and atexit handlers for child-process cleanup.

    Must be called from the main thread before any SDK subprocesses are
    spawned.
    """
    signal.signal(signal.SIGHUP, _fatal_signal_handler)
    signal.signal(signal.SIGTERM, _fatal_signal_handler)
    atexit.register(_kill_child_processes)


def _run_orchestrator(orchestrator: Orchestrator) -> None:
    """Run the orchestrator with user-friendly error handling."""
    _install_cleanup_handlers()
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
            elif "require provider" in issue and "SDK" in issue:
                parts.append(f"  - {issue}")
                parts.append(
                    "    Fix: SDK agents must use 'anthropic' or 'openai' provider. "
                    "Use --provider to switch, or override per-agent in YAML config."
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


def _detect_retry_agent(version_dir: Path) -> str | None:
    """Detect which agent to resume from within a failed iteration.

    Walks the agent execution order and returns the last agent whose
    artifacts are present on disk, so that earlier agents are skipped
    on retry.  Returns None when no artifacts exist (full restart).
    """
    from auto_scientist.resume import _AGENT_ARTIFACTS, AGENT_ORDER

    last_completed = None
    for agent in AGENT_ORDER:
        artifacts = _AGENT_ARTIFACTS[agent]
        if not artifacts:
            # Coder has no tracked artifacts; check for experiment.py instead
            if agent == "coder" and (version_dir / "experiment.py").exists():
                last_completed = agent
            continue
        if all((version_dir / a).exists() for a in artifacts):
            last_completed = agent

    # Resume from the agent that failed (= last_completed itself if it
    # produced partial output, or the one after if it fully completed).
    # The coder is the final agent: if its script exists, re-run the coder.
    if last_completed == "coder":
        return "coder"
    if last_completed is not None:
        idx = AGENT_ORDER.index(last_completed)
        if idx + 1 < len(AGENT_ORDER):
            return AGENT_ORDER[idx + 1]
    return None


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
    provider: str | None = None,
) -> ModelConfig:
    """Resolve ModelConfig from CLI flags (TOML configs only)."""
    if config_path and preset:
        raise click.UsageError("--config and --preset are mutually exclusive")

    # Resolve preset name, applying provider variant if applicable
    preset_name = preset or "default"
    if provider and provider != "anthropic":
        variant = f"{preset_name}-{provider}"
        from auto_scientist.model_config import BUILTIN_PRESETS

        if variant in BUILTIN_PRESETS:
            preset_name = variant

    if config_path:
        model_config = ModelConfig.from_toml(Path(config_path))
    else:
        model_config = ModelConfig.builtin_preset(preset_name)

    # If provider variant didn't exist, override defaults.provider
    if provider and provider != "anthropic" and not preset_name.endswith(f"-{provider}"):
        model_config.defaults = model_config.defaults.model_copy(update={"provider": provider})

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
        max_iterations=exp_config.max_iterations,
    )

    orchestrator = Orchestrator(
        state=state,
        data_path=data_path,
        output_dir=resolved_output,
        max_iterations=exp_config.max_iterations,
        model_config=model_config,
        interactive=exp_config.interactive,
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
    "--output-dir",
    default="experiments/runs",
    type=click.Path(),
    help="Output directory for experiment runs",
)
@click.option(
    "--provider",
    "-p",
    default=None,
    type=click.Choice(["anthropic", "openai"], case_sensitive=False),
    help="Default provider for SDK agents: anthropic (default) or openai (uses Codex CLI).",
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
    output_dir: str,
    provider: str | None,
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
        if ctx.get_parameter_source("provider") == _cli and provider is not None:
            exp_config.provider = cast("Literal['anthropic', 'openai']", provider)

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

    model_config = _resolve_model_config(config_path, preset, no_summaries, provider)

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
        max_iterations=max_iterations,
    )

    orchestrator = Orchestrator(
        state=state,
        data_path=Path(data),
        output_dir=resolved_output,
        max_iterations=max_iterations,
        model_config=model_config,
        interactive=interactive,
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
    "--from-iteration",
    "--resume-from",
    "from_iteration",
    default=None,
    type=click.IntRange(min=0),
    help="Resume from this iteration (keeps all prior iterations intact). Requires --fork.",
)
@click.option(
    "--from-agent",
    "from_agent",
    default=None,
    type=click.Choice(
        ["analyst", "scientist", "debate", "revision", "coder"],
        case_sensitive=False,
    ),
    help=(
        "Resume from this agent within the target iteration "
        "(earlier agents loaded from disk). Requires --fork."
    ),
)
@click.option("--max-iterations", default=None, type=int, help="Maximum iteration count")
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
    "--provider",
    "-p",
    default=None,
    type=click.Choice(["anthropic", "openai"], case_sensitive=False),
    help="Default provider for SDK agents: anthropic (default) or openai (uses Codex CLI).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Show debug log messages on console (always written to debug.log).",
)
@click.pass_context
def resume(
    ctx: click.Context,
    source: str,
    fork: bool,
    from_iteration: int | None,
    from_agent: str | None,
    max_iterations: int | None,
    output_dir: str | None,
    config_path: str | None,
    preset: str | None,
    no_summaries: bool,
    provider: str | None,
    verbose: bool,
):
    """Resume a previously paused or crashed run.

    By default, resumes in-place. With --fork, copies to a new directory
    first (original untouched). With --fork --from-iteration N, rewinds to
    iteration N in the copy (keeps iterations 0 through N-1).

    Use --from-agent to resume from a specific agent within an iteration,
    loading earlier agents' artifacts from disk.

    Examples:

      auto-scientist resume --from experiments/runs/my-run
      auto-scientist resume --from experiments/runs/my-run --fork
      auto-scientist resume --from experiments/runs/my-run --fork --from-iteration 3
      auto-scientist resume --from runs/my-run --fork --from-iteration 3 --from-agent scientist
      auto-scientist resume --from runs/my-run --fork --from-agent coder
    """
    import shutil

    from auto_scientist.resume import rewind_run

    # Validate flag combinations
    if from_iteration is not None and not fork:
        raise click.UsageError(
            "--from-iteration requires --fork (rewinding in-place would destroy data)"
        )
    if from_agent is not None and not fork:
        raise click.UsageError(
            "--from-agent requires --fork (modifying artifacts in-place would destroy data)"
        )
    if output_dir is not None and not fork:
        raise click.UsageError("--output-dir requires --fork")

    src, source_state = _resolve_source(source)

    # --- Resolve max_iterations ---
    default_max = 20
    _cli_source = click.core.ParameterSource.COMMANDLINE
    user_set_max_iter = ctx.get_parameter_source("max_iterations") == _cli_source
    saved_max = source_state.max_iterations
    is_completed = source_state.phase in ("stopped", "report")

    if user_set_max_iter:
        assert max_iterations is not None
        effective_max = max_iterations
        console.print(
            f"Max iterations: {effective_max} (from --max-iterations)",
            style="bold",
        )
    elif is_completed:
        # Completed run being forked: the original cap is exhausted, so extend
        # to the default to give new iterations room to run.
        effective_max = default_max
        console.print(
            f"[yellow]Original run completed with max_iterations={saved_max or '?'}. "
            f"Extending to {effective_max} for the forked run.[/yellow]",
        )
        console.print(
            "[dim]Use --max-iterations to set a different cap.[/dim]",
        )
    elif saved_max is not None:
        effective_max = saved_max
        console.print(
            f"Max iterations: {effective_max} (restored from original run)",
            style="bold",
        )
        console.print(
            "[dim]Use --max-iterations to override.[/dim]",
        )
    else:
        # Old state file without max_iterations saved
        effective_max = default_max
        if source_state.iteration >= effective_max:
            raise click.UsageError(
                f"No max_iterations in saved state (old format) and current "
                f"iteration ({source_state.iteration}) already >= default ({effective_max}). "
                f"Pass --max-iterations explicitly."
            )
        console.print(
            f"[yellow]No max_iterations in saved state (old format). "
            f"Using default: {effective_max}.[/yellow]",
        )
        console.print(
            "[dim]Use --max-iterations to set a different cap.[/dim]",
        )

    max_iterations = effective_max

    if fork:
        # Determine output directory
        if output_dir is None:
            run_dir = _next_output_dir(src)
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
        target = from_iteration if from_iteration is not None else source_state.iteration
        try:
            result = rewind_run(run_dir, target, from_agent=from_agent)
        except ValueError as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise click.UsageError(str(e)) from None
        except Exception as e:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise click.ClickException(f"Failed to rewind run: {e}") from None

        state = result.state
        from_agent = result.from_agent  # may have been normalized
        restored_panels = result.restored_panels

        agent_info = f", from agent '{from_agent}'" if from_agent else ""
        console.print(
            f"Resuming from iteration {state.iteration}{agent_info} "
            f"({len(state.versions)} prior iterations preserved)"
        )
    else:
        run_dir = src
        state = source_state
        restored_panels = None

        if state.phase in ("stopped", "report"):
            # Allow in-place retry when the latest version failed
            latest = state.versions[-1] if state.versions else None
            if latest and latest.status == "failed":
                target = state.iteration - 1
                retry_agent = _detect_retry_agent(run_dir / f"v{target:02d}")
                result = rewind_run(run_dir, target, from_agent=retry_agent)
                state = result.state
                from_agent = result.from_agent
                restored_panels = result.restored_panels
                agent_info = f" from agent '{from_agent}'" if from_agent else ""
                console.print(
                    f"[yellow]Retrying failed iteration {target}{agent_info} in-place[/yellow]"
                )
            else:
                raise click.UsageError(
                    "This run has already completed. Use --fork to continue "
                    "from a copy (the original run is preserved)."
                )

    # Persist resolved max_iterations so future resumes have it
    state.max_iterations = max_iterations
    state.save(run_dir / "state.json")

    # Resolve model config
    if config_path or preset or provider:
        model_config = _resolve_model_config(config_path, preset, no_summaries, provider)
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
        verbose=verbose,
        skip_to_agent=from_agent,
        restored_panels=restored_panels,
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

    Shows iteration layout and which agents have artifacts on disk,
    so you know what --from-iteration and --from-agent values are valid
    for the resume command.

    Examples:

      auto-scientist status --from experiments/runs/my-run
    """
    import json
    import re

    from auto_scientist.resume import _AGENT_ARTIFACTS, STOP_GATE_AGENTS

    run_dir, loaded_state = _resolve_source(source)

    # Truncate goal to a single display line
    goal = loaded_state.goal
    max_goal = 72
    goal_display = f"{goal[:max_goal]}..." if len(goal) > max_goal else goal

    click.echo(f"Domain:     {loaded_state.domain}")
    click.echo(f"Goal:       {goal_display}")
    click.echo(f"Phase:      {loaded_state.phase}")
    click.echo(f"Iteration:  {loaded_state.iteration}")
    if loaded_state.max_iterations is not None:
        click.echo(f"Max iter:   {loaded_state.max_iterations}")
    click.echo(f"Run dir:    {run_dir}")
    if loaded_state.data_path:
        click.echo(f"Data:       {loaded_state.data_path}")

    # Version status summary (completed/failed/crashed)
    if loaded_state.versions:
        counts: dict[str, int] = {}
        for v in loaded_state.versions:
            counts[v.status] = counts.get(v.status, 0) + 1
        parts = [f"{n} {s}" for s, n in counts.items()]
        click.echo(f"Runs:       {', '.join(parts)}")

    if loaded_state.dead_ends:
        click.echo(f"Dead ends:  {len(loaded_state.dead_ends)}")

    # Show per-iteration agent artifacts (granular, including stop gate + revision)
    step_artifacts: list[tuple[str, str]] = [
        (agent, artifacts[0])
        for agent, artifacts in _AGENT_ARTIFACTS.items()
        if artifacts  # skip coder (no fixed artifact)
    ]
    step_artifacts.append(("coder", "experiment.py"))

    version_re = re.compile(r"^v(\d+)$")
    try:
        version_dirs = sorted(
            (int(m.group(1)), child)
            for child in run_dir.iterdir()
            if child.is_dir() and (m := version_re.match(child.name))
        )
    except OSError as e:
        click.echo(f"\n(Could not scan iteration directories: {e})")
        return
    if version_dirs:
        click.echo()
        click.echo("Iterations on disk:")
        for idx, vdir in version_dirs:
            steps_present = []
            for step_name, artifact in step_artifacts:
                if (vdir / artifact).exists():
                    steps_present.append(step_name)
            steps_str = ", ".join(steps_present) if steps_present else "(empty)"
            click.echo(f"  v{idx:02d} (--from-iteration {idx}): {steps_str}")

        last_idx, last_vdir = version_dirs[-1]

        # Show stop reason if the scientist wants to stop
        plan_path = last_vdir / "plan.json"
        if plan_path.exists():
            try:
                plan = json.loads(plan_path.read_text())
                if plan.get("should_stop"):
                    stop_reason = plan.get("stop_reason", "unknown")
                    click.echo(f"\nStop requested: {stop_reason}")
            except (json.JSONDecodeError, OSError) as e:
                click.echo(f"\n(Could not read {plan_path.name}: {e})")

        # Resume suggestion: find the next missing step.
        # Conditional steps: stop gate (only if any stop gate artifact exists),
        # revision (only if debate ran). Only suggest resumable agents.
        present = {name for name, artifact in step_artifacts if (last_vdir / artifact).exists()}
        stop_gate_active = bool(present & STOP_GATE_AGENTS)
        debate_ran = "debate" in present

        # Steps that are conditional on context
        conditional_agents = STOP_GATE_AGENTS | (set() if debate_ran else {"revision"})

        expected_steps = [
            name
            for name, _ in step_artifacts
            if name not in conditional_agents
            or (name in STOP_GATE_AGENTS and stop_gate_active)
            or (name == "revision" and debate_ran)
        ]
        # Only suggest agents that are actually resumable via --from-agent
        _resumable = {"analyst", "scientist", "debate", "revision", "coder"}
        next_step = next(
            (s for s in expected_steps if s not in present and s in _resumable),
            None,
        )
        if next_step:
            click.echo()
            click.echo("Resume examples:")
            click.echo(
                f"  auto-scientist resume --from {run_dir} --fork "
                f"--from-iteration {last_idx} --from-agent {next_step}"
            )


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
