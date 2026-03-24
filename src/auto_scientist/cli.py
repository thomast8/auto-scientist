"""CLI entry point: run, resume, status commands."""

from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from auto_scientist.console import PipelineApp, console
from auto_scientist.model_config import ModelConfig
from auto_scientist.orchestrator import Orchestrator
from auto_scientist.state import ExperimentState


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
                parts.append("    Fix: check the model ID at https://platform.openai.com/docs/models")
            elif "not found" in issue and "google" in issue.lower():
                parts.append(f"  - {issue}")
                parts.append("    Fix: check the model ID at https://ai.google.dev/gemini-api/docs/models")
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


def _resolve_model_config(
    config_path: str | None,
    preset: str | None,
    no_summaries: bool = False,
) -> ModelConfig:
    """Resolve ModelConfig from CLI flags."""
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


@click.group()
def cli():
    """Auto-Scientist: Autonomous scientific investigation framework."""


@cli.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Path to dataset")
@click.option("--goal", required=True, help="Problem statement / investigation goal")
@click.option("--max-iterations", default=20, help="Maximum iteration count")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to models.toml config file",
)
@click.option("--preset", default=None, help="Named preset: default, fast")
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
    default="experiments",
    type=click.Path(),
    help="Output directory for experiments",
)
@click.option(
    "--no-stream",
    is_flag=True,
    help="Disable live token streaming during debate phase",
)
@click.option(
    "-v", "--verbose", is_flag=True,
    help="Show debug log messages on console (always written to debug.log).",
)
def run(
    data: str,
    goal: str,
    max_iterations: int,
    config_path: str | None,
    preset: str | None,
    no_summaries: bool,
    schedule: str | None,
    interactive: bool,
    debate_rounds: int,
    output_dir: str,
    no_stream: bool,
    verbose: bool,
):
    """Run autonomous scientific investigation from raw data."""
    model_config = _resolve_model_config(config_path, preset, no_summaries)

    data_abs = str(Path(data).resolve())

    resolved_output = _next_output_dir(Path(output_dir))
    if resolved_output != Path(output_dir):
        console.print(
            f"Previous run detected in {output_dir}/. "
            f"Using {resolved_output}/ instead.",
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
        stream=not no_stream,
        verbose=verbose,
    )

    _run_orchestrator(orchestrator)


@cli.command()
@click.option(
    "--state",
    required=True,
    type=click.Path(exists=True),
    help="Path to state.json for resumption",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to models.toml config file (overrides saved config)",
)
@click.option("--preset", default=None, help="Named preset: default, fast")
@click.option("--no-summaries", is_flag=True, help="Disable periodic agent summaries")
@click.option(
    "-v", "--verbose", is_flag=True,
    help="Show debug log messages on console (always written to debug.log).",
)
def resume(
    state: str,
    config_path: str | None,
    preset: str | None,
    no_summaries: bool,
    verbose: bool,
):
    """Resume a previously paused or crashed run."""
    loaded_state = ExperimentState.load(Path(state))
    output_dir = Path(state).parent

    if config_path or preset:
        model_config = _resolve_model_config(config_path, preset, no_summaries)
    else:
        saved_mc = output_dir / "model_config.json"
        if saved_mc.exists():
            model_config = ModelConfig.model_validate_json(saved_mc.read_text())
        else:
            model_config = ModelConfig.builtin_preset("default")
        if no_summaries:
            model_config.summarizer = None

    data_path = Path(loaded_state.data_path) if loaded_state.data_path else None

    orchestrator = Orchestrator(
        state=loaded_state,
        data_path=data_path,
        output_dir=output_dir,
        model_config=model_config,
        verbose=verbose,
    )

    _run_orchestrator(orchestrator)


@cli.command()
@click.option(
    "--state",
    required=True,
    type=click.Path(exists=True),
    help="Path to state.json to inspect",
)
def status(state: str):
    """Check progress of an experiment run."""
    loaded_state = ExperimentState.load(Path(state))
    click.echo(f"Domain:     {loaded_state.domain}")
    click.echo(f"Phase:      {loaded_state.phase}")
    click.echo(f"Iteration:  {loaded_state.iteration}")
    click.echo(f"Best:       {loaded_state.best_version} (score {loaded_state.best_score})")
    click.echo(f"Versions:   {len(loaded_state.versions)}")
    click.echo(f"Dead ends:  {len(loaded_state.dead_ends)}")


if __name__ == "__main__":
    cli()
