"""CLI entry point: run, resume, status commands."""

import asyncio
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

from auto_scientist.orchestrator import Orchestrator
from auto_scientist.state import ExperimentState


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


@click.group()
def cli():
    """Auto-Scientist: Autonomous scientific investigation framework."""


@cli.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Path to dataset")
@click.option("--goal", required=True, help="Problem statement / investigation goal")
@click.option("--max-iterations", default=20, help="Maximum iteration count")
@click.option(
    "--critics",
    default="",
    help="Comma-separated critic models (e.g., 'openai:gpt-4o,google:gemini-2.5-pro')",
)
@click.option("--schedule", default=None, help="Time window for execution (e.g., '22:00-06:00')")
@click.option("--interactive", is_flag=True, help="Enable interactive mode")
@click.option(
    "--debate-rounds",
    default=2,
    type=int,
    help="Number of critic-scientist debate rounds (1 = single-pass, default 2)",
)
@click.option(
    "--output-dir",
    default="experiments",
    type=click.Path(),
    help="Output directory for experiments",
)
@click.option(
    "--model",
    default=None,
    help="Claude model for agents (e.g., 'claude-sonnet-4-6'). Uses SDK default if omitted.",
)
@click.option(
    "--no-stream",
    is_flag=True,
    help="Disable live token streaming during debate phase",
)
@click.option(
    "--summary-model",
    default=None,
    help="OpenAI model for periodic agent summaries (e.g., 'gpt-4o-mini'). Opt-in.",
)
def run(
    data: str,
    goal: str,
    max_iterations: int,
    critics: str,
    schedule: str | None,
    interactive: bool,
    debate_rounds: int,
    output_dir: str,
    model: str | None,
    no_stream: bool,
    summary_model: str | None,
):
    """Run autonomous scientific investigation from raw data."""
    critic_list = [c.strip() for c in critics.split(",") if c.strip()] if critics else []

    data_abs = str(Path(data).resolve())

    resolved_output = _next_output_dir(Path(output_dir))
    if resolved_output != Path(output_dir):
        click.echo(f"Previous run detected in {output_dir}/. Using {resolved_output}/ instead.")

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
        critic_models=critic_list,
        interactive=interactive,
        debate_rounds=debate_rounds,
        model=model,
        stream=not no_stream,
        summary_model=summary_model,
    )

    asyncio.run(orchestrator.run())


@cli.command()
@click.option(
    "--state",
    required=True,
    type=click.Path(exists=True),
    help="Path to state.json for resumption",
)
def resume(state: str):
    """Resume a previously paused or crashed run."""
    loaded_state = ExperimentState.load(Path(state))
    output_dir = Path(state).parent

    data_path = Path(loaded_state.data_path) if loaded_state.data_path else None

    orchestrator = Orchestrator(
        state=loaded_state,
        data_path=data_path,
        output_dir=output_dir,
    )

    asyncio.run(orchestrator.run())


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
