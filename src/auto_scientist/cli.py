"""CLI entry point: run, resume, status commands."""

import asyncio
import importlib
from pathlib import Path

import click

from auto_scientist.config import DomainConfig
from auto_scientist.orchestrator import Orchestrator
from auto_scientist.state import ExperimentState


def load_domain_config(name: str) -> DomainConfig:
    """Dynamically load a domain config and inject its domain knowledge.

    Convention: domains/{name}/config.py exports {NAME}_CONFIG,
    domains/{name}/prompts.py exports {NAME}_DOMAIN_KNOWLEDGE.
    """
    upper = name.upper()

    config_mod = importlib.import_module(f"domains.{name}.config")
    config: DomainConfig = getattr(config_mod, f"{upper}_CONFIG")

    try:
        prompts_mod = importlib.import_module(f"domains.{name}.prompts")
        knowledge: str = getattr(prompts_mod, f"{upper}_DOMAIN_KNOWLEDGE", "")
    except (ModuleNotFoundError, AttributeError):
        knowledge = ""

    if knowledge:
        config = config.model_copy(update={"domain_knowledge": knowledge})

    return config


@click.group()
def cli():
    """Auto-Scientist: Autonomous scientific modelling framework."""


@cli.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="Path to dataset")
@click.option("--goal", required=True, help="Problem statement / modelling goal")
@click.option("--domain", default=None, help="Domain name (e.g., 'spo2'). Auto-detected if omitted")
@click.option("--max-iterations", default=20, help="Maximum iteration count")
@click.option(
    "--critics",
    default="",
    help="Comma-separated critic models (e.g., 'openai:gpt-4o,google:gemini-2.5-pro')",
)
@click.option("--schedule", default=None, help="Time window for execution (e.g., '22:00-06:00')")
@click.option("--interactive", is_flag=True, help="Enable interactive discovery mode")
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
    "--synthesis-interval",
    default=0,
    type=int,
    help="Condense notebook every N iterations (0 = disabled)",
)
def run(
    data: str,
    goal: str,
    domain: str | None,
    max_iterations: int,
    critics: str,
    schedule: str | None,
    interactive: bool,
    debate_rounds: int,
    output_dir: str,
    synthesis_interval: int,
):
    """Run autonomous scientific modelling from raw data."""
    critic_list = [c.strip() for c in critics.split(",") if c.strip()] if critics else []

    # Load domain config if a domain name is specified
    config: DomainConfig | None = None
    if domain:
        config = load_domain_config(domain)

    data_abs = str(Path(data).resolve())

    state = ExperimentState(
        domain=domain or "auto",
        goal=goal,
        phase="ingestion",
        schedule=schedule,
        data_path=data_abs,
    )

    orchestrator = Orchestrator(
        state=state,
        data_path=Path(data),
        output_dir=Path(output_dir),
        max_iterations=max_iterations,
        critic_models=critic_list,
        interactive=interactive,
        debate_rounds=debate_rounds,
        config=config,
        synthesis_interval=synthesis_interval,
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

    # Reload domain config if available
    config: DomainConfig | None = None
    if loaded_state.domain and loaded_state.domain != "auto":
        config = load_domain_config(loaded_state.domain)

    data_path = Path(loaded_state.data_path) if loaded_state.data_path else None

    orchestrator = Orchestrator(
        state=loaded_state,
        data_path=data_path,
        output_dir=output_dir,
        config=config,
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
