"""Auto-Reviewer CLI entrypoint.

Minimal wrapper over the shared auto_core.orchestrator.Orchestrator. The
reviewer registry is installed on package import (see
`auto_reviewer/__init__.py`), so by the time CLI commands run the core
runtime already knows the reviewer's styles, prompts, and agent fields.

Commands:
    auto-reviewer review --pr <ref> --repo-path <path> --goal "<text>"
    auto-reviewer resume <run_dir>
    auto-reviewer status <run_dir>
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click
from auto_core.model_config import ModelConfig
from auto_core.orchestrator import Orchestrator
from auto_core.resume import RewindResult
from auto_core.state import RunState

from auto_reviewer.config import ReviewConfig

logger = logging.getLogger("auto_reviewer")


def _default_workspace(pr_ref: str) -> Path:
    """Compute a default review workspace directory for a given PR ref."""
    slug = pr_ref.replace("/", "_").replace("#", "_").replace(" ", "_").strip("_")
    base = Path.cwd() / "review_workspace"
    base.mkdir(exist_ok=True)
    candidate = base / slug
    i = 1
    while candidate.exists():
        i += 1
        candidate = base / f"{slug}.{i:02d}"
    return candidate


@click.group()
def cli() -> None:
    """Auto-Reviewer: autonomous bug-hunting PR reviewer."""


@cli.command()
@click.option("--pr", "pr_ref", required=True, help="PR ref (e.g. owner/repo#123 or branch name).")
@click.option(
    "--repo-path",
    "repo_path",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Filesystem path to the repository being reviewed.",
)
@click.option(
    "--goal",
    default="Find correctness bugs introduced or exposed by the PR.",
    help="Review goal in natural language.",
)
@click.option("--base-ref", default=None, help="Base ref the PR targets (e.g. main).")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Review workspace directory. Defaults to ./review_workspace/<pr_slug>.",
)
@click.option(
    "--max-iterations",
    type=int,
    default=5,
    help="Maximum investigation iterations before stopping.",
)
@click.option(
    "--preset",
    default="default",
    help="Model preset name (passed to auto_core.model_config.ModelConfig.builtin_preset).",
)
@click.option(
    "--critics",
    default=None,
    help=(
        "Comma-separated adversary model specs, e.g. "
        "'openai:gpt-4o,google:gemini-2.5-pro'. Empty string disables debate."
    ),
)
@click.option("-v", "--verbose", is_flag=True)
def review(
    pr_ref: str,
    repo_path: str,
    goal: str,
    base_ref: str | None,
    output_dir: str | None,
    max_iterations: int,
    preset: str,
    critics: str | None,
    verbose: bool,
) -> None:
    """Review a pull request end-to-end."""
    repo_abs = Path(repo_path).resolve()
    workspace = Path(output_dir).resolve() if output_dir else _default_workspace(pr_ref)
    workspace.mkdir(parents=True, exist_ok=True)

    # Build ReviewConfig and persist it; the intake agent will refine it
    # once it has pulled the PR diff.
    review_config = ReviewConfig(
        name=pr_ref.replace("/", "_").replace("#", "_"),
        description=f"PR review of {pr_ref} against {repo_abs}",
        run_command="uv run pytest -x -s {script_path}",
        repo_path=str(repo_abs),
        pr_ref=pr_ref,
        base_ref=base_ref,
    )
    config_path = workspace / "review_config.json"
    config_path.write_text(review_config.model_dump_json(indent=2))

    # Initial RunState: the intake phase will update data_path / raw_data_path.
    state = RunState(
        domain=pr_ref,
        goal=goal,
        phase="ingestion",
        max_iterations=max_iterations,
        config_path=str(config_path),
    )

    model_config = ModelConfig.builtin_preset(preset)
    if critics is not None:
        # Override critics list from the flag. Empty string means no debate.
        if not critics.strip():
            model_config.critics = []
        else:
            from auto_core.model_config import AgentModelConfig

            new_critics = []
            valid_providers = {"anthropic", "openai", "google"}
            for spec in critics.split(","):
                spec = spec.strip()
                if not spec:
                    continue
                provider, model = spec.split(":", 1)
                if provider not in valid_providers:
                    raise click.BadParameter(
                        f"Unknown critic provider {provider!r}; valid: {sorted(valid_providers)}"
                    )
                new_critics.append(
                    AgentModelConfig.model_validate(
                        {"provider": provider, "model": model, "mode": "api"}
                    )
                )
            model_config.critics = new_critics

    orchestrator = Orchestrator(
        state=state,
        data_path=None,  # intake will pull the PR and set workspace paths
        output_dir=workspace,
        max_iterations=max_iterations,
        model_config=model_config,
        verbose=verbose,
    )
    orchestrator.config = review_config  # pre-seed so intake doesn't overwrite

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        click.echo(
            "Interrupted. State is persisted at state.json; use `auto-reviewer resume` to continue."
        )
        sys.exit(130)


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
def status(run_dir: str) -> None:
    """Print the current phase + iteration for a review run."""
    run_path = Path(run_dir)
    state_path = run_path / "state.json"
    if not state_path.exists():
        click.echo(f"No state.json in {run_path}")
        sys.exit(1)
    state = RunState.load(state_path)
    click.echo(
        json.dumps(
            {
                "pr": state.domain,
                "goal": state.goal,
                "phase": state.phase,
                "iteration": state.iteration,
                "probes": len(state.versions),
                "predictions": len(state.prediction_history),
                "pending_abductions": len(state.pending_abductions),
            },
            indent=2,
        )
    )


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--from-iteration",
    type=int,
    default=None,
    help="Rewind to this iteration before resuming.",
)
def resume(run_dir: str, from_iteration: int | None) -> None:
    """Resume a paused review run."""
    from auto_core.resume import rewind_run

    run_path = Path(run_dir).resolve()
    state_path = run_path / "state.json"
    if not state_path.exists():
        click.echo(f"No state.json in {run_path}", err=True)
        sys.exit(1)
    current_state = RunState.load(state_path)
    target_iteration = from_iteration if from_iteration is not None else current_state.iteration
    rewound: RewindResult = rewind_run(run_path, target_iteration=target_iteration, from_agent=None)
    state = rewound.state

    mc_path = run_path / "model_config.json"
    model_config = (
        ModelConfig.model_validate_json(mc_path.read_text())
        if mc_path.exists()
        else ModelConfig.builtin_preset("default")
    )

    orchestrator = Orchestrator(
        state=state,
        data_path=Path(state.data_path) if state.data_path else None,
        output_dir=run_path,
        max_iterations=state.max_iterations or 5,
        model_config=model_config,
        restored_panels=rewound.restored_panels,
    )

    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        click.echo("Interrupted again. Re-run `auto-reviewer resume` to continue.")
        sys.exit(130)


if __name__ == "__main__":
    cli()
