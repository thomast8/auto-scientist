"""Auto-Reviewer CLI entrypoint.

Minimal wrapper over the shared auto_core.orchestrator.Orchestrator. The
reviewer registry is installed on package import (see
`auto_reviewer/__init__.py`), so by the time CLI commands run the core
runtime already knows the reviewer's styles, prompts, and agent fields.

The `review` command takes a single natural-language prompt. The Intake
agent parses the prompt, locates (or clones) the target repository,
resolves base/head refs, and writes a populated ReviewConfig before the
downstream pipeline runs.

Commands:
    auto-reviewer review "<prompt>"
    auto-reviewer resume <run_dir>
    auto-reviewer status <run_dir>
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import click
from auto_core.app import PipelineApp
from auto_core.cli_cleanup import install_child_cleanup_handlers
from auto_core.model_config import ModelConfig
from auto_core.orchestrator import Orchestrator
from auto_core.resume import RewindResult
from auto_core.state import RunState
from dotenv import load_dotenv

# Pick up API keys from the repo's .env before any agent tries to talk to
# an LLM provider.
load_dotenv()

logger = logging.getLogger("auto_reviewer")


def _run_orchestrator(orchestrator: Orchestrator, interrupted_message: str) -> None:
    """Run the orchestrator with reviewer-specific interruption handling."""
    install_child_cleanup_handlers()
    try:
        PipelineApp(orchestrator).run()
    except KeyboardInterrupt:
        click.echo(interrupted_message)
        sys.exit(130)


def _slug(text: str, max_len: int = 60) -> str:
    """Filesystem-safe slug derived from free-form text."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("_")
    return (cleaned[:max_len] or "review").rstrip("_")


def _default_workspace() -> Path:
    """Timestamp-named workspace directory under ./review_workspace."""
    base = Path.cwd() / "review_workspace"
    base.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"review_{stamp}"
    i = 1
    while candidate.exists():
        i += 1
        candidate = base / f"review_{stamp}.{i:02d}"
    return candidate


@click.group()
def cli() -> None:
    """Auto-Reviewer: autonomous bug-hunting PR reviewer."""


@cli.command()
@click.argument("prompt")
@click.option(
    "--cwd",
    "cwd_opt",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Starting directory the intake agent uses to locate the repo. Defaults to cwd.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Review workspace directory. Defaults to ./review_workspace/review_<timestamp>.",
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
@click.option(
    "--interactive",
    is_flag=True,
    help="Allow the intake agent to ask clarifying questions (AskUserQuestion).",
)
@click.option("-v", "--verbose", is_flag=True)
def review(
    prompt: str,
    cwd_opt: str | None,
    output_dir: str | None,
    max_iterations: int,
    preset: str,
    critics: str | None,
    interactive: bool,
    verbose: bool,
) -> None:
    """Review a pull request end-to-end.

    The PROMPT is a natural-language description pointing at the code to
    review - a GitHub PR URL, an owner/repo#N reference, a branch name,
    or simply "my current branch". The intake agent resolves the pointer.
    """
    cwd = Path(cwd_opt).resolve() if cwd_opt else Path.cwd()
    workspace = Path(output_dir).resolve() if output_dir else _default_workspace()
    workspace.mkdir(parents=True, exist_ok=True)

    # Initial RunState. The intake agent will refine `data_path` to point at
    # the canonical data directory once it has populated it. Seeding with
    # cwd gives `validate_prerequisites` a real directory to check.
    state = RunState(
        domain=_slug(prompt),
        goal=prompt,
        phase="ingestion",
        max_iterations=max_iterations,
        config_path=str(workspace / "domain_config.json"),
        data_path=str(cwd),
    )

    model_config = ModelConfig.builtin_preset(preset)
    if critics is not None:
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
        data_path=cwd,
        output_dir=workspace,
        max_iterations=max_iterations,
        model_config=model_config,
        interactive=interactive,
        verbose=verbose,
    )
    _run_orchestrator(
        orchestrator,
        "Interrupted. State is persisted at state.json; use `auto-reviewer resume` to continue.",
    )


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
    _run_orchestrator(
        orchestrator,
        "Interrupted again. Re-run `auto-reviewer resume` to continue.",
    )


if __name__ == "__main__":
    cli()
