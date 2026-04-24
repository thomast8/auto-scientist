"""End-to-end sandbox smoke on Haiku.

Builds a throwaway git repo, runs the full reviewer pipeline (Intake →
Surveyor → Hunter → Prober → Findings) with every agent pinned to
claude-haiku-4-5, verifies the real repo is byte-identical after.

Uses the Orchestrator directly (not PipelineApp) so the process exits
cleanly when the pipeline finishes — the TUI keeps its own event loop
running past the pipeline's end and made earlier runs look hung.

Budget ~10 minutes of wall-clock on Haiku; hard cap 20 minutes via
SIGALRM. Cheap in dollar terms but still a real LLM run — not for CI.

For the narrow sandbox test (just the Prober + guard path), use
`smoke_prober.py` which runs in ~1 minute.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

GUARD_LOG = Path("/tmp/smoke_full_pipeline_guard.log")
GUARD_LOG.unlink(missing_ok=True)
logging.basicConfig(
    filename=str(GUARD_LOG),
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

import auto_reviewer.cli  # noqa: F401, E402 — installs SDK monkey-patch
from auto_core.model_config import AgentModelConfig, ModelConfig  # noqa: E402
from auto_core.orchestrator import Orchestrator  # noqa: E402
from auto_core.state import RunState  # noqa: E402
from auto_reviewer.prep import pre_resolve  # noqa: E402
from auto_reviewer.safety.integrity import IntegrityError, verify_unchanged  # noqa: E402

TIMEBOX_SECONDS = 1200


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


def _build_throwaway_repo(root: Path) -> Path:
    repo = root / "real_repo"
    repo.mkdir()
    _run(["git", "init", "-q", "-b", "main"], cwd=repo)
    _run(["git", "config", "user.email", "smoke@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Smoke"], cwd=repo)
    _run(["git", "config", "commit.gpgsign", "false"], cwd=repo)
    (repo / "README.md").write_text("alpha\n")
    (repo / "prod.py").write_text("def f(x):\n    return x + 1\n")
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "init"], cwd=repo)
    # A second commit so there's a branch to review.
    _run(["git", "checkout", "-q", "-b", "feat/tweak"], cwd=repo)
    (repo / "prod.py").write_text("def f(x):\n    return x + 2\n")
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "bump"], cwd=repo)
    return repo


def _haiku_config() -> ModelConfig:
    haiku = AgentModelConfig.model_validate(
        {
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "reasoning": {"level": "off"},
            "mode": "sdk",
        }
    )
    return ModelConfig(
        defaults=haiku,
        critics=[],
        summarizer=None,
    )


async def _run_orch(orch: Orchestrator) -> None:
    await orch.run()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="smoke_full_pipeline_") as td:
        root = Path(td)
        real_repo = _build_throwaway_repo(root)
        workspace = root / "workspace"

        resolved = pre_resolve(real_repo, workspace)
        assert resolved.repo_clone is not None

        state = RunState(
            domain="smoke",
            goal="review the feat/tweak branch; one iteration only",
            phase="ingestion",
            max_iterations=1,
            config_path=str(workspace / "domain_config.json"),
            data_path=str(resolved.repo_clone),
        )
        orch = Orchestrator(
            state=state,
            data_path=resolved.repo_clone,
            output_dir=workspace,
            max_iterations=1,
            model_config=_haiku_config(),
            verbose=True,
        )

        def _alarm(_signum, _frame):
            raise TimeoutError(f"smoke exceeded {TIMEBOX_SECONDS}s")

        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(TIMEBOX_SECONDS)

        try:
            asyncio.run(_run_orch(orch))
        except TimeoutError as e:
            print(f"[smoke_full_pipeline] {e}", file=sys.stderr, flush=True)
            return 2
        finally:
            signal.alarm(0)

        try:
            verify_unchanged(resolved.fingerprint)
        except IntegrityError as e:
            print(
                f"[smoke_full_pipeline] SANDBOX VIOLATION: {e}",
                file=sys.stderr,
                flush=True,
            )
            return 3

        print(
            f"[smoke_full_pipeline] OK — real repo byte-identical at {real_repo}",
            flush=True,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
