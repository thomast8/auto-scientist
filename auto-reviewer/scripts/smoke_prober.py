"""Prober-only sandbox smoke. Skips Intake/Analyst/Hunter — fabricates
the inputs the Prober expects and runs it directly.

Tests the exact piece of the pipeline that needs the sandbox: the
code-running agent with Bash/Write/Edit under the probe-mode guard.
Runs in 1-2 minutes on Haiku instead of 6+ for the full pipeline.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GUARD_LOG = Path("/tmp/smoke_prober_guard.log")
GUARD_LOG.unlink(missing_ok=True)
logging.basicConfig(
    filename=str(GUARD_LOG),
    filemode="w",
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
# Silence noisy libraries at DEBUG
for noisy in ("httpx", "httpcore", "anyio", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

import asyncio  # noqa: E402

import auto_reviewer.cli  # noqa: F401, E402 — installs monkey-patch
from auto_reviewer.agents.prober import run_prober  # noqa: E402
from auto_reviewer.prep import pre_resolve  # noqa: E402
from auto_reviewer.safety.integrity import IntegrityError, verify_unchanged  # noqa: E402

TIMEBOX = 300


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


def _build_throwaway_repo(root: Path) -> Path:
    repo = root / "real_repo"
    repo.mkdir()
    _run(["git", "init", "-q", "-b", "main"], cwd=repo)
    _run(["git", "config", "user.email", "smoke@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Smoke"], cwd=repo)
    _run(["git", "config", "commit.gpgsign", "false"], cwd=repo)
    # A tiny pytest-testable module with a deliberate bug for the
    # Prober to (try to) characterize.
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "throwaway"\nversion = "0.0.0"\nrequires-python = ">=3.12"\n'
    )
    (repo / "mymod.py").write_text(
        "def f(x):\n    # BUG: subtracts instead of adds.\n    return x - 1\n"
    )
    (repo / "test_basic.py").write_text(
        "from mymod import f\n\ndef test_positive():\n    assert f(1) == 2\n"
    )
    (repo / "CRITICAL.txt").write_text("precious\n")
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "init"], cwd=repo)
    return repo


def _write_prober_inputs(workspace: Path, version_dir: Path, clone: Path) -> None:
    domain_config = {
        "name": "smoke",
        "description": "Prober-only smoke",
        "repo_path": str(clone),
        "pr_ref": "main",
        "base_ref": "main",
        "head_ref": "main",
        "run_cwd": str(clone),
        "run_command": "uv run pytest -x -s {script_path}",
        "run_timeout_minutes": 5,
        "version_prefix": "v",
        "protected_paths": [],
    }
    (workspace / "domain_config.json").write_text(json.dumps(domain_config))

    plan = {
        "suspected_bug": (
            "The function mymod.f is documented to increment by 1 but appears to decrement by 1."
        ),
        "testable_predictions": [
            {
                "pred_id": "p1",
                "claim": "f(1) returns 2 (expected behaviour).",
                "experiment": "Import mymod.f from the target repo and "
                "assert f(1) == 2 using pytest.",
                "expected_outcome": "passes if f adds; fails if f subtracts.",
            }
        ],
    }
    (version_dir / "plan.json").write_text(json.dumps(plan))

    # previous_script placeholder (v00 from a prior iteration); path must
    # exist but can be empty.
    prev_dir = workspace / "v00"
    prev_dir.mkdir(parents=True, exist_ok=True)
    prev_script = prev_dir / "run_result.json"
    prev_script.write_text("{}")


def main() -> int:
    td = Path(tempfile.mkdtemp(prefix="smoke_prober_"))
    try:
        real_repo = _build_throwaway_repo(td)
        workspace = td / "workspace"
        resolved = pre_resolve(real_repo, workspace)
        assert resolved.repo_clone is not None

        version = "v01"
        version_dir = workspace / version
        version_dir.mkdir(parents=True, exist_ok=True)
        _write_prober_inputs(workspace, version_dir, resolved.repo_clone)

        t0 = time.time()
        print(f"[smoke_prober] starting run_prober (clone={resolved.repo_clone})", flush=True)

        plan = json.loads((version_dir / "plan.json").read_text())
        prev_script = workspace / "v00" / "run_result.json"

        async def go() -> Path:
            return await run_prober(
                plan=plan,
                previous_script=prev_script,
                output_dir=workspace,
                version=version,
                domain_knowledge="throwaway",
                data_path=str(resolved.repo_clone),
                model="claude-haiku-4-5-20251001",
                run_timeout_minutes=5,
                run_command="uv run pytest -x -s {script_path}",
                provider="anthropic",
            )

        try:
            result_path = asyncio.run(asyncio.wait_for(go(), timeout=TIMEBOX))
        except TimeoutError:
            print(f"[smoke_prober] TIMEOUT after {TIMEBOX}s", file=sys.stderr, flush=True)
            return 2

        elapsed = time.time() - t0
        print(f"[smoke_prober] run_prober finished in {elapsed:.1f}s", flush=True)

        if result_path.exists():
            print(f"[smoke_prober] wrote {result_path}:", flush=True)
            print(result_path.read_text(), flush=True)

        # The critical assertion: real repo untouched.
        try:
            verify_unchanged(resolved.fingerprint)
        except IntegrityError as e:
            print(f"[smoke_prober] SANDBOX VIOLATION: {e}", file=sys.stderr, flush=True)
            return 3

        denies = sum(1 for line in GUARD_LOG.read_text().splitlines() if "guard deny:" in line)
        print(f"[smoke_prober] guard denies: {denies}", flush=True)
        print(f"[smoke_prober] OK — real repo at {real_repo} byte-identical", flush=True)
        return 0
    finally:
        # keep the tmp dir on disk if non-zero exit for post-mortem
        pass


if __name__ == "__main__":
    sys.exit(main())
