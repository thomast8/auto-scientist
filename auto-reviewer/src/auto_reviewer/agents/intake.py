"""Deterministic PR canonicalization for auto-reviewer.

Unlike auto-scientist's Ingestor, which needs an LLM to canonicalize
heterogeneous data (CSV / Excel / PDFs with ad-hoc schemas), a PR's "raw
data" is entirely machine-readable: `git diff base..head` + a bit of
metadata. An LLM agent with a 30-turn budget + retry loop is pure
overhead.

`run_intake` is a plain async function wrapping subprocess calls to `git`
(and `gh` when available). It has the same signature as auto-scientist's
`run_ingestor` so the shared orchestrator dispatches to it unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from pathlib import Path

from auto_core.notebook import NOTEBOOK_FILENAME, append_entry

from auto_reviewer.config import ReviewConfig

logger = logging.getLogger(__name__)


async def run_intake(
    raw_data_path: Path,
    output_dir: Path,
    goal: str,
    interactive: bool = False,
    config_path: Path | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
    provider: str = "anthropic",
) -> Path:
    """Canonicalize a PR into a review workspace.

    Signature mirrors `auto_scientist.agents.ingestor.run_ingestor` so the
    auto_core orchestrator's generic dispatch works unchanged. `model`,
    `provider`, and `interactive` are accepted and ignored - this
    canonicalizer does no LLM work.

    Expected state:
        * `config_path` points at a partially-filled ReviewConfig JSON that
          already carries `repo_path`, `pr_ref`, and `base_ref` (the CLI
          writes this before the orchestrator starts).
        * `raw_data_path` is the target repo root (filesystem path).
        * `output_dir` is the review workspace root; we write canonical
          artifacts to `{output_dir}/data/`.

    Writes:
        {output_dir}/data/diff.patch
        {output_dir}/data/pr_metadata.json
        {output_dir}/data/touched_files/<flattened-path>   (verbatim at head)
        {output_dir}/{config_path basename}                (refined config)
        {output_dir}/{lab notebook}                        (intake entry)

    Returns:
        Path to the canonical data directory.
    """
    # Keep the expensive-ish work off the event loop in case it grows.
    return await asyncio.to_thread(
        _canonicalize,
        raw_data_path=raw_data_path,
        output_dir=output_dir,
        goal=goal,
        config_path=config_path,
        message_buffer=message_buffer,
    )


def _canonicalize(
    *,
    raw_data_path: Path,
    output_dir: Path,
    goal: str,
    config_path: Path | None,
    message_buffer: list[str] | None,
) -> Path:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Load the seed ReviewConfig the CLI wrote. Without it we cannot know
    # which PR ref the caller meant.
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(
            f"Review config not found at {config_path}. "
            "The auto-reviewer CLI writes this before calling intake; "
            "if you invoked the orchestrator directly, seed it yourself."
        )
    seed = json.loads(config_path.read_text())
    pr_ref = seed.get("pr_ref")
    base_ref = seed.get("base_ref") or "main"
    repo_path = Path(seed.get("repo_path", raw_data_path))

    if not pr_ref:
        raise ValueError(f"Review config at {config_path} is missing pr_ref.")

    _emit(message_buffer, f"Canonicalizing PR {pr_ref} against {base_ref}...")

    pr_metadata = _load_pr_metadata(repo_path, pr_ref, base_ref)
    diff_text = _load_diff(repo_path, pr_ref, base_ref, pr_metadata)
    changed_files = _load_changed_files(repo_path, pr_ref, base_ref)

    (data_dir / "diff.patch").write_text(diff_text)
    (data_dir / "pr_metadata.json").write_text(json.dumps(pr_metadata, indent=2))

    touched_dir = data_dir / "touched_files"
    touched_dir.mkdir(exist_ok=True)
    head_ref = pr_metadata.get("headRefName") or pr_ref
    for rel in changed_files:
        _capture_file_at_head(repo_path, rel, head_ref, touched_dir)

    # Refine the review config with `head_ref` (if we learned it) and write
    # it back with `{script_path}` placeholder preserved in run_command.
    config = ReviewConfig.model_validate(
        {
            **seed,
            "head_ref": pr_metadata.get("headRefName") or seed.get("head_ref") or pr_ref,
            "run_command": seed.get("run_command") or "uv run pytest {script_path}",
        }
    )
    config_path.write_text(config.model_dump_json(indent=2))

    # One-line notebook entry describing scope and the PR title.
    notebook_path = output_dir / NOTEBOOK_FILENAME
    title = pr_metadata.get("title") or pr_ref
    n_files = len(changed_files)
    diff_lines = diff_text.count("\n")
    append_entry(
        notebook_path,
        content=(
            f"## Intake\n"
            f"- PR: {pr_ref} (title: {title})\n"
            f"- Base: {base_ref}, head: {config.head_ref}\n"
            f"- Files changed: {n_files}\n"
            f"- Diff lines: {diff_lines}\n"
            f"- Goal: {goal}\n"
        ),
        version="intake",
        source="intake",
    )

    _emit(
        message_buffer,
        f"Intake done. {n_files} touched files, {diff_lines} diff lines -> {data_dir}",
    )
    return data_dir


def _emit(buffer: list[str] | None, msg: str) -> None:
    """Log + append to the TUI buffer when one is supplied."""
    logger.info(msg)
    if buffer is not None:
        buffer.append(msg)


def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _load_pr_metadata(repo_path: Path, pr_ref: str, base_ref: str) -> dict:
    """Best-effort metadata pull.

    Tries `gh pr view <pr_ref> --json ...` first. Falls back to git log
    when gh is unavailable or the ref isn't a real PR on GitHub.
    """
    if _gh_available():
        try:
            out = subprocess.run(
                [
                    "gh",
                    "pr",
                    "view",
                    pr_ref,
                    "--json",
                    "title,body,url,author,baseRefName,headRefName",
                ],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=20,
            )
            meta: dict = json.loads(out.stdout)
            # gh returns author as {login, name}; flatten.
            if isinstance(meta.get("author"), dict):
                meta["author"] = meta["author"].get("login") or meta["author"].get("name")
            return meta
        except (subprocess.SubprocessError, subprocess.TimeoutExpired, json.JSONDecodeError):
            logger.debug("gh pr view failed for %s; falling back to git", pr_ref, exc_info=True)

    # Fallback: synthesize metadata from the tip commit on the PR branch.
    title = _git(repo_path, ["log", "-1", "--pretty=%s", pr_ref])
    body = _git(repo_path, ["log", "-1", "--pretty=%b", pr_ref])
    author = _git(repo_path, ["log", "-1", "--pretty=%an", pr_ref])
    return {
        "title": title,
        "body": body,
        "author": author,
        "url": None,
        "baseRefName": base_ref,
        "headRefName": pr_ref,
    }


def _load_diff(
    repo_path: Path,
    pr_ref: str,
    base_ref: str,
    pr_metadata: dict,
) -> str:
    """Prefer `gh pr diff`; fall back to `git diff`.

    Used ref pair comes from pr_metadata when gh provided one, else the
    caller-supplied (base_ref, pr_ref).
    """
    if _gh_available() and pr_metadata.get("url"):
        try:
            out = subprocess.run(
                ["gh", "pr", "diff", pr_ref],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return out.stdout
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            logger.debug(
                "gh pr diff failed for %s; falling back to git diff", pr_ref, exc_info=True
            )

    return _git(repo_path, ["diff", f"{base_ref}...{pr_ref}"])


def _load_changed_files(repo_path: Path, pr_ref: str, base_ref: str) -> list[str]:
    output = _git(repo_path, ["diff", "--name-only", f"{base_ref}...{pr_ref}"])
    return [line for line in output.splitlines() if line.strip()]


def _capture_file_at_head(
    repo_path: Path,
    rel: str,
    head_ref: str,
    touched_dir: Path,
) -> None:
    """Copy the head-ref version of `rel` under touched_dir, flattening its path."""
    try:
        content = _git(repo_path, ["show", f"{head_ref}:{rel}"])
    except subprocess.CalledProcessError:
        # Deleted in the PR - write a tombstone so the Surveyor sees it.
        (touched_dir / (rel.replace("/", "__") + ".DELETED")).write_text(
            f"(file deleted at {head_ref})\n"
        )
        return
    flat = rel.replace("/", "__")
    (touched_dir / flat).write_text(content)


def _git(repo_path: Path, args: list[str]) -> str:
    """Run a git subcommand and return stdout, raising on non-zero."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return result.stdout
