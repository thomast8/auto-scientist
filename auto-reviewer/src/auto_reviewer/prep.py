"""Pre-Intake resolution: everything that touches the user's real repo
happens here (non-LLM Python code), so downstream agents never see the
original path and cannot write to it.

Flow:

1. Resolve the user's `--cwd` to a git top-level when possible, then
   fingerprint that source so we can check at the end of the run that
   the original is byte-identical.
2. If the source is a local git repo, clone it into
   ``<workspace>/repo_clone/`` via ``git clone --local``. The clone is
   a fully independent repository; even if the LLM defeats every other
   layer and writes inside ``repo_clone``, the user's original is
   unaffected.
3. Write ``<workspace>/data/cwd_hint.json`` with the metadata Intake
   needs to resolve natural-language pointers (remotes, current branch,
   HEAD ref). The LLM reads this JSON rather than the user's real
   filesystem.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from auto_reviewer.safety.integrity import RepoFingerprint, snapshot_repo

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreResolved:
    """Outcome of the pre-resolution step.

    Attributes:
        fingerprint: Fingerprint of the user's source repo or cwd at run start.
            Pass to ``verify_unchanged`` at end of run.
        repo_clone: Workspace-internal clone path, or ``None`` if the
            source wasn't a git repo (Intake will have to clone from a
            remote URL).
        hint_path: Path to the JSON file with repo metadata Intake reads.
    """

    fingerprint: RepoFingerprint
    repo_clone: Path | None
    hint_path: Path


def pre_resolve(cwd: Path, workspace: Path) -> PreResolved:
    """Run the pre-resolution step for a review.

    Args:
        cwd: The user's `--cwd`. Its git top-level is snapshotted when
            available; otherwise cwd itself is snapshotted.
        workspace: The review workspace. The clone (if any) lands under
            `workspace/repo_clone/`, the hint file under `workspace/data/`.
    """
    cwd = cwd.expanduser().resolve()
    workspace = workspace.expanduser().resolve()
    source_root = _find_git_toplevel(cwd)
    source_path = source_root or cwd
    if _is_within(workspace, source_path):
        raise RuntimeError(
            f"Review workspace {workspace} is inside the source path {source_path}. "
            "Choose an --output-dir outside the repository being reviewed."
        )

    workspace.mkdir(parents=True, exist_ok=True)
    data_dir = workspace / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = snapshot_repo(source_path)
    is_git = source_root is not None

    clone_path: Path | None = None
    hint: dict[str, object] = {
        "is_git": is_git,
    }

    if is_git:
        clone_path = workspace / "repo_clone"
        if clone_path.exists():
            raise RuntimeError(
                f"pre_resolve: {clone_path} already exists; refusing to "
                "overwrite. Point --output-dir at a fresh workspace."
            )
        _clone_local(source_path, clone_path)
        hint.update(_collect_git_hint(source_path))
        hint["repo_clone"] = str(clone_path)

    hint_path = data_dir / "cwd_hint.json"
    hint_path.write_text(json.dumps(hint, indent=2, sort_keys=True))
    logger.info(
        "pre_resolve: workspace=%s is_git=%s clone=%s",
        workspace,
        is_git,
        clone_path,
    )
    return PreResolved(fingerprint=fingerprint, repo_clone=clone_path, hint_path=hint_path)


def _clone_local(source: Path, destination: Path) -> None:
    """`git clone --local` into ``destination``.

    --local hardlinks ``.git/objects`` when source and destination share a
    filesystem, which is fast and doesn't risk mutating the source:
    git objects are content-addressable and immutable, so hardlinks are
    read-only for integrity purposes. The clone gets its own HEAD / refs /
    working tree, fully independent of the original.
    """
    result = subprocess.run(
        ["git", "clone", "--local", "--no-hardlinks", str(source), str(destination)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git clone --local {source} -> {destination} failed "
            f"(rc={result.returncode}): {result.stderr.strip()}"
        )


def _find_git_toplevel(path: Path) -> Path | None:
    """Return the containing git repo's top-level directory, if any."""
    raw = _git(path, ["rev-parse", "--show-toplevel"])
    return Path(raw).resolve() if raw else None


def _collect_git_hint(path: Path) -> dict[str, object]:
    """Gather repo metadata Intake uses to resolve natural-language pointers."""
    hint: dict[str, object] = {}
    hint["head_sha"] = _git(path, ["rev-parse", "HEAD"]) or ""
    hint["current_branch"] = _git(path, ["symbolic-ref", "--short", "-q", "HEAD"]) or "(detached)"
    remotes_raw = _git(path, ["remote", "-v"]) or ""
    hint["remotes"] = _parse_remotes(remotes_raw)
    return hint


def _git(path: Path, args: list[str]) -> str | None:
    """Run ``git`` at ``path`` and return stdout, or ``None`` on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _parse_remotes(raw: str) -> list[dict[str, str]]:
    """Turn `git remote -v` output into JSON-friendly entries."""
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, url = parts[0], parts[1]
        key = (name, url)
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "url": _sanitize_remote_url(url)})
    return out


def _sanitize_remote_url(url: str) -> str:
    """Strip credentials from remote URLs before writing LLM-readable hints."""
    split = urlsplit(url)
    if split.scheme and split.netloc and "@" in split.netloc:
        host = split.hostname or ""
        if split.port is not None:
            host = f"{host}:{split.port}"
        return urlunsplit((split.scheme, host, split.path, split.query, split.fragment))

    slash_index = url.find("/")
    at_index = url.find("@")
    if at_index != -1 and (slash_index == -1 or at_index < slash_index):
        user = url[:at_index]
        if user != "git":
            return url[at_index + 1 :]
    return url


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
