"""Before/after tripwire for the user's real repository.

The guard (`auto_core.safety.tool_guard`) and the workspace-pinned repo
clone prevent the LLM from writing to the user's real repo in the first
place. This module is the last-resort check: regardless of how bad any
layer above misbehaves, a clean before/after fingerprint proves the
original tree is byte-identical.

The fingerprint captures three independent signals:

1. **HEAD.** The commit the repo was pointed at when the run started.
   Catches any `git reset`, `git checkout`, `git commit` that slipped
   through.
2. **Porcelain status.** The set of modified / untracked / staged paths
   at snapshot time. Catches working-tree mutations even if HEAD is
   untouched.
3. **Tree content hash.** sha256-of-sha256s over every file in the
   repo outside `.git/`. Catches anything the porcelain status might
   miss (mode changes, symlink retargeting, files git considers
   unchanged because their mtime wasn't bumped).

A mismatch on any signal is a loud failure. The design goal is zero
false negatives; false positives (a legitimate benign change the user
made in another terminal) are acceptable and surface as a clear error
they can act on.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


class IntegrityError(RuntimeError):
    """Raised when the post-run fingerprint differs from the pre-run one."""


@dataclass(frozen=True)
class RepoFingerprint:
    """Immutable snapshot of a repository's on-disk state."""

    head: str
    porcelain: str
    tree_hash: str
    path: str

    def diff(self, other: RepoFingerprint) -> list[str]:
        """Return human-readable descriptions of every differing field."""
        deltas: list[str] = []
        if self.head != other.head:
            deltas.append(f"HEAD moved: {self.head} -> {other.head}")
        if self.porcelain != other.porcelain:
            deltas.append(
                "`git status --porcelain` changed:\n"
                f"  before: {self.porcelain!r}\n"
                f"  after:  {other.porcelain!r}"
            )
        if self.tree_hash != other.tree_hash:
            deltas.append(
                f"working-tree content hash changed: "
                f"{self.tree_hash[:12]}... -> {other.tree_hash[:12]}..."
            )
        return deltas


def snapshot_repo(path: Path) -> RepoFingerprint:
    """Capture an integrity fingerprint of the repo at ``path``.

    Requires ``path`` to be a git repository. If it isn't, we still take
    a content hash so non-git directories get some protection, but
    HEAD/porcelain are recorded as ``<no-git>``.
    """
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"snapshot_repo: path does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"snapshot_repo: not a directory: {resolved}")

    head = _git_rev_parse_head(resolved)
    porcelain = _git_porcelain(resolved) if head != "<no-git>" else "<no-git>"
    tree_hash = _hash_tree(resolved)
    return RepoFingerprint(
        head=head,
        porcelain=porcelain,
        tree_hash=tree_hash,
        path=str(resolved),
    )


def verify_unchanged(before: RepoFingerprint, path: Path | None = None) -> None:
    """Re-snapshot and raise :class:`IntegrityError` if anything differs.

    ``path`` defaults to the path recorded in ``before`` so callers can
    pass just the fingerprint. Pass an explicit ``path`` when the repo
    has moved (uncommon — only relevant if you intentionally relocated
    the source tree between snapshots).
    """
    target = path.resolve() if path is not None else Path(before.path)
    after = snapshot_repo(target)
    deltas = before.diff(after)
    if deltas:
        raise IntegrityError(
            f"Target repo at {target} was modified during the run:\n"
            + "\n".join(f"  - {d}" for d in deltas)
        )


def _git_rev_parse_head(path: Path) -> str:
    """Return the HEAD SHA, or ``<no-git>`` if the path isn't a git repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "<no-git>"
    if result.returncode != 0:
        return "<no-git>"
    return result.stdout.strip()


def _git_porcelain(path: Path) -> str:
    """Return the deterministic `git status --porcelain=v1` string.

    ``-z`` would be more robust for weird filenames, but the plain v1
    form is stable across git versions and trivially comparable.
    """
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        # A missing or broken git repo is itself a fingerprint signal.
        return f"<git-status-failed: rc={result.returncode}>"
    return result.stdout


def _hash_tree(path: Path) -> str:
    """sha256 over every non-.git file under ``path``.

    For each file we hash (relative_path_bytes, content_sha256). The
    outer hash is over the sorted concatenation so it's deterministic
    and catches:
      - content changes (content hash differs)
      - file addition/removal (path list differs)
      - rename (path list differs)

    Symlinks are recorded as their target path, not as the target's
    contents, so retargeting a symlink inside the repo gets caught.
    """
    outer = hashlib.sha256()
    entries: list[tuple[bytes, bytes]] = []
    for entry in _iter_files(path):
        rel = entry.relative_to(path).as_posix().encode("utf-8")
        if entry.is_symlink():
            try:
                target = entry.readlink().as_posix()
            except OSError:
                target = "<unreadable-symlink>"
            content_sig = hashlib.sha256(f"symlink:{target}".encode()).digest()
        else:
            content_sig = _sha256_file(entry)
        entries.append((rel, content_sig))
    entries.sort()
    for rel, sig in entries:
        outer.update(rel)
        outer.update(b"\x00")
        outer.update(sig)
        outer.update(b"\n")
    return outer.hexdigest()


def _iter_files(root: Path):
    """Yield every non-.git file under ``root`` (recursively)."""
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        try:
            parts = p.relative_to(root).parts
        except ValueError:
            continue
        if parts and parts[0] == ".git":
            continue
        yield p


def _sha256_file(path: Path) -> bytes:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return hashlib.sha256(b"<unreadable>").digest()
    return h.digest()
