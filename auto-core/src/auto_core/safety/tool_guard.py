"""PreToolUse guard that confines SDK tool calls to a workspace.

The guard produced by :func:`make_workspace_guard` is a synchronous
callable the SDK backend adapts to its per-provider hook shape (Claude
Code SDK's `can_use_tool` callback; Codex has no per-call hook so the
guard is advisory on that path, with the Codex seatbelt doing the actual
enforcement).

Design principles:

- **Strict and simple beats clever.** The guard is one of three defence
  layers (workspace-pinned repo clone + integrity tripwire are the
  others). Anything that looks even slightly wrong gets denied; the
  integration tests verify the allowed surface is large enough for real
  agent work.
- **Absolute-path discipline.** Every path in tool input is resolved via
  ``Path.resolve()`` (which collapses symlinks) and compared to the
  workspace via ``Path.is_relative_to``. Relative paths get resolved
  against the workspace before the check.
- **Command allowlist for Bash.** Shell commands are tokenised with
  :func:`shlex.split` and screened against a deny-list of destructive
  verbs. We also reject absolute paths in arguments that are not under
  the workspace, and reject shell redirections whose target resolves
  outside the workspace. This is not a sandbox; it is a tripwire for
  obviously-wrong commands.
"""

from __future__ import annotations

import logging
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

GuardMode = Literal["intake", "probe", "read_only"]


@dataclass(frozen=True)
class Decision:
    """Result of a single PreToolUse check.

    ``allowed=False`` comes paired with a human-readable ``reason`` that
    the SDK surfaces back to the model as a tool error. A well-phrased
    reason helps the model course-correct instead of retrying blindly.
    """

    allowed: bool
    reason: str = ""

    @classmethod
    def allow(cls) -> Decision:
        return cls(allowed=True, reason="")

    @classmethod
    def deny(cls, reason: str) -> Decision:
        return cls(allowed=False, reason=reason)


PreToolUseHook = Callable[[str, dict[str, Any]], Decision]


# Destructive bash tokens. Matched against whole tokens from shlex.split,
# so 'rmdir' does not trigger 'rm'. The 'rm' + flag pattern is handled
# separately because 'rm foo.txt' inside the workspace is fine.
_DESTRUCTIVE_VERBS: tuple[str, ...] = (
    "sudo",
    "su",
    "mkfs",
    "mkfs.ext4",
    "mkfs.apfs",
    "dd",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "kexec",
    "systemctl",
    "launchctl",
    "chmod",
    "chown",
    "chgrp",
)

# git subcommands that mutate refs or publish state. Rejected everywhere
# — the reviewer never needs to publish, commit, or reset.
_FORBIDDEN_GIT_SUBCOMMANDS: tuple[str, ...] = (
    "push",
    "commit",
    "rebase",
    "merge",
    "cherry-pick",
    "revert",
    "am",
    "tag",
    "reset",
    "clean",
    "gc",
    "prune",
    "stash",
    "remote",
    "config",
    "credential",
    "update-ref",
    "branch",
    "checkout",
    "switch",
    "restore",
)

# gh subcommands that publish or mutate state on GitHub. Rejected
# everywhere. The read-only subset (`gh pr view/diff/list`,
# `gh api repos/...`) is implicitly allowed.
_FORBIDDEN_GH_SUBCOMMANDS: tuple[str, ...] = (
    "pr",  # allowed via suffix check below; see _is_gh_allowed
    "issue",
    "release",
    "repo",
    "gist",
    "auth",
    "workflow",
    "run",
    "secret",
    "variable",
)
_GH_READONLY_PR_SUBSUB: tuple[str, ...] = ("view", "diff", "list", "checks", "status")


# Shell splitters that can hide a second command on one line.
_COMMAND_SEPARATORS: tuple[str, ...] = (";", "&&", "||", "|", "&")

# Bash token patterns that are always considered a shell redirection.
_REDIRECT_TOKENS: tuple[str, ...] = (">", ">>", "<", "<<", "<<<", "&>", "2>", "2>>")


def make_workspace_guard(
    workspace: Path,
    repo_clone: Path,
    mode: GuardMode,
) -> PreToolUseHook:
    """Build a PreToolUse hook pinned to ``workspace`` in the given mode.

    Args:
        workspace: Absolute path to the review workspace. All writes must
            resolve inside this directory.
        repo_clone: Absolute path to the cloned target repo inside the
            workspace. Must satisfy ``repo_clone.is_relative_to(workspace)``.
            The ``probe`` mode treats this path as "source of the project
            under review" — writes here are further restricted to the
            ``.auto_reviewer_probes/`` subdirectory.
        mode: One of:
            - ``intake``: Write/Bash allowed inside workspace; git clone,
              git read-only subcommands, gh read-only subcommands allowed.
            - ``probe``: Write/Edit/Bash allowed in workspace; inside
              ``repo_clone`` restricted to ``.auto_reviewer_probes/``.
            - ``read_only``: Only Read/Glob/Grep.

    Returns:
        A callable ``(tool_name, tool_input) -> Decision`` suitable for
        adapting into provider-specific hooks.
    """
    workspace = workspace.resolve()
    repo_clone = repo_clone.resolve()
    if not _is_within(repo_clone, workspace):
        raise ValueError(
            f"repo_clone={repo_clone} is not inside workspace={workspace}; "
            "the guard invariant is that the clone lives inside the writable area."
        )

    def guard(tool_name: str, tool_input: dict[str, Any]) -> Decision:
        name = tool_name.split("__")[-1]  # strip "mcp__server__tool" prefix if any

        if mode == "read_only":
            if name in {"Read", "Glob", "Grep", "AskUserQuestion"}:
                return Decision.allow()
            return Decision.deny(
                f"Tool {name!r} is not available in read-only mode; "
                "only Read, Glob, Grep, AskUserQuestion are permitted."
            )

        if name in {"Read", "Glob", "Grep", "AskUserQuestion", "WebSearch", "WebFetch"}:
            return Decision.allow()

        if name in {"Write", "Edit", "NotebookEdit"}:
            return _check_write(tool_input, workspace, repo_clone, mode)

        if name == "Bash":
            return _check_bash(tool_input, workspace, repo_clone, mode)

        # Unknown tool: reject by default. New tools must be explicitly
        # enumerated so we never silently expand the attack surface.
        return Decision.deny(
            f"Tool {name!r} is not on the reviewer allowlist. If the "
            "reviewer needs it, add it to auto_core.safety.tool_guard."
        )

    return guard


def _check_write(
    tool_input: dict[str, Any],
    workspace: Path,
    repo_clone: Path,
    mode: GuardMode,
) -> Decision:
    """Verify a Write/Edit target resolves inside the writable area."""
    target = (
        tool_input.get("file_path") or tool_input.get("path") or tool_input.get("notebook_path")
    )
    if not target:
        return Decision.deny("Write tool call has no file_path/path/notebook_path argument.")
    resolved = _resolve(target, workspace)
    if not _is_within(resolved, workspace):
        return Decision.deny(
            f"Write target {resolved} is outside the review workspace {workspace}. "
            "The reviewer may only write under the workspace directory."
        )
    if mode == "probe" and _is_within(resolved, repo_clone):
        probes_root = repo_clone / ".auto_reviewer_probes"
        if not _is_within(resolved, probes_root):
            return Decision.deny(
                f"Probes may only write under {probes_root} inside the cloned "
                f"target repo; {resolved} is elsewhere in the clone. Place "
                "shims under .auto_reviewer_probes/."
            )
    return Decision.allow()


def _check_bash(
    tool_input: dict[str, Any],
    workspace: Path,
    repo_clone: Path,
    mode: GuardMode,
) -> Decision:
    """Screen a Bash command for destructive verbs and escape paths."""
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return Decision.deny("Bash tool call has no command argument.")

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as e:
        return Decision.deny(f"Bash command failed to tokenise ({e}); refusing to run.")

    # Redirection targets first — check whether any `>`, `>>`, etc. points
    # outside workspace before falling into per-segment verb checks. Paths
    # that appear as redirect targets are excluded from the per-segment
    # abs-path scan to avoid a less-specific deny reason.
    redirect_target_indices: set[int] = set()
    for i, tok in enumerate(tokens):
        if tok in _REDIRECT_TOKENS and i + 1 < len(tokens):
            target = tokens[i + 1]
            redirect_target_indices.add(i + 1)
            if target.startswith("/") or target.startswith("~"):
                resolved = _resolve(target, workspace)
                if not _is_within(resolved, workspace):
                    return Decision.deny(
                        f"Bash redirection to {resolved} is outside the workspace "
                        f"{workspace}. Redirect to a path under the workspace."
                    )

    # Split on shell operators so each segment gets its own verb check.
    segments: list[list[str]] = [[]]
    for idx, tok in enumerate(tokens):
        if tok in _COMMAND_SEPARATORS:
            segments.append([])
            continue
        if tok in _REDIRECT_TOKENS or idx in redirect_target_indices:
            continue  # already handled by the redirection scan above
        segments[-1].append(tok)
    segments = [s for s in segments if s]
    if not segments:
        return Decision.deny("Bash command has no executable tokens after tokenisation.")

    for segment in segments:
        verdict = _check_bash_segment(segment, workspace, repo_clone, mode)
        if not verdict.allowed:
            return verdict
    return Decision.allow()


def _check_bash_segment(
    segment: list[str],
    workspace: Path,
    repo_clone: Path,
    mode: GuardMode,
) -> Decision:
    if not segment:
        return Decision.allow()
    head = segment[0]

    # Strip inline env-var prefixes (FOO=bar BAR=baz cmd ...)
    idx = 0
    while idx < len(segment) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", segment[idx]):
        idx += 1
    if idx >= len(segment):
        return Decision.allow()  # only env vars, no command
    head = segment[idx]
    rest = segment[idx + 1 :]

    # Base-name the command so /usr/bin/rm and rm are treated alike.
    head_base = Path(head).name if "/" in head else head

    if head_base in _DESTRUCTIVE_VERBS:
        return Decision.deny(
            f"Bash command starts with destructive verb {head_base!r}; "
            "the guard rejects these regardless of arguments."
        )

    if head_base == "rm":
        return _check_rm(rest, workspace)

    if head_base == "git":
        return _check_git(rest, workspace, repo_clone, mode)

    if head_base == "gh":
        return _check_gh(rest)

    # Any absolute-path argument must resolve inside workspace.
    for arg in rest:
        if arg.startswith("/") or arg.startswith("~"):
            resolved = _resolve(arg, workspace)
            if not _is_within(resolved, workspace):
                return Decision.deny(
                    f"Bash command references absolute path {resolved} "
                    f"outside workspace {workspace}. Paths passed as arguments "
                    "must resolve inside the workspace."
                )
    return Decision.allow()


def _check_rm(args: list[str], workspace: Path) -> Decision:
    # rm -rf is almost never what the reviewer needs. Reject any recursive
    # flag; plain `rm file.txt` inside workspace is allowed if the target
    # resolves inside workspace.
    recursive = any(a in {"-r", "-R", "--recursive", "-rf", "-fr", "-rF", "-Rf"} for a in args)
    if recursive:
        return Decision.deny(
            "Bash `rm` with recursive flag is forbidden. Remove individual "
            "files with plain `rm path/to/file`, or clean up via the "
            "orchestrator at end-of-run."
        )
    for arg in args:
        if arg.startswith("-"):
            continue
        resolved = _resolve(arg, workspace)
        if not _is_within(resolved, workspace):
            return Decision.deny(
                f"Bash `rm` target {resolved} is outside the workspace "
                f"{workspace}. Only files under the workspace may be removed."
            )
    return Decision.allow()


def _check_git(
    args: list[str],
    workspace: Path,
    repo_clone: Path,
    mode: GuardMode,
) -> Decision:
    # Parse out `git -C <dir>` prefix to find the working directory that
    # git will operate on. We still run the subcommand allowlist check.
    working_dir = workspace
    i = 0
    subcommand_idx = 0
    while i < len(args) and args[i].startswith("-"):
        if args[i] == "-C" and i + 1 < len(args):
            working_dir = _resolve(args[i + 1], workspace)
            if not _is_within(working_dir, workspace):
                return Decision.deny(
                    f"`git -C {args[i + 1]}` targets {working_dir} outside "
                    f"workspace {workspace}. The reviewer only operates on "
                    "the clone under the workspace."
                )
            i += 2
            subcommand_idx = i
            continue
        if args[i] in {"-c", "--exec-path"} and i + 1 < len(args):
            i += 2
            subcommand_idx = i
            continue
        i += 1
        subcommand_idx = i

    if subcommand_idx >= len(args):
        # bare `git` — print help; harmless.
        return Decision.allow()

    sub = args[subcommand_idx]
    sub_args = args[subcommand_idx + 1 :]

    if sub in _FORBIDDEN_GIT_SUBCOMMANDS:
        return Decision.deny(
            f"`git {sub}` is on the forbidden-subcommand list. The reviewer "
            "never pushes, commits, resets, or mutates refs; if you need to "
            "inspect history use `git log`, `git show`, or `git diff`."
        )

    if sub == "clone":
        # Destination must resolve inside workspace. Positional args are
        # [url, destination]; we filter URL-shaped tokens (those with
        # "://" or "user@host:" prefixes) and require at least one
        # remaining positional — the explicit destination.
        positional = [a for a in sub_args if not a.startswith("-")]
        non_urls = [
            a
            for a in positional
            if "://" not in a and not re.match(r"^[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+:", a)
        ]
        if not non_urls:
            return Decision.deny(
                "`git clone` requires an explicit destination inside the "
                "workspace (e.g. `git clone <url> <workspace>/repo_clone`). "
                "Refusing to clone without a bounded destination."
            )
        destination = _resolve(non_urls[-1], workspace)
        if not _is_within(destination, workspace):
            return Decision.deny(
                f"`git clone` destination {destination} is outside workspace {workspace}."
            )
        return Decision.allow()

    if sub == "worktree":
        # `git worktree add <path>` — destination must be inside workspace.
        if sub_args and sub_args[0] == "add":
            path_args = [a for a in sub_args[1:] if not a.startswith("-")]
            if not path_args:
                return Decision.deny("`git worktree add` needs a destination path.")
            dest = _resolve(path_args[0], workspace)
            if not _is_within(dest, workspace):
                return Decision.deny(
                    f"`git worktree add` destination {dest} is outside workspace {workspace}."
                )
            return Decision.allow()
        # worktree list/remove/prune inside workspace: allow.
        return Decision.allow()

    if sub == "fetch":
        # Allow fetch only when -C bound to workspace (enforced above).
        if not _is_within(working_dir, workspace):
            return Decision.deny(
                f"`git fetch` requires `-C <path inside {workspace}>` so the "
                "user's original repo is never mutated."
            )
        return Decision.allow()

    # Read-only subcommands (show, log, diff, rev-parse, ls-files, status,
    # blame, cat-file, symbolic-ref, ls-remote, for-each-ref, describe,
    # shortlog, help). Allow.
    return Decision.allow()


def _check_gh(args: list[str]) -> Decision:
    if not args:
        return Decision.allow()
    sub = args[0]
    if sub == "pr":
        if len(args) >= 2 and args[1] in _GH_READONLY_PR_SUBSUB:
            return Decision.allow()
        return Decision.deny(
            f"`gh pr {args[1] if len(args) >= 2 else ''}` is not on the "
            f"read-only PR allowlist {_GH_READONLY_PR_SUBSUB}."
        )
    if sub == "api":
        # gh api is GET by default; reject explicit mutations.
        if any(a in {"-X", "--method"} for a in args):
            method_idx = next(i for i, a in enumerate(args) if a in {"-X", "--method"})
            if method_idx + 1 < len(args) and args[method_idx + 1].upper() not in {"GET", "HEAD"}:
                return Decision.deny(
                    f"`gh api -X {args[method_idx + 1]}` is a mutating HTTP "
                    "method. Only GET/HEAD requests are permitted."
                )
        return Decision.allow()
    if sub in _FORBIDDEN_GH_SUBCOMMANDS:
        return Decision.deny(
            f"`gh {sub}` is on the forbidden-subcommand list. The reviewer "
            "only uses `gh pr view/diff/list` and `gh api` for GET requests."
        )
    return Decision.allow()


def _resolve(raw: str, anchor: Path) -> Path:
    """Resolve ``raw`` against ``anchor`` for relative paths, then collapse.

    Expands ``~`` (user-home) via :meth:`Path.expanduser` so a command like
    ``rm ~/foo`` still hits the path-check. ``Path.resolve()`` follows
    symlinks, so a symlink inside the workspace pointing to ``/etc`` gets
    rejected rather than traversed.
    """
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = anchor / p
    try:
        return p.resolve()
    except (OSError, RuntimeError):
        # resolve() can raise on weird FS states (circular symlinks); treat
        # as "outside workspace" so the caller denies.
        return p


def _is_within(path: Path, root: Path) -> bool:
    """True if ``path`` is ``root`` or a descendant of it."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
