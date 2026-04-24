"""Review configuration schema.

Extends `auto_core.config.RunConfig` with review-specific fields: the repo
path the PR targets, the PR ref, and the base ref. Operational fields
(run_command, run_cwd, run_timeout_minutes, version_prefix, protected_paths)
live in the base class.
"""

from pathlib import Path

from auto_core.config import RunConfig


class ReviewConfig(RunConfig):
    """Operational configuration for a review run.

    Review concerns (repo_knowledge, prediction_history / suspected bugs)
    live in `RunState`; this carries only runtime / infrastructure settings.
    """

    repo_path: str
    pr_ref: str | None = None
    base_ref: str | None = None
    head_ref: str | None = None

    def require_inside_workspace(self, workspace: Path) -> None:
        """Raise if ``repo_path`` resolves outside ``workspace``.

        The sandbox invariant is that the clone the reviewer operates on
        lives inside the review workspace. If Intake (or a resumed state
        file) sets ``repo_path`` to the user's real repo, downstream
        agents could `cd` there and the Codex seatbelt would no longer
        protect the original. Calling this right after
        ``ReviewConfig.model_validate`` fails the run before any LLM
        sees the bad value.
        """
        workspace_resolved = workspace.resolve()
        repo_resolved = Path(self.repo_path).resolve()
        try:
            repo_resolved.relative_to(workspace_resolved)
        except ValueError as e:
            raise ValueError(
                f"ReviewConfig.repo_path ({repo_resolved}) is not inside "
                f"workspace ({workspace_resolved}). The reviewer only "
                "operates on clones inside the workspace; pointing at the "
                "user's original repo would bypass the sandbox."
            ) from e
