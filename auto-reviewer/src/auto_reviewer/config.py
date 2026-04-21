"""Review configuration schema.

Extends `auto_core.config.RunConfig` with review-specific fields: the repo
path the PR targets, the PR ref, and the base ref. Operational fields
(run_command, run_cwd, run_timeout_minutes, version_prefix, protected_paths)
live in the base class.
"""

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
