"""Reviewer-only safety primitives (orchestrator-owned state).

The shared guard lives in `auto_core.safety`; anything here is reviewer-
specific because the auto-scientist loop doesn't need it (it works on
`experiments/` that it creates, never on a user-supplied repo).
"""

from auto_reviewer.safety.integrity import (
    IntegrityError,
    RepoFingerprint,
    snapshot_repo,
    verify_unchanged,
)

__all__ = [
    "IntegrityError",
    "RepoFingerprint",
    "snapshot_repo",
    "verify_unchanged",
]
