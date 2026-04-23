"""Review state types for auto-reviewer.

The generic shape comes from `auto_core.state.RunState` / `ProbeEntry` /
`SuspectedBug`. This module just re-exports the review-oriented aliases so
call sites in auto-reviewer read naturally.
"""

# Re-export review-oriented aliases from the shared runtime.
from auto_core.state import (  # noqa: F401
    ProbeEntry,
    SuspectedBug,
)
from auto_core.state import RunState as ReviewState  # noqa: F401
