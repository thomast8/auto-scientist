"""Run state persistence and crash recovery.

Shared by every app that runs on the auto_core orchestrator. The shape is
deliberately agnostic to the domain (science vs. PR review etc.):

    * `domain`             -> app-specific name / identifier (SpO2 dataset for
                              auto-scientist, PR ref for auto-reviewer)
    * `goal`               -> app-specific objective
    * `data_path`          -> canonical workspace the implementer writes into
    * `raw_data_path`      -> raw inputs the canonicalizer received
    * `domain_knowledge`   -> stable facts surfaced at intake
    * `versions`           -> implementer runs (experiments / probes)
    * `prediction_history` -> suspected bugs / hypotheses, carried forward

Apps with distinct vocabularies reinterpret the generic fields in their
prompts and display strings; the storage shape stays the same.

Type aliases (`ExperimentState` <-> `RunState`, `VersionEntry` <-> `ProbeEntry`,
`PredictionRecord` <-> `SuspectedBug`) let each app import under the name that
reads best in its codebase. Phase literals ("ingestion"/"iteration"/"report"/
"stopped") are internal state-machine labels kept for back-compat with
existing runs; UI labels come from the role registry.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Phase = Literal["ingestion", "iteration", "report", "stopped"]


class VersionEntry(BaseModel):
    """Record of a single implementer run (experiment / probe)."""

    version: str
    iteration: int
    script_path: str
    results_path: str | None = None
    hypothesis: str = ""
    status: Literal["pending", "running", "completed", "failed", "crashed"] = "pending"
    failure_reason: Literal["timed_out", "crash", "no_script", "no_result"] | None = None


# Review-oriented alias.
ProbeEntry = VersionEntry


class PredictionRecord(BaseModel):
    """A testable prediction + its outcome, persisted across iterations.

    Auto-scientist reads this as a scientific hypothesis. Auto-reviewer reads it
    as a suspected bug: `prediction` = "the bug is X", `diagnostic` =
    "reproduce by Y", `outcome` in {confirmed (reproduced), refuted (ran
    clean), inconclusive (flaky / timed out / could not build the probe)}.
    """

    pred_id: str = ""  # "{iteration}.{index}" assigned by the orchestrator
    iteration_prescribed: int
    iteration_evaluated: int | None = None
    prediction: str
    diagnostic: str
    if_confirmed: str
    if_refuted: str
    follows_from: str | None = None
    outcome: Literal["pending", "confirmed", "refuted", "inconclusive"] = "pending"
    evidence: str = ""
    summary: str = ""  # one-line compact summary for tree display


# Review-oriented alias.
SuspectedBug = PredictionRecord


class RunState(BaseModel):
    """Full state of a run, persisted to JSON after every phase transition."""

    domain: str
    goal: str
    phase: Phase = "ingestion"
    iteration: int = 0
    versions: list[VersionEntry] = Field(default_factory=list)
    dead_ends: list[str] = Field(default_factory=list)
    schedule: str | None = None
    consecutive_failures: int = 0
    data_path: str | None = None
    raw_data_path: str | None = None
    config_path: str | None = None
    domain_knowledge: str = ""
    max_iterations: int | None = Field(default=None, ge=1)
    prediction_history: list[PredictionRecord] = Field(default_factory=list)
    pending_abductions: list[dict] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        """Persist state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "RunState":
        """Load state from JSON file, migrating legacy phase names."""
        data = json.loads(path.read_text())
        # Legacy "discovery" phase maps to "iteration".
        if data.get("phase") == "discovery":
            data["phase"] = "iteration"
            data.setdefault("iteration", 0)
        return cls.model_validate(data)

    def record_version(self, entry: VersionEntry) -> None:
        """Add a version / probe entry."""
        self.versions.append(entry)

    def record_failure(self) -> None:
        """Increment consecutive failure counter."""
        self.consecutive_failures += 1

    def record_success(self) -> None:
        """Reset consecutive failure counter on success."""
        self.consecutive_failures = 0

    def should_stop_on_failures(self, max_consecutive: int = 3) -> bool:
        """Check if we've hit the consecutive failure cap."""
        return self.consecutive_failures >= max_consecutive


# Science-oriented alias kept for ergonomics in auto-scientist call sites.
ExperimentState = RunState
