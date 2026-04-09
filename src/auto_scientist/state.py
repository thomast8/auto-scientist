"""Experiment state persistence and crash recovery."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class VersionEntry(BaseModel):
    """Record of a single experiment version."""

    version: str
    iteration: int
    script_path: str
    results_path: str | None = None
    hypothesis: str = ""
    status: Literal["pending", "running", "completed", "failed", "crashed"] = "pending"
    failure_reason: Literal["timed_out", "crash", "no_script", "no_result"] | None = None


class PredictionRecord(BaseModel):
    """A testable prediction and its outcome, persisted across iterations."""

    pred_id: str = ""  # "{iteration}.{index}" assigned by orchestrator
    iteration_prescribed: int
    iteration_evaluated: int | None = None
    prediction: str
    diagnostic: str
    if_confirmed: str
    if_refuted: str
    follows_from: str | None = None
    outcome: Literal["pending", "confirmed", "refuted", "inconclusive"] = "pending"
    evidence: str = ""
    summary: str = ""  # One-line compact summary for tree display


class ExperimentState(BaseModel):
    """Full state of an experiment run, persisted to JSON after every phase transition."""

    domain: str
    goal: str
    phase: Literal["ingestion", "iteration", "report", "stopped"] = "ingestion"
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
    def load(cls, path: Path) -> "ExperimentState":
        """Load state from JSON file, migrating legacy formats."""
        data = json.loads(path.read_text())
        # Migrate legacy "discovery" phase to "iteration"
        if data.get("phase") == "discovery":
            data["phase"] = "iteration"
            data.setdefault("iteration", 0)
        return cls.model_validate(data)

    def record_version(self, entry: VersionEntry) -> None:
        """Add a version entry."""
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
