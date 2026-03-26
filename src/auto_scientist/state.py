"""Experiment state persistence and crash recovery."""

import json
from pathlib import Path

from pydantic import BaseModel, Field

from auto_scientist.config import SuccessCriterion


class VersionEntry(BaseModel):
    """Record of a single experiment version."""

    version: str
    iteration: int
    script_path: str
    results_path: str | None = None
    score: int | None = None
    hypothesis: str = ""
    status: str = "pending"  # pending, running, completed, failed, crashed


class CriteriaRevision(BaseModel):
    """Audit trail entry for when top-level success criteria are defined or revised."""

    iteration: int
    action: str  # "defined" | "revised"
    changes: str  # What changed and why
    criteria_snapshot: list[SuccessCriterion]


class ExperimentState(BaseModel):
    """Full state of an experiment run, persisted to JSON after every phase transition."""

    domain: str
    goal: str
    phase: str = "ingestion"  # ingestion, iteration, report, stopped
    iteration: int = 0
    versions: list[VersionEntry] = Field(default_factory=list)
    dead_ends: list[str] = Field(default_factory=list)
    best_version: str | None = None
    best_score: int = 0
    schedule: str | None = None
    consecutive_failures: int = 0
    data_path: str | None = None
    raw_data_path: str | None = None
    config_path: str | None = None
    success_criteria: list[SuccessCriterion] | None = None
    domain_knowledge: str = ""
    criteria_history: list[CriteriaRevision] = Field(default_factory=list)
    ingestion_source: str | None = None

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
        """Add a version entry and update best tracking."""
        self.versions.append(entry)
        if entry.score is not None and entry.score > self.best_score:
            self.best_score = entry.score
            self.best_version = entry.version

    def record_failure(self) -> None:
        """Increment consecutive failure counter."""
        self.consecutive_failures += 1

    def record_success(self) -> None:
        """Reset consecutive failure counter on success."""
        self.consecutive_failures = 0

    def should_stop_on_failures(self, max_consecutive: int = 5) -> bool:
        """Check if we've hit the consecutive failure cap."""
        return self.consecutive_failures >= max_consecutive
