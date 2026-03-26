"""ExperimentStore protocol and FilesystemStore implementation.

Provides an abstraction layer for discovering, listing, and inspecting
past experiment runs. The first implementation (FilesystemStore) scans
the experiments directory for state.json files.
"""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from auto_scientist.model_config import ModelConfig
from auto_scientist.state import ExperimentState

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class ExperimentSummary(BaseModel):
    """Lightweight summary of an experiment for the HomeScreen table."""

    id: str
    goal: str
    preset_name: str | None = None
    iteration: int = 0
    status: str = "paused"  # running | paused | completed | crashed
    output_dir: str = ""
    started_at: str = ""


class IngestionSource(BaseModel):
    """An experiment whose ingested data can be reused."""

    id: str
    goal: str
    data_paths: list[str] = []
    output_dir: str = ""


class ExperimentDetail(BaseModel):
    """Full experiment state + model config for resume/view."""

    state: ExperimentState
    models: ModelConfig | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ExperimentStore(Protocol):
    def list_experiments(self) -> list[ExperimentSummary]: ...
    def get_experiment(self, experiment_id: str) -> ExperimentDetail | None: ...
    def get_ingestion_sources(self) -> list[IngestionSource]: ...
    def register_experiment(
        self, output_dir: Path, state: ExperimentState, model_config: ModelConfig,
    ) -> str: ...
    def update_status(self, experiment_id: str, status: str, iteration: int) -> None: ...


# ---------------------------------------------------------------------------
# Filesystem implementation
# ---------------------------------------------------------------------------


def _derive_status(phase: str) -> str:
    """Derive experiment status from the state phase field."""
    if phase == "stopped":
        return "completed"
    if phase == "report":
        return "completed"
    # ingestion, iteration - we can't check process liveness from FS, so mark as paused
    return "paused"


class FilesystemStore:
    """Scans a directory for experiment state.json files.

    No extra metadata files are maintained. The filesystem is the source of truth.
    register_experiment and update_status are no-ops since the orchestrator
    already writes state.json and model_config.json.
    """

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def list_experiments(self) -> list[ExperimentSummary]:
        """Scan base directory for experiments, sorted by ID."""
        if not self._base_dir.exists():
            return []

        results: list[ExperimentSummary] = []
        for entry in sorted(self._base_dir.iterdir()):
            if not entry.is_dir():
                continue
            state_path = entry / "state.json"
            if not state_path.exists():
                continue
            try:
                state = ExperimentState.load(state_path)
            except Exception:
                continue

            preset_name = self._read_preset_name(entry)
            started_at = self._get_started_at(entry)

            results.append(ExperimentSummary(
                id=entry.name,
                goal=state.goal,
                preset_name=preset_name,
                iteration=state.iteration,
                status=_derive_status(state.phase),
                output_dir=str(entry),
                started_at=started_at,
            ))

        return results

    def get_experiment(self, experiment_id: str) -> ExperimentDetail | None:
        """Load full experiment detail by ID (directory name)."""
        exp_dir = self._base_dir / experiment_id
        state_path = exp_dir / "state.json"
        if not state_path.exists():
            return None

        state = ExperimentState.load(state_path)
        mc = self._load_model_config(exp_dir)

        return ExperimentDetail(state=state, models=mc)

    def get_ingestion_sources(self) -> list[IngestionSource]:
        """Return experiments that have completed ingestion (have a data_path)."""
        sources: list[IngestionSource] = []
        for summary in self.list_experiments():
            exp_dir = self._base_dir / summary.id
            state_path = exp_dir / "state.json"
            try:
                state = ExperimentState.load(state_path)
            except Exception:
                continue
            if state.data_path:
                sources.append(IngestionSource(
                    id=summary.id,
                    goal=state.goal,
                    data_paths=[state.data_path],
                    output_dir=str(exp_dir),
                ))
        return sources

    def register_experiment(
        self, output_dir: Path, state: ExperimentState, model_config: ModelConfig,
    ) -> str:
        """No-op. The orchestrator writes state.json directly."""
        return output_dir.name

    def update_status(self, experiment_id: str, status: str, iteration: int) -> None:
        """No-op. The orchestrator updates state.json directly."""

    def _read_preset_name(self, exp_dir: Path) -> str | None:
        """Read preset_name from model_config.json if it exists."""
        mc_path = exp_dir / "model_config.json"
        if not mc_path.exists():
            return None
        try:
            data = json.loads(mc_path.read_text())
            return data.get("preset_name")
        except (json.JSONDecodeError, OSError):
            return None

    def _load_model_config(self, exp_dir: Path) -> ModelConfig | None:
        """Load ModelConfig from model_config.json if it exists."""
        mc_path = exp_dir / "model_config.json"
        if not mc_path.exists():
            return None
        try:
            return ModelConfig.model_validate_json(mc_path.read_text())
        except Exception:
            return None

    def _get_started_at(self, exp_dir: Path) -> str:
        """Get the experiment start time from directory creation time."""
        try:
            stat = exp_dir.stat()
            ctime = stat.st_birthtime if hasattr(stat, "st_birthtime") else stat.st_ctime
            from datetime import datetime
            return datetime.fromtimestamp(ctime, tz=UTC).isoformat()
        except OSError:
            return ""


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------


def next_output_dir(base: Path) -> Path:
    """If *base* already contains a state.json, return base_001, base_002, etc."""
    if not (base / "state.json").exists():
        return base
    seq = 1
    while True:
        candidate = base.parent / f"{base.name}_{seq:03d}"
        if not (candidate / "state.json").exists():
            return candidate
        seq += 1
