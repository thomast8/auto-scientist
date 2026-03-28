"""Per-iteration TUI metadata, persisted for replay reconstruction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "iteration_manifest.json"


class PanelRecord(BaseModel):
    """Snapshot of one AgentPanel's final state."""

    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    style: str = "cyan"
    done_summary: str = ""
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    num_turns: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    lines: list[str] = Field(default_factory=list)


class IterationRecord(BaseModel):
    """Snapshot of one completed iteration's TUI state."""

    iteration: int | Literal["ingestion"]
    title: str
    result_text: str = "done"
    result_style: str = "green"
    summary: str = ""
    panels: list[PanelRecord] = Field(default_factory=list)


def save_manifest(records: list[IterationRecord], path: Path) -> None:
    """Write the full manifest list to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [r.model_dump() for r in records]
    path.write_text(json.dumps(data, indent=2))


def load_manifest(path: Path) -> list[IterationRecord]:
    """Load manifest from JSON. Returns [] if file is missing or corrupt."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return [IterationRecord.model_validate(item) for item in data]
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Could not load iteration manifest {path}: {e}")
        return []


def append_record(record: IterationRecord, path: Path) -> None:
    """Load existing manifest, append record, write back."""
    records = load_manifest(path)
    records.append(record)
    save_manifest(records, path)
