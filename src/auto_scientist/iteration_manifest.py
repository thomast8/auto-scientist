"""Per-iteration TUI metadata, persisted for replay reconstruction."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

MANIFEST_FILENAME = "iteration_manifest.json"


class PanelRecord(BaseModel):
    """Snapshot of one AgentPanel's final state."""

    name: str
    model: str
    style: str = "cyan"
    done_summary: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    num_turns: int = 0
    elapsed_seconds: float = 0.0
    lines: list[str] = Field(default_factory=list)


class IterationRecord(BaseModel):
    """Snapshot of one completed iteration's TUI state."""

    iteration: int | str  # int for iterations, "ingestion" for ingestion phase
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
    """Load manifest from JSON. Returns [] if file is missing."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [IterationRecord.model_validate(item) for item in data]


def append_record(record: IterationRecord, path: Path) -> None:
    """Load existing manifest, append record, write back."""
    records = load_manifest(path)
    records.append(record)
    save_manifest(records, path)
