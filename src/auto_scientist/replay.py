"""Rewind a saved run directory to a target iteration for replay."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from auto_scientist.iteration_manifest import MANIFEST_FILENAME, load_manifest, save_manifest
from auto_scientist.notebook import NOTEBOOK_FILENAME
from auto_scientist.state import ExperimentState

# Regex to extract the two-digit iteration number from buffer filenames
# e.g. "analyst_02.txt" -> 2, "debate_novelty_skeptic_04.txt" -> 4
_BUFFER_ITER_RE = re.compile(r"_(\d{2})\.txt$")

# Regex to extract version directory names like "v00", "v01"
_VERSION_DIR_RE = re.compile(r"^v(\d{2})$")

# Files generated only during the report phase or that should be regenerated
_FINAL_ARTIFACTS = ("report.md", "exegesis.md", "console.log", "debug.log")


def _detect_old_output_dir(state: ExperimentState) -> str | None:
    """Infer the original output directory from absolute paths in state."""
    if state.config_path:
        return str(Path(state.config_path).parent)
    if state.versions:
        return str(Path(state.versions[0].script_path).parent.parent)
    if state.data_path:
        parent = Path(state.data_path).parent
        # data_path is typically <output_dir>/data
        if parent.name == "data":
            return str(parent.parent)
    return None


def _rewrite_path(path: str | None, old_base: str, new_base: str) -> str | None:
    """Replace old_base prefix with new_base in an absolute path."""
    if path is None:
        return None
    if path.startswith(old_base):
        return new_base + path[len(old_base):]
    return path


def _truncate_notebook(notebook_path: Path, allowed_versions: set[str]) -> None:
    """Keep only notebook entries whose version attribute is in allowed_versions."""
    if not notebook_path.exists():
        return

    text = notebook_path.read_text()

    # Match <entry version="...">...</entry> blocks (multiline, non-greedy)
    entry_pattern = re.compile(
        r'<entry\s+version="([^"]*)"[^>]*>.*?</entry>',
        re.DOTALL,
    )

    kept_entries = []
    for match in entry_pattern.finditer(text):
        version = match.group(1)
        if version in allowed_versions:
            kept_entries.append(match.group(0))

    if kept_entries:
        new_text = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<lab_notebook>\n"
            + "\n".join(kept_entries)
            + "\n</lab_notebook>\n"
        )
    else:
        new_text = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<lab_notebook>\n</lab_notebook>\n"
        )
    notebook_path.write_text(new_text)


def _reconstruct_domain_knowledge(run_dir: Path, target_iteration: int) -> str:
    """Rebuild domain_knowledge by replaying analysis.json files up to target_iteration."""
    if target_iteration == 0:
        return ""

    domain_knowledge = ""
    for i in range(target_iteration):
        analysis_path = run_dir / f"v{i:02d}" / "analysis.json"
        if not analysis_path.exists():
            continue
        analysis = json.loads(analysis_path.read_text())
        dk = analysis.get("domain_knowledge", "")
        if dk:
            domain_knowledge = dk
    return domain_knowledge


def rewind_run(run_dir: Path, target_iteration: int) -> ExperimentState:
    """Rewind a run directory in-place to the given iteration.

    The caller is responsible for copying the source run directory first.
    This function modifies run_dir to look as if the orchestrator just
    finished iteration target_iteration - 1 and is about to start
    target_iteration.

    Returns the rewound ExperimentState.
    """
    state_path = run_dir / "state.json"
    state = ExperimentState.load(state_path)

    # --- Validation ---
    if state.phase not in ("iteration", "report", "stopped"):
        raise ValueError(
            f"Cannot rewind a run in phase '{state.phase}'. "
            "Only completed or in-progress iteration runs can be rewound."
        )
    if target_iteration < 0:
        raise ValueError(f"target_iteration must be >= 0, got {target_iteration}")
    if target_iteration >= state.iteration:
        raise ValueError(
            f"target_iteration ({target_iteration}) must be < "
            f"current iteration ({state.iteration})"
        )

    # --- Detect old output dir for path rewriting ---
    old_base = _detect_old_output_dir(state)
    new_base = str(run_dir.resolve())

    # --- Trim state fields ---
    state.phase = "iteration"
    state.iteration = target_iteration
    state.versions = state.versions[:target_iteration]
    state.prediction_history = [
        p for p in state.prediction_history
        if p.iteration_prescribed < target_iteration
    ]
    state.consecutive_failures = 0
    state.dead_ends = []

    # --- Reconstruct domain_knowledge ---
    state.domain_knowledge = _reconstruct_domain_knowledge(run_dir, target_iteration)

    # --- Rewrite paths ---
    if old_base:
        state.data_path = _rewrite_path(state.data_path, old_base, new_base)
        state.config_path = _rewrite_path(state.config_path, old_base, new_base)
        for v in state.versions:
            v.script_path = _rewrite_path(v.script_path, old_base, new_base) or ""
            v.results_path = _rewrite_path(v.results_path, old_base, new_base)

    # --- Truncate lab notebook ---
    allowed_versions = {"ingestion"} | {f"v{i:02d}" for i in range(target_iteration)}
    _truncate_notebook(run_dir / NOTEBOOK_FILENAME, allowed_versions)

    # --- Delete version directories at or beyond target ---
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _VERSION_DIR_RE.match(child.name)
        if m and int(m.group(1)) >= target_iteration:
            shutil.rmtree(child)

    # --- Delete buffers at or beyond target ---
    buffers_dir = run_dir / "buffers"
    if buffers_dir.is_dir():
        for buf_file in sorted(buffers_dir.iterdir()):
            name = buf_file.name
            # Always delete report buffers (they belong to the report phase)
            if name.startswith("report_"):
                buf_file.unlink()
                continue
            m = _BUFFER_ITER_RE.search(name)
            if m and int(m.group(1)) >= target_iteration:
                buf_file.unlink()

    # --- Delete final-run artifacts ---
    for artifact in _FINAL_ARTIFACTS:
        p = run_dir / artifact
        if p.exists():
            p.unlink()

    # --- Trim iteration manifest (if present) ---
    manifest_path = run_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        records = load_manifest(manifest_path)
        kept = [
            r for r in records
            if r.iteration == "ingestion"
            or (isinstance(r.iteration, int) and r.iteration < target_iteration)
        ]
        save_manifest(kept, manifest_path)

    # --- Save rewound state ---
    state.save(state_path)
    return state
