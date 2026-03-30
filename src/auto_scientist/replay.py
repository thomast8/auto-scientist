"""Rewind a saved run directory to a target iteration for resumption."""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from auto_scientist.iteration_manifest import MANIFEST_FILENAME, load_manifest, save_manifest
from auto_scientist.notebook import NOTEBOOK_FILENAME
from auto_scientist.state import ExperimentState

logger = logging.getLogger(__name__)

# Regex to extract the two-digit iteration number from buffer filenames
# e.g. "analyst_02.txt" -> 2, "debate_novelty_skeptic_04.txt" -> 4
_BUFFER_ITER_RE = re.compile(r"_(\d{2})\.txt$")

# Regex to extract version directory names like "v00", "v01"
_VERSION_DIR_RE = re.compile(r"^v(\d{2})$")

# Artifacts to remove when rewinding: report output + run-wide logs that will be regenerated
_FINAL_ARTIFACTS = ("report.md", "console.log", "debug.log")


def _detect_old_output_dir(state: ExperimentState) -> str | None:
    """Infer the original output directory from paths stored in state."""
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
        return new_base + path[len(old_base) :]
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

    all_matches = list(entry_pattern.finditer(text))
    kept_entries = [m.group(0) for m in all_matches if m.group(1) in allowed_versions]

    if all_matches and not kept_entries:
        logger.warning(
            f"Notebook had {len(all_matches)} entries but none matched "
            f"allowed versions {allowed_versions}; notebook will be empty"
        )

    if kept_entries:
        new_text = (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            "<lab_notebook>\n" + "\n".join(kept_entries) + "\n</lab_notebook>\n"
        )
    else:
        new_text = '<?xml version="1.0" encoding="utf-8"?>\n<lab_notebook>\n</lab_notebook>\n'
    notebook_path.write_text(new_text)


def _reconstruct_domain_knowledge(run_dir: Path, target_iteration: int) -> str:
    """Rebuild domain_knowledge from analysis.json files.

    Reads iterations 0 through target_iteration - 1.
    """
    if target_iteration == 0:
        return ""

    domain_knowledge = ""
    for i in range(target_iteration):
        analysis_path = run_dir / f"v{i:02d}" / "analysis.json"
        if not analysis_path.exists():
            continue
        try:
            analysis = json.loads(analysis_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Skipping corrupt analysis file {analysis_path}: {e}")
            continue
        dk = analysis.get("domain_knowledge", "")
        if dk:
            domain_knowledge = dk
    return domain_knowledge


def rewind_run(run_dir: Path, target_iteration: int) -> ExperimentState:
    """Rewind a run directory in-place to the given iteration.

    The caller is responsible for copying the source run directory first
    (when forking). This function can also be called in-place for resume.

    **Rewind mode** (target_iteration < state.iteration): discards
    iterations from target_iteration onward. The orchestrator will begin
    fresh at target_iteration.

    **Extend mode** (target_iteration == state.iteration): preserves all
    completed iterations, strips report-phase artifacts, and bumps
    state.iteration past any iteration that already has a manifest record
    (so the TUI doesn't show a duplicate). The analyst buffer at
    target_iteration is kept in extend mode (it contains prediction
    resolution context from the report phase).

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
    if target_iteration > state.iteration:
        raise ValueError(
            f"target_iteration ({target_iteration}) must be <= "
            f"current iteration ({state.iteration})"
        )

    # --- Detect old output dir for path rewriting ---
    old_base = _detect_old_output_dir(state)
    if old_base is None:
        logger.warning("Could not detect original output dir; path rewriting skipped")
    new_base = str(run_dir.resolve())
    extending = target_iteration == state.iteration

    # --- Compute effective iteration ---
    # In extend mode, bump past any iteration that already has a manifest
    # record (e.g. scientist stopped at iteration 3, manifest has it,
    # so we resume from iteration 4).
    effective_iteration = target_iteration
    if extending:
        manifest_path_check = run_dir / MANIFEST_FILENAME
        if manifest_path_check.exists():
            records_check = load_manifest(manifest_path_check)
            max_recorded = max(
                (r.iteration for r in records_check if isinstance(r.iteration, int)),
                default=-1,
            )
            if isinstance(max_recorded, int) and max_recorded >= target_iteration:
                effective_iteration = max_recorded + 1

    # --- Trim state fields ---
    state.phase = "iteration"
    state.iteration = effective_iteration
    state.versions = state.versions[:effective_iteration]
    state.prediction_history = [
        p for p in state.prediction_history if p.iteration_prescribed < effective_iteration
    ]
    state.consecutive_failures = 0
    state.dead_ends = []

    # --- Reconstruct domain_knowledge ---
    state.domain_knowledge = _reconstruct_domain_knowledge(run_dir, effective_iteration)

    # --- Rewrite paths ---
    if old_base:
        state.data_path = _rewrite_path(state.data_path, old_base, new_base)
        state.config_path = _rewrite_path(state.config_path, old_base, new_base)
        for v in state.versions:
            v.script_path = _rewrite_path(v.script_path, old_base, new_base) or ""
            v.results_path = _rewrite_path(v.results_path, old_base, new_base)

    # --- Truncate lab notebook ---
    allowed_versions = {"ingestion"} | {f"v{i:02d}" for i in range(effective_iteration)}
    _truncate_notebook(run_dir / NOTEBOOK_FILENAME, allowed_versions)

    # --- Delete version directories at or beyond effective iteration ---
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _VERSION_DIR_RE.match(child.name)
        if m and int(m.group(1)) >= effective_iteration:
            shutil.rmtree(child)

    # --- Delete buffers at or beyond effective iteration ---
    buffers_dir = run_dir / "buffers"
    if buffers_dir.is_dir():
        for buf_file in sorted(buffers_dir.iterdir()):
            name = buf_file.name
            # Always delete report buffers (they belong to the report phase)
            if name.startswith("report_"):
                buf_file.unlink()
                continue
            m = _BUFFER_ITER_RE.search(name)
            if not m:
                continue
            buf_iter = int(m.group(1))
            if buf_iter >= effective_iteration:
                # In extend mode, keep the analyst buffer at target_iteration
                # (produced by _resolve_final_predictions during report phase)
                if extending and name.startswith("analyst_") and buf_iter == target_iteration:
                    continue
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
            r
            for r in records
            if (
                r.iteration == "ingestion"
                # Guard against old manifests where report was tagged as "ingestion"
                and r.title != "Report"
            )
            or (isinstance(r.iteration, int) and r.iteration < effective_iteration)
        ]
        # Normalize the last iteration's border to green
        if kept and kept[-1].iteration != "ingestion":
            kept[-1].result_style = "green"
            kept[-1].result_text = "done"
        save_manifest(kept, manifest_path)

    # --- Save rewound state ---
    state.save(state_path)
    return state
