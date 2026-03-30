"""Rewind a saved run directory to a target iteration for resumption."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
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

# Canonical agent execution order within an iteration
AGENT_ORDER: list[str] = ["analyst", "scientist", "debate", "coder"]

# Files each agent produces in the version directory.
# Note: debate also overwrites plan.json; coder produces unpredictable files.
_AGENT_ARTIFACTS: dict[str, list[str]] = {
    "analyst": ["analysis.json"],
    "scientist": ["plan.json"],
    "debate": ["debate.json"],
    "coder": [],
}

# Buffer filename prefixes per agent
_AGENT_BUFFER_PREFIXES: dict[str, list[str]] = {
    "analyst": ["analyst_"],
    "scientist": ["scientist_"],
    "debate": ["debate_", "scientist_revision_"],
    "coder": ["coder_"],
}

# Notebook entry source attributes per agent
_AGENT_NOTEBOOK_SOURCES: dict[str, list[str]] = {
    "analyst": [],
    "scientist": ["scientist"],
    "debate": ["revision"],
    "coder": [],
}


@dataclass
class RewindResult:
    """Result of rewinding a run directory."""

    state: ExperimentState
    from_agent: str | None = None
    # Panel records from the target iteration for agents loaded from disk.
    # Each dict has keys: name, model, style, done_summary, input_tokens,
    # output_tokens, thinking_tokens, num_turns, elapsed_seconds, lines.
    restored_panels: list[dict] | None = None


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


def _strip_notebook_entries_for_agents(
    notebook_path: Path,
    target_version: str,
    from_agent: str,
) -> None:
    """Remove notebook entries for the target version whose source matches agents being re-run.

    Keeps entries from agents before from_agent; removes entries from from_agent onward.
    """
    if not notebook_path.exists():
        return

    agent_idx = AGENT_ORDER.index(from_agent)
    sources_to_remove: set[str] = set()
    for agent in AGENT_ORDER[agent_idx:]:
        sources_to_remove.update(_AGENT_NOTEBOOK_SOURCES[agent])

    if not sources_to_remove:
        return

    text = notebook_path.read_text()

    entry_pattern = re.compile(
        r'<entry\s+version="([^"]*)"\s+source="([^"]*)"[^>]*>.*?</entry>',
        re.DOTALL,
    )

    all_matches = list(entry_pattern.finditer(text))
    kept_entries = [
        m.group(0)
        for m in all_matches
        if not (m.group(1) == target_version and m.group(2) in sources_to_remove)
    ]

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


def _validate_agent_prerequisites(version_dir: Path, from_agent: str) -> None:
    """Check that all artifacts from agents before from_agent exist on disk.

    Raises ValueError if any required artifact is missing.
    """
    agent_idx = AGENT_ORDER.index(from_agent)
    for agent in AGENT_ORDER[:agent_idx]:
        for artifact in _AGENT_ARTIFACTS[agent]:
            path = version_dir / artifact
            if not path.exists():
                raise ValueError(
                    f"Cannot resume from '{from_agent}' at {version_dir.name}: "
                    f"required artifact '{artifact}' not found (expected from '{agent}' agent)"
                )


def _clean_version_dir_for_agent(version_dir: Path, from_agent: str) -> None:
    """Remove artifacts from from_agent onward, keeping earlier agents' output.

    Special handling for plan.json when resuming from debate: the on-disk
    plan.json is the post-debate revision, so we restore the pre-debate
    original from debate.json["original_plan"].
    """
    agent_idx = AGENT_ORDER.index(from_agent)

    # Compute files to keep (from agents strictly before from_agent)
    files_to_keep: set[str] = set()
    for agent in AGENT_ORDER[:agent_idx]:
        files_to_keep.update(_AGENT_ARTIFACTS[agent])

    # Special case: when resuming from debate, restore the pre-debate plan
    if from_agent == "debate":
        debate_path = version_dir / "debate.json"
        if debate_path.exists():
            try:
                debate_data = json.loads(debate_path.read_text())
                original_plan = debate_data.get("original_plan")
                if original_plan:
                    (version_dir / "plan.json").write_text(json.dumps(original_plan, indent=2))
                    logger.info("Restored pre-debate plan.json from debate.json")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Could not extract original plan from debate.json: {e}")

    # Delete everything except files to keep
    for child in version_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        elif child.name not in files_to_keep:
            child.unlink()


def rewind_run(
    run_dir: Path,
    target_iteration: int,
    from_agent: str | None = None,
) -> RewindResult:
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

    **Agent-level resume** (from_agent is set): within the target iteration,
    keeps artifacts from agents before from_agent and deletes the rest.
    The orchestrator will load preserved artifacts from disk and re-run
    from the specified agent onward.

    Returns a RewindResult with the rewound state and the normalized from_agent.
    """
    state_path = run_dir / "state.json"
    state = ExperimentState.load(state_path)

    # --- Validate from_agent ---
    if from_agent is not None:
        if from_agent not in AGENT_ORDER:
            raise ValueError(
                f"Unknown agent '{from_agent}'. Must be one of: {', '.join(AGENT_ORDER)}"
            )
        # analyst is the first agent, so resuming from it is a full restart
        if from_agent == "analyst":
            from_agent = None

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
    # In extend mode (target == current), bump past any iteration that
    # already has a manifest record so we don't create duplicates.
    # But when from_agent is set, the user wants to re-run *within* that
    # iteration, so skip the bump entirely.
    effective_iteration = target_iteration
    if extending and not from_agent:
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
    state.consecutive_failures = 0
    state.dead_ends = []

    # --- Prediction history trimming ---
    if from_agent and from_agent in ("scientist", "debate"):
        # Keep predictions from before this iteration + predictions evaluated
        # at this iteration (analyst ran). Remove predictions prescribed at
        # this iteration (scientist will re-prescribe).
        state.prediction_history = [
            p
            for p in state.prediction_history
            if p.iteration_prescribed < effective_iteration
            or (p.iteration_prescribed == effective_iteration and p.iteration_evaluated is not None)
        ]
    elif from_agent == "coder":
        # Coder doesn't touch predictions; keep everything up to and including
        # this iteration.
        state.prediction_history = [
            p for p in state.prediction_history if p.iteration_prescribed <= effective_iteration
        ]
    else:
        # Full restart (no from_agent): standard trim
        state.prediction_history = [
            p for p in state.prediction_history if p.iteration_prescribed < effective_iteration
        ]

    # --- Reconstruct domain_knowledge ---
    if from_agent:
        # Analyst output is preserved, so include the target iteration
        state.domain_knowledge = _reconstruct_domain_knowledge(run_dir, effective_iteration + 1)
    else:
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
    if from_agent:
        # Include the target iteration's version (some entries will be kept)
        target_version = f"v{effective_iteration:02d}"
        allowed_versions.add(target_version)
    _truncate_notebook(run_dir / NOTEBOOK_FILENAME, allowed_versions)
    # Selectively strip entries from agents being re-run
    if from_agent:
        _strip_notebook_entries_for_agents(
            run_dir / NOTEBOOK_FILENAME,
            f"v{effective_iteration:02d}",
            from_agent,
        )

    # --- Delete version directories ---
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        m = _VERSION_DIR_RE.match(child.name)
        if not m:
            continue
        dir_iter = int(m.group(1))
        if dir_iter > effective_iteration:
            # Always delete iterations above the target
            shutil.rmtree(child)
        elif dir_iter == effective_iteration:
            if from_agent:
                # Selectively clean: keep artifacts from agents before from_agent
                _validate_agent_prerequisites(child, from_agent)
                _clean_version_dir_for_agent(child, from_agent)
            else:
                # Full restart at this iteration: delete entire dir
                shutil.rmtree(child)

    # --- Delete buffers at or beyond effective iteration ---
    buffers_dir = run_dir / "buffers"
    if buffers_dir.is_dir():
        # Compute buffer prefixes to remove for the target iteration
        agent_idx = AGENT_ORDER.index(from_agent) if from_agent else 0
        prefixes_to_remove: set[str] = set()
        for agent in AGENT_ORDER[agent_idx:]:
            prefixes_to_remove.update(_AGENT_BUFFER_PREFIXES[agent])

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
            if buf_iter > effective_iteration:
                # Above target: always delete
                buf_file.unlink()
            elif buf_iter == effective_iteration:
                if from_agent:
                    # Agent-level: only delete buffers from agents being re-run
                    if any(name.startswith(p) for p in prefixes_to_remove):
                        buf_file.unlink()
                else:
                    # Full restart: delete all buffers at this iteration
                    # In extend mode, keep the analyst buffer (prediction context)
                    if extending and name.startswith("analyst_") and buf_iter == target_iteration:
                        continue
                    buf_file.unlink()

    # --- Delete final-run artifacts ---
    for artifact in _FINAL_ARTIFACTS:
        p = run_dir / artifact
        if p.exists():
            p.unlink()

    # --- Trim iteration manifest and extract panels for skipped agents ---
    restored_panels: list[dict] | None = None
    manifest_path = run_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        records = load_manifest(manifest_path)

        # Extract panel records for agents loaded from disk
        if from_agent:
            agent_idx = AGENT_ORDER.index(from_agent)
            # Map panel names to agent order indices
            # Analyst -> 0, Scientist -> 1, Critic/* -> 2, Coder -> 3
            panel_agent_idx = {"Analyst": 0, "Scientist": 1, "Coder": 3}

            for r in records:
                if r.iteration == effective_iteration:
                    restored_panels = [
                        p.model_dump()
                        for p in r.panels
                        if panel_agent_idx.get(
                            p.name.split("/")[0],
                            2,  # Critic/* defaults to debate idx
                        )
                        < agent_idx
                    ]
                    break

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
    return RewindResult(state=state, from_agent=from_agent, restored_panels=restored_panels)
