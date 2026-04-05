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

# Canonical agent execution order within an iteration.
# Stop gate sub-steps (assessment, stop_debate, stop_revision) are conditional:
# they only run when the scientist recommends stopping.
AGENT_ORDER: list[str] = [
    "analyst",
    "scientist",
    "assessment",
    "stop_debate",
    "stop_revision",
    "debate",
    "revision",
    "coder",
]

# Stop gate agents are conditional (only present when should_stop=True)
STOP_GATE_AGENTS: frozenset[str] = frozenset({"assessment", "stop_debate", "stop_revision"})

# Agents that re-prescribe predictions (scientist or anything after it that
# re-plans): keep evaluated predictions, discard prescribed-only ones.
_REPLANNING_AGENTS: frozenset[str] = frozenset(
    {
        "scientist",
        "assessment",
        "stop_debate",
        "stop_revision",
        "debate",
        "revision",
    }
)

# Files each agent produces in the version directory.
_AGENT_ARTIFACTS: dict[str, list[str]] = {
    "analyst": ["analysis.json"],
    "scientist": ["plan.json"],
    "assessment": ["completeness_assessment.json"],
    "stop_debate": ["stop_debate.json"],
    "stop_revision": ["stop_revision_plan.json"],
    "debate": ["debate.json"],
    "revision": ["revision_plan.json"],
    "coder": [],
}

# Buffer filename prefixes per agent
_AGENT_BUFFER_PREFIXES: dict[str, list[str]] = {
    "analyst": ["analyst_"],
    "scientist": ["scientist_"],
    "assessment": ["completeness_assessment_"],
    "stop_debate": ["stop_debate_"],
    "stop_revision": ["stop_revision_"],
    "debate": ["debate_"],
    "revision": ["scientist_revision_"],
    "coder": ["coder_"],
}

# Notebook entry source attributes per agent
_AGENT_NOTEBOOK_SOURCES: dict[str, list[str]] = {
    "analyst": [],
    "scientist": ["scientist"],
    "assessment": [],
    "stop_debate": [],
    "stop_revision": ["stop_revision", "stop_gate"],
    "debate": [],
    "revision": ["revision"],
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


def _extract_done_summary(agent: str, version_dir: Path) -> str:
    """Extract a human-readable summary from an agent's artifact file.

    Returns an empty string if no meaningful summary can be extracted.
    """
    try:
        if agent == "analyst":
            artifact = version_dir / "analysis.json"
            if not artifact.exists():
                return ""
            analysis = json.loads(artifact.read_text())
            # Prefer data_summary, fall back to a brief from key findings
            ds: str = analysis.get("data_summary") or ""
            if ds:
                return ds[:300]
            # Build from structured fields if data_summary is missing
            parts = []
            if analysis.get("patterns_observed"):
                parts.append(str(analysis["patterns_observed"])[:200])
            if analysis.get("next_unknowns"):
                parts.append(f"unknowns: {analysis['next_unknowns'][:100]}")
            return ", ".join(parts)[:300] if parts else ""

        if agent == "scientist":
            artifact = version_dir / "plan.json"
            if not artifact.exists():
                return ""
            plan = json.loads(artifact.read_text())
            parts = []
            if plan.get("should_stop"):
                parts.append(f"should_stop=True: {plan.get('stop_reason', '')}")
            else:
                if plan.get("strategy"):
                    parts.append(f"strategy={plan['strategy']}")
                if plan.get("hypothesis"):
                    parts.append(plan["hypothesis"][:200])
            return ", ".join(parts) if parts else ""

        if agent == "assessment":
            artifact = version_dir / "completeness_assessment.json"
            if not artifact.exists():
                return ""
            assessment = json.loads(artifact.read_text())
            coverage = assessment.get("overall_coverage", "?")
            gaps = [
                sq["question"][:60]
                for sq in assessment.get("sub_questions", [])
                if sq.get("coverage") in ("shallow", "unexplored")
            ]
            summary = f"coverage={coverage}"
            if gaps:
                summary += f", gaps: {', '.join(gaps[:3])}"
            return summary[:300]

        if agent == "revision":
            artifact = version_dir / "revision_plan.json"
            if not artifact.exists():
                return ""
            plan = json.loads(artifact.read_text())
            return f"strategy={plan.get('strategy', '?')}"

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read artifact for {agent} panel summary: {e}")
    return ""


def _build_panels_from_buffers(
    run_dir: Path,
    iteration: int,
    from_agent: str,
    model_config_path: Path | None = None,
) -> list[dict]:
    """Build restored panel records from buffer files for agents before from_agent.

    When the manifest doesn't have panel records for the current iteration
    (because it never completed), this reconstructs minimal panel data from
    the buffer files and artifact files on disk.

    Prefers panels.json (written incrementally by the orchestrator as each
    agent completes) over artifact-based reconstruction when available.
    """
    version_tag = f"{iteration:02d}"
    version_dir = run_dir / f"v{version_tag}"

    # Check for incrementally-saved panel snapshots first
    panels_path = version_dir / "panels.json"
    if panels_path.exists():
        try:
            saved_panels: list[dict] = json.loads(panels_path.read_text())
            if isinstance(saved_panels, list) and saved_panels:
                logger.info(f"Loaded {len(saved_panels)} panel(s) from {panels_path}")
                return saved_panels
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load {panels_path}, falling back to buffers: {e}")

    # Fallback: reconstruct from buffer files and artifacts
    agent_idx = AGENT_ORDER.index(from_agent)
    agents_to_restore = AGENT_ORDER[:agent_idx]

    # Map AGENT_ORDER names to model config keys (they differ for some agents)
    _config_key = {
        "assessment": "assessor",
        "revision": "scientist",
        "stop_revision": "scientist",
    }

    # Load model config to get agent model names
    model_names: dict[str, str] = {}
    config_path = model_config_path or (run_dir / "model_config.json")
    if config_path.exists():
        try:
            from auto_scientist.model_config import ModelConfig

            raw = json.loads(config_path.read_text())
            mc = ModelConfig._from_dict(raw)
            for agent in agents_to_restore:
                cfg = mc.resolve(_config_key.get(agent, agent))
                model_names[agent] = cfg.model
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logger.warning(f"Could not load model config from {config_path}: {e}")

    # Map agents to their buffer file prefix and panel display name
    agent_panel_info = {
        "analyst": {"panel_name": "Analyst", "style": "green", "buffer_prefix": "analyst_"},
        "scientist": {"panel_name": "Scientist", "style": "cyan", "buffer_prefix": "scientist_"},
        "assessment": {
            "panel_name": "Assessor",
            "style": "blue",
            "buffer_prefix": "completeness_assessment_",
        },
        "revision": {
            "panel_name": "Revision",
            "style": "cyan",
            "buffer_prefix": "scientist_revision_",
        },
    }

    panels: list[dict] = []
    buffers_dir = run_dir / "buffers"

    # Resolve critic model label from config
    critic_label = "unknown"
    if config_path.exists():
        try:
            from auto_scientist.model_config import ModelConfig

            mc = ModelConfig._from_dict(json.loads(config_path.read_text()))
            if mc.critics:
                c = mc.critics[0]
                critic_label = f"{c.provider}:{c.model}"
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            pass

    # Regex to extract persona name from critic buffer filenames
    # e.g. "stop_debate_depth_challenger_05.txt" -> "Depth Challenger"
    # e.g. "debate_methodologist_05.txt" -> "Methodologist"
    _stop_debate_re = re.compile(rf"^stop_debate_(.+)_{version_tag}\.txt$")
    _debate_re = re.compile(rf"^debate_(.+)_{version_tag}\.txt$")

    for agent in agents_to_restore:
        # Handle critic panels (stop_debate / debate) by scanning buffer files
        if agent in ("stop_debate", "debate"):
            pattern = _stop_debate_re if agent == "stop_debate" else _debate_re
            if buffers_dir.is_dir():
                for buf_file in sorted(buffers_dir.iterdir()):
                    m = pattern.match(buf_file.name)
                    if not m:
                        continue
                    persona_slug = m.group(1)
                    persona_name = persona_slug.replace("_", " ").title()
                    panels.append(
                        {
                            "name": f"Critic/{persona_name}",
                            "model": critic_label,
                            "style": "yellow",
                            "done_summary": f"{persona_name} loaded from disk",
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "thinking_tokens": 0,
                            "num_turns": 0,
                            "elapsed_seconds": 0,
                            "lines": [],
                        }
                    )
            continue

        info = agent_panel_info.get(agent)
        if not info:
            logger.debug(f"Skipping panel reconstruction for '{agent}' (not supported)")
            continue

        # Read buffer content for summary
        buf_path = buffers_dir / f"{info['buffer_prefix']}{version_tag}.txt"
        lines: list[str] = []
        if buf_path.exists():
            lines = buf_path.read_text().splitlines()

        # Build a meaningful done_summary from the artifact
        done_summary = _extract_done_summary(agent, version_dir)

        if not done_summary and lines:
            done_summary = lines[-1][:300]

        panels.append(
            {
                "name": info["panel_name"],
                "model": model_names.get(agent, "unknown"),
                "style": info["style"],
                "done_summary": done_summary or f"{agent} loaded from disk",
                "input_tokens": 0,
                "output_tokens": 0,
                "thinking_tokens": 0,
                "num_turns": 0,
                "elapsed_seconds": 0,
                "lines": [],
            }
        )

    return panels


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

    Stop gate agents and revision are conditional, so their artifacts are
    only required when they actually exist on disk.

    Raises ValueError if any required artifact is missing.
    """
    # Agents whose artifacts are conditional (only present in some iterations)
    conditional_agents = STOP_GATE_AGENTS | {"revision"}

    agent_idx = AGENT_ORDER.index(from_agent)
    for agent in AGENT_ORDER[:agent_idx]:
        if agent in conditional_agents:
            continue
        for artifact in _AGENT_ARTIFACTS[agent]:
            path = version_dir / artifact
            if not path.exists():
                raise ValueError(
                    f"Cannot resume from '{from_agent}' at {version_dir.name}: "
                    f"required artifact '{artifact}' not found (expected from '{agent}' agent)"
                )


def _clean_version_dir_for_agent(version_dir: Path, from_agent: str) -> None:
    """Remove artifacts from from_agent onward, keeping earlier agents' output.

    Special handling for plan.json when resuming from debate or revision:
    the on-disk plan.json may be the post-debate revision, so we restore
    the pre-debate original from debate.json["original_plan"].
    """
    agent_idx = AGENT_ORDER.index(from_agent)

    # Compute files to keep (from agents strictly before from_agent).
    # Skip conditional agents whose artifacts may not exist.
    conditional_agents = STOP_GATE_AGENTS | {"revision"}
    files_to_keep: set[str] = set()
    for agent in AGENT_ORDER[:agent_idx]:
        if agent in conditional_agents:
            # Only keep conditional artifacts that actually exist
            for artifact in _AGENT_ARTIFACTS[agent]:
                if (version_dir / artifact).exists():
                    files_to_keep.add(artifact)
        else:
            files_to_keep.update(_AGENT_ARTIFACTS[agent])

    # When resuming from debate or revision, restore the pre-debate plan
    # from debate.json["original_plan"] (plan.json may have been overwritten
    # by the revision step)
    if from_agent in ("debate", "revision"):
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
            f"--from-iteration {target_iteration + 1} is beyond "
            f"current iteration ({state.iteration + 1})"
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

    # --- Validate version dir exists when resuming from a specific agent ---
    # Must happen before state mutations so we don't corrupt state on error.
    if from_agent:
        target_version_dir = run_dir / f"v{effective_iteration:02d}"
        if not target_version_dir.exists():
            existing = sorted(
                int(m.group(1))
                for child in run_dir.iterdir()
                if child.is_dir() and (m := _VERSION_DIR_RE.match(child.name))
            )
            max_existing = existing[-1] if existing else -1
            raise ValueError(
                f"Cannot resume from '{from_agent}' at iteration {effective_iteration + 1}: "
                f"directory '{target_version_dir.name}/' does not exist. "
                f"The run has iterations up to v{max_existing:02d}. "
                f"Did you mean --from-iteration {max_existing + 1}?"
            )

    # --- Trim state fields ---
    state.phase = "iteration"
    state.iteration = effective_iteration
    state.versions = state.versions[:effective_iteration]
    state.consecutive_failures = 0
    state.dead_ends = []

    # --- Prediction history trimming ---
    if from_agent and from_agent in _REPLANNING_AGENTS:
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
            # Map panel names to AGENT_ORDER indices.
            # Critic/* panels (both stop-debate and normal debate) default
            # to the debate index; this is approximate but sufficient for
            # determining which panels to restore in the TUI.
            _debate_idx = AGENT_ORDER.index("debate")
            panel_agent_idx = {
                "Analyst": AGENT_ORDER.index("analyst"),
                "Scientist": AGENT_ORDER.index("scientist"),
                "Assessor": AGENT_ORDER.index("assessment"),
                "Stop Revision": AGENT_ORDER.index("stop_revision"),
                "Revision": AGENT_ORDER.index("revision"),
                "Coder": AGENT_ORDER.index("coder"),
            }

            for r in records:
                if r.iteration == effective_iteration:
                    restored_panels = [
                        p.model_dump()
                        for p in r.panels
                        if panel_agent_idx.get(
                            p.name.split("/")[0],
                            _debate_idx,  # Critic/* defaults to debate idx
                        )
                        < agent_idx
                    ]
                    break

            # If the manifest didn't have records for this iteration
            # (it never completed), build panels from buffer/artifact files
            if not restored_panels:
                restored_panels = _build_panels_from_buffers(
                    run_dir, effective_iteration, from_agent
                )

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
