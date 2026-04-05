"""Disk I/O, manifest management, resume loaders, and prediction logic.

All functions are standalone (no dependency on the Orchestrator class).
They take explicit parameters instead of reading from self.*.
"""

import json
import logging
from pathlib import Path
from typing import Any, Literal

from auto_scientist.iteration_manifest import (
    MANIFEST_FILENAME,
    IterationRecord,
    PanelRecord,
    append_record,
    load_manifest,
)
from auto_scientist.runner import RunResult
from auto_scientist.state import ExperimentState, PredictionRecord, VersionEntry

logger = logging.getLogger(__name__)

# Module-level constants (previously class attributes on Orchestrator)
VALID_OUTCOMES = {"confirmed", "refuted", "inconclusive"}

INFRA_FILES = {
    "run_result.json",
    "exitcode.txt",
    "stderr.txt",
    "analysis.json",
    "plan.json",
    "debate.json",
}


def persist_buffer(
    output_dir: Path,
    agent_name: str,
    buffer: list[str],
    iteration: int,
) -> None:
    """Write an agent's message buffer to disk for debugging."""
    if not buffer:
        return
    buffers_dir = output_dir / "buffers"
    buffers_dir.mkdir(exist_ok=True)
    filename = f"{agent_name.lower().replace(' ', '_')}_{iteration:02d}.txt"
    (buffers_dir / filename).write_text("\n".join(buffer))


def persist_artifact(version_dir: Path, filename: str, data: Any) -> None:
    """Save a structured JSON artifact to a version directory."""
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / filename).write_text(json.dumps(data, indent=2))


def collect_iteration_record(
    live: Any,
    state: ExperimentState,
    title: str | int,
    result_text: str,
    result_style: str,
    summary: str,
) -> IterationRecord:
    """Snapshot current iteration's panel metadata for the manifest."""
    container = live._current_iteration
    panels = []
    for p in getattr(container, "_panels", []):
        panels.append(
            PanelRecord(
                name=p.panel_name,
                model=p.model,
                style=p.panel_style,
                done_summary=p.done_summary,
                input_tokens=p.input_tokens,
                output_tokens=p.output_tokens,
                thinking_tokens=p.thinking_tokens,
                num_turns=p.num_turns,
                elapsed_seconds=p.elapsed,
                lines=list(p.all_lines),
            )
        )
    iteration_key: int | Literal["ingestion", "report"]
    if isinstance(title, int):
        iteration_key = state.iteration
        display_title = f"Iteration {title + 1}"
    elif title == "Report":
        iteration_key = "report"
        display_title = "Report"
    else:
        iteration_key = "ingestion"
        display_title = str(title)
    return IterationRecord(
        iteration=iteration_key,
        title=display_title,
        result_text=result_text,
        result_style=result_style,
        summary=summary,
        panels=panels,
    )


def save_iteration_manifest(
    live: Any,
    state: ExperimentState,
    output_dir: Path,
    title: str | int,
    result_text: str,
    result_style: str,
    summary: str,
) -> None:
    """Collect and persist an iteration record to the manifest file."""
    record = collect_iteration_record(live, state, title, result_text, result_style, summary)
    append_record(record, output_dir / MANIFEST_FILENAME)


def save_partial_panels(live: Any, version_dir: Path) -> None:
    """Snapshot current iteration's completed panels to version_dir/panels.json.

    Called after each agent completes so that if a later agent crashes,
    the panel data (summaries, lines, token counts) is preserved on disk
    for resume/fork reconstruction.
    """
    container = live._current_iteration
    if container is None:
        return
    panels = []
    for p in getattr(container, "_panels", []):
        if not p.done:
            continue
        panels.append(
            {
                "name": p.panel_name,
                "model": p.model,
                "style": p.panel_style,
                "done_summary": p.done_summary,
                "input_tokens": p.input_tokens,
                "output_tokens": p.output_tokens,
                "thinking_tokens": p.thinking_tokens,
                "num_turns": p.num_turns,
                "elapsed_seconds": p.elapsed,
                "lines": list(p.all_lines),
            }
        )
    if panels:
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "panels.json").write_text(json.dumps(panels, indent=2))


def restore_iterations_from_manifest(live: Any, output_dir: Path) -> None:
    """Mount collapsed iteration panels from a saved manifest.

    Called at startup to show previous iterations in the TUI when
    resuming or forking. Silently skips if no manifest exists.
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    records = load_manifest(manifest_path)
    if not records:
        return

    for record in records:
        live.mount_restored_iteration(
            title=record.title,
            result_text=record.result_text,
            result_style=record.result_style,
            summary=record.summary,
            panels=[p.model_dump() for p in record.panels],
        )


def load_analyst_from_disk(version_dir: Path, state: ExperimentState) -> dict[str, Any] | None:
    """Load analyst output from disk and replay state side-effects."""
    analysis_path = version_dir / "analysis.json"
    if not analysis_path.exists():
        logger.error(f"Cannot load analyst output: {analysis_path} not found")
        return None
    try:
        analysis: dict[str, Any] = json.loads(analysis_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Cannot load analyst output: {e}")
        return None

    # Replay side-effects
    if analysis.get("domain_knowledge"):
        state.domain_knowledge = analysis["domain_knowledge"]
        logger.info("Domain knowledge restored from disk")
    resolve_prediction_outcomes(analysis, state)

    logger.info(f"Loaded analyst output from {analysis_path}")
    return analysis


def load_scientist_plan_from_disk(version_dir: Path) -> dict[str, Any] | None:
    """Load scientist plan from disk.

    When resuming from debate, rewind_run has already restored the
    pre-debate plan in plan.json.
    """
    plan_path = version_dir / "plan.json"
    if not plan_path.exists():
        logger.error(f"Cannot load scientist plan: {plan_path} not found")
        return None
    try:
        plan: dict[str, Any] = json.loads(plan_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Cannot load scientist plan: {e}")
        return None

    logger.info(f"Loaded scientist plan from {plan_path}")
    return plan


def load_final_plan_from_disk(version_dir: Path, state: ExperimentState) -> dict[str, Any] | None:
    """Load the final (post-debate) plan from disk and replay prediction updates.

    Used when resuming from coder: the plan on disk is the post-debate
    revision.  Prediction updates are applied only if they are not
    already present in state (rewind_run preserves them on coder resume).
    """
    plan_path = version_dir / "plan.json"
    if not plan_path.exists():
        logger.error(f"Cannot load final plan: {plan_path} not found")
        return None
    try:
        plan: dict[str, Any] = json.loads(plan_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Cannot load final plan: {e}")
        return None

    existing = {
        p.pred_id for p in state.prediction_history if p.iteration_prescribed == state.iteration
    }
    if not existing:
        apply_prediction_updates(plan, state)
    else:
        logger.info(
            f"Predictions for iteration {state.iteration + 1} already in state, "
            f"skipping re-application ({len(existing)} found)"
        )

    logger.info(f"Loaded final plan from {plan_path}")
    return plan


def read_run_result(version_dir: Path) -> RunResult:
    """Read run_result.json and companion files from a version directory.

    Returns a populated RunResult. If run_result.json is missing or
    malformed, returns a failure RunResult.
    """
    run_result_path = version_dir / "run_result.json"
    if not run_result_path.exists():
        logger.warning(f"run_result.json missing from {version_dir}")
        return RunResult(
            success=False,
            stderr="Coder did not produce run_result.json",
        )

    from pydantic import ValidationError

    from auto_scientist.schemas import CoderRunResult

    try:
        raw = json.loads(run_result_path.read_text())
        validated = CoderRunResult.model_validate(raw)
        data = validated.model_dump()
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to parse run_result.json: {e}")
        return RunResult(
            success=False,
            stderr=f"Failed to parse run_result.json: {e}",
        )
    except ValidationError as e:
        logger.warning(f"run_result.json schema validation failed: {e}")
        # Fall back to raw data if schema validation fails
        data = raw

    # Build stderr from error field + stderr.txt
    stderr_parts = []
    if data.get("error"):
        stderr_parts.append(data["error"])
    stderr_path = version_dir / "stderr.txt"
    if stderr_path.exists():
        try:
            stderr_parts.append(stderr_path.read_text())
        except OSError as e:
            stderr_parts.append(f"(could not read stderr.txt: {e})")

    stderr = "\n".join(stderr_parts)

    # Read stdout from results.txt
    results_path = version_dir / "results.txt"
    try:
        stdout = results_path.read_text() if results_path.exists() else ""
    except OSError as e:
        logger.warning(f"Could not read results.txt: {e}")
        stdout = ""

    # Discover output files (exclude infra files)
    output_files = [
        str(f)
        for f in version_dir.iterdir()
        if f.suffix in (".png", ".txt", ".csv", ".json") and f.name not in INFRA_FILES
    ]

    return RunResult(
        success=data.get("success", False),
        return_code=data.get("return_code", -1),
        timed_out=data.get("timed_out", False),
        stdout=stdout,
        stderr=stderr,
        output_files=output_files,
    )


def evaluate(
    run_result: RunResult | None, version_entry: VersionEntry, state: ExperimentState
) -> None:
    """Evaluate results and update the version entry."""
    if run_result is None:
        version_entry.status = "failed"
        version_entry.failure_reason = "no_result"
        state.record_failure()
        return

    if run_result.timed_out:
        version_entry.status = "failed"
        version_entry.failure_reason = "timed_out"
        state.record_failure()
        return

    if not run_result.success:
        version_entry.status = "failed"
        version_entry.failure_reason = "crash"
        state.record_failure()
        return

    # Success path
    version_entry.status = "completed"
    state.record_success()

    # Set results path if stdout was saved
    results_path = Path(version_entry.script_path).parent / "results.txt"
    if results_path.exists():
        version_entry.results_path = str(results_path)


def normalize_follows_from(raw: str | None, known_pred_ids: set[str]) -> str | None:
    """Validate a follows_from value against known prediction IDs.

    Returns the value unchanged if it matches a known pred_id,
    otherwise returns None with a warning log.
    """
    if not raw or not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()
    if raw in known_pred_ids:
        return raw
    logger.warning(f"follows_from '{raw}' is not a known pred_id; setting to null")
    return None


def apply_prediction_updates(plan: dict[str, Any], state: ExperimentState) -> None:
    """Extract testable predictions from the Scientist plan and store as pending records.

    Assigns ordinal pred_ids like "1.1", "1.2" and injects them back into the
    plan dict so the Coder can include them in HYPOTHESIS TESTS output.
    Skips malformed or empty prediction entries and carried-forward predictions
    (which already have PredictionRecords).
    """
    known_pred_ids = {r.pred_id for r in state.prediction_history if r.pred_id}

    predictions = plan.get("testable_predictions", [])
    stored = []
    for i, pred in enumerate(predictions, 1):
        if not isinstance(pred, dict):
            logger.warning(f"Prediction {i}: expected dict, got {type(pred).__name__}; skipping")
            continue
        if pred.get("_carried_forward"):
            continue
        text = pred.get("prediction", "")
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning(f"Prediction {i}: empty or invalid prediction text; skipping")
            continue
        pred_id = f"{state.iteration}.{i}"
        follows_from = normalize_follows_from(pred.get("follows_from"), known_pred_ids)
        record = PredictionRecord(
            pred_id=pred_id,
            iteration_prescribed=state.iteration,
            prediction=text,
            diagnostic=pred.get("diagnostic", ""),
            if_confirmed=pred.get("if_confirmed", ""),
            if_refuted=pred.get("if_refuted", ""),
            follows_from=follows_from,
        )
        state.prediction_history.append(record)
        known_pred_ids.add(pred_id)
        pred["pred_id"] = pred_id
        pred["follows_from"] = follows_from
        stored.append(pred_id)
    if stored:
        logger.info(f"Stored {len(stored)} predictions: {', '.join(stored)}")


def get_pending_carryforward_predictions(state: ExperimentState) -> list[dict[str, Any]]:
    """Collect pending predictions from prior iterations for carry-forward.

    Returns prediction dicts (matching testable_predictions schema) with a
    ``_carried_forward`` marker so ``apply_prediction_updates`` skips them.
    """
    current_iter = state.iteration
    carried: list[dict[str, Any]] = []
    for rec in state.prediction_history:
        if rec.outcome != "pending":
            continue
        if rec.iteration_prescribed >= current_iter:
            continue
        carried.append(
            {
                "pred_id": rec.pred_id,
                "prediction": rec.prediction,
                "diagnostic": rec.diagnostic,
                "if_confirmed": rec.if_confirmed,
                "if_refuted": rec.if_refuted,
                "follows_from": rec.follows_from,
                "_carried_forward": True,
            }
        )
    if carried:
        ids = [p["pred_id"] for p in carried]
        logger.info(
            f"Carrying forward {len(carried)} pending predictions "
            f"from prior iterations: {', '.join(ids)}"
        )
    return carried


def _normalize_pred_id(raw: str, pending: list[PredictionRecord]) -> str:
    """Resolve a raw analyst pred_id to a known pending pred_id.

    Handles common Coder/Analyst format mismatches:
    - Bracketed IDs: "[1]" -> "1", "[0.1]" -> "0.1"
    - Bare integers: "1" -> "0.1" (if unambiguous among pending)
    """
    cleaned = raw.strip().strip("[]").strip()
    if not cleaned:
        return raw
    if "." in cleaned:
        return cleaned
    # Bare integer: match against pending pred_ids ending with .{cleaned}
    candidates = [p for p in pending if p.pred_id.endswith(f".{cleaned}")]
    if len(candidates) == 1:
        return candidates[0].pred_id
    return cleaned


def _token_overlap_score(a: str, b: str) -> float:
    """Fraction of the shorter text's tokens found in the longer text."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    shorter = tokens_a if len(tokens_a) <= len(tokens_b) else tokens_b
    overlap = tokens_a & tokens_b
    return len(overlap) / len(shorter)


def resolve_prediction_outcomes(analysis: dict[str, Any] | None, state: ExperimentState) -> None:
    """Match Analyst prediction outcomes against pending records in state.

    Primary matching by pred_id with normalization (strips brackets, resolves
    bare integers). Falls back to token-overlap text matching with a minimum
    length guard.
    """
    if not analysis:
        return
    outcomes = analysis.get("prediction_outcomes", [])
    if not outcomes:
        return

    pending = [r for r in state.prediction_history if r.outcome == "pending"]
    if not pending:
        return

    pending_by_id = {r.pred_id: r for r in pending if r.pred_id}

    def _resolve(record, outcome: dict) -> bool:
        raw_outcome = outcome.get("outcome", "")
        normalized = raw_outcome.strip().lower() if isinstance(raw_outcome, str) else ""
        if normalized not in VALID_OUTCOMES:
            logger.warning(
                f"Prediction {record.pred_id}: invalid outcome '{raw_outcome}', leaving pending"
            )
            return False
        record.outcome = normalized
        record.evidence = outcome.get("evidence", "")
        record.summary = outcome.get("summary", "")
        record.iteration_evaluated = state.iteration
        return True

    for outcome in outcomes:
        if not isinstance(outcome, dict):
            logger.warning(
                f"Prediction outcome: expected dict, got {type(outcome).__name__}; skipping"
            )
            continue

        # Primary: match by pred_id (with normalization)
        raw_oid = outcome.get("pred_id", "")
        oid = _normalize_pred_id(raw_oid, pending) if raw_oid else ""
        if oid and oid in pending_by_id:
            record = pending_by_id[oid]
            if _resolve(record, outcome):
                pending_by_id.pop(oid)
                pending.remove(record)
                label = f" (normalized from '{raw_oid}')" if oid != raw_oid else ""
                logger.info(f"Prediction {oid}: resolved by ID{label} as '{record.outcome}'")
            continue

        # Fallback: text matching (minimum length guard)
        outcome_text = outcome.get("prediction", "")
        outcome_text = outcome_text.lower().strip() if isinstance(outcome_text, str) else ""
        if len(outcome_text) < 10:
            logger.warning(f"Prediction outcome text too short for text matching: '{outcome_text}'")
            continue

        # Try substring match first (original behavior), then token overlap
        best_match = None
        best_score = 0
        for record in pending:
            record_text = record.prediction.lower()
            if outcome_text in record_text or record_text in outcome_text:
                score = len(record_text)
                if score > best_score:
                    best_match = record
                    best_score = score
        if best_match is None:
            # Token overlap fallback: require clear margin over runner-up
            scored: list[tuple[float, PredictionRecord]] = []
            for record in pending:
                record_text = record.prediction.lower()
                overlap = _token_overlap_score(outcome_text, record_text)
                if overlap >= 0.4:
                    scored.append((overlap, record))
            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                best_overlap, best_candidate = scored[0]
                runner_up = scored[1][0] if len(scored) > 1 else 0.0
                if best_overlap >= runner_up + 0.15:
                    best_match = best_candidate
                    logger.debug(
                        f"Token overlap match: best={best_overlap:.2f} "
                        f"('{best_candidate.pred_id}'), "
                        f"runner_up={runner_up:.2f}"
                    )
                else:
                    ids = ", ".join(f"{s:.2f}={r.pred_id}" for s, r in scored)
                    logger.warning(f"Token overlap ambiguous (no clear margin): {ids}")
        if best_match:
            if _resolve(best_match, outcome):
                logger.info(
                    f"Prediction {best_match.pred_id}: "
                    f"resolved by text fallback as '{best_match.outcome}'"
                )
                pending.remove(best_match)
                pending_by_id.pop(best_match.pred_id, None)
        else:
            logger.warning(
                f"Prediction outcome unmatched: pred_id='{raw_oid}', text='{outcome_text[:80]}'..."
            )


def build_concern_ledger(debate_results: list) -> list[dict[str, Any]]:
    """Build a concern ledger from structured debate results.

    For each concern in the CriticOutput, attach the persona and model.
    """
    from auto_scientist.agents.debate_models import ConcernLedgerEntry, DebateResult

    ledger: list[dict[str, Any]] = []
    for result in debate_results:
        if not isinstance(result, DebateResult):
            continue

        for concern in result.critic_output.concerns:
            entry = ConcernLedgerEntry(
                claim=concern.claim,
                severity=concern.severity,
                confidence=concern.confidence,
                category=concern.category,
                persona=result.persona,
                critic_model=result.critic_model,
            )
            ledger.append(entry.model_dump())

    return ledger
