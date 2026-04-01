"""Main orchestration loop and state machine."""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from auto_scientist.config import DomainConfig
from auto_scientist.model_config import AgentModelConfig, ModelConfig
from auto_scientist.notebook import NOTEBOOK_FILENAME, append_entry, read_notebook
from auto_scientist.persistence import (
    apply_prediction_updates,
    build_concern_ledger,
    evaluate,
    get_pending_carryforward_predictions,
    load_analyst_from_disk,
    load_final_plan_from_disk,
    load_scientist_plan_from_disk,
    persist_artifact,
    persist_buffer,
    read_run_result,
    resolve_prediction_outcomes,
    restore_iterations_from_manifest,
    save_iteration_manifest,
    save_partial_panels,
)
from auto_scientist.pipeline_live import (
    PipelineLive,
    collapse_panel,
    generate_iteration_summary,
    with_summaries,
)
from auto_scientist.scheduler import wait_for_window
from auto_scientist.sdk_backend import CODEX_MODEL_OVERRIDES
from auto_scientist.state import ExperimentState, VersionEntry
from auto_scientist.summarizer import run_with_summaries, summarize_results
from auto_scientist.validation import validate_prerequisites
from auto_scientist.widgets import AGENT_STYLES, AgentPanel, console

logger = logging.getLogger(__name__)

# Default from DomainConfig, used when self.config is None (e.g. tests)
_DEFAULT_RUN_TIMEOUT = DomainConfig.model_fields["run_timeout_minutes"].default


class Orchestrator:
    """Drives the Ingestion -> Iteration -> Report pipeline.

    State machine phases:
        INGESTION -> ANALYZE -> PLAN -> STOP_CHECK -> (DEBATE) ->
        IMPLEMENT -> VALIDATE -> RUN -> EVALUATE -> ANALYZE (loop) or STOP
    """

    def __init__(
        self,
        state: ExperimentState,
        data_path: Path | None,
        output_dir: Path,
        max_iterations: int = 20,
        model_config: ModelConfig | None = None,
        interactive: bool = False,
        max_consecutive_failures: int = 3,
        verbose: bool = False,
        skip_to_agent: str | None = None,
        restored_panels: list[dict] | None = None,
    ):
        self.state = state
        self.data_path = data_path.resolve() if data_path else data_path
        self.output_dir = output_dir.resolve()
        self.max_iterations = max_iterations
        self.model_config = model_config or ModelConfig.builtin_preset("default")
        self.interactive = interactive
        self.max_consecutive_failures = max_consecutive_failures
        self.config: DomainConfig | None = None
        self.verbose = verbose
        self._live: PipelineLive = PipelineLive()
        self.pause_requested: bool = False
        self.skip_to_report: bool = False
        self._skip_to_agent: str | None = skip_to_agent
        self._restored_panels: list[dict] | None = restored_panels

    @property
    def _summary_model(self) -> str | None:
        """Return the summarizer model name, or None if not configured."""
        if self.model_config.summarizer is None:
            return None
        return self.model_config.summarizer.model

    async def run(self) -> None:
        """Execute the full orchestration loop."""
        from auto_scientist.log_setup import setup_file_logging

        validate_prerequisites(
            self.state, self.data_path, self.output_dir, self.model_config, self.config
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_file_logging(self.output_dir, verbose=self.verbose)
        logger.info(
            f"Run started: output_dir={self.output_dir}, "
            f"defaults={self.model_config.defaults.model}, "
            f"max_iterations={self.max_iterations}, "
            f"critics={len(self.model_config.critics)}"
        )

        state_path = self.output_dir / "state.json"

        # Persist model config for resume
        mc_path = self.output_dir / "model_config.json"
        mc_path.write_text(self.model_config.model_dump_json(indent=2))

        # In app mode, PipelineApp already created and wired _live.
        # In headless mode (smoke test, direct call), create our own.
        if self._live._app is None:
            self._live = PipelineLive()
            self._live.start(log_path=self.output_dir / "console.log")
            console.print(self._build_startup_banner())
        else:
            # Open log file now that the output dir exists
            self._live.start(log_path=self.output_dir / "console.log")
            self._live.mount_banner(self._build_startup_banner())

        # Expose max_iterations to the metrics bar from the start
        self._live.update_status(max_iterations=self.max_iterations)

        # Restore previous iterations from manifest (for fork / resume)
        restore_iterations_from_manifest(self._live, self.output_dir)

        try:
            # Phase 0: Ingestion (with its own border)
            if self.state.phase == "ingestion":
                self._live.start_iteration("Ingestion")

                self.state.raw_data_path = self.state.data_path
                self.state.save(state_path)

                try:
                    canonical_data_dir = await self._run_ingestion()
                except Exception:
                    try:
                        self._live.end_iteration("failed", "red", "")
                        self._live.flush_completed()
                    except Exception:
                        logger.warning("Failed to finalize iteration box", exc_info=True)
                    raise

                if canonical_data_dir is None:
                    self.state.phase = "stopped"
                    self.state.save(state_path)
                    iter_summary = await generate_iteration_summary(self._live, self._summary_model)
                    save_iteration_manifest(
                        self._live,
                        self.state,
                        self.output_dir,
                        "Ingestion",
                        "failed (ingestor error)",
                        "red",
                        iter_summary,
                    )
                    self._live.end_iteration("failed (ingestor error)", "red", iter_summary)
                    self._live.flush_completed()
                else:
                    self.state.data_path = str(canonical_data_dir)
                    self.data_path = canonical_data_dir

                    # Load config if produced by ingestor
                    config_path = self.output_dir / "domain_config.json"
                    if config_path.exists():
                        config_data = json.loads(config_path.read_text())
                        self.config = DomainConfig.model_validate(config_data)
                        self.state.config_path = str(config_path)

                    self.state.phase = "iteration"
                    self.state.save(state_path)
                    logger.info("Ingestion complete, entering iteration phase")

                    iter_summary = await generate_iteration_summary(self._live, self._summary_model)
                    save_iteration_manifest(
                        self._live,
                        self.state,
                        self.output_dir,
                        "Ingestion",
                        "done",
                        "green",
                        iter_summary,
                    )
                    self._live.end_iteration("done", "green", iter_summary)
                    self._live.flush_completed()

            # Phase 1: Unified iteration loop
            while self.state.phase == "iteration":
                if self.state.should_stop_on_failures(self.max_consecutive_failures):
                    self._live.add_rule(
                        Rule(
                            f"Hit {self.max_consecutive_failures} consecutive failures. Stopping.",
                            style="red",
                        )
                    )
                    self._live.flush_completed()
                    self.state.phase = "stopped"
                    self.state.save(state_path)
                    break

                # When the latest version has no results, the analyst
                # cannot proceed and would just cascade into another
                # failure. Stop before wasting a doomed iteration.
                # When results exist, let the threshold above govern.
                if self.state.consecutive_failures > 0:
                    latest = self.state.versions[-1] if self.state.versions else None
                    if latest and latest.results_path is None:
                        self.state.phase = "stopped"
                        self.state.save(state_path)
                        break

                if self.state.iteration >= self.max_iterations:
                    self._live.add_rule(
                        Rule(
                            f"Reached max iterations ({self.max_iterations}). Stopping.",
                            style="yellow",
                        )
                    )
                    self._live.flush_completed()
                    self.state.phase = "report"
                    break

                if self.pause_requested:
                    self._live.add_rule(Rule("Paused by user.", style="yellow"))
                    self._live.flush_completed()
                    self.state.phase = "report"
                    break

                if self.skip_to_report:
                    self._live.add_rule(Rule("Skipping to report.", style="yellow"))
                    self._live.flush_completed()
                    self.state.phase = "report"
                    break

                # Schedule check
                await wait_for_window(self.state.schedule)

                try:
                    await self._run_iteration_body()
                except Exception:
                    # Catches orchestration errors (state, evaluation, summary).
                    # Individual agent failures are handled inside
                    # _run_iteration_body via _fail_iteration().
                    try:
                        self._live.end_iteration("failed", "red", "")
                        self._live.flush_completed()
                    except Exception:
                        logger.warning("Failed to finalize iteration box", exc_info=True)
                    raise
                self.state.save(state_path)

            # Phase 2: Report (with its own border)
            if self.state.phase == "report":
                self._live.start_iteration("Report")

                # Resolve pending predictions for the final version
                if self.state.versions:
                    await self._resolve_final_predictions()
                    self.state.save(state_path)

                try:
                    report_ok = await self._run_report()
                except Exception:
                    try:
                        self._live.end_iteration("failed", "red", "")
                        self._live.flush_completed()
                    except Exception:
                        logger.warning("Failed to finalize iteration box", exc_info=True)
                    raise
                self.state.phase = "stopped"
                self.state.save(state_path)

                if report_ok:
                    label, style = "done", "green"
                else:
                    label, style = "failed (report error)", "red"
                iter_summary = await generate_iteration_summary(self._live, self._summary_model)
                save_iteration_manifest(
                    self._live,
                    self.state,
                    self.output_dir,
                    "Report",
                    label,
                    style,
                    iter_summary,
                )
                self._live.end_iteration(label, style, iter_summary)
                self._live.flush_completed()

            logger.info("Run finished successfully")
        finally:
            self._live.wait_for_dismiss()
            self._live.stop()
            try:
                from auto_scientist.sdk_backend import close_all_backends

                await close_all_backends()
            except Exception:
                logger.debug("Backend cleanup failed", exc_info=True)
            try:
                from auto_scientist.sdk_backend import cleanup_sessions

                cleanup_sessions()
            except Exception:
                logger.debug("Session cleanup failed", exc_info=True)

    async def _run_ingestion(self) -> Path | None:
        """Phase 0: Canonicalize raw data into experiments/data/."""
        from auto_scientist.agents.ingestor import run_ingestor

        # On resume, use raw_data_path (original); on fresh run, use data_path
        source_path = Path(self.state.raw_data_path) if self.state.raw_data_path else self.data_path
        if source_path is None:
            raise ValueError(
                "Cannot run ingestion without a data path. "
                "Provide --data when starting a new experiment."
            )

        config_path = self.output_dir / "domain_config.json"

        cfg = self.model_config.resolve("ingestor")
        panel = AgentPanel(
            name="Ingestor",
            model=self._display_model(cfg),
            style=AGENT_STYLES.get("Ingestor", "bright_red"),
        )
        self._live.add_panel(panel)
        self._live.update_status(phase="INGESTION")

        async def _ingestor_coro(buf):
            return await run_ingestor(
                raw_data_path=source_path,
                output_dir=self.output_dir,
                goal=self.state.goal,
                interactive=self.interactive,
                config_path=config_path,
                model=cfg.model,
                message_buffer=buf,
                provider=cfg.provider,
            )

        buffer: list[str] = []
        try:
            canonical_data_dir: Path = await with_summaries(
                _ingestor_coro,
                "Ingestor",
                buffer,
                panel=panel,
                live=self._live,
                summary_model=self._summary_model,
            )
            data_files = sorted(canonical_data_dir.iterdir())
            file_list = ", ".join(f.name for f in data_files)
            collapse_panel(
                panel,
                self._live,
                self._summary_model,
                f"Canonicalized {len(data_files)} files: {file_list}",
            )
        except Exception as e:
            logger.exception(f"Ingestor error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
            raise
        finally:
            persist_buffer(self.output_dir, "ingestor", buffer, self.state.iteration)

        return canonical_data_dir

    async def _fail_iteration(
        self,
        label: str,
        failure_reason: Literal["timed_out", "crash", "no_script", "no_result"] | None = None,
    ) -> None:
        """Record a failed iteration and finalize the TUI with a red border."""
        version = f"v{self.state.iteration:02d}"
        version_entry = VersionEntry(
            version=version,
            iteration=self.state.iteration,
            script_path="",
            hypothesis="",
            status="failed",
            failure_reason=failure_reason,
        )
        self.state.record_failure()
        self.state.record_version(version_entry)
        logger.info(f"Iteration {self.state.iteration + 1} complete: status={label}")
        iter_summary = await generate_iteration_summary(self._live, self._summary_model)
        save_iteration_manifest(
            self._live,
            self.state,
            self.output_dir,
            self.state.iteration,
            label,
            "red",
            iter_summary,
        )
        self._live.end_iteration(label, "red", iter_summary)
        self._live.flush_completed()
        self.state.iteration += 1

    async def _run_iteration_body(self) -> None:
        """Run one iteration of the pipeline (inlined, not _run_iteration)."""
        from auto_scientist.resume import AGENT_ORDER

        logger.info(f"=== Iteration {self.state.iteration + 1} start ===")
        self._live.start_iteration(self.state.iteration + 1, max_iterations=self.max_iterations)
        self._live.update_status(
            iteration=self.state.iteration + 1, max_iterations=self.max_iterations
        )

        version = f"v{self.state.iteration:02d}"
        version_dir = self.output_dir / version

        # Consume skip_to_agent (only applies to the first iteration after resume)
        skip_to = self._skip_to_agent
        self._skip_to_agent = None
        # Build the set of agents to skip (everything before skip_to)
        if skip_to:
            skip_idx = AGENT_ORDER.index(skip_to)
            agents_to_skip = set(AGENT_ORDER[:skip_idx])
        else:
            agents_to_skip = set()

        # Mount TUI panels for agents loaded from disk (with full original stats)
        if self._restored_panels:
            for panel_data in self._restored_panels:
                self._live.mount_restored_panel(panel_data)
            self._restored_panels = None

        # Step 1: Analyst observes latest results (or raw data on iteration 0)
        if "analyst" in agents_to_skip:
            analysis = load_analyst_from_disk(version_dir, self.state)
        else:
            analysis = await self._run_analyst()

        if analysis is None:
            await self._fail_iteration("failed (analyst error)")
            return

        if "analyst" not in agents_to_skip:
            # Persist analysis for audit trail
            persist_artifact(version_dir, "analysis.json", analysis)

            # Apply domain_knowledge from Analyst if present
            if analysis.get("domain_knowledge"):
                self.state.domain_knowledge = analysis["domain_knowledge"]
                logger.info("Domain knowledge updated from Analyst")

            # Resolve any pending prediction outcomes from the Analyst
            resolve_prediction_outcomes(analysis, self.state)
            save_partial_panels(self._live, version_dir)

        # Step 2: Scientist plans next iteration
        if "scientist" in agents_to_skip:
            plan = load_scientist_plan_from_disk(version_dir)
        else:
            plan = await self._run_scientist_plan(analysis)

        if plan is None:
            await self._fail_iteration("failed (scientist error)")
            return

        if "scientist" not in agents_to_skip:
            save_partial_panels(self._live, version_dir)

        # Step 3: Stop gate (if Scientist recommends stopping)
        # Skip entirely when resuming past the stop gate (debate or later)
        if plan and plan.get("should_stop") and "debate" not in agents_to_skip:
            revised_stop_plan = await self._run_stop_gate(plan, analysis, version_dir)

            if revised_stop_plan is None:
                # Gate crashed - cannot validate the stop, continue investigating
                logger.error(
                    "Stop gate failed to produce a verdict. "
                    "Treating as 'stop not validated' and continuing investigation."
                )
                self._live.log(
                    "STOP GATE: error during validation - stop proposal NOT upheld, continuing"
                )
                plan["should_stop"] = False
            elif revised_stop_plan.get("should_stop"):
                # Stop upheld after the gate
                stop_msg = revised_stop_plan.get("stop_reason", "unknown")
                logger.info(f"Scientist stop upheld: {stop_msg}")
                self.state.phase = "report"
                iter_summary = await generate_iteration_summary(self._live, self._summary_model)
                stop_label = f"stopped: {revised_stop_plan.get('stop_reason', 'unknown')}"
                save_iteration_manifest(
                    self._live,
                    self.state,
                    self.output_dir,
                    self.state.iteration,
                    stop_label,
                    "yellow",
                    iter_summary,
                )
                self._live.end_iteration(stop_label, "yellow", iter_summary)
                self._live.flush_completed()
                return
            else:
                # Stop withdrawn - use the new plan and fall through to normal debate
                logger.info("Scientist withdrew stop after stop gate, continuing")
                plan = revised_stop_plan
            save_partial_panels(self._live, version_dir)

        # Step 4: Debate + Revision
        if "debate" in agents_to_skip and "revision" in agents_to_skip:
            # Both done - load the final (post-revision) plan from disk
            final_plan = load_final_plan_from_disk(version_dir, self.state)
            if final_plan is None:
                await self._fail_iteration("failed (could not load plan from disk)")
                return
        elif "debate" in agents_to_skip:
            # Debate done but revision needs to re-run
            final_plan = await self._resume_from_revision(version_dir, analysis)
            if final_plan is None:
                await self._fail_iteration("failed (revision error on resume)")
                return
        else:
            debate_result = await self._run_debate(plan, analysis)
            revised_plan = await self._run_scientist_revision(plan, debate_result, analysis)
            final_plan = revised_plan or plan

            # Persist debate transcript with original plan for context
            if debate_result:
                from auto_scientist.agents.debate_models import DebateResult

                serialized_results = [
                    r.model_dump() if isinstance(r, DebateResult) else r for r in debate_result
                ]
                concern_ledger = build_concern_ledger(debate_result)
                persist_artifact(
                    version_dir,
                    "debate.json",
                    {
                        "original_plan": plan,
                        "debate_results": serialized_results,
                        "concern_ledger": concern_ledger,
                    },
                )

            # Persist revision artifact (marks revision as completed for resume detection)
            persist_artifact(version_dir, "revision_plan.json", final_plan)

            # Apply prediction updates from the final plan (after debate revision)
            if final_plan:
                apply_prediction_updates(final_plan, self.state)

            # Persist the final plan (post-debate revision if applicable).
            # NOTE: this must happen BEFORE carry-forward injection below,
            # so plan.json reflects only the Scientist's own predictions.
            # Carry-forward entries are ephemeral (re-derived on resume).
            if final_plan:
                persist_artifact(version_dir, "plan.json", final_plan)

        if "debate" not in agents_to_skip or "revision" not in agents_to_skip:
            save_partial_panels(self._live, version_dir)

        # Carry forward pending predictions from prior iterations so the
        # Coder includes them in HYPOTHESIS TESTS and the Analyst can evaluate them.
        if final_plan:
            carryforward = get_pending_carryforward_predictions(self.state)
            if carryforward:
                existing = final_plan.get("testable_predictions", [])
                final_plan["testable_predictions"] = existing + carryforward

        # Step 5: Coder implements and runs the plan
        new_script = await self._run_coder(final_plan)

        if new_script is None:
            await self._fail_iteration("failed (no script)", failure_reason="no_script")
            return

        # Step 6: Read run result from Coder's output files
        run_result = read_run_result(new_script.parent)

        # Log run result
        if run_result.timed_out:
            self._live.log("RUN RESULT: timed out")
        elif not run_result.success:
            self._live.log(f"RUN RESULT: failed (rc={run_result.return_code})")
        else:
            self._live.log(f"RUN RESULT: success ({len(run_result.output_files)} output files)")

        # Results summary
        if self._summary_model and run_result.success:
            results_path = new_script.parent / "results.txt"
            if results_path.exists():
                try:
                    results_text = results_path.read_text()
                    summary = await summarize_results(
                        results_text,
                        self._summary_model,
                    )
                    if summary:
                        self._live.log(f"Results: {summary}")
                except Exception as e:
                    logger.warning(f"SUMMARY: error summarizing results: {e}")

        # Step 7: Evaluate
        version_entry = VersionEntry(
            version=version,
            iteration=self.state.iteration,
            script_path=str(new_script),
            hypothesis=final_plan.get("hypothesis", "") if final_plan else "",
        )
        evaluate(run_result, version_entry, self.state)
        self.state.record_version(version_entry)

        # Iteration border color: green=completed, red=failed
        status_style = "red" if version_entry.status != "completed" else "green"
        iter_summary = await generate_iteration_summary(self._live, self._summary_model)
        save_iteration_manifest(
            self._live,
            self.state,
            self.output_dir,
            self.state.iteration,
            version_entry.status,
            status_style,
            iter_summary,
        )
        self._live.end_iteration(version_entry.status, status_style, iter_summary)
        self._live.flush_completed()

        # Increment at end of loop body
        logger.info(f"Iteration {self.state.iteration + 1} complete: status={version_entry.status}")
        self.state.iteration += 1

    @staticmethod
    def _display_model(cfg: AgentModelConfig) -> str:
        """Return the model name that will actually be used at runtime.

        SDK-mode agents go through the Codex backend, which silently swaps
        unsupported models (see CODEX_MODEL_OVERRIDES).  Show the effective
        model so the banner isn't misleading.
        """
        model = cfg.model
        if cfg.mode == "sdk":
            model = CODEX_MODEL_OVERRIDES.get(model, model)
        return model

    def _build_startup_banner(self) -> Panel:
        """Build the Rich startup banner with run config and model info."""
        mc = self.model_config
        goal_preview = " ".join(self.state.goal.split())

        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("key", style="bold", no_wrap=True)
        table.add_column("value", ratio=1, overflow="fold")

        table.add_row("Input", str(self.data_path) if self.data_path else "(none)")
        table.add_row("Output", str(self.output_dir))
        table.add_row("Goal", goal_preview)
        table.add_row("", "")  # spacer

        # Agent models
        agent_map = [
            ("Ingestor", "ingestor"),
            ("Analyst", "analyst"),
            ("Scientist", "scientist"),
        ]
        for display_name, field_name in agent_map:
            cfg = mc.resolve(field_name)
            style = AGENT_STYLES.get(display_name, "")
            table.add_row(
                Text(display_name, style=style),
                Text(f"{self._display_model(cfg)}  [{cfg.reasoning.level}]", style=style),
            )

        for i, critic in enumerate(mc.critics):
            label = f"Critic {i + 1}" if len(mc.critics) > 1 else "Critic"
            table.add_row(
                Text(label, style="yellow"),
                Text(
                    f"{critic.provider}:{self._display_model(critic)}  [{critic.reasoning.level}]",
                    style="yellow",
                ),
            )

        # Coder + Report go after critics
        for display_name, field_name in [("Coder", "coder"), ("Report", "report")]:
            cfg = mc.resolve(field_name)
            style = AGENT_STYLES.get(display_name, "")
            table.add_row(
                Text(display_name, style=style),
                Text(f"{self._display_model(cfg)}  [{cfg.reasoning.level}]", style=style),
            )

        if mc.summarizer:
            s = mc.summarizer
            table.add_row(
                Text("Summarizer", style="dim"),
                Text(f"{s.provider}:{self._display_model(s)}  [{s.reasoning.level}]", style="dim"),
            )

        return Panel(table, title="Auto-Scientist", title_align="left", border_style="bold")

    def _notebook_content(self) -> str:
        """Return notebook content."""
        return read_notebook(self.output_dir / NOTEBOOK_FILENAME)

    async def _run_analyst_initial(self) -> dict[str, Any] | None:
        """Iteration 0: analyze raw canonical data instead of experiment results."""
        from auto_scientist.agents.analyst import run_analyst

        notebook_path = self.output_dir / NOTEBOOK_FILENAME
        domain_knowledge = self.state.domain_knowledge
        cfg = self.model_config.resolve("analyst")

        panel = AgentPanel(
            name="Analyst",
            model=self._display_model(cfg),
            style=AGENT_STYLES.get("Analyst", "green"),
        )
        self._live.add_panel(panel)
        self._live.update_status(phase="ANALYZE")

        buffer: list[str] = []
        try:

            async def _analyst_coro(buf):
                return await run_analyst(
                    results_path=None,
                    plot_paths=[],
                    notebook_path=notebook_path,
                    domain_knowledge=domain_knowledge,
                    data_dir=self.data_path,
                    model=cfg.model,
                    message_buffer=buf,
                    provider=cfg.provider,
                )

            analysis: dict[str, Any] | None = await with_summaries(
                _analyst_coro,
                "Analyst",
                buffer,
                panel=panel,
                live=self._live,
                summary_model=self._summary_model,
            )
            logger.info("Analyst initial: data characterization complete")
            collapse_panel(panel, self._live, self._summary_model, "Data characterization complete")
            return analysis
        except Exception as e:
            logger.exception(f"Analyst initial error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
            return None
        finally:
            persist_buffer(self.output_dir, "analyst", buffer, self.state.iteration)

    async def _run_analyst(self) -> dict[str, Any] | None:
        """Invoke the Analyst agent on latest results + plots."""
        from auto_scientist.agents.analyst import run_analyst

        if not self.state.versions:
            # Iteration 0: analyze raw data instead of experiment results
            return await self._run_analyst_initial()

        latest = self.state.versions[-1]
        notebook_path = self.output_dir / NOTEBOOK_FILENAME
        domain_knowledge = self.state.domain_knowledge

        # Find results file (resolve to absolute for agent cwd consistency)
        results_path = Path(latest.results_path).resolve() if latest.results_path else None

        # Build timeout context if previous version timed out
        timeout_context: dict[str, Any] | None = None
        if latest.failure_reason == "timed_out":
            run_timeout = self.config.run_timeout_minutes if self.config else _DEFAULT_RUN_TIMEOUT
            timeout_context = {
                "timeout_minutes": run_timeout,
                "hypothesis": latest.hypothesis,
            }

        if not results_path or not results_path.exists():
            if timeout_context is None:
                self._live.log("ANALYZE: skipped (no results file)")
                return None
            # Timeout: check for partial results/plots in the version directory
            version_dir = Path(latest.script_path).parent if latest.script_path else None
            if version_dir:
                partial_results = version_dir / "results.txt"
                results_path = partial_results.resolve() if partial_results.exists() else None

        # Find plot PNGs in the version directory
        if latest.script_path:
            version_dir = Path(latest.script_path).parent
        else:
            version_dir = notebook_path.parent
        plot_paths = sorted(version_dir.glob("*.png"))

        cfg = self.model_config.resolve("analyst")

        panel = AgentPanel(
            name="Analyst",
            model=self._display_model(cfg),
            style=AGENT_STYLES.get("Analyst", "green"),
        )
        self._live.add_panel(panel)
        self._live.update_status(phase="ANALYZE")

        buffer: list[str] = []
        try:

            async def _analyst_coro(buf):
                return await run_analyst(
                    results_path=results_path,
                    plot_paths=plot_paths,
                    notebook_path=notebook_path,
                    domain_knowledge=domain_knowledge,
                    model=cfg.model,
                    message_buffer=buf,
                    provider=cfg.provider,
                    timeout_context=timeout_context,
                )

            analysis: dict[str, Any] = await with_summaries(
                _analyst_coro,
                "Analyst",
                buffer,
                panel=panel,
                live=self._live,
                summary_model=self._summary_model,
            )
            logger.info(
                f"Analyst complete: data_summary={'yes' if analysis.get('data_summary') else 'no'}"
            )
            collapse_panel(panel, self._live, self._summary_model, "Analysis complete")
            return analysis
        except Exception as e:
            logger.exception(f"Analyst error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
            return None
        finally:
            persist_buffer(self.output_dir, "analyst", buffer, self.state.iteration)

    async def _run_scientist_plan(self, analysis: dict | None) -> dict[str, Any] | None:
        """Invoke the Scientist agent to formulate a plan."""
        from auto_scientist.agents.scientist import run_scientist

        version = f"v{self.state.iteration:02d}"
        notebook_path = self.output_dir / NOTEBOOK_FILENAME
        domain_knowledge = self.state.domain_knowledge

        cfg = self.model_config.resolve("scientist")

        panel = AgentPanel(
            name="Scientist",
            model=self._display_model(cfg),
            style=AGENT_STYLES.get("Scientist", "cyan"),
        )
        self._live.add_panel(panel)
        self._live.update_status(phase="PLAN")

        buffer: list[str] = []
        try:

            async def _scientist_coro(buf):
                return await run_scientist(
                    analysis=analysis or {},
                    notebook_path=notebook_path,
                    version=version,
                    domain_knowledge=domain_knowledge,
                    prediction_history=self.state.prediction_history,
                    model=cfg.model,
                    message_buffer=buf,
                    goal=self.state.goal,
                    provider=cfg.provider,
                    reasoning=cfg.reasoning,
                    output_dir=self.output_dir,
                )

            plan: dict[str, Any] = await with_summaries(
                _scientist_coro,
                "Scientist",
                buffer,
                panel=panel,
                live=self._live,
                summary_model=self._summary_model,
            )

            # Write the notebook entry from the plan
            if plan.get("notebook_entry"):
                append_entry(notebook_path, plan["notebook_entry"], version, "scientist")

            logger.info(
                f"Scientist plan: strategy={plan.get('strategy', '?')}, "
                f"changes={len(plan.get('changes', []))}, "
                f"hypothesis={plan.get('hypothesis', '?')[:100]}, "
                f"should_stop={plan.get('should_stop', False)}"
            )
            collapse_panel(
                panel,
                self._live,
                self._summary_model,
                f"strategy={plan.get('strategy', '?')}, changes={len(plan.get('changes', []))}",
            )
            return plan
        except Exception as e:
            logger.exception(f"Scientist plan error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
            return None
        finally:
            persist_buffer(self.output_dir, "scientist", buffer, self.state.iteration)

    async def _run_stop_gate(
        self,
        plan: dict,
        analysis: dict | None,
        version_dir: Path,
    ) -> dict[str, Any] | None:
        """Run the stop gate: assessment, stop debate, and stop revision.

        Returns the revised plan dict. If should_stop is still true, the stop
        is upheld. If should_stop is false, the plan contains a real experiment.
        Returns None if the gate encounters an error (stop is NOT upheld;
        investigation continues as a safety measure).
        """
        from auto_scientist.agents.stop_gate import (
            run_completeness_assessment,
            run_scientist_stop_revision,
        )
        from auto_scientist.prompts.stop_gate import STOP_PERSONAS

        stop_reason = plan.get("stop_reason", "unknown")
        version = f"v{self.state.iteration:02d}"
        notebook_path = self.output_dir / NOTEBOOK_FILENAME

        # --- Step 3a: Completeness Assessment ---
        cfg = self.model_config.resolve("assessor")
        panel = AgentPanel(
            name="Assessor",
            model=self._display_model(cfg),
            style=AGENT_STYLES.get("Assessor", "blue"),
        )
        self._live.add_panel(panel)
        self._live.update_status(phase="ASSESS")

        buffer: list[str] = []
        try:

            async def _assess_coro(buf):
                return await run_completeness_assessment(
                    goal=self.state.goal,
                    stop_reason=stop_reason,
                    notebook_path=notebook_path,
                    domain_knowledge=self.state.domain_knowledge,
                    prediction_history=self.state.prediction_history,
                    model=cfg.model,
                    message_buffer=buf,
                    provider=cfg.provider,
                    output_dir=self.output_dir,
                )

            assessment: dict[str, Any] = await with_summaries(
                _assess_coro,
                "Completeness Assessment",
                buffer,
                panel=panel,
                live=self._live,
                summary_model=self._summary_model,
            )
            collapse_panel(
                panel,
                self._live,
                self._summary_model,
                f"coverage={assessment.get('overall_coverage', '?')}",
            )
            persist_artifact(version_dir, "completeness_assessment.json", assessment)
        except Exception as e:
            logger.exception(f"Completeness assessment error: {e}")
            logger.error("Assessment failure aborts stop gate. Debate and revision skipped.")
            panel.error(str(e))
            self._live.collapse_panel(panel)
            return None  # Error -> stop not validated, investigation continues
        finally:
            persist_buffer(self.output_dir, "completeness_assessment", buffer, self.state.iteration)

        # --- Step 3b: Stop Debate ---
        if not self.model_config.critics:
            self._live.log("STOP DEBATE: skipped (no critics configured)")
            debate_results: list = []
        else:
            import asyncio
            import contextlib

            from auto_scientist.agents.stop_gate import run_single_stop_debate

            analysis_json = json.dumps(analysis, indent=2) if analysis else ""
            from auto_scientist.agents.scientist import _format_predictions_for_prompt

            prediction_history_text = _format_predictions_for_prompt(
                self.state.prediction_history,
            )
            notebook_content = self._notebook_content()

            self._live.update_status(phase="STOP_DEBATE")

            # Create per-persona buffers and panels
            stop_buffers: dict[str, list[str]] = {}
            stop_panels: dict[str, AgentPanel] = {}
            collectors: dict[str, list[tuple[str, str, str]]] = {}
            for i, persona in enumerate(STOP_PERSONAS):
                name = persona["name"]
                stop_buffers[name] = []
                collectors[name] = []
                config_idx = i % len(self.model_config.critics)
                critic_cfg = self.model_config.critics[config_idx]
                critic_label = f"{critic_cfg.provider}:{self._display_model(critic_cfg)}"
                stop_panel = AgentPanel(
                    name=f"Critic/{name}",
                    model=critic_label,
                    style=AGENT_STYLES.get("Critic", "yellow"),
                )
                stop_panels[name] = stop_panel
                self._live.add_panel(stop_panel)

            seen: dict[str, int] = {p["name"]: 0 for p in STOP_PERSONAS}

            def _flush_collectors():
                for name in seen:
                    entries = collectors[name]
                    for _agent_name, summary, time_label in entries[seen[name] :]:
                        stop_panels[name].add_line(f"[{time_label}] {summary}")
                    seen[name] = len(entries)
                self._live.refresh()

            async def _drain_loop():
                while True:
                    await asyncio.sleep(0.5)
                    _flush_collectors()

            summary_model = self._summary_model

            def _collapse_persona(name, result):
                panel = stop_panels[name]
                panel.set_stats(
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    thinking_tokens=result.thinking_tokens,
                    num_turns=len(result.raw_transcript),
                )
                # Flush remaining summary entries
                entries = collectors[name]
                for _n, summary, time_label in entries[seen[name] :]:
                    panel.add_line(f"[{time_label}] {summary}")
                seen[name] = len(entries)

                done_entries = [e for e in collectors[name] if e[2].endswith("done")]
                if done_entries:
                    done_summary = done_entries[-1][1]
                else:
                    done_summary = result.critic_output.overall_assessment
                self._live.collapse_panel(panel, done_summary or "Critique complete")

            async def _summarized_stop_debate(persona_index, persona):
                name = persona["name"]
                config_idx = persona_index % len(self.model_config.critics)
                config = self.model_config.critics[config_idx]
                stop_buffers.setdefault(name, [])

                async def coro(buf):
                    return await run_single_stop_debate(
                        config=config,
                        stop_reason=stop_reason,
                        completeness_assessment=assessment,
                        notebook_content=notebook_content,
                        domain_knowledge=self.state.domain_knowledge,
                        message_buffer=buf,
                        persona=persona,
                        analysis_json=analysis_json,
                        prediction_history=prediction_history_text,
                        goal=self.state.goal,
                        prediction_history_records=self.state.prediction_history,
                        output_dir=self.output_dir,
                    )

                try:
                    result = await run_with_summaries(
                        coro,
                        f"Stop Debate: {name}",
                        summary_model,
                        stop_buffers[name],
                        label_prefix="",
                        summary_collector=collectors[name],
                    )
                    _collapse_persona(name, result)
                    return result
                except Exception as e:
                    stop_panels[name].error(str(e))
                    self._live.collapse_panel(stop_panels[name])
                    raise

            drain_task = asyncio.create_task(_drain_loop())
            try:
                tasks = [
                    _summarized_stop_debate(i, persona) for i, persona in enumerate(STOP_PERSONAS)
                ]
                raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                drain_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drain_task

            debate_results = []
            for persona, result in zip(STOP_PERSONAS, raw_results, strict=True):
                name = persona["name"]
                if isinstance(result, BaseException):
                    logger.error(
                        f"Stop debate failed for {name}: {result}",
                        exc_info=result,
                    )
                    if not stop_panels[name].done:
                        stop_panels[name].error(str(result))
                        self._live.collapse_panel(stop_panels[name])
                else:
                    debate_results.append(result)

            # Persist buffers regardless of success/failure for debugging
            for label, buf in stop_buffers.items():
                clean = label.replace(":", "_").replace("/", "_").replace(" ", "_")
                persist_buffer(self.output_dir, f"stop_debate_{clean}", buf, self.state.iteration)

            if not debate_results and raw_results:
                failed_msgs = [str(r) for r in raw_results if isinstance(r, BaseException)]
                logger.error(
                    f"All {len(raw_results)} stop debates failed. "
                    f"Continuing investigation (unvalidated stop). Errors: {failed_msgs}"
                )
                self._live.log(
                    f"STOP DEBATE: all {len(raw_results)} debates failed, "
                    f"continuing investigation (stop not validated)"
                )
                return None

            self._live.log(f"STOP DEBATE: received {len(debate_results)} critique(s)")

        # Build concern ledger and persist stop debate
        concern_ledger: list[dict[str, Any]] = []
        if debate_results:
            from auto_scientist.agents.debate_models import DebateResult

            serialized = [
                r.model_dump() if isinstance(r, DebateResult) else r for r in debate_results
            ]
            concern_ledger = build_concern_ledger(debate_results)
            persist_artifact(
                version_dir,
                "stop_debate.json",
                {
                    "assessment": assessment,
                    "debate_results": serialized,
                    "concern_ledger": concern_ledger,
                },
            )

        # --- Step 3c: Scientist Stop Revision ---
        self._live.update_status(phase="STOP_REVISE")
        revision_cfg = self.model_config.resolve("scientist")
        revision_panel = AgentPanel(
            name="Stop Revision",
            model=self._display_model(revision_cfg),
            style=AGENT_STYLES.get("Scientist", "cyan"),
        )
        self._live.add_panel(revision_panel)

        revision_buffer: list[str] = []
        try:

            async def _revision_coro(buf):
                return await run_scientist_stop_revision(
                    stop_reason=stop_reason,
                    completeness_assessment=assessment,
                    concern_ledger=concern_ledger,
                    analysis=analysis or {},
                    notebook_path=notebook_path,
                    version=version,
                    domain_knowledge=self.state.domain_knowledge,
                    prediction_history=self.state.prediction_history,
                    model=revision_cfg.model,
                    message_buffer=buf,
                    goal=self.state.goal,
                    provider=revision_cfg.provider,
                    output_dir=self.output_dir,
                )

            revised: dict[str, Any] = await with_summaries(
                _revision_coro,
                "Stop Revision",
                revision_buffer,
                panel=revision_panel,
                live=self._live,
                summary_model=self._summary_model,
            )

            # Write notebook entry
            if revised.get("notebook_entry"):
                append_entry(notebook_path, revised["notebook_entry"], version, "stop_revision")

            # If stop was withdrawn, write a stop-gate notebook entry
            if not revised.get("should_stop"):
                gap_names = [
                    sq["question"]
                    for sq in assessment.get("sub_questions", [])
                    if sq.get("coverage") in ("shallow", "unexplored")
                ]
                gaps_str = ", ".join(gap_names) or "N/A"
                hyp = revised.get("hypothesis", "N/A")
                it = self.state.iteration
                gate_entry = (
                    f"Stop proposal withdrawn\n\n"
                    f"The Scientist proposed stopping at iteration {it}. "
                    f"The completeness assessment identified gaps in: "
                    f"{gaps_str}.\n\n"
                    f"The stop debate challenged these gaps. The Scientist "
                    f"withdrew the stop and proposed investigating: "
                    f"{hyp}.\n\n"
                    f"These gaps must be addressed before stopping can be "
                    f"reconsidered."
                )
                append_entry(notebook_path, gate_entry, version, "stop_gate")

            # Apply prediction updates only if stop is upheld (the withdrawn
            # path falls through to normal debate which handles predictions)
            if revised.get("should_stop"):
                apply_prediction_updates(revised, self.state)

            collapse_panel(
                revision_panel,
                self._live,
                self._summary_model,
                f"should_stop={revised.get('should_stop', '?')}",
            )

            # Persist as stop_revision_plan.json (plan.json will be written by
            # the normal flow on the withdrawn path, or here on the upheld path)
            persist_artifact(version_dir, "stop_revision_plan.json", revised)
            if revised.get("should_stop"):
                persist_artifact(version_dir, "plan.json", revised)

            return revised
        except Exception as e:
            logger.exception(f"Stop revision error: {e}")
            revision_panel.error(str(e))
            self._live.collapse_panel(revision_panel)
            return None  # Error -> stop not validated, investigation continues
        finally:
            persist_buffer(self.output_dir, "stop_revision", revision_buffer, self.state.iteration)

    async def _run_debate(
        self,
        plan: dict | None,
        analysis: dict | None,
    ) -> list[dict[str, Any]] | None:
        """Send plan to critic model(s) for parallel debate with the Scientist."""
        if not self.model_config.critics or plan is None:
            self._live.log("DEBATE: skipped (no critics configured or no plan)")
            return None

        from auto_scientist.agents.critic import run_debate
        from auto_scientist.agents.scientist import _format_predictions_for_prompt

        notebook_content = self._notebook_content()
        domain_knowledge = self.state.domain_knowledge
        scientist_cfg = self.model_config.resolve("scientist")

        # Full-detail text as API-mode fallback; SDK critics override with tool hint
        analysis_json = json.dumps(analysis, indent=2) if analysis else ""
        prediction_history = _format_predictions_for_prompt(
            self.state.prediction_history,
        )

        self._live.update_status(phase="DEBATE")

        # Per-persona buffers (run_debate keys buffers by persona name)
        from auto_scientist.prompts.critic import ITERATION_0_PERSONAS, PERSONAS

        active_personas = [
            p for p in PERSONAS if self.state.iteration > 0 or p["name"] in ITERATION_0_PERSONAS
        ]

        buffers: dict[str, list[str]] = {}
        for persona in active_personas:
            buffers[persona["name"]] = []

        try:
            if self._summary_model:
                critiques = await self._run_debate_with_summaries(
                    buffers,
                    plan,
                    notebook_content,
                    domain_knowledge,
                    scientist_cfg,
                    analysis_json,
                    prediction_history,
                    self.state.goal,
                    prediction_history_records=self.state.prediction_history,
                    output_dir=self.output_dir,
                )
            else:
                critiques = await run_debate(
                    critic_configs=self.model_config.critics,
                    plan=plan,
                    notebook_content=notebook_content,
                    domain_knowledge=domain_knowledge,
                    message_buffers=buffers,
                    iteration=self.state.iteration,
                    analysis_json=analysis_json,
                    prediction_history=prediction_history,
                    goal=self.state.goal,
                    prediction_history_records=self.state.prediction_history,
                    output_dir=self.output_dir,
                )
            self._live.log(f"DEBATE: received {len(critiques)} critique(s)")
            return critiques
        except Exception as e:
            logger.exception(f"Debate error: {e}")
            self._live.log(f"DEBATE: error - {e}")
            return None
        finally:
            for label, buf in buffers.items():
                safe_name = f"debate_{label.replace(':', '_').replace('/', '_')}"
                persist_buffer(self.output_dir, safe_name, buf, self.state.iteration)

    async def _run_debate_with_summaries(
        self,
        buffers: dict[str, list[str]],
        plan: dict,
        notebook_content: str,
        domain_knowledge: str,
        scientist_cfg: Any,
        analysis_json: str,
        prediction_history: str,
        goal: str = "",
        prediction_history_records: list | None = None,
        output_dir: Path | None = None,
    ) -> list:
        """Run per-persona debates in parallel, each with its own summarizer and panel."""
        import asyncio
        import contextlib

        from auto_scientist.agents.critic import run_single_critic_debate
        from auto_scientist.agents.debate_models import DebateResult
        from auto_scientist.prompts.critic import (
            ITERATION_0_PERSONAS,
            PERSONAS,
            get_model_index_for_debate,
        )

        active_personas = [
            p for p in PERSONAS if self.state.iteration > 0 or p["name"] in ITERATION_0_PERSONAS
        ]

        summary_model = self._summary_model

        # Create per-persona panels and collectors
        panels: dict[str, AgentPanel] = {}
        collectors: dict[str, list[tuple[str, str, str]]] = {}
        persona_names = [p["name"] for p in active_personas]
        for persona in active_personas:
            name = persona["name"]
            model_idx = get_model_index_for_debate(
                active_personas.index(persona),
                self.state.iteration,
                len(self.model_config.critics),
            )
            config = self.model_config.critics[model_idx]
            label = f"{config.provider}:{self._display_model(config)}"
            panel = AgentPanel(
                name=f"Critic/{name}", model=label, style=AGENT_STYLES.get("Critic", "yellow")
            )
            panels[name] = panel
            collectors[name] = []
            self._live.add_panel(panel)

        # Track how many entries per persona have been flushed to panels
        seen: dict[str, int] = {n: 0 for n in persona_names}

        def _flush_collectors():
            for name in persona_names:
                entries = collectors[name]
                new_entries = entries[seen[name] :]
                for _agent_name, summary, time_label in new_entries:
                    panels[name].add_line(f"[{time_label}] {summary}")
                seen[name] = len(entries)
            self._live.refresh()

        async def _drain_loop():
            while True:
                await asyncio.sleep(0.5)
                _flush_collectors()

        def _collapse_persona(name: str, result: DebateResult) -> None:
            """Collapse a persona panel with stats from its result."""
            panel = panels[name]
            panel.set_stats(
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                thinking_tokens=result.thinking_tokens,
                num_turns=len(result.raw_transcript),
            )
            # Flush collector entries before collapsing
            entries = collectors[name]
            new_entries = entries[seen[name] :]
            for _agent_name, summary, time_label in new_entries:
                panel.add_line(f"[{time_label}] {summary}")
            seen[name] = len(entries)

            done_entries = [e for e in collectors[name] if e[2].endswith("done")]
            if done_entries:
                done_summary = done_entries[-1][1]
            else:
                done_summary = result.critic_output.overall_assessment
            self._live.collapse_panel(panel, done_summary or "Debate complete")

        async def _summarized_debate(persona_index, persona):
            name = persona["name"]
            model_idx = get_model_index_for_debate(
                persona_index,
                self.state.iteration,
                len(self.model_config.critics),
            )
            config = self.model_config.critics[model_idx]
            buf_key = name
            buffers.setdefault(buf_key, [])

            async def coro(buf):
                return await run_single_critic_debate(
                    config=config,
                    plan=plan,
                    notebook_content=notebook_content,
                    domain_knowledge=domain_knowledge,
                    message_buffer=buf,
                    persona=persona,
                    analysis_json=analysis_json,
                    prediction_history=prediction_history,
                    goal=goal,
                    prediction_history_records=prediction_history_records,
                    output_dir=output_dir,
                )

            try:
                result = await run_with_summaries(
                    coro,
                    f"Debate: {name}",
                    summary_model,
                    buffers[buf_key],
                    label_prefix="",
                    summary_collector=collectors[name],
                )
                _collapse_persona(name, result)
                return result
            except Exception as e:
                panels[name].error(str(e))
                self._live.collapse_panel(panels[name])
                raise

        drain_task = asyncio.create_task(_drain_loop())
        raw_results: list[DebateResult | BaseException] = []
        try:
            tasks = [_summarized_debate(i, persona) for i, persona in enumerate(active_personas)]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task

        # Log failures, return successes
        successful = []
        persona_names = [p["name"] for p in active_personas]
        for name, r in zip(persona_names, raw_results, strict=True):
            if isinstance(r, BaseException):
                logger.error(f"Critic debate failed for {name}: {r}", exc_info=r)
            else:
                successful.append(r)
        if not successful:
            failed_msgs = [str(r) for r in raw_results if isinstance(r, BaseException)]
            raise RuntimeError(
                f"All {len(raw_results)} critic debates failed. "
                f"Check API keys and network connectivity. Errors: {failed_msgs}"
            )
        if len(successful) < len(raw_results):
            lost = [
                n
                for n, r in zip(persona_names, raw_results, strict=True)
                if isinstance(r, BaseException)
            ]
            logger.warning(
                f"Debate: {len(successful)}/{len(raw_results)} debates succeeded. "
                f"Lost perspectives: {lost}"
            )
        return successful

    async def _run_scientist_revision(
        self,
        plan: dict | None,
        debate_result: list | None,
        analysis: dict | None,
    ) -> dict[str, Any] | None:
        """Scientist revises plan based on debate."""
        if plan is None or not debate_result:
            self._live.log("REVISE: skipped (no plan or no debate)")
            return None

        from auto_scientist.agents.scientist import run_scientist_revision

        version = f"v{self.state.iteration:02d}"
        notebook_path = self.output_dir / NOTEBOOK_FILENAME
        domain_knowledge = self.state.domain_knowledge

        # Build structured concern ledger from debate results
        concern_ledger = build_concern_ledger(debate_result)

        cfg = self.model_config.resolve("scientist")

        panel = AgentPanel(
            name="Revision",
            model=self._display_model(cfg),
            style=AGENT_STYLES.get("Scientist", "cyan"),
        )
        self._live.add_panel(panel)
        self._live.update_status(phase="REVISE")

        buffer: list[str] = []
        try:

            async def _revision_coro(buf):
                return await run_scientist_revision(
                    original_plan=plan,
                    concern_ledger=concern_ledger,
                    analysis=analysis or {},
                    notebook_path=notebook_path,
                    version=version,
                    domain_knowledge=domain_knowledge,
                    prediction_history=self.state.prediction_history,
                    model=cfg.model,
                    message_buffer=buf,
                    goal=self.state.goal,
                    provider=cfg.provider,
                    reasoning=cfg.reasoning,
                    output_dir=self.output_dir,
                )

            revised: dict[str, Any] = await with_summaries(
                _revision_coro,
                "Scientist Revision",
                buffer,
                panel=panel,
                live=self._live,
                summary_model=self._summary_model,
            )

            # Write revised notebook entry
            if revised.get("notebook_entry"):
                append_entry(notebook_path, revised["notebook_entry"], version, "revision")

            collapse_panel(
                panel,
                self._live,
                self._summary_model,
                f"strategy={revised.get('strategy', '?')}",
            )
            return revised
        except Exception as e:
            logger.exception(f"Scientist revision error: {e}")
            panel.error(f"{e}, using original plan")
            self._live.collapse_panel(panel)
            return None
        finally:
            persist_buffer(self.output_dir, "scientist_revision", buffer, self.state.iteration)

    async def _resume_from_revision(
        self, version_dir: Path, analysis: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Load debate data from disk and re-run only the scientist revision.

        Used when resuming from the 'revision' agent: debate completed but
        the post-debate scientist revision crashed. Uses the pre-built
        concern_ledger from debate.json (raw debate_results are serialized
        dicts, not DebateResult objects, so build_concern_ledger won't work).
        """
        from auto_scientist.agents.scientist import run_scientist_revision

        debate_path = version_dir / "debate.json"
        if not debate_path.exists():
            logger.error(f"Cannot resume revision: {debate_path} not found")
            return None

        try:
            debate_data = json.loads(debate_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Cannot load debate data for revision resume: {e}")
            return None

        original_plan = debate_data.get("original_plan")
        concern_ledger = debate_data.get("concern_ledger", [])

        if not original_plan:
            logger.error("debate.json missing original_plan, cannot resume revision")
            return None

        version = f"v{self.state.iteration:02d}"
        notebook_path = self.output_dir / NOTEBOOK_FILENAME
        cfg = self.model_config.resolve("scientist")

        panel = AgentPanel(
            name="Revision", model=cfg.model, style=AGENT_STYLES.get("Scientist", "cyan")
        )
        self._live.add_panel(panel)
        self._live.update_status(phase="REVISE")

        buffer: list[str] = []
        try:

            async def _revision_coro(buf):
                return await run_scientist_revision(
                    original_plan=original_plan,
                    concern_ledger=concern_ledger,
                    analysis=analysis or {},
                    notebook_path=notebook_path,
                    version=version,
                    domain_knowledge=self.state.domain_knowledge,
                    prediction_history=self.state.prediction_history,
                    model=cfg.model,
                    message_buffer=buf,
                    goal=self.state.goal,
                    provider=cfg.provider,
                    reasoning=cfg.reasoning,
                )

            revised: dict[str, Any] = await with_summaries(
                _revision_coro,
                "Scientist Revision",
                buffer,
                panel=panel,
                live=self._live,
                summary_model=self._summary_model,
            )

            # Write revised notebook entry
            if revised.get("notebook_entry"):
                from auto_scientist.notebook import append_entry

                append_entry(notebook_path, revised["notebook_entry"], version, "revision")

            collapse_panel(
                panel,
                self._live,
                self._summary_model,
                f"strategy={revised.get('strategy', '?')}",
            )

            final_plan = revised
        except Exception as e:
            logger.exception(f"Scientist revision error on resume: {e}")
            panel.error(f"{e}, using original plan")
            self._live.collapse_panel(panel)
            final_plan = original_plan
        finally:
            persist_buffer(self.output_dir, "scientist_revision", buffer, self.state.iteration)

        if final_plan:
            persist_artifact(version_dir, "revision_plan.json", final_plan)
            apply_prediction_updates(final_plan, self.state)
            persist_artifact(version_dir, "plan.json", final_plan)

        logger.info("Resumed from revision: scientist revision re-run complete")
        return final_plan

    async def _run_coder(self, plan: dict | None) -> Path | None:
        """Invoke the Coder agent to implement the plan."""
        from auto_scientist.agents.coder import run_coder

        if plan is None:
            self._live.log("IMPLEMENT: skipped (no plan)")
            return None

        version = f"v{self.state.iteration:02d}"
        domain_knowledge = self.state.domain_knowledge
        data_path = str(Path(self.state.data_path).resolve()) if self.state.data_path else ""

        # On iteration 0 (no previous versions), use a nonexistent path
        if self.state.versions:
            latest = self.state.versions[-1]
            previous_script = Path(latest.script_path)
        else:
            previous_script = Path("nonexistent")

        run_timeout = self.config.run_timeout_minutes if self.config else _DEFAULT_RUN_TIMEOUT
        run_cmd = self.config.run_command if self.config else "uv run {script_path}"

        cfg = self.model_config.resolve("coder")

        # Resolve the executable to an absolute path so the coder's Bash
        # subprocess can find it even when ~/.local/bin isn't in PATH.
        # Skip for Codex: its seatbelt sandbox has its own filesystem,
        # so host absolute paths (e.g. /Users/.../python3) don't exist there.
        if cfg.provider != "openai":
            parts = run_cmd.split()
            if parts:
                exe = parts[0]
                abs_exe = shutil.which(exe)
                if abs_exe and abs_exe != exe:
                    run_cmd = abs_exe + run_cmd[len(exe) :]
                elif not abs_exe:
                    logger.warning(
                        f"Executable '{exe}' not found on PATH; "
                        f"coder may fail to run the experiment script"
                    )

        # Prepend ensure_deps to auto-patch PEP 723 deps before each run.
        # For CC: invoke as a module (auto_scientist is importable).
        # For Codex: copy the script into the output dir (sandbox can't
        # import auto_scientist, but ensure_deps is pure stdlib).
        if "{script_path}" in run_cmd:
            if cfg.provider == "openai":
                import auto_scientist.ensure_deps as _ed_mod

                # Bootstrap pip and pre-install common scientific packages
                # from the HOST as a performance optimization (avoids
                # download time inside the sandbox).  The coder sandbox
                # has network access (danger-full-access) so it can also
                # pip-install at runtime via ensure_deps --install.
                try:
                    _ed_mod._ensure_pip()
                    _ed_mod._preinstall_scientific_packages()
                except Exception:
                    logger.warning(
                        "Failed to pre-install scientific packages from host; "
                        "coder will fall back to in-sandbox pip install"
                    )

                ed_src = Path(_ed_mod.__file__)
                ed_dst = self.output_dir / "_ensure_deps.py"
                shutil.copy2(ed_src, ed_dst)
                # Use absolute path so it works even if the coder cd's into a subdirectory
                run_cmd = f"python3 {ed_dst} --install {{script_path}} && {run_cmd}"
            else:
                run_cmd = (
                    f"{sys.executable} -m auto_scientist.ensure_deps {{script_path}} && {run_cmd}"
                )

        # Pre-compute data directory listing so coder doesn't waste turns
        data_dir = Path(data_path) if data_path else None
        if data_dir and data_dir.is_dir():
            data_files_listing = "\n".join(
                f"- {f.name} ({f.stat().st_size} bytes)"
                for f in sorted(data_dir.iterdir())
                if f.is_file() and not f.name.startswith(".")
            )
        else:
            data_files_listing = ""

        panel = AgentPanel(
            name="Coder",
            model=self._display_model(cfg),
            style=AGENT_STYLES.get("Coder", "magenta1"),
        )
        self._live.add_panel(panel)
        self._live.update_status(phase="IMPLEMENT")

        buffer: list[str] = []
        try:

            async def _coder_coro(buf):
                return await run_coder(
                    plan=plan,
                    previous_script=previous_script,
                    output_dir=self.output_dir,
                    version=version,
                    domain_knowledge=domain_knowledge,
                    data_path=data_path,
                    model=cfg.model,
                    message_buffer=buf,
                    run_timeout_minutes=run_timeout,
                    run_command=run_cmd,
                    data_files_listing=data_files_listing,
                    provider=cfg.provider,
                )

            new_script: Path | None = await with_summaries(
                _coder_coro,
                "Coder",
                buffer,
                panel=panel,
                live=self._live,
                summary_model=self._summary_model,
            )
            collapse_panel(panel, self._live, self._summary_model, f"Created {new_script}")
            return new_script
        except Exception as e:
            logger.exception(f"Coder error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
            self.state.record_failure()
            return None
        finally:
            persist_buffer(self.output_dir, "coder", buffer, self.state.iteration)

    async def _resolve_final_predictions(self) -> None:
        """Run the Analyst on the last version to resolve pending predictions."""
        analysis = await self._run_analyst()
        if analysis:
            resolve_prediction_outcomes(analysis, self.state)
            if self.state.versions:
                version_dir = self.output_dir / self.state.versions[-1].version
                persist_artifact(version_dir, "final_analysis.json", analysis)
        still_pending = [r for r in self.state.prediction_history if r.outcome == "pending"]
        if still_pending:
            ids = [r.pred_id or "?" for r in still_pending]
            logger.warning(
                f"{len(still_pending)} predictions still pending after "
                f"final resolution: {', '.join(ids)}"
            )

    async def _run_report(self) -> bool:
        """Phase 2: Generate final summary report."""
        from auto_scientist.agents.report import run_report

        notebook_path = self.output_dir / NOTEBOOK_FILENAME

        cfg = self.model_config.resolve("report")
        panel = AgentPanel(
            name="Report", model=self._display_model(cfg), style=AGENT_STYLES.get("Report", "blue")
        )
        self._live.add_panel(panel)
        self._live.update_status(phase="REPORT")

        buffer: list[str] = []
        try:

            async def _report_coro(buf):
                return await run_report(
                    state=self.state,
                    notebook_path=notebook_path,
                    output_dir=self.output_dir,
                    model=cfg.model,
                    message_buffer=buf,
                    provider=cfg.provider,
                )

            report_content = await with_summaries(
                _report_coro,
                "Report",
                buffer,
                panel=panel,
                live=self._live,
                summary_model=self._summary_model,
            )
            report_path = self.output_dir / "report.md"
            report_path.write_text(report_content)
            collapse_panel(panel, self._live, self._summary_model, f"Written to {report_path}")
            return True
        except Exception as e:
            logger.exception(f"Report error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
            self.state.record_failure()
            return False
        finally:
            persist_buffer(self.output_dir, "report", buffer, self.state.iteration)
