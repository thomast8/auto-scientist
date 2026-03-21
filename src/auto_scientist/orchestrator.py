"""Main orchestration loop and state machine."""

import json
import logging
import re
from pathlib import Path
from typing import Any

from auto_scientist.config import DomainConfig, SuccessCriterion
from auto_scientist.console import (
    AGENT_COLORS,
    BOLD,
    DIM,
    GREEN,
    RED,
    RESET,
    STEP_COLORS,
    YELLOW,
    _use_color,
    close_console_log,
    init_console_log,
    print_header,
    print_iteration_header,
    print_step,
    print_summary,
)
from auto_scientist.log_setup import setup_file_logging
from auto_scientist.model_config import ModelConfig
from auto_scientist.notebook import NOTEBOOK_FILENAME, append_entry, read_notebook
from auto_scientist.runner import RunResult
from auto_scientist.scheduler import wait_for_window
from auto_scientist.state import CriteriaRevision, ExperimentState, VersionEntry
from auto_scientist.summarizer import run_with_summaries, summarize_results

logger = logging.getLogger(__name__)


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
        max_consecutive_failures: int = 5,
        debate_rounds: int = 2,
        stream: bool = True,
        verbose: bool = False,
    ):
        self.state = state
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.model_config = model_config or ModelConfig.builtin_preset("default")
        self.interactive = interactive
        self.max_consecutive_failures = max_consecutive_failures
        self.debate_rounds = debate_rounds
        self.config: DomainConfig | None = None
        self.stream = stream
        self.verbose = verbose

    async def run(self) -> None:
        """Execute the full orchestration loop."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_file_logging(self.output_dir, verbose=self.verbose)
        init_console_log(self.output_dir / "console.log")
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

        goal_preview = self.state.goal[:80] + ("..." if len(self.state.goal) > 80 else "")
        fields = {
            "Output": str(self.output_dir),
            "Goal": goal_preview,
        }
        print_header("Auto-Scientist", fields)
        self._print_model_banner()

        try:
            # Phase 0: Ingestion
            if self.state.phase == "ingestion":
                self.state.raw_data_path = self.state.data_path
                self.state.save(state_path)

                canonical_data_dir = await self._run_ingestion()

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

            # Phase 1: Unified iteration loop
            while self.state.phase == "iteration":
                if self.state.iteration >= self.max_iterations:
                    print_step(
                        f"Reached max iterations ({self.max_iterations}). Stopping.",
                        color=YELLOW,
                    )
                    self.state.phase = "report"
                    break

                if self.state.should_stop_on_failures(self.max_consecutive_failures):
                    print_step(
                        f"Hit {self.max_consecutive_failures} consecutive failures. Stopping.",
                        color=RED,
                    )
                    self.state.phase = "report"
                    break

                # Schedule check
                await wait_for_window(self.state.schedule)

                await self._run_iteration_body()
                self.state.save(state_path)

            # Score the final version if it was never evaluated
            if self.state.versions and self.state.versions[-1].score is None:
                await self._score_final_version()
                self.state.save(state_path)

            # Phase 2: Report
            if self.state.phase == "report":
                await self._run_report()
                self.state.phase = "stopped"
                self.state.save(state_path)

            print_step(f"Experiment completed. Final state saved to {state_path}", color=GREEN)
            logger.info("Run finished successfully")
        finally:
            close_console_log()

    async def _run_ingestion(self) -> Path:
        """Phase 0: Canonicalize raw data into experiments/data/."""
        from auto_scientist.agents.ingestor import run_ingestor

        # On resume, use raw_data_path (original); on fresh run, use data_path
        source_path = (
            Path(self.state.raw_data_path)
            if self.state.raw_data_path
            else self.data_path
        )
        if source_path is None:
            raise ValueError(
                "Cannot run ingestion without a data path. "
                "Provide --data when starting a new experiment."
            )

        config_path = self.output_dir / "domain_config.json"

        print_step("INGESTION phase: canonicalizing raw data")

        cfg = self.model_config.resolve("ingestor")

        async def _ingestor_coro(buf):
            return await run_ingestor(
                raw_data_path=source_path,
                output_dir=self.output_dir,
                goal=self.state.goal,
                interactive=self.interactive,
                config_path=config_path,
                model=cfg.model,
                message_buffer=buf,
            )

        buffer: list[str] = []
        try:
            canonical_data_dir = await self._with_summaries(
                _ingestor_coro, "Ingestor", buffer,
            )
        finally:
            self._persist_buffer("ingestor", buffer)

        # Summary
        data_files = sorted(canonical_data_dir.iterdir())
        print_step("\nINGESTION complete:")
        for f in data_files:
            print_step(f"  {f}", color=STEP_COLORS["INGESTION"])
        print_step("")

        return canonical_data_dir

    async def _run_iteration_body(self) -> None:
        """Run one iteration of the pipeline (inlined, not _run_iteration)."""
        logger.info(f"=== Iteration {self.state.iteration} start ===")
        print_iteration_header(self.state.iteration)

        # Step 1: Analyst observes latest results (or raw data on iteration 0)
        analysis = await self._run_analyst()

        # Apply domain_knowledge from Analyst if present
        if analysis and analysis.get("domain_knowledge"):
            self.state.domain_knowledge = analysis["domain_knowledge"]
            logger.info("Domain knowledge updated from Analyst")

        # Score the evaluated version with current criteria
        self._score_latest(analysis)

        # Step 2: Scientist plans next iteration
        plan = await self._run_scientist_plan(analysis)

        # Apply criteria updates from Scientist if present
        if plan:
            self._apply_criteria_updates(plan)

        # Step 3: Check if Scientist recommends stopping
        if plan and plan.get("should_stop"):
            print_step(
                f"Scientist recommends stopping: {plan.get('stop_reason', 'unknown')}",
                color=YELLOW,
            )
            logger.info(f"Scientist stop: {plan.get('stop_reason', 'unknown')}")
            self.state.phase = "report"
            return

        # Step 4: Debate (skip on iteration 0, nothing to challenge)
        final_plan = plan
        if self.state.iteration > 0:
            debate_result = await self._run_debate(plan)
            revised_plan = await self._run_scientist_revision(plan, debate_result, analysis)
            final_plan = revised_plan or plan

        # Step 5: Coder implements and runs the plan
        new_script = await self._run_coder(final_plan)

        if new_script is None:
            # Coder failed to produce a script; record failure and move on
            version = f"v{self.state.iteration:02d}"
            version_entry = VersionEntry(
                version=version,
                iteration=self.state.iteration,
                script_path="",
                hypothesis=final_plan.get("hypothesis", "") if final_plan else "",
                status="failed",
            )
            self.state.record_failure()
            self.state.record_version(version_entry)
            print_step(f"\nITERATION {self.state.iteration} complete:", color=BOLD)
            print_step(f"  Status:     {version_entry.status}", color=RED)
            logger.info(
                f"Iteration {self.state.iteration} complete: "
                f"status=failed (coder produced no script)"
            )
            self.state.iteration += 1
            return

        # Step 6: Read run result from Coder's output files
        run_result = self._read_run_result(new_script.parent)

        if run_result.timed_out:
            print_step("  RUN RESULT: timed out", color=RED)
        elif not run_result.success:
            print_step(f"  RUN RESULT: failed (rc={run_result.return_code})", color=RED)
        else:
            print_step(
                f"  RUN RESULT: success "
                f"({len(run_result.output_files)} output files)",
                color=GREEN,
            )

        # Results summary
        if self._should_summarize() and run_result.success:
            results_path = new_script.parent / "results.txt"
            if results_path.exists():
                try:
                    results_text = results_path.read_text()
                    summary = await summarize_results(
                        results_text, self._summary_model,
                    )
                    if summary:
                        print_summary("Results", summary)
                except Exception as e:
                    logger.warning(f"SUMMARY: error summarizing results: {e}")

        # Step 7: Evaluate
        version = f"v{self.state.iteration:02d}"
        version_entry = VersionEntry(
            version=version,
            iteration=self.state.iteration,
            script_path=str(new_script),
            hypothesis=final_plan.get("hypothesis", "") if final_plan else "",
        )
        self._evaluate(run_result, version_entry)
        self.state.record_version(version_entry)

        # Iteration summary
        version_dir = new_script.parent if new_script else None
        status_color = GREEN if version_entry.status == "completed" else RED
        print_step(f"\nITERATION {self.state.iteration} complete:", color=BOLD)
        if new_script:
            print_step(f"  Script:     {new_script}", color=DIM)
        print_step(f"  Status:     {version_entry.status}", color=status_color)
        if version_entry.score is not None:
            print_step(f"  Score:      {version_entry.score}", color=BOLD)
        print_step(
            f"  Best:       {self.state.best_version} (score {self.state.best_score})",
            color=BOLD,
        )
        if version_dir and version_dir.exists():
            suffixes = (".png", ".txt", ".json")
            artifacts = sorted(
                f
                for f in version_dir.iterdir()
                if f.suffix in suffixes
                and f.name not in self._INFRA_FILES
            )
            if artifacts:
                print_step("  Outputs:", color=DIM)
                for f in artifacts:
                    print_step(f"    {f}", color=DIM)

        # Increment at end of loop body
        logger.info(
            f"Iteration {self.state.iteration} complete: "
            f"status={version_entry.status}, score={version_entry.score}, "
            f"best={self.state.best_version} ({self.state.best_score})"
        )
        self.state.iteration += 1

    def _print_model_banner(self) -> None:
        """Print a color-coded per-agent model configuration banner."""
        import sys

        use_color = _use_color()
        mc = self.model_config

        # Agent -> (color, resolved model, reasoning level)
        agent_map = {
            "Analyst": ("Analyst", "analyst"),
            "Scientist": ("Scientist", "scientist"),
            "Coder": ("Coder", "coder"),
            "Ingestor": ("Ingestor", "ingestor"),
            "Report": ("Report", "report"),
        }

        lines: list[str] = []
        for display_name, (color_key, field_name) in agent_map.items():
            cfg = mc.resolve(field_name)
            color = AGENT_COLORS.get(color_key, "")
            reasoning_label = cfg.reasoning.level
            if cfg.reasoning.budget:
                reasoning_label += f" ({cfg.reasoning.budget} tokens)"
            line = f"  {display_name:<12s}{cfg.model}  [{reasoning_label}]"
            if use_color:
                lines.append(f"{color}{line}{RESET}")
            else:
                lines.append(line)

        # Critics
        if mc.critics:
            for i, critic in enumerate(mc.critics):
                color = AGENT_COLORS.get("Critic", "")
                label = f"Critic {i + 1}" if len(mc.critics) > 1 else "Critic"
                r_label = critic.reasoning.level
                line = f"  {label:<12s}{critic.provider}:{critic.model}  [{r_label}]"
                if use_color:
                    lines.append(f"{color}{line}{RESET}")
                else:
                    lines.append(line)

        # Summarizer
        if mc.summarizer:
            r_label = mc.summarizer.reasoning.level
            s = mc.summarizer
            line = f"  {'Summarizer':<12s}{s.provider}:{s.model}  [{r_label}]"
            if use_color:
                lines.append(f"{DIM}{line}{RESET}")
            else:
                lines.append(line)

        sys.stdout.write("\n".join(lines) + "\n\n")
        sys.stdout.flush()

    def _should_summarize(self) -> bool:
        """Check if summaries are enabled."""
        return self.model_config.summarizer is not None

    @property
    def _summary_model(self) -> str:
        """Return the summarizer model name."""
        if self.model_config.summarizer is None:
            raise RuntimeError("Summarizer not configured")
        return self.model_config.summarizer.model

    async def _with_summaries(self, coro_fn, agent_name: str, message_buffer: list[str]):
        """Wrap an agent call in run_with_summaries if enabled.

        Agent errors propagate normally. Only summary infrastructure
        errors are caught here (auth errors disable future summaries).
        """
        if not self._should_summarize():
            return await coro_fn(message_buffer)
        return await run_with_summaries(
            coro_fn, agent_name, self._summary_model, message_buffer,
        )

    def _persist_buffer(
        self, agent_name: str, buffer: list[str], iteration: int | None = None,
    ) -> None:
        """Write an agent's message buffer to disk for debugging."""
        if not buffer:
            return
        buffers_dir = self.output_dir / "buffers"
        buffers_dir.mkdir(exist_ok=True)
        if iteration is None:
            iteration = self.state.iteration
        filename = f"{agent_name.lower().replace(' ', '_')}_{iteration:02d}.txt"
        (buffers_dir / filename).write_text("\n".join(buffer))

    def _notebook_content(self) -> str:
        """Return notebook content."""
        return read_notebook(self.output_dir / NOTEBOOK_FILENAME)

    async def _run_analyst_initial(self) -> dict[str, Any] | None:
        """Iteration 0: analyze raw canonical data instead of experiment results."""
        from auto_scientist.agents.analyst import run_analyst

        notebook_path = self.output_dir / NOTEBOOK_FILENAME
        domain_knowledge = self.state.domain_knowledge
        cfg = self.model_config.resolve("analyst")

        print_step("  ANALYZE: initial data characterization")
        buffer: list[str] = []
        try:
            async def _analyst_coro(buf):
                return await run_analyst(
                    results_path=None,
                    plot_paths=[],
                    notebook_path=notebook_path,
                    domain_knowledge=domain_knowledge,
                    success_criteria=self.state.success_criteria,
                    data_dir=self.data_path,
                    model=cfg.model,
                    message_buffer=buf,
                )

            analysis = await self._with_summaries(_analyst_coro, "Analyst", buffer)
            logger.info("Analyst initial: data characterization complete")
            print_step("  ANALYZE: data characterization complete")
            return analysis
        except Exception as e:
            logger.exception(f"Analyst initial error: {e}")
            print_step(f"  ANALYZE: error - {e}")
            return None
        finally:
            self._persist_buffer("analyst", buffer)

    async def _run_analyst(self) -> dict[str, Any] | None:
        """Invoke the Analyst agent on latest results + plots."""
        from auto_scientist.agents.analyst import run_analyst

        if not self.state.versions:
            # Iteration 0: analyze raw data instead of experiment results
            return await self._run_analyst_initial()

        latest = self.state.versions[-1]
        notebook_path = self.output_dir / NOTEBOOK_FILENAME
        domain_knowledge = self.state.domain_knowledge

        # Find results file
        results_path = Path(latest.results_path) if latest.results_path else None
        if not results_path or not results_path.exists():
            print_step("  ANALYZE: skipped (no results file)")
            return None

        # Find plot PNGs in the version directory
        version_dir = Path(latest.script_path).parent
        plot_paths = sorted(version_dir.glob("*.png"))

        success_criteria = self.state.success_criteria or []

        cfg = self.model_config.resolve("analyst")

        print_step(f"  ANALYZE: analyzing {latest.version} ({len(plot_paths)} plots)")
        buffer: list[str] = []
        try:
            async def _analyst_coro(buf):
                return await run_analyst(
                    results_path=results_path,
                    plot_paths=plot_paths,
                    notebook_path=notebook_path,
                    domain_knowledge=domain_knowledge,
                    success_criteria=success_criteria,
                    model=cfg.model,
                    message_buffer=buf,
                )

            analysis = await self._with_summaries(_analyst_coro, "Analyst", buffer)
            n_criteria = len(analysis.get("criteria_results", []))
            logger.info(
                f"Analyst complete: {n_criteria} criteria evaluated, "
                f"data_summary={'yes' if analysis.get('data_summary') else 'no'}"
            )
            print_step(f"  ANALYZE: complete ({n_criteria} criteria evaluated)")
            return analysis
        except Exception as e:
            logger.exception(f"Analyst error: {e}")
            print_step(f"  ANALYZE: error - {e}")
            return None
        finally:
            self._persist_buffer("analyst", buffer)

    async def _run_scientist_plan(self, analysis: dict | None) -> dict[str, Any] | None:
        """Invoke the Scientist agent to formulate a plan."""
        from auto_scientist.agents.scientist import run_scientist

        version = f"v{self.state.iteration:02d}"
        notebook_path = self.output_dir / NOTEBOOK_FILENAME
        domain_knowledge = self.state.domain_knowledge

        cfg = self.model_config.resolve("scientist")

        print_step(f"  PLAN: scientist planning {version}")
        buffer: list[str] = []
        try:
            async def _scientist_coro(buf):
                return await run_scientist(
                    analysis=analysis or {},
                    notebook_path=notebook_path,
                    version=version,
                    domain_knowledge=domain_knowledge,
                    success_criteria=self.state.success_criteria,
                    model=cfg.model,
                    message_buffer=buf,
                )

            plan = await self._with_summaries(_scientist_coro, "Scientist", buffer)

            # Write the notebook entry from the plan
            if plan.get("notebook_entry"):
                append_entry(notebook_path, plan["notebook_entry"], version, "scientist")

            logger.info(
                f"Scientist plan: strategy={plan.get('strategy', '?')}, "
                f"changes={len(plan.get('changes', []))}, "
                f"hypothesis={plan.get('hypothesis', '?')[:100]}, "
                f"should_stop={plan.get('should_stop', False)}"
            )
            print_step(f"  PLAN: strategy={plan.get('strategy', '?')}, "
                       f"changes={len(plan.get('changes', []))}")
            return plan
        except Exception as e:
            logger.exception(f"Scientist plan error: {e}")
            print_step(f"  PLAN: error - {e}")
            return None
        finally:
            self._persist_buffer("scientist", buffer)

    async def _run_debate(self, plan: dict | None) -> list[dict[str, Any]] | None:
        """Send plan to critic model(s) for debate with the Scientist."""
        if not self.model_config.critics or plan is None:
            print_step("  DEBATE: skipped (no critics configured or no plan)")
            return None

        from auto_scientist.agents.critic import run_debate

        notebook_content = self._notebook_content()
        domain_knowledge = self.state.domain_knowledge
        scientist_cfg = self.model_config.resolve("scientist")

        from auto_scientist.console import make_stream_printer

        factory = make_stream_printer if self.stream else None

        n_critics = len(self.model_config.critics)
        print_step(f"  DEBATE: {n_critics} critic(s), {self.debate_rounds} round(s)")

        buffer: list[str] = []

        async def _debate_coro(buf):
            return await run_debate(
                critic_configs=self.model_config.critics,
                plan=plan,
                notebook_content=notebook_content,
                domain_knowledge=domain_knowledge,
                max_rounds=self.debate_rounds,
                scientist_config=scientist_cfg,
                on_token_factory=factory,
                message_buffer=buf,
            )

        try:
            critiques = await self._with_summaries(_debate_coro, "Debate", buffer)
            print_step(f"  DEBATE: received {len(critiques)} critique(s)")
            return critiques
        except Exception as e:
            logger.exception(f"Debate error: {e}")
            print_step(f"  DEBATE: error - {e}")
            return None
        finally:
            self._persist_buffer("debate", buffer)

    async def _run_scientist_revision(
        self,
        plan: dict | None,
        debate_result: list[dict[str, Any]] | None,
        analysis: dict | None,
    ) -> dict[str, Any] | None:
        """Scientist revises plan based on debate."""
        if plan is None or not debate_result:
            print_step("  REVISE: skipped (no plan or no debate)")
            return None

        from auto_scientist.agents.scientist import run_scientist_revision

        version = f"v{self.state.iteration:02d}"
        notebook_path = self.output_dir / NOTEBOOK_FILENAME
        domain_knowledge = self.state.domain_knowledge

        # Combine transcripts from all critics
        all_transcript: list[dict[str, str]] = []
        for entry in debate_result:
            all_transcript.append({"role": "critic", "content": f"[{entry['model']}]"})
            all_transcript.extend(entry.get("transcript", []))

        cfg = self.model_config.resolve("scientist")

        print_step("  REVISE: scientist revising plan after debate")
        buffer: list[str] = []
        try:
            async def _revision_coro(buf):
                return await run_scientist_revision(
                    original_plan=plan,
                    debate_transcript=all_transcript,
                    analysis=analysis or {},
                    notebook_path=notebook_path,
                    version=version,
                    domain_knowledge=domain_knowledge,
                    model=cfg.model,
                    message_buffer=buf,
                )

            revised = await self._with_summaries(
                _revision_coro, "Scientist Revision", buffer,
            )

            # Write revised notebook entry
            if revised.get("notebook_entry"):
                append_entry(notebook_path, revised["notebook_entry"], version, "revision")

            print_step(f"  REVISE: strategy={revised.get('strategy', '?')}")
            return revised
        except Exception as e:
            logger.exception(f"Scientist revision error: {e}")
            print_step(f"  REVISE: error - {e}, using original plan")
            return None
        finally:
            self._persist_buffer("scientist_revision", buffer)

    async def _run_coder(self, plan: dict | None) -> Path | None:
        """Invoke the Coder agent to implement the plan."""
        from auto_scientist.agents.coder import run_coder

        if plan is None:
            print_step("  IMPLEMENT: skipped (no plan)")
            return None

        version = f"v{self.state.iteration:02d}"
        domain_knowledge = self.state.domain_knowledge
        data_path = self.state.data_path or ""

        # On iteration 0 (no previous versions), use a nonexistent path
        if self.state.versions:
            latest = self.state.versions[-1]
            previous_script = Path(latest.script_path)
        else:
            previous_script = Path("nonexistent")

        run_timeout = self.config.run_timeout_minutes if self.config else 120
        run_cmd = self.config.run_command if self.config else "uv run {script_path}"

        cfg = self.model_config.resolve("coder")

        print_step(f"  IMPLEMENT: coder writing and running {version}")
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
                )

            new_script = await self._with_summaries(_coder_coro, "Coder", buffer)
            print_step(f"  IMPLEMENT: created {new_script}")
            return new_script
        except Exception as e:
            logger.exception(f"Coder error: {e}")
            print_step(f"  IMPLEMENT: error - {e}")
            self.state.record_failure()
            return None
        finally:
            self._persist_buffer("coder", buffer)

    def _apply_criteria_updates(self, plan: dict[str, Any]) -> None:
        """Check Scientist plan for top_level_criteria or criteria_revision, update state.

        TODO: criteria revision is purely voluntary (Scientist includes criteria_revision
        in plan output). In toy_function_022, a v02 regression was never addressed because
        the loop hit max iterations before the Scientist could evaluate it. Consider
        making revision more systematic, e.g. prompting the Scientist more strongly on
        score regressions or adding an automatic revision consideration step.
        """
        if plan.get("top_level_criteria"):
            criteria = [
                c for raw in plan["top_level_criteria"]
                if (c := self._parse_criterion(raw)) is not None
            ]
            self.state.success_criteria = criteria
            self.state.criteria_history.append(
                CriteriaRevision(
                    iteration=self.state.iteration,
                    action="defined",
                    changes="Initial criteria definition",
                    criteria_snapshot=criteria,
                )
            )

        elif plan.get("criteria_revision"):
            rev = plan["criteria_revision"]
            criteria = [
                c for raw in rev.get("revised_criteria", [])
                if (c := self._parse_criterion(raw)) is not None
            ]
            self.state.success_criteria = criteria
            self.state.criteria_history.append(
                CriteriaRevision(
                    iteration=self.state.iteration,
                    action="revised",
                    changes=rev.get("changes", ""),
                    criteria_snapshot=criteria,
                )
            )

    @staticmethod
    def _compute_score(
        criteria_results: list[dict[str, Any]],
        success_criteria: list[SuccessCriterion],
    ) -> int:
        """Compute score deterministically from criteria results.

        Formula: (passing_required / total_required) * 100, rounded to int.
        Optional criteria are tracked but don't affect the score.
        """
        required = [c for c in success_criteria if c.required]
        if not required:
            return 0

        # Build a lookup from criteria results by name
        result_by_name = {r["name"]: r["status"] for r in criteria_results}

        passing = sum(
            1 for c in required if result_by_name.get(c.name) == "pass"
        )
        return round((passing / len(required)) * 100)

    def _score_latest(self, analysis: dict[str, Any] | None) -> None:
        """Score the latest version using criteria_results and current success_criteria.

        Called after _apply_criteria_updates so that criteria defined in the
        same iteration are available for scoring.
        """
        if not self.state.versions:
            return
        latest = self.state.versions[-1]
        criteria_results = (analysis or {}).get("criteria_results", [])
        score = self._compute_score(
            criteria_results, self.state.success_criteria or [],
        )
        latest.score = score
        if score > self.state.best_score:
            self.state.best_score = score
            self.state.best_version = latest.version
        print_step(f"  SCORE: {score}", color=GREEN)

    async def _score_final_version(self) -> None:
        """Run the Analyst on the last version so it gets a score before report."""
        print_step("\nScoring final version before report...", color=BOLD)
        analysis = await self._run_analyst()
        if analysis:
            self._score_latest(analysis)

    @staticmethod
    def _parse_criterion(raw: dict[str, Any]) -> SuccessCriterion | None:
        """Parse a criterion dict from Scientist output into SuccessCriterion.

        Returns None if the criterion has no numeric target (unmeasurable).
        """
        target_min = None
        target_max = None
        condition = raw.get("condition", "")

        # Parse condition string: "> 0.95", "< 500", ">= 0.9", "<= 10"
        match = re.match(r"^\s*([<>]=?)\s*([\d.]+)\s*$", condition)
        if match:
            op, val = match.group(1), float(match.group(2))
            if op in (">", ">="):
                target_min = val
            elif op in ("<", "<="):
                target_max = val

        if target_min is None and target_max is None:
            logger.warning(
                f"Rejecting criterion '{raw.get('name', '?')}': "
                f"no numeric target (condition='{condition}')"
            )
            return None

        return SuccessCriterion(
            name=raw["name"],
            description=raw.get("description", ""),
            metric_key=raw.get("metric_key", ""),
            target_min=target_min,
            target_max=target_max,
        )

    _INFRA_FILES = {"run_result.json", "exitcode.txt", "stderr.txt"}

    def _read_run_result(self, version_dir: Path) -> RunResult:
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

        try:
            data = json.loads(run_result_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse run_result.json: {e}")
            return RunResult(
                success=False,
                stderr=f"Failed to parse run_result.json: {e}",
            )

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
            str(f) for f in version_dir.iterdir()
            if f.suffix in (".png", ".txt", ".csv", ".json")
            and f.name not in self._INFRA_FILES
        ]

        return RunResult(
            success=data.get("success", False),
            return_code=data.get("return_code", -1),
            timed_out=data.get("timed_out", False),
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
        )

    def _evaluate(self, run_result: RunResult | None, version_entry: VersionEntry) -> None:
        """Evaluate results and update the version entry."""
        if run_result is None:
            version_entry.status = "failed"
            self.state.record_failure()
            return

        if run_result.timed_out:
            version_entry.status = "failed"
            self.state.record_failure()
            return

        if not run_result.success:
            version_entry.status = "failed"
            self.state.record_failure()
            return

        # Success path
        version_entry.status = "completed"
        self.state.record_success()

        # Set results path if stdout was saved
        results_path = Path(version_entry.script_path).parent / "results.txt"
        if results_path.exists():
            version_entry.results_path = str(results_path)

    async def _run_report(self) -> None:
        """Phase 2: Generate final summary report."""
        from auto_scientist.agents.report import run_report

        notebook_path = self.output_dir / NOTEBOOK_FILENAME

        print_step("\nREPORT phase: generating final summary")
        buffer: list[str] = []
        cfg = self.model_config.resolve("report")

        try:
            async def _report_coro(buf):
                return await run_report(
                    state=self.state,
                    notebook_path=notebook_path,
                    output_dir=self.output_dir,
                    model=cfg.model,
                    message_buffer=buf,
                )

            report_path = await self._with_summaries(_report_coro, "Report", buffer)
            print_step(f"REPORT: written to {report_path}")
        except Exception as e:
            logger.exception(f"Report error: {e}")
            print_step(f"REPORT: error - {e}")
        finally:
            self._persist_buffer("report", buffer)
