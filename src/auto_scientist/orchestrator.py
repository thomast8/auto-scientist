"""Main orchestration loop and state machine."""

import json
import re
from pathlib import Path
from typing import Any

from auto_scientist.config import DomainConfig, SuccessCriterion
from auto_scientist.runner import RunResult
from auto_scientist.scheduler import wait_for_window
from auto_scientist.state import CriteriaRevision, ExperimentState, VersionEntry


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
        critic_models: list[str] | None = None,
        interactive: bool = False,
        max_consecutive_failures: int = 5,
        debate_rounds: int = 2,
        model: str | None = None,
        stream: bool = True,
    ):
        self.state = state
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.critic_models = critic_models or []
        self.interactive = interactive
        self.max_consecutive_failures = max_consecutive_failures
        self.debate_rounds = debate_rounds
        self.config: DomainConfig | None = None
        self.model = model
        self.stream = stream

    async def run(self) -> None:
        """Execute the full orchestration loop."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.output_dir / "state.json"

        model_label = self.model or "(SDK default)"
        print("Auto-Scientist starting")
        print(f"  Model:      {model_label}")
        print(f"  Output:     {self.output_dir}")
        print(f"  Goal:       {self.state.goal[:80]}{'...' if len(self.state.goal) > 80 else ''}")
        if self.critic_models:
            print(f"  Critics:    {', '.join(self.critic_models)}")
        print()

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

        # Phase 1: Unified iteration loop
        while self.state.phase == "iteration":
            if self.state.iteration >= self.max_iterations:
                print(f"Reached max iterations ({self.max_iterations}). Stopping.")
                self.state.phase = "report"
                break

            if self.state.should_stop_on_failures(self.max_consecutive_failures):
                print(
                    f"Hit {self.max_consecutive_failures} consecutive failures. Stopping."
                )
                self.state.phase = "report"
                break

            # Schedule check
            await wait_for_window(self.state.schedule)

            await self._run_iteration_body()
            self.state.save(state_path)

        # Phase 2: Report
        if self.state.phase == "report":
            await self._run_report()
            self.state.phase = "stopped"
            self.state.save(state_path)

        print(f"Experiment completed. Final state saved to {state_path}")

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

        print("INGESTION phase: canonicalizing raw data")
        canonical_data_dir = await run_ingestor(
            raw_data_path=source_path,
            output_dir=self.output_dir,
            goal=self.state.goal,
            interactive=self.interactive,
            config_path=config_path,
            model=self.model,
        )

        # Summary
        data_files = sorted(canonical_data_dir.iterdir())
        print("\nINGESTION complete:")
        for f in data_files:
            print(f"  {f}")
        print()

        return canonical_data_dir

    async def _run_iteration_body(self) -> None:
        """Run one iteration of the pipeline (inlined, not _run_iteration)."""
        print(f"\n{'='*60}")
        print(f"ITERATION {self.state.iteration}")
        print(f"{'='*60}")

        # Step 1: Analyst observes latest results (or raw data on iteration 0)
        analysis = await self._run_analyst()

        # Apply domain_knowledge from Analyst if present
        if analysis and analysis.get("domain_knowledge"):
            self.state.domain_knowledge = analysis["domain_knowledge"]

        # Step 2: Scientist plans next iteration
        plan = await self._run_scientist_plan(analysis)

        # Apply criteria updates from Scientist if present
        if plan:
            self._apply_criteria_updates(plan)

        # Step 3: Check if Scientist recommends stopping
        if plan and plan.get("should_stop"):
            print(f"Scientist recommends stopping: {plan.get('stop_reason', 'unknown')}")
            self.state.phase = "report"
            return

        # Step 4: Debate (skip on iteration 0, nothing to challenge)
        final_plan = plan
        if self.state.iteration > 0:
            debate_result = await self._run_debate(plan)
            revised_plan = await self._run_scientist_revision(plan, debate_result, analysis)
            final_plan = revised_plan or plan

        # Step 5: Coder implements the plan
        new_script = await self._run_coder(final_plan)

        # Step 6: Validate (syntax check)
        if new_script:
            valid = await self._validate_script(new_script)
            if not valid:
                self.state.record_failure()
                self.state.iteration += 1
                return

        # Step 7: Run
        run_result = await self._run_experiment(new_script)

        # Step 8: Evaluate
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
        print(f"\nITERATION {self.state.iteration} complete:")
        if new_script:
            print(f"  Script:     {new_script}")
        print(f"  Status:     {version_entry.status}")
        if version_entry.score is not None:
            print(f"  Score:      {version_entry.score}")
        print(f"  Best:       {self.state.best_version} (score {self.state.best_score})")
        if version_dir and version_dir.exists():
            suffixes = (".png", ".txt", ".json")
            artifacts = sorted(
                f for f in version_dir.iterdir() if f.suffix in suffixes
            )
            if artifacts:
                print("  Outputs:")
                for f in artifacts:
                    print(f"    {f}")

        # Increment at end of loop body
        self.state.iteration += 1

    def _notebook_content(self) -> str:
        """Return notebook content."""
        notebook_path = self.output_dir / "lab_notebook.md"
        return notebook_path.read_text() if notebook_path.exists() else ""

    async def _run_analyst_initial(self) -> dict[str, Any] | None:
        """Iteration 0: analyze raw canonical data instead of experiment results."""
        from auto_scientist.agents.analyst import run_analyst

        notebook_path = self.output_dir / "lab_notebook.md"
        domain_knowledge = self.state.domain_knowledge

        print("  ANALYZE: initial data characterization")
        try:
            analysis = await run_analyst(
                results_path=None,
                plot_paths=[],
                notebook_path=notebook_path,
                domain_knowledge=domain_knowledge,
                success_criteria=self.state.success_criteria,
                data_dir=self.data_path,
                model=self.model,
            )
            print("  ANALYZE: data characterization complete")
            return analysis
        except Exception as e:
            print(f"  ANALYZE: error - {e}")
            return None

    async def _run_analyst(self) -> dict[str, Any] | None:
        """Invoke the Analyst agent on latest results + plots."""
        from auto_scientist.agents.analyst import run_analyst

        if not self.state.versions:
            # Iteration 0: analyze raw data instead of experiment results
            return await self._run_analyst_initial()

        latest = self.state.versions[-1]
        notebook_path = self.output_dir / "lab_notebook.md"
        domain_knowledge = self.state.domain_knowledge

        # Find results file
        results_path = Path(latest.results_path) if latest.results_path else None
        if not results_path or not results_path.exists():
            print("  ANALYZE: skipped (no results file)")
            return None

        # Find plot PNGs in the version directory
        version_dir = Path(latest.script_path).parent
        plot_paths = sorted(version_dir.glob("*.png"))

        success_criteria = self.state.success_criteria or []

        print(f"  ANALYZE: analyzing {latest.version} ({len(plot_paths)} plots)")
        try:
            analysis = await run_analyst(
                results_path=results_path,
                plot_paths=plot_paths,
                notebook_path=notebook_path,
                domain_knowledge=domain_knowledge,
                success_criteria=success_criteria,
                model=self.model,
            )
            # Update latest version score
            if "success_score" in analysis and analysis["success_score"] is not None:
                latest.score = analysis["success_score"]
                # Update best tracking
                if latest.score > self.state.best_score:
                    self.state.best_score = latest.score
                    self.state.best_version = latest.version
            print(f"  ANALYZE: score={analysis.get('success_score', '?')}")
            return analysis
        except Exception as e:
            print(f"  ANALYZE: error - {e}")
            return None

    async def _run_scientist_plan(self, analysis: dict | None) -> dict[str, Any] | None:
        """Invoke the Scientist agent to formulate a plan."""
        from auto_scientist.agents.scientist import run_scientist

        version = f"v{self.state.iteration:02d}"
        notebook_path = self.output_dir / "lab_notebook.md"
        domain_knowledge = self.state.domain_knowledge

        print(f"  PLAN: scientist planning {version}")
        try:
            plan = await run_scientist(
                analysis=analysis or {},
                notebook_path=notebook_path,
                version=version,
                domain_knowledge=domain_knowledge,
                success_criteria=self.state.success_criteria,
                model=self.model,
            )

            # Write the notebook entry from the plan
            if plan.get("notebook_entry"):
                with notebook_path.open("a") as f:
                    f.write(plan["notebook_entry"] + "\n\n---\n\n")

            print(f"  PLAN: strategy={plan.get('strategy', '?')}, "
                  f"changes={len(plan.get('changes', []))}")
            return plan
        except Exception as e:
            print(f"  PLAN: error - {e}")
            return None

    async def _run_debate(self, plan: dict | None) -> list[dict[str, Any]] | None:
        """Send plan to critic model(s) for debate with the Scientist."""
        if not self.critic_models or plan is None:
            print("  DEBATE: skipped (no critics configured or no plan)")
            return None

        from auto_scientist.agents.critic import run_debate

        notebook_content = self._notebook_content()
        domain_knowledge = self.state.domain_knowledge

        from auto_scientist.console import make_stream_printer

        factory = make_stream_printer if self.stream else None

        n_critics = len(self.critic_models)
        print(f"  DEBATE: {n_critics} critic(s), {self.debate_rounds} round(s)")
        critiques = await run_debate(
            critic_specs=self.critic_models,
            plan=plan,
            notebook_content=notebook_content,
            domain_knowledge=domain_knowledge,
            max_rounds=self.debate_rounds,
            on_token_factory=factory,
        )

        print(f"  DEBATE: received {len(critiques)} critique(s)")
        return critiques

    async def _run_scientist_revision(
        self,
        plan: dict | None,
        debate_result: list[dict[str, Any]] | None,
        analysis: dict | None,
    ) -> dict[str, Any] | None:
        """Scientist revises plan based on debate."""
        if plan is None or not debate_result:
            print("  REVISE: skipped (no plan or no debate)")
            return None

        from auto_scientist.agents.scientist import run_scientist_revision

        version = f"v{self.state.iteration:02d}"
        notebook_path = self.output_dir / "lab_notebook.md"
        domain_knowledge = self.state.domain_knowledge

        # Combine transcripts from all critics
        all_transcript: list[dict[str, str]] = []
        for entry in debate_result:
            all_transcript.append({"role": "critic", "content": f"[{entry['model']}]"})
            all_transcript.extend(entry.get("transcript", []))

        print("  REVISE: scientist revising plan after debate")
        try:
            revised = await run_scientist_revision(
                original_plan=plan,
                debate_transcript=all_transcript,
                analysis=analysis or {},
                notebook_path=notebook_path,
                version=version,
                domain_knowledge=domain_knowledge,
                model=self.model,
            )

            # Write revised notebook entry
            if revised.get("notebook_entry"):
                with notebook_path.open("a") as f:
                    f.write(revised["notebook_entry"] + "\n\n---\n\n")

            print(f"  REVISE: strategy={revised.get('strategy', '?')}")
            return revised
        except Exception as e:
            print(f"  REVISE: error - {e}, using original plan")
            return None

    async def _run_coder(self, plan: dict | None) -> Path | None:
        """Invoke the Coder agent to implement the plan."""
        from auto_scientist.agents.coder import run_coder

        if plan is None:
            print("  IMPLEMENT: skipped (no plan)")
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

        print(f"  IMPLEMENT: coder writing {version}")
        try:
            new_script = await run_coder(
                plan=plan,
                previous_script=previous_script,
                output_dir=self.output_dir,
                version=version,
                domain_knowledge=domain_knowledge,
                data_path=data_path,
                model=self.model,
            )
            print(f"  IMPLEMENT: created {new_script}")
            return new_script
        except Exception as e:
            print(f"  IMPLEMENT: error - {e}")
            self.state.record_failure()
            return None

    def _apply_criteria_updates(self, plan: dict[str, Any]) -> None:
        """Check Scientist plan for top_level_criteria or criteria_revision, update state."""
        if plan.get("top_level_criteria"):
            criteria = [
                self._parse_criterion(c) for c in plan["top_level_criteria"]
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
                self._parse_criterion(c) for c in rev.get("revised_criteria", [])
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
    def _parse_criterion(raw: dict[str, Any]) -> SuccessCriterion:
        """Parse a criterion dict from Scientist output into SuccessCriterion."""
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

        return SuccessCriterion(
            name=raw["name"],
            description=raw.get("description", ""),
            metric_key=raw.get("metric_key", ""),
            target_min=target_min,
            target_max=target_max,
        )

    async def _validate_script(self, script_path: Path) -> bool:
        """Syntax-check the generated script."""
        from auto_scientist.runner import validate_syntax

        valid, error = validate_syntax(script_path)
        if not valid:
            print(f"  VALIDATE: syntax error - {error}")
        else:
            print("  VALIDATE: syntax OK")
        return valid

    async def _run_experiment(self, script_path: Path | None) -> RunResult | None:
        """Execute the experiment script."""
        from auto_scientist.runner import run_experiment

        if script_path is None:
            print("  RUN: skipped (no script)")
            return None

        command = "uv run {script_path}"
        cwd = "."
        timeout = 120

        if self.config:
            command = self.config.run_command
            cwd = self.config.run_cwd
            timeout = self.config.run_timeout_minutes

        print(f"  RUN: executing {script_path.name} (timeout {timeout}m)")
        result = await run_experiment(
            script_path=script_path,
            command_template=command,
            cwd=cwd,
            timeout_minutes=timeout,
        )

        if result.timed_out:
            print(f"  RUN: timed out after {timeout} minutes")
        elif not result.success:
            print(f"  RUN: failed (rc={result.return_code})")
            if result.stderr:
                # Print last few lines of stderr
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines[-5:]:
                    print(f"  RUN: {line}")
        else:
            print(f"  RUN: success ({len(result.output_files)} output files)")

        # Save stdout to results file
        if result.stdout:
            results_path = script_path.parent / "results.txt"
            results_path.write_text(result.stdout)

        return result

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

        notebook_path = self.output_dir / "lab_notebook.md"

        print("\nREPORT phase: generating final summary")
        try:
            report_path = await run_report(
                state=self.state,
                notebook_path=notebook_path,
                output_dir=self.output_dir,
                model=self.model,
            )
            print(f"REPORT: written to {report_path}")
        except Exception as e:
            print(f"REPORT: error - {e}")
