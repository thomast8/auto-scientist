"""Main orchestration loop and state machine."""

from pathlib import Path
from typing import Any

from auto_scientist.config import DomainConfig
from auto_scientist.runner import RunResult
from auto_scientist.scheduler import wait_for_window
from auto_scientist.state import ExperimentState, VersionEntry


class Orchestrator:
    """Drives the Ingestion -> Discovery -> Iteration -> Report pipeline.

    State machine phases:
        INGESTION -> DISCOVERY -> ANALYZE -> PLAN -> STOP_CHECK -> CRITIQUE ->
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
        config: DomainConfig | None = None,
        synthesis_interval: int = 0,
    ):
        self.state = state
        self.data_path = data_path
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.critic_models = critic_models or []
        self.interactive = interactive
        self.max_consecutive_failures = max_consecutive_failures
        self.debate_rounds = debate_rounds
        self.config = config
        self.synthesis_interval = synthesis_interval
        self._notebook_override: str | None = None  # set by synthesis

    async def run(self) -> None:
        """Execute the full orchestration loop."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.output_dir / "state.json"

        # Phase 0: Ingestion
        if self.state.phase == "ingestion":
            self.state.raw_data_path = self.state.data_path
            self.state.save(state_path)

            canonical_data_dir = await self._run_ingestion()

            self.state.data_path = str(canonical_data_dir)
            self.data_path = canonical_data_dir

            if self.config:
                data_dir_str = str(canonical_data_dir.resolve())
                if data_dir_str not in self.config.protected_paths:
                    self.config.protected_paths.append(data_dir_str)

            self.state.phase = "discovery"
            self.state.save(state_path)

        # Phase 1: Discovery
        if self.state.phase == "discovery":
            await self._run_discovery()
            self.state.phase = "iteration"
            self.state.save(state_path)

        # Phase 2: Iteration loop
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

            await self._run_iteration()
            self.state.save(state_path)

        # Phase 3: Report
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

        print("INGESTION phase: canonicalizing raw data")
        canonical_data_dir = await run_ingestor(
            raw_data_path=source_path,
            output_dir=self.output_dir,
            goal=self.state.goal,
            interactive=self.interactive,
        )
        return canonical_data_dir

    async def _run_discovery(self) -> None:
        """Phase 1: Explore data, research domain, design first model."""
        from auto_scientist.agents.coder import run_coder
        from auto_scientist.agents.discovery import run_discovery

        notebook_path = self.output_dir / "lab_notebook.md"

        if self.config is not None:
            # Pre-loaded domain config: use Coder to write the first script
            print("DISCOVERY phase: using pre-loaded domain config")
            version_dir = self.output_dir / "v00"
            version_dir.mkdir(parents=True, exist_ok=True)

            # Create initial lab notebook
            if not notebook_path.exists():
                notebook_path.write_text(
                    f"# Lab Notebook\n\n"
                    f"## Goal\n{self.state.goal}\n\n"
                    f"## Domain\n{self.config.name}: {self.config.description}\n\n"
                    f"---\n\n"
                )

            # Create a synthetic initial plan for the Coder
            initial_plan = {
                "hypothesis": "Baseline approach based on domain knowledge",
                "strategy": "exploratory",
                "changes": [
                    {
                        "what": "Create initial experiment from scratch",
                        "why": "First iteration, need a baseline to iterate from",
                        "how": (
                            "Design a simple baseline based on the domain knowledge"
                            " and goal. Start simple."
                        ),
                        "priority": 1,
                    }
                ],
                "expected_impact": "Establish a baseline for future iterations",
                "should_stop": False,
                "stop_reason": None,
                "notebook_entry": (
                    "## v00 - Initial Baseline\n\n"
                    "Baseline approach from domain config.\n"
                ),
            }

            # Write the notebook entry
            with notebook_path.open("a") as f:
                f.write(initial_plan["notebook_entry"] + "\n---\n\n")

            # Use Coder agent to write the first script
            script_path = await run_coder(
                plan=initial_plan,
                previous_script=Path("nonexistent"),  # No previous script
                output_dir=self.output_dir,
                version="v00",
                domain_knowledge=self.config.domain_knowledge,
                data_path=self.state.data_path or "",
                experiment_dependencies=self.config.experiment_dependencies,
            )
        else:
            # Auto-discovery mode
            print("DISCOVERY phase: exploring dataset and building first model")
            self.config, script_path = await run_discovery(
                state=self.state,
                data_path=self.data_path,
                output_dir=self.output_dir,
                interactive=self.interactive,
            )
            self.state.config_path = str(self.output_dir / "domain_config.json")

        # Run and evaluate the initial script
        print("DISCOVERY phase: running initial experiment (v00)")
        run_result = await self._run_experiment(script_path)
        version_entry = VersionEntry(
            version="v00",
            iteration=0,
            script_path=str(script_path),
            hypothesis="Initial model from discovery phase",
        )
        self._evaluate(run_result, version_entry)
        self.state.record_version(version_entry)

    def _notebook_content(self) -> str:
        """Return notebook content, using synthesis override if available."""
        if self._notebook_override:
            return self._notebook_override
        notebook_path = self.output_dir / "lab_notebook.md"
        return notebook_path.read_text() if notebook_path.exists() else ""

    async def _run_synthesis(self) -> None:
        """Condense the notebook if synthesis interval is reached."""
        if self.synthesis_interval <= 0:
            return
        if self.state.iteration % self.synthesis_interval != 0:
            return

        from auto_scientist.synthesis import run_synthesis

        notebook_path = self.output_dir / "lab_notebook.md"
        if not notebook_path.exists():
            return

        notebook_content = notebook_path.read_text()
        domain_knowledge = self.config.domain_knowledge if self.config else ""

        print("  SYNTHESIS: condensing notebook")
        try:
            self._notebook_override = await run_synthesis(
                notebook_content=notebook_content,
                domain_knowledge=domain_knowledge,
            )
            print("  SYNTHESIS: done")
        except Exception as e:
            print(f"  SYNTHESIS: error - {e}")
            self._notebook_override = None

    async def _run_iteration(self) -> None:
        """Run one iteration of the pipeline."""
        self.state.iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {self.state.iteration}")
        print(f"{'='*60}")

        # Step 0: Periodic synthesis (condense notebook every N iterations)
        await self._run_synthesis()

        # Step 1: Analyst observes latest results
        analysis = await self._run_analyst()

        # Step 2: Scientist plans next iteration
        plan = await self._run_scientist_plan(analysis)

        # Step 3: Check if Scientist recommends stopping
        if plan and plan.get("should_stop"):
            print(f"Scientist recommends stopping: {plan.get('stop_reason', 'unknown')}")
            self.state.phase = "report"
            return

        # Step 4: Critic debates the Scientist's plan
        debate_result = await self._run_debate(plan)

        # Step 5: Scientist revises plan based on debate
        revised_plan = await self._run_scientist_revision(plan, debate_result, analysis)

        # Step 6: Coder implements the revised plan (or original if no revision)
        final_plan = revised_plan or plan
        new_script = await self._run_coder(final_plan)

        # Step 7: Validate (syntax check)
        if new_script:
            valid = await self._validate_script(new_script)
            if not valid:
                self.state.record_failure()
                return

        # Step 8: Run
        run_result = await self._run_experiment(new_script)

        # Step 9: Evaluate
        version = f"v{self.state.iteration:02d}"
        version_entry = VersionEntry(
            version=version,
            iteration=self.state.iteration,
            script_path=str(new_script),
            hypothesis=final_plan.get("hypothesis", "") if final_plan else "",
        )
        self._evaluate(run_result, version_entry)
        self.state.record_version(version_entry)

        # Reset synthesis override for next iteration
        self._notebook_override = None

    async def _run_analyst(self) -> dict[str, Any] | None:
        """Invoke the Analyst agent on latest results + plots."""
        from auto_scientist.agents.analyst import run_analyst

        if not self.state.versions:
            print("  ANALYZE: skipped (no previous versions)")
            return None

        latest = self.state.versions[-1]
        notebook_path = self.output_dir / "lab_notebook.md"
        domain_knowledge = self.config.domain_knowledge if self.config else ""

        # Find results file
        results_path = Path(latest.results_path) if latest.results_path else None
        if not results_path or not results_path.exists():
            print("  ANALYZE: skipped (no results file)")
            return None

        # Find plot PNGs in the version directory
        version_dir = Path(latest.script_path).parent
        plot_paths = sorted(version_dir.glob("*.png"))

        success_criteria = self.config.success_criteria if self.config else []

        print(f"  ANALYZE: analyzing {latest.version} ({len(plot_paths)} plots)")
        try:
            analysis = await run_analyst(
                results_path=results_path,
                plot_paths=plot_paths,
                notebook_path=notebook_path,
                domain_knowledge=domain_knowledge,
                success_criteria=success_criteria,
            )
            # Update latest version score
            if "success_score" in analysis:
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

        if not self.state.versions:
            print("  PLAN: skipped (no previous versions)")
            return None

        version = f"v{self.state.iteration:02d}"
        notebook_path = self.output_dir / "lab_notebook.md"
        domain_knowledge = self.config.domain_knowledge if self.config else ""

        print(f"  PLAN: scientist planning {version}")
        try:
            plan = await run_scientist(
                analysis=analysis or {},
                notebook_path=notebook_path,
                version=version,
                domain_knowledge=domain_knowledge,
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
        domain_knowledge = self.config.domain_knowledge if self.config else ""

        n_critics = len(self.critic_models)
        print(f"  DEBATE: {n_critics} critic(s), {self.debate_rounds} round(s)")
        critiques = await run_debate(
            critic_specs=self.critic_models,
            plan=plan,
            notebook_content=notebook_content,
            domain_knowledge=domain_knowledge,
            max_rounds=self.debate_rounds,
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
        domain_knowledge = self.config.domain_knowledge if self.config else ""

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

        if not self.state.versions:
            print("  IMPLEMENT: skipped (no previous versions)")
            return None

        version = f"v{self.state.iteration:02d}"
        latest = self.state.versions[-1]
        previous_script = Path(latest.script_path)
        domain_knowledge = self.config.domain_knowledge if self.config else ""

        data_path = self.state.data_path or ""
        deps = self.config.experiment_dependencies if self.config else []

        print(f"  IMPLEMENT: coder writing {version}")
        try:
            new_script = await run_coder(
                plan=plan,
                previous_script=previous_script,
                output_dir=self.output_dir,
                version=version,
                domain_knowledge=domain_knowledge,
                data_path=data_path,
                experiment_dependencies=deps,
            )
            print(f"  IMPLEMENT: created {new_script}")
            return new_script
        except Exception as e:
            print(f"  IMPLEMENT: error - {e}")
            self.state.record_failure()
            return None

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

        command = "uv run python -u {script_path}"
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
        """Phase 3: Generate final summary report."""
        from auto_scientist.agents.report import run_report

        notebook_path = self.output_dir / "lab_notebook.md"

        print("\nREPORT phase: generating final summary")
        try:
            report_path = await run_report(
                state=self.state,
                notebook_path=notebook_path,
                output_dir=self.output_dir,
            )
            print(f"REPORT: written to {report_path}")
        except Exception as e:
            print(f"REPORT: error - {e}")
