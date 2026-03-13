"""Main orchestration loop and state machine."""

from pathlib import Path

from auto_scientist.config import DomainConfig
from auto_scientist.scheduler import wait_for_window
from auto_scientist.state import ExperimentState


class Orchestrator:
    """Drives the Discovery -> Iteration -> Report pipeline.

    State machine phases:
        DISCOVERY -> ANALYZE -> CRITIQUE -> IMPLEMENT -> VALIDATE -> RUN -> EVALUATE
                                                                              |
                                                                      ANALYZE (loop)
                                                                      or STOP
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

    async def run(self) -> None:
        """Execute the full orchestration loop."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        state_path = self.output_dir / "state.json"

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

    async def _run_discovery(self) -> None:
        """Phase 1: Explore data, research domain, design first model."""
        # TODO: Invoke Discovery agent
        print("DISCOVERY phase: not yet implemented")

    async def _run_iteration(self) -> None:
        """Single iteration: analyze -> critique -> implement -> validate -> run -> evaluate."""
        self.state.iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {self.state.iteration}")
        print(f"{'='*60}")

        # Step 1: Analyst
        analysis = await self._run_analyst()

        # Step 2: Check if Analyst recommends stopping
        if analysis and analysis.get("should_stop"):
            print(f"Analyst recommends stopping: {analysis.get('stop_reason', 'unknown')}")
            self.state.phase = "report"
            return

        # Step 3: Critic
        critique = await self._run_critic(analysis)

        # Step 4: Scientist
        new_script = await self._run_scientist(analysis, critique)

        # Step 5: Validate (syntax check)
        if new_script:
            valid = await self._validate_script(new_script)
            if not valid:
                self.state.record_failure()
                return

        # Step 6: Run
        run_result = await self._run_experiment(new_script)

        # Step 7: Evaluate
        await self._evaluate(run_result)

    async def _run_analyst(self) -> dict | None:
        """Invoke the Analyst agent on latest results + plots."""
        # TODO: Invoke Analyst agent
        print("  ANALYZE: not yet implemented")
        return None

    async def _run_critic(self, analysis: dict | None) -> str | None:
        """Send analysis to critic model(s) for debate."""
        if not self.critic_models or analysis is None:
            print("  CRITIQUE: skipped (no critics configured or no analysis)")
            return None

        from auto_scientist.agents.critic import run_debate
        from auto_scientist.history import build_compressed_history

        compressed_history = build_compressed_history(self.state)
        notebook_path = self.output_dir / "lab_notebook.md"
        notebook_content = notebook_path.read_text() if notebook_path.exists() else ""

        domain_knowledge = ""
        if self.config:
            domain_knowledge = self.config.domain_knowledge

        # Read the latest script content for the defender
        script_content = ""
        if self.state.versions:
            latest = self.state.versions[-1]
            script_path = Path(latest.script_path)
            if script_path.exists():
                script_content = script_path.read_text()

        print(f"  CRITIQUE: debating with {len(self.critic_models)} critic(s), {self.debate_rounds} round(s)")
        critiques = await run_debate(
            critic_specs=self.critic_models,
            analysis=analysis,
            compressed_history=compressed_history,
            notebook_content=notebook_content,
            domain_knowledge=domain_knowledge,
            script_content=script_content,
            max_rounds=self.debate_rounds,
        )

        # Combine all critiques into a single string for the Scientist
        parts = []
        for entry in critiques:
            parts.append(f"### Critique from {entry['model']}\n\n{entry['critique']}")
        combined = "\n\n---\n\n".join(parts)
        print(f"  CRITIQUE: received {len(critiques)} critique(s)")
        return combined

    async def _run_scientist(self, analysis: dict | None, critique: str | None) -> Path | None:
        """Invoke the Scientist agent to implement changes."""
        # TODO: Invoke Scientist agent
        print("  IMPLEMENT: not yet implemented")
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

    async def _run_experiment(self, script_path: Path | None) -> dict | None:
        """Execute the experiment script."""
        # TODO: Invoke Runner
        print("  RUN: not yet implemented")
        return None

    async def _evaluate(self, run_result: dict | None) -> None:
        """Evaluate results and update state."""
        # TODO: Parse results and update state
        print("  EVALUATE: not yet implemented")

    async def _run_report(self) -> None:
        """Phase 3: Generate final summary report."""
        # TODO: Invoke Report agent
        print("REPORT phase: not yet implemented")
