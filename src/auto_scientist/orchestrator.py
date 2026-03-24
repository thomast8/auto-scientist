"""Main orchestration loop and state machine."""

import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from auto_scientist.config import DomainConfig, SuccessCriterion
from auto_scientist.console import (
    AGENT_STYLES,
    AgentPanel,
    PipelineLive,
    console,
)
from auto_scientist.log_setup import setup_file_logging
from auto_scientist.model_config import AgentModelConfig, ModelConfig
from auto_scientist.notebook import NOTEBOOK_FILENAME, append_entry, read_notebook
from auto_scientist.runner import RunResult
from auto_scientist.scheduler import wait_for_window
from auto_scientist.state import CriteriaRevision, ExperimentState, VersionEntry
from auto_scientist.summarizer import run_with_summaries, summarize_results

logger = logging.getLogger(__name__)


def _check_provider_auth(provider: str) -> str | None:
    """Try to instantiate the SDK client for a provider. Returns error message or None."""
    if provider == "anthropic":
        try:
            from anthropic import Anthropic

            Anthropic()
            return None
        except Exception as e:
            return f"Anthropic SDK authentication failed: {e}"
    elif provider == "openai":
        try:
            from openai import OpenAI

            OpenAI()
            return None
        except Exception as e:
            return f"OpenAI SDK authentication failed: {e}"
    elif provider == "google":
        if not os.environ.get("GOOGLE_API_KEY"):
            return "GOOGLE_API_KEY is not set (required for google provider)"
        return None
    else:
        return f"Unknown provider: {provider}"


def _validate_reasoning_configs(mc: ModelConfig) -> list[str]:
    """Validate reasoning configs are compatible with their provider and model.

    Returns a list of error messages for invalid configurations.
    """
    from auto_scientist.models.anthropic_client import ANTHROPIC_BUDGET_DEFAULTS
    from auto_scientist.models.google_client import GOOGLE_LEVEL_MAP
    from auto_scientist.models.openai_client import OPENAI_EFFORT_MAP

    errors: list[str] = []

    # Collect (agent_name, AgentModelConfig) pairs
    entries: list[tuple[str, AgentModelConfig]] = []
    for agent_name in ["analyst", "scientist", "coder", "ingestor", "report", "summarizer"]:
        entries.append((agent_name, mc.resolve(agent_name)))
    for i, critic in enumerate(mc.critics):
        entries.append((f"critic[{i}]", critic))

    for agent_name, cfg in entries:
        r = cfg.reasoning
        if r.level in ("default", "off"):
            continue

        label = f"{agent_name} ({cfg.provider}/{cfg.model}, reasoning={r.level})"

        if cfg.provider == "anthropic":
            budget = r.budget or ANTHROPIC_BUDGET_DEFAULTS.get(r.level)
            if budget is None:
                errors.append(
                    f"{label}: no budget_tokens mapping for level '{r.level}'"
                )
            elif budget < 1024:
                errors.append(
                    f"{label}: budget_tokens={budget} is below Anthropic minimum (1024)"
                )
            elif budget > 128_000:
                errors.append(
                    f"{label}: budget_tokens={budget} exceeds Anthropic maximum (128000)"
                )

        elif cfg.provider == "openai":
            effort = OPENAI_EFFORT_MAP.get(r.level)
            if effort is None:
                errors.append(
                    f"{label}: no reasoning effort mapping for level '{r.level}'"
                )

        elif cfg.provider == "google":
            model = cfg.model
            is_3x = "3" in model and "2.5" not in model and "2.0" not in model
            is_25 = "2.5" in model

            if is_3x:
                mapped = GOOGLE_LEVEL_MAP.get(r.level)
                if mapped is None:
                    errors.append(
                        f"{label}: no thinkingLevel mapping for level '{r.level}'"
                    )
                elif mapped in ("MINIMAL", "MEDIUM") and "flash" not in model.lower():
                    errors.append(
                        f"{label}: thinkingLevel={mapped} is only valid for Gemini 3 Flash models"
                    )
            elif is_25:
                # Budget-based; just verify we have a default
                if r.budget is None:
                    from auto_scientist.models.google_client import GOOGLE_BUDGET_DEFAULTS
                    if r.level not in GOOGLE_BUDGET_DEFAULTS:
                        errors.append(
                            f"{label}: no thinkingBudget mapping for level '{r.level}'"
                        )

    return errors


def _validate_model_names(mc: ModelConfig) -> list[str]:
    """Validate all configured model names against their provider APIs.

    Returns a list of error messages for invalid models.
    """
    errors: list[str] = []

    # Collect unique (provider, model) pairs to avoid duplicate checks
    pairs: dict[tuple[str, str], list[str]] = {}
    for agent_name in ["analyst", "scientist", "coder", "ingestor", "report"]:
        cfg = mc.resolve(agent_name)
        key = (cfg.provider, cfg.model)
        pairs.setdefault(key, []).append(agent_name)

    for critic in mc.critics:
        key = (critic.provider, critic.model)
        pairs.setdefault(key, []).append("critic")

    if mc.summarizer:
        key = (mc.summarizer.provider, mc.summarizer.model)
        pairs.setdefault(key, []).append("summarizer")

    # Only validate models for providers that authenticate successfully
    # (auth failures are reported separately by _check_provider_auth)
    authenticated_providers: set[str] = set()
    for provider in {p for p, _ in pairs}:
        if _check_provider_auth(provider) is None:
            authenticated_providers.add(provider)

    for (provider, model), agents in pairs.items():
        if provider not in authenticated_providers:
            continue
        err = _check_model_exists(provider, model)
        if err:
            agent_list = ", ".join(sorted(set(agents)))
            errors.append(f"Model '{model}' ({provider}) not found (used by: {agent_list}): {err}")

    return errors


def _check_model_exists(provider: str, model: str) -> str | None:
    """Check if a model exists by querying the provider API.

    Returns an error message if the model doesn't exist, None if it does.
    Ignores auth errors (handled separately by _check_provider_auth).
    """
    if provider == "anthropic":
        # Anthropic SDK agents run through the Claude Code CLI which handles
        # its own auth (OAuth/session). The Anthropic SDK's models.retrieve()
        # requires an API key which may not be set. Skip validation here;
        # the CLI will catch invalid model names at runtime.
        return None
    elif provider == "openai":
        try:
            from openai import AuthenticationError, OpenAI

            client = OpenAI()
            client.models.retrieve(model)
            return None
        except AuthenticationError:
            return None  # auth handled separately
        except Exception as e:
            return str(e)
    elif provider == "google":
        try:
            from google import genai
            from google.genai import errors as genai_errors

            client = genai.Client()
            client.models.get(model=model)
            return None
        except genai_errors.APIError as e:
            if e.code == 401 or e.code == 403:
                return None  # auth handled separately
            return str(e)
        except Exception as e:
            return str(e)
    return None  # unknown provider, skip validation


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
        self._live: PipelineLive = PipelineLive()

    def _validate_prerequisites(self) -> None:
        """Validate directories, API keys, and config before starting.

        Raises RuntimeError with all problems at once so the user can fix
        everything in a single pass.
        """
        errors: list[str] = []

        # Data path must exist when starting from ingestion
        if self.state.phase == "ingestion":
            if self.data_path is None:
                errors.append("--data is required for a new run")
            elif not self.data_path.exists():
                errors.append(f"Data path does not exist: {self.data_path}")

        # Output dir parent must be writable
        parent = self.output_dir.parent
        if parent.exists() and not os.access(parent, os.W_OK):
            errors.append(f"Output directory parent is not writable: {parent}")

        # Claude Code CLI must be installed (powers all main agents)
        if not shutil.which("claude"):
            errors.append(
                "Claude Code CLI not found on PATH. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )

        # uv must be installed (runs experiment scripts)
        run_cmd = self.config.run_command if self.config else "uv run {script_path}"
        exe = run_cmd.split()[0] if run_cmd.strip() else ""
        if exe and not shutil.which(exe):
            errors.append(
                f"'{exe}' not found on PATH (needed for run_command: {run_cmd}). "
                f"Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
            )

        # SDK agents must use Anthropic models (they run through claude CLI)
        mc = self.model_config
        sdk_agents = ["analyst", "scientist", "coder", "ingestor", "report"]
        for agent_name in sdk_agents:
            cfg = mc.resolve(agent_name)
            if cfg.provider != "anthropic":
                errors.append(
                    f"{agent_name} uses provider '{cfg.provider}' (model: {cfg.model}), "
                    f"but SDK agents require provider 'anthropic'. "
                    f"Non-Anthropic models can only be used for critics and summarizer."
                )

        # Validate model names against provider APIs
        model_errors = _validate_model_names(mc)
        errors.extend(model_errors)

        # Validate reasoning configs against provider constraints
        reasoning_errors = _validate_reasoning_configs(mc)
        errors.extend(reasoning_errors)

        # Collect required providers from model config
        required_providers: set[str] = set()

        # Claude Code SDK powers all main agents, always needs Anthropic
        required_providers.add("anthropic")

        # Summarizer always uses OpenAI directly
        if mc.summarizer is not None:
            required_providers.add("openai")

        # Critics use their configured provider
        for critic in mc.critics:
            required_providers.add(critic.provider)

        # Validate each provider by trying to instantiate its SDK client
        for provider in sorted(required_providers):
            err = _check_provider_auth(provider)
            if err:
                errors.append(err)

        if errors:
            raise RuntimeError(
                "Pre-flight check failed:\n  - " + "\n  - ".join(errors)
            )

    async def run(self) -> None:
        """Execute the full orchestration loop."""
        self._validate_prerequisites()
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

        # Startup banner (printed before Live starts)
        console.print(self._build_startup_banner())

        self._live = PipelineLive()
        self._live.start(log_path=self.output_dir / "console.log")

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
                    self._live.add_rule(
                        Rule(f"Reached max iterations ({self.max_iterations}). Stopping.", style="yellow")
                    )
                    self._live.flush_completed()
                    self.state.phase = "report"
                    break

                if self.state.should_stop_on_failures(self.max_consecutive_failures):
                    self._live.add_rule(
                        Rule(f"Hit {self.max_consecutive_failures} consecutive failures. Stopping.", style="red")
                    )
                    self._live.flush_completed()
                    self.state.phase = "report"
                    break

                # Schedule check
                await wait_for_window(self.state.schedule)

                await self._run_iteration_body()
                self.state.save(state_path)

            # Phase 2: Report (with its own border)
            if self.state.phase == "report":
                self._live.start_iteration("Report")

                # Score the final version if it was never evaluated
                if self.state.versions and self.state.versions[-1].score is None:
                    await self._score_final_version()
                    self.state.save(state_path)

                await self._run_report()
                self.state.phase = "stopped"
                self.state.save(state_path)

                self._live.end_iteration("done", "blue")
                self._live.flush_completed()

            logger.info("Run finished successfully")
        finally:
            self._live.wait_for_dismiss()
            self._live.stop()

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

        cfg = self.model_config.resolve("ingestor")
        panel = AgentPanel(name="Ingestor", model=cfg.model, style="red")
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
            )

        buffer: list[str] = []
        try:
            canonical_data_dir = await self._with_summaries(
                _ingestor_coro, "Ingestor", buffer, panel=panel,
            )
            data_files = sorted(canonical_data_dir.iterdir())
            file_list = ", ".join(f.name for f in data_files)
            self._collapse(panel, f"Canonicalized {len(data_files)} files: {file_list}")
        finally:
            self._persist_buffer("ingestor", buffer)

        return canonical_data_dir

    async def _run_iteration_body(self) -> None:
        """Run one iteration of the pipeline (inlined, not _run_iteration)."""
        logger.info(f"=== Iteration {self.state.iteration} start ===")
        self._live.start_iteration(self.state.iteration)
        self._live.update_status(iteration=self.state.iteration)
        prev_best = self.state.best_score

        version = f"v{self.state.iteration:02d}"
        version_dir = self.output_dir / version

        # Step 1: Analyst observes latest results (or raw data on iteration 0)
        analysis = await self._run_analyst()

        # Persist analysis for audit trail
        if analysis:
            self._persist_artifact(version_dir, "analysis.json", analysis)

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
            self._persist_artifact(version_dir, "plan.json", plan)
            logger.info(f"Scientist stop: {plan.get('stop_reason', 'unknown')}")
            self.state.phase = "report"
            self._live.end_iteration(
                f"stopped: {plan.get('stop_reason', 'unknown')}", "yellow",
            )
            self._live.flush_completed()
            return

        # Step 4: Debate (skip on iteration 0, nothing to challenge)
        final_plan = plan
        if self.state.iteration > 0:
            debate_result = await self._run_debate(plan)
            revised_plan = await self._run_scientist_revision(plan, debate_result, analysis)
            final_plan = revised_plan or plan

            # Persist debate transcript with original plan for context
            if debate_result:
                self._persist_artifact(version_dir, "debate.json", {
                    "original_plan": plan,
                    "critiques": debate_result,
                })

        # Persist the final plan (post-debate revision if applicable)
        if final_plan:
            self._persist_artifact(version_dir, "plan.json", final_plan)

        # Step 5: Coder implements and runs the plan
        new_script = await self._run_coder(final_plan)

        if new_script is None:
            # Coder failed to produce a script; record failure and move on
            version_entry = VersionEntry(
                version=version,
                iteration=self.state.iteration,
                script_path="",
                hypothesis=final_plan.get("hypothesis", "") if final_plan else "",
                status="failed",
            )
            self.state.record_failure()
            self.state.record_version(version_entry)
            logger.info(
                f"Iteration {self.state.iteration} complete: "
                f"status=failed (coder produced no script)"
            )
            self._live.end_iteration("failed (no script)", "red")
            self._live.flush_completed()
            self.state.iteration += 1
            return

        # Step 6: Read run result from Coder's output files
        run_result = self._read_run_result(new_script.parent)

        # Log run result
        if run_result.timed_out:
            self._live.log("RUN RESULT: timed out")
        elif not run_result.success:
            self._live.log(f"RUN RESULT: failed (rc={run_result.return_code})")
        else:
            self._live.log(f"RUN RESULT: success ({len(run_result.output_files)} output files)")

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
        self._evaluate(run_result, version_entry)
        self.state.record_version(version_entry)

        # Iteration border color: green=improved, yellow=no gain, red=failed
        best_now = self.state.best_score
        if version_entry.status != "completed":
            status_style = "red"
        elif best_now > prev_best:
            status_style = "green"
        elif best_now > 0:
            status_style = "yellow"
        else:
            status_style = "bold"
        score_str = f" (best: {best_now})" if best_now > 0 else ""
        # Update status bar with best score
        self._live.update_status(
            best_version=self.state.best_version,
            best_score=self.state.best_score,
        )
        self._live.end_iteration(f"{version_entry.status}{score_str}", status_style)
        self._live.flush_completed()

        # Increment at end of loop body
        logger.info(
            f"Iteration {self.state.iteration} complete: "
            f"status={version_entry.status}, score={version_entry.score}, "
            f"best={self.state.best_version} ({self.state.best_score})"
        )
        self.state.iteration += 1

    def _build_startup_banner(self) -> Panel:
        """Build the Rich startup banner with run config and model info."""
        mc = self.model_config
        goal_preview = self.state.goal[:80] + ("..." if len(self.state.goal) > 80 else "")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("key", style="bold", no_wrap=True)
        table.add_column("value")

        table.add_row("Input", str(self.data_path) if self.data_path else "(none)")
        table.add_row("Output", str(self.output_dir))
        table.add_row("Goal", goal_preview)
        table.add_row("", "")  # spacer

        # Agent models
        agent_map = [
            ("Ingestor", "ingestor"), ("Analyst", "analyst"),
            ("Scientist", "scientist"),
        ]
        for display_name, field_name in agent_map:
            cfg = mc.resolve(field_name)
            style = AGENT_STYLES.get(display_name, "")
            table.add_row(
                Text(display_name, style=style),
                Text(f"{cfg.model}  [{cfg.reasoning.level}]", style=style),
            )

        for i, critic in enumerate(mc.critics):
            label = f"Critic {i + 1}" if len(mc.critics) > 1 else "Critic"
            table.add_row(
                Text(label, style="yellow"),
                Text(f"{critic.provider}:{critic.model}  [{critic.reasoning.level}]", style="yellow"),
            )

        # Coder + Report go after critics
        for display_name, field_name in [("Coder", "coder"), ("Report", "report")]:
            cfg = mc.resolve(field_name)
            style = AGENT_STYLES.get(display_name, "")
            table.add_row(
                Text(display_name, style=style),
                Text(f"{cfg.model}  [{cfg.reasoning.level}]", style=style),
            )

        if mc.summarizer:
            s = mc.summarizer
            table.add_row(
                Text("Summarizer", style="dim"),
                Text(f"{s.provider}:{s.model}  [{s.reasoning.level}]", style="dim"),
            )

        return Panel(table, title="Auto-Scientist", title_align="left", border_style="bold")

    def _should_summarize(self) -> bool:
        """Check if summaries are enabled."""
        return self.model_config.summarizer is not None

    @property
    def _summary_model(self) -> str:
        """Return the summarizer model name."""
        if self.model_config.summarizer is None:
            raise RuntimeError("Summarizer not configured")
        return self.model_config.summarizer.model

    async def _with_summaries(
        self, coro_fn, agent_name: str, message_buffer: list[str],
        panel: AgentPanel | None = None,
    ):
        """Wrap an agent call in run_with_summaries if enabled.

        When a panel is provided, summaries are routed to it via
        summary_collector callback instead of being printed directly.
        """
        if not self._should_summarize():
            result = await coro_fn(message_buffer)
            if panel is not None:
                self._apply_sdk_usage(panel)
            return result

        if panel is not None:
            import asyncio
            import contextlib

            summary_collector: list[tuple[str, str, str]] = []
            seen = 0

            async def _poll_collector():
                """Poll collector and push to panel."""
                nonlocal seen
                while True:
                    await asyncio.sleep(0.5)
                    new_entries = summary_collector[seen:]
                    for _name, summary, label in new_entries:
                        panel.add_line(f"[{label}] {summary}")
                        self._live.refresh()
                    seen = len(summary_collector)

            poll_task = asyncio.create_task(_poll_collector())
            try:
                result = await run_with_summaries(
                    coro_fn, agent_name, self._summary_model, message_buffer,
                    summary_collector=summary_collector,
                )
                self._apply_sdk_usage(panel)
            finally:
                poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poll_task
                # Flush entries added since last poll drain
                for _name, summary, label in summary_collector[seen:]:
                    panel.add_line(f"[{label}] {summary}")
                self._live.refresh()
            return result

        return await run_with_summaries(
            coro_fn, agent_name, self._summary_model, message_buffer,
        )

    def _collapse(self, panel: AgentPanel, fallback: str = "") -> None:
        """Collapse a panel, preferring the summarizer's done line over a fallback."""
        if self._should_summarize() and panel.lines:
            # Summarizer populated the panel; let complete() use the last line
            self._live.collapse_panel(panel)
        else:
            self._live.collapse_panel(panel, fallback)

    @staticmethod
    def _apply_sdk_usage(panel: AgentPanel) -> None:
        """Read token usage from the last SDK query and apply it to a panel."""
        from auto_scientist.sdk_utils import collect_text_from_query

        usage = getattr(collect_text_from_query, "last_usage", {})
        if not usage:
            return
        panel.set_stats(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            num_turns=usage.get("num_turns", 0),
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

    @staticmethod
    def _persist_artifact(version_dir: Path, filename: str, data: Any) -> None:
        """Save a structured JSON artifact to a version directory."""
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / filename).write_text(json.dumps(data, indent=2))

    def _notebook_content(self) -> str:
        """Return notebook content."""
        return read_notebook(self.output_dir / NOTEBOOK_FILENAME)

    async def _run_analyst_initial(self) -> dict[str, Any] | None:
        """Iteration 0: analyze raw canonical data instead of experiment results."""
        from auto_scientist.agents.analyst import run_analyst

        notebook_path = self.output_dir / NOTEBOOK_FILENAME
        domain_knowledge = self.state.domain_knowledge
        cfg = self.model_config.resolve("analyst")

        panel = AgentPanel(name="Analyst", model=cfg.model, style="green")
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
                    success_criteria=self.state.success_criteria,
                    data_dir=self.data_path,
                    model=cfg.model,
                    message_buffer=buf,
                )

            analysis = await self._with_summaries(_analyst_coro, "Analyst", buffer, panel=panel)
            logger.info("Analyst initial: data characterization complete")
            self._collapse(panel, "Data characterization complete")
            return analysis
        except Exception as e:
            logger.exception(f"Analyst initial error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
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
            self._live.log("ANALYZE: skipped (no results file)")
            return None

        # Find plot PNGs in the version directory
        version_dir = Path(latest.script_path).parent
        plot_paths = sorted(version_dir.glob("*.png"))

        success_criteria = self.state.success_criteria or []

        cfg = self.model_config.resolve("analyst")

        panel = AgentPanel(name="Analyst", model=cfg.model, style="green")
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
                    success_criteria=success_criteria,
                    model=cfg.model,
                    message_buffer=buf,
                )

            analysis = await self._with_summaries(_analyst_coro, "Analyst", buffer, panel=panel)
            n_criteria = len(analysis.get("criteria_results", []))
            logger.info(
                f"Analyst complete: {n_criteria} criteria evaluated, "
                f"data_summary={'yes' if analysis.get('data_summary') else 'no'}"
            )
            self._collapse(panel, f"Complete ({n_criteria} criteria evaluated)")
            return analysis
        except Exception as e:
            logger.exception(f"Analyst error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
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

        panel = AgentPanel(name="Scientist", model=cfg.model, style="cyan")
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
                    success_criteria=self.state.success_criteria,
                    model=cfg.model,
                    message_buffer=buf,
                )

            plan = await self._with_summaries(_scientist_coro, "Scientist", buffer, panel=panel)

            # Write the notebook entry from the plan
            if plan.get("notebook_entry"):
                append_entry(notebook_path, plan["notebook_entry"], version, "scientist")

            logger.info(
                f"Scientist plan: strategy={plan.get('strategy', '?')}, "
                f"changes={len(plan.get('changes', []))}, "
                f"hypothesis={plan.get('hypothesis', '?')[:100]}, "
                f"should_stop={plan.get('should_stop', False)}"
            )
            self._collapse(
                panel, f"strategy={plan.get('strategy', '?')}, changes={len(plan.get('changes', []))}"
            )
            return plan
        except Exception as e:
            logger.exception(f"Scientist plan error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
            return None
        finally:
            self._persist_buffer("scientist", buffer)

    async def _run_debate(self, plan: dict | None) -> list[dict[str, Any]] | None:
        """Send plan to critic model(s) for parallel debate with the Scientist."""
        if not self.model_config.critics or plan is None:
            self._live.log("DEBATE: skipped (no critics configured or no plan)")
            return None

        from auto_scientist.agents.critic import run_debate
        from auto_scientist.agents.scientist import _format_criteria_for_prompt

        notebook_content = self._notebook_content()
        domain_knowledge = self.state.domain_knowledge
        success_criteria = _format_criteria_for_prompt(self.state.success_criteria)
        scientist_cfg = self.model_config.resolve("scientist")

        # Gather plot PNGs from the latest version directory
        plot_paths: list[Path] = []
        if self.state.versions:
            latest = self.state.versions[-1]
            if latest.script_path:
                version_dir = Path(latest.script_path).parent
                plot_paths = sorted(version_dir.glob("*.png"))

        self._live.update_status(phase="DEBATE")

        # Per-critic buffers
        buffers: dict[str, list[str]] = {}
        for config in self.model_config.critics:
            label = f"{config.provider}:{config.model}"
            buffers[label] = []

        try:
            if self._should_summarize():
                critiques = await self._run_debate_with_summaries(
                    buffers, plan, notebook_content, domain_knowledge,
                    success_criteria, scientist_cfg, plot_paths,
                )
            else:
                critiques = await run_debate(
                    critic_configs=self.model_config.critics,
                    plan=plan,
                    notebook_content=notebook_content,
                    domain_knowledge=domain_knowledge,
                    success_criteria=success_criteria,
                    max_rounds=self.debate_rounds,
                    scientist_config=scientist_cfg,
                    message_buffers=buffers,
                    plot_paths=plot_paths,
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
                self._persist_buffer(safe_name, buf)

    async def _run_debate_with_summaries(
        self,
        buffers: dict[str, list[str]],
        plan: dict,
        notebook_content: str,
        domain_knowledge: str,
        success_criteria: str,
        scientist_cfg: Any,
        plot_paths: list[Path],
    ) -> list[dict[str, Any]]:
        """Run per-critic debates in parallel, each with its own summarizer and panel."""
        import asyncio
        import contextlib

        from auto_scientist.agents.critic import run_single_critic_debate
        from auto_scientist.images import ImageData, encode_images_from_paths

        summary_model = self._summary_model

        # Encode plot images once for all critics
        images: list[ImageData] = []
        if plot_paths:
            images = encode_images_from_paths(plot_paths)
        has_plots = bool(images)

        # Create per-critic panels and collectors
        critic_labels = [
            f"{c.provider}:{c.model}" for c in self.model_config.critics
        ]
        panels: dict[str, AgentPanel] = {}
        collectors: dict[str, list[tuple[str, str, str]]] = {}
        for config, label in zip(self.model_config.critics, critic_labels, strict=True):
            panel = AgentPanel(name="Critic", model=label, style="yellow")
            panels[label] = panel
            collectors[label] = []
            self._live.add_panel(panel)

        # Track how many entries per critic have been flushed to panels
        seen: dict[str, int] = {lb: 0 for lb in critic_labels}

        def _flush_collectors():
            for label in critic_labels:
                entries = collectors[label]
                new_entries = entries[seen[label]:]
                for _agent_name, summary, time_label in new_entries:
                    panels[label].add_line(f"[{time_label}] {summary}")
                seen[label] = len(entries)
            self._live.refresh()

        async def _drain_loop():
            while True:
                await asyncio.sleep(0.5)
                _flush_collectors()

        def _collapse_critic(label: str, result: dict[str, Any]) -> None:
            """Collapse a critic panel with stats from its result."""
            panel = panels[label]
            panel.set_stats(
                input_tokens=result.get("input_tokens", 0),
                output_tokens=result.get("output_tokens", 0),
                num_turns=len(result.get("transcript", [])),
            )
            # Flush collector entries before collapsing
            entries = collectors[label]
            new_entries = entries[seen[label]:]
            for _agent_name, summary, time_label in new_entries:
                panel.add_line(f"[{time_label}] {summary}")
            seen[label] = len(entries)

            done_entries = [
                e for e in collectors[label] if e[2].endswith("done")
            ]
            if done_entries:
                done_summary = done_entries[-1][1]
            else:
                critique = result.get("critique", "")
                done_summary = critique[:200] + "..." if len(critique) > 200 else critique
            self._live.collapse_panel(panel, done_summary or "Debate complete")

        async def _summarized_debate(config, label):
            async def coro(buf):
                return await run_single_critic_debate(
                    config=config,
                    plan=plan,
                    notebook_content=notebook_content,
                    domain_knowledge=domain_knowledge,
                    success_criteria=success_criteria,
                    max_rounds=self.debate_rounds,
                    scientist_config=scientist_cfg,
                    message_buffer=buf,
                    plot_paths=plot_paths,
                    images=images,
                    has_plots=has_plots,
                )
            try:
                result = await run_with_summaries(
                    coro, f"Debate: {label}", summary_model, buffers[label],
                    label_prefix="",
                    summary_collector=collectors[label],
                )
                _collapse_critic(label, result)
                return result
            except Exception as e:
                panels[label].error(str(e))
                self._live.collapse_panel(panels[label])
                raise

        drain_task = asyncio.create_task(_drain_loop())
        raw_results: list[dict[str, Any] | BaseException] = []
        try:
            tasks = [
                _summarized_debate(config, label)
                for config, label in zip(
                    self.model_config.critics, critic_labels, strict=True,
                )
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await drain_task

        # Log failures, return successes
        successful = []
        for r in raw_results:
            if isinstance(r, BaseException):
                logger.error(f"Critic debate failed: {r}")
            else:
                successful.append(r)
        return successful

    async def _run_scientist_revision(
        self,
        plan: dict | None,
        debate_result: list[dict[str, Any]] | None,
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

        # Combine transcripts from all critics
        all_transcript: list[dict[str, str]] = []
        for entry in debate_result:
            all_transcript.append({"role": "critic", "content": f"[{entry['model']}]"})
            all_transcript.extend(entry.get("transcript", []))

        cfg = self.model_config.resolve("scientist")

        panel = AgentPanel(name="Revision", model=cfg.model, style="cyan")
        self._live.add_panel(panel)
        self._live.update_status(phase="REVISE")

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
                _revision_coro, "Scientist Revision", buffer, panel=panel,
            )

            # Write revised notebook entry
            if revised.get("notebook_entry"):
                append_entry(notebook_path, revised["notebook_entry"], version, "revision")

            self._collapse(panel, f"strategy={revised.get('strategy', '?')}")
            return revised
        except Exception as e:
            logger.exception(f"Scientist revision error: {e}")
            panel.error(f"{e}, using original plan")
            self._live.collapse_panel(panel)
            return None
        finally:
            self._persist_buffer("scientist_revision", buffer)

    async def _run_coder(self, plan: dict | None) -> Path | None:
        """Invoke the Coder agent to implement the plan."""
        from auto_scientist.agents.coder import run_coder

        if plan is None:
            self._live.log("IMPLEMENT: skipped (no plan)")
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

        # Resolve the executable to an absolute path so the coder's Bash
        # subprocess can find it even when ~/.local/bin isn't in PATH.
        parts = run_cmd.split()
        if parts:
            exe = parts[0]
            abs_exe = shutil.which(exe)
            if abs_exe and abs_exe != exe:
                run_cmd = abs_exe + run_cmd[len(exe):]
            elif not abs_exe:
                logger.warning(
                    f"Executable '{exe}' not found on PATH; "
                    f"coder may fail to run the experiment script"
                )

        cfg = self.model_config.resolve("coder")

        # Serialize top-level criteria before entering coder try block
        top_criteria = [
            c.model_dump() for c in (self.state.success_criteria or [])
        ]

        panel = AgentPanel(name="Coder", model=cfg.model, style="magenta")
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
                    top_level_criteria=top_criteria or None,
                )

            new_script = await self._with_summaries(_coder_coro, "Coder", buffer, panel=panel)
            self._collapse(panel, f"Created {new_script}")
            return new_script
        except Exception as e:
            logger.exception(f"Coder error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
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
        self._live.log(f"SCORE: {score}")

    async def _score_final_version(self) -> None:
        """Run the Analyst on the last version so it gets a score before report."""
        analysis = await self._run_analyst()
        if analysis:
            self._score_latest(analysis)
            if self.state.versions:
                version_dir = self.output_dir / self.state.versions[-1].version
                self._persist_artifact(version_dir, "final_analysis.json", analysis)

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

    _INFRA_FILES = {
        "run_result.json", "exitcode.txt", "stderr.txt",
        "analysis.json", "plan.json", "debate.json",
    }

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

        from auto_scientist.schemas import CoderRunResult

        from pydantic import ValidationError

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

        cfg = self.model_config.resolve("report")
        panel = AgentPanel(name="Report", model=cfg.model, style="blue")
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
                )

            report_content = await self._with_summaries(_report_coro, "Report", buffer, panel=panel)
            report_path = self.output_dir / "report.md"
            report_path.write_text(report_content)
            self._collapse(panel, f"Written to {report_path}")
        except Exception as e:
            logger.exception(f"Report error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
        finally:
            self._persist_buffer("report", buffer)
