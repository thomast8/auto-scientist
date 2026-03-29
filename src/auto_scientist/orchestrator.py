"""Main orchestration loop and state machine."""

import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from auto_scientist.config import DomainConfig
from auto_scientist.console import (
    AGENT_STYLES,
    AgentPanel,
    PipelineLive,
    console,
)
from auto_scientist.iteration_manifest import (
    MANIFEST_FILENAME,
    IterationRecord,
    PanelRecord,
    append_record,
    load_manifest,
)
from auto_scientist.log_setup import setup_file_logging
from auto_scientist.model_config import AgentModelConfig, ModelConfig
from auto_scientist.notebook import NOTEBOOK_FILENAME, append_entry, read_notebook
from auto_scientist.runner import RunResult
from auto_scientist.scheduler import wait_for_window
from auto_scientist.state import ExperimentState, VersionEntry
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
        if r.level == "off":
            continue

        label = f"{agent_name} ({cfg.provider}/{cfg.model}, reasoning={r.level})"

        if cfg.provider == "anthropic":
            budget = r.budget or ANTHROPIC_BUDGET_DEFAULTS.get(r.level)
            if budget is None:
                errors.append(f"{label}: no budget_tokens mapping for level '{r.level}'")
            elif budget < 1024:
                errors.append(f"{label}: budget_tokens={budget} is below Anthropic minimum (1024)")
            elif budget > 128_000:
                errors.append(f"{label}: budget_tokens={budget} exceeds Anthropic maximum (128000)")

        elif cfg.provider == "openai":
            effort = OPENAI_EFFORT_MAP.get(r.level)
            if effort is None:
                errors.append(f"{label}: no reasoning effort mapping for level '{r.level}'")

        elif cfg.provider == "google":
            model = cfg.model
            is_3x = "gemini-3" in model
            is_25 = "2.5" in model

            if is_3x:
                mapped = GOOGLE_LEVEL_MAP.get(r.level)
                if mapped is None:
                    errors.append(f"{label}: no thinkingLevel mapping for level '{r.level}'")
                elif mapped in ("MINIMAL", "MEDIUM") and "flash" not in model.lower():
                    errors.append(
                        f"{label}: thinkingLevel={mapped} is only valid for Gemini 3 Flash models"
                    )
            elif is_25:
                # Budget-based; just verify we have a default
                if r.budget is None:
                    from auto_scientist.models.google_client import GOOGLE_BUDGET_DEFAULTS

                    if r.level not in GOOGLE_BUDGET_DEFAULTS:
                        errors.append(f"{label}: no thinkingBudget mapping for level '{r.level}'")

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

            google_client = genai.Client()
            google_client.models.get(model=model)
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
        debate_rounds: int = 1,
        stream: bool = True,
        verbose: bool = False,
    ):
        self.state = state
        self.data_path = data_path.resolve() if data_path else data_path
        self.output_dir = output_dir.resolve()
        self.max_iterations = max_iterations
        self.model_config = model_config or ModelConfig.builtin_preset("default")
        self.interactive = interactive
        self.max_consecutive_failures = max_consecutive_failures
        self.debate_rounds = debate_rounds
        self.config: DomainConfig | None = None
        self.stream = stream
        self.verbose = verbose
        self._live: PipelineLive = PipelineLive()
        self.pause_requested: bool = False
        self.skip_to_report: bool = False

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
            raise RuntimeError("Pre-flight check failed:\n  - " + "\n  - ".join(errors))

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

        # Restore previous iterations from manifest (for replay / resume)
        self._restore_iterations_from_manifest()

        try:
            # Phase 0: Ingestion (with its own border)
            if self.state.phase == "ingestion":
                self._live.start_iteration("Ingestion")

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

                iter_summary = await self._generate_iteration_summary()
                self._save_iteration_manifest("Ingestion", "done", "green", iter_summary)
                self._live.end_iteration("done", "green", iter_summary)
                self._live.flush_completed()

            # Phase 1: Unified iteration loop
            while self.state.phase == "iteration":
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

                if self.state.should_stop_on_failures(self.max_consecutive_failures):
                    self._live.add_rule(
                        Rule(
                            f"Hit {self.max_consecutive_failures} consecutive failures. Stopping.",
                            style="red",
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

                await self._run_iteration_body()
                self.state.save(state_path)

            # Phase 2: Report (with its own border)
            if self.state.phase == "report":
                self._live.start_iteration("Report")

                # Resolve pending predictions for the final version
                if self.state.versions:
                    await self._resolve_final_predictions()
                    self.state.save(state_path)

                await self._run_report()
                self.state.phase = "stopped"
                self.state.save(state_path)

                iter_summary = await self._generate_iteration_summary()
                self._save_iteration_manifest(
                    "Report",
                    "done",
                    "green",
                    iter_summary,
                )
                self._live.end_iteration("done", "green", iter_summary)
                self._live.flush_completed()

            logger.info("Run finished successfully")
        finally:
            self._live.wait_for_dismiss()
            self._live.stop()

    async def _run_ingestion(self) -> Path:
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
            name="Ingestor", model=cfg.model, style=AGENT_STYLES.get("Ingestor", "bright_red")
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
            )

        buffer: list[str] = []
        try:
            canonical_data_dir: Path = await self._with_summaries(
                _ingestor_coro,
                "Ingestor",
                buffer,
                panel=panel,
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

        # Resolve any pending prediction outcomes from the Analyst
        self._resolve_prediction_outcomes(analysis)

        # Step 2: Scientist plans next iteration
        plan = await self._run_scientist_plan(analysis)

        # Step 3: Check if Scientist recommends stopping
        if plan and plan.get("should_stop"):
            self._persist_artifact(version_dir, "plan.json", plan)
            logger.info(f"Scientist stop: {plan.get('stop_reason', 'unknown')}")
            self.state.phase = "report"
            iter_summary = await self._generate_iteration_summary()
            stop_label = f"stopped: {plan.get('stop_reason', 'unknown')}"
            self._save_iteration_manifest(self.state.iteration, stop_label, "yellow", iter_summary)
            self._live.end_iteration(stop_label, "yellow", iter_summary)
            self._live.flush_completed()
            return

        # Step 4: Debate (subset personas on iteration 0, all on iteration 1+)
        debate_result = await self._run_debate(plan, analysis)
        revised_plan = await self._run_scientist_revision(plan, debate_result, analysis)
        final_plan = revised_plan or plan

        # Persist debate transcript with original plan for context
        if debate_result:
            from auto_scientist.agents.debate_models import DebateResult

            serialized_results = [
                r.model_dump() if isinstance(r, DebateResult) else r for r in debate_result
            ]
            concern_ledger = self._build_concern_ledger(debate_result)
            self._persist_artifact(
                version_dir,
                "debate.json",
                {
                    "original_plan": plan,
                    "debate_results": serialized_results,
                    "concern_ledger": concern_ledger,
                },
            )

        # Apply prediction updates from the final plan (after debate revision)
        if final_plan:
            self._apply_prediction_updates(final_plan)

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
            iter_summary = await self._generate_iteration_summary()
            self._save_iteration_manifest(
                self.state.iteration, "failed (no script)", "red", iter_summary
            )
            self._live.end_iteration("failed (no script)", "red", iter_summary)
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
        self._evaluate(run_result, version_entry)
        self.state.record_version(version_entry)

        # Iteration border color: green=completed, red=failed
        status_style = "red" if version_entry.status != "completed" else "green"
        iter_summary = await self._generate_iteration_summary()
        self._save_iteration_manifest(
            self.state.iteration, version_entry.status, status_style, iter_summary
        )
        self._live.end_iteration(version_entry.status, status_style, iter_summary)
        self._live.flush_completed()

        # Increment at end of loop body
        logger.info(f"Iteration {self.state.iteration} complete: status={version_entry.status}")
        self.state.iteration += 1

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
                Text(f"{cfg.model}  [{cfg.reasoning.level}]", style=style),
            )

        for i, critic in enumerate(mc.critics):
            label = f"Critic {i + 1}" if len(mc.critics) > 1 else "Critic"
            table.add_row(
                Text(label, style="yellow"),
                Text(
                    f"{critic.provider}:{critic.model}  [{critic.reasoning.level}]",
                    style="yellow",
                ),
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

    def _collect_iteration_record(
        self,
        title: str | int,
        result_text: str,
        result_style: str,
        summary: str,
    ) -> IterationRecord:
        """Snapshot current iteration's panel metadata for the manifest."""
        container = self._live._current_iteration
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
        return IterationRecord(
            iteration="ingestion" if isinstance(title, str) else self.state.iteration,
            title=f"Iteration {title}" if isinstance(title, int) else str(title),
            result_text=result_text,
            result_style=result_style,
            summary=summary,
            panels=panels,
        )

    def _save_iteration_manifest(
        self,
        title: str | int,
        result_text: str,
        result_style: str,
        summary: str,
    ) -> None:
        """Collect and persist an iteration record to the manifest file."""
        record = self._collect_iteration_record(title, result_text, result_style, summary)
        append_record(record, self.output_dir / MANIFEST_FILENAME)

    def _restore_iterations_from_manifest(self) -> None:
        """Mount collapsed iteration panels from a saved manifest.

        Called at startup to show previous iterations in the TUI when
        resuming or replaying. Silently skips if no manifest exists.
        """
        manifest_path = self.output_dir / MANIFEST_FILENAME
        records = load_manifest(manifest_path)
        if not records:
            return

        for record in records:
            self._live.mount_restored_iteration(
                title=record.title,
                result_text=record.result_text,
                result_style=record.result_style,
                summary=record.summary,
                panels=[p.model_dump() for p in record.panels],
            )

    async def _generate_iteration_summary(self) -> str:
        """Generate a combined recap from all agents' done_summaries for the current iteration."""
        if not self._should_summarize():
            return ""
        container = self._live._current_iteration
        if container is None:
            return ""
        summaries = [
            (p.panel_name, p.done_summary)
            for p in getattr(container, "_panels", [])
            if p.done and p.done_summary
        ]
        if not summaries:
            return ""
        from auto_scientist.summarizer import summarize_iteration

        return await summarize_iteration(summaries, self._summary_model)

    @property
    def _summary_model(self) -> str:
        """Return the summarizer model name."""
        if self.model_config.summarizer is None:
            raise RuntimeError("Summarizer not configured")
        return self.model_config.summarizer.model

    async def _with_summaries(
        self,
        coro_fn: Callable[..., Any],
        agent_name: str,
        message_buffer: list[str],
        panel: AgentPanel | None = None,
    ) -> Any:
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
                    coro_fn,
                    agent_name,
                    self._summary_model,
                    message_buffer,
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
            coro_fn,
            agent_name,
            self._summary_model,
            message_buffer,
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
        # Claude Code SDK splits input tokens across cache buckets:
        # input_tokens (non-cached) + cache_creation + cache_read = total input
        in_tok = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        panel.set_stats(
            input_tokens=in_tok,
            output_tokens=usage.get("output_tokens", 0),
            thinking_tokens=usage.get("thinking_tokens", 0),
            num_turns=usage.get("num_turns", 0),
        )

    def _persist_buffer(
        self,
        agent_name: str,
        buffer: list[str],
        iteration: int | None = None,
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

        panel = AgentPanel(
            name="Analyst", model=cfg.model, style=AGENT_STYLES.get("Analyst", "green")
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
                )

            analysis: dict[str, Any] | None = await self._with_summaries(
                _analyst_coro,
                "Analyst",
                buffer,
                panel=panel,
            )
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

        # Find results file (resolve to absolute for agent cwd consistency)
        results_path = Path(latest.results_path).resolve() if latest.results_path else None
        if not results_path or not results_path.exists():
            self._live.log("ANALYZE: skipped (no results file)")
            return None

        # Find plot PNGs in the version directory
        version_dir = Path(latest.script_path).parent
        plot_paths = sorted(version_dir.glob("*.png"))

        cfg = self.model_config.resolve("analyst")

        panel = AgentPanel(
            name="Analyst", model=cfg.model, style=AGENT_STYLES.get("Analyst", "green")
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
                )

            analysis: dict[str, Any] = await self._with_summaries(
                _analyst_coro,
                "Analyst",
                buffer,
                panel=panel,
            )
            logger.info(
                f"Analyst complete: data_summary={'yes' if analysis.get('data_summary') else 'no'}"
            )
            self._collapse(panel, "Analysis complete")
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

        panel = AgentPanel(
            name="Scientist", model=cfg.model, style=AGENT_STYLES.get("Scientist", "cyan")
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
                )

            plan: dict[str, Any] = await self._with_summaries(
                _scientist_coro,
                "Scientist",
                buffer,
                panel=panel,
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
            self._collapse(
                panel,
                f"strategy={plan.get('strategy', '?')}, changes={len(plan.get('changes', []))}",
            )
            return plan
        except Exception as e:
            logger.exception(f"Scientist plan error: {e}")
            panel.error(str(e))
            self._live.collapse_panel(panel)
            return None
        finally:
            self._persist_buffer("scientist", buffer)

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

        # Pre-format analysis and prediction history as strings for debate
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
            if self._should_summarize():
                critiques = await self._run_debate_with_summaries(
                    buffers,
                    plan,
                    notebook_content,
                    domain_knowledge,
                    scientist_cfg,
                    analysis_json,
                    prediction_history,
                    self.state.goal,
                )
            else:
                critiques = await run_debate(
                    critic_configs=self.model_config.critics,
                    plan=plan,
                    notebook_content=notebook_content,
                    domain_knowledge=domain_knowledge,
                    max_rounds=self.debate_rounds,
                    scientist_config=scientist_cfg,
                    message_buffers=buffers,
                    iteration=self.state.iteration,
                    analysis_json=analysis_json,
                    prediction_history=prediction_history,
                    goal=self.state.goal,
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
        scientist_cfg: Any,
        analysis_json: str,
        prediction_history: str,
        goal: str = "",
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
            label = f"{config.provider}:{config.model}"
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
                assessment = ""
                if result.rounds:
                    assessment = result.rounds[-1].critic_output.overall_assessment
                done_summary = assessment[:200] + "..." if len(assessment) > 200 else assessment
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
                    max_rounds=self.debate_rounds,
                    scientist_config=scientist_cfg,
                    message_buffer=buf,
                    persona=persona,
                    analysis_json=analysis_json,
                    prediction_history=prediction_history,
                    goal=goal,
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

    @staticmethod
    def _build_concern_ledger(debate_results: list) -> list[dict[str, Any]]:
        """Build a concern ledger from structured debate results.

        For each concern in the final round's CriticOutput, attach the persona,
        model, and (if available) the scientist's defense verdict.
        """
        from auto_scientist.agents.debate_models import ConcernLedgerEntry, DebateResult

        ledger: list[dict[str, Any]] = []
        for result in debate_results:
            if not isinstance(result, DebateResult):
                continue
            if not result.rounds:
                continue

            # Build ledger from each round's concerns paired with that
            # round's defense (not cross-round positional matching).
            seen_claims: set[str] = set()
            for rnd in result.rounds:
                critic_output = rnd.critic_output
                defense_responses = rnd.scientist_defense.responses if rnd.scientist_defense else []
                # Build a lookup by concern text for fuzzy matching
                defense_by_text = {}
                for resp in defense_responses:
                    defense_by_text[resp.concern.lower().strip()] = resp

                for i, concern in enumerate(critic_output.concerns):
                    # Deduplicate concerns that appear across rounds
                    claim_key = concern.claim.lower().strip()
                    if claim_key in seen_claims:
                        continue
                    seen_claims.add(claim_key)

                    # Try text match first, fall back to positional
                    matched_resp = defense_by_text.get(claim_key)
                    if matched_resp is None and i < len(defense_responses):
                        matched_resp = defense_responses[i]
                    verdict = matched_resp.verdict if matched_resp else None
                    reasoning = matched_resp.reasoning if matched_resp else None

                    entry = ConcernLedgerEntry(
                        claim=concern.claim,
                        severity=concern.severity,
                        confidence=concern.confidence,
                        category=concern.category,
                        persona=result.persona,
                        critic_model=result.critic_model,
                        scientist_verdict=verdict,
                        scientist_reasoning=reasoning,
                    )
                    ledger.append(entry.model_dump())

        return ledger

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
        concern_ledger = self._build_concern_ledger(debate_result)

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
                )

            revised: dict[str, Any] = await self._with_summaries(
                _revision_coro,
                "Scientist Revision",
                buffer,
                panel=panel,
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
        data_path = str(Path(self.state.data_path).resolve()) if self.state.data_path else ""

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
                run_cmd = abs_exe + run_cmd[len(exe) :]
            elif not abs_exe:
                logger.warning(
                    f"Executable '{exe}' not found on PATH; "
                    f"coder may fail to run the experiment script"
                )

        cfg = self.model_config.resolve("coder")

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
            name="Coder", model=cfg.model, style=AGENT_STYLES.get("Coder", "magenta1")
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
                )

            new_script: Path | None = await self._with_summaries(
                _coder_coro,
                "Coder",
                buffer,
                panel=panel,
            )
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

    _VALID_OUTCOMES = {"confirmed", "refuted", "inconclusive"}

    def _apply_prediction_updates(self, plan: dict[str, Any]) -> None:
        """Extract testable predictions from the Scientist plan and store as pending records.

        Assigns ordinal pred_ids like "1.1", "1.2" and injects them back into the
        plan dict so the Coder can include them in HYPOTHESIS TESTS output.
        Skips malformed or empty prediction entries.
        """
        from auto_scientist.state import PredictionRecord

        predictions = plan.get("testable_predictions", [])
        stored = []
        for i, pred in enumerate(predictions, 1):
            if not isinstance(pred, dict):
                logger.warning(
                    f"Prediction {i}: expected dict, got {type(pred).__name__}; skipping"
                )
                continue
            text = pred.get("prediction", "")
            if not text or not isinstance(text, str) or not text.strip():
                logger.warning(f"Prediction {i}: empty or invalid prediction text; skipping")
                continue
            pred_id = f"{self.state.iteration}.{i}"
            record = PredictionRecord(
                pred_id=pred_id,
                iteration_prescribed=self.state.iteration,
                prediction=text,
                diagnostic=pred.get("diagnostic", ""),
                if_confirmed=pred.get("if_confirmed", ""),
                if_refuted=pred.get("if_refuted", ""),
                follows_from=pred.get("follows_from"),
            )
            self.state.prediction_history.append(record)
            pred["pred_id"] = pred_id
            stored.append(pred_id)
        if stored:
            logger.info(f"Stored {len(stored)} predictions: {', '.join(stored)}")

    def _resolve_prediction_outcomes(self, analysis: dict[str, Any] | None) -> None:
        """Match Analyst prediction outcomes against pending records in state.

        Primary matching by pred_id (reliable). Falls back to substring text
        matching for backward compatibility with a minimum length guard.
        """
        if not analysis:
            return
        outcomes = analysis.get("prediction_outcomes", [])
        if not outcomes:
            return

        pending = [r for r in self.state.prediction_history if r.outcome == "pending"]
        if not pending:
            return

        pending_by_id = {r.pred_id: r for r in pending if r.pred_id}

        def _resolve(record, outcome: dict) -> bool:
            raw_outcome = outcome.get("outcome", "")
            normalized = raw_outcome.strip().lower() if isinstance(raw_outcome, str) else ""
            if normalized not in self._VALID_OUTCOMES:
                logger.warning(
                    f"Prediction {record.pred_id}: invalid outcome '{raw_outcome}', leaving pending"
                )
                return False
            record.outcome = normalized
            record.evidence = outcome.get("evidence", "")
            record.iteration_evaluated = self.state.iteration
            return True

        for outcome in outcomes:
            if not isinstance(outcome, dict):
                logger.warning(
                    f"Prediction outcome: expected dict, got {type(outcome).__name__}; skipping"
                )
                continue

            # Primary: match by pred_id
            oid = outcome.get("pred_id", "")
            if oid and oid in pending_by_id:
                record = pending_by_id[oid]
                if _resolve(record, outcome):
                    pending_by_id.pop(oid)
                    pending.remove(record)
                    logger.info(f"Prediction {oid}: resolved by ID as '{record.outcome}'")
                continue

            # Fallback: substring text matching (minimum length guard)
            outcome_text = outcome.get("prediction", "")
            outcome_text = outcome_text.lower().strip() if isinstance(outcome_text, str) else ""
            if len(outcome_text) < 10:
                logger.warning(
                    f"Prediction outcome text too short for text matching: '{outcome_text}'"
                )
                continue

            best_match = None
            best_score = 0
            for record in pending:
                record_text = record.prediction.lower()
                if outcome_text in record_text or record_text in outcome_text:
                    score = len(record_text)
                    if score > best_score:
                        best_match = record
                        best_score = score
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
                    f"Prediction outcome unmatched: pred_id='{oid}', text='{outcome_text[:80]}'..."
                )

    async def _resolve_final_predictions(self) -> None:
        """Run the Analyst on the last version to resolve pending predictions."""
        analysis = await self._run_analyst()
        if analysis:
            self._resolve_prediction_outcomes(analysis)
            if self.state.versions:
                version_dir = self.output_dir / self.state.versions[-1].version
                self._persist_artifact(version_dir, "final_analysis.json", analysis)

    _INFRA_FILES = {
        "run_result.json",
        "exitcode.txt",
        "stderr.txt",
        "analysis.json",
        "plan.json",
        "debate.json",
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
            if f.suffix in (".png", ".txt", ".csv", ".json") and f.name not in self._INFRA_FILES
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
        panel = AgentPanel(name="Report", model=cfg.model, style=AGENT_STYLES.get("Report", "blue"))
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
