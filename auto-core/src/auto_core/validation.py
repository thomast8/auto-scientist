"""Pre-flight validation for the orchestrator pipeline.

Validates environment (PATH, executables), provider authentication
(API keys, CLI login status), model availability, and reasoning config
compatibility before the pipeline starts. Functions may perform subprocess
calls, network requests, and filesystem checks.
"""

import json
import logging
import os
import shutil
from pathlib import Path

from auto_core.config import RunConfig
from auto_core.model_config import AgentModelConfig, ModelConfig
from auto_core.state import RunState

logger = logging.getLogger(__name__)


def check_provider_auth(provider: str) -> str | None:
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


def check_claude_cli_auth(claude_bin: str) -> str | None:
    """Check that the Claude Code CLI is logged in. Returns error message or None."""
    import subprocess

    try:
        result = subprocess.run(
            [claude_bin, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return (
                "Claude Code CLI is not logged in (needed by Anthropic SDK agents). "
                "Run: claude login"
            )
        # Parse JSON output to check loggedIn field
        try:
            status = json.loads(result.stdout)
            if not status.get("loggedIn"):
                return (
                    "Claude Code CLI is not logged in (needed by Anthropic SDK agents). "
                    "Run: claude login"
                )
        except (json.JSONDecodeError, KeyError):
            pass  # If we can't parse, assume OK (returncode was 0)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning(f"Could not check Claude CLI auth status: {e}")
    return None


def check_codex_cli_auth() -> str | None:
    """Check that the Codex CLI is logged in. Returns error message or None."""
    auth_path = Path.home() / ".codex" / "auth.json"
    if not auth_path.exists():
        return "Codex CLI is not logged in (needed by OpenAI SDK agents). Run: codex login"
    try:
        auth = json.loads(auth_path.read_text())
        has_auth = bool(auth.get("tokens")) or bool(auth.get("OPENAI_API_KEY"))
        if not has_auth:
            return (
                "Codex CLI has no valid credentials (needed by OpenAI SDK agents). Run: codex login"
            )
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read Codex auth file: {e}")
    return None


def validate_reasoning_configs(mc: ModelConfig) -> list[str]:
    """Validate reasoning configs are compatible with their provider and model.

    Returns a list of error messages for invalid configurations.
    """
    from auto_core.models.anthropic_client import ANTHROPIC_BUDGET_DEFAULTS
    from auto_core.models.google_client import GOOGLE_LEVEL_MAP
    from auto_core.models.openai_client import OPENAI_EFFORT_MAP

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
                    from auto_core.models.google_client import GOOGLE_BUDGET_DEFAULTS

                    if r.level not in GOOGLE_BUDGET_DEFAULTS:
                        errors.append(f"{label}: no thinkingBudget mapping for level '{r.level}'")

    return errors


def validate_model_names(mc: ModelConfig) -> list[str]:
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
    # (auth failures are reported separately by check_provider_auth)
    authenticated_providers: set[str] = set()
    for provider in {p for p, _ in pairs}:
        if check_provider_auth(provider) is None:
            authenticated_providers.add(provider)

    for (provider, model), agents in pairs.items():
        if provider not in authenticated_providers:
            continue
        err = check_model_exists(provider, model)
        if err:
            agent_list = ", ".join(sorted(set(agents)))
            errors.append(f"Model '{model}' ({provider}) not found (used by: {agent_list}): {err}")

    return errors


def check_model_exists(provider: str, model: str) -> str | None:
    """Check if a model exists by querying the provider API.

    Returns an error message if the model doesn't exist, None if it does.
    Ignores auth errors (handled separately by check_provider_auth).
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


def validate_prerequisites(
    state: RunState,
    data_path: Path | None,
    output_dir: Path,
    model_config: ModelConfig,
    config: RunConfig | None,
) -> None:
    """Validate directories, API keys, and config before starting.

    Raises RuntimeError with all problems at once so the user can fix
    everything in a single pass.
    """
    errors: list[str] = []

    # Data path must exist when starting from ingestion
    if state.phase == "ingestion":
        if data_path is None:
            errors.append("--data is required for a new run")
        elif not data_path.exists():
            errors.append(f"Data path does not exist: {data_path}")

    # Output dir parent must be writable
    parent = output_dir.parent
    if parent.exists() and not os.access(parent, os.W_OK):
        errors.append(f"Output directory parent is not writable: {parent}")

    # uv must be installed (runs experiment scripts) - unless the coder
    # uses OpenAI/Codex, which rewrites uv run -> python3 in the prompt.
    coder_cfg = model_config.resolve("coder")
    coder_uses_codex = coder_cfg.provider == "openai" and coder_cfg.mode == "sdk"
    run_cmd = config.run_command if config else "uv run {script_path}"
    exe = run_cmd.split()[0] if run_cmd.strip() else ""
    if exe and not coder_uses_codex and not shutil.which(exe):
        errors.append(
            f"'{exe}' not found on PATH (needed for run_command: {run_cmd}). "
            f"Install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )

    # Validate SDK agent provider+mode combinations.
    #
    # `scientist` (reused by auto-reviewer as the Hunter slot) also belongs
    # in `sdk_only_agents`: the scientist / hunter / hunter-revision /
    # stop-revision implementations call `get_backend(provider)` directly
    # with no API-mode fallback, so mode='api' would crash at runtime. The
    # `sdk_capable_agents` list is therefore identical; keep both spellings
    # for readability.
    mc = model_config
    sdk_only_agents = ["analyst", "scientist", "coder", "ingestor", "report", "assessor"]
    sdk_capable_agents = ["analyst", "scientist", "coder", "ingestor", "report", "assessor"]
    needs_claude_cli = False
    needs_codex_cli = False

    for agent_name in sdk_capable_agents:
        cfg = mc.resolve(agent_name)

        # SDK-only agents cannot use mode=api
        if agent_name in sdk_only_agents and cfg.mode == "api":
            errors.append(
                f"{agent_name} uses mode='api', but it requires mode='sdk' "
                f"(needs file tools). Remove the mode override or set mode='sdk'."
            )

        # SDK mode requires anthropic or openai (not google)
        if cfg.mode == "sdk" and cfg.provider == "google":
            errors.append(
                f"{agent_name} uses mode='sdk' with provider='google', "
                f"but no Google coding agent CLI exists. "
                f"Use provider='anthropic' or provider='openai' for SDK mode."
            )

        # Track which CLIs are needed
        if cfg.mode == "sdk":
            if cfg.provider == "anthropic":
                needs_claude_cli = True
            elif cfg.provider == "openai":
                needs_codex_cli = True

    # Also validate critic configs for SDK prerequisites
    for i, critic_cfg in enumerate(mc.critics):
        if critic_cfg.mode == "sdk" and critic_cfg.provider == "google":
            errors.append(
                f"critic[{i}] uses mode='sdk' with provider='google', "
                f"but no Google coding agent CLI exists."
            )
        if critic_cfg.mode == "sdk":
            if critic_cfg.provider == "anthropic":
                needs_claude_cli = True
            elif critic_cfg.provider == "openai":
                needs_codex_cli = True

    # Check CLI availability and login status based on actual needs
    if needs_claude_cli:
        claude_bin = shutil.which("claude")
        if not claude_bin:
            errors.append(
                "Claude Code CLI not found on PATH (needed by Anthropic SDK agents). "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )
        else:
            err = check_claude_cli_auth(claude_bin)
            if err:
                errors.append(err)

    if needs_codex_cli:
        codex_bin = shutil.which("codex")
        if not codex_bin:
            errors.append(
                "Codex CLI not found on PATH (needed by OpenAI SDK agents). "
                "Install with: npm install -g @openai/codex"
            )
        else:
            err = check_codex_cli_auth()
            if err:
                errors.append(err)

    # Validate model names against provider APIs
    model_errors = validate_model_names(mc)
    errors.extend(model_errors)

    # Validate reasoning configs against provider constraints
    reasoning_errors = validate_reasoning_configs(mc)
    errors.extend(reasoning_errors)

    # Collect required providers from model config
    required_providers: set[str] = set()

    # Add providers for SDK agents (API key needed for api mode,
    # subscription or key for sdk mode)
    for agent_name in sdk_capable_agents:
        cfg = mc.resolve(agent_name)
        if cfg.mode == "api":
            required_providers.add(cfg.provider)
        # SDK mode: CLI handles auth (subscription or key), skip API key check

    # Summarizer uses direct API
    if mc.summarizer is not None:
        required_providers.add(mc.summarizer.provider)

    # Critics use their configured provider (api mode by default)
    for critic in mc.critics:
        if critic.mode == "api":
            required_providers.add(critic.provider)

    # Validate each provider by trying to instantiate its SDK client
    for provider in sorted(required_providers):
        err = check_provider_auth(provider)
        if err:
            errors.append(err)

    if errors:
        raise RuntimeError("Pre-flight check failed:\n  - " + "\n  - ".join(errors))
