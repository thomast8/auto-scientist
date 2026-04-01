"""SDK backend abstraction for multi-provider agentic support.

Provides a unified interface for both Claude Code SDK (Anthropic) and
Codex SDK (OpenAI), allowing agents to run on either provider without
changing their code.

Monkey-patches the Claude Code SDK message parser at import time so that
unknown message types are silently skipped instead of crashing the stream.
"""

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import claude_code_sdk._internal.client as _client_mod
import claude_code_sdk._internal.message_parser as _parser_mod
from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
)
from claude_code_sdk import query as claude_query
from claude_code_sdk._errors import MessageParseError
from codex_app_server_sdk import CodexClient, ThreadConfig, TurnOverrides
from codex_app_server_sdk.errors import CodexProtocolError

logger = logging.getLogger(__name__)

# Models broken with Codex + ChatGPT subscription auth.
# Workaround: auto-upgrade to the cheapest working model.
# Track: https://github.com/openai/codex/issues/14266
# Remove this when OpenAI fixes the issue.
CODEX_MODEL_OVERRIDES: dict[str, str] = {
    "gpt-5.4-nano": "gpt-5.4-mini",
}

# Codex effort mapping: our levels -> Codex ReasoningEffort strings
_CODEX_EFFORT_MAP: dict[str, str] = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "xhigh",
}


# ---------------------------------------------------------------------------
# Monkey-patch: make parse_message return None for unknown types
# ---------------------------------------------------------------------------

_original_parse_message = _parser_mod.parse_message


def _tolerant_parse_message(data: dict[str, Any]) -> Any:
    """parse_message wrapper that returns None for unknown message types."""
    try:
        return _original_parse_message(data)
    except MessageParseError as exc:
        if "Unknown message type" in str(exc):
            msg_type = data.get("type", "<missing>")
            logger.debug(f"Skipping unknown SDK message type: {msg_type}")
            return None
        raise


_parser_mod.parse_message = _tolerant_parse_message  # type: ignore[assignment]
_client_mod.parse_message = _tolerant_parse_message  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Unified types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SDKOptions:
    """Unified options for both Claude Code SDK and Codex SDK."""

    system_prompt: str
    allowed_tools: tuple[str, ...] | list[str]
    max_turns: int
    model: str | None = None
    cwd: Path | None = None
    permission_mode: str = "default"
    extra_args: dict[str, str | None] = field(default_factory=dict)
    resume: str | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class SDKMessage:
    """Unified message envelope for both SDK backends."""

    type: Literal["assistant", "result"]
    text: str | None = None
    content_blocks: list[Any] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    session_id: str | None = None


class SDKBackend(Protocol):
    """Protocol that both Claude and Codex backends implement."""

    def query(self, prompt: str, options: SDKOptions) -> AsyncIterator[SDKMessage]: ...


# ---------------------------------------------------------------------------
# Claude Code SDK Backend
# ---------------------------------------------------------------------------


class ClaudeBackend:
    """Wraps claude_code_sdk.query() behind the SDKBackend interface."""

    def _build_claude_options(self, options: SDKOptions) -> ClaudeCodeOptions:
        """Convert SDKOptions to ClaudeCodeOptions."""
        env = dict(options.env)

        # Strip ANTHROPIC_API_KEY so the CLI uses subscription auth
        if "ANTHROPIC_API_KEY" not in env and os.environ.get("ANTHROPIC_API_KEY"):
            logger.info(
                "Stripping ANTHROPIC_API_KEY from SDK subprocess env "
                "(using Claude Code subscription instead of direct API billing)"
            )
            env["ANTHROPIC_API_KEY"] = ""

        kwargs: dict[str, Any] = {
            "system_prompt": options.system_prompt,
            "allowed_tools": options.allowed_tools,
            "max_turns": options.max_turns,
            "permission_mode": options.permission_mode,
            "extra_args": dict(options.extra_args),
        }
        if options.model:
            kwargs["model"] = options.model
        if options.cwd:
            kwargs["cwd"] = str(options.cwd)
        if options.resume:
            kwargs["resume"] = options.resume
        if env:
            kwargs["env"] = env

        return ClaudeCodeOptions(**kwargs)

    async def query(self, prompt: str, options: SDKOptions) -> AsyncIterator[SDKMessage]:
        """Run a Claude Code SDK query, yielding unified SDKMessages."""
        cc_opts = self._build_claude_options(options)

        logger.debug(
            f"ClaudeBackend query: model={cc_opts.model}, "
            f"max_turns={cc_opts.max_turns}, prompt_len={len(prompt)}"
        )

        async for msg in claude_query(prompt=prompt, options=cc_opts):
            if msg is None:
                continue

            if isinstance(msg, AssistantMessage):
                yield SDKMessage(
                    type="assistant",
                    content_blocks=list(msg.content),
                )
            elif isinstance(msg, ResultMessage):
                usage = getattr(msg, "usage", None) or {}
                usage["num_turns"] = getattr(msg, "num_turns", 0)
                usage["total_cost_usd"] = getattr(msg, "total_cost_usd", None)
                yield SDKMessage(
                    type="result",
                    result=msg.result if msg.result else None,
                    usage=usage,
                    session_id=getattr(msg, "session_id", None),
                )


# ---------------------------------------------------------------------------
# Codex SDK Backend
# ---------------------------------------------------------------------------

# Sandbox mode type for Codex SDK
_CODEX_APPROVAL_POLICY: Any = "never"


class CodexBackend:
    """Wraps codex_app_server_sdk behind the SDKBackend interface."""

    def _resolve_sandbox(self, allowed_tools: list[str] | tuple[str, ...]) -> str:
        """Map allowed tools to Codex sandbox mode."""
        write_tools = {"Write", "Edit", "Bash"}
        if write_tools & set(allowed_tools):
            return "workspace-write"
        return "read-only"

    def _resolve_effort(self, extra_args: dict[str, str | None]) -> str | None:
        """Map extra_args effort to Codex reasoning effort string."""
        effort = extra_args.get("effort")
        if effort is None:
            return None
        return _CODEX_EFFORT_MAP.get(effort, effort)

    async def query(self, prompt: str, options: SDKOptions) -> AsyncIterator[SDKMessage]:
        """Run a Codex SDK query, yielding unified SDKMessages."""
        sandbox_mode = self._resolve_sandbox(options.allowed_tools)
        effort = self._resolve_effort(options.extra_args)
        model = options.model

        # Build environment for the Codex subprocess.
        # IMPORTANT: create_subprocess_exec replaces the entire env when a
        # dict is passed, so we must start from os.environ and layer overrides.
        env: dict[str, str] | None = None
        needs_env = bool(options.env) or os.environ.get("OPENAI_API_KEY")
        if needs_env:
            env = {**os.environ, **options.env}
            if "OPENAI_API_KEY" not in options.env and os.environ.get("OPENAI_API_KEY"):
                logger.info(
                    "Stripping OPENAI_API_KEY from Codex subprocess env "
                    "(using ChatGPT subscription instead of direct API billing)"
                )
                env["OPENAI_API_KEY"] = ""

        # Build thread config
        thread_config = ThreadConfig(
            model=model,
            base_instructions=options.system_prompt,
            sandbox=sandbox_mode,  # type: ignore[arg-type]
            approval_policy=_CODEX_APPROVAL_POLICY,
        )
        if options.cwd:
            thread_config.cwd = str(options.cwd)

        # Build turn overrides
        turn_overrides = TurnOverrides()
        if effort:
            turn_overrides.effort = effort  # type: ignore[assignment]

        logger.debug(
            f"CodexBackend query: model={model}, "
            f"sandbox={sandbox_mode}, effort={effort}, prompt_len={len(prompt)}"
        )

        # Connect and run using chat() for streaming (enables progress
        # summaries) instead of chat_once() which blocks until turn completes.
        client = CodexClient.connect_stdio(
            cwd=str(options.cwd) if options.cwd else None,
            env=env,
        )
        try:
            await client.start()

            chat_kwargs: dict[str, Any] = {"turn_overrides": turn_overrides}
            if options.resume:
                chat_kwargs["thread_id"] = options.resume
            else:
                chat_kwargs["thread_config"] = thread_config

            final_text_parts: list[str] = []
            thread_id: str | None = None
            step_count = 0

            async for step in client.chat(prompt, **chat_kwargs):
                thread_id = step.thread_id
                step_count += 1
                if step.text:
                    final_text_parts.append(step.text)
                    # Yield each step as an assistant message so the
                    # message_buffer gets populated during the turn
                    # (enables progress summaries).
                    synthetic_block = type("_SyntheticTextBlock", (), {"text": step.text})()
                    yield SDKMessage(
                        type="assistant",
                        content_blocks=[synthetic_block],
                    )

            final_text = "\n".join(final_text_parts) if final_text_parts else ""

            # Codex SDK doesn't expose token usage; report step count as turns.
            yield SDKMessage(
                type="result",
                result=final_text if final_text else None,
                session_id=thread_id,
                usage={"num_turns": step_count},
            )
        except CodexProtocolError as e:
            raise RuntimeError(
                f"Codex query failed (model={model}, sandbox={sandbox_mode}): {e}\n"
                f"If using ChatGPT subscription, note that gpt-5.4-nano is not supported "
                f"by Codex. Use gpt-5.4-mini or higher."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Codex query failed (model={model}, sandbox={sandbox_mode}): {e}"
            ) from e
        finally:
            try:
                await client.close()
            except Exception:
                logger.debug("Error closing Codex client", exc_info=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(provider: str) -> SDKBackend:
    """Return the SDK backend for the given provider."""
    if provider == "anthropic":
        return ClaudeBackend()
    if provider == "openai":
        return CodexBackend()
    raise ValueError(
        f"No SDK backend for provider {provider!r}. SDK mode requires 'anthropic' or 'openai'."
    )


def apply_codex_model_overrides(mc: Any) -> None:
    """Patch a ModelConfig in-place, replacing broken Codex models.

    Call once at startup so the TUI, agent panels, and actual queries
    all show the correct (overridden) model name.

    Remove this function when OpenAI fixes the issue:
    https://github.com/openai/codex/issues/14266
    """
    if not CODEX_MODEL_OVERRIDES:
        return

    def _patch(cfg: Any) -> None:
        if cfg is None or cfg.provider != "openai" or cfg.mode != "sdk":
            return
        replacement = CODEX_MODEL_OVERRIDES.get(cfg.model)
        if replacement:
            logger.warning(
                f"Model {cfg.model!r} is unsupported by Codex with ChatGPT subscription. "
                f"Overriding to {replacement!r}. "
                f"(https://github.com/openai/codex/issues/14266)"
            )
            cfg.model = replacement

    _patch(mc.defaults)
    for agent in ("analyst", "scientist", "coder", "ingestor", "report", "summarizer", "assessor"):
        _patch(getattr(mc, agent, None))
    for critic in mc.critics:
        _patch(critic)
