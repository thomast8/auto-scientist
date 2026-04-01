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
from typing import Any

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

logger = logging.getLogger(__name__)

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


@dataclass
class SDKOptions:
    """Unified options for both Claude Code SDK and Codex SDK."""

    system_prompt: str
    allowed_tools: list[str]
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

    type: str  # "assistant" or "result"
    text: str | None = None
    content_blocks: list[Any] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    session_id: str | None = None


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
_CODEX_SANDBOX_WRITE: Any = "workspace-write"
_CODEX_SANDBOX_READ: Any = "read-only"
_CODEX_APPROVAL_POLICY: Any = "never"


class CodexBackend:
    """Wraps codex_app_server_sdk behind the SDKBackend interface."""

    def _resolve_sandbox(self, allowed_tools: list[str]) -> str:
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

        # Build environment, stripping OPENAI_API_KEY for subscription mode
        env: dict[str, str] = dict(options.env)
        if "OPENAI_API_KEY" not in env and os.environ.get("OPENAI_API_KEY"):
            logger.info(
                "Stripping OPENAI_API_KEY from Codex subprocess env "
                "(using ChatGPT subscription instead of direct API billing)"
            )
            env["OPENAI_API_KEY"] = ""

        # Build thread config
        thread_config = ThreadConfig(
            model=options.model,
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
            f"CodexBackend query: model={options.model}, "
            f"sandbox={sandbox_mode}, effort={effort}, prompt_len={len(prompt)}"
        )

        # Connect and run
        client = CodexClient.connect_stdio(
            cwd=str(options.cwd) if options.cwd else None,
            env=env if env else None,
        )
        try:
            await client.start()

            # Use chat_once for simple request-response
            if options.resume:
                result = await client.chat_once(
                    prompt,
                    thread_id=options.resume,
                    turn_overrides=turn_overrides,
                )
            else:
                result = await client.chat_once(
                    prompt,
                    thread_config=thread_config,
                    turn_overrides=turn_overrides,
                )

            # Synthesize an assistant message so agents that read from
            # assistant blocks (report, coder, ingestor) get the text content
            # for message buffers and report assembly.
            if result.final_text:
                synthetic_block = type("_SyntheticTextBlock", (), {"text": result.final_text})()
                yield SDKMessage(
                    type="assistant",
                    content_blocks=[synthetic_block],
                )

            yield SDKMessage(
                type="result",
                result=result.final_text,
                session_id=result.thread_id,
                usage={"_provider": "codex", "_usage_unavailable": True},
            )
        except Exception as e:
            raise RuntimeError(
                f"Codex query failed (model={options.model}, sandbox={sandbox_mode}): {e}"
            ) from e
        finally:
            try:
                await client.close()
            except Exception:
                logger.debug("Error closing Codex client", exc_info=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_backend(provider: str) -> "ClaudeBackend | CodexBackend":
    """Return the SDK backend for the given provider."""
    if provider == "anthropic":
        return ClaudeBackend()
    if provider == "openai":
        return CodexBackend()
    raise ValueError(
        f"No SDK backend for provider {provider!r}. SDK mode requires 'anthropic' or 'openai'."
    )
