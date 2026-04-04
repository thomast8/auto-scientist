"""SDK backend abstraction for multi-provider agentic support.

Provides a unified interface for both Claude Code SDK (Anthropic) and
Codex SDK (OpenAI), allowing agents to run on either provider without
changing their code.

Monkey-patches the Claude Code SDK message parser at import time so that
unknown message types are silently skipped instead of crashing the stream.
"""

import asyncio
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import claude_code_sdk._internal.client as _client_mod
import claude_code_sdk._internal.message_parser as _parser_mod
import codex_app_server_sdk.transport as _codex_transport_mod
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

# Codex sandbox addendum for tool-using agents.
# uv panics inside the Codex macOS seatbelt sandbox because the
# system-configuration Rust crate can't access SCDynamicStore.
# Track: https://github.com/astral-sh/uv/issues/16664
# Injected into system prompts only when provider == "openai".
CODEX_SANDBOX_ADDENDUM = """\

<sandbox_environment>
You are running inside a sandboxed environment where `uv` is not available.
The run command in the task instructions already uses `python3` instead.

Before running any script, install its dependencies:
1. Read the PEP 723 metadata block at the top of the script to find dependencies
2. Install them: `pip install <dep1> <dep2> ...`
3. Then run the script using the command from the task instructions

IMPORTANT: Every time you edit the script to add a new import, you MUST also:
- Add the package to the PEP 723 dependencies block
- Run `pip install <new_package>` before re-running the script

The script must still declare dependencies in the PEP 723 block for
reproducibility outside this environment.
</sandbox_environment>
"""

# Tools that should never be available to SDK subprocesses.
# Agent/Skill could recurse into host plugins; we block them explicitly.
_DISALLOWED_SUBPROCESS_TOOLS = "Agent,Skill"


@dataclass(frozen=True)
class _IsolationConfig:
    """CLI extra_args and env vars for subprocess isolation."""

    extra_args: dict[str, str | None]
    env: dict[str, str]


def _isolation_config() -> _IsolationConfig:
    """Build isolation settings for SDK subprocesses.

    Instead of --bare (which strips all tools except Bash/Edit/Read),
    we use targeted flags and env vars to isolate from host config
    while keeping the full tool set (WebSearch, Glob, Grep, Write):

    CLI flags (extra_args):
    - ``--setting-sources ''`` skips host user/project/local settings,
      hooks, and CLAUDE.md auto-discovery.
    - ``--disallowed-tools Agent,Skill`` prevents recursion into host
      plugins and skill invocations.

    Env vars:
    - ``CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`` prevents host memory files
      (MEMORY.md) from leaking into subprocess context.
    - ``CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`` prevents CLAUDE.md discovery
      (belt-and-suspenders with --setting-sources '').

    Auth works natively (keychain on macOS, ANTHROPIC_API_KEY env
    on other platforms) since we no longer block keychain reads.
    """
    return _IsolationConfig(
        extra_args={
            "setting-sources": "",
            "disallowed-tools": _DISALLOWED_SUBPROCESS_TOOLS,
        },
        env={
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
        },
    )


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

if hasattr(_parser_mod, "parse_message"):
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
    if hasattr(_client_mod, "parse_message"):
        _client_mod.parse_message = _tolerant_parse_message  # type: ignore[assignment]
    else:
        logger.warning("claude_code_sdk._internal.client.parse_message not found; patch skipped")
else:
    logger.warning(
        "claude_code_sdk._internal.message_parser.parse_message not found; patch skipped"
    )


# ---------------------------------------------------------------------------
# Monkey-patch: raise asyncio StreamReader limit for Codex stdio transport
# ---------------------------------------------------------------------------
# Python's asyncio.StreamReader.readline() has a 64 KiB default limit.
# The Codex app-server echoes input content in its JSON-RPC responses,
# so large prompts (>~65 KB) produce stdout lines that exceed the limit,
# causing LimitOverrunError -> "failed reading from stdio transport".
# Fix: raise the limit to 1 MB when spawning the app-server subprocess.

if hasattr(_codex_transport_mod, "StdioTransport") and hasattr(
    _codex_transport_mod.StdioTransport, "connect"
):
    _original_stdio_connect = _codex_transport_mod.StdioTransport.connect

    async def _stdio_connect_with_large_limit(self: Any) -> None:
        """StdioTransport.connect() with a 1 MB StreamReader limit."""
        if self._proc is not None:
            return
        try:
            self._proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *self._command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                    cwd=self._cwd,
                    env=self._env,
                    limit=1024 * 1024,
                ),
                timeout=self._connect_timeout,
            )
        except Exception as exc:
            from codex_app_server_sdk.errors import CodexTransportError

            raise CodexTransportError(
                f"failed to start stdio transport command: {self._command!r}"
            ) from exc

    _codex_transport_mod.StdioTransport.connect = _stdio_connect_with_large_limit  # type: ignore[assignment]
else:
    logger.warning("codex_app_server_sdk.transport.StdioTransport.connect not found; patch skipped")


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
    mcp_servers: dict[str, Any] = field(default_factory=dict)


@dataclass
class SDKMessage:
    """Unified message envelope for both SDK backends.

    Types:
        assistant: Complete message from the model (text, tool use, thinking blocks).
        result: Final result with usage stats and session info.
        stream: Partial content delta from streaming (text/thinking chunks).
                Only populated to message_buffer, not used for final output.
    """

    type: Literal["assistant", "result", "stream"]
    text: str | None = None
    content_blocks: list[Any] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    session_id: str | None = None


class SDKBackend(Protocol):
    """Protocol that both Claude and Codex backends implement."""

    def query(self, prompt: str, options: SDKOptions) -> AsyncIterator[SDKMessage]: ...

    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Claude Code SDK Backend
# ---------------------------------------------------------------------------


# Minimum characters to accumulate before yielding a stream chunk.
# Avoids per-token buffer noise while keeping summarizer updates timely.
_STREAM_CHUNK_SIZE = 200


def _extract_stream_delta(event: dict[str, Any]) -> tuple[str, str] | None:
    """Extract a text or thinking delta from a raw Anthropic stream event.

    Returns:
        ("text", delta_string) or ("thinking", delta_string), or None if
        the event is not a content delta we care about.
    """
    if event.get("type") != "content_block_delta":
        return None
    delta = event.get("delta", {})
    delta_type = delta.get("type")
    if delta_type == "text_delta":
        text = delta.get("text", "")
        if text:
            return ("text", text)
    elif delta_type == "thinking_delta":
        thinking = delta.get("thinking", "")
        if thinking:
            return ("thinking", thinking)
    return None


class ClaudeBackend:
    """Wraps claude_code_sdk.query() behind the SDKBackend interface."""

    async def close(self) -> None:
        """No-op: Claude Code CLI manages its own session lifecycle."""

    def _build_claude_options(self, options: SDKOptions) -> ClaudeCodeOptions:
        """Convert SDKOptions to ClaudeCodeOptions."""
        env = dict(options.env)
        extra_args = dict(options.extra_args)

        # When reasoning is "off" (no effort key in extra_args), disable
        # extended thinking via env var.  The Claude Code CLI has no
        # --effort off/none flag; omitting --effort lets the model use
        # adaptive thinking by default.  MAX_THINKING_TOKENS=0 is the
        # only way to fully suppress it.
        if "effort" not in extra_args:
            env["MAX_THINKING_TOKENS"] = "0"

        # Isolate SDK subprocesses from the user's personal Claude Code
        # environment (hooks, plugins, CLAUDE.md, memory, settings) while
        # keeping the full tool set (WebSearch, Glob, Grep, Write, etc.).
        isolation = _isolation_config()
        extra_args.update(isolation.extra_args)
        env.update(isolation.env)

        kwargs: dict[str, Any] = {
            "system_prompt": options.system_prompt,
            "allowed_tools": options.allowed_tools,
            "max_turns": options.max_turns,
            "permission_mode": options.permission_mode,
            "extra_args": extra_args,
            "include_partial_messages": True,
        }
        if options.model:
            kwargs["model"] = options.model
        if options.cwd:
            kwargs["cwd"] = str(options.cwd)
        if options.resume:
            kwargs["resume"] = options.resume
        if env:
            kwargs["env"] = env
        if options.mcp_servers:
            kwargs["mcp_servers"] = options.mcp_servers

        return ClaudeCodeOptions(**kwargs)

    async def query(self, prompt: str, options: SDKOptions) -> AsyncIterator[SDKMessage]:
        """Run a Claude Code SDK query, yielding unified SDKMessages.

        With ``include_partial_messages`` enabled, the CLI also emits
        ``StreamEvent`` objects carrying raw Anthropic API content deltas.
        We accumulate text and thinking deltas into ~200-char chunks and
        yield them as ``SDKMessage(type="stream")`` so the summarizer gets
        real-time buffer content instead of sitting idle.
        """
        cc_opts = self._build_claude_options(options)

        logger.debug(
            f"ClaudeBackend query: model={cc_opts.model}, "
            f"max_turns={cc_opts.max_turns}, prompt_len={len(prompt)}"
        )

        # Accumulators for streaming deltas (flushed at chunk boundaries)
        text_acc = ""
        thinking_acc = ""

        async for msg in claude_query(prompt=prompt, options=cc_opts):
            if msg is None:
                continue

            if isinstance(msg, AssistantMessage):
                # Flush any remaining accumulated stream content before
                # yielding the complete message (avoids lost tail).
                if text_acc:
                    block = type("_SyntheticTextBlock", (), {"text": text_acc})()
                    yield SDKMessage(type="stream", content_blocks=[block])
                    text_acc = ""
                if thinking_acc:
                    block = type("_SyntheticThinkingBlock", (), {"thinking": thinking_acc})()
                    yield SDKMessage(type="stream", content_blocks=[block])
                    thinking_acc = ""

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
            elif hasattr(msg, "event"):
                # StreamEvent - extract text/thinking deltas and accumulate
                try:
                    parsed = _extract_stream_delta(msg.event)
                except Exception:
                    continue
                if parsed is None:
                    continue

                kind, delta = parsed
                if kind == "text":
                    text_acc += delta
                    if len(text_acc) >= _STREAM_CHUNK_SIZE:
                        block = type("_SyntheticTextBlock", (), {"text": text_acc})()
                        yield SDKMessage(type="stream", content_blocks=[block])
                        text_acc = ""
                elif kind == "thinking":
                    thinking_acc += delta
                    if len(thinking_acc) >= _STREAM_CHUNK_SIZE:
                        block = type("_SyntheticThinkingBlock", (), {"thinking": thinking_acc})()
                        yield SDKMessage(type="stream", content_blocks=[block])
                        thinking_acc = ""


# ---------------------------------------------------------------------------
# Codex SDK Backend
# ---------------------------------------------------------------------------

# Sandbox mode type for Codex SDK
_CODEX_APPROVAL_POLICY: Any = "never"


class CodexBackend:
    """Wraps codex_app_server_sdk behind the SDKBackend interface.

    Stateful: keeps the Codex app-server client alive between ``query()``
    calls so that threads can be resumed via ``options.resume``.  Call
    ``close()`` when done, or rely on ``__del__`` for temp-dir cleanup.
    """

    def __init__(self) -> None:
        self._client: CodexClient | None = None
        self._codex_home: Path | None = None
        self._sandbox_mode: str = "<unknown>"

    async def close(self) -> None:
        """Shut down the live client and remove the temp config directory."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                logger.debug("Error closing Codex client", exc_info=True)
            self._client = None
        if self._codex_home is not None:
            shutil.rmtree(self._codex_home, ignore_errors=True)
            self._codex_home = None

    def __del__(self) -> None:
        """Safety net: remove the temp directory if close() was never called."""
        if self._codex_home is not None:
            shutil.rmtree(self._codex_home, ignore_errors=True)
            self._codex_home = None

    @staticmethod
    def _resolve_sandbox(
        allowed_tools: list[str] | tuple[str, ...],
        has_mcp: bool = False,
    ) -> str:
        """Map allowed tools to Codex sandbox mode.

        MCP servers run as child subprocesses of the Codex app-server.
        The read-only and workspace-write sandboxes block subprocess
        creation, which prevents MCP servers from starting.  When MCP
        servers are configured we must use danger-full-access; the
        security boundary is maintained by the allowed_tools list, not
        the sandbox.
        """
        if has_mcp:
            logger.debug("Sandbox escalated to danger-full-access for MCP subprocess spawning")
            return "danger-full-access"
        write_tools = {"Write", "Edit", "Bash"}
        if write_tools & set(allowed_tools):
            return "workspace-write"
        return "read-only"

    def _resolve_effort(self, extra_args: dict[str, str | None]) -> str:
        """Map extra_args effort to Codex reasoning effort string.

        Unlike the Claude Code CLI (which defaults to no extended thinking
        when --effort is omitted), the Codex SDK lets the model choose its
        own reasoning level when effort is unset.  For gpt-5.4-mini this
        can mean uncapped reasoning that produces no streaming events,
        triggering the inactivity timeout on large prompts.  We therefore
        default to ``"none"`` so reasoning is always explicitly controlled.
        """
        effort = extra_args.get("effort")
        if effort is None:
            return "none"
        return _CODEX_EFFORT_MAP.get(effort, effort)

    @staticmethod
    def _write_codex_mcp_config(mcp_servers: dict[str, Any], codex_home: Path) -> bool:
        """Write MCP server config to ``$CODEX_HOME/config.toml``.

        Codex reads config from ``$CODEX_HOME/config.toml`` (defaults to
        ``~/.codex/config.toml``).  We write to an isolated temp directory
        so the subprocess sees only the servers we configure, not the
        host machine's global MCP servers, AGENTS.md, skills, or rules.

        Only stdio servers with ``command`` + ``args`` are supported.

        Returns True if any servers were written, False if all were skipped.
        """
        config_path = codex_home / "config.toml"

        lines: list[str] = []
        for name, cfg in mcp_servers.items():
            srv_type = cfg.get("type", "")
            if srv_type != "stdio":
                logger.warning(f"Codex MCP: skipping non-stdio server '{name}' (type={srv_type})")
                continue
            command = cfg.get("command", "")
            args = cfg.get("args", [])
            lines.append(f"[mcp_servers.{name}]")
            lines.append(f'command = "{command}"')
            # Format args as TOML array of strings
            args_str = ", ".join(f'"{a}"' for a in args)
            lines.append(f"args = [{args_str}]")
            lines.append("")

        if lines:
            config_path.write_text("\n".join(lines))
            logger.debug(f"Wrote Codex MCP config to {config_path}")
            return True
        return False

    async def _ensure_client(
        self, options: SDKOptions, model: str | None
    ) -> tuple[CodexClient, dict[str, Any], str]:
        """Return ``(client, chat_kwargs, sandbox_mode)`` for this call.

        On a fresh call (no ``options.resume``), tears down any existing
        client and spins up a new app-server subprocess with an isolated
        ``CODEX_HOME``.

        On a resume call, reuses the existing client so the thread is
        still accessible.  Falls back to a fresh client if the previous
        one was torn down (e.g. after an error).
        """
        effort = self._resolve_effort(options.extra_args)
        turn_overrides = TurnOverrides()
        if effort:
            turn_overrides.effort = effort  # type: ignore[assignment]

        # --- Resume path: reuse existing client ---
        if options.resume:
            if self._client is not None:
                logger.debug(f"Resuming Codex thread {options.resume} on existing client")
                chat_kwargs: dict[str, Any] = {
                    "turn_overrides": turn_overrides,
                    "thread_id": options.resume,
                }
                return self._client, chat_kwargs, self._sandbox_mode

            logger.warning(
                f"Codex resume requested (thread {options.resume}) but no live "
                f"client exists; falling back to a fresh thread"
            )

        # --- Fresh path: close any stale client, create new one ---
        await self.close()

        has_mcp = False
        codex_home = Path(tempfile.mkdtemp(prefix="codex_home_"))
        self._codex_home = codex_home

        # Copy auth.json so ChatGPT subscription auth works in isolation.
        real_auth = Path.home() / ".codex" / "auth.json"
        if real_auth.exists():
            shutil.copy2(real_auth, codex_home / "auth.json")

        # Write MCP server config to the isolated home (not cwd).
        if options.mcp_servers:
            has_mcp = self._write_codex_mcp_config(options.mcp_servers, codex_home)

        sandbox_mode = self._resolve_sandbox(options.allowed_tools, has_mcp=has_mcp)
        self._sandbox_mode = sandbox_mode

        # Build environment for the Codex subprocess.
        # IMPORTANT: create_subprocess_exec replaces the entire env when a
        # dict is passed, so we must start from os.environ and layer overrides.
        env: dict[str, str] = {
            **os.environ,
            **options.env,
            "CODEX_HOME": str(codex_home),
        }
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

        logger.debug(
            f"CodexBackend new client: model={model}, sandbox={sandbox_mode}, effort={effort}"
        )

        codex_cwd = Path(options.cwd) if options.cwd else Path.cwd()
        client = CodexClient.connect_stdio(
            cwd=str(codex_cwd),
            env=env,
            inactivity_timeout=600.0,
        )
        try:
            await client.start()
        except Exception:
            try:
                await client.close()
            except Exception:
                logger.debug("Error closing half-started Codex client", exc_info=True)
            raise
        self._client = client

        chat_kwargs = {
            "turn_overrides": turn_overrides,
            "thread_config": thread_config,
        }
        return client, chat_kwargs, sandbox_mode

    async def query(self, prompt: str, options: SDKOptions) -> AsyncIterator[SDKMessage]:
        """Run a Codex SDK query, yielding unified SDKMessages.

        The client is kept alive after a successful call so that a
        subsequent call with ``options.resume`` can continue the same
        thread.  On error the client is torn down.
        """
        model = options.model

        # Apply model overrides at query time (not globally) to avoid
        # mutating shared config that gets persisted to disk.
        replacement = CODEX_MODEL_OVERRIDES.get(model) if model else None
        if replacement:
            logger.warning(
                f"Model {model!r} is unsupported by Codex with ChatGPT subscription. "
                f"Using {replacement!r} instead. "
                f"(https://github.com/openai/codex/issues/14266)"
            )
            model = replacement

        sandbox_mode = self._sandbox_mode
        try:
            client, chat_kwargs, sandbox_mode = await self._ensure_client(options, model)

            final_text_parts: list[str] = []
            thread_id: str | None = None
            step_count = 0

            async for step in client.chat(prompt, **chat_kwargs):
                thread_id = step.thread_id
                step_count += 1
                if step.text:
                    final_text_parts.append(step.text)
                # Always yield a message so the message_buffer gets
                # populated during the turn (enables progress summaries).
                # Steps without text (thinking, exec) use step_type as
                # fallback so the summarizer sees activity.
                display_text = step.text or f"[{step.step_type}]"
                synthetic_block = type("_SyntheticTextBlock", (), {"text": display_text})()
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
            await self.close()
            raise RuntimeError(
                f"Codex query failed (model={model}, sandbox={sandbox_mode}): {e}\n"
                f"If using ChatGPT subscription, note that gpt-5.4-nano is not supported "
                f"by Codex. Use gpt-5.4-mini or higher."
            ) from e
        except Exception as e:
            await self.close()
            raise RuntimeError(
                f"Codex query failed (model={model}, sandbox={sandbox_mode}): {e}"
            ) from e


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
