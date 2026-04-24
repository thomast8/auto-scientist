"""SDK backend abstraction for multi-provider agentic support.

Provides a unified interface for both Claude Code SDK (Anthropic) and
Codex SDK (OpenAI), allowing agents to run on either provider without
changing their code.

Monkey-patches the Claude Code SDK message parser at import time so that
unknown message types are silently skipped instead of crashing the stream.
"""

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import claude_code_sdk._internal.client as _client_mod
import claude_code_sdk._internal.message_parser as _parser_mod
import codex_app_server_sdk.transport as _codex_transport_mod
from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    HookContext,
    HookMatcher,
    ResultMessage,
)
from claude_code_sdk import query as claude_query
from claude_code_sdk._errors import MessageParseError
from claude_code_sdk.types import HookJSONOutput
from codex_app_server_sdk import CodexClient, ThreadConfig, TurnOverrides
from codex_app_server_sdk.errors import CodexProtocolError
from pydantic import BaseModel

from auto_core.cost_ceiling import record_cost
from auto_core.safety.tool_guard import PreToolUseHook

logger = logging.getLogger(__name__)

_CLAUDE_QUERY_MESSAGE_TIMEOUT_ENV = "CLAUDE_QUERY_MESSAGE_TIMEOUT_SECONDS"
_CLAUDE_QUERY_MESSAGE_TIMEOUT_DEFAULT = 900.0


def _resolve_message_timeout() -> float:
    raw = os.environ.get(_CLAUDE_QUERY_MESSAGE_TIMEOUT_ENV)
    if raw is None:
        return _CLAUDE_QUERY_MESSAGE_TIMEOUT_DEFAULT
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"{_CLAUDE_QUERY_MESSAGE_TIMEOUT_ENV}={raw!r} is not a valid float"
        ) from exc


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

Dependencies are installed automatically by the run command. You do NOT need
to run `pip install` manually. Just declare all third-party packages in the
PEP 723 metadata block and use the exact run command from the task
instructions. The framework will install everything before executing the
script.

IMPORTANT: Every time you edit the script to add a new import, you MUST also
add the package to the PEP 723 dependencies block. Do NOT remove imports to
work around installation failures; the framework handles installation.
</sandbox_environment>
"""

# Additional sandbox policy shown only to agents that run under the
# reviewer's workspace guard. The Codex seatbelt already blocks writes
# outside cwd; this blurb keeps the model's model-of-the-world accurate.
CODEX_REVIEWER_POLICY_ADDENDUM = """\

<reviewer_sandbox_policy>
The seatbelt confines every write to the review workspace. The following
are kernel-blocked and will fail with EPERM, regardless of what you try:

- Writes outside the workspace (including the user's real repository
  elsewhere on disk). Use `Read` / `Glob` / `Grep` if you need to
  inspect it.
- Subprocess attempts to mutate directories outside the workspace.

The policy additionally rejects these operations even when they would
resolve inside the workspace — mirroring the Claude-backend guard so
behaviour is identical across providers:

- `rm -r` / `rm -rf` anywhere (clean up via the orchestrator, not via
  Bash).
- `sudo`, `chmod`, `chown`, `dd`, `mkfs`, `systemctl`, `launchctl`.
- `git push`, `git commit`, `git reset --hard`, `git clean`,
  `git rebase`, `git checkout`, `git branch`, `git remote`.
- `gh pr merge/close/edit`, `gh issue *`, `gh repo *`, `gh api -X POST`.

Probes must run the reviewer's script from the workspace clone. Do not
`cd` to any path that isn't under the workspace.
</reviewer_sandbox_policy>
"""

# Tools that should never be available to SDK subprocesses.
# Agent/Skill could recurse into host plugins; we block them explicitly.
_DISALLOWED_SUBPROCESS_TOOLS = "Agent,Skill"


def rewrite_uv_run_for_codex(run_command: str) -> str:
    """Rewrite a ``uv run ...`` command so it executes inside the Codex sandbox.

    ``uv`` panics inside the macOS seatbelt sandbox because the Rust
    system-configuration crate cannot reach SCDynamicStore, so any
    ``uv run`` invocation has to be rewritten to use the host ``python3``
    directly. The patterns Intake / DomainConfig produce:

    - ``uv run {script_path}``  -> ``python3 {script_path}``
    - ``uv run foo.py [args]``  -> ``python3 foo.py [args]``
    - ``uv run python [args]``  -> ``python3 [args]``
    - ``uv run <tool> [args]``  -> ``python3 -m <tool> [args]``  (e.g. pytest)

    Anything that does not start with ``uv run `` is returned unchanged,
    so ``node {script_path}`` etc pass through. Only the leading ``uv run``
    is rewritten; the caller is responsible for not chaining further
    ``uv`` invocations, because ``uv`` is unavailable in the sandbox.
    """
    prefix = "uv run "
    if not run_command.startswith(prefix):
        return run_command

    rest = run_command[len(prefix) :]
    tokens = rest.split(None, 1)
    if not tokens:
        return run_command

    first = tokens[0]
    remainder = tokens[1] if len(tokens) > 1 else ""
    tail = f" {remainder}" if remainder else ""

    if first in {"python", "python3"}:
        return f"python3{tail}"
    if first == "{script_path}" or first.endswith(".py"):
        return f"python3 {rest}"
    # Any other first token is a CLI tool installed by uv; run it as a
    # module so python3 can locate it inside the sandbox (e.g. pytest).
    return f"python3 -m {first}{tail}"


@dataclass(frozen=True)
class _IsolationConfig:
    """CLI extra_args and env vars for subprocess isolation."""

    extra_args: dict[str, str | None]
    env: dict[str, str]


def _isolation_config() -> _IsolationConfig:
    """Build isolation settings for SDK subprocesses.

    This helper is **Claude-Code-specific**. It returns CLI flags + env
    vars that only the ``claude`` binary reads, so it is wired into
    :meth:`ClaudeBackend._build_claude_options` but must *not* be
    applied to Codex. The Codex backend isolates through a separate
    mechanism - see :meth:`CodexBackend._ensure_client`, which
    allocates a fresh ``$CODEX_HOME`` per run so the Codex
    subprocess never sees the host's ``~/.codex/config.toml``,
    AGENTS.md, skills, or global MCP servers. Any future parity fix
    (e.g. blocking specific Codex built-in tools) belongs there, not
    here.

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
# Session cleanup: track Claude Code sessions for post-run deletion
# ---------------------------------------------------------------------------

# Registry of (session_id, encoded_cwd) pairs created during this process.
_claude_session_registry: list[tuple[str, str]] = []


def _encode_cwd(cwd: Path) -> str:
    """Encode a path the way Claude Code does for project directories."""
    return re.sub(r"[^a-zA-Z0-9]", "-", str(cwd))


def _register_session(session_id: str, cwd: Path) -> None:
    """Track a Claude Code session for cleanup on exit."""
    encoded = _encode_cwd(cwd)
    _claude_session_registry.append((session_id, encoded))
    logger.debug(f"Registered session {session_id} (cwd={encoded})")


def cleanup_sessions() -> int:
    """Delete all tracked Claude Code session files from ~/.claude/projects/.

    Returns the number of sessions cleaned up.
    """
    projects_dir = Path.home() / ".claude" / "projects"
    cleaned = 0
    for session_id, encoded_cwd in _claude_session_registry:
        session_dir = projects_dir / encoded_cwd
        for suffix in (f"{session_id}.jsonl", session_id):
            path = session_dir / suffix
            if path.is_file():
                path.unlink()
                cleaned += 1
                logger.debug(f"Cleaned up session file: {path}")
            elif path.is_dir():
                shutil.rmtree(path)
                cleaned += 1
                logger.debug(f"Cleaned up session dir: {path}")
    count = len(_claude_session_registry)
    _claude_session_registry.clear()
    if count:
        logger.info(f"Cleaned up {cleaned} session artifacts from {count} agent sessions")
    return cleaned


# ---------------------------------------------------------------------------
# Unified types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SDKOptions:
    """Unified options for both Claude Code SDK and Codex SDK.

    ``pre_tool_use_hook``: optional workspace guard (see
    :mod:`auto_core.safety.tool_guard`). When set:
      - Claude backend wires it to ``ClaudeCodeOptions.can_use_tool``;
        ``permission_mode`` is forced to ``"default"`` because
        ``"acceptEdits"`` silently auto-approves edit/write tools and
        bypasses the callback.
      - Codex backend cannot call back into Python per-tool; instead it
        asserts that ``cwd`` matches the guard's workspace so the
        macOS-seatbelt / Linux-namespace ``workspace-write`` sandbox
        enforces the boundary at the kernel level.
    """

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
    network_access: bool = False
    response_schema: type[BaseModel] | None = None
    pre_tool_use_hook: PreToolUseHook | None = None


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


async def _single_user_message_stream(
    prompt: str,
    shutdown: asyncio.Event,
) -> AsyncIterator[dict[str, Any]]:
    """Yield a single user-turn message, stay open until ``shutdown`` is set.

    Streaming mode is required for both `can_use_tool` and `hooks` to
    actually wire up (see SDK's `Query.initialize_if_needed`). More
    subtly, the hook-dispatch channel is shared with the input stream:
    if this generator returns too early, the SDK's JS side logs
    ``error: Stream closed`` every time it tries to send a hook
    request back, and the hook silently does nothing (tools run
    unguarded).

    So we yield the one user message and block on ``shutdown`` until
    the outer consumer has drained the result message and is ready
    to close the input side. The SDK's prompt-reader task then sees
    StopAsyncIteration and terminates cleanly.
    """
    yield {
        "type": "user",
        "message": {"role": "user", "content": prompt},
        "parent_tool_use_id": None,
        "session_id": "default",
    }
    await shutdown.wait()


def _make_claude_pretooluse_hook(
    hook: PreToolUseHook,
) -> Callable[
    [dict[str, Any], str | None, HookContext],
    Awaitable[HookJSONOutput],
]:
    """Adapt a workspace guard to the Claude SDK's PreToolUse hook.

    Unlike `can_use_tool` (which is bypassed for tools in
    `allowed_tools`), PreToolUse runs for every tool invocation. The
    hook input dict is shaped per
    https://docs.anthropic.com/en/docs/claude-code/hooks#hook-input —
    we pull `tool_name` and `tool_input` from it and call the guard.

    Return shape per Anthropic docs: ``hookSpecificOutput`` with
    ``permissionDecision`` set to ``"deny"`` (plus a
    ``permissionDecisionReason``) on denial; an empty dict on allow.
    """

    async def pre_tool_use(
        input_: dict[str, Any],
        tool_use_id: str | None,  # noqa: ARG001 — SDK signature
        context: HookContext,  # noqa: ARG001 — SDK signature
    ) -> HookJSONOutput:
        tool_name = input_.get("tool_name", "")
        tool_input = input_.get("tool_input", {}) or {}
        decision = hook(tool_name, tool_input)
        if decision.allowed:
            logger.debug("guard allow: %s", tool_name)
            return {}
        tool_summary = _summarise_tool_input(tool_name, tool_input)
        logger.info(
            "guard deny: tool=%s input=%s reason=%s",
            tool_name,
            tool_summary,
            decision.reason,
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": decision.reason,
            }
        }

    return pre_tool_use


def _summarise_tool_input(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Short deterministic summary of a tool call for log context."""
    if tool_name == "Bash":
        cmd = str(tool_input.get("command", ""))
        return f"command={cmd[:200]!r}"
    for key in ("file_path", "path", "notebook_path"):
        if key in tool_input:
            return f"{key}={tool_input[key]!r}"
    return str(tool_input)[:200]


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

        # Strip ANTHROPIC_API_KEY so Claude Code CLI uses OAuth instead
        # of direct API billing (the key may leak in via .env / dotenv).
        if "ANTHROPIC_API_KEY" not in options.env and os.environ.get("ANTHROPIC_API_KEY"):
            logger.info(
                "Stripping ANTHROPIC_API_KEY from Claude Code subprocess env "
                "(using OAuth instead of direct API billing)"
            )
            env["ANTHROPIC_API_KEY"] = ""

        # A pre_tool_use_hook is the reviewer's workspace guard. Wire it
        # via `hooks={"PreToolUse": ...}` rather than `can_use_tool`: the
        # hook fires for *every* tool invocation, whereas `can_use_tool`
        # is bypassed whenever the tool is already in `allowed_tools`.
        # For a guard whose whole point is to catch escapes, the hook
        # path is the right primitive — `allowed_tools` is about
        # ergonomics, not security.
        hooks = None
        if options.pre_tool_use_hook is not None:
            hooks = {
                "PreToolUse": [
                    HookMatcher(
                        matcher="*",
                        hooks=[_make_claude_pretooluse_hook(options.pre_tool_use_hook)],
                    )
                ]
            }

        kwargs: dict[str, Any] = {
            "system_prompt": options.system_prompt,
            "allowed_tools": options.allowed_tools,
            "max_turns": options.max_turns,
            "permission_mode": options.permission_mode,
            "extra_args": extra_args,
            "include_partial_messages": True,
        }
        if hooks is not None:
            kwargs["hooks"] = hooks
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

        if options.response_schema is not None:
            schema_str = json.dumps(options.response_schema.model_json_schema(), indent=2)
            kwargs["append_system_prompt"] = (
                "\n\nMANDATORY OUTPUT FORMAT: Your final response must be ONLY "
                "valid JSON matching this exact schema. No markdown fencing "
                "(```), no prose before or after the JSON, no trailing text. "
                "Output the raw JSON object and nothing else.\n\n"
                f"Schema:\n{schema_str}"
            )

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

        # The SDK only initialises the hook IPC in streaming mode (see
        # claude_code_sdk/_internal/query.py initialize_if_needed: hooks
        # config is skipped when `is_streaming_mode=False`). Additionally,
        # hooks share the input stream for their control-channel IPC —
        # closing the stream too early makes every hook fail with
        # "Stream closed" JS-side (silent unguarded tool calls). So we
        # keep a long-lived stream and close it only after the final
        # ResultMessage arrives (see the shutdown_prompt_stream set
        # below). Non-guard callers keep the cheap string path.
        sdk_prompt: str | AsyncIterable[dict[str, Any]]
        shutdown_prompt_stream: asyncio.Event | None = None
        if options.pre_tool_use_hook is not None:
            shutdown_prompt_stream = asyncio.Event()
            sdk_prompt = _single_user_message_stream(prompt, shutdown_prompt_stream)
        else:
            sdk_prompt = prompt

        message_timeout = _resolve_message_timeout()
        query_iter = claude_query(prompt=sdk_prompt, options=cc_opts).__aiter__()
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(query_iter.__anext__(), timeout=message_timeout)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    logger.error(
                        f"Claude CLI stalled for {message_timeout:.0f}s between messages; "
                        f"aborting query (set {_CLAUDE_QUERY_MESSAGE_TIMEOUT_ENV} to adjust)"
                    )
                    raise
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
                    cost = getattr(msg, "total_cost_usd", None)
                    usage["total_cost_usd"] = cost
                    record_cost(cost)
                    sid = getattr(msg, "session_id", None)
                    if sid:
                        effective_cwd = Path(options.cwd) if options.cwd else Path.cwd()
                        _register_session(sid, effective_cwd)
                    # Final message — safe to close the prompt stream now
                    # that no more hook control-requests can arrive. The
                    # SDK's prompt-reader task sees StopAsyncIteration and
                    # the query terminates cleanly instead of hanging.
                    if shutdown_prompt_stream is not None:
                        shutdown_prompt_stream.set()
                    yield SDKMessage(
                        type="result",
                        result=msg.result if msg.result else None,
                        usage=usage,
                        session_id=sid,
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
                            block = type(
                                "_SyntheticThinkingBlock",
                                (),
                                {"thinking": thinking_acc},
                            )()
                            yield SDKMessage(type="stream", content_blocks=[block])
                            thinking_acc = ""
        finally:
            # Ensure the prompt stream is closed on any exit path
            # (error, timeout, caller breaking out of the outer loop).
            if shutdown_prompt_stream is not None:
                shutdown_prompt_stream.set()


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
        network_access: bool = False,
    ) -> str:
        """Map allowed tools to Codex sandbox mode.

        MCP servers run as child subprocesses of the Codex app-server.
        The read-only and workspace-write sandboxes block subprocess
        creation, which prevents MCP servers from starting.  When MCP
        servers are configured we must escalate the sandbox.

        Only ``network_access=True`` (coder needing pip) escalates all the
        way to ``danger-full-access``.  MCP-only agents (analyst, scientist,
        critics) escalate to ``workspace-write`` which allows subprocess
        creation without granting unrestricted filesystem access.
        """
        if network_access:
            logger.debug("Sandbox escalated to danger-full-access for network access (pip install)")
            return "danger-full-access"
        if has_mcp:
            logger.debug("Sandbox escalated to workspace-write for MCP subprocess spawning")
            return "workspace-write"
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
    def _resolve_disabled_features(
        allowed_tools: list[str] | tuple[str, ...],
    ) -> list[str]:
        """Determine which Codex features to disable based on allowed tools.

        Codex exposes 16+ built-in tools by default.  For agents that only
        need web search and MCP tools, the extra tools (shell, agents,
        tool_suggest) dilute the model's attention and reduce MCP usage.
        """
        disabled: list[str] = []
        allowed = set(allowed_tools)
        shell_tools = {"Write", "Edit", "Bash", "Read", "Glob", "Grep"}
        if not (shell_tools & allowed):
            disabled.append("shell_tool")
            disabled.append("unified_exec")
        # Always disable: no auto-scientist agent needs sub-agents or tool discovery.
        disabled.append("multi_agent")
        disabled.append("tool_suggest")
        return disabled

    @staticmethod
    def _write_codex_home_config(
        codex_home: Path,
        mcp_servers: dict[str, Any] | None = None,
        disabled_features: list[str] | None = None,
    ) -> bool:
        """Write ``$CODEX_HOME/config.toml`` with MCP servers and feature flags.

        Codex reads config from ``$CODEX_HOME/config.toml`` (defaults to
        ``~/.codex/config.toml``).  We write to an isolated temp directory
        so the subprocess sees only the servers we configure, not the
        host machine's global MCP servers, AGENTS.md, skills, or rules.

        Only stdio servers with ``command`` + ``args`` are supported.

        Returns True if any MCP servers were written, False otherwise.
        """
        config_path = codex_home / "config.toml"

        lines: list[str] = []
        has_mcp = False

        # Feature flags
        if disabled_features:
            lines.append("[features]")
            for feat in disabled_features:
                lines.append(f"{feat} = false")
            lines.append("")

        # MCP servers
        if mcp_servers:
            for name, cfg in mcp_servers.items():
                srv_type = cfg.get("type", "")
                if srv_type != "stdio":
                    logger.warning(
                        f"Codex MCP: skipping non-stdio server '{name}' (type={srv_type})"
                    )
                    continue
                command = cfg.get("command", "")
                args = cfg.get("args", [])
                lines.append(f"[mcp_servers.{name}]")
                lines.append(f'command = "{command}"')
                # Format args as TOML array of strings
                args_str = ", ".join(f'"{a}"' for a in args)
                lines.append(f"args = [{args_str}]")
                lines.append("")
                has_mcp = True

        if lines:
            config_path.write_text("\n".join(lines))
            logger.debug(f"Wrote Codex config to {config_path}")

        return has_mcp

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
        # Codex has no per-tool Python callback; the workspace-write
        # sandbox is the enforcement layer and its boundary is the
        # client's cwd. If the caller passed a workspace guard, verify
        # options.cwd resolves to the same directory so the kernel-level
        # seatbelt matches what the guard thinks it's protecting. A
        # mismatch here would silently put the model outside the guard's
        # jurisdiction — the one case where "fail fast" is strictly
        # better than trying to recover.
        if options.pre_tool_use_hook is not None:
            if options.cwd is None:
                raise RuntimeError(
                    "SDKOptions.pre_tool_use_hook is set but options.cwd "
                    "is None; the Codex seatbelt needs an explicit cwd "
                    "to pin its workspace-write sandbox to."
                )
            cwd_resolved = Path(options.cwd).resolve()
            guard_workspace = options.pre_tool_use_hook.workspace
            if cwd_resolved != guard_workspace:
                raise RuntimeError(
                    f"Codex cwd {cwd_resolved} does not match the workspace "
                    f"guard's workspace {guard_workspace}. The seatbelt "
                    "would confine writes to a different directory than "
                    "the guard expects; refusing to proceed."
                )

        effort = self._resolve_effort(options.extra_args)
        turn_overrides = TurnOverrides()
        if effort:
            turn_overrides.effort = effort  # type: ignore[assignment]

        # Structured output: set output_schema on the turn overrides so
        # Codex mechanically constrains the model to valid JSON.
        if options.response_schema is not None:
            from auto_core.sdk_utils import make_strict_schema

            strict = make_strict_schema(options.response_schema.model_json_schema())
            turn_overrides.output_schema = strict

        # --- Resume path: reuse existing client ---
        is_resume_fallback = False
        if options.resume:
            if self._client is not None:
                logger.debug(f"Resuming Codex thread {options.resume} on existing client")
                chat_kwargs: dict[str, Any] = {
                    "turn_overrides": turn_overrides,
                    "thread_id": options.resume,
                }
                return self._client, chat_kwargs, self._sandbox_mode

            # The caller asked to continue a specific thread, but we no
            # longer have a live client for it (usually because a prior
            # turn raised and query() tore the client down). There is no
            # way to resume from scratch, so we start a fresh thread and
            # prior conversation context is lost. Log loudly and make this
            # fallback thread VISIBLE in the Codex sidebar (ephemeral=False
            # below) so the operator has an obvious signal that the
            # correction loop silently restarted.
            logger.error(
                f"Codex resume requested (thread {options.resume}) but no "
                f"live client exists; falling back to a FRESH thread. Prior "
                f"conversation context is LOST. The fallback thread will be "
                f"visible in the Codex sidebar as a debugging signal."
            )
            is_resume_fallback = True

        # --- Fresh path: close any stale client, create new one ---
        await self.close()

        has_mcp = False
        codex_home = Path(tempfile.mkdtemp(prefix="codex_home_"))
        self._codex_home = codex_home

        # Copy auth.json so ChatGPT subscription auth works in isolation.
        real_auth = Path.home() / ".codex" / "auth.json"
        if real_auth.exists():
            shutil.copy2(real_auth, codex_home / "auth.json")

        # Write config.toml with MCP servers and feature flags.
        disabled_features = self._resolve_disabled_features(options.allowed_tools)
        has_mcp = self._write_codex_home_config(
            codex_home,
            mcp_servers=options.mcp_servers or None,
            disabled_features=disabled_features,
        )

        sandbox_mode = self._resolve_sandbox(
            options.allowed_tools,
            has_mcp=has_mcp,
            network_access=options.network_access,
        )
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

        # Build thread config.
        #
        # ephemeral is a server-side property set once at thread/start and
        # never re-specified on subsequent turns. The resume path above
        # reuses the same live client and passes only thread_id into
        # client.chat(), so corrections from the Coder's error loop
        # continue on the same (ephemeral) thread without rebuilding this
        # config.
        #
        # We default ephemeral=True to hide auto-scientist agent chatter
        # from the Codex desktop app's history sidebar. A pipeline run
        # fires dozens of agent calls and each one would otherwise show up
        # as its own persistent thread. The one exception is the
        # resume-fallback path: if the caller asked to continue a thread
        # but the live client is gone, we have already lost prior context,
        # and the fresh thread we're about to start is created as VISIBLE
        # (ephemeral=False) so the operator has a sidebar signal that the
        # correction loop silently restarted.
        thread_config = ThreadConfig(
            model=model,
            base_instructions=options.system_prompt,
            sandbox=sandbox_mode,  # type: ignore[arg-type]
            approval_policy=_CODEX_APPROVAL_POLICY,
            ephemeral=not is_resume_fallback,
        )
        if options.cwd:
            thread_config.cwd = str(options.cwd)

        # developer_instructions carry higher priority than base_instructions
        # for behavioral directives in Codex.
        dev_instructions: list[str] = []

        # NOTE: An earlier MCP-call mandate lived here. It used "you MUST call
        # at least one of these tools, otherwise your output will be rejected"
        # framing to push GPT models toward MCP usage on long prompts. In
        # practice this caused Codex agents to fabricate "rejection" responses
        # and fall back to shell commands when their tool-call serialization
        # tripped (observed live in the Report agent). Use prompt-level tool
        # guidance in the per-agent system prompts instead, not a global
        # threat-shaped mandate. If MCP usage drops on the GPT path we will
        # re-introduce a positive nudge here, not a coercive one.

        # Structured output behavioral reinforcement (complements the
        # mechanical output_schema on TurnOverrides).
        if options.response_schema is not None:
            dev_instructions.append(
                "MANDATORY OUTPUT FORMAT: Your final output MUST be valid JSON "
                "matching the structured output schema. No markdown fencing, "
                "no prose before or after the JSON. Output ONLY the raw JSON object."
            )

        if dev_instructions:
            thread_config.developer_instructions = "\n\n".join(dev_instructions)

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

# Cache of backends by provider, so repeated calls reuse the same instance
# instead of accumulating temp dirs and subprocess handles.
_backend_cache: dict[str, SDKBackend] = {}


def get_backend(provider: str) -> SDKBackend:
    """Return a cached SDK backend for the given provider.

    Creates a new instance on first call for each provider, then reuses it.
    All backends are closed at shutdown via ``close_all_backends()``.
    """
    cached = _backend_cache.get(provider)
    if cached is not None:
        return cached

    if provider == "anthropic":
        backend: SDKBackend = ClaudeBackend()
    elif provider == "openai":
        backend = CodexBackend()
    else:
        raise ValueError(
            f"No SDK backend for provider {provider!r}. SDK mode requires 'anthropic' or 'openai'."
        )
    _backend_cache[provider] = backend
    return backend


async def close_all_backends() -> None:
    """Close every backend created via ``get_backend()`` and clear the cache.

    Called at shutdown to properly terminate subprocess transports before
    the event loop closes, preventing asyncio child-watcher warnings.
    """
    for backend in _backend_cache.values():
        try:
            await backend.close()
        except Exception:
            logger.debug("Error closing backend %s", type(backend).__name__, exc_info=True)
    _backend_cache.clear()


@asynccontextmanager
async def create_backend(provider: str) -> AsyncIterator[SDKBackend]:
    """Yield a fresh, non-cached SDK backend instance.

    Unlike :func:`get_backend` which returns a cached singleton, this creates
    a new instance each time.  The backend is closed automatically when the
    context manager exits.

    Use this for any bounded scope that needs private backend state:
    concurrent callers (parallel critics, stop-gate debates) that must not
    share a ``CodexBackend`` subprocess, and retryable conversations where
    the backend must stay alive across validation retries for session resume.

    :func:`get_backend` is appropriate only for long-lived sequential agents
    (analyst, scientist, coder) where a single shared instance with cross-call
    resume is desired and no concurrent access occurs.
    """
    if provider == "anthropic":
        backend: SDKBackend = ClaudeBackend()
    elif provider == "openai":
        backend = CodexBackend()
    else:
        raise ValueError(
            f"No SDK backend for provider {provider!r}. SDK mode requires 'anthropic' or 'openai'."
        )
    try:
        yield backend
    finally:
        await backend.close()
