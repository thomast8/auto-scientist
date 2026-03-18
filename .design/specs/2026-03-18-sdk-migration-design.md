# SDK Migration: `claude_agent_sdk` -> `claude_code_sdk`

**Date:** 2026-03-18
**Status:** Approved

## Problem

The codebase references `claude_agent_sdk`, which does not exist as an installable package. The real SDK is `claude_code_sdk` (already in `pyproject.toml` dependencies). Five agent files import from the non-existent package, and tests mock it at the `sys.modules` level to avoid import errors.

The ingestor agent already uses `claude_code_sdk` correctly, confirming the target API.

## Changes

### 1. Import Rename (5 files)

Replace `from claude_agent_sdk import ...` with `from claude_code_sdk import ...` and rename `ClaudeAgentOptions` to `ClaudeCodeOptions` in:

- `src/auto_scientist/agents/analyst.py`
- `src/auto_scientist/agents/scientist.py`
- `src/auto_scientist/agents/coder.py`
- `src/auto_scientist/agents/report.py`
- `src/auto_scientist/agents/discovery.py` (also imports `ClaudeSDKClient`, which exists in `claude_code_sdk` with the same API)

### 2. Remove `output_format` Parameter (3 call sites)

`claude_code_sdk.ClaudeCodeOptions` does not support `output_format`. The three call sites that use it:

- `analyst.py` line 136
- `scientist.py` line 113
- `scientist.py` line 202

**Workaround:** Append a JSON output instruction block to the system prompt at each call site. The instruction embeds the JSON schema and tells the agent to respond with raw JSON only. The existing fence-stripping and `json.loads()` parsing in each agent already handles responses correctly.

Example suffix appended to the system prompt:

```
## Output Format
You MUST respond with ONLY valid JSON matching the schema below.
No markdown fencing. No explanation. No other text.

Schema:
{schema_json}
```

### 3. Test Mock Update (`tests/conftest.py`)

- Change `sys.modules` key from `"claude_agent_sdk"` to `"claude_code_sdk"`
- Rename mock attribute `ClaudeAgentOptions` to `ClaudeCodeOptions`
- Update the comment text from "claude_agent_sdk" to "claude_code_sdk"

### 4. Docstring Cleanup

Fix stale references to `claude_agent_sdk` or `ClaudeAgentOptions` in module docstrings (e.g., discovery.py line 2).

## Out of Scope

- `critic.py`: uses raw API clients (OpenAI/Google/Anthropic), no SDK dependency.
- `ingestor.py`: already uses `claude_code_sdk` correctly.
- Historical plan/design docs: not updated (they describe past decisions).

## Verification

- `uv run ruff check src/ tests/` passes
- `uv run pytest` passes (all existing tests use the mock, which will be updated)
- Imports resolve correctly: `uv run python -c "from auto_scientist.agents.analyst import run_analyst"`
