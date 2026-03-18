# Data Ingestion Agent Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Ingestor agent as the first pipeline phase that canonicalizes raw user data (any local file/directory) into a clean dataset, with optional human-in-the-loop conversation.

**Architecture:** New `"ingestion"` phase before `"discovery"` in the state machine. The Ingestor uses `ClaudeSDKClient` (like Discovery) with Bash tools to inspect and convert data. When `interactive=True`, it uses `AskUserQuestion` for multi-round clarification with the human. Output goes to `experiments/data/`.

**Tech Stack:** Python, Pydantic (state model), `claude_agent_sdk` (`ClaudeSDKClient`, `AskUserQuestion`), Click (CLI)

**Spec:** `docs/superpowers/specs/2026-03-17-data-ingestion-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/auto_scientist/state.py` | Add `raw_data_path` field, change phase default to `"ingestion"` |
| Create | `src/auto_scientist/prompts/ingestor.py` | System + user prompt templates for Ingestor |
| Create | `src/auto_scientist/agents/ingestor.py` | `run_ingestor()` using `ClaudeSDKClient` |
| Modify | `src/auto_scientist/orchestrator.py` | Add ingestion phase dispatch, update data paths after ingestion |
| Modify | `src/auto_scientist/cli.py` | Change initial phase to `"ingestion"` |
| Create | `tests/test_ingestor.py` | Tests for state changes and ingestor module |

---

### Task 1: State Model Changes

**Files:**
- Modify: `src/auto_scientist/state.py:21-35`
- Modify: `tests/test_state.py`

- [ ] **Step 1: Write failing tests for new state fields**

Add the following methods inside the existing `TestExperimentState` class in `tests/test_state.py`:

```python
def test_raw_data_path_field(self):
    state = ExperimentState(domain="test", goal="g", raw_data_path="/raw/data.csv")
    assert state.raw_data_path == "/raw/data.csv"

def test_raw_data_path_defaults_to_none(self):
    state = ExperimentState(domain="test", goal="g")
    assert state.raw_data_path is None

def test_default_phase_is_ingestion(self):
    state = ExperimentState(domain="test", goal="g")
    assert state.phase == "ingestion"

def test_raw_data_path_roundtrip(self, tmp_state_path):
    state = ExperimentState(domain="test", goal="g", raw_data_path="/raw/data.csv")
    state.save(tmp_state_path)
    loaded = ExperimentState.load(tmp_state_path)
    assert loaded.raw_data_path == "/raw/data.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py::TestExperimentState::test_raw_data_path_field tests/test_state.py::TestExperimentState::test_default_phase_is_ingestion -v`
Expected: FAIL (no `raw_data_path` field, phase defaults to `"discovery"`)

- [ ] **Step 3: Implement state changes**

In `src/auto_scientist/state.py`, in the `ExperimentState` class:
- Change `phase` default from `"discovery"` to `"ingestion"` and update the comment: `# ingestion, discovery, iteration, report, stopped`
- Add `raw_data_path: str | None = None` field after `data_path`

- [ ] **Step 4: Run full state test suite**

Run: `uv run pytest tests/test_state.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```
feat: add raw_data_path to ExperimentState and change default phase to ingestion
```

---

### Task 2: Ingestor Prompt Templates

**Files:**
- Create: `src/auto_scientist/prompts/ingestor.py`

- [ ] **Step 1: Create prompt file**

Create `src/auto_scientist/prompts/ingestor.py` with two templates: `INGESTOR_SYSTEM` and `INGESTOR_USER`.

The system prompt should instruct the agent to:
- Inspect the raw data at the given path (file or directory)
- Detect format, schema, column types, row counts
- For large files, sample first to understand structure before full conversion
- Choose the best canonical format based on heuristics (SQLite for multi-table/relational, CSV for simple flat tables, Parquet for large single-table datasets)
- When in interactive mode: ask the human about anything ambiguous (column semantics, table relationships, units, encodings) via `AskUserQuestion`
- When in autonomous mode: make best-effort decisions and log assumptions
- Write a conversion script to `{data_dir}/ingest.py`
- Run the script to produce the canonical output in `{data_dir}/`
- Show the human a summary of what was produced for final approval (interactive) or log it (autonomous)
- Create the lab notebook file if it doesn't exist, and append ingestion findings/assumptions to it
- NEVER modify the original data files

The user prompt should be formatted with: `{raw_data_path}`, `{goal}`, `{data_dir}`, `{notebook_path}`, `{mode}` (interactive/autonomous).

Follow the same pattern as `src/auto_scientist/prompts/discovery.py` for structure and tone.

- [ ] **Step 2: Verify import works**

Run: `uv run python -c "from auto_scientist.prompts.ingestor import INGESTOR_SYSTEM, INGESTOR_USER; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```
feat: add prompt templates for Ingestor agent
```

---

### Task 3: Ingestor Agent Implementation

**Files:**
- Create: `src/auto_scientist/agents/ingestor.py`
- Create: `tests/test_ingestor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_ingestor.py`:

```python
"""Tests for the Ingestor agent module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.agents.ingestor import run_ingestor


def test_run_ingestor_is_async():
    """Verify the function exists and is async."""
    assert asyncio.iscoroutinefunction(run_ingestor)


class TestRunIngestorToolSelection:
    """Verify correct tools are passed based on interactive flag."""

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.ClaudeSDKClient")
    async def test_interactive_mode_includes_ask_user(self, mock_client_cls, tmp_path):
        """In interactive mode, AskUserQuestion tool should be included."""
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        # Create a fake data output so verification passes
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        mock_instance = AsyncMock()
        mock_instance.receive_response = MagicMock(return_value=AsyncMock(
            __aiter__=AsyncMock(return_value=iter([]))
        ))
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        await run_ingestor(raw_data, output_dir, "test goal", interactive=True)

        # Check the ClaudeAgentOptions passed to ClaudeSDKClient
        call_kwargs = mock_client_cls.call_args
        options = call_kwargs[1]["options"] if "options" in call_kwargs[1] else call_kwargs[0][0]
        assert "AskUserQuestion" in options.allowed_tools

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.ClaudeSDKClient")
    async def test_autonomous_mode_excludes_ask_user(self, mock_client_cls, tmp_path):
        """In autonomous mode, AskUserQuestion tool should NOT be included."""
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        mock_instance = AsyncMock()
        mock_instance.receive_response = MagicMock(return_value=AsyncMock(
            __aiter__=AsyncMock(return_value=iter([]))
        ))
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        await run_ingestor(raw_data, output_dir, "test goal", interactive=False)

        call_kwargs = mock_client_cls.call_args
        options = call_kwargs[1]["options"] if "options" in call_kwargs[1] else call_kwargs[0][0]
        assert "AskUserQuestion" not in options.allowed_tools


class TestRunIngestorValidation:
    """Verify error handling when agent produces no output."""

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.ClaudeSDKClient")
    async def test_raises_when_no_data_produced(self, mock_client_cls, tmp_path):
        """Should raise FileNotFoundError if data dir is empty after agent runs."""
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()

        mock_instance = AsyncMock()
        mock_instance.receive_response = MagicMock(return_value=AsyncMock(
            __aiter__=AsyncMock(return_value=iter([]))
        ))
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        with pytest.raises(FileNotFoundError, match="did not produce any data files"):
            await run_ingestor(raw_data, output_dir, "test goal")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingestor.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement `run_ingestor`**

Create `src/auto_scientist/agents/ingestor.py`:

```python
"""Ingestor agent: Phase 0 data canonicalization.

Uses ClaudeSDKClient for persistent session (may need multi-round human Q&A).
Tools: Bash (data inspection, conversion), Read/Write, Glob, Grep.
When interactive: also AskUserQuestion.
Produces: canonical dataset in {output_dir}/data/.
"""

from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from auto_scientist.prompts.ingestor import INGESTOR_SYSTEM, INGESTOR_USER


async def run_ingestor(
    raw_data_path: Path,
    output_dir: Path,
    goal: str,
    interactive: bool = False,
) -> Path:
    """Inspect raw data and produce a canonical dataset.

    Args:
        raw_data_path: Path to raw data file or directory.
        output_dir: Experiment output directory (experiments/).
        goal: The user's modelling goal.
        interactive: If True, agent can ask user questions.

    Returns:
        Path to the canonical data directory (output_dir/data/).
    """
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "lab_notebook.md"

    tools = ["Bash", "Read", "Write", "Glob", "Grep"]
    if interactive:
        tools.append("AskUserQuestion")

    mode = "interactive" if interactive else "autonomous"

    options = ClaudeAgentOptions(
        system_prompt=INGESTOR_SYSTEM,
        allowed_tools=tools,
        max_turns=30,
        permission_mode="acceptEdits",
        cwd=output_dir,
    )

    prompt = INGESTOR_USER.format(
        raw_data_path=str(raw_data_path.resolve()),
        goal=goal,
        data_dir=str(data_dir),
        notebook_path=str(notebook_path),
        mode=mode,
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(f"  [ingestor] {block.text[:200]}")
            elif isinstance(msg, ResultMessage):
                pass

    # Verify something was produced in data_dir
    data_files = list(data_dir.iterdir())
    output_files = [f for f in data_files if f.name != "ingest.py"]
    if not output_files:
        raise FileNotFoundError(
            f"Ingestor agent did not produce any data files in {data_dir}"
        )

    return data_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingestor.py -v`
Expected: ALL PASS (the mock tests may need adjustment based on exact `ClaudeSDKClient` constructor API; adapt the mock setup to match actual constructor signature seen in `discovery.py`)

- [ ] **Step 5: Commit**

```
feat: implement Ingestor agent with ClaudeSDKClient
```

---

### Task 4: Orchestrator Integration

**Files:**
- Modify: `src/auto_scientist/orchestrator.py:45-55`

- [ ] **Step 1: Add `_run_ingestion` method to Orchestrator**

Add a new method:

```python
async def _run_ingestion(self) -> Path:
    """Phase 0: Canonicalize raw data into experiments/data/."""
    from auto_scientist.agents.ingestor import run_ingestor

    if self.data_path is None:
        raise ValueError(
            "Cannot run ingestion without a data path. "
            "Provide --data when starting a new experiment."
        )

    print("INGESTION phase: canonicalizing raw data")
    canonical_data_dir = await run_ingestor(
        raw_data_path=self.data_path,
        output_dir=self.output_dir,
        goal=self.state.goal,
        interactive=self.interactive,
    )
    return canonical_data_dir
```

Note the explicit `None` guard on `self.data_path`.

- [ ] **Step 2: Add ingestion phase dispatch to `run()`**

In `Orchestrator.run()`, add a new block **before** the existing discovery block (before `if self.state.phase == "discovery":`):

```python
# Phase 0: Ingestion
if self.state.phase == "ingestion":
    # Save raw_data_path before ingestion so resume always uses original
    self.state.raw_data_path = self.state.data_path
    self.state.save(state_path)

    canonical_data_dir = await self._run_ingestion()

    # Update paths so downstream agents see canonical data
    self.state.data_path = str(canonical_data_dir)
    self.data_path = canonical_data_dir

    # Add data dir to protected paths if config already exists
    if self.config:
        data_dir_str = str(canonical_data_dir.resolve())
        if data_dir_str not in self.config.protected_paths:
            self.config.protected_paths.append(data_dir_str)

    self.state.phase = "discovery"
    self.state.save(state_path)
```

Key design decisions:
- `raw_data_path` is saved **before** running the ingestor, so on resume the orchestrator can always find the original raw path
- `canonical_data_dir` is scoped inside the `if` block, avoiding `NameError` on resume at later phases
- Protected paths are added inside the ingestion block (only when config exists)

- [ ] **Step 3: Update `_run_ingestion` to use `raw_data_path` on resume**

The `_run_ingestion` method should use `self.state.raw_data_path` if available (resume case), falling back to `self.data_path` (fresh run case). Update the method:

```python
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
```

- [ ] **Step 4: Update Orchestrator docstring**

Update the class docstring to include INGESTION in the state machine:

```python
"""Drives the Ingestion -> Discovery -> Iteration -> Report pipeline.

State machine phases:
    INGESTION -> DISCOVERY -> ANALYZE -> PLAN -> STOP_CHECK -> CRITIQUE ->
    IMPLEMENT -> VALIDATE -> RUN -> EVALUATE -> ANALYZE (loop) or STOP
"""
```

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```
feat: wire Ingestor phase into orchestrator before Discovery
```

---

### Task 5: CLI Changes

**Files:**
- Modify: `src/auto_scientist/cli.py:94-100`

- [ ] **Step 1: Change initial phase in `run` command**

In the `run()` function, change `phase="discovery"` to `phase="ingestion"`:

```python
state = ExperimentState(
    domain=domain or "auto",
    goal=goal,
    phase="ingestion",
    schedule=schedule,
    data_path=data_abs,
)
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -v`
Expected: ALL PASS

- [ ] **Step 3: Run a quick smoke test with the CLI**

Run: `uv run auto-scientist run --help`
Expected: shows help text without errors, `--data`, `--goal`, `--interactive` flags all present.

- [ ] **Step 4: Commit**

```
feat: update CLI to start with ingestion phase
```

---

### Task 6: Update Pipeline Visualizer and Docs

**Files:**
- Modify: `docs/pipeline-visualizer.html` (add Ingestion phase box)
- Modify: `.claude/CLAUDE.md` (update architecture summary)

- [ ] **Step 1: Update CLAUDE.md architecture summary**

Update the "Orchestrator flow" line to include Ingestion:

```
Orchestrator flow: Ingest (interactive) -> [Synthesis] -> Analyst -> Scientist (plan) -> stop check -> Critic <-> Scientist (debate) -> Scientist (revise) -> Coder -> Validate -> Run -> Evaluate
```

Add Ingestor to the pipeline description:
```
0. **Ingestor** (canonicalizer): inspects raw data, asks human for clarification (interactive mode), produces canonical dataset. Uses Bash tools.
```

Add to the "Key Components" list:
```
- `agents/ingestor.py`: `run_ingestor()` canonicalizes raw data into experiments/data/
```

- [ ] **Step 2: Update pipeline visualizer**

Add an Ingestion phase box to `docs/pipeline-visualizer.html` before the Discovery section. Follow the existing visual style (use a distinct color accent, e.g., cyan).

- [ ] **Step 3: Commit**

```
docs: add Ingestor agent to pipeline visualization and architecture docs
```

---

## Notes

**DomainConfig.data_paths interaction:** The `--data` CLI argument always points to a single file or directory. When using a pre-loaded domain config (`--domain spo2`), the `--data` flag still controls what the Ingestor receives. `DomainConfig.data_paths` describes the domain's original data locations and is not modified by ingestion. After ingestion, `state.data_path` points to the canonical output.

**Coder protected_paths enforcement (follow-up):** The Coder agent's `_make_permission_callback` currently only checks that writes are within the `output_dir`. It does not read `DomainConfig.protected_paths`. Enforcing write protection for `experiments/data/` requires updating the Coder's permission callback to also block writes to protected paths. This is tracked as a follow-up, not part of this plan. The Ingestor's prompt instructs generated scripts not to modify files in `data/` as a soft guard.
