: # SDK Migration Implementation Plan

**Goal:** Replace non-existent `claude_agent_sdk` with the real `claude_code_sdk` package across all agent files and tests.

**Architecture:** Mechanical rename of imports and class references. For the 3 call sites using `output_format` (not available in `claude_code_sdk`), append JSON schema instructions to the system prompt instead. Existing JSON parsing logic stays unchanged.

**Tech Stack:** Python, `claude_code_sdk`, pytest

---

### Task 1: Update conftest mock

**Files:**
- Modify: `tests/conftest.py`

**What:** Update the test mock to use the real SDK module name and class names. This must happen first so tests can validate subsequent changes.

- [ ] **Step 1: Update mock** - Change module key to `"claude_code_sdk"`, rename `ClaudeAgentOptions` -> `ClaudeCodeOptions`, update comment text.
- [ ] **Step 2: Run tests** - `uv run pytest` to confirm mock still works.
- [ ] **Step 3: Commit**

---

### Task 2: Rename imports in simple agents (report, coder, discovery)

**Files:**
- Modify: `src/auto_scientist/agents/report.py` (import line 9)
- Modify: `src/auto_scientist/agents/coder.py` (import lines 15-22)
- Modify: `src/auto_scientist/agents/discovery.py` (import lines 12-18, docstring line 2)

**What:** Straight find-and-replace: `claude_agent_sdk` -> `claude_code_sdk`, `ClaudeAgentOptions` -> `ClaudeCodeOptions`. These agents don't use `output_format`, so no prompt changes needed. Discovery also imports `ClaudeSDKClient`, which exists in `claude_code_sdk` unchanged.

- [ ] **Step 1: Update imports and usages** in all three files.
- [ ] **Step 2: Run lint and tests** - `uv run ruff check src/ tests/ && uv run pytest`
- [ ] **Step 3: Commit**

---

### Task 3: Migrate analyst.py (has output_format)

**Files:**
- Modify: `src/auto_scientist/agents/analyst.py` (import lines 14-20, options line 130-137)

**What:** Rename imports. Remove `output_format` kwarg from `ClaudeCodeOptions()`. Append JSON schema instruction block to the system prompt passed to `options`. The existing fence-stripping + `json.loads()` at lines 160-166 already handles parsing.

- [ ] **Step 1: Update imports and remove output_format** - Replace import, rename class, drop the `output_format=` line. Append JSON output instruction with `ANALYST_SCHEMA` to the system prompt string.
- [ ] **Step 2: Run tests** - `uv run pytest tests/test_analyst.py`
- [ ] **Step 3: Commit**

---

### Task 4: Migrate scientist.py (has output_format x2)

**Files:**
- Modify: `src/auto_scientist/agents/scientist.py` (import lines 11-17, options lines 109-114 and 198-203)

**What:** Same pattern as analyst. Two call sites: `run_scientist()` and `run_scientist_revision()`. Both use `SCIENTIST_PLAN_SCHEMA`. Both already have fence-stripping + `json.loads()`.

- [ ] **Step 1: Update imports and both call sites** - Replace import, rename class, drop both `output_format=` lines. Append JSON output instruction with `SCIENTIST_PLAN_SCHEMA` to each system prompt.
- [ ] **Step 2: Run tests** - `uv run pytest tests/test_scientist.py`
- [ ] **Step 3: Commit**

---

### Task 5: Final verification

- [ ] **Step 1: Full lint** - `uv run ruff check src/ tests/`
- [ ] **Step 2: Full test suite** - `uv run pytest`
- [ ] **Step 3: Import smoke test** - `uv run python -c "from auto_scientist.agents.analyst import run_analyst; from auto_scientist.agents.scientist import run_scientist; from auto_scientist.agents.coder import run_coder; from auto_scientist.agents.report import run_report; from auto_scientist.agents.discovery import run_discovery; print('all imports ok')"`
- [ ] **Step 4: Verify no stale references** - grep for `claude_agent_sdk` in `src/` and `tests/`
