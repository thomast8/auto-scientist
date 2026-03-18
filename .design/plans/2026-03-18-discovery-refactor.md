# Discovery Agent Refactoring - Implementation Plan

**Goal:** Separate exploration from code generation in Discovery, remove the `--domain` path, and wire Discovery -> Scientist -> Coder for v00.

**Architecture:** Discovery produces only config + notebook. The Scientist plans v00 from notebook findings. The Coder implements v00. The domain-config path (`--domain` CLI flag) is removed entirely.

**Tech Stack:** Existing framework, no new dependencies.

---

### Task 1: Refactor Discovery Agent

**Files:**
- Modify: `src/auto_scientist/agents/discovery.py`
- Modify: `src/auto_scientist/prompts/discovery.py`
- Modify: `tests/test_discovery.py`

**What:** Strip script generation from Discovery. Return type becomes `DomainConfig` (not `tuple[DomainConfig, Path]`). Remove Step 3 from prompt. Remove v00 dir creation and script verification.

- [ ] **Step 1: Update tests** - Change `test_interactive_mode_includes_ask_user`: remove script file setup (lines 40-42), change `config, script = await run_discovery(...)` to `config = await run_discovery(...)`. Remove `test_missing_script_raises` entirely.
- [ ] **Step 2: Run tests, verify failures** - Discovery tests should fail (return type mismatch)
- [ ] **Step 3: Update discovery.py** - Change return type to `DomainConfig`. Remove `version_dir`, `script_path`, `notebook_path` variables and script verification. Remove `version_dir` and `script_name` from the `DISCOVERY_USER.format()` call (they'll no longer exist in the template). Update module docstring. Keep config loading/validation.
- [ ] **Step 4: Update prompts/discovery.py** - Remove Step 3 (Write Experiment Script) and all script-related template variables (`{version_dir}`, `{script_name}`). Refocus Step 2 on conceptual design. Renumber to Steps 1-4.
- [ ] **Step 5: Run tests, verify they pass**
- [ ] **Step 6: Commit** - "refactor: strip script generation from Discovery agent"

---

### Task 2: Update Scientist Prompt for v00

**Files:**
- Modify: `src/auto_scientist/prompts/scientist.py`

**What:** Change first-iteration guidance from "(v01)" to "(v00)" and "baseline results" to "exploration findings in the notebook".

- [ ] **Step 1: Edit SCIENTIST_SYSTEM** - One-line wording change in the notebook entry section
- [ ] **Step 2: Run existing scientist tests, verify they pass**
- [ ] **Step 3: Commit** - "fix: update Scientist prompt to handle v00 planning"

---

### Task 3: Remove Domain-Config Path from Orchestrator

**Files:**
- Modify: `src/auto_scientist/orchestrator.py`
- Modify: `tests/test_orchestrator.py`

**What:** Remove `config` parameter from `Orchestrator.__init__`. Remove the `if self.config is not None` branch. Rewrite `_run_discovery()` to: Discovery -> Scientist -> Coder. Update all test fixtures.

- [ ] **Step 1: Update orchestrator tests** - Remove `config` fixture and `config` param from `orchestrator` fixture. For `TestPhaseTransitions` tests: remove `config=` from Orchestrator construction and set `orchestrator.config` directly on the instance after construction (these tests start in "iteration" phase, so `_run_discovery` never runs). Add a test for the new `_run_discovery` flow: mock `run_discovery` returning DomainConfig, mock `run_scientist` returning a plan dict, mock `run_coder` returning a script path, verify all three are called in sequence.
- [ ] **Step 2: Run tests, verify failures**
- [ ] **Step 3: Update orchestrator.py** - Remove `config` param from `__init__`, initialize `self.config = None`. Rewrite `_run_discovery()`: call `run_discovery()` -> set `self.config`, call `run_scientist(analysis={}, ...)` -> get plan, append notebook entry, call `run_coder()` -> get script path. Keep the run+evaluate v00 logic at the end. Note: `self.config` is `None` during ingestion, so the `if self.config:` guard for protected_paths in `_run_ingestion()` won't execute; this is fine because Discovery's config will set protected_paths later.
- [ ] **Step 4: Run tests, verify they pass**
- [ ] **Step 5: Commit** - "refactor: wire Discovery -> Scientist -> Coder in orchestrator"

---

### Task 4: Remove `--domain` from CLI

**Files:**
- Modify: `src/auto_scientist/cli.py`
- Modify: `tests/test_cli.py`

**What:** Remove `--domain` flag, `load_domain_config()`, and domain config handling from `run` and `resume` commands.

- [ ] **Step 1: Update CLI tests** - Remove `TestLoadDomainConfig` class and the `load_domain_config` import. Remove `config` from `Orchestrator` call assertions in `TestRunCommand`. Update `resume` test: remove domain config reload logic.
- [ ] **Step 2: Run tests, verify failures**
- [ ] **Step 3: Update cli.py** - Remove `load_domain_config()` and `importlib` import. Remove `--domain` option from `run`. Remove `config` from Orchestrator construction. Simplify `resume`: remove domain config reload (config comes from persisted state).
- [ ] **Step 4: Run tests, verify they pass**
- [ ] **Step 5: Commit** - "refactor: remove --domain CLI flag"

---

### Task 5: Update Documentation

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/pipeline-visualizer.html` (if it exists)

**What:** Update Discovery agent description and pipeline visualization.

- [ ] **Step 1: Update architecture.md** - Discovery "Produces" list: remove "first experiment script". Update CLI examples to remove `--domain`. Add note that domain configs are planned but not yet supported.
- [ ] **Step 2: Update pipeline-visualizer.html** - Update data flow to show Discovery -> Scientist -> Coder for v00 (if the file exists).
- [ ] **Step 3: Commit** - "docs: update architecture for Discovery refactoring"
