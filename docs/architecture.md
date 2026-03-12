# Auto-Scientist: Autonomous Scientific Modelling Framework

## Context

The user has been manually iterating on SpO2 models (v5.x to v7.x, 18 versions) through a cycle of: run experiment, analyze results, hypothesize changes, implement, repeat. Each iteration involves a ~2500-line Python script, ~500-line results file, and diagnostic plots. The manual process also involved cross-pollinating ideas between Claude and GPT (Claude implements/reports, GPT critiques/proposes, Claude responds and implements).

**Goal**: Build a general-purpose autonomous scientific modelling framework where the user provides a dataset and problem statement, and the system autonomously discovers, iterates, and refines models. SpO2 is the first domain and test case: reproduce the manual v5 to v7 journey starting from just the raw data. This is a **new standalone repo**.

**Key design decisions**:
- Fully autonomous by default (optional interactive mode)
- LLM as scientist with full creative latitude (decides when to stop, when to pivot)
- Multi-model debate (Claude implements, GPT/Gemini/other critiques)
- Three-phase flow: Discovery, Iteration, Report
- Analyst/Scientist agent split for context efficiency
- Lab notebook pattern for cross-iteration memory
- Multimodal analysis (plots + text)
- Runs on Claude Pro/Max subscription (no API cost tracking, token-aware scheduling)
- Scheduling support (run overnight to preserve daytime token budget)

---

## Architecture

### Three Phases

```
Phase 1: DISCOVERY (autonomous, one-time)
  User provides: dataset path + problem statement
  System: explores data, researches domain, designs first model, writes v1 script

Phase 2: ITERATION (autonomous loop)
  [1] Analyst Agent (Claude, read-only)
      Input: results text + plots + lab notebook
      Output: structured analysis JSON

  [2] Critic (GPT/Gemini/other LLM, text-only API call)
      Input: analysis JSON + lab notebook
      Output: critique, alternative hypotheses, challenges

  [3] Scientist Agent (Claude, read/write)
      Input: analysis + critique + previous script + lab notebook
      Output: new script + updated lab notebook

  [4] Runner (Python subprocess, no LLM)
      Executes script, captures stdout + plots, saves results

  Loop back to [1]

Phase 3: REPORT (one-time)
  Generates final summary: best model, journey, recommendations
```

### Agent Details

**Discovery Agent** (Phase 1):
- Uses `ClaudeSDKClient` for persistent session (exploratory, may need multiple queries)
- Tools: Bash (data exploration, stats, plots), WebSearch (literature), Read/Write
- Produces: domain config (success criteria, metric definitions), first experiment script, lab notebook entry #0
- In `--interactive` mode, uses AskUserQuestion to clarify with the user

**Analyst Agent** (Phase 2, step 1):
- Uses `query()` (fresh session each iteration, bounded context)
- Tools: Read (results file + plot PNGs), Glob (find output files)
- Input: results text + plot images + lab notebook
- Output: structured JSON: `success_score`, `failures[]`, `key_metrics`, `what_worked`, `what_didnt_work`, `stagnation_detected`, `paradigm_shift_recommended`, `should_stop`, `stop_reason`
- `max_turns`: 5

**Critic** (Phase 2, step 2):
- Plain API call (OpenAI/Google/Anthropic SDK), no agent tools needed
- Input: analysis JSON + lab notebook + compressed history
- Output: free-text critique with challenges, alternative hypotheses, suggestions
- Configurable: list of models to consult (can be empty for no debate)

**Scientist Agent** (Phase 2, step 3):
- Uses `query()` (fresh session, reads files via tools)
- Tools: Read, Write, Edit, Bash (for syntax check), Glob, Grep
- Input (via prompt): analysis JSON + critic's feedback + lab notebook
- Input (via tools): reads previous script, writes new script
- Output: new experiment script + updated lab notebook + hypothesis doc
- `max_turns`: 30
- Safety hooks: block writes outside experiments/ dir, block writes to data files

**Runner** (Phase 2, step 4):
- Plain Python `asyncio.create_subprocess_exec`
- Runs: domain-configured command (e.g., `uv run python -u {script_path}`)
- Captures: stdout to results file, checks for output plots
- Timeout: configurable (default 120 min)
- Syntax validation before run: `python -m py_compile`

### State Machine

```
DISCOVERY -> ANALYZE -> CRITIQUE -> IMPLEMENT -> VALIDATE -> RUN -> EVALUATE
                                                                      |
                                                              ANALYZE (loop)
                                                              or STOP
```

`VALIDATE` = syntax check on generated script. If fails, re-invoke Scientist with error (max 3 retries).

### Lab Notebook

A markdown file (`experiments/lab_notebook.md`) maintained by the Scientist. The Scientist reads this at the start of each iteration and appends to it. The orchestrator never modifies it.

### Compressed History (for Critic)

The orchestrator maintains a separate compact summary (~1 line per iteration) for the Critic, extracted mechanically from the Analyst's structured JSON.

### Scheduling

The CLI supports a `--schedule` flag for time-based execution:
```bash
auto-scientist run --data ... --goal "..." --schedule "22:00-06:00"
```

The orchestrator checks the current time before each iteration. If outside the scheduled window, it saves state and sleeps until the window opens. This preserves the user's Claude Pro/Max token budget for daytime interactive use.

Implementation: simple time check in the orchestrator loop. No external scheduler (cron) needed - the process stays alive and self-pauses.

---

## Project Structure

```
auto-scientist/
  pyproject.toml
  README.md
  src/
    auto_scientist/
      __init__.py
      cli.py                         # CLI entry: run, resume, status
      orchestrator.py                # Main loop, state machine, scheduling
      state.py                       # ExperimentState + JSON persistence
      config.py                      # DomainConfig schema
      runner.py                      # Subprocess experiment runner
      history.py                     # Compressed history builder
      scheduler.py                   # Time-window scheduling logic
      agents/
        __init__.py
        discovery.py                 # Phase 1: data exploration + first model
        analyst.py                   # Phase 2: results + plots -> structured JSON
        critic.py                    # Phase 2: multi-model critique
        scientist.py                 # Phase 2: implement changes + update notebook
        report.py                    # Phase 3: final summary
      prompts/
        __init__.py
        discovery.py
        analyst.py
        scientist.py
        report.py
      models/
        __init__.py
        openai_client.py             # OpenAI API wrapper for critic
        google_client.py             # Google AI wrapper for critic
        anthropic_client.py          # Anthropic API wrapper for critic
  domains/
    spo2/
      __init__.py
      config.py                      # SpO2 DomainConfig
      prompts.py                     # SpO2 domain knowledge
      seed/
        data/                        # spo2.db
      examples/                      # Real examples from the manual process
        iteration_example.md         # Annotated v7.04 to v7.05 transition
        results_example.txt          # v7.05 results (reference format)
        analysis_example.json        # What the Analyst should produce
    example_template/
      __init__.py
      config.py
      prompts.py
  experiments/                       # Output dir (gitignored)
    .gitkeep
  tests/
    __init__.py
    test_state.py
    test_history.py
    test_runner.py
    test_scheduler.py
  docs/
    architecture.md
```

### Dependencies (`pyproject.toml`)

```toml
[project]
name = "auto-scientist"
requires-python = ">=3.11"
dependencies = [
    "claude-code-sdk",
    "pydantic>=2.0",
    "click",
    "openai",
    "google-genai",
    "anthropic",
]
```

---

## Key Data Models

### DomainConfig

```python
class DomainConfig(BaseModel):
    name: str
    description: str
    data_paths: list[str]
    run_command: str                          # e.g., "uv run python -u {script_path}"
    run_cwd: str
    run_timeout_minutes: int = 120
    version_prefix: str = "v"
    success_criteria: list[SuccessCriterion]
    domain_knowledge: str                    # Injected into agent prompts
    protected_paths: list[str] = []
```

### ExperimentState

```python
class ExperimentState(BaseModel):
    domain: str
    goal: str
    phase: str                               # "discovery", "iteration", "report", "stopped"
    iteration: int = 0
    versions: list[VersionEntry] = []
    dead_ends: list[str] = []
    best_version: str | None = None
    best_score: int = 0
    config: DomainConfig | None = None
    schedule: str | None = None              # e.g., "22:00-06:00"
```

---

## Safety Mechanisms

1. **Write protection**: PreToolUse hook blocks writes outside `experiments/` and to data files
2. **No destructive bash**: Hook blocks `rm -rf`, `git push`, `git reset`, etc.
3. **Syntax validation**: `python -m py_compile` before running generated scripts
4. **Iteration cap**: Hard stop at `--max-iterations`
5. **Crash recovery**: State persisted to JSON after every phase transition; `auto-scientist resume` picks up
6. **Consecutive failure cap**: Stop after N crashes/failures in a row (default 5)

---

## CLI Interface

```bash
# Fully autonomous from raw data (the primary use case)
auto-scientist run \
  --data ./my_data.csv \
  --goal "Model the relationship between X and Y with a physically grounded model" \
  --max-iterations 20 \
  --critics openai:gpt-4o,google:gemini-2.5-pro \
  --schedule "22:00-06:00"

# Resume after crash, pause, or overnight stop
auto-scientist resume --state experiments/state.json

# Check progress
auto-scientist status --state experiments/state.json

# Interactive discovery mode
auto-scientist run --data ./data.csv --goal "..." --interactive
```

---

## Real Examples from SpO2 Manual Process

These examples from the actual v5 to v7 journey are included in `domains/spo2/examples/` to guide the agents. They show what good scientific iteration looks like.

### Example: Version-to-Version Changes (v7.04 to v7.05)

This shows the kind of structural reasoning the Scientist should produce:

```
1. Power-law descent replaces piecewise-linear. Global parameter p controls
   curvature. For p > 1, the derivative at t=0 is zero (natural plateau), then
   descent accelerates toward the nadir. This decouples latent shape from sensor
   delay, so the kernel doesn't need to act as a per-hold shape knob.

2. S_start locked to B_h = median(SpO2[t<=20]) by construction. Removed from
   parameter vector. Eliminates baseline ambiguity entirely.

3. r_offset removed. Measurement equation is now baseline-corrected:
   pred = B_h + b_s * (filtered - B_h). Pred = B_h at plateau regardless of b_s.
   Saturation eliminated by construction.

4. m_h removed (fixed at 0). t_turn = t_end for all holds. With the power-law
   providing curvature, nadir delay is carried by the kernel (tau_0 + delta_h).

Parameter reduction: Stage A 28 -> 18 (-10), Stage B 18 -> 17 (-1).
```

### Example: Analysis Output (what the Analyst should produce)

```
What worked:
  - Baseline-corrected measurement equation eliminated saturation: 0.4% vs 17.6%.
  - Power-law curvature is identifiable. Both tau_0 and p profiles are non-monotone.
  - LOHO-Inference R2 dramatically improved for QC-passing holds.
  - Parameter efficiency: 18 params vs 28 in v7.04 (-36%), with better fits.

What didn't work:
  - b_s = 1.75, far from the ideal 1.0. The baseline-corrected equation makes b_s
    a gain around baseline. b_s > 1 means the model amplifies the deviation.
  - Delta range = 20.2s, still too wide.
  - gamma = 1.30 at upper bound in Stage B.

Recommended next steps:
  - Widen gamma bounds to [0.8, 1.6]
  - Investigate b_s = 1.75: try reducing cv or adding per-hold b_s
  - Consider excluding RV#4 entirely
```

### Example: Success Criteria (what the system evaluates)

```
 # Criterion                              Result    Status
 1 b_s in (0.8, 1.2)                      1.75      FAIL
 2 tau_0 in [10, 30]                       15.0      PASS
 3 p in [1.5, 3.5]                         3.52      FAIL (borderline)
 4 Saturation < 5%                         0.4%      PASS
 5 Deltas not at bounds                    0/5       PASS
 ...
Score: 9/15 PASS, 6 FAIL
```

### Example: Paradigm Shift Pattern (v6 to v7)

The v6 series used piecewise-linear latent shapes. After multiple iterations showed that:
- The linear latent + normalized kernel had no interior curvature
- Sensor delay (tau_0) was forced to compensate for shape, making it unidentifiable
- The effective lag range was 37x between holds (should be <5x)

The system recognized these as structural limitations (not tuning issues) and shifted to power-law descent in v7, which decoupled shape from delay. This is the kind of paradigm shift the Scientist should be able to make when stagnation is detected.

### Example: Comparison Table (cross-version tracking)

```
Param         v7.04     v7.05     Change
tau_0         17.26     15.01     -13% (less inflated)
b_s            1.04      1.75     +68% (regression)
saturation    17.6%      0.4%    fixed
delta range   36.3s     20.2s    -44% (improved)
QC-pass R2a    0.57      0.97    +70% (dramatic)
```

---

## Implementation Plan

### Commit 1: Scaffold (DONE)

Created repo with full directory structure, all modules with docstrings/stubs, pyproject.toml, 34 passing tests, domain examples, docs, and CLAUDE.md.

### Commit 2: Runner + Scheduler + History
- Subprocess execution with timeout
- Time-window scheduling
- Compressed history builder

### Commit 3: Analyst Agent
- `query()` with structured output, multimodal (text + plots)

### Commit 4: Critic
- Multi-model critique dispatcher (OpenAI, Google, Anthropic wrappers)

### Commit 5: Scientist Agent
- `query()` with file tools, safety hooks, lab notebook

### Commit 6: Orchestrator + Iteration Loop
- State machine, error handling, crash recovery

### Commit 7: Discovery Agent
- Autonomous data exploration, domain research, first model

### Commit 8: Report Agent
- Final summary generation

### Commit 9: SpO2 Domain + Examples
- DomainConfig, domain knowledge prompts, real v5 to v7 examples

### Commit 10: CLI polish + README + domain template

---

## Verification

### Unit tests
- `test_state.py`: persistence round-trip, crash recovery
- `test_history.py`: compressed history generation
- `test_runner.py`: subprocess execution, timeout, syntax validation
- `test_scheduler.py`: time window logic

### Integration test (1 iteration)
1. Use SpO2 domain with a simple seed script and `--max-iterations 1`
2. Verify: Analyst produces valid JSON, Critic produces text, Scientist writes valid script, Runner executes, state updated, lab notebook has entry

### Reproduction test (the real validation)
1. `auto-scientist run --data domains/spo2/seed/data/ --goal "Model SpO2 dynamics during breath-holds in a physiologically grounded, transferable way" --max-iterations 20 --schedule "22:00-06:00"`
2. Let it run from scratch (no seed script - Discovery creates v1)
3. Compare: does it converge to something comparable to the manual v7.05 result? Does it discover similar structural insights (baseline correction, power-law curvature)?
4. This is the ultimate test of the framework

---

## Reference Files (from spO2_modelling repo)

- `backend/scripts/experiments/exp_v7_05/exp_v7_05_powerlaw.py` - Latest script (2322 lines)
- `backend/scripts/experiments/exp_v7_05/exp_v7_05_results.txt` - Latest results (493 lines)
- `backend/scripts/experiments/exp_v7_04/exp_v7_04_results.txt` - Previous results (423 lines)
- `backend/pyproject.toml` - Dependencies for experiment scripts
- `data/spo2.db` - Experimental data
- Selected earlier experiment results (v5, v6 series) for paradigm shift examples
