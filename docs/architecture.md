# Auto-Scientist: Autonomous Scientific Investigation Framework

## Context

The user has been manually iterating on SpO2 models (v5.x to v7.x, 18 versions) through a cycle of: run experiment, analyze results, hypothesize changes, implement, repeat. Each iteration involves a ~2500-line Python script, ~500-line results file, and diagnostic plots. The manual process also involved cross-pollinating ideas between Claude and GPT (Claude implements/reports, GPT critiques/proposes, Claude responds and implements).

**Goal**: Build a general-purpose autonomous scientific investigation framework where the user provides a dataset and problem statement, and the system autonomously discovers, iterates, and refines approaches. SpO2 is the first domain and test case: reproduce the manual v5 to v7 journey starting from just the raw data. This is a **new standalone repo**.

**Key design decisions**:
- Fully autonomous by default (optional interactive mode)
- LLM as scientist with full creative latitude (decides when to stop, when to pivot)
- Multi-model debate (Claude plans, GPT/Gemini/other critiques the plan)
- Three-phase flow: Discovery, Iteration, Report
- Four-agent iteration loop: Analyst (observe), Scientist (plan), Critic (challenge), Coder (implement)
- Strict information boundaries: only the Coder sees Python code
- Lab notebook as strategic journal; results.txt compiled by the script itself
- Multimodal analysis (plots + text)
- Runs on Claude Pro/Max subscription (no API cost tracking, token-aware scheduling)
- Scheduling support (run overnight to preserve daytime token budget)

---

## Architecture

### Three Phases

```
Phase 1: DISCOVERY (autonomous, one-time)
  User provides: dataset path + problem statement
  System: explores data, researches domain, designs first approach, writes v1 script

Phase 2: ITERATION (autonomous loop)
  [0] Synthesis (optional, every N iterations)
      Condenses the lab notebook into a concise narrative for prompt injection.
      Raw notebook stays on disk untouched.

  [1] Analyst Agent (Claude, read-only observer)
      Input: results text + plots + lab notebook + success criteria
      Output: structured JSON observation (score, criteria, metrics, observations)

  [2] Scientist Agent (Claude, no tools, prompt-in/JSON-out)
      Input: analysis JSON + lab notebook + domain knowledge
      Output: structured JSON plan (hypothesis, strategy, changes, notebook entry,
              per-iteration success criteria)
      Does NOT read Python code. Analysis + notebook is sufficient for planning.
      Defines 3-8 testable predictions (success criteria) for each hypothesis.

  [3] Stop Check
      If Scientist sets should_stop=true, skip to Report phase

  [4] Critic-Scientist Debate (configurable rounds, default 2)
      Round 1: Critic (GPT/Gemini/other) critiques the Scientist's plan
      Round 2+: Scientist (Claude API) responds, Critic refines
      Input: plan JSON + lab notebook + domain knowledge
      Both sides get symmetric context (no analysis, no script)
      Both sides have web search for claim verification and literature lookup
      Output: debate transcript (all rounds)

  [5] Scientist Revision (Claude, no tools, prompt-in/JSON-out)
      Input: original plan + debate transcript + analysis + notebook
      Output: revised plan JSON (same schema, incorporating valid critique)

  [6] Coder Agent (Claude, read/write implementer)
      Input: revised plan JSON + previous script
      Output: new experiment script
      Only agent that reads/writes Python code

  [7] Runner (Python subprocess, no LLM)
      Executes script, captures stdout + plots, saves results
      results.txt is compiled by the script itself (print statements)

  Loop back to [1]

Phase 3: REPORT (one-time)
  Generates final summary: best approach, journey, recommendations
```

### Information Boundaries

A core design principle: each agent sees only the information relevant to its role.

| Agent | Sees code? | Sees analysis JSON? | Sees notebook? | Has web search? |
|-------|-----------|-------------------|---------------|----------------|
| Analyst | No | Produces it | Yes | No |
| Scientist | No | Yes (input) | Yes | No |
| Critic | No | No | Yes | Yes |
| Scientist (debate) | No | No | Yes | Yes |
| Coder | Yes (only agent) | No | No | No |

Why: The plan already incorporates the analysis, so passing both to the Critic
is redundant. Code is an implementation detail that only the implementer needs.
The Critic and Scientist debate strategy on equal footing with symmetric context.

### Agent Details

**Discovery Agent** (Phase 1):
- Uses `ClaudeSDKClient` for persistent session (exploratory, may need multiple queries)
- Tools: Bash (data exploration, stats, plots), WebSearch (literature), Read/Write
- Produces: domain config (success criteria, metric definitions), first experiment script, lab notebook entry #0
- In `--interactive` mode, uses AskUserQuestion to clarify with the user

**Analyst Agent** (Phase 2, step 1):
- Uses `query()` (fresh session each iteration, bounded context)
- Tools: Read (results file + plot PNGs), Glob (find output files)
- Input: results text + plot images + lab notebook + success criteria
- Output: structured JSON: `success_score`, `criteria_results[]`, `key_metrics`, `improvements`, `regressions`, `observations`, `iteration_criteria_results[]`
- Role: pure observer, reports facts only, no recommendations
- Evaluates two tiers: top-level criteria (from config, drives stopping) and per-iteration criteria (from Scientist, transcribed from script output)
- `max_turns`: 5

**Scientist Agent** (Phase 2, step 2):
- Pure prompt-in, JSON-out call (no tools, `max_turns`: 1)
- Input (via prompt injection): analysis JSON + lab notebook + domain knowledge
- Does NOT read Python code; analysis + notebook is sufficient for strategic planning
- Output: structured JSON plan: `hypothesis`, `strategy`, `changes[]`, `expected_impact`, `should_stop`, `stop_reason`, `notebook_entry`, `success_criteria[]`
- Role: strategic thinker, formulates hypotheses and plans, does NOT write code
- Defines 3-8 per-iteration success criteria: concrete, measurable predictions of the hypothesis that the experiment script evaluates

**Critic-Scientist Debate** (Phase 2, step 4):
- Multi-round debate between external critic models and the Scientist (Claude via API)
- Round 1: plain API call to critic model (OpenAI/Google/Anthropic SDK)
- Round 2+: Scientist responds to critique, then critic refines
- Input: plan JSON + lab notebook + domain knowledge
- Neither side sees analysis JSON or experiment scripts (symmetric context)
- Both Critic and Scientist have web search (verify claims, look up papers, check methods)
- Returns full debate transcript for the Scientist revision step
- Configurable: `--debate-rounds N` (default 2; 1 = single-pass, no debate)
- Configurable: list of critic models to consult (can be empty to skip debate entirely)
- Scientist debate model defaults to `claude-sonnet-4-6`

**Scientist Revision** (Phase 2, step 5):
- Second `query()` call to the Scientist after the debate
- Input: original plan + debate transcript + analysis JSON + notebook + domain knowledge
- Output: revised plan JSON (same schema, incorporating valid critique)
- The Coder never sees the debate transcript or critique directly

**Coder Agent** (Phase 2, step 6):
- Uses `query()` (fresh session, reads/writes files via tools)
- Tools: Read, Write, Edit, Bash (for syntax check), Glob, Grep
- Input (via prompt): revised plan JSON + previous script path
- Output: new experiment script at `{version_dir}/experiment.py`
- Role: pure implementer, follows the plan without making strategic decisions
- Only agent that reads or writes Python code
- `max_turns`: 30
- Safety hooks: block writes outside experiments/ dir, block writes to data files

**Runner** (Phase 2, step 7):
- Plain Python `asyncio.create_subprocess_exec`
- Runs: domain-configured command (e.g., `uv run python -u {script_path}`)
- Captures: stdout to results file, checks for output plots
- `results.txt` is compiled by the experiment script itself via print statements.
  The Coder writes scripts that print structured output (approach spec, parameters,
  metrics, diagnostics, success criteria). No LLM post-processing needed.
- Timeout: configurable (default 120 min)
- Syntax validation before run: `python -m py_compile`

### State Machine

```
DISCOVERY -> [SYNTHESIS] -> ANALYZE -> PLAN -> STOP_CHECK -> CRITIQUE
                                                                |
                                                          critic -> defender -> critic
                                                          (repeats for debate_rounds - 1)
                                                                |
                                                             IMPLEMENT -> VALIDATE -> RUN -> EVALUATE
                                                                                                |
                                                                                    [SYNTHESIS] -> ANALYZE (loop)
                                                                                    or STOP
```

`SYNTHESIS` = optional, runs every N iterations (configurable via `--synthesis-interval`).
`VALIDATE` = syntax check on generated script. If fails, re-invoke Coder with error (max 3 retries).

### Lab Notebook

A markdown file (`experiments/lab_notebook.md`) maintained by the orchestrator. The Scientist writes the notebook entry as part of its plan, and the orchestrator appends it to the file.

The notebook is a **strategic journal**: hypothesis, reasoning, key lessons, dead ends. It is NOT a copy of the approach spec or results. That lives in `results.txt`.

### Periodic Investigation Synthesis

After 10+ iterations, the notebook can become bloated. The synthesis step (plain Anthropic API call) condenses the full notebook + compressed history into a concise narrative (~30-50% of original). The synthesis replaces the raw notebook in agent prompts for that iteration; the raw notebook file stays on disk untouched.

Configurable via `--synthesis-interval N` (default 0 = disabled).

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
      synthesis.py                   # Periodic notebook condensation
      agents/
        __init__.py
        discovery.py                 # Phase 1: data exploration + first approach
        analyst.py                   # Phase 2: results + plots -> structured JSON
        scientist.py                 # Phase 2: analysis -> hypothesis + plan
        critic.py                    # Phase 2: multi-model critique of plan
        coder.py                     # Phase 2: plan -> experiment script
        report.py                    # Phase 3: final summary
      prompts/
        __init__.py
        discovery.py
        analyst.py
        scientist.py
        coder.py
        report.py
      models/
        __init__.py
        openai_client.py             # OpenAI API wrapper (web search via Responses API)
        google_client.py             # Google AI wrapper (Google Search grounding)
        anthropic_client.py          # Anthropic API wrapper (web_search tool)
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
    test_critic.py
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
  --goal "Investigate the relationship between X and Y with a physically grounded approach" \
  --max-iterations 20 \
  --critics openai:gpt-4o,google:gemini-2.5-pro \
  --debate-rounds 2 \
  --synthesis-interval 5 \
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
Improvements:
  - Baseline-corrected measurement equation eliminated saturation: 0.4% vs 17.6%.
  - Power-law curvature is identifiable. Both tau_0 and p profiles are non-monotone.
  - LOHO-Inference R2 dramatically improved for QC-passing holds.
  - Parameter efficiency: 18 params vs 28 in v7.04 (-36%), with better fits.

Regressions:
  - b_s = 1.75, far from the ideal 1.0. The baseline-corrected equation makes b_s
    a gain around baseline. b_s > 1 means the model amplifies the deviation.
  - Delta range = 20.2s, still too wide.
  - gamma = 1.30 at upper bound in Stage B.
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

## Verification

### Unit tests
- `test_state.py`: persistence round-trip, crash recovery
- `test_history.py`: compressed history generation
- `test_runner.py`: subprocess execution, timeout, syntax validation
- `test_scheduler.py`: time window logic
- `test_critic.py`: debate loop, plan injection, symmetric context, web search

### Integration test (1 iteration)
1. Use SpO2 domain with a simple seed script and `--max-iterations 1`
2. Verify: Analyst produces valid JSON, Scientist produces plan, Critic produces text, Coder writes valid script, Runner executes, state updated, lab notebook has entry

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
