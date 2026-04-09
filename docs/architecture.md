# Auto-Scientist: Autonomous Scientific Investigation Framework

## Context

The user has been manually iterating on SpO2 models (v5.x to v7.x, 18 versions) through a cycle of: run experiment, analyze results, hypothesize changes, implement, repeat. Each iteration involves a ~2500-line Python script, ~500-line results file, and diagnostic plots. The manual process also involved cross-pollinating ideas between Claude and GPT (Claude implements/reports, GPT critiques/proposes, Claude responds and implements).

**Goal**: Build a general-purpose autonomous scientific investigation framework where the user provides a dataset and problem statement, and the system autonomously discovers, iterates, and refines approaches. SpO2 is the first domain and test case: reproduce the manual v5 to v7 journey starting from just the raw data. This is a **new standalone repo**.

**Key design decisions**:
- Fully autonomous by default (optional interactive mode)
- LLM as scientist with full creative latitude (decides when to stop, when to pivot)
- Multi-model debate (Claude plans, GPT/Gemini/other critiques the plan)
- Two-phase flow: Ingestion, Iteration (unified loop), Report
- Four-agent iteration loop: Analyst (observe), Scientist (plan), Critic (challenge), Coder (implement)
- Strict information boundaries: only the Coder sees Python code
- Lab notebook as strategic journal; results.txt compiled by the script itself
- Multimodal analysis (plots + text)
- Runs on Claude Pro/Max subscription (no API cost tracking, token-aware scheduling)
- Scheduling support (run overnight to preserve daytime token budget)

---

## Architecture

### Two Phases

```
Phase 0: INGESTION (one-time)
  User provides: dataset path + problem statement
  Ingestor: canonicalizes raw data, produces slim DomainConfig

Phase 1: ITERATION (unified loop)
  Every iteration follows: Analyst -> Scientist -> (Debate) -> Coder -> Run -> Evaluate
  Behavior adapts based on what data is available:

  Iteration 0 (exploration):
    Analyst reads raw data files, produces data characterization
    Scientist plans data exploration (distributions, baselines)
    Debate skipped (nothing to challenge yet)
    Coder implements exploration script

  Iteration 1 (first hypothesis):
    Analyst reads exploration results + plots
    Scientist formulates first hypothesis with testable predictions
    Debate challenges the plan
    Coder implements first real experiment

  Iteration 2+ (normal science):
    Analyst reads results, evaluates prediction outcomes
    Scientist plans next hypothesis informed by prediction trajectory
    Debate challenges the plan
    Coder implements

  [1] Analyst Agent (Claude, read-only observer)
      Input: results text + plots + lab notebook
      (Iteration 0: raw data files instead of results)
      Output: structured JSON observation (key_metrics, improvements, regressions,
              observations, prediction_outcomes)
      (Iteration 0: also domain_knowledge, data_summary)

  [2] Scientist Agent (Claude, prompt-in/JSON-out, web search)
      Input: analysis JSON + lab notebook + domain knowledge + prediction history
      Output: structured JSON plan (hypothesis, strategy, changes, expected_impact,
              should_stop, stop_reason, notebook_entry, testable_predictions)
      Does NOT read Python code. Analysis + notebook is sufficient for planning.

  [3] Stop Check
      If Scientist sets should_stop=true, skip to Report phase

  [4] Critic-Scientist Debate (skipped on iteration 0)
      Round 1: Critic (GPT/Gemini/other) critiques the Scientist's plan
      Round 2+: Scientist (Claude API) responds, Critic refines
      Input: plan JSON + analysis JSON + prediction history + lab notebook + domain knowledge
      Both sides share the full evidence base; neither sees experiment scripts
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

Phase 2: REPORT (one-time)
  Generates final summary: best approach, journey, recommendations
```

### Information Boundaries

A core design principle: each agent sees only the information relevant to its role.

| Agent | Sees code? | Sees analysis JSON? | Sees prediction history? | Sees notebook? | Has tools? |
|-------|-----------|-------------------|------------------------|---------------|------------|
| Ingestor | No | No | No | Writes structure summary | Bash, Read, Write, Glob, Grep |
| Analyst | No | Produces it | No | Yes | Read, Glob |
| Scientist | No | Yes (input) | Yes | Yes | Web search |
| Critic (debate) | No | Yes | Yes | Yes | Web search |
| Coder | Yes (only agent) | No | No | No | Bash, Read, Write, Glob |

Why: Debate participants share the full evidence base (analysis, prediction
history, notebook, domain knowledge) so critics can make grounded arguments
based on actual metrics and what has already been tested. Code is an
implementation detail that only the Coder needs.

### Agent Details

**Ingestor Agent** (Phase 0):
- Uses the SDK backend for a persistent session (may need interactive human Q&A)
- Tools: Bash (data inspection, conversion), Read/Write, Glob, Grep
- Produces: canonical dataset in experiments/data/ + slim DomainConfig JSON
- In `--interactive` mode, uses AskUserQuestion to clarify with the user

**Analyst Agent** (Phase 1, step 1):
- Uses `query()` (fresh session each iteration, bounded context)
- Tools: Read (results file + plot PNGs), Glob (find output files)
- Input: results text + plot images + lab notebook
- (Iteration 0: raw data directory instead of results)
- Output: structured JSON: `key_metrics`, `improvements`, `regressions`, `observations`, `prediction_outcomes[]`, `data_diagnostics[]`
- (Iteration 0: also `domain_knowledge`, `data_summary`)
- `prediction_outcomes` entries: `{pred_id, prediction, outcome, evidence}` where outcome is "confirmed", "refuted", or "inconclusive"
- `data_diagnostics` entries: `{variables, pattern, evidence}` - cross-cutting structural patterns the Analyst notices across its own observations (co-moving variables, distributional boundaries, sign-flip anomalies). Domain-agnostic meta-observation step that gives the Scientist raw material for abductive reasoning about hidden factors.
- Role: pure observer, reports facts only, no recommendations
- `max_turns`: 30

**Scientist Agent** (Phase 1, step 2):
- Prompt-in, JSON-out call with web search (`max_turns`: 10)
- Input (via prompt injection): analysis JSON + lab notebook + domain knowledge + prediction history + pending abductions from prior iterations
- Does NOT read Python code; analysis + notebook is sufficient for strategic planning
- Output: structured JSON plan: `hypothesis`, `strategy`, `changes[]`, `expected_impact`, `should_stop`, `stop_reason`, `notebook_entry`, `testable_predictions[]`, `refutation_reasoning[]`, `deprioritized_abductions[]`
- `testable_predictions` entries: `{prediction, diagnostic, if_confirmed, if_refuted, follows_from}` where `follows_from` links to a prior pred_id to form reasoning trajectories
- `refutation_reasoning` entries: `{refuted_pred_id, assumptions_violated, alternative_explanation, testable_consequence}` - structured abductive reasoning generated for each refuted prediction. The `alternative_explanation` must describe the system under study (naming specific measured or unmeasured entities and the mechanism connecting them), not concerns about the analysis pipeline. The `testable_consequence` becomes a candidate for the next iteration's testable_predictions via follows_from.
- `deprioritized_abductions` entries: `{refuted_pred_id, reason}` - explicit decisions to not pursue a pending abduction's testable consequence, so the carry-forward mechanism knows the thread is closed by judgment, not by oversight.
- Role: strategic thinker, formulates hypotheses and plans, does NOT write code
- Decides when to stop via `should_stop` based on scientific judgment (goal satisfaction, diminishing returns, structural limits), not metric thresholds

**Critic** (Phase 1, step 4):
- Single-pass critique from external critic models (OpenAI/Google/Anthropic SDK)
- Each persona produces structured concerns, no back-and-forth
- Input: plan JSON + analysis JSON + prediction history + lab notebook + domain knowledge
- Critics receive the full evidence base but do not see experiment scripts
- Critics have web search (verify claims, look up papers, check methods)
- Returns structured critique for the Scientist revision step
- Configurable: list of critic models to consult (can be empty to skip debate entirely)

**Scientist Revision** (Phase 1, step 5):
- Second `query()` call to the Scientist after the debate
- Input: original plan + debate transcript + analysis JSON + notebook + domain knowledge
- Output: revised plan JSON (same schema, incorporating valid critique)
- The Coder never sees the debate transcript or critique directly

**Coder Agent** (Phase 1, step 6):
- Uses `query()` (fresh session, reads/writes files via tools)
- Tools: Read, Write, Edit, Bash, Glob, Grep
- Input (via prompt): revised plan JSON + previous script path + run config
- Output: experiment script at `{version_dir}/experiment.py` + `run_result.json`
- Role: pure implementer, follows the plan without making strategic decisions
- Only agent that reads/writes and runs Python code
- Writes the script, runs it (with timeout), fixes errors, and reports results
- `max_turns`: 50 (to accommodate write-run-fix cycles)
- `results.txt` is compiled by the experiment script itself via print statements.
  The Coder writes scripts that print structured output (approach spec, parameters,
  metrics, diagnostics, HYPOTHESIS TESTS). No LLM post-processing needed.
- HYPOTHESIS TESTS section: if the plan includes `testable_predictions`, the script
  prints each prediction's outcome with a bracketed pred_id (e.g., `[1.1]`) so the
  Analyst can match results back to predictions
- `run_result.json` reports success/failure, return code, timeout status, and attempts
- Safety hooks: block writes outside experiments/ dir, block writes to data files

### State Machine

```
INGESTION -> ANALYZE -> PLAN -> STOP_CHECK -> (DEBATE, skipped iter 0) ->
             IMPLEMENT (write + run) -> EVALUATE -> ANALYZE (loop) or REPORT
```

Phases: `ingestion -> iteration (loop) -> report -> stopped`

### Lab Notebook

A markdown file (`experiments/lab_notebook.md`) maintained by the orchestrator. The Scientist writes the notebook entry as part of its plan, and the orchestrator appends it to the file.

The notebook is a **strategic journal**: hypothesis, reasoning, key lessons, dead ends. It is NOT a copy of the approach spec or results. That lives in `results.txt`.

### Abduction Carry-Forward

When the Scientist generates `refutation_reasoning` for a refuted prediction, the `testable_consequence` is typically tested in the next iteration via a new prediction with `follows_from` set to the refuted pred_id. But the Scientist may also raise an abduction without immediately testing it (for example, because the current iteration is already tightly scoped).

The orchestrator tracks these as `pending_abductions` in `ExperimentState`. On every subsequent iteration, pending abductions are injected into the Scientist's prompt with explicit instructions: address each via a new `follows_from` prediction or list it in `deprioritized_abductions` with a reason. An abduction is considered "addressed" if either condition is met (pure string matching, no LLM inference in the orchestrator).

Three downstream agents also receive the pending abductions:

- **Critics** during debate flag any unaddressed abduction as a dropped thread
- **Assessor** at stop time treats each unaddressed abduction as an open sub-question blocking "thorough" coverage rating
- **Report** documents remaining open abductions in the Limitations section with explicit "what was raised, why not tested, what would resolve it"

This machinery exists specifically to solve a failure mode observed in early runs: models would generate thoughtful alternative explanations for refuted predictions and then silently forget them. Without structured persistence and multi-agent enforcement, abductive reasoning becomes a one-shot exercise that doesn't drive the investigation forward.

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
        ingestor.py                  # Phase 0: data canonicalization + config
        analyst.py                   # Phase 1: results + plots -> structured JSON
        scientist.py                 # Phase 1: analysis -> hypothesis + plan
        critic.py                    # Phase 1: multi-model critique of plan
        coder.py                     # Phase 1: plan -> experiment script
        report.py                    # Phase 2: final summary
      prompts/
        __init__.py
        ingestor.py
        analyst.py
        scientist.py
        coder.py
        critic.py
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

### DomainConfig (operational only)

```python
class DomainConfig(BaseModel):
    name: str
    description: str
    data_paths: list[str]
    run_command: str                          # e.g., "uv run {script_path}"
    run_cwd: str
    run_timeout_minutes: int = 120
    version_prefix: str = "v"
    protected_paths: list[str] = []
```

### ExperimentState

```python
class ExperimentState(BaseModel):
    domain: str
    goal: str
    phase: str                               # "ingestion", "iteration", "report", "stopped"
    iteration: int = 0
    versions: list[VersionEntry] = []
    dead_ends: list[str] = []
    best_version: str | None = None
    schedule: str | None = None              # e.g., "22:00-06:00"
    consecutive_failures: int = 0
    data_path: str | None = None
    raw_data_path: str | None = None
    config_path: str | None = None
    domain_knowledge: str = ""               # Set by Analyst iteration 0
    prediction_history: list[PredictionRecord] = []

class PredictionRecord(BaseModel):
    pred_id: str                             # "{iteration}.{index}", e.g., "1.1"
    iteration_prescribed: int
    iteration_evaluated: int | None = None
    prediction: str
    diagnostic: str
    if_confirmed: str
    if_refuted: str
    follows_from: str | None = None          # pred_id of parent prediction
    outcome: str = "pending"                 # "pending", "confirmed", "refuted", "inconclusive"
    evidence: str = ""
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
  --schedule "22:00-06:00"

# Resume after crash, pause, or overnight stop
auto-scientist resume --from experiments/runs/my-run

# Fork a run and resume from iteration 3
auto-scientist resume --from experiments/runs/my-run --fork --from-iteration 3

# Fork and resume from the scientist agent within iteration 3
auto-scientist resume --from experiments/runs/my-run --fork --from-iteration 3 --from-agent scientist

# Check progress
auto-scientist status --from experiments/runs/my-run

# Interactive mode
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

### Example: Hypothesis Tests (what the script outputs)

```
HYPOTHESIS TESTS
----------------
[1.1] Power-law descent produces identifiable curvature: CONFIRMED (p=2.8, non-monotone tau_0 profile)
[1.2] Baseline correction eliminates saturation: CONFIRMED (0.4% vs 17.6% in v7.04)
[1.3] b_s converges near 1.0: REFUTED (b_s=1.75, amplifying deviation)
[1.4] Delta range < 10s: REFUTED (delta range=20.2s, improved but still wide)
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
2. Let it run from scratch (iteration 0 explores, iteration 1 formulates first hypothesis with testable predictions)
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
