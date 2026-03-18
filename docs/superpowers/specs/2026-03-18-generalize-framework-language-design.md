# Generalize Framework Language

**Date:** 2026-03-18
**Status:** Draft
**Branch:** `feat/data-ingestion-agent` (or new branch)

## Problem

The framework's prompts, docs, and domain configs use ML/model-fitting language that biases agents toward thinking of every investigation as a model-building exercise. The word "model" appears 11 times in the Discovery prompt alone. Examples reference R2 scores, regression, and parameter fitting.

This makes the framework a model optimizer, not a general scientist. The system should work equally well for:
- Physics modeling (SpO2)
- Algorithm optimization ("make this sorting faster")
- Data analysis ("what explains this trend?")
- Meta-investigation ("improve this framework's own prompts")
- Any iterative investigation with measurable outcomes

## Design

### Principle

Static prompts describe generic roles (observe, plan, challenge, implement) without suggesting any methodology. The only domain-specific content comes from `domain_knowledge`, which is user-provided or Discovery-generated. Agents are free to decide the approach.

### Central terminology shift

| Concept | Current | New |
|---------|---------|-----|
| What we're doing | "modelling experiment" | "investigation" |
| What the script produces | "model" | "artifact" / "approach" |
| What we measure | "model quality" | "outcome quality" |
| The central output | "Best Model" | "Best Approach" |
| What we do with data | "fit the model" | "implement and evaluate" |
| The framework title | "autonomous scientific modelling" | "autonomous scientific investigation" |

### File-by-file changes

#### 1. `src/auto_scientist/prompts/discovery.py`

**Word-level:**
- "design a first model to explain the data" -> "design a first approach to investigate the data"
- "a working first model, not perfection" -> "a working first approach, not perfection"
- "a first experiment script implementing a reasonable baseline model" -> "a first experiment script implementing a reasonable baseline"
**Notebook template (Step 4):**
- "## v00 - Initial Exploration and Baseline Model" -> "## v00 - Initial Exploration and Baseline"
- "### Model Design" -> "### Approach Design"
- "[Describe the model, its assumptions, and why you chose it]" -> "[Describe the approach, its assumptions, and why you chose it]"
- "[What you expect the model to do well and poorly]" -> "[What you expect the approach to do well and poorly]"

**Structural: rewrite Step 2**

Current:
```
## Step 2: Design a First Model
Based on your exploration:
- Identify the key variables and relationships
- Choose a model family appropriate for the data (regression, ODE, etc.)
- Define success criteria that are measurable from the model output
- Keep the model simple for this first version
```

New:
```
## Step 2: Design the First Approach
Based on your exploration:
- Identify the core question and what needs to be measured
- Decide what the experiment script should do and produce
- Define success criteria that are measurable from the script's output
- Start simple - this is a baseline for future iterations to improve upon
```

**Structural: update Step 3 guidance**
- "Implement and fit the model" -> "Implement and evaluate the approach"

**Success criteria guidance:**
- "A mix of required (core model quality) and optional (nice-to-have)" -> "A mix of required (core outcome quality) and optional (nice-to-have)"

#### 2. `src/auto_scientist/prompts/scientist.py`

**Word-level in SCIENTIST_SYSTEM:**
- Strategy type "incremental": "Tune the existing approach (parameters, bounds, priors)" -> "Tune the existing approach (adjust configuration, inputs, or parameters)"
- Success criteria example: 'Good criteria are specific ("R2 > 0.95 for all holds") not vague ("model fits well")' -> 'Good criteria are specific ("error < 10% across all test cases") not vague ("results look good")'

No structural changes needed. The strategy types (incremental/structural/exploratory) and plan structure (what/why/how) are already general.

#### 3. `src/auto_scientist/prompts/report.py`

**Word-level in REPORT_SYSTEM:**
- "autonomous modelling experiment" -> "autonomous scientific investigation"
- "Best Model" -> "Best Approach"
- "from first to best model" -> "from first to best approach"
- "What the model can't do" -> "What the current approach can't do"
- "understand the best model in detail" (REPORT_USER) -> "understand the best approach in detail"

**Structural: generalize section 5**

Current:
```
5. **Best Model** - Full specification: equations, parameters, constraints.
   Include the fitted parameter values.
```

New:
```
5. **Best Approach** - Complete description of what was built and how it works.
   Include key configuration, parameters, or design choices.
```

**Structural: generalize section 6**
- "Best model metrics and diagnostics" -> "Best approach results and diagnostics"

#### 4. `src/auto_scientist/prompts/coder.py`

**Word-level in CODER_SYSTEM results format:**
- "Full specification of the approach (equations, parameters, configuration)" -> "Full specification of the approach and its key design choices"

No other structural changes needed. The Coder prompt is already quite generic.

#### 5. `src/auto_scientist/prompts/analyst.py`

No changes needed. The Analyst prompt is already fully generic - it observes results, evaluates criteria, and reports facts.

#### 6. `domains/spo2/config.py`

- Description: "Model SpO2 dynamics during voluntary breath-holds using a two-stage approach: sensor calibration (latent SaO2 + gamma kernel) then physiology (Severinghaus ODC)." -> "Explain SpO2 dynamics during voluntary breath-holds using a two-stage approach: sensor calibration (latent SaO2 + gamma kernel) then physiology (Severinghaus ODC)."
  - "Model" -> "Explain" in the description opening

#### 7. `domains/spo2/prompts.py`

- "### Model Structure (evolved through v5-v7)" -> "### Approach Structure (evolved through v5-v7)"
- "**Stage A (Sensor Calibration)**: Fit a latent SaO2 shape" -> "**Stage A (Sensor Calibration)**: Estimate a latent SaO2 shape"
- "fit a Severinghaus ODC model to the apnea-only data" -> "identify Severinghaus ODC parameters from the apnea-only data"

#### 8. `docs/architecture.md`

- Title: "Autonomous Scientific Modelling Framework" -> "Autonomous Scientific Investigation Framework"
- "general-purpose autonomous scientific modelling framework" -> "general-purpose autonomous scientific investigation framework"
- "discovers, iterates, and refines models" -> "discovers, iterates, and refines approaches"
- "designs first model, writes v1 script" -> "designs first approach, writes v1 script"
- "first experiment script" -> keep as-is (experiment is neutral)
- "Initial model from discovery phase" (in orchestrator.py) -> "Initial approach from discovery phase"

#### 9. `src/auto_scientist/orchestrator.py`

- Line 206: `hypothesis="Initial model from discovery phase"` -> `hypothesis="Initial approach from discovery phase"`
- Line 128: docstring "design first model" -> "design first approach"
- Line 189: print "exploring dataset and building first model" -> "exploring dataset and designing first approach"

#### 10. `docs/pipeline-visualizer.html`

- "Bi-exponential Decay Model" -> "Bi-exponential Decay" (this is example content in a tooltip, remove the generic "Model" suffix)
- "FULL MODEL" -> "FULL SPECIFICATION"
- "Not a copy of model specs" -> "Not a copy of approach specs"
- "model spec, parameters, metrics" -> "approach spec, parameters, metrics"
- "not a copy of model specs or detailed results" -> "not a copy of approach specs or detailed results"

#### 11. `src/auto_scientist/cli.py`

- "Autonomous scientific modelling framework." -> "Autonomous scientific investigation framework."
- "modelling goal" -> "investigation goal"
- "Run autonomous scientific modelling from raw data." -> "Run autonomous scientific investigation from raw data."

#### 12. `src/auto_scientist/__init__.py`

- "Autonomous scientific modelling framework." -> "Autonomous scientific investigation framework."

#### 13. `pyproject.toml`

- description: "Autonomous scientific modelling framework" -> "Autonomous scientific investigation framework"
- "LLM-driven model discovery, iteration, and refinement" -> "LLM-driven discovery, iteration, and refinement"

#### 14. `README.md`

- Apply the terminology shift table throughout: "modelling" -> "investigation", "models" -> "approaches", "first model" -> "first approach", "best model" -> "best approach"

#### 15. `.claude/CLAUDE.md`

- Project overview: "Autonomous scientific modelling framework" -> "Autonomous scientific investigation framework"
- "discovers, iterates, and refines models" -> "discovers, iterates, and refines approaches"

### What we're NOT changing

- **Architecture:** The pipeline (Ingest -> Discovery -> Iteration -> Report), agent roles, and information boundaries are already general.
- **State machine and data models:** `ExperimentState`, `DomainConfig`, `VersionEntry` are domain-agnostic.
- **Agent implementations:** The Python code in `agents/` uses neutral language.
- **Ingestor, Analyst, and Critic prompts:** Already general enough.
- **The word "experiment":** This is appropriate - every investigation runs experiment scripts.
- **"model" referring to LLM models:** e.g., `critic_models` in orchestrator.py refers to language models, not scientific models. These stay as-is.
- **Pydantic API calls:** e.g., `.model_validate()`, `.model_copy()` - these are framework API names, not domain terms.

## Risks

- **SpO2 domain breakage:** Changing SpO2 prompts/config language could affect agent behavior for this specific domain. Low risk since we're only changing descriptions, not metrics or criteria.
- **Scope creep:** Easy to start rewriting prompts beyond what's needed. Stick to the specific changes listed.

## Verification

1. `uv run ruff check src/ tests/` passes
2. `uv run pytest` passes
3. Manual review: grep for "model" in `prompts/`, `docs/architecture.md`, and `docs/pipeline-visualizer.html` to verify no ML-biased uses remain (domain-specific uses in SpO2 prompts.py are OK where "model" refers to a specific mathematical model, not the framework concept; "model" referring to LLM models is also OK)
4. Architecture doc and pipeline visualizer read coherently with new terminology
