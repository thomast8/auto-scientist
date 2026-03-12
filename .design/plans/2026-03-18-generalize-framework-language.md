# Generalize Framework Language Implementation Plan

**Goal:** Replace ML/model-fitting terminology with domain-agnostic investigation language across all prompts, docs, and configs.

**Architecture:** Pure text replacement across 12 files (no logic changes). The framework's code and data models are already general; only the human-facing strings need updating. Domain-specific files (e.g., SpO2) are left untouched - they should use their own domain language.

**Tech Stack:** Python string constants, Markdown docs, HTML, TOML

**Spec:** `docs/superpowers/specs/2026-03-18-generalize-framework-language-design.md`

---

### Task 1: Generalize core prompt templates

**Files:**
- Modify: `src/auto_scientist/prompts/discovery.py`
- Modify: `src/auto_scientist/prompts/scientist.py`
- Modify: `src/auto_scientist/prompts/report.py`
- Modify: `src/auto_scientist/prompts/coder.py`

- [ ] **Step 1: Update discovery.py DISCOVERY_SYSTEM**

Replace in `src/auto_scientist/prompts/discovery.py`:
```
"design a first model to explain the data"  ->  "design a first approach to investigate the data"
"a first experiment script implementing a reasonable baseline model"  ->  "a first experiment script implementing a reasonable baseline"
"a working first model, not perfection"  ->  "a working first approach, not perfection"
```

- [ ] **Step 2: Rewrite discovery.py Step 2 (DISCOVERY_USER)**

Replace the entire Step 2 block:
```python
# OLD
## Step 2: Design a First Model
Based on your exploration:
- Identify the key variables and relationships
- Choose a model family appropriate for the data (regression, ODE, etc.)
- Define success criteria that are measurable from the model output
- Keep the model simple for this first version

# NEW
## Step 2: Design the First Approach
Based on your exploration:
- Identify the core question and what needs to be measured
- Decide what the experiment script should do and produce
- Define success criteria that are measurable from the script's output
- Start simple - this is a baseline for future iterations to improve upon
```

- [ ] **Step 3: Update discovery.py Step 3 and success criteria**

```
"Implement and fit the model"  ->  "Implement and evaluate the approach"
"core model quality"  ->  "core outcome quality"
```

- [ ] **Step 4: Update discovery.py notebook template (Step 4)**

```
"Initial Exploration and Baseline Model"  ->  "Initial Exploration and Baseline"
"### Model Design"  ->  "### Approach Design"
"Describe the model, its assumptions, and why you chose it"  ->  "Describe the approach, its assumptions, and why you chose it"
"What you expect the model to do well and poorly"  ->  "What you expect the approach to do well and poorly"
```

- [ ] **Step 5: Update scientist.py**

In `SCIENTIST_SYSTEM`:
```
"Tune the existing approach (parameters, bounds, priors)"  ->  "Tune the existing approach (adjust configuration, inputs, or parameters)"
```
```
'Good criteria are specific ("R2 > 0.95 for all holds") not vague ("model fits\nwell")'  ->  'Good criteria are specific ("error < 10% across all test cases") not vague ("results look\ngood")'
```

- [ ] **Step 6: Update report.py REPORT_SYSTEM**

```
"autonomous modelling experiment"  ->  "autonomous scientific investigation"
"from first to best model"  ->  "from first to best approach"
"What the model can't do"  ->  "What the current approach can't do"
```

Replace section 5:
```python
# OLD
5. **Best Model** - Full specification: equations, parameters, constraints.
   Include the fitted parameter values.

# NEW
5. **Best Approach** - Complete description of what was built and how it works.
   Include key configuration, parameters, or design choices.
```

Replace section 6:
```
"Best model metrics and diagnostics"  ->  "Best approach results and diagnostics"
```

- [ ] **Step 7: Update report.py REPORT_USER**

```
"understand the best model in detail"  ->  "understand the best approach in detail"
```

- [ ] **Step 8: Update coder.py**

In `CODER_SYSTEM` results output section:
```
"Full specification of the approach (equations, parameters, configuration)"  ->  "Full specification of the approach and its key design choices"
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (no logic changes, just string constants)

- [ ] **Step 10: Commit**

```bash
git add src/auto_scientist/prompts/discovery.py src/auto_scientist/prompts/scientist.py src/auto_scientist/prompts/report.py src/auto_scientist/prompts/coder.py
git commit -m "refactor: generalize core prompt language from ML to domain-agnostic"
```

---

### Task 2: Generalize framework identity (code files)

**Files:**
- Modify: `src/auto_scientist/orchestrator.py`
- Modify: `src/auto_scientist/cli.py`
- Modify: `src/auto_scientist/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update orchestrator.py**

```
hypothesis="Initial model from discovery phase"  ->  hypothesis="Initial approach from discovery phase"
```
Docstring (line ~128):
```
"design first model"  ->  "design first approach"
```
Print statement (line ~189):
```
"exploring dataset and building first model"  ->  "exploring dataset and designing first approach"
```

- [ ] **Step 2: Update cli.py**

```
"Autonomous scientific modelling framework."  ->  "Autonomous scientific investigation framework."
"modelling goal"  ->  "investigation goal"
"Run autonomous scientific modelling from raw data."  ->  "Run autonomous scientific investigation from raw data."
```

- [ ] **Step 3: Update __init__.py**

```
"Autonomous scientific modelling framework."  ->  "Autonomous scientific investigation framework."
```

- [ ] **Step 4: Update pyproject.toml**

```
"Autonomous scientific modelling framework - LLM-driven model discovery, iteration, and refinement"
->
"Autonomous scientific investigation framework - LLM-driven discovery, iteration, and refinement"
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/auto_scientist/orchestrator.py src/auto_scientist/cli.py src/auto_scientist/__init__.py pyproject.toml
git commit -m "refactor: generalize framework identity from modelling to investigation"
```

---

### Task 3: Generalize documentation

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/pipeline-visualizer.html`
- Modify: `README.md`
- Modify: `.claude/CLAUDE.md`

- [ ] **Step 1: Update architecture.md**

```
Title: "Autonomous Scientific Modelling Framework"  ->  "Autonomous Scientific Investigation Framework"
"general-purpose autonomous scientific modelling framework"  ->  "general-purpose autonomous scientific investigation framework"
"discovers, iterates, and refines models"  ->  "discovers, iterates, and refines approaches"
"designs first model, writes v1 script"  ->  "designs first approach, writes v1 script"
```

Scan the full file for any other framework-level "model" references and update them. Leave "model" alone where it refers to LLM models (e.g., "critic model"), Pydantic APIs, or the SpO2 domain examples section.

- [ ] **Step 2: Update pipeline-visualizer.html**

```
"Bi-exponential Decay Model"  ->  "Bi-exponential Decay"
"FULL MODEL"  ->  "FULL SPECIFICATION"
"Not a copy of model specs"  ->  "Not a copy of approach specs"
"model spec, parameters, metrics"  ->  "approach spec, parameters, metrics"
"not a copy of model specs or detailed results"  ->  "not a copy of approach specs or detailed results"
```

- [ ] **Step 3: Update README.md**

Apply terminology shift throughout:
```
"Autonomous scientific modelling framework"  ->  "Autonomous scientific investigation framework"
"discovers, iterates, and refines models"  ->  "discovers, iterates, and refines approaches"
"designs a first model"  ->  "designs a first approach"
"best model"  ->  "best approach"
"Model the relationship"  ->  "Investigate the relationship"
```

- [ ] **Step 4: Update .claude/CLAUDE.md**

```
"Autonomous scientific modelling framework"  ->  "Autonomous scientific investigation framework"
"discovers, iterates, and refines models"  ->  "discovers, iterates, and refines approaches"
```

- [ ] **Step 5: Commit**

```bash
git add docs/architecture.md docs/pipeline-visualizer.html README.md .claude/CLAUDE.md
git commit -m "docs: generalize documentation language from modelling to investigation"
```

---

### Task 4: Verify completeness

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Run linter**

Run: `uv run ruff check src/ tests/`
Expected: No errors

- [ ] **Step 3: Grep for remaining ML-biased "model" in prompts**

Run: `grep -rn "model" src/auto_scientist/prompts/ --include="*.py"`

Review each hit. Acceptable uses:
- SpO2 domain references to a specific mathematical model (e.g., "Severinghaus ODC model")
- Pydantic API (`.model_validate()`)
- Already-general uses

Flag any framework-level "model" that should have been "approach" or "investigation".

- [ ] **Step 4: Grep for remaining ML-biased "model" in docs**

Run: `grep -n "model" docs/architecture.md docs/pipeline-visualizer.html`

Same review criteria. LLM model references and SpO2 examples are OK.

- [ ] **Step 5: Grep for "modelling" anywhere**

Run: `grep -rn "modelling" src/ docs/ README.md .claude/ pyproject.toml domains/`

Expected: No hits (all instances should have been changed to "investigation").
