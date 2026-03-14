# Per-Iteration Success Criteria Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-iteration success criteria that the Scientist defines and the Coder evaluates in the experiment script, alongside the existing top-level criteria.

**Architecture:** The Scientist's plan JSON gains a `success_criteria` array. The Coder prints a SUCCESS CRITERIA section in stdout. The Analyst reads both tiers. The Critic challenges whether criteria are well-chosen.

**Tech Stack:** Python, claude-code-sdk, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-03-13-per-iteration-success-criteria-design.md`

---

### Task 1: Add success_criteria to Scientist plan schema

**Files:**
- Modify: `src/auto_scientist/agents/scientist.py:22-54`
- Modify: `src/auto_scientist/prompts/scientist.py`

- [ ] **Step 1: Add success_criteria to SCIENTIST_PLAN_SCHEMA**

In `src/auto_scientist/agents/scientist.py`, add to the `properties` dict after `notebook_entry`:

```python
"success_criteria": {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "metric_key": {"type": "string"},
            "condition": {"type": "string"},
        },
        "required": ["name", "description", "metric_key", "condition"],
    },
},
```

Add `"success_criteria"` to the `required` list.

- [ ] **Step 2: Add criteria guidance to SCIENTIST_SYSTEM prompt**

In `src/auto_scientist/prompts/scientist.py`, add after the "Lab Notebook Entry" section and before the JSON keys list:

```
## Success Criteria

Define 3-8 success criteria that are concrete, measurable predictions of your
hypothesis. Each criterion should be testable from the experiment's output.
Good criteria are specific ("R2 > 0.95 for all holds") not vague ("model fits
well"). The experiment script will evaluate these and print pass/fail results.

For each criterion, provide:
- name: human-readable label
- description: what it tests and why
- metric_key: the key the script will use to report the measured value
- condition: human-readable target (e.g., "> 0.95", "== true", "< 10%")
```

Add `success_criteria` to the JSON keys list at the end of the system prompt:

```
- success_criteria: list[object] (each with: name, description, metric_key, condition)
```

- [ ] **Step 3: Run tests to verify nothing broke**

Run: `uv run pytest tests/ -v`
Expected: all existing tests pass (no tests directly exercise scientist schema yet)

- [ ] **Step 4: Commit**

```
feat: add success_criteria to Scientist plan schema
```

---

### Task 2: Add SUCCESS CRITERIA guidance to Coder prompt

**Files:**
- Modify: `src/auto_scientist/prompts/coder.py:33-41`

- [ ] **Step 1: Add SUCCESS CRITERIA section to CODER_SYSTEM Results Output**

In `src/auto_scientist/prompts/coder.py`, replace items 7-8 in the Results Output list with:

```
7. Success criteria evaluation: the plan includes a `success_criteria` list.
   For EACH criterion, compute the measured value in code and print a
   SUCCESS CRITERIA section at the end of stdout in this exact format:

   SUCCESS CRITERIA
   ----------------
   1. {name}: PASS ({measured_value})
   2. {name}: FAIL ({measured_value}, expected {condition})

   Score: X/Y PASS, Z FAIL

   The pass/fail evaluation MUST be computed by the script in code, not
   hardcoded. This is the honest record of whether the hypothesis held.
8. Summary of findings
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/ -v`
Expected: all pass

- [ ] **Step 3: Commit**

```
feat: add SUCCESS CRITERIA printing guidance to Coder prompt
```

---

### Task 3: Add iteration_criteria_results to Analyst

**Files:**
- Modify: `src/auto_scientist/agents/analyst.py:26-56`
- Modify: `src/auto_scientist/prompts/analyst.py`

- [ ] **Step 1: Add iteration_criteria_results to ANALYST_SCHEMA**

In `src/auto_scientist/agents/analyst.py`, add to the `properties` dict after `observations`:

```python
"iteration_criteria_results": {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "status": {"type": "string", "enum": ["pass", "fail"]},
            "measured_value": {"type": "string"},
        },
        "required": ["name", "status", "measured_value"],
    },
},
```

Add `"iteration_criteria_results"` to the `required` list.

- [ ] **Step 2: Add iteration criteria guidance to ANALYST_SYSTEM**

In `src/auto_scientist/prompts/analyst.py`, add after the existing JSON keys list:

```
- iteration_criteria_results: list[object] (each with: name, status, measured_value)
  The experiment output may include a SUCCESS CRITERIA section with per-iteration
  criteria defined by the Scientist. Transcribe these results into
  iteration_criteria_results. These are separate from the top-level success
  criteria and do not affect the success_score.
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/ -v`
Expected: all pass

- [ ] **Step 4: Commit**

```
feat: add iteration_criteria_results to Analyst schema
```

---

### Task 4: Add criteria challenge to Critic prompt

**Files:**
- Modify: `src/auto_scientist/agents/critic.py:144-181`
- Modify: `tests/test_critic.py`

- [ ] **Step 1: Update _build_critic_prompt**

In `src/auto_scientist/agents/critic.py`, add item 5 to the "Your Task" list in `_build_critic_prompt()`:

```python
"5. Whether the success criteria are well-chosen tests of the hypothesis",
"   (too lenient? redundant? missing obvious failure modes?)",
```

- [ ] **Step 2: Add success_criteria to the plan fixture in tests**

In `tests/test_critic.py`, update the `plan` fixture to include:

```python
"success_criteria": [
    {
        "name": "Convergence improves",
        "description": "Final loss should decrease with lower learning rate",
        "metric_key": "final_loss_decreased",
        "condition": "== true",
    }
],
```

- [ ] **Step 3: Add a test for criteria in critic prompt**

In `tests/test_critic.py`, add to `TestRunDebate`:

```python
@pytest.mark.asyncio
async def test_criteria_in_critic_prompt(self, base_kwargs):
    """Critic prompt includes success criteria from the plan."""
    with patch(
        "auto_scientist.agents.critic.query_openai",
        new_callable=AsyncMock,
        return_value="Critique",
    ) as mock_openai:
        await run_debate(**base_kwargs, max_rounds=1)

    critic_prompt = mock_openai.call_args[0][1]
    assert "success_criteria" in critic_prompt
    assert "Convergence improves" in critic_prompt
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_critic.py -v`
Expected: all pass, including the new test

- [ ] **Step 5: Commit**

```
feat: add criteria challenge guidance to Critic prompt
```

---

### Task 5: Fix visualization issues in pipeline-visualizer.html

**Files:**
- Modify: `docs/pipeline-visualizer.html`

- [ ] **Step 1: Add dataset-to-Coder arrow**

Add a dashed blue read arrow from dataset (bottom) to Coder (middle):

```svg
<!-- Dataset → Coder -->
<path class="arrow-line" data-agents="coder" data-artifacts="dataset"
      d="M 1078,518 C 1060,450 900,380 880,340"
      fill="none" stroke="#4e8fd4" stroke-width="1" stroke-opacity="0.4" stroke-dasharray="4 3"
      marker-end="url(#ah-blue-dim)"/>
```

- [ ] **Step 2: Add amber "writes" arrows for persistent stores**

Add solid amber arrows showing who writes to each evolving store:

```svg
<!-- Scientist writes to lab notebook -->
<path class="arrow-line" data-agents="scientist" data-artifacts="notebook"
      d="M 367,338 C 367,420 310,480 297,516"
      fill="none" stroke="#d4a24e" stroke-width="1.5" marker-end="url(#ah-amber)"/>

<!-- Orchestrator writes compressed history (from Runner side) -->
<path class="arrow-line" data-agents="runner" data-artifacts="history"
      d="M 1117,338 C 1100,420 820,480 786,516"
      fill="none" stroke="#d4a24e" stroke-width="1.2" stroke-opacity="0.5" marker-end="url(#ah-amber)"/>
```

- [ ] **Step 3: Add "pre-set" labels to static stores**

Add small text labels below success criteria, domain knowledge, and dataset:

```svg
<text x="95" y="562" text-anchor="middle" fill="#5a6178" font-size="7" opacity="0.5">pre-set</text>
<text x="540" y="562" text-anchor="middle" fill="#5a6178" font-size="7" opacity="0.5">pre-set</text>
<text x="1098" y="562" text-anchor="middle" fill="#5a6178" font-size="7" opacity="0.5">pre-set</text>
```

- [ ] **Step 4: Open in browser and verify**

Run: `open docs/pipeline-visualizer.html`
Check: dataset arrow reaches Coder, notebook has amber write arrow from Scientist, pre-set labels visible on static stores.

- [ ] **Step 5: Commit**

```
fix: correct data flow arrows in pipeline visualization
```

---

### Task 6: Update docs

**Files:**
- Modify: `docs/architecture.md` (information boundaries table)
- Modify: `.claude/CLAUDE.md` (architecture summary)
- Modify: `TODO.md`

- [ ] **Step 1: Update architecture.md**

Add per-iteration success criteria to the agent descriptions and data flow section.

- [ ] **Step 2: Update CLAUDE.md**

Add note about two-tier criteria to the Architecture Summary.

- [ ] **Step 3: Update TODO.md**

Move "Per-iteration success criteria" to Completed with today's date.

- [ ] **Step 4: Commit**

```
docs: update architecture for per-iteration success criteria
```
