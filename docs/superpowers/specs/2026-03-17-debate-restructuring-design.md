# Debate Restructuring and Compressed History Removal

## Problem

Three design issues in the current iteration loop:

1. **Defender is a fake Scientist.** The Defender is a fresh Claude API call told "you are the scientist who formulated this plan" but has none of the reasoning context from actually having made it. The Scientist and Defender are semantically the same role.

2. **Critique goes to the Coder.** The Coder receives the original plan + the critique and must reconcile them. But reconciling critique with a plan is a strategic decision, violating the principle that the Coder is a pure implementer.

3. **Compressed history is redundant.** It's a one-line-per-iteration summary built from state entries. The lab notebook already contains the same information with richer context, and the synthesis step condenses the notebook when it grows too long.

## Design

### New iteration flow

```
Analyst
  → Scientist (initial plan)
  → Debate: Critic ↔ Scientist for N rounds
  → Scientist (revised plan incorporating the debate)
  → Coder (implements revised plan only, no critique)
```

### Scientist replaces Defender in the debate loop

The debate loop becomes Critic ↔ Scientist instead of Critic ↔ Defender:

1. Scientist produces initial plan (`query()` call, structured JSON output)
2. Critic critiques the plan (provider API call, same as today)
3. Scientist responds, defending choices or conceding points (`query_anthropic()` call, same mechanism as current Defender)
4. Critic refines (provider API call)
5. Scientist responds again
6. ... for N configured rounds
7. Scientist produces revised plan (`query()` call, structured JSON output, receives initial plan + full debate transcript)

Steps 2-6 are mechanically identical to the current debate loop, with "Defender" renamed to "Scientist" in the prompts. Step 7 is a new final call.

The Scientist's debate responses use `query_anthropic()` (same as current Defender). The prompt changes from "you are a defender" to "you are the scientist who formulated this plan." This gives the Critic a more authentic counterpart.

### Scientist revision call

After the debate, a second `query()` call to the Scientist:

- Input: initial plan JSON + full debate transcript + analysis JSON + notebook + domain knowledge
- Output: revised plan JSON (same schema as the initial plan)
- The Scientist can accept valid critique, reject bad points, adjust success criteria, change strategy, or even change the hypothesis entirely

New prompt templates: `SCIENTIST_REVISION_SYSTEM` and `SCIENTIST_REVISION_USER`.

The revised plan is what the Coder receives. The Coder never sees the debate transcript or the critique.

### Debate return value

`run_debate()` currently returns `list[dict[str, str]]` (one entry per critic with `model` and `critique`). It needs to also return the full debate transcript (all rounds of critic + scientist responses) so the Scientist revision call can see the entire discussion.

New return: `list[dict]` where each entry has `model`, `critique` (final refined), and `transcript` (list of `{"role": "critic"|"scientist", "content": str}`).

### Compressed history removal

Remove `compressed_history` as a parameter and artifact everywhere:

- Delete `src/auto_scientist/history.py` and `tests/test_history.py`
- Remove `compressed_history` parameter from: `run_debate()`, `run_synthesis()`, `run_report()`, and all prompt builder functions
- Agents that received compressed history now rely on the notebook (potentially synthesized) for iteration context

The dead ends list and best version tracking already live in `ExperimentState`. The orchestrator can inject `state.dead_ends` and `state.best_version` into prompts where useful, without a separate compressed history artifact.

### Coder simplification

The Coder no longer receives the critique. It gets only the revised plan (which already incorporates the debate). The prompt drops any critique references.

## Data flow summary

```
Analyst    --[analysis JSON]----> Scientist (initial plan)
Scientist  --[initial plan]-----> Critic (critique)
Critic     --[critique]---------> Scientist (debate response)
  ... N rounds ...
Scientist  --[revised plan]-----> Coder (implements)
Coder      --[experiment.py]----> Runner
```

## Files changed

**Deleted:**
- `src/auto_scientist/history.py`
- `tests/test_history.py`

**Modified (debate restructuring):**
- `src/auto_scientist/agents/critic.py` - rename Defender to Scientist in prompts, remove `compressed_history` parameter, return debate transcript
- `src/auto_scientist/agents/scientist.py` - add `run_scientist_revision()` function
- `src/auto_scientist/prompts/scientist.py` - add `SCIENTIST_REVISION_SYSTEM` and `SCIENTIST_REVISION_USER`
- `src/auto_scientist/orchestrator.py` - wire new flow, remove compressed_history, Coder gets revised plan only

**Modified (compressed history removal):**
- `src/auto_scientist/synthesis.py` - remove `compressed_history` parameter
- `src/auto_scientist/agents/report.py` - remove compressed_history
- `src/auto_scientist/prompts/report.py` - remove `{compressed_history}` section

**Modified (coder simplification):**
- `src/auto_scientist/agents/coder.py` - remove critique from inputs (if present)
- `src/auto_scientist/prompts/coder.py` - remove critique references

**Tests:**
- `tests/test_critic.py` - update for renamed prompts, removed compressed_history, transcript return

**Docs:**
- `docs/architecture.md`, `.claude/CLAUDE.md`, `TODO.md`
- `docs/pipeline-visualizer.html` - remove compressed history node, rename Defender to Scientist, update arrows
