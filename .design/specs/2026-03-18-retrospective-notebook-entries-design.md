# Design: Retrospective Notebook Entries

## Problem

The Scientist's notebook entries are purely forward-looking: hypothesis, strategy, planned changes. They capture *what will be tried* but not *how the investigation arc is developing*. The kind of retrospective insight that makes a real lab notebook valuable - identifying breakthroughs vs dead ends, explaining failures with hindsight, tracking diagnostic indicators of investigation health - is missing entirely.

## Solution

Enhance the Scientist's prompt to produce richer notebook entries that include retrospective narrative alongside the forward-looking plan.

Each notebook entry should flow naturally through three layers:

1. **Arc reflection** - Where does this iteration sit in the investigation's trajectory? Was the last result a genuine breakthrough (changed understanding), an incremental improvement (refined existing approach), or a dead end? For dead ends, explain *why* with hindsight, not just "metrics didn't improve" but the structural reason (e.g., "the optimizer couldn't distinguish delay from tau_washout because both just slow the curve").

2. **Diagnostic indicators** - Signs of investigation health beyond the score. Are parameters well-identified or compensating for each other? Is the approach structurally adequate or are we curve-fitting? Are we converging, stuck, or circling? These are domain-dependent observations, not a fixed checklist.

3. **Forward plan** - The hypothesis, strategy, and planned changes (what already exists today).

## Scope

Two changes: enrich the Scientist's notebook entries with retrospective narrative, and remove the synthesis module (now redundant).

### Files Modified

- `src/auto_scientist/prompts/scientist.py`: Expand the "Lab Notebook Entry" section in `SCIENTIST_SYSTEM` to include retrospective guidance. Also update the `notebook_entry` instruction in `SCIENTIST_REVISION_SYSTEM` for consistency.
- `src/auto_scientist/synthesis.py`: Delete entirely.
- `src/auto_scientist/orchestrator.py`: Remove `synthesis_interval` parameter, `_notebook_override` field, `_run_synthesis()` method, the synthesis call in `_run_iteration()`, and the override reset. Simplify `_notebook_content()` to just read the file directly.
- `src/auto_scientist/cli.py`: Remove `--synthesis-interval` CLI option.
- Documentation (`docs/architecture.md`, `docs/pipeline-visualizer.html`, `.claude/CLAUDE.md`): Remove synthesis references.

### Why Remove Synthesis

Synthesis was periodic notebook compression to manage context limits. Two things make it unnecessary:
1. Modern context windows (1M tokens for Opus/Gemini, 200K for Sonnet) can hold even very long notebooks comfortably.
2. Retrospective entries are self-compressing: each entry reflects on the full arc, so the latest entry naturally distills the accumulated understanding. The synthesis module would be doing redundant work.

### What Does NOT Change

- The `notebook_entry` field stays a `str` in the output schema - no structural change
- The notebook file format stays as accumulating markdown
- No changes to who reads the notebook (Analyst, Scientist, Critic, Report agent all continue as-is)

## Design Details

### Updated Prompt Section

The "Lab Notebook Entry" section in `SCIENTIST_SYSTEM` (currently lines 47-50) expands from:

> Write a notebook entry documenting your hypothesis, strategy, and planned changes. This becomes the permanent record of your reasoning for this iteration.

To guidance that asks the Scientist to:

- Reflect on the arc before planning forward. Label the previous iteration's outcome: breakthrough (changed understanding), incremental (refined approach), or dead end (abandoned direction). For dead ends, explain the structural reason with hindsight.
- Note diagnostic indicators relevant to the domain. These are signs of investigation health: parameter identifiability, model adequacy, convergence behavior, whether improvements are genuine or artifacts of overfitting/compensation.
- Then plan forward with hypothesis, strategy, and changes as before.
- On the first iteration (v01), skip the arc reflection since there's only the baseline to work from. Focus on the initial assessment of the baseline results and the forward plan.

The entry should read as a continuous narrative, not three separate sections with headers. The format stays `## {version} - [Brief Title]` followed by prose.

The prompt should include a short domain-agnostic example of good retrospection to anchor the quality bar. Examples must avoid domain-specific language per project convention.

### Revision Prompt

The revision prompt (`SCIENTIST_REVISION_SYSTEM`) currently says the notebook_entry should "document what changed from the debate and why." This stays, but adds: also update the arc reflection if the debate changed the assessment of where the investigation stands (e.g., what was thought to be a breakthrough is actually a dead end). The revision entry should NOT repeat the same arc reflection and diagnostics from the initial entry; it focuses on what the debate changed.

## Risks

- **Longer notebook entries** without synthesis could grow context usage over many iterations. In practice, even 100 iterations at 2000 tokens/entry is 200K tokens, well within modern context windows. If this ever becomes a problem, it can be addressed by the model selection for the Scientist, not by a separate compression step.
- **Scientist might produce generic reflections** ("we are making progress"). Mitigation: the prompt guidance should emphasize concrete, domain-specific observations over vague summaries. The examples in the prompt should model what good retrospection looks like.

## Success Criteria

- Notebook entries include backward-looking narrative, not just forward-looking plans
- Dead ends are explained with structural reasoning, not just "metrics didn't improve"
- The investigation arc is visible when reading the notebook sequentially
- Synthesis module cleanly removed with no orphaned references
- No changes to agent boundaries or output schemas
