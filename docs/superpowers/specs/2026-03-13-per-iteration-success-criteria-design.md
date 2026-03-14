# Per-Iteration Success Criteria

## Problem

Success criteria are currently static: defined once by Discovery, evaluated by the Analyst, never revised. A real scientist defines testable predictions for each experiment. The spO2 reference implementation (exp_v7_07) shows the pattern: each iteration has its own SUCCESS CRITERIA section with hypothesis-specific pass/fail tests ("Gamma profile has interior minimum", "FL#6 exclusion changes gamma by < 10%").

The framework needs two tiers of criteria:

- **Top-level** (from Discovery/config): define when the investigation is done
- **Per-iteration** (from the Scientist): define whether a specific hypothesis held

## Design

### Scientist produces per-iteration criteria

The Scientist's plan JSON gains a `success_criteria` field:

```json
{
  "hypothesis": "...",
  "strategy": "...",
  "changes": [...],
  "expected_impact": "...",
  "should_stop": false,
  "stop_reason": null,
  "notebook_entry": "...",
  "success_criteria": [
    {
      "name": "Gamma profile has interior minimum",
      "description": "The profile likelihood should show a non-monotone shape with minimum between bounds",
      "metric_key": "gamma_profile_interior",
      "condition": "== true"
    }
  ]
}
```

Fields: `name` (human label), `description` (what it tests and why), `metric_key` (key the script will print), `condition` (human-readable target).

Scientist prompt guidance: "Define 3-8 success criteria that are concrete, measurable predictions of your hypothesis. Each criterion should be testable from the experiment's output."

### Coder implements criteria evaluation in the script

The Coder already receives the full plan JSON. New prompt guidance tells it to print a SUCCESS CRITERIA section at the end of stdout:

```
SUCCESS CRITERIA
----------------
1. Gamma profile has interior minimum:    PASS (gamma_opt=2.50, interior)
2. Free-fit gamma interior to bounds:     PASS (gamma=2.405, bounds [0.8, 4.0])
3. FL#6 exclusion changes gamma by < 10%: FAIL (21.2% change)

Score: 2/3 PASS, 1 FAIL
```

The script computes pass/fail in code, not via LLM judgment. This keeps evaluation honest and matches the existing principle that results.txt is self-compiled.

### Analyst reports both tiers

The Analyst schema gains:

```json
"iteration_criteria_results": [
  {"name": "string", "status": "pass|fail", "measured_value": "string"}
]
```

The Analyst reads the SUCCESS CRITERIA section from results.txt and transcribes it into `iteration_criteria_results`. No re-evaluation needed since the script already computed pass/fail.

`success_score` stays based on top-level criteria only (drives the stopping decision). Per-iteration criteria are informational and flow back to the Scientist via the analysis JSON.

### Critic/Defender challenge the criteria

The plan JSON (already passed to Critic/Defender) now includes `success_criteria`. The Critic prompt gets: "Examine whether the success criteria are well-chosen tests of the hypothesis. Challenge criteria that are too lenient, redundant, or that miss obvious failure modes."

No schema or function signature changes for the debate.

## Data flow summary

```
Scientist  --[plan with success_criteria]--> Critic/Defender (challenge criteria)
Scientist  --[plan with success_criteria]--> Coder (implements evaluation in script)
Script     --[SUCCESS CRITERIA in stdout]--> results.txt
Analyst    --[reads results.txt]-----------> iteration_criteria_results in analysis JSON
Analyst    --[evaluates top-level]---------> criteria_results + success_score in analysis JSON
Scientist  --[reads analysis JSON]---------> sees both tiers, plans next iteration
```

## State persistence

No changes to `state.py` or `VersionEntry`. The iteration criteria results live in the analysis JSON, which the orchestrator passes directly to the next Scientist call. The Scientist's notebook entry can also reference which criteria passed/failed, so the information persists in the notebook for later iterations and synthesis.

## Files changed

- `src/auto_scientist/prompts/scientist.py` - add criteria guidance to system prompt, add field to schema
- `src/auto_scientist/agents/scientist.py` - update SCIENTIST_PLAN_SCHEMA with success_criteria
- `src/auto_scientist/prompts/coder.py` - add SUCCESS CRITERIA printing guidance
- `src/auto_scientist/prompts/analyst.py` - add guidance to read iteration criteria from results
- `src/auto_scientist/agents/analyst.py` - add iteration_criteria_results to ANALYST_SCHEMA
- `src/auto_scientist/prompts/critic.py` (within critic.py) - add criteria challenge guidance
- `docs/pipeline-visualizer.html` - fix visualization issues (see below)

## Visualization fixes

Three issues in the SVG diagram:

1. **Dataset to Coder**: add dashed read arrow from dataset to Coder (Coder needs the data path to write the script)
2. **Persistent store writers**: add amber "writes" arrows showing who produces each store:
   - Scientist writes to lab notebook (via notebook_entry)
   - Orchestrator writes compressed history (built from version entries)
3. **Pre-set stores**: success criteria, domain knowledge, and dataset are set up before the iteration loop. Add a small "pre-set" label to distinguish them from stores that evolve during iteration.
