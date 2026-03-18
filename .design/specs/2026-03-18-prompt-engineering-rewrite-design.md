# Prompt Engineering Rewrite - Design Spec

**Date:** 2026-03-18
**Scope:** Clean rewrite of all LLM prompts to follow prompt engineering best practices

## Problem

The current prompts work but were written ad-hoc. They lack consistency in structure, use markdown delimiters, have no few-shot examples, mix positive and negative framing, scatter output format specs across sections, and provide no fallback instructions for missing data. As the framework matures, prompt quality directly affects agent reliability.

## Standards

All prompts will follow these rules (provided by the team):

### Structure
- Section order: Role → Instructions → Examples → Context/Data → Output Format → Task/Query
- For long-context prompts (5K+ tokens): put documents/data at the top, query at the bottom
- Repeat critical instructions in a `<recap>` block at the end (sandwich method)
- Use XML tags as delimiters for multi-section prompts. No mixing XML and Markdown delimiters.

### Role/Persona
- Open every system prompt with: "You are a [domain] [function] system. [Purpose]. [Constraints]."
- Use functional roles ("extraction system") not personality roles ("brilliant analyst")

### Few-Shot Examples
- Include 3-5 examples for any non-trivial task
- Always include one example showing null/missing/empty handling
- Show reasoning BEFORE output in examples, never after
- Place the most typical example last

### Output Format
- Always specify format three ways: verbal description, JSON schema, and example output
- Use provider-native structured output APIs when available (tool-based for Claude)
- State explicit fallback for missing data ("return null", "return empty array")

### Instruction Style
- Positive framing: say what to do, not what to avoid
- Number sequential steps
- Use concrete constraints ("exactly 2-3 sentences") not vague ones ("keep it brief")
- Include motivation for non-obvious rules (Claude generalises from explanations)
- One task per prompt, decompose complex work into sequential calls

### Prompt File Convention
- System prompts and user prompts are defined separately (existing pattern in .py files)
- Static content (role, instructions, examples, format) goes in system prompt
- Dynamic content (documents, query, RAG chunks) goes in user prompt
- Keep static prefixes identical across calls for prompt caching

## Approach

**Clean rewrite (Approach B).** Rewrite every prompt from scratch using the standards as a template. The system has not been tested with the current prompts, so there is no working behavior to preserve. This gives maximum compliance with no regression risk.

## Deliverables

### Files to rewrite (7 prompt files):
1. `src/auto_scientist/prompts/analyst.py` - ANALYST_SYSTEM, ANALYST_USER
2. `src/auto_scientist/prompts/scientist.py` - SCIENTIST_SYSTEM, SCIENTIST_USER, SCIENTIST_REVISION_SYSTEM, SCIENTIST_REVISION_USER
3. `src/auto_scientist/prompts/coder.py` - CODER_SYSTEM, CODER_USER, CODER_NO_PREVIOUS, CODER_HAS_PREVIOUS
4. `src/auto_scientist/prompts/discovery.py` - DISCOVERY_SYSTEM, DISCOVERY_USER
5. `src/auto_scientist/prompts/ingestor.py` - INGESTOR_SYSTEM, INGESTOR_USER
6. `src/auto_scientist/prompts/report.py` - REPORT_SYSTEM, REPORT_USER

### Files to create (1 new file):
7. `src/auto_scientist/prompts/critic.py` - CRITIC_SYSTEM, CRITIC_USER, CRITIC_REFINEMENT_SYSTEM, CRITIC_REFINEMENT_USER, SCIENTIST_DEBATE_SYSTEM, SCIENTIST_DEBATE_USER

### Files to modify (1 agent file):
8. `src/auto_scientist/agents/critic.py` - Replace inline prompt building with imports from `prompts/critic.py`

## XML Skeleton

Every system prompt follows this structure:

```python
AGENT_SYSTEM = """\
<role>
You are a [domain] [function] system. [Purpose]. [Constraints].
</role>

<instructions>
1. [First step]
2. [Second step]
...
</instructions>

<examples>
<example>
<input>
[Sample input matching what appears in the user prompt]
</input>
<reasoning>
[How to process the input - shown BEFORE output]
</reasoning>
<output>
[Expected output in the exact target format]
</output>
</example>
</examples>

<output_format>
[Verbal description of what to produce]

[JSON schema or format specification]

[Fallback rules for missing/null data]
</output_format>

<recap>
[Critical instructions repeated - only for prompts that will exceed ~5K tokens after variable expansion]
</recap>
"""
```

Every user prompt follows this structure:

```python
AGENT_USER = """\
<context>
[Domain knowledge, notebook history, and other background]
</context>

<data>
[Dynamic content: results files, scripts, paths, analysis JSON]
</data>

<task>
[Specific instruction for this invocation]
</task>
"""
```

## Per-Prompt Design

### 1. Analyst (structured JSON output)

**Role:** "You are a scientific observation and measurement system. You read experiment results, examine diagnostic plots, and produce structured JSON assessments. Your output is strictly factual and quantitative."

**Instructions (positively framed):**
1. Read the results file and extract all numeric metrics
2. Examine each plot file using the Read tool; describe trends, patterns, outliers factually
3. For each success criterion, measure the value and compare to target
4. Compare to previous iterations using the notebook; state what improved and regressed with numbers
5. Compute success_score as percentage of weighted criteria passing (required criteria weighted more heavily)
6. Transcribe any SUCCESS CRITERIA section from experiment output into iteration_criteria_results

**Examples (3, using water quality monitoring domain):**
1. Normal case: 2 criteria pass, 1 fails, 3 plots examined, comparison to previous iteration shows improvement
2. Null/empty case: first iteration (no previous to compare), one criterion "unable_to_measure" because the script crashed before producing that metric, no plots generated
3. Most typical case (last): mixed results, some improvement, some regression, partial criteria pass

**Output format:** Verbal description, full JSON schema with types, example JSON, fallbacks:
- No plots → empty `observations` list
- No previous iteration → empty `improvements` and `regressions` lists
- Metric not found in output → status: "unable_to_measure", measured_value: null
- No SUCCESS CRITERIA section → empty `iteration_criteria_results` list

**Recap:** "Report only what you observe. Every claim references a specific number from the results. Produce valid JSON with all required keys."

### 2. Scientist (structured JSON output)

**Role:** "You are a scientific hypothesis and planning system. You analyze experimental assessments, formulate hypotheses, and produce detailed implementation plans as JSON. You plan from results, observations, and your notebook. A separate Coder implements your plans."

**Instructions:**
1. Read the analysis and notebook to understand the current state
2. Reflect on the investigation arc (breakthrough / incremental improvement / dead end) with specific reasoning
3. Formulate a hypothesis about what to change and why
4. Choose a strategy: incremental (tune existing approach), structural (fundamental change), exploratory (fresh direction)
5. Create prioritized changes, each with what/why/how and priority 1-3
6. Define 3-8 success criteria as concrete, measurable predictions of the hypothesis
7. Write a notebook entry as a continuous narrative under `## {version} - [Brief Title]`
8. Set should_stop=true when all required criteria pass, or when stagnation persists after structural changes

**Examples (4, covering all strategy types + stop + v00):**
1. Incremental: crop yield prediction, adjusting soil sampling depth. Previous iteration was an incremental improvement. Criteria are specific numeric thresholds.
2. Structural: traffic flow analysis, switching from regression to simulation. Previous iteration was a dead end with structural reasoning for why.
3. Exploratory: weather prediction, abandoning physics model for data-driven approach after stagnation.
4. v00 / first iteration: no analysis JSON yet, just discovery findings in notebook. Plans first experiment from exploration.
5. Most typical (last): should_stop=true, all criteria pass, stop_reason explains convergence.

**Output format:** Verbal, JSON schema with enums and constraints, example JSON, fallbacks:
- First iteration (no analysis) → base plan on discovery findings in notebook
- No domain_knowledge → plan from data patterns alone
- Analysis shows script crash → plan must address the crash before advancing the investigation

**Recap:** "Output valid JSON with all required keys. Each change has what/why/how/priority. Success criteria are testable numeric predictions, not subjective assessments."

### 3. Scientist Revision (structured JSON output)

**Role:** "You are a scientific plan revision system. You incorporate feedback from a critic debate into a revised experiment plan. You produce a complete revised plan as JSON, not a diff."

**Instructions:**
1. Read the original plan and debate transcript
2. Identify which critique points are valid and which were adequately addressed
3. Accept valid critique: adjust hypothesis, strategy, changes, or criteria accordingly
4. Reject points already resolved in the debate, with brief reasoning
5. If the debate revealed fundamental issues, change strategy or hypothesis entirely
6. Write a notebook_entry documenting what changed from the original plan and why
7. Output a complete revised plan (all fields, not just changes)

**Examples (2):**
1. Revision accepting critique: critic identified a missing confounding variable, plan adds a control. Criteria adjusted. Notebook entry explains the shift.
2. Revision mostly rejecting: critic suggested an overly complex approach, scientist explains in notebook why the simpler plan is better. Minor criterion tweak accepted.

**Output format:** Same JSON schema as Scientist. Verbal + schema + example + fallbacks:
- Empty debate transcript → return original plan unchanged
- Debate only about criteria → adjust criteria, keep hypothesis/changes intact

**Recap:** "Output a complete plan with all required keys. notebook_entry documents what the debate changed, not a repeat of the original reflection."

### 4. Coder (tool-using agent, stdout output)

**Role:** "You are a scientific software implementation system. You translate experiment plans into complete, self-contained, runnable Python scripts. You follow plans faithfully without making strategic decisions."

**Instructions:**
1. Read the previous script (if any) to understand current implementation
2. Implement all priority-1 (must-do) changes from the plan
3. Implement priority-2 (should-do) changes if feasible
4. Priority-3 (nice-to-have) changes are optional
5. The script must be completely self-contained: all imports at top, all code in one file, load data directly from the provided path
6. Only use allowed dependencies (provided in prompt)
7. Print structured results to stdout (header, data summary, approach spec, changes, parameters, metrics, SUCCESS CRITERIA section, summary)
8. Save diagnostic plots as PNGs in the script directory
9. The SUCCESS CRITERIA section must be computed by the script in code, with pass/fail evaluated programmatically
10. Verify syntax after writing

**Motivation for key rules:**
- Self-contained scripts ensure reproducibility: anyone can rerun a version without the framework installed
- SUCCESS CRITERIA computed in code (not hardcoded) ensures honest evaluation of whether the hypothesis held

**Examples:** None (Coder is a tool-using agent that produces files, not structured JSON. The instructions are procedural and the output format is specified concretely via the SUCCESS CRITERIA template.)

**Output format (SUCCESS CRITERIA in stdout):**
```
SUCCESS CRITERIA
----------------
1. {name}: PASS ({measured_value})
2. {name}: FAIL ({measured_value}, expected {condition})

Score: X/Y PASS, Z FAIL
```

**No recap needed** (prompt stays under 5K tokens).

### 5. Discovery (tool-using agent, writes files)

**Role:** "You are a scientific data exploration and experiment design system. You explore datasets, understand their structure, and design conceptual approaches for investigation. You document findings and define success criteria. A separate Scientist and Coder will plan and implement the actual experiments."

**Instructions:**
1. Examine the dataset: file type, schema, row counts, distributions, summary statistics, correlations, anomalies
2. Create exploratory plots to visualize the data
3. Identify the core question and what needs to be measured
4. Describe what a good first experiment should try and why
5. Define 5-10 success criteria that are measurable from experiment stdout output, specific (numeric targets where possible), and a mix of required and optional
6. Write initial lab notebook entry (#0) documenting exploration, approach design, and expected behavior
7. Write domain config JSON file

**Examples:** None (tool-using agent producing files, not structured JSON).

**Output format:** Two files:
- Lab notebook (markdown, specified template)
- Domain config JSON (schema provided in prompt with all fields documented)

**No recap needed** (prompt is under 5K tokens).

### 6. Ingestor (tool-using agent, writes files)

**Role:** "You are a data canonicalization system. You inspect raw data, understand its structure, and produce a clean canonical dataset for downstream scientific analysis. You write a conversion script for auditability."

**Instructions:**
1. Examine raw data: file types, schema, data types, encodings, row counts, sample rows
2. In interactive mode: ask the human about ambiguities (column semantics, relationships, units) using AskUserQuestion. In autonomous mode: make best-effort decisions and log every assumption.
3. Choose canonical format based on data characteristics: SQLite for relational, CSV for simple flat (<100MB), Parquet for large single-table (>100MB). If data is already clean, copy through with minimal transformation.
4. Write a self-contained conversion script to {data_dir}/ingest.py
5. Run the script to produce canonical output
6. Update lab notebook with findings, assumptions, and output summary
7. Present a final summary: input received, output produced, decisions made

**Examples:** None (tool-using, procedural).

**No recap needed.**

### 7. Report (tool-using agent, writes markdown)

**Role:** "You are a scientific report writing system. You produce comprehensive final reports summarizing autonomous scientific investigations. Reports are accessible to readers with domain knowledge but no familiarity with the specific experiment."

**Instructions:**
1. Read the best version's script and results using Read tool
2. Read other notable versions (paradigm shifts, regressions) for the journey section
3. Write the report following the 10-section structure (provided)
4. Use precise, quantitative language; reference specific numbers from results
5. Include a version comparison table covering all versions

**Report sections** (same 10 sections as current, kept as a numbered list in `<instructions>`).

**Examples:** None (produces a full markdown report, not structured JSON).

**No recap needed.**

### 8. Critic Prompts (NEW: prompts/critic.py)

Three prompt pairs extracted from agents/critic.py:

#### CRITIC_SYSTEM / CRITIC_USER
**Role:** "You are a scientific critique system. You challenge experiment plans, propose alternative hypotheses, and identify blind spots. You have web search available to verify claims and look up relevant methods."

**Instructions:**
1. Challenge the proposed hypothesis and strategy with specific reasoning
2. Propose alternative hypotheses the scientist has not considered
3. Evaluate whether the success criteria are well-chosen tests of the hypothesis (too lenient? redundant? missing failure modes?)
4. Assess whether a different strategy type is needed (incremental/structural/exploratory)
5. Evaluate feasibility and expected impact
6. Use web search to verify scientific claims and check methods

**Examples:** None (free-form critique, not structured JSON).

#### SCIENTIST_DEBATE_SYSTEM / SCIENTIST_DEBATE_USER
**Role:** "You are a scientist defending your experiment plan during a critique debate. You have web search available to support your claims."

**Instructions:**
1. Defend well-motivated choices with specific reasoning
2. Acknowledge valid critique points and suggest adjustments
3. Clarify misunderstandings about the plan
4. Be concise and substantive; focus on the most important points

#### CRITIC_REFINEMENT_SYSTEM / CRITIC_REFINEMENT_USER
**Role:** "You are a scientific critique system refining your critique after hearing the scientist's defense."

**Instructions:**
1. Drop points the scientist adequately addressed
2. Sharpen points where the defense was weak or evasive
3. Add new observations prompted by the defense
4. Produce a final, self-contained, actionable critique

## Agent File Changes

### agents/critic.py

Replace the three `_build_*_prompt` functions with imports from `prompts/critic.py`. The functions become thin wrappers that format the template strings with the dynamic content (plan JSON, notebook, domain knowledge, prior critique, defense).

Current:
```python
def _build_critic_prompt(plan, notebook_content, domain_knowledge):
    parts = ["You are a scientific critic...", ...]
    return "\n".join(parts)
```

New:
```python
from auto_scientist.prompts.critic import CRITIC_SYSTEM, CRITIC_USER

def _build_critic_prompt(plan, notebook_content, domain_knowledge):
    system = CRITIC_SYSTEM
    user = CRITIC_USER.format(
        domain_knowledge=domain_knowledge or "(none provided)",
        notebook_content=notebook_content or "(empty)",
        plan_json=json.dumps(plan, indent=2),
    )
    return f"{system}\n\n{user}"
```

Note: Since the critic uses plain API calls (not Claude agent), system and user are concatenated into a single prompt string. The structural separation is maintained in the template for clarity and caching.

## Example Domains for Few-Shots

To keep examples domain-agnostic and broadly representative:

| Example Domain | Why It's Useful |
|---|---|
| Water quality monitoring | Numeric thresholds, spatial sampling, clear pass/fail criteria |
| Crop yield prediction | Time-series data, multiple input variables, seasonal patterns |
| Traffic flow analysis | Simulation vs. regression, throughput metrics, structural approach changes |
| Weather station calibration | Sensor data, calibration curves, null/missing readings common |

Each example uses a different domain to demonstrate that the framework is general-purpose.

## Testing

- Existing tests in `tests/test_critic.py` verify information boundaries (no analysis or script in prompts, no compressed history). These must still pass after the rewrite.
- Run `uv run pytest` to verify no regressions.
- Manual review of each prompt against the standards checklist.

## Out of Scope

- Changing agent logic or orchestrator flow
- Modifying how prompts are consumed (the agent .py files just import and format, that stays the same)
- Adding prompt caching infrastructure (future work, but the static prefix pattern enables it)
- Domain-specific prompt overrides (future: domains/ could provide custom prompt fragments)
