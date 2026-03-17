# Debate Restructuring Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge Defender into Scientist (Scientist defends its own plan in the debate loop, then revises), route critique back to Scientist instead of Coder, and eliminate compressed history.

**Architecture:** The debate loop becomes Critic ↔ Scientist (replacing Critic ↔ Defender). After the debate, a second Scientist `query()` call produces a revised plan. The Coder receives only the revised plan. Compressed history is removed everywhere; agents rely on the notebook (potentially synthesized).

**Tech Stack:** Python, claude-code-sdk, pytest

**Spec:** `docs/superpowers/specs/2026-03-17-debate-restructuring-design.md`

---

### Task 1: Remove compressed history from synthesis

**Files:**
- Modify: `src/auto_scientist/synthesis.py`

- [ ] **Step 1: Remove compressed_history from SYNTHESIS_PROMPT**

In `src/auto_scientist/synthesis.py`, replace the prompt template. Remove the `## Compressed History` / `{compressed_history}` section entirely. The notebook already has the information.

```python
SYNTHESIS_PROMPT = """\
You are a scientific editor. Your task is to condense a lab notebook into a
concise investigation narrative that captures the essential context.

## Lab Notebook (Full)
{notebook_content}

## Domain Knowledge
{domain_knowledge}

## Your Task
Produce a condensed narrative (target: 30-50% of the original notebook length)
that preserves:
1. The overall goal and what has been tried
2. Key hypotheses and whether they panned out
3. Major structural changes and their outcomes
4. Dead ends and why they were abandoned
5. The current state: what works, what doesn't, what's next

Write in a direct, factual style. Use section headers for clarity.
Do NOT include raw metric tables or full parameter lists; summarize trends.
This narrative will replace the full notebook in agent prompts, so it must
contain enough context for a scientist to plan the next iteration.
"""
```

- [ ] **Step 2: Remove compressed_history from run_synthesis()**

Update the function signature and body:

```python
async def run_synthesis(
    notebook_content: str,
    domain_knowledge: str = "",
    model: str = "claude-sonnet-4-6",
) -> str:
    """Condense the lab notebook into a concise narrative.

    Args:
        notebook_content: Full text of the lab notebook.
        domain_knowledge: Domain-specific context.
        model: Anthropic model to use.

    Returns:
        Condensed narrative string.
    """
    prompt = SYNTHESIS_PROMPT.format(
        notebook_content=notebook_content,
        domain_knowledge=domain_knowledge or "(none provided)",
    )
    return await query_anthropic(model, prompt)
```

- [ ] **Step 3: Update the module docstring**

Replace line 4 `condensing the full notebook + compressed history` with `condensing the full notebook`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/ -v`
Expected: all pass (synthesis has no dedicated tests, but nothing should break)

- [ ] **Step 5: Commit**

```
refactor: remove compressed_history from synthesis
```

---

### Task 2: Remove compressed history from report

**Files:**
- Modify: `src/auto_scientist/agents/report.py`
- Modify: `src/auto_scientist/prompts/report.py`

- [ ] **Step 1: Update REPORT_USER prompt**

In `src/auto_scientist/prompts/report.py`, remove the compressed history section from `REPORT_USER`:

```python
REPORT_USER = """\
## Experiment Metadata
- Domain: {domain}
- Goal: {goal}
- Total iterations: {total_iterations}
- Best version: {best_version} (score: {best_score})

## Lab Notebook
{notebook_content}

## Instructions
1. Use Glob to find the best version's directory, then Read its results file
   and script to understand the best model in detail
2. If there are other notable versions (paradigm shifts, regressions), read
   their results too for the journey section
3. Write the report to: {report_path}
"""
```

- [ ] **Step 2: Update run_report()**

In `src/auto_scientist/agents/report.py`, remove the `build_compressed_history` import and call:

```python
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

from auto_scientist.prompts.report import REPORT_SYSTEM, REPORT_USER
from auto_scientist.state import ExperimentState
```

Remove `compressed_history = build_compressed_history(state)` (line 32) and remove `compressed_history=compressed_history` from the `REPORT_USER.format()` call (line 42).

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/ -v`

- [ ] **Step 4: Commit**

```
refactor: remove compressed_history from report
```

---

### Task 3: Delete history.py

**Files:**
- Delete: `src/auto_scientist/history.py`
- Delete: `tests/test_history.py`

- [ ] **Step 1: Verify no remaining imports**

Run: `uv run ruff check src/ tests/` after deletion to confirm nothing imports `history`.

Check the orchestrator still imports it (it does, in `_run_critic` and `_run_synthesis`). Those will be fixed in Task 5. For now, just delete the files and note the expected breakage.

- [ ] **Step 2: Delete both files**

Remove `src/auto_scientist/history.py` and `tests/test_history.py`.

- [ ] **Step 3: Commit**

```
refactor: delete history.py (compressed history removed)
```

---

### Task 4: Restructure debate loop (Scientist replaces Defender)

**Files:**
- Modify: `src/auto_scientist/agents/critic.py`
- Modify: `tests/test_critic.py`

- [ ] **Step 1: Update run_debate() signature and return type**

Remove `compressed_history` parameter. Change return type to include transcript. Rename `defender_model` to `scientist_model`:

```python
async def run_debate(
    critic_specs: list[str],
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
    max_rounds: int = 2,
    scientist_model: str = "claude-sonnet-4-6",
) -> list[dict[str, Any]]:
```

- [ ] **Step 2: Update debate loop body to track transcript**

```python
    if not critic_specs:
        return []

    critiques = []
    for spec in critic_specs:
        provider, model = parse_critic_spec(spec)
        transcript: list[dict[str, str]] = []

        # Round 1: initial critique
        critic_prompt = _build_critic_prompt(
            plan, notebook_content, domain_knowledge
        )
        critique_text = await _query_critic(provider, model, critic_prompt)
        transcript.append({"role": "critic", "content": critique_text})

        # Rounds 2+: scientist responds, critic refines
        for _ in range(1, max_rounds):
            scientist_response = await query_anthropic(
                scientist_model,
                _build_scientist_response_prompt(
                    plan=plan,
                    notebook_content=notebook_content,
                    domain_knowledge=domain_knowledge,
                    critique=critique_text,
                ),
                web_search=True,
            )
            transcript.append({"role": "scientist", "content": scientist_response})
            refinement_prompt = _build_critic_refinement_prompt(
                plan=plan,
                notebook_content=notebook_content,
                domain_knowledge=domain_knowledge,
                critique=critique_text,
                defense=scientist_response,
            )
            critique_text = await _query_critic(provider, model, refinement_prompt)
            transcript.append({"role": "critic", "content": critique_text})

        critiques.append({
            "model": spec,
            "critique": critique_text,
            "transcript": transcript,
        })

    return critiques
```

- [ ] **Step 3: Rename _build_defender_prompt to _build_scientist_response_prompt**

Update the function name and docstring. Remove `compressed_history` parameter. Update the prompt text:

```python
def _build_scientist_response_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    critique: str,
) -> str:
    """Build the prompt for the Scientist responding to a critique during debate."""
    parts = [
        "You are the scientist who formulated this plan. A critic has reviewed it.",
        "Respond to their critique:",
        "- Defend choices that are well-motivated (explain your reasoning).",
        "- Acknowledge valid points and suggest how to address them.",
        "- Clarify any misunderstandings the critic may have about your plan.",
        "Be concise and substantive. Focus on the most important points.",
        "You have web search available to back up your claims with references.",
        "",
        "## Domain Knowledge",
        domain_knowledge or "(none provided)",
        "",
        "## Lab Notebook",
        notebook_content or "(empty)",
        "",
        "## Your Plan",
        json.dumps(plan, indent=2),
        "",
        "## Critic's Feedback",
        critique,
        "",
        "## Your Response",
        "Address each major point from the critic. Be honest about weaknesses",
        "but defend the reasoning behind your plan where it is sound.",
    ]
    return "\n".join(parts)
```

- [ ] **Step 4: Update _build_critic_prompt - remove compressed_history**

```python
def _build_critic_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
) -> str:
```

Remove the `## Experiment History` / `compressed_history` section from the prompt parts.

- [ ] **Step 5: Update _build_critic_refinement_prompt - remove compressed_history**

Same pattern: remove `compressed_history` parameter and the `## Experiment History` section.

- [ ] **Step 6: Update run_critic() backward-compat wrapper**

Remove `compressed_history` parameter:

```python
async def run_critic(
    critic_specs: list[str],
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
) -> list[dict[str, Any]]:
    return await run_debate(
        critic_specs=critic_specs,
        plan=plan,
        notebook_content=notebook_content,
        domain_knowledge=domain_knowledge,
        max_rounds=1,
    )
```

- [ ] **Step 7: Update module docstring**

Remove references to "compressed history" and "Defender". Update to describe Critic ↔ Scientist debate.

- [ ] **Step 8: Update tests**

In `tests/test_critic.py`:

Remove `compressed_history` from `base_kwargs` fixture. Update `plan` fixture (already has success_criteria from prior work). Rename any "defender" references to "scientist". Update assertions for transcript in return value. Add a test that verifies transcript is returned:

```python
@pytest.mark.asyncio
async def test_debate_returns_transcript(self, base_kwargs):
    """Debate returns transcript with all rounds."""
    with (
        patch(
            "auto_scientist.agents.critic.query_openai",
            new_callable=AsyncMock,
            side_effect=["Critique R1", "Critique R2"],
        ),
        patch(
            "auto_scientist.agents.critic.query_anthropic",
            new_callable=AsyncMock,
            return_value="Scientist response",
        ),
    ):
        result = await run_debate(**base_kwargs, max_rounds=2)

    assert len(result) == 1
    transcript = result[0]["transcript"]
    assert len(transcript) == 3  # critic, scientist, critic
    assert transcript[0]["role"] == "critic"
    assert transcript[1]["role"] == "scientist"
    assert transcript[2]["role"] == "critic"
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest tests/test_critic.py -v`

- [ ] **Step 10: Commit**

```
refactor: replace Defender with Scientist in debate loop
```

---

### Task 5: Add Scientist revision call

**Files:**
- Modify: `src/auto_scientist/agents/scientist.py`
- Modify: `src/auto_scientist/prompts/scientist.py`

- [ ] **Step 1: Add revision prompts**

In `src/auto_scientist/prompts/scientist.py`, add:

```python
SCIENTIST_REVISION_SYSTEM = """\
You are a scientist revising your plan after a debate with a critic.

You previously formulated a plan (hypothesis, strategy, changes, success criteria).
A critic challenged it and you debated. Now produce a REVISED plan that
incorporates the valid points from the debate.

You may:
- Accept valid critique and adjust your plan accordingly
- Reject points that were adequately addressed in the debate
- Adjust success criteria based on the discussion
- Change strategy or hypothesis entirely if the debate revealed fundamental issues

Your revised plan must use the same JSON schema as the original plan.
Output a complete revised plan, not just the changes.

Your output must be a JSON object with these exact keys:
- hypothesis: str (revised if needed)
- strategy: str (one of "incremental", "structural", "exploratory")
- changes: list[object] (each with: what, why, how, priority)
- expected_impact: str
- should_stop: bool
- stop_reason: str | null
- notebook_entry: str (document what changed from the debate and why)
- success_criteria: list[object] (each with: name, description, metric_key, condition)
"""

SCIENTIST_REVISION_USER = """\
## Domain Knowledge
{domain_knowledge}

## Analysis of Previous Version
{analysis_json}

## Lab Notebook
{notebook_content}

## Your Original Plan
{original_plan}

## Debate Transcript
{debate_transcript}

## Your Task
Produce a revised plan incorporating valid critique from the debate. Use the
same JSON schema as the original plan. The notebook_entry should document
what you changed from your original plan and why.

The new version is: {version}
"""
```

- [ ] **Step 2: Add run_scientist_revision() function**

In `src/auto_scientist/agents/scientist.py`, add:

```python
async def run_scientist_revision(
    original_plan: dict[str, Any],
    debate_transcript: list[dict[str, str]],
    analysis: dict[str, Any],
    notebook_path: Path,
    version: str,
    domain_knowledge: str = "",
) -> dict[str, Any]:
    """Revise the plan after a critic debate.

    Args:
        original_plan: The initial plan that was debated.
        debate_transcript: List of {"role": "critic"|"scientist", "content": str}.
        analysis: Structured analysis JSON from the Analyst.
        notebook_path: Path to the lab notebook.
        version: Version string.
        domain_knowledge: Domain-specific context.

    Returns:
        Revised plan dict (same schema as the initial plan).
    """
    from auto_scientist.prompts.scientist import (
        SCIENTIST_REVISION_SYSTEM,
        SCIENTIST_REVISION_USER,
    )

    notebook_path = Path(notebook_path)
    notebook_content = notebook_path.read_text() if notebook_path.exists() else ""

    # Format debate transcript
    transcript_text = ""
    for entry in debate_transcript:
        role = entry["role"].capitalize()
        transcript_text += f"### {role}\n{entry['content']}\n\n"

    user_prompt = SCIENTIST_REVISION_USER.format(
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        analysis_json=(
            json.dumps(analysis, indent=2) if analysis else "(no analysis)"
        ),
        notebook_content=notebook_content or "(empty notebook)",
        original_plan=json.dumps(original_plan, indent=2),
        debate_transcript=transcript_text or "(no debate - critique was skipped)",
        version=version,
    )

    options = ClaudeAgentOptions(
        system_prompt=SCIENTIST_REVISION_SYSTEM,
        allowed_tools=[],
        max_turns=1,
        output_format={"type": "json_schema", "schema": SCIENTIST_PLAN_SCHEMA},
    )

    result_text = ""
    assistant_texts: list[str] = []

    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, ResultMessage):
            if message.result:
                result_text = message.result
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    assistant_texts.append(block.text)

    raw = result_text
    if not raw:
        raw = "\n".join(assistant_texts)

    if not raw:
        raise RuntimeError("Scientist revision returned no output")

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        raw = "\n".join(lines)

    return json.loads(raw)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/ -v`

- [ ] **Step 4: Commit**

```
feat: add Scientist revision call for post-debate plan refinement
```

---

### Task 6: Wire new flow into orchestrator

**Files:**
- Modify: `src/auto_scientist/orchestrator.py`

- [ ] **Step 1: Update _run_iteration flow**

Replace the current steps 4-5 with the new flow:

```python
    async def _run_iteration(self) -> None:
        """Run one iteration: synthesize, analyze, plan, debate, revise, implement, run, evaluate."""
        self.state.iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {self.state.iteration}")
        print(f"{'='*60}")

        # Step 0: Periodic synthesis
        await self._run_synthesis()

        # Step 1: Analyst observes latest results
        analysis = await self._run_analyst()

        # Step 2: Scientist plans next iteration
        plan = await self._run_scientist_plan(analysis)

        # Step 3: Check if Scientist recommends stopping
        if plan and plan.get("should_stop"):
            print(f"Scientist recommends stopping: {plan.get('stop_reason', 'unknown')}")
            self.state.phase = "report"
            return

        # Step 4: Critic debates the Scientist's plan
        debate_result = await self._run_debate(plan)

        # Step 5: Scientist revises plan based on debate
        revised_plan = await self._run_scientist_revision(plan, debate_result, analysis)

        # Step 6: Coder implements the revised plan
        new_script = await self._run_coder(revised_plan or plan)

        # Step 7: Validate (syntax check)
        if new_script:
            valid = await self._validate_script(new_script)
            if not valid:
                self.state.record_failure()
                return

        # Step 8: Run
        run_result = await self._run_experiment(new_script)

        # Step 9: Evaluate
        final_plan = revised_plan or plan
        version = f"v{self.state.iteration:02d}"
        version_entry = VersionEntry(
            version=version,
            iteration=self.state.iteration,
            script_path=str(new_script),
            hypothesis=final_plan.get("hypothesis", "") if final_plan else "",
        )
        self._evaluate(run_result, version_entry)
        self.state.record_version(version_entry)

        self._notebook_override = None
```

- [ ] **Step 2: Rename _run_critic to _run_debate, return full result**

```python
    async def _run_debate(self, plan: dict | None) -> list[dict[str, Any]] | None:
        """Send plan to critic model(s) for debate with the Scientist."""
        if not self.critic_models or plan is None:
            print("  DEBATE: skipped (no critics configured or no plan)")
            return None

        from auto_scientist.agents.critic import run_debate

        notebook_content = self._notebook_content()
        domain_knowledge = self.config.domain_knowledge if self.config else ""

        n_critics = len(self.critic_models)
        print(f"  DEBATE: {n_critics} critic(s), {self.debate_rounds} round(s)")
        critiques = await run_debate(
            critic_specs=self.critic_models,
            plan=plan,
            notebook_content=notebook_content,
            domain_knowledge=domain_knowledge,
            max_rounds=self.debate_rounds,
        )

        print(f"  DEBATE: received {len(critiques)} critique(s)")
        return critiques
```

- [ ] **Step 3: Add _run_scientist_revision method**

```python
    async def _run_scientist_revision(
        self,
        plan: dict | None,
        debate_result: list[dict[str, Any]] | None,
        analysis: dict | None,
    ) -> dict[str, Any] | None:
        """Scientist revises plan based on debate."""
        if plan is None or not debate_result:
            print("  REVISE: skipped (no plan or no debate)")
            return None

        from auto_scientist.agents.scientist import run_scientist_revision

        version = f"v{self.state.iteration:02d}"
        notebook_path = self.output_dir / "lab_notebook.md"
        domain_knowledge = self.config.domain_knowledge if self.config else ""

        # Combine transcripts from all critics
        all_transcript: list[dict[str, str]] = []
        for entry in debate_result:
            all_transcript.append({"role": "critic", "content": f"[{entry['model']}]"})
            all_transcript.extend(entry.get("transcript", []))

        print("  REVISE: scientist revising plan after debate")
        try:
            revised = await run_scientist_revision(
                original_plan=plan,
                debate_transcript=all_transcript,
                analysis=analysis or {},
                notebook_path=notebook_path,
                version=version,
                domain_knowledge=domain_knowledge,
            )

            # Write revised notebook entry
            if revised.get("notebook_entry"):
                with notebook_path.open("a") as f:
                    f.write(revised["notebook_entry"] + "\n\n---\n\n")

            print(f"  REVISE: strategy={revised.get('strategy', '?')}")
            return revised
        except Exception as e:
            print(f"  REVISE: error - {e}, using original plan")
            return None
```

- [ ] **Step 4: Simplify _run_coder - remove critique parameter**

Change signature from `_run_coder(self, plan, critique)` to `_run_coder(self, plan)`. Remove the `critique_feedback` injection block (lines 388-391).

- [ ] **Step 5: Remove compressed_history from _run_synthesis**

Remove the `from auto_scientist.history import build_compressed_history` import and `compressed_history = build_compressed_history(self.state)` call. Update `run_synthesis()` call to not pass `compressed_history`.

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/ -v`

- [ ] **Step 7: Commit**

```
feat: wire Scientist revision and debate restructuring into orchestrator
```

---

### Task 7: Update docs and visualization

**Files:**
- Modify: `docs/architecture.md`
- Modify: `.claude/CLAUDE.md`
- Modify: `TODO.md`
- Modify: `docs/pipeline-visualizer.html`

- [ ] **Step 1: Update architecture.md**

Update the iteration flow diagram, agent descriptions, and information boundaries. Remove compressed history references. Rename Defender to Scientist in the debate description. Add Scientist revision step.

- [ ] **Step 2: Update CLAUDE.md**

Update the orchestrator flow line to: `[Synthesis] -> Analyst -> Scientist (plan) -> Critic ↔ Scientist (debate) -> Scientist (revise) -> Coder -> Validate -> Run -> Evaluate`

Remove compressed history from information boundaries. Remove Defender references.

- [ ] **Step 3: Update TODO.md**

Add completed item: "Debate restructuring: Scientist replaces Defender, critique flows back to Scientist for revision, compressed history removed"

- [ ] **Step 4: Update pipeline-visualizer.html**

- Remove compressed history node and all its arrows
- Rename "Critic / Defender" to "Critic / Scientist" in the agent node
- Remove critique arrow to Coder
- Add arrow from debate back to Scientist (revised plan)
- Add arrow from Scientist to Coder (revised plan)
- Update tooltips

- [ ] **Step 5: Commit**

```
docs: update architecture and visualization for debate restructuring
```
