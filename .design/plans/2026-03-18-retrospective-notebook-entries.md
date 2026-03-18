# Retrospective Notebook Entries Implementation Plan

**Goal:** Enrich the Scientist's notebook entries with retrospective narrative (arc reflection, diagnostic indicators, dead-end reasoning) and remove the now-redundant synthesis module.

**Architecture:** Prompt-only change to the Scientist agent plus clean removal of synthesis.py and all its integration points (orchestrator, CLI, docs).

**Tech Stack:** Python, Click (CLI), prompt templates (plain strings)

---

### Task 1: Remove Synthesis Module

**Files:**
- Delete: `src/auto_scientist/synthesis.py`
- Modify: `src/auto_scientist/orchestrator.py`
- Modify: `src/auto_scientist/cli.py`

- [ ] **Step 1: Remove synthesis from orchestrator**

In `src/auto_scientist/orchestrator.py`:

Remove the `synthesis_interval` parameter and `_notebook_override` field from `__init__`:
```python
# Remove these two lines from __init__:
self.synthesis_interval = synthesis_interval
self._notebook_override: str | None = None  # set by synthesis
# And remove synthesis_interval from the __init__ signature
```

Simplify `_notebook_content()` (line 210-215) from:
```python
def _notebook_content(self) -> str:
    """Return notebook content, using synthesis override if available."""
    if self._notebook_override:
        return self._notebook_override
    notebook_path = self.output_dir / "lab_notebook.md"
    return notebook_path.read_text() if notebook_path.exists() else ""
```
To:
```python
def _notebook_content(self) -> str:
    """Return notebook content."""
    notebook_path = self.output_dir / "lab_notebook.md"
    return notebook_path.read_text() if notebook_path.exists() else ""
```

Delete the entire `_run_synthesis()` method (lines 217-242).

Remove the synthesis call from `_run_iteration()`:
```python
# Remove these two lines:
# Step 0: Periodic synthesis (condense notebook every N iterations)
await self._run_synthesis()
```

Remove the override reset at end of `_run_iteration()`:
```python
# Remove these two lines:
# Reset synthesis override for next iteration
self._notebook_override = None
```

- [ ] **Step 2: Remove synthesis from CLI**

In `src/auto_scientist/cli.py`:

Remove the `--synthesis-interval` option (lines 66-71):
```python
# Remove this entire @click.option block:
@click.option(
    "--synthesis-interval",
    default=0,
    type=int,
    help="Condense notebook every N iterations (0 = disabled)",
)
```

Remove `synthesis_interval: int` from the `run()` function signature (line 82).

Remove `synthesis_interval=synthesis_interval` from the Orchestrator constructor call (line 111).

- [ ] **Step 3: Delete synthesis.py**

Delete `src/auto_scientist/synthesis.py`.

- [ ] **Step 4: Run tests to verify nothing breaks**

Run: `uv run pytest tests/ -v`
Expected: All existing tests pass (no tests reference synthesis).

- [ ] **Step 5: Commit**

```bash
git add -u src/auto_scientist/synthesis.py src/auto_scientist/orchestrator.py src/auto_scientist/cli.py
git commit -m "refactor: remove synthesis module

Retrospective notebook entries make periodic compression redundant,
and modern context windows (1M tokens) can hold long notebooks."
```

---

### Task 2: Enrich Scientist Prompt with Retrospective Guidance

**Files:**
- Modify: `src/auto_scientist/prompts/scientist.py`

- [ ] **Step 1: Expand the "Lab Notebook Entry" section in SCIENTIST_SYSTEM**

In `src/auto_scientist/prompts/scientist.py`, replace the current "Lab Notebook Entry" section (lines 47-50):

```python
## Lab Notebook Entry

Write a notebook entry documenting your hypothesis, strategy, and planned
changes. This becomes the permanent record of your reasoning for this iteration.
```

With:

```python
## Lab Notebook Entry

Your notebook entry is the permanent record of this iteration. It should read
as a continuous narrative under the heading `## {version} - [Brief Title]`.

Before planning forward, reflect on the investigation arc:
- Label the previous iteration's outcome: was it a breakthrough (changed your
  understanding of the problem), an incremental improvement (refined the
  existing approach), or a dead end (abandoned direction)? For dead ends,
  explain the structural reason it failed, not just that metrics didn't improve.
- Note diagnostic indicators of investigation health beyond the score: are
  results genuine or artifacts of overfitting? Is the approach structurally
  sound? Are you converging, stuck in a local minimum, or circling?

Then describe your hypothesis, strategy, and planned changes for the next
iteration.

On the first iteration (v01), there is no prior arc to reflect on. Focus on
your initial assessment of the baseline results and your forward plan.

Good retrospection is concrete and specific to the domain:
  "v03 was a dead end: adding sensor delay sounded physiologically correct,
  but the optimizer can't distinguish delay from washout rate since both just
  shift the curve rightward. We need a data regime that breaks this degeneracy."
Not vague: "v03 didn't work well, so we'll try something different."
```

- [ ] **Step 2: Update the revision prompt's notebook_entry instruction**

In `SCIENTIST_REVISION_SYSTEM`, replace the current notebook_entry line (line 121):

```python
- notebook_entry: str (document what changed from the debate and why)
```

With:

```python
- notebook_entry: str (document what the debate changed and why; update the
  arc assessment if the debate shifted your understanding of where the
  investigation stands; do NOT repeat the full arc reflection from your
  initial entry)
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (prompt changes don't break any existing tests).

- [ ] **Step 4: Commit**

```bash
git add src/auto_scientist/prompts/scientist.py
git commit -m "feat: add retrospective narrative to scientist notebook entries

Notebook entries now include arc reflection (breakthrough/incremental/dead-end
labeling), diagnostic indicators, and dead-end reasoning with hindsight,
alongside the existing forward-looking plan."
```

---

### Task 3: Update Documentation

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/pipeline-visualizer.html`
- Modify: `.claude/CLAUDE.md`
- Modify: `TODO.md`

- [ ] **Step 1: Update architecture.md**

Remove the "Periodic Investigation Synthesis" section (lines 180-184) and the `--synthesis-interval` reference in the CLI example (line 343). Remove `synthesis.py` from the file tree (line 219). Update the `SYNTHESIS` reference in the state machine description (line 171).

- [ ] **Step 2: Update pipeline-visualizer.html**

Remove the entire Synthesis agent card (lines 1385-1415). Remove the CSS rules for `.agent-card.synthesis` (lines 238, 284-285). Also remove the `[Synthesis]` step from the flow-strip (line ~587) and its adjacent arrow.

- [ ] **Step 3: Update .claude/CLAUDE.md**

Remove the `synthesis.py` bullet from the "Key Components" section (line 31). Also remove `[Synthesis]` from the "Orchestrator flow" line (line 17).

- [ ] **Step 4: Update TODO.md**

Mark "Periodic investigation synthesis" as removed/superseded by retrospective notebook entries. Add "Retrospective notebook entries" to completed features.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture.md docs/pipeline-visualizer.html .claude/CLAUDE.md TODO.md
git commit -m "docs: remove synthesis references, document retrospective entries"
```
