# Discovery Agent Refactoring - Design Spec

**Date**: 2026-03-18
**Goal**: Separate exploration from code generation in the Discovery agent, enforcing the architecture's information boundary principle.

## Problem

In auto-discovery mode (no `--domain`), the Discovery agent currently does everything in a single session: explores data, writes the domain config, writes the lab notebook, AND writes the v00 experiment script. This violates the architecture's information boundary principle where only the Coder agent should write Python experiment code.

The domain-config path in the orchestrator already does it correctly: it delegates v00 script creation to the Coder. The auto-discovery path needs to follow the same pattern, but with the Scientist producing the plan rather than a hardcoded synthetic plan.

## Design

### Flow Change

**Before (auto-discovery):**
```
Discovery -> [config + notebook + v00 script]
```

**After (auto-discovery):**
```
Discovery -> [config + notebook]
Scientist -> [v00 plan from notebook + domain knowledge]
Coder -> [v00 script from plan]
```

**Before (domain-config):**
```
Config loaded -> synthetic hardcoded plan -> Coder -> [v00 script]
```

**After (domain-config):**
```
Config loaded -> Scientist -> [v00 plan] -> Coder -> [v00 script]
```

Both paths converge: config source (Discovery or pre-loaded) -> Scientist plans v00 -> Coder implements v00.

### Change 1: Discovery Agent - Remove Script Generation

**`src/auto_scientist/agents/discovery.py`:**
- Return type changes from `tuple[DomainConfig, Path]` to `DomainConfig`
- Remove `version_dir` creation (`v00/` directory)
- Remove `script_path` variable and verification
- Keep: domain config creation, lab notebook creation, data exploration

**`src/auto_scientist/prompts/discovery.py`:**
- Remove Step 3 ("Write the Experiment Script") entirely
- Refocus Step 2: "Design the approach conceptually" instead of "Design what the script should do"
- Renumber remaining steps: Explore (1), Design (2), Notebook (3), Config (4)
- Remove references to script file paths and script requirements from the prompt

### Change 2: Orchestrator - Wire Discovery -> Scientist -> Coder

**`src/auto_scientist/orchestrator.py` - `_run_discovery()` method:**

The auto-discovery path becomes:

```python
# Auto-discovery mode
config = await run_discovery(state, data_path, output_dir, interactive, model)
self.config = config
self.state.config_path = str(self.output_dir / "domain_config.json")
```

Then both paths (auto-discovery and domain-config) converge into shared code:

```python
notebook_path = self.output_dir / "lab_notebook.md"

# Scientist plans v00 from notebook + domain knowledge
plan = await run_scientist(
    analysis={},
    notebook_path=notebook_path,
    version="v00",
    domain_knowledge=config.domain_knowledge,
    model=self.model,
)

# Append notebook entry
with notebook_path.open("a") as f:
    f.write(plan["notebook_entry"] + "\n---\n\n")

# Coder implements v00
script_path = await run_coder(
    plan=plan,
    previous_script=Path("nonexistent"),
    output_dir=output_dir,
    version="v00",
    domain_knowledge=config.domain_knowledge,
    data_path=state.data_path or "",
    experiment_dependencies=config.experiment_dependencies,
    model=self.model,
)
```

### Change 3: Domain-Config Path Uses Scientist

The existing domain-config path in `_run_discovery()` currently creates a hardcoded synthetic plan dict. This is replaced with a `run_scientist()` call, making both paths identical after the config is obtained.

The hardcoded synthetic plan is removed. However, the domain-config path still needs to create the initial notebook skeleton (Goal + Domain header) before calling the Scientist, since Discovery doesn't run in this path. In auto-discovery mode, the Discovery agent creates the notebook as part of its exploration.

```python
if self.config is not None:
    # Domain-config path: create notebook skeleton (Discovery doesn't run)
    if not notebook_path.exists():
        notebook_path.write_text(
            f"# Lab Notebook\n\n## Goal\n{self.state.goal}\n\n"
            f"## Domain\n{self.config.name}: {self.config.description}\n\n---\n\n"
        )
else:
    # Auto-discovery path: Discovery creates notebook + config
    self.config = await run_discovery(...)

# Both paths converge: Scientist -> Coder (shared code below)
```

Note: `v00/` directory creation is also removed from the orchestrator since `run_coder()` already handles it (coder.py:102-103).

### Change 4: Minor Scientist Prompt Tweak

The Scientist system prompt currently says: "On the first iteration (v01), there is no prior arc to reflect on. Focus on your initial assessment of the baseline results and your forward plan."

This needs two small fixes now that the Scientist handles v00:
1. Change "(v01)" to "(v00)" since the Scientist now plans the very first version
2. Change "baseline results" to "the exploration findings in the notebook" since at v00 there are no results yet, only the Discovery agent's notebook entry

This is a one-line wording change in `SCIENTIST_SYSTEM`, not a new prompt or function.

### Why This Works Without New Functions

The Scientist's existing `run_scientist()` function handles the v00 case:
- It already formats empty analysis as `"(no analysis yet - first iteration)"`
- The Scientist receives the notebook (which contains Discovery's exploration findings) and domain knowledge, giving it enough context to plan a baseline approach
- The plan JSON schema is the same regardless of version

### What Doesn't Change

- `run_scientist()`, `run_coder()` functions - no modifications
- Coder prompts - no modifications
- The iteration loop - no changes
- Tests for Scientist, Coder, iteration pipeline - no changes

### Files Modified

| File | Change |
|---|---|
| `src/auto_scientist/agents/discovery.py` | Remove script generation, change return type |
| `src/auto_scientist/prompts/discovery.py` | Remove Step 3, refocus Step 2 |
| `src/auto_scientist/prompts/scientist.py` | Update first-iteration guidance (v01 -> v00, results -> notebook) |
| `src/auto_scientist/orchestrator.py` | Rewire `_run_discovery()` for both paths |
| `tests/test_discovery.py` | Update assertions for new return type, remove script verification tests |
| `docs/architecture.md` | Remove "first experiment script" from Discovery produces list |
| `docs/pipeline-visualizer.html` | Update data flow to show Discovery -> Scientist -> Coder for v00 |

### Test Plan

- [ ] Unit: Discovery returns `DomainConfig` only (no script path)
- [ ] Unit: Orchestrator calls Scientist then Coder after Discovery
- [ ] Integration: auto-discovery mode produces v00 via Scientist -> Coder pipeline
- [ ] Integration: domain-config mode produces v00 via Scientist -> Coder pipeline
- [ ] Verify: information boundary holds (Discovery never writes `.py` files)
