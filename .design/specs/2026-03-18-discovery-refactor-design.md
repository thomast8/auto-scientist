# Discovery Agent Refactoring - Design Spec

**Date**: 2026-03-18
**Goal**: Separate exploration from code generation in the Discovery agent, enforcing the architecture's information boundary principle. Also remove the `--domain` path (pre-loaded domain config) pending a redesign.

## Problem

In auto-discovery mode (no `--domain`), the Discovery agent currently does everything in a single session: explores data, writes the domain config, writes the lab notebook, AND writes the v00 experiment script. This violates the architecture's information boundary principle where only the Coder agent should write Python experiment code.

Additionally, the `--domain` path (pre-loaded domain config) uses a hardcoded synthetic plan instead of the Scientist. Rather than fixing both paths now, the domain-config path is removed entirely to be rethought later. Only auto-discovery mode is supported.

## Design

### Flow Change

**Before:**
```
Discovery -> [config + notebook + v00 script]
```

**After:**
```
Discovery -> [config + notebook]
Scientist -> [v00 plan from notebook + domain knowledge]
Coder -> [v00 script from plan]
```

### Change 1: Remove `--domain` CLI Flag and Domain-Config Path

**`src/auto_scientist/cli.py`:**
- Remove `--domain` option from the `run` command
- Remove `load_domain_config()` function
- Remove `config` parameter from `Orchestrator()` construction
- The `resume` command also stops reloading domain configs (the config is already persisted in state via `config_path`)

**`src/auto_scientist/orchestrator.py`:**
- Remove `config` parameter from `__init__`
- Remove the `if self.config is not None` branch in `_run_discovery()` (the entire domain-config path)
- `_run_discovery()` always runs the auto-discovery flow: Discovery -> Scientist -> Coder

**Note**: The `domains/` directory and domain config files stay in the repo. They're just not loadable via CLI for now.

### Change 2: Discovery Agent - Remove Script Generation

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

### Change 3: Orchestrator - Wire Discovery -> Scientist -> Coder

**`src/auto_scientist/orchestrator.py` - `_run_discovery()` method:**

```python
async def _run_discovery(self) -> None:
    from auto_scientist.agents.discovery import run_discovery
    from auto_scientist.agents.scientist import run_scientist
    from auto_scientist.agents.coder import run_coder

    notebook_path = self.output_dir / "lab_notebook.md"

    # Step 1: Discovery explores data, writes config + notebook
    print("DISCOVERY phase: exploring dataset and designing first approach")
    self.config = await run_discovery(
        state=self.state,
        data_path=self.data_path,
        output_dir=self.output_dir,
        interactive=self.interactive,
        model=self.model,
    )
    self.state.config_path = str(self.output_dir / "domain_config.json")

    # Step 2: Scientist plans v00 from notebook + domain knowledge
    print("DISCOVERY phase: scientist planning v00")
    plan = await run_scientist(
        analysis={},
        notebook_path=notebook_path,
        version="v00",
        domain_knowledge=self.config.domain_knowledge,
        model=self.model,
    )

    # Append scientist's notebook entry
    with notebook_path.open("a") as f:
        f.write(plan["notebook_entry"] + "\n---\n\n")

    # Step 3: Coder implements v00
    print("DISCOVERY phase: coder writing v00")
    script_path = await run_coder(
        plan=plan,
        previous_script=Path("nonexistent"),
        output_dir=self.output_dir,
        version="v00",
        domain_knowledge=self.config.domain_knowledge,
        data_path=self.state.data_path or "",
        experiment_dependencies=self.config.experiment_dependencies,
        model=self.model,
    )

    # Run and evaluate v00 (unchanged from current code)
    ...
```

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
| `src/auto_scientist/orchestrator.py` | Remove domain-config path, wire Discovery -> Scientist -> Coder |
| `src/auto_scientist/cli.py` | Remove `--domain` flag and `load_domain_config()` |
| `tests/test_discovery.py` | Update assertions for new return type, remove script verification tests |
| `docs/architecture.md` | Remove "first experiment script" from Discovery produces list |
| `docs/pipeline-visualizer.html` | Update data flow to show Discovery -> Scientist -> Coder for v00 |

### Test Plan

- [ ] Unit: Discovery returns `DomainConfig` only (no script path)
- [ ] Unit: Orchestrator calls Scientist then Coder after Discovery
- [ ] Integration: auto-discovery mode produces v00 via Scientist -> Coder pipeline
- [ ] Verify: information boundary holds (Discovery never writes `.py` files)
- [ ] Verify: `--domain` flag is gone from CLI help
