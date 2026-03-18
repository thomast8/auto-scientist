# Data Ingestion Agent

## Problem

The auto-scientist pipeline currently has no data canonicalization layer. Users pass a raw file path via `--data`, and every generated script reinvents data loading from scratch. The Discovery phase gets raw, unpredictable input. There's no structured understanding of the data's shape, and certain things about the data (column semantics, table relationships, encoding meanings) simply can't be inferred without human input.

## Solution

A new **Ingestor agent** that runs as the first phase of the pipeline, before Discovery. It takes raw user data (any local file or directory), inspects it, has a multi-round interactive conversation with the human to resolve ambiguities, and produces a clean canonical dataset.

## Pipeline Change

**Before:** Discovery -> Iteration -> Report

**After:** **Ingest** -> Discovery -> Iteration -> Report

## Agent Design

### Responsibilities

- Detect input format (CSV, Excel, JSON, SQLite, Parquet, directory of mixed files, etc.)
- Run code (Bash access) to inspect, profile, and convert data
- When `interactive=True`: ask the human clarifying questions when things can't be inferred (column semantics, table relationships, encoding meanings, units, join keys, etc.)
- When `interactive=False` (autonomous mode): make best-effort decisions based on the data, log assumptions to the notebook, and proceed without human confirmation
- Produce the canonical dataset in whatever format best fits the data
- Update `ExperimentState.data_path` to point at the canonical output

### What It Does NOT Do

- No schema manifests or metadata files (Discovery handles exploration)
- No data quality reports (Discovery's job)
- No preprocessing, feature engineering, or outlier removal (that's for iteration scripts)

### Tools

Uses `ClaudeSDKClient` for a persistent multi-turn session (same pattern as Discovery). Tools: `Bash`, `Read`, `Write`, `Glob`, `Grep`. When `interactive=True`, also gets `AskUserQuestion`.

### Prompt Strategy

The Ingestor receives:
- The raw data path (file or directory)
- The user's `--goal` (knowing the goal can inform ingestion decisions, e.g., which tables to join, which columns matter in a wide dataset)
- Suggested heuristics for format selection (e.g., "prefer SQLite for multi-table relational data, CSV for simple flat tables, Parquet for large single-table datasets")
- When interactive: instructions to ask the human when uncertain rather than guessing
- When autonomous: instructions to make reasonable defaults and document assumptions

### Conversation Flow (Interactive Mode)

1. Agent inspects the raw data (file type, size, structure, sample rows)
2. Agent presents its understanding to the human: "Here's what I see - N rows, these columns, this structure. I have some questions..."
3. Multi-round Q&A until the human is satisfied
4. Agent writes and runs a conversion script to produce the canonical output in `experiments/data/`
5. Agent shows the human the result (schema summary, row counts, sample) for final approval
6. Pipeline continues to Discovery

### Autonomous Mode

When `interactive=False`, the agent:
1. Inspects the raw data
2. Makes best-effort format and structure decisions
3. Logs all assumptions to the lab notebook (e.g., "Assumed column 'cat' is nominal, joined sheets on shared 'id' column")
4. Produces the canonical output
5. Pipeline continues without pausing

This is important for scheduled/overnight runs where no human is available.

### Idempotent Behavior

Always runs, even if the input is already in a clean format. If the data is already well-structured, the agent recognizes that, confirms with the human (interactive) or logs it (autonomous), and copies it through.

### Directory Input

When pointed at a directory, the Ingestor inspects all files, asks the human how they relate (interactive) or infers relationships (autonomous), and produces the appropriate canonical output.

### Large Files

For multi-GB files, the Ingestor samples for profiling (first N rows, random sample) rather than loading everything into memory. Full conversion still processes the entire file.

## Integration

### CLI

No new commands. `auto-scientist run --data ...` gains an ingestion phase at the start. The `--data` argument continues to accept any local file or directory path.

### Orchestrator

- New `"ingestion"` phase before `"discovery"` in the state machine
- Phase dispatch in `Orchestrator.run()` gains an ingestion block before the discovery block
- Calls `run_ingestor(raw_data_path, output_dir, goal, interactive)`
- After ingestion completes: updates `state.data_path` to canonical output path, updates `self.data_path` instance variable so Discovery and downstream agents receive the canonical path (not the raw path)
- Transitions state phase from `"ingestion"` to `"discovery"`

### State Changes

`ExperimentState` changes:
- `phase` default changes from `"discovery"` to `"ingestion"`. Valid phases become: `"ingestion"`, `"discovery"`, `"iteration"`, `"report"`, `"stopped"`
- Add `raw_data_path: str | None` - original user-provided path (file or directory)

`data_path` is repurposed: before ingestion it holds the raw path, after ingestion it points at the canonical output. `raw_data_path` preserves the original for reference.

No separate `ingestion_complete` boolean needed; the phase state machine handles resume: if `phase == "ingestion"`, ingestion reruns. Once phase advances to `"discovery"`, ingestion is skipped on resume.

### DomainConfig Reconciliation

`DomainConfig.data_paths` is a `list[str]`. After ingestion:
- For pre-loaded domain configs (`--domain spo2`): ingestion still runs on the files listed in `data_paths`, producing a canonical output. `state.data_path` points to the canonical output. `DomainConfig.data_paths` is left unchanged (it describes the original domain setup).
- For auto-discovery mode: `DomainConfig` is created by Discovery after ingestion, so Discovery sees the canonical path and sets `data_paths` accordingly.

### Conversion Script Location

The Ingestor's conversion script is saved to `experiments/data/ingest.py` for auditability. This is a one-off utility, not an experiment version. The canonical data output also lives in `experiments/data/`.

### Write Protection

After ingestion, `experiments/data/` is added to the Coder agent's `protected_paths` so generated experiment scripts cannot accidentally overwrite the canonical dataset.

### Resume Behavior

`auto-scientist resume` dispatches on `state.phase`:
- `"ingestion"`: restarts ingestion from scratch (it's cheap and needs human input anyway in interactive mode)
- `"discovery"` or later: ingestion already completed, skips it

### Impact on Existing Agents

- **Orchestrator:** needs changes to add ingestion phase dispatch, update `self.data_path` after ingestion
- **Discovery:** no code changes needed. It receives `self.data_path` from the orchestrator, which now points to canonical data after ingestion.
- **Coder:** same as before, gets `data_path` in prompt. Path now points to canonical data.
- **Analyst, Scientist, Critic:** no changes, they never touch data directly.

## Files

### New

- `src/auto_scientist/agents/ingestor.py` - agent implementation with `run_ingestor()`, uses `ClaudeSDKClient`
- `src/auto_scientist/prompts/ingestor.py` - prompt templates (system + user)

### Modified

- `src/auto_scientist/orchestrator.py` - add ingestion phase before Discovery, update `self.data_path` after ingestion
- `src/auto_scientist/state.py` - add `raw_data_path` field, change `phase` default to `"ingestion"`
- `src/auto_scientist/cli.py` - update initial phase from `"discovery"` to `"ingestion"` when creating new `ExperimentState`

### Runtime Output

```
experiments/
├── data/              <- Ingestor writes here
│   ├── ingest.py      <- the conversion script (for auditability)
│   └── (dataset.db, data.csv, etc.)
├── state.json
├── lab_notebook.md
├── domain_config.json
├── v00/
│   └── ...
```
