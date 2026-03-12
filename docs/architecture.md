# Auto-Scientist Architecture

## Overview

Auto-Scientist is an autonomous scientific modelling framework. Given a dataset and problem statement, it discovers, iterates, and refines models through an LLM-driven loop.

## Three Phases

```
Phase 1: DISCOVERY (one-time)
  User provides: dataset path + problem statement
  System: explores data -> researches domain -> designs first model -> writes v1 script

Phase 2: ITERATION (autonomous loop)
  [1] Analyst Agent: results text + plots + lab notebook -> structured analysis JSON
  [2] Critic: analysis JSON + lab notebook -> critique + alternative hypotheses
  [3] Scientist Agent: analysis + critique + previous script -> new script + updated notebook
  [4] Runner: executes script -> captures stdout + plots -> saves results
  -> Loop back to [1]

Phase 3: REPORT (one-time)
  Generates final summary: best model, journey, recommendations
```

## Agent Details

### Discovery Agent (Phase 1)
- Uses `ClaudeSDKClient` for persistent session
- Tools: Bash, WebSearch, Read/Write
- Produces: domain config, first experiment script, lab notebook entry #0

### Analyst Agent (Phase 2, step 1)
- Uses `query()` (fresh session each iteration)
- Tools: Read (results + plot PNGs), Glob
- Output: structured JSON (success_score, failures, metrics, recommendations)
- `max_turns`: 5

### Critic (Phase 2, step 2)
- Plain API call (OpenAI/Google/Anthropic SDK)
- Input: analysis JSON + lab notebook + compressed history
- Output: free-text critique

### Scientist Agent (Phase 2, step 3)
- Uses `query()` (fresh session)
- Tools: Read, Write, Edit, Bash, Glob, Grep
- Output: new experiment script + updated lab notebook
- `max_turns`: 30
- Safety hooks: block writes outside experiments/ dir

### Runner (Phase 2, step 4)
- Python `asyncio.create_subprocess_exec`
- Syntax validation before run (`py_compile`)
- Configurable timeout (default 120 min)

## State Machine

```
DISCOVERY -> ANALYZE -> CRITIQUE -> IMPLEMENT -> VALIDATE -> RUN -> EVALUATE
                                                                      |
                                                              ANALYZE (loop)
                                                              or STOP
```

VALIDATE = syntax check. If fails, re-invoke Scientist (max 3 retries).

## Safety Mechanisms

1. Write protection: PreToolUse hook blocks writes outside experiments/ and to data files
2. No destructive bash: Hook blocks rm -rf, git push, git reset, etc.
3. Syntax validation: py_compile before running generated scripts
4. Iteration cap: Hard stop at --max-iterations
5. Crash recovery: State persisted to JSON after every phase transition
6. Consecutive failure cap: Stop after N crashes/failures in a row (default 5)

## Implementation Plan

### Commit 1 (this session): Scaffold
- Full directory structure, all modules with docstrings
- pyproject.toml, tests, domain examples, docs

### Commit 2: Runner + Scheduler + History
- Subprocess execution with timeout
- Time-window scheduling
- Compressed history builder

### Commit 3: Analyst Agent
- query() with structured JSON output
- Multimodal (text + plot images)

### Commit 4: Critic
- Multi-model critique dispatcher
- OpenAI, Google, Anthropic wrappers

### Commit 5: Scientist Agent
- query() with file tools
- Safety hooks
- Lab notebook management

### Commit 6: Orchestrator + Iteration Loop
- State machine implementation
- Error handling, crash recovery

### Commit 7: Discovery Agent
- Autonomous data exploration
- Domain research
- First model generation

### Commit 8: Report Agent
- Final summary generation

### Commit 9: SpO2 Domain
- DomainConfig, domain knowledge prompts
- Real v5->v7 examples

### Commit 10: CLI polish + README + domain template
