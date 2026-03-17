# Auto-Scientist

## Project Overview
Autonomous scientific modelling framework. Given a dataset and problem statement, the system discovers, iterates, and refines models through an LLM-driven loop. See `docs/architecture.md` for the full spec.

## Architecture Summary
Three-phase pipeline: Discovery -> Iteration -> Report.

Iteration loop (four agents):
1. **Analyst** (observer): reads results + plots, outputs structured JSON. No recommendations.
2. **Scientist** (planner): pure prompt-in/JSON-out, no tools, no code access. Outputs plan + per-iteration success criteria.
3. **Critic** (challenger): multi-round debate with the Scientist. Both have web search. Symmetric context (plan + notebook + domain knowledge). No analysis JSON, no script.
4. **Coder** (implementer): only agent that reads/writes Python code. Follows the revised plan.

Orchestrator flow: [Synthesis] -> Analyst -> Scientist (plan) -> stop check -> Critic ↔ Scientist (debate) -> Scientist (revise) -> Coder -> Validate -> Run -> Evaluate

### Success Criteria (two tiers)
- **Top-level** (from Discovery/config): define when the investigation is done
- **Per-iteration** (from Scientist): testable predictions of each hypothesis, evaluated by the script in code, transcribed by the Analyst

### Information Boundaries
- Only the Coder sees Python code
- The Scientist plans from analysis JSON + notebook (no code)
- Critic and Scientist (during debate) get symmetric context (no analysis, no script); they challenge the plan's criteria too
- After debate, the Scientist revises the plan; Coder gets only the revised plan (no critique)
- results.txt is compiled by the script itself (print statements), no LLM post-processing

### Key Components
- `synthesis.py`: plain API call, condenses notebook every N iterations (`--synthesis-interval`)
- `models/*.py`: OpenAI/Google/Anthropic wrappers with optional `web_search=True`
- `agents/critic.py`: `run_debate()` orchestrates multi-round critic-scientist loop
- `agents/scientist.py`: `run_scientist()` for initial plan, `run_scientist_revision()` for post-debate revision

## Key Directories
- `src/auto_scientist/` - Core framework code
- `src/auto_scientist/agents/` - Agent implementations (discovery, analyst, scientist, critic, coder, report)
- `src/auto_scientist/prompts/` - Prompt templates for each agent
- `src/auto_scientist/models/` - LLM API client wrappers (with web search support)
- `domains/` - Domain-specific configs, prompts, and seed data
- `domains/spo2/` - SpO2 domain (first test case)
- `experiments/` - Runtime output directory (gitignored)
- `tests/` - Pytest test suite
- `docs/architecture.md` - Full architecture specification

## Development
- Python >= 3.11, managed with `uv`
- Run tests: `uv run pytest`
- CLI: `uv run auto-scientist run --data ... --goal "..."`
- Lint: `uv run ruff check src/ tests/`

## Conventions
- All agent implementations go in `src/auto_scientist/agents/`
- Prompt templates go in `src/auto_scientist/prompts/`
- New domains get their own directory under `domains/`
- State is persisted as JSON via Pydantic models
- Experiment scripts must be self-contained (no framework imports)
- Prompts must be domain-agnostic (no ML/model-fitting specific language at framework level)
