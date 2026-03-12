# Auto-Scientist

## Project Overview
Autonomous scientific modelling framework. Given a dataset and problem statement, the system discovers, iterates, and refines models through an LLM-driven loop. See `docs/architecture.md` for the full spec.

## Architecture Summary
Three-phase pipeline: Discovery -> Iteration -> Report.
Iteration loop: Analyst -> Critic -> Scientist -> Runner -> (repeat).
Agents use `claude-code-sdk`. Critic uses direct API calls (OpenAI/Google/Anthropic).

## Key Directories
- `src/auto_scientist/` - Core framework code
- `src/auto_scientist/agents/` - Agent implementations (discovery, analyst, critic, scientist, report)
- `src/auto_scientist/prompts/` - Prompt templates for each agent
- `src/auto_scientist/models/` - LLM API client wrappers for the Critic
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
