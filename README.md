# Auto-Scientist

Autonomous scientific investigation framework. Provide a dataset and problem statement, and the system discovers, iterates, and refines approaches through an LLM-driven loop.

## How It Works

1. **Ingestion**: Canonicalizes your raw data and produces operational config
2. **Iteration**: Runs a unified loop: explore data -> define criteria -> analyze -> plan -> debate -> implement -> run
3. **Report**: Generates a final summary of the best approach and key insights

The iteration loop adapts based on what's available: iteration 0 explores the data, iteration 1 defines success criteria and the first hypothesis, and subsequent iterations do normal science with optional criteria revision.

The system uses Claude (via claude-code-sdk) as the primary scientist, with optional multi-round critic-scientist debate (GPT, Gemini, or any supported model critiques; Claude defends; critic refines).

## Quick Start

```bash
# Install
uv sync

# Run from raw data
auto-scientist run \
  --data ./my_data.csv \
  --goal "Investigate the relationship between X and Y" \
  --max-iterations 20

# With multi-model critique (2-round debate by default)
auto-scientist run \
  --data ./my_data.csv \
  --goal "..." \
  --critics openai:gpt-4o,google:gemini-2.5-pro

# Single-pass critique (no debate)
auto-scientist run \
  --data ./my_data.csv \
  --goal "..." \
  --critics openai:gpt-4o \
  --debate-rounds 1

# Schedule for overnight (preserves daytime token budget)
auto-scientist run \
  --data ./my_data.csv \
  --goal "..." \
  --schedule "22:00-06:00"

# Resume after crash or pause
auto-scientist resume --state experiments/state.json

# Check progress
auto-scientist status --state experiments/state.json
```

## Adding a New Domain

1. Copy `domains/example_template/` to `domains/your_domain/`
2. Edit `config.py` with your DomainConfig (data paths, run command, dependencies)
3. Edit `prompts.py` with domain-specific knowledge for the agents
4. Place seed data in `domains/your_domain/seed/data/`

See `domains/spo2/` for a complete real-world example.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full specification.

## Status

This project is in **alpha** (v0.1.0-alpha). The core pipeline works end-to-end but expect breaking changes.

### Implemented
- [x] Project scaffold, CLI, and state machine
- [x] Ingestor agent (data canonicalization with human-in-the-loop)
- [x] Analyst agent (structured observation, no recommendations)
- [x] Scientist agent (pure prompt-in/JSON-out planning)
- [x] Critic agent (multi-model, multi-round debate with web search)
- [x] Coder agent (experiment implementation with error correction)
- [x] Report agent (final summary generation)
- [x] Orchestrator with adaptive iteration loop
- [x] Two-tier success criteria (top-level + per-iteration)
- [x] Strict information boundaries (only Coder sees code)
- [x] Retrospective notebook entries
- [x] Multi-provider LLM support (OpenAI, Anthropic, Google)
- [x] Agent output validation with structured output and retry
- [x] Pre-flight model validation with user-friendly error messages

### Planned
- [ ] Interactive report visualizations (HTML charts, explorable data views)
- [ ] Redis + Celery integration (state store, caching, task queue)
- [ ] Criteria revision reliability improvements
