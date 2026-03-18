# Auto-Scientist

Autonomous scientific investigation framework. Provide a dataset and problem statement, and the system discovers, iterates, and refines approaches through an LLM-driven loop.

## How It Works

1. **Discovery**: Explores your data, researches the domain, and designs a first approach
2. **Iteration**: Runs an autonomous loop of analyze -> critique -> implement -> run
3. **Report**: Generates a final summary of the best approach and key insights

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
2. Edit `config.py` with your DomainConfig (data paths, success criteria, run command)
3. Edit `prompts.py` with domain-specific knowledge for the agents
4. Place seed data in `domains/your_domain/seed/data/`

See `domains/spo2/` for a complete real-world example.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full specification.

## Status

This project is under active development. Current state:
- [x] Project scaffold and structure
- [ ] Runner + Scheduler + History
- [ ] Analyst Agent
- [x] Critic (multi-model, multi-round debate)
- [ ] Scientist Agent
- [ ] Orchestrator + Iteration Loop
- [ ] Discovery Agent
- [ ] Report Agent
- [ ] SpO2 domain integration
- [ ] CLI polish
