# Auto-Scientist

Autonomous scientific investigation framework. Provide a dataset and problem statement, and the system discovers, iterates, and refines approaches through an LLM-driven loop.

## How It Works

1. **Ingestion**: Canonicalizes your raw data and produces operational config
2. **Iteration**: Runs a unified loop: explore data -> define criteria -> analyze -> plan -> debate -> implement -> run
3. **Report**: Generates a final summary of the best approach and key insights

The iteration loop adapts based on what's available: iteration 0 explores the data, iteration 1 defines success criteria and the first hypothesis, and subsequent iterations do normal science with optional criteria revision.

The system uses Claude (via claude-code-sdk) as the primary scientist, with optional multi-round critic-scientist debate (GPT, Gemini, or any supported model critiques; Claude defends; critic refines).

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`npm install -g @anthropic-ai/claude-code`)
- API keys for model providers (see [Environment Variables](#environment-variables))

## Quick Start

```bash
# Install
uv sync

# Launch the interactive TUI
auto-scientist

# Or run directly from a domain config
auto-scientist run -c domains/toy_function/experiment.yaml
```

## Usage

### TUI Launcher (recommended)

Run bare to open an interactive form where you pick domain, data, goal, and settings:

```bash
auto-scientist
```

Pre-fill the TUI from a domain config:

```bash
auto-scientist -c domains/spo2/experiment.yaml
auto-scientist -c domains/toy_function/experiment.yaml
auto-scientist -c domains/alien_minerals/experiment.yaml
auto-scientist -c domains/alloy_design/experiment.yaml
auto-scientist -c domains/water_treatment/experiment.yaml
```

TUI keyboard shortcuts:
- `Ctrl+R` - Run the experiment
- `Ctrl+S` - Save config to YAML
- `Ctrl+Q` - Quit

### Direct CLI Run (no TUI)

Use `auto-scientist run` to skip the TUI and start immediately:

```bash
# Run a domain from its YAML config
auto-scientist run -c domains/spo2/experiment.yaml

# Run from raw data (no YAML needed)
auto-scientist run \
  --data ./my_data.csv \
  --goal "Investigate the relationship between X and Y" \
  --max-iterations 10

# Override YAML settings from the command line
auto-scientist run -c domains/spo2/experiment.yaml --max-iterations 5 --preset fast

# Schedule for overnight (preserves daytime token budget)
auto-scientist run -c domains/spo2/experiment.yaml --schedule "22:00-06:00"
```

### Resume and Status

State is persisted after every phase transition, so you can safely kill and resume without data loss.

```bash
# Resume after crash or pause
auto-scientist resume --state experiments/state.json

# Resume with different model config
auto-scientist resume --state experiments/state.json --preset fast

# Check progress
auto-scientist status --state experiments/state.json
```

## CLI Reference

### `auto-scientist` (bare)

Launches the TUI launcher form.

| Flag | Description |
|------|-------------|
| `-c, --config <path>` | Pre-fill TUI from an `experiment.yaml` |

### `auto-scientist run`

Run an investigation directly (no TUI).

| Flag | Default | Description |
|------|---------|-------------|
| `--data <path>` | *(required without YAML)* | Path to dataset (file or directory) |
| `--goal <text>` | *(required without YAML)* | Problem statement / investigation goal |
| `-c, --config <path>` | | Path to `experiment.yaml` or `models.toml` |
| `--preset <name>` | `default` | Model preset: `default`, `fast`, `high`, `max` |
| `--max-iterations <int>` | `20` | Maximum iteration count |
| `--debate-rounds <int>` | `1` | Critic-scientist debate rounds (0 = skip debate) |
| `--output-dir <path>` | `experiments` | Output directory |
| `--schedule <window>` | | Time window, e.g. `"22:00-06:00"` |
| `--interactive` | `false` | Enable human-in-the-loop at decision points |
| `--no-stream` | | Disable live token streaming during debate |
| `--no-summaries` | | Disable periodic agent progress summaries |
| `-v, --verbose` | | Show debug log messages on console |

When using `-c` with a YAML config, all other flags are optional overrides. Without YAML, `--data` and `--goal` are required.

### `auto-scientist resume`

Resume a previously paused or crashed run.

| Flag | Default | Description |
|------|---------|-------------|
| `--state <path>` | *(required)* | Path to `state.json` |
| `--config <path>` | | Override saved model config with `models.toml` |
| `--preset <name>` | | Override saved preset |
| `--no-summaries` | | Disable summaries on resume |
| `-v, --verbose` | | Debug logging to console |

### `auto-scientist status`

Check progress of a run (domain, phase, iteration, versions, dead ends).

| Flag | Description |
|------|-------------|
| `--state <path>` | *(required)* Path to `state.json` |

## Configuration

### YAML Experiment Config

Each domain has an `experiment.yaml` that captures the full experiment setup. All paths are resolved relative to the YAML file's directory.

```yaml
# Required
data: seed/data/my_data.csv
goal: >
  Describe the investigation goal here.
  Use YAML folded scalar (>) for multiline goals.

# Optional (shown with defaults)
max_iterations: 20
preset: default                    # default, fast, high, max
debate_rounds: 1                   # 0 = skip debate entirely
output_dir: experiments
schedule: "22:00-06:00"            # Optional time window
interactive: false
stream: true
verbose: false
summaries: true

# Optional per-agent model overrides
models:
  analyst:
    provider: anthropic
    model: claude-sonnet-4-6
    reasoning: medium              # off, minimal, low, medium, high, max
  scientist:
    provider: anthropic
    model: claude-opus-4-6
    reasoning:
      level: high
      budget: 16384                # Custom token budget
  critics:
    - provider: openai
      model: gpt-5.4
      reasoning: medium
    - provider: google
      model: gemini-3.1-pro-preview
      reasoning: high
```

Per-agent overrides replace the preset defaults (they are not merged). The critics array replaces the entire preset critics list.

**Constraint:** SDK agents (analyst, scientist, coder, ingestor, report) must use Anthropic models. Non-Anthropic models are only allowed for critics and summarizer.

### Model Presets

Four built-in presets control model assignment and reasoning levels:

| Preset | Use case | Defaults | Scientist | Critics |
|--------|----------|----------|-----------|---------|
| `default` | Balanced | sonnet-4-6 (medium) | opus-4-6 (medium) | Gemini 3.1 Pro (low), GPT-5.4 (medium) |
| `fast` | Speed/cost | haiku-4-5 (off) | haiku-4-5 (off) | Gemini 3.1 Flash Lite (off), GPT-5.4-nano (off) |
| `high` | Quality | sonnet-4-6 (high) | opus-4-6 (high) | Gemini 3.1 Pro (high), GPT-5.4 (high) |
| `max` | Maximum | opus-4-6 (max) | opus-4-6 (max) | Gemini 3.1 Pro (max), GPT-5.4 (max) |

### Environment Variables

Set these in your shell or in a `.env` file in the project root:

```
ANTHROPIC_API_KEY      # For Claude models (required)
OPENAI_API_KEY         # For GPT models (critics, summarizer)
GOOGLE_API_KEY         # For Gemini models (critics)
```

## Available Domains

| Domain | Difficulty | Description | Data |
|--------|------------|-------------|------|
| `toy_function` | Easy | Discover the hidden mathematical function from noisy x/y data | CSV |
| `alien_minerals` | Medium | Classify 6 alien mineral types from physical/chemical properties | Multi-file |
| `alloy_design` | Medium | Discover composition-property relationships in metal alloys | Multi-file |
| `water_treatment` | Hard | Causal relationships in water treatment plant SCADA data | Multi-file |
| `spo2` | Hard | SpO2 dynamics during breath-holds (sensor calibration + physiology) | SQLite DB |

## Features

### Adaptive Iteration

- **Iteration 0** (exploration): Analyst characterizes raw data, Scientist plans exploration, no debate
- **Iteration 1** (criteria definition): Analyst reads exploration results, Scientist defines top-level success criteria and first hypothesis
- **Iteration 2+** (normal science): Full loop with analysis, planning, debate, implementation, and evaluation against criteria

### Multi-Round Debate

Each iteration (from iteration 1 onward) includes a critic-scientist debate phase:

1. Critic challenges the Scientist's plan from a randomly assigned persona
2. Scientist defends via API response
3. Critic refines their critique (repeat for N rounds)
4. Scientist produces a revised plan incorporating valid criticism

Both Critic and Scientist have web search enabled during debate. Neither sees experiment code (that's the Coder's domain). Set `--debate-rounds 0` to skip debate entirely.

### Information Boundaries

Strict separation of concerns between agents:

| Agent | Sees | Does not see |
|-------|------|-------------|
| **Analyst** | Results, plots, raw data, notebook | Code |
| **Scientist** | Analysis JSON, notebook, domain knowledge, predictions | Code |
| **Critic** | Plan, analysis, notebook, predictions, domain knowledge | Code |
| **Coder** | Plan, previous script | Analysis details, debate transcript |
| **Report** | Notebook, all versions, results | (comprehensive, read-only) |

### Prediction Tracking

The Scientist defines testable predictions for each hypothesis. The experiment script evaluates them, and the Analyst transcribes outcomes (confirmed/refuted/inconclusive). The full history informs future iterations.

### Schedule (Time Windows)

Restrict execution to specific hours to manage token budgets or run overnight:

```yaml
schedule: "22:00-06:00"   # Overnight window (wraps past midnight)
schedule: "09:00-17:00"   # Business hours only
```

The run sleeps when outside the window and resumes when it opens.

### Summarizer

Optional progress summaries during long-running agent phases. A lightweight model (GPT-5.4-nano by default) polls agent output and generates one-line status updates. Disable with `--no-summaries` or `summaries: false` in YAML.

### Interactive Mode

When `--interactive` is enabled, the system pauses at key decision points for human approval or corrections before proceeding.

### Live TUI Dashboard

The pipeline runs inside a Textual-based terminal UI showing:

- Bordered iteration containers (v00, v01, ...) with collapsible agent panels
- Agent name, model, and reasoning level for each phase
- Persistent metrics bar with sparkline, token count, phase name, and elapsed time
- Command palette (`Ctrl+\`) for navigation and control

## Output Structure

Each run produces the following in the output directory:

```
experiments/
  state.json                  # Full experiment state (for resume)
  model_config.json           # Saved model configuration
  lab_notebook.xml            # Iteration journal
  report.md                   # Final report
  debug.log                   # Full debug logging
  console.log                 # TUI output capture
  data/                       # Canonicalized data (from Ingestor)
  buffers/
    ingestor_00.txt           # Raw agent output per phase
    analyst_00.txt
    scientist_00.txt
    debate_*.txt              # Debate transcripts
    coder_00.txt
    ...
  v00/                        # Per-version outputs
    experiment.py             # Generated experiment script
    plan.json                 # Scientist's plan
    analysis.json             # Analyst's structured observation
    results.txt               # Script output
    run_result.json           # Execution result metadata
    *.png                     # Generated plots
```

If the output directory already contains a `state.json`, a new suffixed directory (e.g., `experiments_001`) is created automatically.

## Adding a New Domain

1. Copy `domains/example_template/` to `domains/your_domain/`
2. Edit `config.py` with your `DomainConfig` (data paths, run command, timeout, protected paths)
3. Edit `prompts.py` with domain-specific knowledge for the agents
4. Place seed data in `domains/your_domain/seed/data/`
5. Create an `experiment.yaml` with your data path and goal

See `domains/spo2/` for a complete real-world example.

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_cli.py -v

# Lint
uv run ruff check src/ tests/
```

## Case Study

See [Classifying Alien Minerals](docs/showcase-alien-minerals.md) for a detailed walkthrough of an end-to-end autonomous investigation - how the system explored, failed, recovered, and delivered interpretable classification rules in under an hour.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full specification, or open [docs/pipeline-visualizer.html](docs/pipeline-visualizer.html) for an interactive diagram of the pipeline.

## Status

This project is in **alpha** (v0.1.0-alpha). The core pipeline works end-to-end but expect breaking changes.

### Implemented
- [x] Project scaffold, CLI, and state machine
- [x] Interactive TUI launcher with domain picker and config editor
- [x] YAML experiment config with CLI override support
- [x] Ingestor agent (data canonicalization with human-in-the-loop)
- [x] Analyst agent (structured observation, no recommendations)
- [x] Scientist agent (pure prompt-in/JSON-out planning)
- [x] Critic agent (multi-model, multi-round debate with web search)
- [x] Coder agent (experiment implementation with error correction)
- [x] Report agent (final summary generation)
- [x] Orchestrator with adaptive iteration loop
- [x] Two-tier success criteria (top-level + per-iteration)
- [x] Strict information boundaries (only Coder sees code)
- [x] Prediction tracking and evaluation
- [x] Retrospective notebook entries
- [x] Multi-provider LLM support (OpenAI, Anthropic, Google)
- [x] Four model presets (default, fast, high, max)
- [x] Per-agent model overrides in YAML
- [x] Agent output validation with structured output and retry
- [x] Pre-flight model validation with user-friendly error messages
- [x] Live TUI dashboard with agent panels and metrics
- [x] Schedule/time-window support
- [x] Crash recovery via state persistence and resume
- [x] Five built-in domains (toy_function, alien_minerals, alloy_design, water_treatment, spo2)

### Planned
- [ ] Interactive report visualizations (HTML charts, explorable data views)
- [ ] Redis + Celery integration (state store, caching, task queue)
- [ ] Criteria revision reliability improvements
