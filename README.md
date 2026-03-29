# Auto-Scientist

**An autonomous scientific investigation framework.** Give it a dataset and a question. It explores, hypothesizes, debates, implements, fails, recovers, and converges on an answer, without human intervention.

![Live pipeline dashboard showing agents running an investigation](docs/images/tui-pipeline.png)

## How It Works

Four specialized agents run in a loop, each with strict information boundaries:

```
Ingest -> Analyst -> Scientist -> Debate -> Coder -> Run -> Evaluate -> loop
```

| Agent | Role | Key constraint |
|-------|------|----------------|
| **Analyst** | Observes results and plots, reports structured JSON | Cannot recommend, only observe |
| **Scientist** | Forms hypotheses, defines testable predictions, plans experiments | Never sees code |
| **Critics** | Challenge the plan from multiple angles (adversarial debate) | Never see code |
| **Coder** | Implements and runs experiments | Never sees the debate |

The Scientist never sees code. The Coder never sees the debate. The Analyst cannot recommend. These boundaries prevent confirmation bias: the entity that plans is not the one that evaluates, and the one that implements has no stake in the hypothesis.

Both Critics and Scientist have web search during debate to verify claims and look up literature.

![Pipeline Architecture](docs/images/pipeline-diagram.png)

### Adaptive iterations

- **Iteration 0**: Explore raw data, characterize distributions, establish baselines
- **Iteration 1**: Define top-level success criteria, formulate first hypothesis with testable predictions
- **Iteration 2+**: Full scientific loop with analysis, planning, multi-round debate, implementation, and evaluation

The Scientist decides when to stop based on scientific judgment (goal satisfied, noise floor reached, diminishing returns), not a fixed threshold.

### Prediction tracking

Every hypothesis comes with pre-registered testable predictions, each with diagnostics and branching plans ("if confirmed, do X; if refuted, do Y"). The experiment script evaluates them in code, and the Analyst transcribes outcomes. This creates accountability: wrong predictions redirect the investigation instead of being ignored.

## Quick Start

```bash
# Install
git clone https://github.com/thomast8/auto-scientist.git
cd auto-scientist
uv sync

# Set API keys (only needed for non-Anthropic critics)
export OPENAI_API_KEY="..."      # Optional (GPT as critic)
export GOOGLE_API_KEY="..."      # Optional (Gemini as critic)

# Launch the interactive TUI
auto-scientist

# Or run directly from a domain config
auto-scientist run -c domains/alien_minerals/experiment.yaml
```

![TUI launcher with domain config and model selection](docs/images/tui-launcher.png)

## Built-in Domains

| Domain | Difficulty | What the system investigates |
|--------|------------|------------------------------|
| `toy_function` | Easy | Discover the hidden mathematical function from noisy x/y data |
| `alien_minerals` | Medium | Classify 6 alien mineral types from physical/chemical properties |
| `alloy_design` | Medium | Discover composition-property relationships in metal alloys |
| `water_treatment` | Hard | Causal discovery with latent confounders, feedback loops, nonlinearity, regime changes, and MNAR missingness |
| `spo2` | Expert | SpO2 dynamics during breath-holds (nonlinear physiology, latent variables, domain expertise) |

Each domain is a directory under `domains/` with an `experiment.yaml`, seed data, and optional domain-specific prompts. Adding a new domain means copying the template and filling in your data path and goal.

## Examples

### Alien Minerals Classification

The system classified six alien mineral types from nine physical measurements, noisy labels, and calibration artifacts. Full writeup: [docs/showcase-alien-minerals.md](docs/showcase-alien-minerals.md).

It went down a dead end (a hand-crafted hierarchy that dropped F1 from 0.92 to 0.60), got called out by its own critics for sunk-cost reasoning, pivoted to validating the decision tree's own rules, and delivered interpretable classification rules in 58 minutes. No human touched the data or wrote any code.

| Metric | Value |
|--------|-------|
| Wall time | 58 minutes |
| Iterations | 3 |
| Final macro F1 | 0.93 (5x5 CV) |
| Unanimous specimens misclassified | 0 / 475 |
| Features in final rules | 5 of 12 |

### Toy Function Discovery

Given 200 noisy (x, y) points and zero hints, the system recovered the exact generating formula in two iterations. Full writeup: [docs/showcase-toy-function.md](docs/showcase-toy-function.md).

It screened polynomials and trig composites, identified a phase ambiguity in the cosine fit, reparameterized to sine, proved the clean constants (0.3, 2.5, 1.5) are preferred over free parameters by BIC, and confirmed residual whiteness with Fisher's g-test. Critics caught test-set leakage and goal drift before they could corrupt the results.

| Metric | Value |
|--------|-------|
| Wall time | 39 minutes |
| Iterations | 2 |
| Discovered formula | y = 0.3x² + 2.5·sin(1.5x) |
| Parameter accuracy | within 1.5% |
| Test R² | 0.959 (ceiling: 0.978) |

## Usage

### TUI Launcher

Run bare to open an interactive form:

```bash
auto-scientist
```

Pre-fill from a domain config:

```bash
auto-scientist -c domains/alien_minerals/experiment.yaml
```

Keyboard shortcuts: `Ctrl+R` run, `Ctrl+S` save config, `Ctrl+Q` quit.

### Direct CLI

```bash
# From YAML config
auto-scientist run -c domains/spo2/experiment.yaml

# From raw data
auto-scientist run \
  --data ./my_data.csv \
  --goal "Investigate the relationship between X and Y" \
  --max-iterations 10

# Override settings
auto-scientist run -c domains/spo2/experiment.yaml --max-iterations 5 --preset fast

# Schedule for overnight
auto-scientist run -c domains/spo2/experiment.yaml --schedule "22:00-06:00"
```

### Resume, Replay, and Status

State is persisted after every phase transition. Kill and resume without data loss.

```bash
# Resume a crashed or paused run from where it left off
auto-scientist resume --state experiments/state.json

# Resume with a different model config
auto-scientist resume --state experiments/state.json --preset high

# Replay: fork a saved run from a specific iteration (new output dir, original untouched)
auto-scientist replay --from experiments/runs/my-run --at-iteration 1 --max-iterations 10

# Replay without --at-iteration: continue from where it stopped with more iterations
auto-scientist replay --from experiments/runs/my-run --max-iterations 10

# Same, but with different models
auto-scientist replay --from experiments/runs/my-run --max-iterations 10 --preset fast

# Check progress of any run
auto-scientist status --state experiments/state.json
```

`resume` picks up in-place from the last saved phase. `replay` copies the run to a new directory and never touches the original. With `--at-iteration`, it rewinds to that iteration and re-runs from there (useful for testing prompt changes or alternative paths). Without `--at-iteration`, it continues from where the run stopped with a fresh iteration budget (useful when the Scientist stopped too early or you want more iterations).

## Configuration

### YAML Experiment Config

```yaml
data: seed/data/my_data.csv
goal: >
  Describe the investigation goal here.

# Optional
max_iterations: 20
preset: default          # default, fast, high, max
debate_rounds: 1         # 0 = skip debate
schedule: "22:00-06:00"  # Time window for execution
interactive: false       # Human-in-the-loop at decision points

# Per-agent model overrides
models:
  scientist:
    provider: anthropic
    model: claude-opus-4-6
    reasoning: high
  critics:
    - provider: openai
      model: gpt-5.4
      reasoning: medium
    - provider: google
      model: gemini-3.1-pro-preview
      reasoning: high
```

### Model Presets

| Preset | Use case | Scientist | Critics |
|--------|----------|-----------|---------|
| `default` | Balanced | opus-4-6 (medium reasoning) | Gemini 3.1 Pro + GPT-5.4 |
| `fast` | Speed/cost | haiku-4-5 | Gemini Flash Lite + GPT-5.4-nano |
| `high` | Quality | opus-4-6 (high reasoning) | Gemini 3.1 Pro + GPT-5.4 (high) |
| `max` | Maximum | opus-4-6 (max reasoning) | Gemini 3.1 Pro + GPT-5.4 (max) |

All core agents (analyst, scientist, coder, ingestor, report) run through the [Claude Code SDK](https://docs.anthropic.com/en/docs/claude-code), which uses your Claude Code subscription instead of per-token API billing. No `ANTHROPIC_API_KEY` is needed. Only the critics support non-Anthropic providers (OpenAI, Google), which do require their respective API keys and are billed per-token.

## CLI Reference

### `auto-scientist run`

| Flag | Default | Description |
|------|---------|-------------|
| `--data <path>` | *(required without YAML)* | Path to dataset |
| `--goal <text>` | *(required without YAML)* | Investigation goal |
| `-c, --config <path>` | | YAML config file |
| `--preset <name>` | `default` | Model preset |
| `--max-iterations <int>` | `20` | Maximum iterations |
| `--debate-rounds <int>` | `1` | Debate rounds (0 = skip) |
| `--output-dir <path>` | `experiments` | Output directory |
| `--schedule <window>` | | Time window, e.g. `"22:00-06:00"` |
| `--interactive` | `false` | Human-in-the-loop mode |
| `--no-stream` | | Disable live token streaming |
| `--no-summaries` | | Disable agent progress summaries |
| `-v, --verbose` | | Debug logging |

### `auto-scientist resume`

Resume a paused or crashed run in-place from the last saved phase.

| Flag | Default | Description |
|------|---------|-------------|
| `--state <path>` | *(required)* | Path to `state.json` |
| `--config <path>` | | Override saved model config with `models.toml` |
| `--preset <name>` | | Override saved preset |
| `--no-summaries` | | Disable agent progress summaries |
| `-v, --verbose` | | Debug logging |

### `auto-scientist replay`

Fork a saved run, rewind to a specific iteration, and continue from there. The original run is never modified.

| Flag | Default | Description |
|------|---------|-------------|
| `--from <path>` | *(required)* | Path to a saved run directory |
| `--at-iteration <int>` | *(current)* | Iteration to rewind to (0-based). Omit to extend from current iteration |
| `--max-iterations <int>` | `20` | Maximum iterations for the replayed run |
| `--output-dir <path>` | *(auto)* | Output directory (default: auto-generated suffix) |
| `-c, --config <path>` | | Override model config |
| `--preset <name>` | | Override preset |
| `--debate-rounds <int>` | `1` | Debate rounds per persona |
| `--no-stream` | | Disable live token streaming |
| `--no-summaries` | | Disable agent progress summaries |
| `-v, --verbose` | | Debug logging |

### `auto-scientist status`

| Flag | Description |
|------|-------------|
| `--state <path>` | *(required)* Path to `state.json` |

## Output Structure

```
experiments/
  state.json              # Full state (for resume)
  lab_notebook.xml         # Iteration journal
  report.md                # Final report
  data/                    # Canonicalized data
  buffers/                 # Raw agent output per phase
  v00/                     # Per-iteration outputs
    experiment.py          #   Generated experiment script
    plan.json              #   Scientist's plan
    analysis.json          #   Analyst's structured observation
    results.txt            #   Script output
    *.png                  #   Generated plots
```

## Adding a New Domain

1. Copy `domains/example_template/` to `domains/your_domain/`
2. Place your data in `domains/your_domain/seed/data/`
3. Create `experiment.yaml` with your data path and goal
4. Optionally add domain-specific prompts in `prompts.py`

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check src/ tests/
```

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`npm install -g @anthropic-ai/claude-code`)
- API keys: `OPENAI_API_KEY` and `GOOGLE_API_KEY` (optional, for non-Anthropic critics). Core agents use Claude Code (subscription-based, no per-token cost)

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full spec, or open [docs/pipeline-visualizer.html](docs/pipeline-visualizer.html) for an interactive diagram.

## Status

**Stable** (v1.0.0). The core pipeline works end-to-end with full autonomous investigation capabilities.
