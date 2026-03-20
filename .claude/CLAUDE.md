# Auto-Scientist

## Project Overview
Autonomous scientific investigation framework. Given a dataset and problem statement, the system discovers, iterates, and refines approaches through an LLM-driven loop. See `docs/architecture.md` for the full spec.

## Architecture Summary
Two-phase pipeline: Ingestion -> Iteration (unified loop) -> Report.

0. **Ingestor** (canonicalizer): inspects raw data, asks human for clarification (interactive mode), produces canonical dataset + slim DomainConfig. Uses Bash tools.

Iteration loop (four agents, adaptive behavior):
1. **Analyst** (observer): reads results + plots (or raw data on iteration 0), outputs structured JSON. No recommendations. On iteration 0, produces domain_knowledge + data_summary.
2. **Scientist** (planner): pure prompt-in/JSON-out, no tools, no code access. Outputs plan + per-iteration success criteria. On iteration 1, defines top-level criteria. On iteration 2+, may revise criteria.
3. **Critic** (challenger): multi-round debate with the Scientist. Both have web search. Symmetric context (plan + notebook + domain knowledge). Skipped on iteration 0.
4. **Coder** (implementer): only agent that reads/writes Python code. Follows the revised plan.

Orchestrator flow: Ingest -> Analyst -> Scientist (plan) -> criteria update -> stop check -> (Debate, iter 1+) -> Scientist (revise) -> Coder -> Validate -> Run -> Evaluate -> increment iteration

### Iteration Lifecycle
- **Iteration 0** (exploration): Analyst characterizes raw data, Scientist plans exploration, no debate
- **Iteration 1** (criteria definition): Analyst reads exploration results, Scientist defines top-level success criteria + first hypothesis
- **Iteration 2+** (normal science): Analyst evaluates against criteria, Scientist plans next experiment, may revise criteria

### Success Criteria (two tiers)
- **Top-level** (defined by Scientist on iteration 1, stored in ExperimentState): define when the investigation is done
- **Per-iteration** (from Scientist each iteration): testable predictions of each hypothesis, evaluated by the script in code, transcribed by the Analyst

### Information Boundaries
- Only the Coder sees Python code
- The Scientist plans from analysis JSON + notebook (no code)
- Critic and Scientist (during debate) get symmetric context (no analysis, no script); they challenge the plan's criteria too
- After debate, the Scientist revises the plan; Coder gets only the revised plan (no critique)
- results.txt is compiled by the script itself (print statements), no LLM post-processing

### Key Components
- `models/*.py`: OpenAI/Google/Anthropic wrappers with optional `web_search=True`
- `agents/critic.py`: `run_debate()` orchestrates multi-round critic-scientist loop
- `agents/scientist.py`: `run_scientist()` for initial plan, `run_scientist_revision()` for post-debate revision
- `agents/ingestor.py`: `run_ingestor()` canonicalizes raw data into experiments/data/

## Key Directories
- `src/auto_scientist/` - Core framework code
- `src/auto_scientist/agents/` - Agent implementations (ingestor, analyst, scientist, critic, coder, report)
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
- Prompt templates go in `src/auto_scientist/prompts/` (XML-delimited, see Prompt Standards below)
- New domains get their own directory under `domains/`
- State is persisted as JSON via Pydantic models
- Experiment scripts must be self-contained (no framework imports)
- Prompts must be domain-agnostic (no ML/model-fitting specific language at framework level)
- `docs/pipeline-visualizer.html` is the canonical pipeline visualization. Update it whenever agent roles, data flow, artifacts, or information boundaries change. It contains an interactive SVG diagram with hover-to-highlight, agent cards, an information boundary matrix, and tooltips with example content for every element.

## Prompt Standards
All prompts in `src/auto_scientist/prompts/` follow these rules:
- **Structure:** `<role>` -> `<instructions>` -> `<examples>` -> `<output_format>` -> `<recap>`. XML tags only, no markdown delimiters.
- **Role:** "You are a [domain] [function] system. [Purpose]. [Constraints]."
- **Framing:** Positive ("your output is strictly observational") not negative ("do NOT recommend").
- **Examples:** 3-5 for structured JSON output agents (Analyst, Scientist, Revision). Include null/empty handling. Reasoning before output.
- **Output format:** Verbal description + JSON schema + fallback rules for missing data.
- **Recap:** Repeat critical instructions for prompts over ~5K tokens.
- **File split:** Static content in system prompt, dynamic in user prompt. Tool-using agents (Coder, Ingestor, Report) skip few-shot examples.
- See `.design/specs/2026-03-18-prompt-engineering-rewrite-design.md` for full spec.
