# Auto-Scientist TODO

## Future

- [ ] Criteria revision reliability: In toy_function_022, v02 regressed (score 17 from 33) but criteria revision was never triggered because the loop hit max iterations before the Scientist could see the regression analysis. Investigate whether (a) the iteration budget is too tight, (b) the Scientist needs stronger prompting to revise criteria on regression, or (c) the mechanism needs restructuring (e.g., automatic revision consideration after score drops). Needs more data from runs to confirm the pattern.
- [ ] Interactive report visualizations: Evolve the report phase to generate interactive HTML visualizations (charts, explorable data views). This likely requires a dedicated post-report sub-pipeline (Analyst to identify what to visualize, Coder to generate the data/JS), or a new "visualization" agent. Scope and approach TBD.
- [ ] Redis + Celery integration: Redis for state store (replace JSON), LLM response caching, pub/sub progress events; Celery (with Redis as broker) for experiment runner tasks and critic fan-out via group/chord; Docker Compose for infrastructure (Redis, Celery worker, Flower monitoring); graceful degradation to current local-only mode when infra not running

## Completed

- [x] Notebook agentic data access: All five agents that consumed `notebook_content` (Scientist, Analyst, Critic, Stop Gate, Report) now receive a compact one-line-per-entry Table of Contents instead of the full XML, plus an in-process `mcp__notebook__read_notebook` MCP tool for on-demand entry detail (queryable by summary, versions, source, search, last_n). On the alloy_design saved run this is a 97% reduction in the notebook prompt slot (~32K full XML → ~1K TOC at iter 5). The orchestrator stops pre-rendering notebook content for critic and stop-debate paths and instead passes `notebook_path` so each persona builds its own MCP server; API-mode critics still receive the full inline XML as a fallback since they cannot call MCP tools. - 2026-04-13
- [x] Prediction tree agentic data access: Replaced the full prediction history dump (~46K chars at iter 20 projection) with a compact one-line-per-prediction tree (~6-8K) plus an in-process `mcp__predictions__read_predictions` MCP tool wired through `SDKOptions.mcp_servers` for the Scientist and prediction-aware Critics. Established the shared `_mcp_base.py` infrastructure (MCPToolSpec, build_mcp_server_config, run_mcp_server_main, registry) that the notebook tool now reuses. - 2026-04-04
- [x] Abductive reasoning and meta-observation diagnostics: Scientist runs a structured protocol on each refuted prediction (enumerate assumptions, identify weakest, generate alternative explanation naming mechanisms in the system under study, derive testable consequence). Analyst emits `data_diagnostics` for cross-cutting structural patterns across its own findings. Pending abductions are carried forward, enforced by critics at debate time, Assessor at stop time, and Report in the Limitations section. On the water_treatment domain vs ground truth, Claude score improved 0.47 -> 0.79 and GPT score improved 0.44 -> 0.62 across two iterations of the prompt design. - 2026-04-09
- [x] Resume and fork: `auto-scientist resume --from <run> --fork --from-iteration N` copies a saved run, keeps iterations 0 through N-1, and re-runs from iteration N. `--from-agent` enables sub-iteration granularity (e.g. resume from scientist within iteration 3). Restored-from-disk agents/iterations show dashed borders and "(restored)" labels in the TUI. Also supports in-place resume of completed runs. - 2026-03-30
- [x] Agent output validation + retry: Pydantic models for Analyst/Scientist JSON outputs, validation + retry (3 attempts) for JSON agents, syntax check + retry for Coder, data/content validation + retry for Ingestor/Report, provider-native structured output (Anthropic tool_use, OpenAI json_schema, Google response_schema) with direct API path for Scientist - 2026-03-22
- [x] Retrospective notebook entries: Scientist reflects on investigation arc (breakthroughs, dead ends, diagnostic indicators) alongside forward-looking plans; synthesis module removed (redundant with 1M context windows + self-compressing retrospective entries) - 2026-03-18
- [x] Debate restructuring: Scientist replaces Defender, critique flows back to Scientist for revision, compressed history removed - 2026-03-17
- [x] Per-iteration success criteria: Scientist defines testable predictions, Coder evaluates in script, Analyst reports both tiers - 2026-03-14
- [x] Web search for Critic and Defender (OpenAI, Google, Anthropic) - 2026-03-13
- [x] Debate simplification: symmetric context, no analysis/script in debate - 2026-03-13
- [x] Scientist simplification: no tools, no code access, pure prompt-in/JSON-out - 2026-03-13
- [x] Strict information boundaries: only Coder sees Python code - 2026-03-13
- [x] Restructure agent roles: Analyst (observer), Scientist (planner), Coder (implementer), Critic (challenger) - 2026-03-13
- [x] Make prompts domain-agnostic (remove ML/model-fitting specific language) - 2026-03-13
- [x] Multi-round critic-scientist debate loop - 2026-03-11
- [x] Full orchestration loop and state machine - 2026-03-11
- [x] Initial scaffold with all agents, runner, scheduler, history - 2026-03-10
