# Auto-Scientist TODO

## In Progress

- [ ] Discovery phase customizing downstream agent prompts based on what it learns about the problem domain

## Completed

- [x] Debate restructuring: Scientist replaces Defender, critique flows back to Scientist for revision, compressed history removed - 2026-03-17
- [x] Per-iteration success criteria: Scientist defines testable predictions, Coder evaluates in script, Analyst reports both tiers - 2026-03-14
- [x] Periodic investigation synthesis (condense notebook every N iterations) - 2026-03-13
- [x] Web search for Critic and Defender (OpenAI, Google, Anthropic) - 2026-03-13
- [x] Debate simplification: symmetric context, no analysis/script in debate - 2026-03-13
- [x] Scientist simplification: no tools, no code access, pure prompt-in/JSON-out - 2026-03-13
- [x] Strict information boundaries: only Coder sees Python code - 2026-03-13
- [x] Restructure agent roles: Analyst (observer), Scientist (planner), Coder (implementer), Critic (challenger) - 2026-03-13
- [x] Make prompts domain-agnostic (remove ML/model-fitting specific language) - 2026-03-13
- [x] Multi-round critic-scientist debate loop - 2026-03-11
- [x] Full orchestration loop and state machine - 2026-03-11
- [x] Initial scaffold with all agents, runner, scheduler, history - 2026-03-10
