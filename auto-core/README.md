# `auto_core`

Shared runtime for Auto-Scientist and Auto-Reviewer. Contains the pieces neither
app should own: the orchestrator, state machine, role registry, persistence,
LLM client wrappers, MCP tool scaffolding, scheduler, retry, notebook.

Apps depend on `auto_core` and supply a `RoleRegistry` + payload type. The
orchestrator does not import any app-specific agent modules.

See `docs/auto-reviewer-deferred-work.md` at the repo root for non-goals and
deferred decisions.
