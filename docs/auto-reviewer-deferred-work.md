# Auto-Reviewer + `auto_core` extraction: deferred work + non-goals

This file is the immutable tracker for decisions we consciously deferred during
the modular extraction of Auto-Scientist into `auto_core` + `auto_scientist` +
`auto_reviewer`. Items here are not forgotten - they are explicitly out of scope
for the extraction and sibling-app milestone. Revisit before shipping each item
into production.

Canonical plan: `/Users/thomastiotto/.claude/plans/yes-but-do-100-immutable-stonebraker.md`
(local to the planning session - mirrored here in case that path changes).

## Non-goals for the initial extraction

1. **Isolation sandbox for Prober executing PR code.** The Prober runs code from
   the PR it is reviewing. Existing PreToolUse hooks (block writes outside the
   review workspace, block destructive bash) carry over, but "don't execute the
   PR's code with the reviewer's credentials" is a new concern. Deferred to a
   later milestone once we have real PRs running through the pipeline.
2. **Adversary persona catalog expansion.** Ship the initial four personas
   (security, concurrency, API-break, input-fuzzing). Evaluate empirically which
   ones earn their cost before adding more.
3. **Migrating existing `experiments/` output directories.** No schema-breaking
   change is made to the state file; existing state.json files load via the
   `RunState.load` legacy-phase migration.
4. **TUI re-theming beyond registry-driven styles.** Styles + descriptions come
   from the `RoleRegistry`; anything requiring new Textual widget types or
   cross-app layout work is out of scope.
5. **Rewriting Auto-Scientist's tests at the `ExperimentState` level.** The
   existing tests keep using the `ExperimentState` alias for `RunState[ExperimentPayload]`.
6. **Stop-criterion choice for Auto-Reviewer.** The review pipeline inherits the
   Auto-Scientist stop gate (completeness assessment -> stop debate -> stop
   revision). After real runs we decide whether "every suspicion is confirmed or
   refuted", "no new suspicions surfaced in N iterations", or token budget is
   the right termination condition for review.
7. **Interactive review mode.** In the scientific case there is no oracle; the
   reviewer has one (the PR author knows the intent). Letting the Prober ask the
   author questions mid-review is a follow-up capability, not part of MVP.
8. **Shared `auto_cli` package.** Each app keeps its own `cli.py`. Extracting
   CLI scaffolding into `auto_core` is a follow-up if `auto_scientist.cli` and
   `auto_reviewer.cli` start drifting in ways that only diverge cosmetically.

## Deferred design decisions

1. **Probe execution isolation boundary.** Options under consideration: fresh
   venv, git worktree, ephemeral container. Each has a cost/benefit against
   real credentials and network isolation. Decide before the first non-fixture
   run.
2. **Stop gate presets for Auto-Reviewer.** The `assessor` preset field in
   `ModelConfig` is reused. If review termination logic diverges enough we will
   either subclass the stop-gate role or introduce a separate `review_assessor`
   preset key.
3. **Auto-Reviewer's notion of "done".** Scientific investigation stops on
   judgment (diminishing returns + goal satisfaction). Review has candidate
   stop signals: exhausted suspicions, timeout, author pushback. MVP uses the
   iteration cap; revisit with data.
4. **Registry key naming convention.** Initial choice: `RoleKey.OBSERVER`,
   `RoleKey.PLANNER`, etc. (role-shaped names, app-agnostic). Alternative:
   app-specific keys per registry. Revisit if the current design accumulates
   boilerplate.

## Open questions the MVP will not answer

1. **Does review depth correlate with adversary personas?** The ablation
   described in the ultraplan (`--critics ""` vs full persona set) needs at
   least 5 real PRs to judge. MVP only proves the pipeline runs.
2. **How does the Hunter plan when it sees only the call-graph summary?** The
   information-boundary principle (Hunter does not read source) is preserved
   from Auto-Scientist. Whether it produces qualitatively different suspicions
   than a source-reading planner is an empirical question.
3. **Does abduction carry-forward find bugs that a one-shot probe cannot?**
   Mirrors the scientific version's value claim. Needs longitudinal review
   data, not MVP wiring.
4. **Per-PR workspace layout.** MVP writes under `review_workspace/`. Whether to
   namespace by PR number, by review session, or by probe is deferred until we
   see actual usage patterns.

## How this file evolves

- Add an item when we intentionally defer a design choice during the extraction.
- Strike an item (do not delete it - leave a struck-through entry with the date
  and resolution) when it is resolved.
- Do not use this file for active TODOs - those belong in `TODO.md`.
- Do not use this file for status updates - those belong in PR descriptions and
  commit messages.
