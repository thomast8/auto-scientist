# Auto-Scientist Agent Guidance

This repo contains the auto-scientist app, the shared `auto_core` runtime, and
the `auto-reviewer` app that shares the same orchestrator. Treat cross-package
changes carefully: behavior that belongs to orchestration, state, persistence,
model resolution, SDK handling, or report cleanup usually belongs in
`auto-core`, then each app should call the shared helper.

## Run And Resume Workflow

- To extend a completed run, prefer resume with a fork instead of starting over:
  `uv run auto-scientist resume --from <run> --fork --output-dir <new-run> --max-iterations N --no-summaries --notify off --verbose`.
- Use this resume pattern when validating carry-forward behavior. It preserves
  prior iterations, state, notebooks, predictions, and dead ends, then continues
  from the next iteration in a safe copy.
- Do not resume a completed run in-place. The CLI intentionally asks for `--fork`
  so the original artifacts remain available for comparison.
- For real LLM smokes, isolate output under `/private/tmp/...` and include
  `--no-summaries --notify off`. Do not reuse production, reviewer, or
  user-facing output directories for exploratory validation.

## Artifact-First Verification

- Treat Rich/TUI repaint output as operational noise. Use persisted artifacts
  for assertions and final claims.
- Use `console.log` to track phase completion, elapsed time, model display, and
  whether the run reached report generation.
- Use `state.json` for authoritative run state: `phase`, `iteration`,
  `versions`, `predictions`, and structured `dead_ends`.
- Use `vNN/plan.json` and `vNN/revision_plan.json` to see what the Scientist
  proposed and what survived debate.
- Use `vNN/debate.json` and `buffers/debate_*_NN.txt` to inspect critic
  feedback. Use `buffers/scientist_*` and `buffers/report_*` when checking
  prompt-visible context or transcript artifacts.
- Use `vNN/exitcode.txt`, `vNN/stderr.txt`, `vNN/run_result.json`, and
  `vNN/results.txt` to verify generated experiment execution.
- Use `report.md` for the final human-facing claim. If report text has SDK
  transcript preamble, fix shared report cleanup in `auto_core`, not only the
  app-specific reporter.

## Dead-End Tracking Checks

- To prove dead-end tracking end to end, check all four layers:
  Scientist output, shared persisted state, later role context, and final report
  surfacing.
- Scientist emission lives in `vNN/plan.json` or `vNN/revision_plan.json` under
  `dead_ends`.
- Shared persistence lives in `state.json` as objects with `iteration`,
  `description`, and `evidence`.
- Later roles should avoid re-treading persisted dead ends unless they explicitly
  reopen them with new evidence. Check later plans, revisions, critiques, and
  reports for this behavior.
- Reports should include a "Ruled Out" or equivalent section with the evidence
  that closed each path. Negative results are part of the scientific output, not
  loose notes.

## Finished TUI Processes

- Sometimes the TUI can keep repainting after artifacts are already written.
  Before terminating anything, confirm `console.log` says the report was written
  or the relevant phase completed, and confirm the expected files exist.
- Inspect the process group with
  `ps -o pid,ppid,pgid,stat,etime,%cpu,command`.
- If the run is complete but still repainting, terminate the run process group
  with `kill -TERM -- -<pgid>`. Do not kill active runs that are still producing
  new artifacts.

## Model And Preset Notes

- OpenAI is the default provider family for current presets, while Anthropic
  remains explicitly supported.
- Codex-backed runs may normalize unavailable model IDs to the nearest available
  Codex model. In particular, Nano IDs can persist in `model_config.json` while
  the live Codex app-server displays Mini if Nano is not exposed.
- When checking model behavior, inspect both `model_config.json` and the startup
  banner in `console.log`.

## Verification Before Claims

- Prefer targeted tests around the changed contract, then at least one real-code
  smoke that imports the actual module or runs the actual CLI.
- For shared behavior, include both auto-scientist and auto-reviewer checks when
  practical.
- If a live run exposes a cleanup issue, add a regression test from the real
  artifact shape before pushing the fix.
