# Auto-Reviewer

Autonomous bug-hunting PR reviewer. Sibling of Auto-Scientist on the shared
`auto_core` runtime. The same orchestrator, state machine, role registry,
multi-model debate, lab notebook + abduction carry-forward, and information
boundaries - repurposed for PR review.

A "confirmed prediction" here is a reproducer (failing test, assertion,
demonstrated misbehavior), not a metric improvement.

Pipeline: **Intake** -> **Investigation** loop (Surveyor -> Hunter ->
(Adversary debate) -> Prober) -> **Findings**.

See `docs/auto-reviewer-deferred-work.md` at the repo root for non-goals and
deferred decisions.

## Smoke test

Run the reviewer against the current branch. The intake agent parses the
prompt, resolves refs, and writes the canonical workspace; add `--critics ""`
to skip the adversary debate and `--max-iterations 1` to stop after one pass.

```bash
uv run auto-reviewer review \
  "review the changes on refactor/extract-auto-core against main" \
  --max-iterations 1 \
  --critics ""
```

Workspace lands at `./review_workspace/review_<timestamp>/`. After intake
completes, `data/diff.patch`, `data/pr_metadata.json`, `data/touched_files/`,
and `domain_config.json` should all be populated. Substitute any
natural-language pointer for the prompt: a PR URL, `owner/repo#N`,
`"my current branch"`, etc.
