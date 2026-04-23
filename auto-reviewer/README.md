# Auto-Reviewer

Autonomous bug-hunting PR reviewer. The Intake agent canonicalizes a PR
into a review workspace; the Surveyor surfaces diff-level suspicions; the
Hunter turns them into testable claims; the Prober writes and runs
reproducer probes; the Findings agent compiles a prioritized report with
reproducers attached. A running **investigation log** carries hypotheses
and open questions across iterations.

A "confirmed prediction" here is a reproducer (failing test, assertion,
demonstrated misbehavior).

Pipeline: **Intake** -> **Investigation** loop (Surveyor -> Hunter ->
(Adversary debate) -> Prober) -> **Findings**.

Auto-Reviewer shares the `auto_core` runtime (orchestrator, state machine,
role registry, multi-model debate, information boundaries) with its
sibling package Auto-Scientist. The runtime is generic; the prompts,
schemas, and agent behaviors here are PR-review specific.

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
