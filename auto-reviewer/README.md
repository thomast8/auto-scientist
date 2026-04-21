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
