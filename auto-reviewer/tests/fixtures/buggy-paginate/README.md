# Fixture: buggy-paginate

Throwaway repo seed for a live auto-reviewer smoke test. A tiny Python project
with a `paginate()` helper and one deliberate off-by-one bug.

## Layout

- `good/` - clean implementation (becomes the `main` branch).
- `buggy/` - the same file with a deliberate off-by-one (becomes the
  `feature/tighten-paginate` branch).
- `setup_fixture.sh` - materializes both as branches in a throwaway git
  repo outside this tree.

## The bug

In `buggy/src/paginate.py` the slice expression is `items[start:end]` where
`end = start + page_size + 1`. Each page returns `page_size + 1` items
instead of `page_size`. The commit message frames it as "tighten slice
bounds" to make it look plausible on review.

A minimal reproducer (what the Prober should write):

```python
from src.paginate import paginate

def test_paginate_returns_page_size_items():
    items = list(range(10))
    assert len(paginate(items, 0, 3)) == 3
```

This fails on `feature/tighten-paginate` (returns 4) and passes on `main`.

## Run

```bash
./setup_fixture.sh           # materializes /tmp/auto-reviewer-buggy-paginate
cd /tmp/auto-reviewer-buggy-paginate
uv sync                      # so `uv run pytest` works

cd -                         # back to auto-scientist repo root
uv run auto-reviewer review \
  --pr feature/tighten-paginate \
  --repo-path /tmp/auto-reviewer-buggy-paginate \
  --base-ref main \
  --goal "Find correctness bugs in the paginate slice change" \
  --max-iterations 2 \
  --critics ""
```

Output lands under `./review_workspace/feature_tighten-paginate/`. Look at
`report.md`, the probe scripts written during iterations, and the `state.json`
transitions.

## Cleaning up

```bash
rm -rf /tmp/auto-reviewer-buggy-paginate
rm -rf ./review_workspace
```

## Known rough edges on first live run

- The Intake agent's prompt says it will use `gh pr view` when available; for
  a local-only branch (no GitHub PR), it should fall back to
  `git diff main..feature/tighten-paginate` and `git log main..HEAD`.
- The Prober runs inside the target repo's Python env. It has no sandbox
  beyond the orchestrator's PreToolUse hooks. This is a throwaway fixture,
  so that is fine here.
- `--critics ""` disables the Adversary debate - keep it off for the MVP
  smoke. Add critics once the spine is proven.
