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
  "review feature/tighten-paginate against main in the repo at /tmp/auto-reviewer-buggy-paginate" \
  --cwd /tmp/auto-reviewer-buggy-paginate \
  --preset turbo \
  --max-iterations 2
```

Output lands under `./review_workspace/review_<timestamp>/`. Look at
`report.md`, the probe scripts written during iterations, and the `state.json`
transitions.

## Expected outcome

A sharp review should produce:
- **1 confirmed bug** tied to `paginate` returning `page_size + 1` items
  instead of `page_size`. Evidence: a failing pytest probe and a reference
  to the docstring's "at most page_size entries" claim.
- **0 phantom bugs** (no HIGH-priority claims without named caller impact).
- **Design Intent** should call out the docstring/code contradiction
  explicitly during the iter-0 debate.

Use this fixture as a calibration target: zero-bug PRs and single-bug PRs
both land here, so regressions in phantom rate or miss rate show up
immediately.

## Cleaning up

```bash
rm -rf /tmp/auto-reviewer-buggy-paginate
rm -rf ./review_workspace
```
