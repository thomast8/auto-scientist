# Fixture: leaky-config-merge

A harder calibration fixture for auto-reviewer. Discriminates between
model tiers: a weak reviewer either rubber-stamps ("tests pass") or
fires a phantom HIGH, while a strong reviewer catches the real bug,
cites the docstring, and produces a multi-key probe.

## The shape

The PR is framed as `refactor: simplify nested env-var handling` - a
"cosmetic consolidation" of `apply_env_overrides` in
`src/config.py`. The change:

```python
# Before (good)
top, sub = key.lower().split("_", 1)
if top in config and isinstance(config[top], dict):
    config[top][sub] = _coerce(value)

# After (buggy)
top, sub = key.lower().split("_", 1)
config[top] = {sub: _coerce(value)}
```

This silently replaces the entire nested dict instead of updating one
key - **any sibling key under `config[top]` gets wiped**. The docstring
on `apply_env_overrides` still promises *"Nested structure is
preserved: sibling keys survive the override."*

`src/database.py::db_connection_args` reads three sibling keys under
`config["db"]` (`host`, `port`, `name`), so real callers break when any
`DB_*` env var is set.

## Why it's hard

1. **Existing tests pass on both branches.** `tests/test_config.py`
   has a `test_nested_override_single_key` test, but it uses a config
   with exactly one nested key. Whole-dict replacement is
   indistinguishable from in-place update at width 1.
2. **The refactor added a new test** (`test_env_override_is_applied_in_place`)
   that passes on the buggy branch. A weak reviewer takes that as
   evidence that the refactor is sound.
3. **The commit message sells the change as cosmetic** ("simplify",
   "consolidate"). Plausible framing.
4. **Catching the bug requires reading the docstring** (Design Intent
   lane: *stated vs observed intent*) AND tracing the caller
   (`src/database.py` reads sibling keys).

## Expected review outcome

See `expected.json`. Short version:

- **1 confirmed bug**: `apply_env_overrides` wipes siblings.
- **Reproducer**: probe with a multi-key nested config (e.g.
  `{"db": {"host": "prod", "port": 5432}}`) + `{"DB_HOST": "dev"}`
  override; assert `result["db"]["port"] == 5432`.
- **Caller impact**: `src/database.py::db_connection_args` reads
  `config["db"]["port"]` and `config["db"]["name"]`; buggy
  `apply_env_overrides` raises `KeyError` here.
- **0 phantom HIGH-priority claims.** The added test
  `test_env_override_is_applied_in_place` is correct (refs match) -
  should not be flagged.
- **Design Intent** should cite the docstring's sibling-preservation
  clause explicitly during iter-0 debate.

## Calibration protocol

Run the same fixture across model tiers and compare:

| Tier | Expected result |
|---|---|
| `--preset turbo` (Haiku) | Likely: misses cross-file caller, produces phantom or rubber-stamps |
| `--preset default` (Sonnet) | Should catch bug, may under-probe |
| `--preset high` (Opus) | Should catch bug, cite docstring, probe with multi-key config |

## Run

```bash
./setup_fixture.sh                          # /tmp/auto-reviewer-leaky-config-merge
cd /tmp/auto-reviewer-leaky-config-merge && uv sync && cd -

uv run auto-reviewer review \
  "review feature/simplify-env-overrides against main in the repo at /tmp/auto-reviewer-leaky-config-merge" \
  --cwd /tmp/auto-reviewer-leaky-config-merge \
  --preset default \
  --max-iterations 2
```

Output lands under `./review_workspace/review_<timestamp>/`. Save the
`report.md` + `state.json` + `buffers/` to a timestamped subdirectory
of this fixture for cross-model comparison.

## Cleaning up

```bash
rm -rf /tmp/auto-reviewer-leaky-config-merge
rm -rf ./review_workspace
```
