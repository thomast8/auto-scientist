#!/usr/bin/env bash
# Build a throwaway git repo with a "simplification" commit on a feature
# branch that silently wipes sibling config keys under nested env
# overrides. Harder than buggy-paginate: requires cross-file reasoning
# (src/config.py docstring + src/database.py caller) and a probe with
# a multi-key nested config to trigger.
#
# Usage:  ./setup_fixture.sh [dest_dir]
# Default destination: /tmp/auto-reviewer-leaky-config-merge
#
# Branches:
#   main                                - correct apply_env_overrides
#   feature/simplify-env-overrides      - buggy apply_env_overrides +
#                                         one unrelated added test

set -euo pipefail

DEST="${1:-/tmp/auto-reviewer-leaky-config-merge}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -e "$DEST" ]]; then
  echo "Destination $DEST already exists. Remove it or pass a different path." >&2
  exit 1
fi

mkdir -p "$DEST"
cd "$DEST"

git init -q -b main
git config user.email "fixture@auto-reviewer.local"
git config user.name "auto-reviewer-fixture"

# main: the correct version
cp -R "$HERE/good/." .
git add .
git commit -q -m "feat: config loader with env-var overrides

load_config() reads JSON from disk and applies env-var overrides.
Nested keys use underscore separators (DB_HOST overrides db.host).
Sibling keys are preserved across overrides.

src/database.py consumes the loaded config and reads three sibling
keys (host, port, name) under config['db']."

# feature branch: 'simplify' that wipes siblings
git checkout -q -b feature/simplify-env-overrides
cp "$HERE/buggy/src/config.py" src/config.py
cp "$HERE/buggy/tests/test_config.py" tests/test_config.py
git add .
git commit -q -m "refactor: simplify nested env-var handling

Consolidate the nested-override branch of apply_env_overrides so the
loop body is a single assignment for both the top-level and nested
shapes. Also adds a test confirming in-place mutation semantics."

echo
echo "Fixture ready at: $DEST"
echo "Branches:"
git --no-pager branch
echo
echo "Try the review with:"
echo
echo "    uv run auto-reviewer review \\"
echo "      \"review feature/simplify-env-overrides against main in the repo at $DEST\" \\"
echo "      --cwd $DEST \\"
echo "      --preset default \\"
echo "      --max-iterations 2"
echo
echo "Expected outcome (see expected.json):"
echo "  - 1 confirmed bug: apply_env_overrides wipes sibling keys"
echo "  - 0 phantom HIGH-priority claims"
echo "  - Design Intent should cite the docstring's 'sibling preservation' clause"
echo
