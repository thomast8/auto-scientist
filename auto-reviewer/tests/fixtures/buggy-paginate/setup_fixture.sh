#!/usr/bin/env bash
# Build a throwaway git repo with a known bug on a feature branch, ready for
# auto-reviewer to chew on.
#
# Usage:  ./setup_fixture.sh [dest_dir]
# Default destination: /tmp/auto-reviewer-buggy-paginate
#
# The resulting repo has two branches:
#   main              - clean paginate() implementation
#   feature/tighten-paginate - introduces an off-by-one: returns one extra
#                              item per page. Looks plausible on review.

set -euo pipefail

DEST="${1:-/tmp/auto-reviewer-buggy-paginate}"
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

# main: the clean version
cp -R "$HERE/good/." .
git add .
git commit -q -m "feat: paginate + page_count helpers"

# feature branch: introduces an off-by-one
git checkout -q -b feature/tighten-paginate
cp "$HERE/buggy/src/paginate.py" src/paginate.py
git add .
git commit -q -m "perf: tighten paginate slice bounds

We were excluding the final element under some conditions. Widen the
slice window by one so it reliably includes page_size items."

echo
echo "Fixture ready at: $DEST"
echo "Branches:"
git --no-pager branch
echo
echo "Try the review with:"
echo
echo "    uv run auto-reviewer review \\"
echo "      --pr feature/tighten-paginate \\"
echo "      --repo-path $DEST \\"
echo "      --base-ref main \\"
echo "      --goal 'Find correctness bugs in the paginate slice change' \\"
echo "      --max-iterations 2 \\"
echo "      --critics ''"
echo
