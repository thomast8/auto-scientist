#!/usr/bin/env bash
# Manual sandbox smoke test.
#
# Creates a throwaway git repo in /tmp, runs the full auto-reviewer CLI
# (spawning a real LLM session), and verifies the throwaway repo is
# byte-identical before and after.
#
# Exit codes:
#   0  success — real repo untouched
#   2  SANDBOX VIOLATION from verify_unchanged — sandbox bug, investigate
#   *  other failure — read the output
#
# Cost: a real LLM run. Set OPENAI/ANTHROPIC credentials appropriately.
# Expect ~10 minutes end-to-end.
#
# The automated test coverage lives in:
#   auto-reviewer/tests/safety/test_cli_no_mutation.py
#   tests/safety/test_hook_adversarial.py
# ...which exercise the sandbox without a live LLM. This script is the
# "did you really stress test this?" check before shipping changes to
# the sandbox.

set -euo pipefail

SMOKE_ROOT="${TMPDIR:-/tmp}/auto-reviewer-smoke-$$"
REAL_REPO="${SMOKE_ROOT}/real_repo"
WORKSPACE="${SMOKE_ROOT}/workspace"

cleanup() {
    rm -rf "${SMOKE_ROOT}"
}
trap cleanup EXIT

mkdir -p "${REAL_REPO}"
cd "${REAL_REPO}"
git init -q -b main
git config user.email "smoke@example.com"
git config user.name "Smoke"
git config commit.gpgsign false
printf 'alpha\n' > README.md
mkdir -p src
printf 'print("hi")\n' > src/main.py
git add .
git commit -q -m "init"

REAL_HEAD_BEFORE=$(git rev-parse HEAD)
REAL_STATUS_BEFORE=$(git status --porcelain=v1)
REAL_HASH_BEFORE=$(find . -type f -not -path './.git/*' -exec shasum -a 256 {} + | sort | shasum -a 256)

echo "=== running auto-reviewer against ${REAL_REPO} ==="
cd "${SMOKE_ROOT}"
uv run auto-reviewer review \
    "review the main branch" \
    --cwd "${REAL_REPO}" \
    --output-dir "${WORKSPACE}" \
    --max-iterations 1 \
    "$@"
CLI_RC=$?

echo "=== verifying ${REAL_REPO} is byte-identical ==="
cd "${REAL_REPO}"
REAL_HEAD_AFTER=$(git rev-parse HEAD)
REAL_STATUS_AFTER=$(git status --porcelain=v1)
REAL_HASH_AFTER=$(find . -type f -not -path './.git/*' -exec shasum -a 256 {} + | sort | shasum -a 256)

[[ "${REAL_HEAD_BEFORE}" = "${REAL_HEAD_AFTER}" ]] || { echo "HEAD moved"; exit 2; }
[[ "${REAL_STATUS_BEFORE}" = "${REAL_STATUS_AFTER}" ]] || { echo "status changed"; exit 2; }
[[ "${REAL_HASH_BEFORE}" = "${REAL_HASH_AFTER}" ]] || { echo "tree hash changed"; exit 2; }

echo "OK: real repo untouched (CLI exit ${CLI_RC})"
exit "${CLI_RC}"
