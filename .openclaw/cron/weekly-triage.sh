#!/bin/bash
# Weekly Repo Triage — SPEC-REPO-METRIC §6
# Runs every Sunday at 00:00 UTC via OpenClaw cron
#
# Usage:
#   ./.openclaw/cron/weekly-triage.sh [workspace_path]
#
# Environment:
#   GITHUB_TOKEN — required for GitHub issue creation/labeling
#   OPENCLAW_WORKSPACE — defaults to /root/.openclaw/workspace

set -euo pipefail

WORKSPACE="${1:-${OPENCLAW_WORKSPACE:-/root/.openclaw/workspace}}"
REPO_ROOT="${WORKSPACE}/sunset-ecosystem"
TRIAGE_MODULE="${REPO_ROOT}/triage/weekly.py"

# Verify we're in a valid repo
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "ERROR: Cannot find sunset-ecosystem at ${REPO_ROOT}" >&2
    exit 1
fi

echo "=== Weekly Triage Started ==="
echo "Workspace: ${WORKSPACE}"
echo "Repo root: ${REPO_ROOT}"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Activate virtualenv if present
if [ -f "${WORKSPACE}/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${WORKSPACE}/.venv/bin/activate"
fi

# Ensure PYTHONPATH includes the repo
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Run health check first
echo "--- Health Check ---"
python -c "
from triage.metrics import run_health_check
score = run_health_check('${REPO_ROOT}')
print(f'Total: {score.total} ({score.traffic_light})')
print(f'  Freshness:       {score.freshness:.1f} / 30')
print(f'  Test Coverage:   {score.test_coverage:.1f} / 25')
print(f'  Documentation:   {score.documentation:.1f} / 15')
print(f'  Dependency:      {score.dependency_health:.1f} / 15')
print(f'  Issue Hygiene:   {score.issue_hygiene:.1f} / 15')
"
echo ""

# Run full triage with drift detection and repo duplicate scan
echo "--- Full Triage ---"
python -m triage.weekly \
    --owner SuperInstance \
    --repo sunset-ecosystem \
    --root "${REPO_ROOT}" \
    --workspace "${WORKSPACE}" \
    --report-format markdown \
    --output "${REPO_ROOT}/docs/triage/$(date -u +%Y-%m-%d).md"

echo ""
echo "=== Weekly Triage Complete ==="
echo "Report: ${REPO_ROOT}/docs/triage/$(date -u +%Y-%m-%d).md"
echo "Cache:  ${REPO_ROOT}/.triage_cache/"
