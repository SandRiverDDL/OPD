#!/usr/bin/env bash
# Generic OPD launcher.
#
# Usage:
#   ./run_opd.sh configs/opd/experiments/opd_prompt32_step80.yaml
#   DRY_RUN=True ./run_opd.sh configs/opd/experiments/opd_prompt32_step80.yaml

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_PATH="${1:-${OPD_CONFIG:-configs/opd/experiments/opd_prompt32_step80.yaml}}"

if [[ -n "${OPD_LOG_FILE:-}" ]]; then
    mkdir -p "$(dirname -- "$OPD_LOG_FILE")"
    exec > >(tee -a "$OPD_LOG_FILE") 2>&1
elif [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p logs
    LOG_FILE="${LOG_FILE:-logs/opd_launcher_$(date +%Y%m%d_%H%M%S).log}"
    exec > >(tee -a "$LOG_FILE") 2>&1
fi

echo "OPD config: $CONFIG_PATH"
RUN_ARGS=(
    python3 scripts/opd/run_opd.py
    --repo-root "$SCRIPT_DIR"
    --config "$CONFIG_PATH"
)
if [[ "${DRY_RUN:-False}" == "True" ]]; then
    RUN_ARGS+=(--dry-run)
fi
exec "${RUN_ARGS[@]}"
