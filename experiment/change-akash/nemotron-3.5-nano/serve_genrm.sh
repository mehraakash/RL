#!/bin/bash
set -euo pipefail

# Launch the external Nano-3.5 GenRM on compute nodes. Each replica is one
# 2-node/8-GPU TP8 Slurm job; it is separate from the 16-node SC allocation.
#
# Usage:
#   bash experiment/change-akash/nemotron-3.5-nano/serve_genrm.sh [N]
#   bash experiment/change-akash/nemotron-3.5-nano/serve_genrm.sh --wait [SECONDS]
#   bash experiment/change-akash/nemotron-3.5-nano/serve_genrm.sh --status
#   bash experiment/change-akash/nemotron-3.5-nano/serve_genrm.sh --url
#   bash experiment/change-akash/nemotron-3.5-nano/serve_genrm.sh --stop

case "${1:-}" in
  --help|-h)
    sed -n '4,12p' "$0"
    exit 0
    ;;
esac

# Site-private service, model, and scheduler values belong in the internal
# runbook, not in git. Management operations need the service and group so the
# underlying manager cannot silently operate on its default Ultra group.
: "${GENRM_SERVICE:?Set GENRM_SERVICE from the internal runbook}"
: "${GENRM_GROUP_ID:?Set GENRM_GROUP_ID from the internal runbook}"
export GENRM_GROUP_ID
MANAGER="${GENRM_SERVICE}/manage.sh"
if [[ ! -x "${MANAGER}" ]]; then
  echo "[ERROR] GenRM manager is not executable: ${MANAGER}" >&2
  exit 1
fi

case "${1:-}" in
  --wait) exec "${MANAGER}" wait "${2:-3600}" ;;
  --status) exec "${MANAGER}" status ;;
  --url) exec "${MANAGER}" url ;;
  --stop) exec "${MANAGER}" stop ;;
esac

required_vars=(MODEL REASONING_PARSER_NAME ACCOUNT PARTITION QOS TIME)
for var_name in "${required_vars[@]}"; do
  if [[ -z "${!var_name:-}" ]]; then
    echo "[ERROR] Set ${var_name} from the internal runbook" >&2
    exit 1
  fi
done
if [[ -z "${REASONING_PARSER+x}" ]]; then
  echo "[ERROR] Set REASONING_PARSER from the internal runbook (it may be empty)" >&2
  exit 1
fi
export MODEL REASONING_PARSER REASONING_PARSER_NAME
export ACCOUNT PARTITION QOS TIME
export LB_PORT="${LB_PORT:-9213}"

# Protect the idle interval between model readiness and the first routed batch.
if [[ -z "${REAPER_COMMENT:-}" ]]; then
  export REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"data_loading","description":"external GenRM serving for nano35 SingleController"}}'
fi

NUM_WORKERS="${NUM_WORKERS:-${1:-1}}"
WAIT_TIMEOUT_SECS="${WAIT_TIMEOUT_SECS:-3600}"

echo "Launching ${NUM_WORKERS} external GenRM replica(s)."
"${MANAGER}" launch "${NUM_WORKERS}"
GENRM_BASE_URL="$("${MANAGER}" wait "${WAIT_TIMEOUT_SECS}")"

echo "GenRM ready: ${GENRM_BASE_URL}"
echo "export GENRM_BASE_URL=${GENRM_BASE_URL}"
