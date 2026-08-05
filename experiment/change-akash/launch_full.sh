#!/bin/bash
# =============================================================================
# launch_full.sh — SingleController in-order/lookahead-1 legacy control.
#
# 8 train + 4 generation + 4 Gym = 16 in-cluster nodes. GenRM stays in its
# separate external allocation.
#
# Batch shape matches the control:
# TP4 CP4 EP8 PP1, PPS=32 x GPP=16 = GBS=512, seq 73728, save_period 5,
# in-order lookahead 1, 64 prompt groups inflight/buffered,
# 4h walltime with checkpoint_must_save_by 03:35. max_num_steps is unbounded;
# the run trains until the save-by margin and auto-resumes via singleton on
# resubmission (same EXP_NAME => same checkpoint dir).
#
# Usage:
#   ./launch_full.sh
#   DRY_RUN=1 ./launch_full.sh     # print resolved TRAIN_CMD, don't submit
# =============================================================================
set -euo pipefail

FULL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Job shape: 8 train + 4 gen + 4 Gym = 16 nodes (multiple of 2) ------------
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-8}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-4}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-4}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"

export CONFIG_PATH="${CONFIG_PATH:-experiment/change-akash/nemotron-3.5-nano/rlvr_dolphin_extgenrm.yaml}"

# --- Reaper: match the legacy control's accepted 90-minute exemption --------
if [[ -z "${REAPER_COMMENT:-}" ]]; then
  export REAPER_COMMENT='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"data_loading","description":"nano35 SC rollout warmup and judge loading"}}'
fi

# --- External GenRM pool; probe before spending a 16-node allocation ---------
: "${GENRM_BASE_URL:?GENRM_BASE_URL must point to the external GenRM /v1 endpoint}"
export GENRM_BASE_URL
_genrm_health="${GENRM_BASE_URL%/v1}/health"
_health_json="$(curl -sf --max-time 10 "${_genrm_health}" || true)"
if [[ -z "${_health_json}" ]]; then
  echo "[ERROR] GenRM LB not responding at ${_genrm_health}." >&2
  exit 1
fi
_healthy="$(printf '%s' "${_health_json}" | grep -o '"healthy_backends": *[0-9]*' | grep -o '[0-9]*' || echo 0)"
if (( _healthy < 1 )); then
  echo "[ERROR] GenRM LB has no healthy backends: ${_health_json}" >&2
  exit 1
fi
echo "[GENRM] healthy backends: ${_healthy} at ${_genrm_health}"

echo "=============================================================="
echo "  SC IN-ORDER / LOOKAHEAD-1 — external GenRM"
echo "  Nodes: ${NUM_TRAIN_NODES} train + ${NUM_GEN_NODES} gen + ${NUM_GYM_NODES} gym"
echo "  Config: ${CONFIG_PATH}"
echo "=============================================================="

exec bash "${FULL_DIR}/nemotron-3.5-nano/nano35_dolphin_launch.sh" "$@"
