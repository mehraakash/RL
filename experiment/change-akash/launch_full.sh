#!/bin/bash
# =============================================================================
# launch_full.sh — FULL-SCALE single-controller run of the akamehra port.
#
# 16 train + 32 generation + 12 gym = 60 nodes. Thin wrapper over
# nemotron-3.5-nano/nano35_dolphin_launch.sh with two deviations from
# akamehra's 64-node shape, both consequences of using the EXTERNAL GenRM
# pool (10 TP=8 replicas, group haitianj-nanov35):
#
#   1. NUM_GYM_NODES 16 -> 12: akamehra's 16 included 4 nodes for in-cluster
#      GenRM (TP4 x DP4). Judge capacity is otherwise identical
#      (nl2bash TP4 x DP8 = 8 nodes, safety 1 GPU, same spare).
#   2. REAPER_COMMENT exemption raised to 240 min: with GenRM external,
#      ~15 gym GPUs are legitimately idle for the whole job, and this
#      cluster's OccupiedIdleGPUsJobReaper kills idle-GPU jobs (default
#      exemption in ultra_launch.sh is 60 min).
#
# Batch shape is akamehra's production recipe, inherited unchanged:
# TP4 CP4 EP16 PP1, PPS=512 x GPP=16 = GBS=8192, seq 73728, save_period 5,
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

# --- Job shape: 16 train + 32 gen + 12 gym = 60 nodes (multiple of 2) --------
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-16}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-32}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-12}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"

export CONFIG_PATH="${CONFIG_PATH:-experiment/change-akash/nemotron-3.5-nano/rlvr_dolphin_extgenrm.yaml}"

# --- Reaper: idle gym GPUs are expected (external GenRM) ---------------------
export REAPER_COMMENT="${REAPER_COMMENT:-{\"OccupiedIdleGPUsJobReaper\":{\"exemptIdleTimeMins\":\"240\",\"reason\":\"data_loading\",\"description\":\"SC full-scale RLVR; gym judge nodes keep spare GPUs (GenRM external)\"}}}"

# --- External GenRM pool; probe before spending a 60-node allocation ---------
export GENRM_BASE_URL="${GENRM_BASE_URL:-http://10.109.26.53:9215/v1}"
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
echo "  SC FULL SCALE (akamehra port) — external GenRM"
echo "  Nodes: ${NUM_TRAIN_NODES} train + ${NUM_GEN_NODES} gen + ${NUM_GYM_NODES} gym"
echo "  Config: ${CONFIG_PATH}"
echo "=============================================================="

exec bash "${FULL_DIR}/nemotron-3.5-nano/nano35_dolphin_launch.sh" "$@"
