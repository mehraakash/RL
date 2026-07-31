#!/bin/bash
# =============================================================================
# launch_smoke.sh — 6-node single-controller smoke for the akamehra port.
#
# 2 train + 2 generation + 2 gym nodes, 10 steps. Thin wrapper over
# nemotron-3.5-nano/nano35_dolphin_launch.sh: sets the small-scale knobs and
# points at rlvr_dolphin_smoke.yaml. Everything else (container, caches,
# mounts, SC entrypoint) comes from the parent chain.
#
# Usage:
#   ./launch_smoke.sh
#   DRY_RUN=1 ./launch_smoke.sh     # print resolved TRAIN_CMD, don't submit
#
# Prerequisite: the external GenRM pool must be serving (the 2-gym-node shape
# cannot fit the in-cluster TP=4 GenRM next to nl2bash TP=4 + safety judge).
# =============================================================================
set -euo pipefail

SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Job shape: 2 train + 2 gen + 2 gym = 6 nodes (multiple of 2) ------------
export NUM_TRAIN_NODES="${NUM_TRAIN_NODES:-2}"
export NUM_GEN_NODES="${NUM_GEN_NODES:-2}"
export NUM_GYM_NODES="${NUM_GYM_NODES:-2}"
export SEGMENT_SIZE="${SEGMENT_SIZE:-2}"

# --- Run length: 10 steps; save_period 5 (inherited) saves at steps 5 and 10 -
export NRL_MAX_STEPS="${NRL_MAX_STEPS:-10}"

# --- Identity: separate checkpoint/W&B/results tree from the full-scale run --
export EXP_NAME="${EXP_NAME:-haitianj-nano35-dolphin-sc-akash-smoke-tp4cp2ep4-pps16gpp4}"
export CONFIG_PATH="${CONFIG_PATH:-experiment/change-akash/nemotron-3.5-nano/rlvr_dolphin_smoke.yaml}"

# --- External GenRM (our own pool; LB on a login node, verified reachable ----
# --- from compute nodes). Probe it before spending a 6-node allocation. ------
export GENRM_BASE_URL="${GENRM_BASE_URL:-http://10.109.26.53:9215/v1}"
_genrm_health="${GENRM_BASE_URL%/v1}/health"
if ! curl -sf --max-time 10 "${_genrm_health}" >/dev/null; then
  echo "[ERROR] GenRM LB not healthy at ${_genrm_health}." >&2
  echo "  Relaunch the pool (nanov35-sc.md §7) or point GENRM_BASE_URL elsewhere." >&2
  exit 1
fi
echo "[GENRM] healthy: ${_genrm_health}"

echo "=============================================================="
echo "  SC SMOKE (akamehra port) — wiring validation, not convergence"
echo "  Nodes: ${NUM_TRAIN_NODES} train + ${NUM_GEN_NODES} gen + ${NUM_GYM_NODES} gym"
echo "  Steps: ${NRL_MAX_STEPS}   Config: ${CONFIG_PATH}"
echo "=============================================================="

exec bash "${SMOKE_DIR}/nemotron-3.5-nano/nano35_dolphin_launch.sh" "$@"
