#!/bin/bash
set -euo pipefail

# =============================================================================
# serve_genrm.sh — stand up the external GenRM pool for the nano-3.5 RLVR run
#
# Wraps geshen's genrm_server_manager.sh: copies it into our own space (it keeps
# .lb_pid_*, logs/ and a flock'd registry beside itself, so running theirs in
# place would collide with their pool), launches N vLLM workers, waits for the
# load balancer to report a healthy backend, and prints the URL to feed the
# training launcher.
#
# Usage:
#   bash examples/nemo_gym/nemotron-3.5-nano/serve_genrm.sh [N]     # default N=4
#   NUM_WORKERS=8 bash .../serve_genrm.sh
#   bash .../serve_genrm.sh --status | --url | --stop
#
# Each worker is a separate SLURM job: 2 nodes x 4 GPUs, TP=8. N=4 -> 8 nodes.
# This pool is SEPARATE from the training job's 64 nodes.
#
# ---------------------------------------------------------------------------
# WHERE THE URL COMES FROM — read this before running
# ---------------------------------------------------------------------------
# The load balancer is a plain nohup'd aiohttp process on THIS login node, and
# genrm_server_manager.sh derives its address as `hostname -I | awk '{print $1}'`
# — the first IP of whatever node you run this from. Consequences:
#
#   * Run this from a login node you intend to stay on. If that node reboots,
#     the LB dies and the training job's GenRM endpoint goes with it (the
#     watchdog restarts the process, not the node).
#   * The URL is an IP, not a DNS name. Compute nodes must be able to reach it.
#   * On a multi-homed login node `hostname -I` may pick the wrong interface.
#     Verify with --status before submitting training.
#
# The URL is also written to <serving-dir>/logs/<timestamp>/url on a successful
# launch, and `--url` recomputes and prints it at any time.
# =============================================================================

GENRM_SRC="${GENRM_SRC:-/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_ppo/users/geshen/mopd_nano_fast/genrm_serving}"
GENRM_DIR="${GENRM_DIR:-/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_llm/users/akamehra/genrm_serving}"

# step1230, the GenRM the dolphin reference used. NOT the manager's built-in
# default, which is the *ultra* GenRM (step_720) — always override it.
export MODEL="${MODEL:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"

export ACCOUNT="${ACCOUNT:-nemotron_sw_post}"
# `batch` by decision — everything uses it. Note the consequence: `batch` caps at
# 4 h, so the pool expires on the same cadence as training and does NOT stay warm
# across training restarts. The 470 GB bf16 Qwen3-235B reload therefore recurs
# either way, and the pool needs relaunching every 4 h.
# (batch_long, 7 d, is permitted for this account if that tradeoff is revisited.)
export PARTITION="${PARTITION:-batch}"
export TIME="${TIME:-4:00:00}"
export QOS="${QOS:-normal}"
export LB_PORT="${LB_PORT:-9213}"
export GENRM_GROUP_ID="${GENRM_GROUP_ID:-nano35_dolphin}"

NUM_WORKERS="${NUM_WORKERS:-${1:-4}}"
WAIT_FOR_BACKENDS="${WAIT_FOR_BACKENDS:-1}"   # proceed once this many are healthy
WAIT_TIMEOUT_SECS="${WAIT_TIMEOUT_SECS:-3600}"

lb_url() { echo "http://$(hostname -I | awk '{print $1}'):${LB_PORT}/v1"; }
health_url() { echo "http://$(hostname -I | awk '{print $1}'):${LB_PORT}/health"; }

case "${1:-}" in
  --url)    lb_url; exit 0 ;;
  --status) cd "${GENRM_DIR}" && exec ./genrm_server_manager.sh status ;;
  --stop)   cd "${GENRM_DIR}" && exec ./genrm_server_manager.sh stop ;;
esac

# --- Copy the serving stack into our space ----------------------------------
# Copy only the executable stack, never the whole tree: the source contains the
# owner's logs/ and registry state (which we must not inherit) and at least one
# unreadable dotfile (.claude/.cc-writes) that makes a recursive cp abort under
# `set -e`. Idempotent — re-running repairs a partial copy.
if [[ ! -x "${GENRM_DIR}/genrm_server_manager.sh" ]]; then
  echo "Copying GenRM serving stack -> ${GENRM_DIR}"
  mkdir -p "${GENRM_DIR}/logs"
  _copied=0
  for _f in genrm_server_manager.sh genrm_worker.sh genrm_lb.py lb_watchdog.sh \
            genrm_registry.sh genrm_auto_add.sh README.md; do
    if [[ -r "${GENRM_SRC}/${_f}" ]]; then
      cp -p "${GENRM_SRC}/${_f}" "${GENRM_DIR}/${_f}"
      _copied=$((_copied + 1))
    else
      echo "[WARN] not readable, skipping: ${GENRM_SRC}/${_f}" >&2
    fi
  done
  chmod +x "${GENRM_DIR}"/*.sh 2>/dev/null || true
  echo "  copied ${_copied} file(s)"
  if [[ ! -x "${GENRM_DIR}/genrm_server_manager.sh" ]]; then
    echo "[ERROR] genrm_server_manager.sh missing after copy — cannot continue." >&2
    exit 1
  fi
else
  echo "Using existing GenRM serving dir: ${GENRM_DIR}"
fi

cd "${GENRM_DIR}"

echo "================================================================"
echo "  GenRM pool — ${NUM_WORKERS} worker(s), $((NUM_WORKERS * 2)) nodes"
echo "================================================================"
echo "  Model     : ${MODEL}"
echo "  Account   : ${ACCOUNT}    Partition: ${PARTITION}    Time: ${TIME}"
echo "  Group     : ${GENRM_GROUP_ID}    LB port: ${LB_PORT}"
echo "  Serving   : ${GENRM_DIR}"
echo "  LB host   : $(hostname -I | awk '{print $1}')  ($(hostname))"
echo "================================================================"
echo ""

./genrm_server_manager.sh launch "${NUM_WORKERS}"

# --- Wait for a healthy backend ---------------------------------------------
# Polls the load balancer's own /health endpoint, NOT the SLURM controller.
URL="$(lb_url)"
HEALTH="$(health_url)"
echo ""
echo "Waiting for >=${WAIT_FOR_BACKENDS} healthy backend(s) at ${HEALTH}"
echo "(workers must allocate and load a 470 GB model — expect several minutes)"

deadline=$(( SECONDS + WAIT_TIMEOUT_SECS ))
while (( SECONDS < deadline )); do
  healthy=$(curl -s -m 5 "${HEALTH}" 2>/dev/null \
    | python3 -c 'import json,sys;
try: print(json.load(sys.stdin).get("healthy_backends",0))
except Exception: print(0)' 2>/dev/null || echo 0)
  if [[ "${healthy}" =~ ^[0-9]+$ ]] && (( healthy >= WAIT_FOR_BACKENDS )); then
    echo ""
    echo "================================================================"
    echo "  GenRM READY — ${healthy} healthy backend(s)"
    echo "================================================================"
    echo ""
    echo "  export GENRM_BASE_URL=${URL}"
    echo ""
    echo "  Then submit training:"
    echo "    GENRM_BASE_URL=${URL} \\"
    echo "      bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch.sh"
    echo ""
    echo "  Monitor : watch -n 5 'curl -s ${HEALTH} | python3 -m json.tool'"
    echo "  Scale   : cd ${GENRM_DIR} && ./genrm_server_manager.sh add 2"
    echo "  Stop    : bash \$0 --stop"
    echo "================================================================"
    exit 0
  fi
  sleep 20
done

echo ""
echo "[ERROR] No healthy backend after ${WAIT_TIMEOUT_SECS}s." >&2
echo "        Check queue state and worker logs:" >&2
echo "          cd ${GENRM_DIR} && ./genrm_server_manager.sh status" >&2
echo "          tail -f ${GENRM_DIR}/logs/*/vllm_*.log" >&2
echo "        LB URL would be: ${URL}" >&2
exit 1
