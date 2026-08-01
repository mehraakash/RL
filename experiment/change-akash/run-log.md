# change-akash — akamehra's nano35 dolphin recipe on the single controller

Chronological record of adapting akamehra's legacy async-1 launch chain to the
single-controller (SC) architecture, every failure hit, the fix, and the
result. Same append-only convention as `experiment/nanov35-run-log.md` (whose
F1–F19 catalogue this port builds on): new entries go at the very end.

- **Branch:** `sc-main-ckpt-exp`
- **Source of truth being adapted:**
  `/lustre/fsw/portfolios/coreai/users/akamehra/code/RL/examples/nemo_gym/`
  `nemotron-3.5-nano/` + inherited `nemotron-3-ultra/` (copied verbatim in
  commit `865bfb4b1`, then modified — `git diff 865bfb4b1 -- experiment/change-akash`
  shows exactly what changed).
- **Constraint:** fix problems via config/env, not code changes. Keep
  akamehra's values except (a) what the SC path requires, (b) W&B
  personalization (joc/nanov35-sc), (c) the small-scale smoke shape.

---

## What was copied (task 1)

| file | role |
|---|---|
| `nemotron-3.5-nano/nano35_dolphin_launch.sh` | thin wrapper: model/data/judges/env, delegates to ultra_launch.sh |
| `nemotron-3.5-nano/rlvr_dolphin.yaml` | nano recipe, inherits `../nemotron-3-ultra/student_rlvr1.yaml` |
| `nemotron-3.5-nano/serve_genrm.sh` | akamehra's GenRM serving notes/script (reference, unused here) |
| `nemotron-3-ultra/ultra_launch.sh` | the real launcher: sbatch, mounts, caches, TRAIN_CMD |
| `nemotron-3-ultra/student_rlvr1.yaml` | 34-env RLVR base config |

Directory structure preserved so the relative `defaults:` inheritance works
unchanged.

## SC adaptation (task 2) — every deliberate change

**`rlvr_dolphin.yaml`** (all marked `SC CHANGE` in-file):

| change | why |
|---|---|
| `grpo.val_period: 10000000 → 0` | SC has no validation path; `setup_single_controller` raises when `val_period > 0` (setup.py:398) |
| `grpo.skip_reference_policy_logprobs_calculation: true` | KL penalty is 0.0 → no reference model is built, but SC computes ref logprobs unless told not to → `AttributeError: reference_state_dict` (F19) |
| `policy.draft` block added (disabled) | `run_grpo_single_controller.py:111` reads `policy["draft"]["enabled"]` unconditionally — KeyError without it |
| top-level `async_rl` block added | required field of SC `MasterConfig`; windowed sampler, staleness 4, `max_buffered_rollouts` 2560 = 512×5 — values from the proven SC reference `experiment/configs/grpo_ultra_512n4g_bf16.yaml` |
| top-level `data_plane` block added | required field of SC `MasterConfig`; TransferQueue, `simple` backend, 2 storage units — same source |
| `env.nemo_gym.{policy_model,policy_model_reasoning_off}...num_groups_nemo_rl: 2` (literalized) | **caught statically, would have killed the first job at config load**: the inherited value is `${add:${grpo.async_grpo.max_trajectory_age_steps}, 1}`, but THIS branch's `register_omegaconf_resolvers` registers only `mul/div/max` — no `add` (akamehra's public-main tree has `add`). Resolution dies with `UnsupportedInterpolationType: add`. 2 is exactly what akamehra's config resolves to, and the key is dead config anyway (zero readers in the pinned Gym 473f446f or in nemo_rl) |
| `logger.wandb`: project `nanov35-sc`, entity `joc` declared | W&B personalization; declaring `entity` gives launcher overrides a key to write into (F16 lesson) |

**`ultra_launch.sh`:**

| change | why |
|---|---|
| entrypoint `examples/nemo_gym/run_grpo_nemo_gym.py → examples/run_grpo_single_controller.py` | the actual SC switch |
| overlay `examples/configs → examples` (whole tree) | the container image (public main 2026-07-26) predates the SC entrypoint; only our checkout has it |
| overlay `experiment/` added | so `CONFIG_PATH=experiment/change-akash/...` resolves inside the container after `cd /opt/nemo-rl` |

Kept unchanged: container (yifuw prebaked `..._20260726_b`), scoped-overlay
strategy (uv resolves the CONTAINER's lock → no Ray drift, the F3/F4/F7
class is structurally impossible), Gym overlay (both sides are 473f446f),
`SANDBOX_COMMAND=""` disable, `PYTHONDONTWRITEBYTECODE=1`, singleton
dependency, cache seeding machinery.

**`nano35_dolphin_launch.sh`:**

| change | why |
|---|---|
| GENRM_BASE_URL hard-error → allowed | akamehra's error was about THEIR unreachable LB (10.244.5.76, job 5733756); our pool's LB (10.109.26.53:9215, login node) is verified reachable from compute nodes. Full-scale keeps in-cluster GenRM by leaving GENRM_BASE_URL unset |
| `EXP_NAME`, `RESULTS_DIR`, `PERSISTENT_CACHE`, `HF_HOME`, `WANDB_PROJ` personalized | akamehra's paths are theirs; HF_HOME reuses the tree where the 62 GB HF→Megatron conversion is already cached |
| `EXTRA_MOUNTS` examples/nemo_gym block removed | superseded by the full `examples/` overlay |
| delegate to `experiment/change-akash/nemotron-3-ultra/ultra_launch.sh` | our SC-adapted copy |

## Smoke shape (task 3) — new files

`rlvr_dolphin_smoke.yaml` (inherits the SC-adapted `rlvr_dolphin.yaml`) +
`launch_smoke.sh`:

- **2 train + 2 gen + 2 gym nodes**, SEGMENT_SIZE=2, 10 steps,
  save_period 5 (inherited) → checkpoints at steps 5 and 10.
- **TP=4 CP=2 PP=1 EP=4 ETP=1** (world 8 ⇒ DP=1; MoE needs EP×ETP == TP×DP).
  CP=2, not 1: `make_sequence_length_divisible_by` and trainer memory at the
  full 73728 window both demand it (F15/F17 lessons).
- **PPS=16 × GPP=4 = GBS=64** (SC validates PPS×GPP == GBS). PPS=16 so the
  deterministic head-of-blend walk actually routes rows to GenRM.
- **Sequence length stays 73728** — the blend is built from `*_len40k`
  datasets; an invented shorter window makes vLLM reject real prompts (F15).
- **GenRM EXTERNAL** (base_url deep-merged onto the inherited in-cluster
  block; `LocalVLLMModel` skips the local vLLM launch when base_url is set,
  local-serving keys become inert). 2 gym nodes cannot fit in-cluster GenRM
  TP4 (4 GPUs) + nl2bash TP4 (4) + safety (1) = 9 > 8 GPUs. nl2bash DP 8→1.
- `async_rl.max_buffered_rollouts` 2560→80 (=16×5), diagnostics on.

## Pre-flight validation (no allocation spent)

- Real-loader config check (`load_config` + the exact ultra_launch override
  list + full interpolation resolution) on a login node: **all 21 SC-required
  keys present with expected values; PPS×GPP==GBS holds**. This is what caught
  the `${add:}` resolver failure above.
- Key-path diff vs the proven SC reference config: 19 paths present there but
  absent here — all Gym-side tunables (gzip/response-cache/keepalive knobs,
  judge `server_env`/`ray_worker_py_executable` variants). All are
  akamehra-proven on the same Gym pin + container; left at akamehra's values.
- `experiment/validate_gym_refs.py` on both new configs: 33 config_paths, all
  exist; all server refs resolve (F5/F6 class clean). Note: `ether0` is kept
  (akamehra keeps it and their runs pass spinup with the prebaked gym venvs;
  the old F11–F13 failures were on a different container + Lustre venv tree).
- `bash -n` clean on all three scripts; `DRY_RUN=1` end-to-end resolves the
  expected TRAIN_CMD (SC entrypoint, smoke config, external GenRM override,
  full mounts).
- GenRM pool live: 1 healthy backend, `/v1/models` serves id `model`,
  worker job 5744069 has ~19 h walltime left.
- Intel from the prior campaign's final run (job 5748614, cancelled
  externally at 26 min to make way for this work): it had cleared F19 and was
  **collecting rollouts** when killed (W&B `joc/nanov35-sc/e7cmxmtp`; the
  TCPStore errors in its log are post-scancel teardown noise). So the SC
  stack downstream of setup is known-good up to rollout collection; the
  never-yet-observed territory is: completed train steps, weight refit, and
  SC checkpoint saves.

---

# ═══ Run log (chronological) ═══

---

## Attempt 1 — job 5749373

Submitted 2026-07-31 16:34 via `experiment/change-akash/launch_smoke.sh`.
6 nodes (2+2+2), 10 steps, TP4 CP2 EP4, PPS16×GPP4=GBS64, seq 73728, external
GenRM (pool probed healthy at submit). W&B: joc/nanov35-sc /
`haitianj-nano35-dolphin-sc-akash-smoke-tp4cp2ep4-pps16gpp4`.

- Slurm logs: `.../akash-results/<exp>/runs/20260731-1634/slurm/`
- Ray logs: `.../akash-results/<exp>/ray_logs/5749373-logs/`

**Outcome — SUCCESS on the first attempt. Zero runtime failures.**

`sacct`: `COMPLETED 0:0`, elapsed **38:59**. From the driver log:

- pending 16:35 → running 16:38; config loaded, `MasterConfig` validated, all
  overrides applied cleanly (the `${add:}` literalization held — no
  `UnsupportedInterpolationType`).
- Megatron workers reused the cached HF→Megatron conversion from `HF_HOME`
  ("Checkpoint already exists … Skipping import"), vLLM engines up, all ~70
  Gym servers up — **including `ether0`**, vindicating "keep akamehra's list"
  (the old F11–F13 failures do not reproduce with the prebaked-venv container).
- W&B: https://wandb.ai/joc/nanov35-sc/runs/d8ylx51m
- `rollout_pump: starting` ~17:00, then the never-before-reached territory:

  ```
  train step 1/10  trainer_v=1  lag=0
  ...
  Saving checkpoint for step 5...
  train step 5/10  trainer_v=5  lag=4
  ...
  Saving checkpoint for step 10...
  train step 10/10 trainer_v=10 lag=1
  SC run complete: {'train_steps': 10, 'trainer_version': 10}
  ```

  All **10 steps** completed (~15 min of training after ~25 min setup), lag
  values 0–4 show the windowed staleness sampler actually cycling through its
  window, and weight refit clearly worked across steps.
- **Checkpoints verified on disk**: `checkpoints/step_5/` and
  `checkpoints/step_10/`, 368 GB each — `policy/` (model + optimizer),
  `replay_buffer.pt` (windowed sampler's buffer round-trip),
  `train_dataloader.pt`, `config.yaml`, `training_info.json`, plus
  `latest_checkpoint_status.json`.
- Post-success teardown noise, recorded so nobody chases it: repeated
  `Error in GPU monitoring collection loop … 'NoneType' object has no
  attribute 'log'` (the monitor outliving the closed wandb logger), the Gym
  1-second shutdown-timeout kill of all ~70 servers, and
  `Error during graceful shutdown … Falling back to force termination` — all
  AFTER `SC run complete`, exit code still 0.

The one genuine launch-killer of this port (the missing `add` omegaconf
resolver) was caught by the login-node pre-flight and never cost an
allocation. Goal met: akamehra's recipe launches and runs successfully on the
single controller.

**Next (not yet run):** full-scale launch via
`nemotron-3.5-nano/nano35_dolphin_launch.sh` (16+32+16 nodes, in-cluster
GenRM, PPS=512, GBS=8192). Before that: re-check the external-GenRM smoke
assumptions don't leak in (full-scale leaves `GENRM_BASE_URL` unset →
in-cluster GenRM via `GENRM_MODEL`), and mind that `async_rl.max_buffered_rollouts`
is already 2560 in `rlvr_dolphin.yaml` for the 512-prompt shape.
