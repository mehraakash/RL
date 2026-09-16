# SWE-E2E telemetry backport

Base: `7cfa1bed795046e1c024c818e8db6ade5e22ca0e`.
Ports the telemetry from PR #4068 (`dcaf1cc017cb0df463071115b82e18e2c9f1ea66`)
and its per-environment valid-count extension onto the existing selected-row
metadata path. No newer controller, replay, masking, or advantage behavior is imported.

`train/environment/<name>/` includes reward, generation length, total tokens,
turns, maximum generation length per turn, truncation, and numeric environment
extras. Distributions include histograms and summaries; core distributions also
include p50/p95. Existing metric names remain available.

`num_samples`, `num_valid_samples`, and `num_valid_tokens` sum over the selected
streaming chunks. Valid counts use the actual final sample weights and weighted
next-token loss mask (excluding the first sequence position), respectively.

This base does not propagate Gym flags into SC's loss mask. Therefore the
diagnostic is called `num_env_flagged_samples`, not `num_mask_sample_filtered`.
It counts flags retained by the existing Gym configuration, without applying them.
Old replay rows without an environment are counted under `unknown`; missing
diagnostic tags suppress the affected distribution instead of reporting a partial one.

Validation: 108 CPU tests passed across SC helpers/controller/train pump, payload,
replay buffer, and rollout manager. One existing test,
`test_nemo_gym_small_task_identity_is_safe_for_rollout_histograms`, fails identically
on untouched `7cfa1bed…` because it expects a suppressed reserved Gym ID metric.
Test runtime: Python 3.13.13, CPU Torch 2.10.0, Transformers 5.6.0; no GPU smoke.
