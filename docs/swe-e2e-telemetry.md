# SWE-E2E telemetry backport

Base: `7cfa1bed795046e1c024c818e8db6ade5e22ca0e`.
Ports the telemetry from PR #4068 (`dcaf1cc017cb0df463071115b82e18e2c9f1ea66`)
and its per-environment valid-count extension onto the existing selected-row
metadata path. The follow-up masking commit backports PR #3766
(`9b25508a3340bffdd8e3a2245ada72279fbc15d6`); no #3837 advantage-baseline changes
or newer replay/controller implementation is imported.

`train/environment/<name>/` includes reward, generation length, total tokens,
turns, maximum generation length per turn, truncation, and numeric environment
extras. Distributions include histograms and summaries; core distributions also
include p50/p95. Existing metric names remain available.

`num_samples`, `num_valid_samples`, and `num_valid_tokens` sum over the selected
streaming chunks. Valid counts use the actual final sample weights and weighted
next-token loss mask (excluding the first sequence position), respectively.

#3766 propagates Gym flags into SC's loss mask and supports configured
`grpo.overlong_filtering`. These masks compose with sequence-logprob filtering.
`num_mask_sample_filtered` now counts Gym flags globally and per environment;
overlapping filters do not double-count flags. `num_env_flagged_samples` remains
available as raw rollout telemetry. `env.should_mask_flagged_samples=false`
still disables Gym flag propagation. Masked rewards still enter the GRPO group
baseline/std, because #3837 is deliberately not included.
Pre-backport replay-buffer tensors lack the new raw flag columns: use fresh
rollouts (or the existing replay-free restore path), not old buffered tensors.
Old replay rows without an environment are counted under `unknown`; missing
diagnostic tags suppress the affected distribution instead of reporting a partial one.

Telemetry-only validation: 108 CPU tests passed across SC helpers/controller/train pump, payload,
replay buffer, and rollout manager. One existing test,
`test_nemo_gym_small_task_identity_is_safe_for_rollout_histograms`, fails identically
on untouched `7cfa1bed…` because it expects a suppressed reserved Gym ID metric.
Test runtime: Python 3.13.13, CPU Torch 2.10.0, Transformers 5.6.0; no GPU smoke.

#3766 backport validation: 169 distinct CPU tests passed across controller/helpers,
payload/replay, reward penalties, Gym flag gates, and wire codec. The 16 codec tests
ran separately without the unrelated Ray session fixtures, with pinned TransferQueue
`c51614308b68c8d7a87c9b3ef62d59e14c69bde2` and its `msgspec` import dependency.
Composition tests cover Gym flags, overlong filtering, logprob filtering, all-masked
batches, and rejection of any `valid_mask` advantage-estimator argument (#3837).
