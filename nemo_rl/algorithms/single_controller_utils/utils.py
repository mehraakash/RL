# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers used by SingleControllerActor."""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.experience.interfaces import (
    NEMO_GYM_RESERVED_KEY_PREFIX,
    ROLLOUT_ENV_EXTRA_TAG_PREFIX,
    ROLLOUT_ENV_FLAG_TAG,
    ROLLOUT_ENVIRONMENT_TAG,
    ROLLOUT_GENERATION_LENGTH_TAG,
    ROLLOUT_MAX_GEN_TOKENS_TAG,
    ROLLOUT_REWARD_TAG,
    ROLLOUT_TOTAL_TOKENS_TAG,
    ROLLOUT_TRUNCATED_TAG,
    ROLLOUT_TURNS_TAG,
)
from nemo_rl.experience.metric_utils import calculate_single_metric

# Reduction rules for all_mb_metrics. Mirror grpo.py / grpo_sync.py.
_MB_METRIC_MIN: frozenset[str] = frozenset(
    {"probs_ratio_min", "probs_ratio_clamped_min"}
)
_MB_METRIC_MAX: frozenset[str] = frozenset(
    {"probs_ratio_max", "probs_ratio_clamped_max"}
)
_MB_METRIC_MEAN: frozenset[str] = frozenset(
    {
        "lr",
        "wd",
        "reward",
        "global_valid_seqs",
        "global_valid_toks",
        "mean_prompt_length",
    }
)
_METRIC_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")


def _rollout_environment_metric_component(environment: str) -> str:
    """Return a readable metric component without silent name collisions."""
    sanitized = _METRIC_COMPONENT_PATTERN.sub("_", environment).strip("_.")
    if sanitized == environment:
        return sanitized
    digest = hashlib.blake2s(environment.encode(), digest_size=8).hexdigest()
    return f"{sanitized or 'unknown'}-{digest}"


def _scalar_summary(
    values: list[float],
    prefix: str,
    *,
    mean_denominator: int | None = None,
) -> dict[str, float]:
    """Return the scalar portion of the legacy rollout metric family.

    V1 divides sparse ``env_extras`` sums by the full environment cohort, not
    by the number of samples that supplied that specific key. The optional
    denominator preserves that behavior without carrying W&B Histogram objects
    through checkpointed SingleController state.
    """
    denominator = mean_denominator if mean_denominator is not None else len(values)
    return {
        f"{prefix}/mean": sum(values) / denominator,
        f"{prefix}/max": max(values),
        f"{prefix}/min": min(values),
        f"{prefix}/median": statistics.median(values),
        f"{prefix}/stddev": statistics.stdev(values)
        if len(values) > 1
        else math.nan,
    }


def aggregate_step_metrics(train_result: dict[str, Any]) -> dict[str, Any]:
    """Reduce per-microbatch metric lists into step-level scalars.

    Args:
        train_result: Output of TQPolicy.finish_train_step.

    Returns:
        Flat dict of step-level scalars ready for logging.
    """
    metrics: dict[str, Any] = {}
    loss = train_result.get("loss")
    if isinstance(loss, torch.Tensor):
        metrics["loss"] = loss.detach().mean().item()
    elif loss is not None:
        metrics["loss"] = float(loss)
    grad_norm = train_result.get("grad_norm")
    if isinstance(grad_norm, torch.Tensor):
        metrics["grad_norm"] = grad_norm.detach().mean().item()
    elif grad_norm is not None:
        metrics["grad_norm"] = float(grad_norm)
    if "total_flops" in train_result:
        metrics["total_flops"] = float(train_result["total_flops"])
    if "num_ranks" in train_result:
        metrics["num_ranks"] = int(train_result["num_ranks"])

    # moe/mtp share the same reduction rules as all_mb_metrics in grpo.py.
    mb: dict[str, list[Any]] = {}
    if "moe_metrics" in train_result:
        mb.update({f"moe/{k}": v for k, v in train_result["moe_metrics"].items()})
    if "mtp_metrics" in train_result:
        mb.update({f"mtp/{k}": v for k, v in train_result["mtp_metrics"].items()})
    mb.update(train_result.get("all_mb_metrics", {}))

    for k, v in mb.items():
        if k in _MB_METRIC_MIN:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.min(valid)) if valid else -1.0
        elif k in _MB_METRIC_MAX:
            valid = [x for x in v if not np.isinf(x)]
            metrics[k] = float(np.max(valid)) if valid else -1.0
        elif k in _MB_METRIC_MEAN:
            metrics[k] = float(np.mean(v))
        else:
            metrics[k] = float(np.sum(v))
    return metrics


def reduce_advantage_pump_metrics(
    rewards: list[torch.Tensor],
    masked_advantages: list[torch.Tensor],
    sequence_lengths: list[int],
    seq_logprob_error_metrics: list[dict[str, float]] | None = None,
    environment_counts: list[dict[str, float]] | None = None,
    num_mask_sample_filtered: list[int] | None = None,
) -> dict[str, float]:
    """Reduce per-step accumulators from _advantage_stage into step scalars.

    Args:
        rewards: One tensor per advantage_stage call; each row a sample reward.
        masked_advantages: Token-masked advantages, one tensor per call.
        sequence_lengths: All input_lengths trained on this step.
        seq_logprob_error_metrics: Sequence-error metrics and their aggregation
            counts, one record per streaming chunk.
        environment_counts: Selected-row counts after the existing loss masks.
        num_mask_sample_filtered: Gym flag counts across selected chunks.

    Returns:
        Step-level reward, advantage, token-count, and optional sequence
        log-probability error metrics.
    """
    out: dict[str, float] = {}
    if rewards:
        reward_values = (
            torch.cat([reward.detach().flatten().cpu() for reward in rewards])
            .tolist()
        )
        if reward_values:
            out["reward"] = statistics.mean(reward_values)
            out.update(_scalar_summary(reward_values, "total_reward"))
    if masked_advantages:
        cat = torch.cat([a.flatten() for a in masked_advantages])
        if cat.numel() > 0:
            out["advantages/mean"] = float(cat.mean())
            out["advantages/max"] = float(cat.max())
            out["advantages/min"] = float(cat.min())
        else:
            out["advantages/mean"] = 0.0
            out["advantages/max"] = 0.0
            out["advantages/min"] = 0.0
    if sequence_lengths:
        out["total_num_tokens"] = float(sum(sequence_lengths))
    for counts in environment_counts or []:
        for key, value in counts.items():
            out[key] = out.get(key, 0.0) + value
    if num_mask_sample_filtered is not None:
        out["num_mask_sample_filtered"] = float(sum(num_mask_sample_filtered))
    if seq_logprob_error_metrics:
        out.update(_reduce_seq_logprob_error_metrics(seq_logprob_error_metrics))
    return out


def environment_sample_counts(
    tags: list[dict[str, Any]] | None,
    *,
    sample_mask: torch.Tensor,
    token_mask: torch.Tensor,
    mask_sample: torch.Tensor | None = None,
) -> dict[str, float]:
    """Observe final sample weights and weighted next-token targets; never mask."""
    size = sample_mask.numel()
    if tags is not None and len(tags) != size:
        raise ValueError("Environment tags must align with selected samples")
    counts: dict[str, float] = {}
    flags = (
        mask_sample.detach().cpu().tolist()
        if mask_sample is not None
        else [None] * size
    )
    for tag, weight, tokens, flagged in zip(
        tags if tags is not None else [{} for _ in range(size)],
        sample_mask.detach().cpu().tolist(),
        token_mask[:, 1:].sum(-1).detach().cpu().tolist(),
        flags,
        strict=True,
    ):
        environment = _rollout_environment_metric_component(
            tag.get(ROLLOUT_ENVIRONMENT_TAG, "unknown")
        )
        for name, value in (
            ("num_samples", 1.0),
            ("num_valid_samples", weight),
            ("num_valid_tokens", tokens),
        ):
            key = f"environment/{environment}/{name}"
            counts[key] = counts.get(key, 0.0) + value
        if flagged is not None:
            key = f"environment/{environment}/num_mask_sample_filtered"
            counts[key] = counts.get(key, 0.0) + int(flagged)
    return counts


def reduce_environment_rollout_metrics(
    tags: list[dict[str, Any]],
) -> dict[str, Any]:
    """Port #4068 distributions to this branch's selected-row metadata path."""
    cohorts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tag in tags:
        cohorts[tag.get(ROLLOUT_ENVIRONMENT_TAG, "unknown")].append(tag)
    metrics: dict[str, Any] = {}
    for environment, rows in cohorts.items():
        prefix = f"environment/{_rollout_environment_metric_component(environment)}"
        metrics[f"{prefix}/sample_count"] = len(rows)
        families = {
            ROLLOUT_REWARD_TAG: "total_reward",
            ROLLOUT_GENERATION_LENGTH_TAG: "gen_tokens_per_sample",
            ROLLOUT_TOTAL_TOKENS_TAG: "total_tokens_per_sample",
            ROLLOUT_TURNS_TAG: "turns_per_sample",
            ROLLOUT_MAX_GEN_TOKENS_TAG: "max_gen_tokens_per_turn",
            ROLLOUT_TRUNCATED_TAG: "truncated",
        }
        for tag_key, name in families.items():
            # Old replay rows must not silently create partial distributions.
            if not all(tag_key in row for row in rows):
                continue
            values = [float(row[tag_key]) for row in rows]
            metric = f"{prefix}/{name}"
            metrics.update(calculate_single_metric(values, len(rows), metric))
            metrics[f"{metric}/p50"] = float(np.percentile(values, 50))
            metrics[f"{metric}/p95"] = float(np.percentile(values, 95))
        if all(ROLLOUT_ENV_FLAG_TAG in row for row in rows):
            # Raw flags, before any overlapping training-time filters.
            metrics[f"{prefix}/num_env_flagged_samples"] = sum(
                bool(row[ROLLOUT_ENV_FLAG_TAG]) for row in rows
            )
        extra_keys = {
            key
            for row in rows
            for key in row
            if key.startswith(ROLLOUT_ENV_EXTRA_TAG_PREFIX)
        }
        for key in sorted(extra_keys):
            values = [float(row[key]) for row in rows if key in row]
            metric = (
                f"{prefix}/env_extra/{key.removeprefix(ROLLOUT_ENV_EXTRA_TAG_PREFIX)}"
            )
            metrics.update(calculate_single_metric(values, len(rows), metric))
    return metrics


def reduce_rollout_length_metrics(
    rollout_tags: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate generated-token lengths and rewards by rollout environment.

    The tags come from the ``KVBatchMeta`` chunks selected for one optimizer
    step, so streaming completion order cannot shift a sample into the wrong
    step's metrics.

    Args:
        rollout_tags: Per-sample metadata tags accumulated across train chunks.

    Returns:
        Global mean generation length, per-environment length summaries, and
        legacy-style per-environment reward summaries when their respective tags
        cover every sample. Missing tags are reported separately for lengths and
        rewards; incomplete cohorts suppress only the affected summary family.
    """
    by_environment: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    rewards_by_environment: dict[str, list[float]] = defaultdict(list)
    extras_by_environment: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    environment_sample_counts: dict[str, int] = defaultdict(int)
    missing_length_samples = 0
    missing_reward_samples = 0
    for tag in rollout_tags:
        environment = tag.get(ROLLOUT_ENVIRONMENT_TAG)
        valid_environment = isinstance(environment, str) and bool(environment)
        if valid_environment:
            environment_sample_counts[environment] += 1
        generation_length = tag.get(ROLLOUT_GENERATION_LENGTH_TAG)
        truncated = tag.get(ROLLOUT_TRUNCATED_TAG)
        if (
            not valid_environment
            or isinstance(generation_length, bool)
            or not isinstance(generation_length, (int, float))
            or not isinstance(truncated, bool)
        ):
            missing_length_samples += 1
        else:
            generation_length = float(generation_length)
            if not math.isfinite(generation_length) or generation_length < 0:
                missing_length_samples += 1
            else:
                by_environment[environment].append((generation_length, truncated))

        reward = tag.get(ROLLOUT_REWARD_TAG)
        if (
            not valid_environment
            or isinstance(reward, bool)
            or not isinstance(reward, (int, float))
        ):
            missing_reward_samples += 1
        else:
            reward = float(reward)
            if not math.isfinite(reward):
                missing_reward_samples += 1
            else:
                rewards_by_environment[environment].append(reward)

        if valid_environment:
            for tag_key, value in tag.items():
                if not tag_key.startswith(ROLLOUT_ENV_EXTRA_TAG_PREFIX):
                    continue
                metric_key = tag_key.removeprefix(ROLLOUT_ENV_EXTRA_TAG_PREFIX)
                if (
                    not metric_key
                    or metric_key.startswith(NEMO_GYM_RESERVED_KEY_PREFIX)
                    or not isinstance(value, (bool, int, float))
                ):
                    continue
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    extras_by_environment[environment][metric_key].append(
                        numeric_value
                    )

    tagged_length_samples = sum(len(rows) for rows in by_environment.values())
    tagged_reward_samples = sum(
        len(values) for values in rewards_by_environment.values()
    )
    total_length_samples = tagged_length_samples + missing_length_samples
    total_reward_samples = tagged_reward_samples + missing_reward_samples
    metrics: dict[str, float] = {
        "rollout_length/tagged_samples": float(tagged_length_samples),
        "rollout_length/missing_samples": float(missing_length_samples),
        "rollout_length/tag_coverage": (
            tagged_length_samples / total_length_samples
            if total_length_samples
            else 0.0
        ),
        "rollout_reward/tagged_samples": float(tagged_reward_samples),
        "rollout_reward/missing_samples": float(missing_reward_samples),
        "rollout_reward/tag_coverage": (
            tagged_reward_samples / total_reward_samples
            if total_reward_samples
            else 0.0
        ),
    }
    # Older checkpoints predate these tags. Mixing their rows with newly
    # generated rows and reporting only the tagged subset would make the first
    # resumed step look complete while excluding exactly the restored samples
    # under investigation.
    if not missing_length_samples and tagged_length_samples:
        all_lengths: list[float] = []
        for environment in sorted(by_environment):
            rows = by_environment[environment]
            lengths = np.asarray([length for length, _ in rows], dtype=np.float64)
            all_lengths.extend(lengths.tolist())
            metric_environment = _rollout_environment_metric_component(environment)
            prefix = f"rollout_length/{metric_environment}"
            metrics.update(
                {
                    f"{prefix}/count": float(len(rows)),
                    f"{prefix}/mean": float(np.mean(lengths)),
                    f"{prefix}/stddev": float(np.std(lengths)),
                    f"{prefix}/min": float(np.min(lengths)),
                    f"{prefix}/p50": float(np.percentile(lengths, 50)),
                    f"{prefix}/p95": float(np.percentile(lengths, 95)),
                    f"{prefix}/max": float(np.max(lengths)),
                    f"{prefix}/truncation_rate": sum(
                        truncated for _, truncated in rows
                    )
                    / len(rows),
                }
            )

        metrics["mean_gen_tokens_per_sample"] = float(np.mean(all_lengths))

    if not missing_reward_samples and tagged_reward_samples:
        for environment in sorted(rewards_by_environment):
            values = rewards_by_environment[environment]
            metric_environment = _rollout_environment_metric_component(environment)
            prefix = f"{metric_environment}/reward"
            metrics[f"{prefix}/count"] = float(len(values))
            metrics.update(_scalar_summary(values, prefix))

    for environment in sorted(extras_by_environment):
        metric_environment = _rollout_environment_metric_component(environment)
        denominator = environment_sample_counts[environment]
        for metric_key in sorted(extras_by_environment[environment]):
            values = extras_by_environment[environment][metric_key]
            prefix = f"{metric_environment}/{metric_key}"
            metrics[f"{prefix}/count"] = float(len(values))
            metrics.update(
                _scalar_summary(
                    values,
                    prefix,
                    mean_denominator=denominator,
                )
            )
    return metrics


def _reduce_seq_logprob_error_metrics(
    records: list[dict[str, float]],
) -> dict[str, float]:
    """Reduce sequence-error metrics across streaming chunks."""

    def reduce_range(
        *,
        count_key: str,
        max_key: str,
        mean_key: str,
        min_key: str,
    ) -> dict[str, float]:
        populated = [record for record in records if record[count_key] > 0]
        count = sum(record[count_key] for record in populated)
        if not count:
            return {max_key: 0.0, mean_key: 0.0, min_key: 0.0}
        return {
            max_key: max(record[max_key] for record in populated),
            mean_key: sum(record[mean_key] * record[count_key] for record in populated)
            / count,
            min_key: min(record[min_key] for record in populated),
        }

    reduced = reduce_range(
        count_key="_num_valid_seqs_before",
        max_key="max_seq_mult_prob_error",
        mean_key="mean_seq_mult_prob_error",
        min_key="min_seq_mult_prob_error",
    )
    reduced.update(
        reduce_range(
            count_key="_num_valid_seqs_after",
            max_key="max_seq_mult_prob_error_after_mask",
            mean_key="mean_seq_mult_prob_error_after_mask",
            min_key="min_seq_mult_prob_error_after_mask",
        )
    )

    masked_count = sum(record["num_masked_seqs_by_logprob_error"] for record in records)
    reduced["num_masked_seqs_by_logprob_error"] = int(masked_count)
    reduced["masked_correct_pct"] = (
        sum(
            record["masked_correct_pct"] * record["num_masked_seqs_by_logprob_error"]
            for record in records
        )
        / masked_count
        if masked_count
        else 0.0
    )
    return reduced


def tensor_field(data: TensorDict, field_name: str) -> torch.Tensor:
    """Read a tensor column from a TensorDict, depadding if nested.

    Args:
        data: TensorDict returned by the data plane.
        field_name: Column name to fetch.

    Returns:
        Dense tensor (nested columns are padded with zeros).
    """
    value = data[field_name]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor field {field_name!r}; got {type(value)}")
    if value.is_nested:
        return torch.nested.to_padded_tensor(value, padding=0)
    return value


def squeeze_trailing_unit_dim(value: torch.Tensor) -> torch.Tensor:
    """Drop a trailing dim of size 1 if present.

    Args:
        value: Input tensor.

    Returns:
        Tensor without the trailing unit dim.
    """
    if value.dim() >= 2 and value.shape[-1] == 1:
        return value.squeeze(-1)
    return value


def fields_for_put(meta: KVBatchMeta, fields: dict[str, torch.Tensor]) -> TensorDict:
    """Pack tensors for DataPlane put, re-nesting jagged rows when needed.

    Args:
        meta: Batch meta whose sequence_lengths drive the nesting.
        fields: Field name to dense tensor.

    Returns:
        TensorDict shaped for dp_client.put_samples.
    """
    packed: dict[str, torch.Tensor] = {}
    if meta.sequence_lengths is None:
        for field_name, value in fields.items():
            packed[field_name] = value.detach().contiguous()
        # pyrefly: ignore[bad-argument-type]
        return TensorDict(packed, batch_size=[meta.size])

    lengths = torch.tensor(meta.sequence_lengths, dtype=torch.long)
    for field_name, value in fields.items():
        if value.dim() >= 2 and value.shape[1] == int(lengths.max().item()):
            rows = [
                value[i, : int(lengths[i].item())].detach().contiguous()
                for i in range(meta.size)
            ]
            packed[field_name] = torch.nested.as_nested_tensor(
                rows,
                layout=torch.jagged,
            )
        else:
            packed[field_name] = value.detach().contiguous()
    # pyrefly: ignore[bad-argument-type]
    return TensorDict(packed, batch_size=[meta.size])
