"""Cost-model expectations for trace step groups: what each kernel bucket should take.

The batch shape of a step is rebuilt from SGLang's detailed step annotations
(``step_span_utils.py::build_detailed_annotation_suffix``): per request group ``g_`` (decode,
verify) or ``c_`` (extend) carries ``(ΣN_q, ΣN_kv, ΣN_q², ΣN_q·N_kv)`` with ``N_kv = prefix + N_q``.
Expectations are kernel-only times (roofline ``max(compute, memory)`` per kernel, floored at the
device's minimum kernel duration; alpha-beta for collectives) because trace kernel durations
exclude launch gaps. Sampling runs outside the step span in SGLang and is excluded.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sglang_maestro.model.ops import BatchShape, OpTiming, StepCostModel

BUCKETS = ("attention", "gemm", "moe", "comm", "small_ops")

_BUCKET_OF: dict[str, str] = {
    "attention": "attention",
    "gemm": "gemm",
    "moe": "moe",
    "moe_routing": "moe",
    "comm": "comm",
    "norm": "small_ops",
    "activation": "small_ops",
    "rope": "small_ops",
    "kv_store": "small_ops",
    "quant": "small_ops",
    "elementwise": "small_ops",
    "copy": "small_ops",
}
"""Shared by trace kernel categories and cost-model op kinds (both use the same vocabulary)."""

COMPARABLE_STAGES = frozenset({"DECODE", "EXTEND", "MIXED", "VERIFY"})


def bucket_of(category_or_kind: str) -> str | None:
    return _BUCKET_OF.get(category_or_kind)


def measured_buckets(by_category_us: dict[str, float]) -> dict[str, float]:
    totals: dict[str, float] = dict.fromkeys(BUCKETS, 0.0)
    for category, value in by_category_us.items():
        bucket = _BUCKET_OF.get(category)
        if bucket is not None:
            totals[bucket] += value
    return totals


def _decode_shape(batch_size: int, annotations: dict[str, int]) -> BatchShape | None:
    new_tokens = annotations.get("g_sq")
    sum_kv = annotations.get("g_sk")
    if not new_tokens or sum_kv is None:
        return None
    sum_q_kv = annotations.get("g_sqsk", sum_kv)
    return BatchShape(
        num_seqs=max(batch_size, 1),
        new_tokens=new_tokens,
        sum_kv=float(max(sum_kv, new_tokens)),
        sum_q_kv=float(max(sum_q_kv, new_tokens)),
        logits_tokens=new_tokens,
        phase="decode",
    )


def _extend_shape(num_seqs: int, annotations: dict[str, int]) -> BatchShape | None:
    query = annotations.get("c_sq")
    sum_kv = annotations.get("c_sk")
    square = annotations.get("c_sqsq")
    cross = annotations.get("c_sqsk")
    if not query or sum_kv is None or square is None or cross is None:
        return None
    causal_pairs = cross - (square - query) / 2.0
    return BatchShape(
        num_seqs=max(num_seqs, 1),
        new_tokens=query,
        sum_kv=float(sum_kv),
        sum_q_kv=float(max(causal_pairs, query)),
        logits_tokens=max(num_seqs, 1),
        phase="prefill",
        prefix_tokens=float(max(sum_kv - query, 0)),
    )


def shape_from_step(
    stage: str,
    batch_size: int,
    extend_tokens: int | None,
    annotations: dict[str, int],
    context_len_hint: int | None = None,
) -> tuple[BatchShape | None, str]:
    """Rebuild the step's batch shape; returns ``(shape, source)``.

    ``source`` is ``"annotations"``, ``"context_hint"`` or a reason the shape is unknown.
    """
    if stage not in COMPARABLE_STAGES:
        return None, f"{stage} 步不与成本模型对比（草稿模型/空 batch/特殊模式）"
    if stage in {"DECODE", "VERIFY"}:
        shape = _decode_shape(batch_size, annotations)
        if shape is not None:
            return shape, "annotations"
        if context_len_hint and stage == "DECODE":
            return BatchShape.decode(batch_size, context_len_hint), "context_hint"
        return None, "缺少 detailed_annotations（g_sq/g_sk），也没有给 --context-len"
    if stage == "EXTEND":
        shape = _extend_shape(batch_size, annotations)
        if shape is not None:
            return shape, "annotations"
        if extend_tokens:
            per_seq = max(extend_tokens // max(batch_size, 1), 1)
            return BatchShape.prefill(batch_size, per_seq, 0), "context_hint"
        return None, "EXTEND 步缺少 toks 与 detailed_annotations"
    context_count = annotations.get("c", 0)
    generate_count = annotations.get("g", 0)
    extend = _extend_shape(context_count, annotations) if context_count else None
    decode = _decode_shape(generate_count, annotations) if generate_count else None
    if extend and decode:
        return extend.merged_with(decode), "annotations"
    if extend or decode:
        return extend or decode, "annotations"
    return None, "MIXED 步缺少 detailed_annotations"


@dataclass
class GroupExpectation:
    shape_source: str
    shape: dict[str, Any]
    buckets_us: dict[str, float]
    timings: list[OpTiming] = field(default_factory=list, repr=False)
    scale: float = 1.0
    """Calibration factor already applied to ``buckets_us`` but not to ``timings``."""

    @property
    def total_us(self) -> float:
        return sum(self.buckets_us.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape_source": self.shape_source,
            "shape": self.shape,
            "buckets_us": self.buckets_us,
            "total_us": self.total_us,
        }


def kernel_time_us(timing: OpTiming) -> float:
    """Launch-free time of one op instance (one layer), including the per-kernel floor."""
    return timing.kernel_us


def expect_shape(model: StepCostModel, shape: BatchShape, shape_source: str) -> GroupExpectation:
    estimate = model.estimate(shape, cuda_graph=True)
    buckets: dict[str, float] = defaultdict(float)
    for timing in estimate.timings:
        bucket = _BUCKET_OF.get(timing.op.kind)
        if bucket is None:
            continue
        buckets[bucket] += kernel_time_us(timing) * timing.op.layers * estimate.scale
    return GroupExpectation(
        shape_source=shape_source,
        shape={
            "phase": shape.phase,
            "num_seqs": shape.num_seqs,
            "new_tokens": shape.new_tokens,
            "sum_kv": shape.sum_kv,
            "sum_q_kv": shape.sum_q_kv,
            "prefix_tokens": shape.prefix_tokens,
        },
        buckets_us={bucket: buckets.get(bucket, 0.0) for bucket in BUCKETS},
        timings=estimate.timings,
        scale=estimate.scale,
    )
