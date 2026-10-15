"""Per-step and per-shape-group timing metrics computed from a parsed trace."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any

from sglang_maestro.perf.classify import COMPUTE_CATEGORIES
from sglang_maestro.perf.trace import (
    DECODE_LIKE_STAGES,
    KernelEvent,
    ParsedTrace,
    StepSpan,
    TraceDialect,
    intersection_length,
    merge_intervals,
    short_kernel_name,
    total_length,
)

GAP_EVIDENCE_THRESHOLD_US = 10.0
TOP_GAPS_PER_STEP = 5
TOP_KERNELS_PER_GROUP = 15


def quantile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


@dataclass(frozen=True)
class Distribution:
    count: int
    mean: float
    median: float
    p90: float
    p99: float
    minimum: float
    maximum: float
    std: float

    @property
    def cv(self) -> float:
        return self.std / self.mean if self.mean > 0 else 0.0

    @staticmethod
    def of(values: list[float]) -> Distribution:
        if not values:
            return Distribution(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        ordered = sorted(values)
        mean = sum(ordered) / len(ordered)
        variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)
        return Distribution(
            count=len(ordered),
            mean=mean,
            median=quantile(ordered, 0.5),
            p90=quantile(ordered, 0.9),
            p99=quantile(ordered, 0.99),
            minimum=ordered[0],
            maximum=ordered[-1],
            std=math.sqrt(variance),
        )


@dataclass(frozen=True)
class Gap:
    start_us: float
    duration_us: float
    before: str
    after: str
    before_kind: str
    before_category: str
    after_category: str


@dataclass
class StepMetrics:
    step: StepSpan
    kernel_count: int
    span_us: float
    busy_us: float
    kernel_sum_us: float
    by_category_us: dict[str, float]
    comm_us: float
    overlapped_comm_us: float
    streams: int
    graph_fraction: float | None
    idle_us: float
    host_sync_idle_us: float
    gaps: list[Gap]

    @property
    def exposed_comm_us(self) -> float:
        return self.comm_us - self.overlapped_comm_us

    @property
    def idle_ratio(self) -> float:
        return self.idle_us / self.span_us if self.span_us > 0 else 0.0


def compute_step_metrics(
    step: StepSpan, kernels: list[KernelEvent], dialect: TraceDialect | None = None
) -> StepMetrics:
    """``kernels`` must be the step's kernels sorted by start time."""
    dialect = dialect or TraceDialect()
    by_category: dict[str, float] = defaultdict(float)
    comm_intervals: list[tuple[float, float]] = []
    compute_intervals: list[tuple[float, float]] = []
    streams: set[tuple[int, int]] = set()
    graph_launched = known_launch = 0
    gaps: list[Gap] = []
    idle = host_sync_idle = 0.0
    union_end = None
    union_owner: KernelEvent | None = None
    busy = kernel_sum = 0.0
    union_start = None
    for kernel in kernels:
        start, end = kernel.start_us, kernel.start_us + kernel.duration_us
        kernel_sum += kernel.duration_us
        by_category[kernel.category] += kernel.duration_us
        streams.add((kernel.device, kernel.stream))
        if kernel.launch_api is not None:
            known_launch += 1
            if kernel.launch_api in dialect.graph_launch_names:
                graph_launched += 1
        if kernel.category == "comm":
            comm_intervals.append((start, end))
        elif kernel.category in COMPUTE_CATEGORIES:
            compute_intervals.append((start, end))
        if union_end is None:
            union_start, union_end, union_owner = start, end, kernel
            continue
        if start > union_end:
            duration = start - union_end
            idle += duration
            if union_owner is not None and union_owner.kind == "memcpy":
                host_sync_idle += duration
            if duration >= GAP_EVIDENCE_THRESHOLD_US and union_owner is not None:
                gaps.append(
                    Gap(
                        start_us=union_end,
                        duration_us=duration,
                        before=short_kernel_name(union_owner.name),
                        after=short_kernel_name(kernel.name),
                        before_kind=union_owner.kind,
                        before_category=union_owner.category,
                        after_category=kernel.category,
                    )
                )
            busy += union_end - union_start
            union_start, union_end, union_owner = start, end, kernel
        elif end > union_end:
            union_end, union_owner = end, kernel
    span = 0.0
    if union_end is not None and union_start is not None:
        busy += union_end - union_start
        span = union_end - kernels[0].start_us
    comm_merged = merge_intervals(sorted(comm_intervals))
    compute_merged = merge_intervals(sorted(compute_intervals))
    gaps.sort(key=lambda gap: -gap.duration_us)
    return StepMetrics(
        step=step,
        kernel_count=len(kernels),
        span_us=span,
        busy_us=busy,
        kernel_sum_us=kernel_sum,
        by_category_us=dict(by_category),
        comm_us=total_length(comm_merged),
        overlapped_comm_us=intersection_length(comm_merged, compute_merged),
        streams=len(streams),
        graph_fraction=graph_launched / known_launch if known_launch else None,
        idle_us=idle,
        host_sync_idle_us=host_sync_idle,
        gaps=gaps[:TOP_GAPS_PER_STEP],
    )


@dataclass(frozen=True)
class KernelStat:
    name: str
    short_name: str
    category: str
    comm_pattern: str | None
    fused: tuple[str, ...]
    grid: tuple[int, ...] | None
    count_per_step: float
    time_per_step_us: float
    share: float
    durations: Distribution


@dataclass
class GroupSummary:
    """All steps of one rank with the same stage and batch shape."""

    rank: str
    stage: str
    label: str
    batch_size: int
    extend_tokens: int | None
    annotations: dict[str, int]
    step_ids: list[int]
    span: Distribution
    busy_us: float
    idle_ratio: float
    kernels_per_step: float
    kernel_launches_per_step: float
    graph_launches_per_step: float
    sync_calls_per_step: float
    memcpy_calls_per_step: float
    graph_fraction: float | None
    streams: int
    by_category_us: dict[str, float]
    comm_us: float
    exposed_comm_us: float
    host_sync_idle_us: float
    gap_to_next: Distribution | None
    top_kernels: list[KernelStat]
    largest_gaps: list[Gap]
    expected: dict[str, Any] | None = None
    kernel_durations: dict[tuple, list[tuple[float, int, int]]] = field(
        default_factory=dict, repr=False
    )

    @property
    def steps(self) -> int:
        return len(self.step_ids)

    @property
    def is_decode_like(self) -> bool:
        return self.stage in DECODE_LIKE_STAGES

    @property
    def step_us(self) -> float:
        return self.span.median

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("kernel_durations", None)
        payload["steps"] = self.steps
        return payload


def _group_key(step: StepSpan) -> tuple:
    return (step.rank, step.stage, step.batch_size, step.extend_tokens)


def summarize_groups(
    trace: ParsedTrace,
    metrics_by_step: dict[int, StepMetrics],
    kernels_by_step: dict[int, list[KernelEvent]],
) -> list[GroupSummary]:
    grouped: dict[tuple, list[StepSpan]] = defaultdict(list)
    for step in trace.steps:
        if step.step_id in metrics_by_step and metrics_by_step[step.step_id].kernel_count:
            grouped[_group_key(step)].append(step)

    next_gap: dict[int, float] = {}
    for rank in trace.ranks:
        ordered = [
            step
            for step in trace.steps_for_rank(rank)
            if step.gpu_start_us is not None and step.step_id in metrics_by_step
        ]
        ordered.sort(key=lambda item: item.gpu_start_us or 0.0)
        for current, following in pairwise(ordered):
            if current.gpu_end_us is not None and following.gpu_start_us is not None:
                next_gap[current.step_id] = max(following.gpu_start_us - current.gpu_end_us, 0.0)

    summaries: list[GroupSummary] = []
    for (_rank, _stage, _batch, _tokens), steps in grouped.items():
        steps.sort(key=lambda item: item.ordinal)
        step_metrics = [metrics_by_step[step.step_id] for step in steps]
        count = len(steps)
        by_category: dict[str, float] = defaultdict(float)
        for metric in step_metrics:
            for category, value in metric.by_category_us.items():
                by_category[category] += value / count
        durations: dict[tuple, list[tuple[float, int, int]]] = defaultdict(list)
        names: dict[tuple, KernelEvent] = {}
        for step in steps:
            for index, kernel in enumerate(kernels_by_step.get(step.step_id, [])):
                signature = kernel.signature
                durations[signature].append((kernel.duration_us, step.step_id, index))
                names.setdefault(signature, kernel)
        total_kernel_us = sum(metric.kernel_sum_us for metric in step_metrics) / count or 1.0
        stats = []
        for signature, values in durations.items():
            sample = names[signature]
            total = sum(value for value, _, _ in values)
            stats.append(
                KernelStat(
                    name=sample.name,
                    short_name=short_kernel_name(sample.name),
                    category=sample.category,
                    comm_pattern=sample.comm_pattern,
                    fused=sample.fused,
                    grid=sample.grid,
                    count_per_step=len(values) / count,
                    time_per_step_us=total / count,
                    share=total / count / total_kernel_us,
                    durations=Distribution.of([value for value, _, _ in values]),
                )
            )
        stats.sort(key=lambda item: -item.time_per_step_us)
        graph_values = [m.graph_fraction for m in step_metrics if m.graph_fraction is not None]
        gaps = sorted(
            (gap for metric in step_metrics for gap in metric.gaps),
            key=lambda gap: -gap.duration_us,
        )
        to_next = [next_gap[step.step_id] for step in steps if step.step_id in next_gap]
        first = steps[0]
        summaries.append(
            GroupSummary(
                rank=first.rank,
                stage=first.stage,
                label=first.label,
                batch_size=first.batch_size,
                extend_tokens=first.extend_tokens,
                annotations=dict(first.annotations),
                step_ids=[step.step_id for step in steps],
                span=Distribution.of([metric.span_us for metric in step_metrics]),
                busy_us=Distribution.of([metric.busy_us for metric in step_metrics]).median,
                idle_ratio=Distribution.of([metric.idle_ratio for metric in step_metrics]).median,
                kernels_per_step=sum(metric.kernel_count for metric in step_metrics) / count,
                kernel_launches_per_step=sum(step.kernel_launches for step in steps) / count,
                graph_launches_per_step=sum(step.graph_launches for step in steps) / count,
                sync_calls_per_step=sum(step.sync_calls for step in steps) / count,
                memcpy_calls_per_step=sum(step.memcpy_calls for step in steps) / count,
                graph_fraction=(
                    sorted(graph_values)[len(graph_values) // 2] if graph_values else None
                ),
                streams=max(metric.streams for metric in step_metrics),
                by_category_us=dict(sorted(by_category.items(), key=lambda item: -item[1])),
                comm_us=sum(metric.comm_us for metric in step_metrics) / count,
                exposed_comm_us=sum(metric.exposed_comm_us for metric in step_metrics) / count,
                host_sync_idle_us=sum(metric.host_sync_idle_us for metric in step_metrics) / count,
                gap_to_next=Distribution.of(to_next) if to_next else None,
                top_kernels=stats[:TOP_KERNELS_PER_GROUP],
                largest_gaps=gaps[:TOP_GAPS_PER_STEP],
                kernel_durations=dict(durations),
            )
        )
    summaries.sort(key=lambda item: (item.rank, -item.span.median * item.steps))
    return summaries
