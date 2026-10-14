"""Operator and step anomalies found in a parsed trace.

* slow kernel occurrences compared with the same occurrence in other steps of the same shape;
* occurrences that are persistently slower than their siblings (per-layer outliers);
* step-time jitter that kernel outliers do not explain;
* rank imbalance (one rank computes longer, the others wait inside collectives);
* eager (non-graph) decode, host synchronisation, idle gaps inside and between steps;
* kernel buckets that run far slower than the roofline estimate for the step's shape.

Thresholds are module constants and are quoted in the evidence so that a reader can judge them.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sgl_agent.findings import Finding, severity_for_share
from sgl_agent.knowledge.facts import source
from sgl_agent.model.moe_tuning import TUNING_SCRIPT, triton_moe_config_key, tuning_command
from sgl_agent.model.parallel import sglang_memory_defaults
from sgl_agent.perf.classify import COMPUTE_CATEGORIES
from sgl_agent.perf.context import AnalysisContext
from sgl_agent.perf.expected import BUCKETS, bucket_of, measured_buckets
from sgl_agent.perf.metrics import Distribution, GroupSummary, quantile
from sgl_agent.perf.trace import KernelEvent, StepSpan, short_kernel_name

MAD_TO_SIGMA = 1.4826
OUTLIER_MIN_STEPS = 3
OUTLIER_RATIO_BY_CATEGORY = {"comm": 3.0, "moe": 4.0, "moe_routing": 4.0}
OUTLIER_DEFAULT_RATIO = 2.0
OUTLIER_MAD_SIGMAS = 8.0
OUTLIER_MIN_EXTRA_US = 5.0
OUTLIER_REPORT_MIN_EXTRA_US = 20.0
OUTLIER_REPORT_MIN_SHARE = 0.005
ANOMALY_MEDIUM_SHARE = 0.01
MAX_OUTLIER_FINDINGS = 8
LAYER_PATTERN_MIN_OCCURRENCES = 8
LAYER_PATTERN_MAX_SPREAD = 0.08
LAYER_PATTERN_MAX_SLOW_FRACTION = 0.25
LAYER_PATTERN_RATIO = 2.0
LAYER_PATTERN_MAD_SIGMAS = 6.0
JITTER_MIN_STEPS = 4
JITTER_CV = 0.10
JITTER_P90_RATIO = 1.2
RANK_IMBALANCE_RATIO = 1.08
RANK_IMBALANCE_MIN_US = 50.0
EAGER_GRAPH_FRACTION = 0.5
EAGER_MIN_KERNELS = 50
PREFILL_GRAPH_MAX_TOKENS = 2048
HOST_SYNC_MIN_SHARE = 0.02
IDLE_MIN_SHARE = 0.15
INTER_STEP_GAP_MIN_US = 500.0
INTER_STEP_GAP_MIN_SHARE = 0.10
BUCKET_RATIO = 1.8
BUCKET_MIN_EXCESS_US = 300.0
BUCKET_MIN_EXCESS_SHARE = 0.05


# ---------------------------------------------------------------------------------- helpers
def _median_mad(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    median = quantile(ordered, 0.5)
    deviations = sorted(abs(value - median) for value in values)
    return median, quantile(deviations, 0.5)


def _format_positions(positions: list[int]) -> str:
    """``[3, 4, 5, 9]`` -> ``"3–5, 9"``."""
    if not positions:
        return ""
    ordered = sorted(set(positions))
    ranges: list[str] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(f"{start}–{previous}" if previous > start else str(start))
        start = previous = value
    ranges.append(f"{start}–{previous}" if previous > start else str(start))
    return ", ".join(ranges)


def _layer_hint(context: AnalysisContext, occurrences_per_step: int) -> str:
    """How call positions map to decoder layers, when the count makes it unambiguous."""
    arch = context.arch
    if arch is None:
        return "（序号从 0 计）"
    pp = context.parallel.pp if context.parallel else 1
    if occurrences_per_step == max(arch.num_layers // pp, 1):
        return "（序号从 0 计；每步调用次数等于层数，序号即层号）"
    if arch.moe and occurrences_per_step == max(arch.num_moe_layers // pp, 1):
        first_moe = arch.moe.moe_layer_indices[0] if arch.moe.moe_layer_indices else 0
        return f"（序号从 0 计；调用次数等于 MoE 层数，层号≈序号+{first_moe}）"
    return "（序号从 0 计）"


def _identifier(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", text).strip("_").lower()[:40]


def _attention_work(step: StepSpan) -> float | None:
    """Attention work proxy from the detailed annotations: Σ N_q·N_kv (causal pairs for extend)."""
    annotations = step.annotations
    work = 0.0
    if "g_sqsk" in annotations:
        work += annotations["g_sqsk"]
    if {"c_sq", "c_sqsq", "c_sqsk"} <= annotations.keys():
        work += annotations["c_sqsk"] - (annotations["c_sqsq"] - annotations["c_sq"]) / 2.0
    return work if work > 0 else None


def _category_of(context: AnalysisContext, step_id: int, index: int) -> str:
    return context.kernels_by_step[step_id][index].category


# --------------------------------------------------------------------- slow occurrences
@dataclass
class _Hit:
    group: GroupSummary
    step_id: int
    name: str
    category: str
    positions: list[int] = field(default_factory=list)
    extra_us: float = 0.0
    worst_ratio: float = 0.0
    occurrences_per_step: int = 0


def find_kernel_outliers(context: AnalysisContext) -> dict[int, list[_Hit]]:
    """Per step id, kernel occurrences far slower than the same occurrence in sibling steps."""
    hits: dict[tuple[int, str], _Hit] = {}
    for group in context.groups:
        if group.steps < OUTLIER_MIN_STEPS:
            continue
        works = {
            step_id: _attention_work(context.steps_by_id[step_id]) for step_id in group.step_ids
        }
        for signature, samples in group.kernel_durations.items():
            per_step: dict[int, list[float]] = defaultdict(list)
            for duration, step_id, _index in samples:
                per_step[step_id].append(duration)
            _, first_step, first_index = samples[0]
            category = _category_of(context, first_step, first_index)
            ratio = OUTLIER_RATIO_BY_CATEGORY.get(category, OUTLIER_DEFAULT_RATIO)
            by_count: dict[int, list[int]] = defaultdict(list)
            for step_id, values in per_step.items():
                by_count[len(values)].append(step_id)
            for count, step_ids in by_count.items():
                if len(step_ids) < OUTLIER_MIN_STEPS:
                    continue
                scale = dict.fromkeys(step_ids, 1.0)
                if category == "attention":
                    step_works = [works.get(step_id) for step_id in step_ids]
                    if all(step_works):
                        reference = sorted(step_works)[len(step_works) // 2]
                        scale = {step_id: reference / works[step_id] for step_id in step_ids}
                for position in range(count):
                    column = [per_step[step_id][position] * scale[step_id] for step_id in step_ids]
                    median, mad = _median_mad(column)
                    threshold = max(
                        ratio * median,
                        median + OUTLIER_MAD_SIGMAS * MAD_TO_SIGMA * mad,
                        median + OUTLIER_MIN_EXTRA_US,
                    )
                    for step_id, value in zip(step_ids, column, strict=True):
                        if value <= threshold:
                            continue
                        expected = median / scale[step_id]
                        raw = per_step[step_id][position]
                        key = (step_id, signature[0])
                        hit = hits.get(key)
                        if hit is None:
                            hit = _Hit(
                                group, step_id, signature[0], category, occurrences_per_step=count
                            )
                            hits[key] = hit
                        hit.positions.append(position)
                        hit.extra_us += raw - expected
                        hit.worst_ratio = max(
                            hit.worst_ratio, raw / expected if expected > 0 else 0.0
                        )
    by_step: dict[int, list[_Hit]] = defaultdict(list)
    for hit in hits.values():
        by_step[hit.step_id].append(hit)
    return by_step


def detect_kernel_outliers(context: AnalysisContext) -> tuple[list[Finding], set[int]]:
    """Findings for steps with slow kernel occurrences, and the step ids they explain."""
    by_step = find_kernel_outliers(context)
    merged: dict[tuple, dict] = {}
    explained: set[int] = set()
    for step_id, step_hits in by_step.items():
        step = context.steps_by_id[step_id]
        group = step_hits[0].group
        extra = sum(hit.extra_us for hit in step_hits)
        if extra < max(OUTLIER_REPORT_MIN_EXTRA_US, OUTLIER_REPORT_MIN_SHARE * group.step_us):
            continue
        explained.add(step_id)
        key = (group.label, step.ordinal)
        entry = merged.setdefault(
            key,
            {"group": group, "ordinal": step.ordinal, "ranks": {}, "kernels": {}, "first": False},
        )
        span = context.metrics_by_step[step_id].span_us
        entry["ranks"][step.rank] = (extra, span)
        entry["first"] = entry["first"] or step_id == group.step_ids[0]
        for hit in step_hits:
            kernel = entry["kernels"].setdefault(
                short_kernel_name(hit.name),
                {
                    "positions": set(),
                    "extra": 0.0,
                    "ratio": 0.0,
                    "count": hit.occurrences_per_step,
                    "category": hit.category,
                },
            )
            kernel["positions"].update(hit.positions)
            kernel["extra"] = max(kernel["extra"], hit.extra_us)
            kernel["ratio"] = max(kernel["ratio"], hit.worst_ratio)

    findings: list[Finding] = []
    all_ranks = set(context.trace.ranks)
    ordered = sorted(
        merged.values(), key=lambda item: -max(extra for extra, _ in item["ranks"].values())
    )
    for entry in ordered[:MAX_OUTLIER_FINDINGS]:
        group: GroupSummary = entry["group"]
        worst_extra = max(extra for extra, _ in entry["ranks"].values())
        share = worst_extra / group.step_us if group.step_us else 0.0
        severity = "high" if share >= 0.10 else "medium" if share >= ANOMALY_MEDIUM_SHARE else "low"
        ranks = sorted(entry["ranks"])
        evidence = [
            f"{group.label} 组共 {group.steps} 个 step（rank {group.rank}），中位 step {group.step_us / 1000:.2f} ms；"
            f"step #{entry['ordinal']}（该 rank 采集到的第几个 step，从 0 计）多出 {worst_extra / 1000:.3f} ms"
            f"（{share:.1%}）。",
        ]
        for name, kernel in sorted(entry["kernels"].items(), key=lambda item: -item[1]["extra"]):
            hint = _layer_hint(context, kernel["count"])
            evidence.append(
                f"{name}（{kernel['category']}）：每步 {kernel['count']} 次调用中的第 "
                f"{_format_positions(sorted(kernel['positions']))} 次{hint}，最慢为同位置中位数的 "
                f"{kernel['ratio']:.1f}×，合计多 {kernel['extra']:.0f} us。"
            )
        evidence.append(
            "判据：与同组其它 step 的同一 kernel 同一调用位置比较（同名同 grid/block）；注意力 kernel 按 "
            "step 注解中的 ΣN_q·N_kv 归一化，阈值 max(2×中位数[通信 3×、MoE 4×], 中位数+8σ_MAD, +5us)。"
        )
        recommendation: list[str] = []
        if len(all_ranks) > 1:
            if set(ranks) == all_ranks:
                evidence.append(f"所有 rank（{', '.join(ranks)}）在同一 step 同时变慢。")
                recommendation.append(
                    "所有 rank 同时变慢多为系统级原因：功耗/温度降频、其它进程占用 GPU、NVLink/PCIe 争用；"
                    "采集时并行记录 `nvidia-smi dmon -s pucvmet`，对照 Clocks Event Reasons。"
                )
            else:
                evidence.append(f"只有 {', '.join(ranks)} 出现，其余 rank 正常。")
                recommendation.append(
                    f"只在 {', '.join(ranks)} 出现：检查对应 GPU 的时钟与降频原因"
                    "（`nvidia-smi -q -d PERFORMANCE,CLOCK`）、XID/ECC 日志以及该卡上是否有其它进程。"
                )
        if entry["first"]:
            recommendation.append(
                "异常出现在该形状的第一个 step，可能是预热/JIT（Triton autotune、DeepGEMM JIT、cuBLAS 选算法）："
                "用 /start_profile 的 `start_step` 跳过前若干步再采一次确认。"
            )
        recommendation.append(
            "形状相同的其它 step 正常，说明 kernel 实现与输入形状本身不是原因；用更大的 `num_steps` 重采，"
            "确认是否周期性复现（与 HiCache 回写、KV/权重传输等后台流量对齐）。"
        )
        findings.append(
            Finding(
                id=f"anomaly.kernel_outlier.{group.stage.lower()}.{entry['ordinal']}",
                kind="anomaly",
                severity=severity,
                title=f"{group.label} 的 step #{entry['ordinal']} 出现 kernel 耗时异常",
                evidence=evidence,
                recommendation=recommendation,
                gain_ms_per_step=None,
                gain_note="偶发异常：影响尾延迟，不计入稳态收益",
                scope=group.label,
            )
        )
    return findings, explained


def _homogeneous_classes(medians: list[float]) -> list[tuple[list[int], float, list[int]]] | None:
    """Split call positions by the shortest period (1–4) whose classes are homogeneous.

    Returns ``(positions, center, slow positions)`` per class. A class is homogeneous when its
    robust spread is small and slow positions are a minority; one kernel signature serving two
    roles (e.g. the same norm kernel before attention and before the MLP) fails the second test.
    """
    count = len(medians)
    for period in (1, 2, 3, 4):
        if count // period < LAYER_PATTERN_MIN_OCCURRENCES:
            break
        classes: list[tuple[list[int], float, list[int]]] = []
        for residue in range(period):
            positions = list(range(residue, count, period))
            center, mad = _median_mad([medians[position] for position in positions])
            if center <= 0 or MAD_TO_SIGMA * mad / center > LAYER_PATTERN_MAX_SPREAD:
                break
            threshold = max(
                LAYER_PATTERN_RATIO * center,
                center + LAYER_PATTERN_MAD_SIGMAS * MAD_TO_SIGMA * mad,
                center + OUTLIER_MIN_EXTRA_US,
            )
            slow = [position for position in positions if medians[position] > threshold]
            if len(slow) > LAYER_PATTERN_MAX_SLOW_FRACTION * len(positions):
                break
            classes.append((positions, center, slow))
        if len(classes) == period:
            return classes
    return None


def detect_layer_outliers(context: AnalysisContext) -> list[Finding]:
    """Occurrences persistently slower than their siblings in every step (same signature)."""
    findings: list[Finding] = []
    for group in context.primary_groups():
        for signature, samples in group.kernel_durations.items():
            per_step: dict[int, list[float]] = defaultdict(list)
            for duration, step_id, _index in samples:
                per_step[step_id].append(duration)
            counts = Counter(len(values) for values in per_step.values())
            count, _ = counts.most_common(1)[0]
            if count < LAYER_PATTERN_MIN_OCCURRENCES:
                continue
            rows = [values for values in per_step.values() if len(values) == count]
            medians = [
                sorted(row[position] for row in rows)[len(rows) // 2] for position in range(count)
            ]
            chosen = _homogeneous_classes(medians)
            if chosen is None:
                continue
            slow_positions: list[int] = []
            extra = worst = 0.0
            for _positions, center, slow in chosen:
                for position in slow:
                    slow_positions.append(position)
                    extra += medians[position] - center
                    worst = max(worst, medians[position] / center)
            slow_positions.sort()
            if not slow_positions or extra < max(
                OUTLIER_REPORT_MIN_EXTRA_US, OUTLIER_REPORT_MIN_SHARE * group.step_us
            ):
                continue
            _, first_step, first_index = samples[0]
            category = _category_of(context, first_step, first_index)
            name = short_kernel_name(signature[0])
            share = extra / group.step_us if group.step_us else 0.0
            recommendation = [
                "同一 kernel 签名在这些调用位置上每个 step 都更慢：先确认这些位置对应的层是否有不同的配置"
                "（滑窗/全局注意力交替、不同量化、被跳过的融合），再用 `ncu --launch-skip/--launch-count` "
                "单独剖析这些调用的访存与占用。",
            ]
            if category in {"moe", "moe_routing"}:
                recommendation.append(
                    "MoE kernel 在特定层持续偏慢通常是这些层的专家负载更不均衡：用 "
                    "`--expert-distribution-recorder-mode` 统计专家分布，EP 部署时考虑 `--enable-eplb`。"
                )
            findings.append(
                Finding(
                    id=f"anomaly.layer_outlier.{group.stage.lower()}.{_identifier(name)}",
                    kind="anomaly",
                    severity=severity_for_share(share, high=0.10, medium=ANOMALY_MEDIUM_SHARE),
                    title=f"{group.label}：{name} 在部分调用位置上持续偏慢",
                    evidence=[
                        f"每步 {count} 次调用，第 {_format_positions(slow_positions)} 次"
                        f"{_layer_hint(context, count)}在所有 step 中都偏慢，最慢为同类位置中位数的 {worst:.1f}×，"
                        f"每步多 {extra:.0f} us（{share:.1%}）。",
                        f"判据：按调用周期 {len(chosen)} 分类后各类离散度 ≤ {LAYER_PATTERN_MAX_SPREAD:.0%} 且偏慢位置"
                        f"不超过 {LAYER_PATTERN_MAX_SLOW_FRACTION:.0%}，超过 max({LAYER_PATTERN_RATIO}×中位数, "
                        f"中位数+{LAYER_PATTERN_MAD_SIGMAS:.0f}σ_MAD, +{OUTLIER_MIN_EXTRA_US:.0f}us)。",
                    ],
                    recommendation=recommendation,
                    gain_ms_per_step=extra / 1000,
                    gain_note="若这些位置恢复到同类中位数",
                    scope=group.label,
                )
            )
    return findings


# --------------------------------------------------------------------------- step jitter
def detect_step_jitter(context: AnalysisContext, explained_steps: set[int]) -> list[Finding]:
    findings: list[Finding] = []
    for group in context.primary_groups():
        if group.steps < JITTER_MIN_STEPS:
            continue
        values = []
        for step_id in group.step_ids:
            metric = context.metrics_by_step[step_id]
            values.append(metric.span_us - metric.by_category_us.get("attention", 0.0))
        distribution = Distribution.of(values)
        if (
            distribution.cv <= JITTER_CV
            or distribution.p90 <= distribution.median * JITTER_P90_RATIO
        ):
            continue
        slow = [
            step_id
            for value, step_id in zip(values, group.step_ids, strict=True)
            if value > distribution.median * JITTER_P90_RATIO and step_id not in explained_steps
        ]
        if not slow:
            continue
        median_idle = Distribution.of(
            [context.metrics_by_step[step_id].idle_us for step_id in group.step_ids]
        ).median
        slow_idle = [context.metrics_by_step[step_id].idle_us for step_id in slow]
        host_side = sum(slow_idle) / len(slow_idle) > median_idle + 0.5 * (
            distribution.p90 - distribution.median
        )
        ordinals = ", ".join(str(context.steps_by_id[step_id].ordinal) for step_id in slow[:10])
        evidence = [
            f"去掉注意力后的 step 时间：中位 {distribution.median / 1000:.2f} ms，P90 {distribution.p90 / 1000:.2f} ms，"
            f"CV {distribution.cv:.2f}（阈值 {JITTER_CV}）；慢 step 序号：{ordinals}。",
            f"慢 step 平均 GPU 空闲 {sum(slow_idle) / len(slow_idle):.0f} us，组中位 {median_idle:.0f} us。",
        ]
        recommendation = (
            [
                "慢 step 的额外时间主要是 GPU 空闲：CPU 侧发射被阻塞（GC、其它线程抢占 GIL、CPU 争用）。"
                "确认调度进程绑定在 GPU 同 NUMA 的独占核上，开启 with_stack 看慢 step 的 CPU 调用栈。"
            ]
            if host_side
            else [
                "慢 step 的额外时间分散在多个 kernel 上（没有单个 kernel 超过离群阈值）：检查 GPU 降频与"
                "同卡其它负载，必要时增加 `num_steps` 重采。"
            ]
        )
        share = (distribution.p90 - distribution.median) / group.step_us if group.step_us else 0.0
        findings.append(
            Finding(
                id=f"anomaly.step_jitter.{group.stage.lower()}.{group.batch_size}",
                kind="anomaly",
                severity="medium" if share >= 0.05 else "low",
                title=f"{group.label} 的 step 时间抖动",
                evidence=evidence,
                recommendation=recommendation,
                gain_note="影响尾延迟（P90-P50）",
                scope=group.label,
            )
        )
    return findings


# ------------------------------------------------------------------------- rank imbalance
def _compute_us(metric) -> float:
    return sum(
        value for category, value in metric.by_category_us.items() if category in COMPUTE_CATEGORIES
    )


def _rank_rows(
    context: AnalysisContext,
    per_rank: dict[str, dict[int, StepSpan]],
    ordinals: list[int],
    phase: str,
) -> list[dict]:
    rows: list[dict] = []
    for rank in context.trace.ranks:
        metrics = [context.metrics_by_step[per_rank[rank][ordinal].step_id] for ordinal in ordinals]
        rows.append(
            {
                "phase": phase,
                "rank": rank,
                "steps": len(metrics),
                "compute_us_median": Distribution.of(
                    [_compute_us(metric) for metric in metrics]
                ).median,
                "comm_us_median": Distribution.of([metric.comm_us for metric in metrics]).median,
                "span_us_median": Distribution.of([metric.span_us for metric in metrics]).median,
                "idle_us_median": Distribution.of([metric.idle_us for metric in metrics]).median,
            }
        )
    return rows


def compare_ranks(context: AnalysisContext) -> tuple[list[dict], list[Finding]]:
    """Per-phase rank table and findings when one rank computes consistently longer."""
    trace = context.trace
    if len(trace.ranks) < 2:
        return [], []
    per_rank: dict[str, dict[int, StepSpan]] = {
        rank: {
            step.ordinal: step
            for step in trace.steps_for_rank(rank)
            if step.step_id in context.metrics_by_step
            and context.metrics_by_step[step.step_id].kernel_count
        }
        for rank in trace.ranks
    }
    common = sorted(set.intersection(*(set(steps) for steps in per_rank.values())))
    if not common:
        return [], []

    all_rows: list[dict] = []
    findings: list[Finding] = []
    for decode_like in (True, False):
        phase = "decode" if decode_like else "prefill/其它"
        ordinals: list[int] = []
        stage_mismatch = 0
        for ordinal in common:
            steps = [per_rank[rank][ordinal] for rank in trace.ranks]
            if len({step.stage for step in steps}) == 1:
                if steps[0].is_decode_like == decode_like:
                    ordinals.append(ordinal)
            elif decode_like and any(step.is_decode_like for step in steps):
                stage_mismatch += 1
        if not ordinals:
            continue
        rows = _rank_rows(context, per_rank, ordinals, phase)
        all_rows += rows
        ratios: list[float] = []
        excess: list[float] = []
        slowest: Counter[str] = Counter()
        batch_mismatch = stage_mismatch
        for ordinal in ordinals:
            steps = {rank: per_rank[rank][ordinal] for rank in trace.ranks}
            compute = {
                rank: _compute_us(context.metrics_by_step[step.step_id])
                for rank, step in steps.items()
            }
            fastest = min(compute.values())
            if fastest <= 0:
                continue
            heaviest = max(compute, key=compute.__getitem__)
            slowest[heaviest] += 1
            ratios.append(compute[heaviest] / fastest)
            excess.append(compute[heaviest] - sum(compute.values()) / len(compute))
            if len({step.batch_size for step in steps.values()}) > 1:
                batch_mismatch += 1
        if not ratios:
            continue
        ratio = Distribution.of(ratios).median
        gain = Distribution.of(excess).median
        if ratio < RANK_IMBALANCE_RATIO or gain < RANK_IMBALANCE_MIN_US:
            continue
        rank, times = slowest.most_common(1)[0]
        row_by_rank = {row["rank"]: row for row in rows}
        evidence = [
            f"{phase} 对齐的 {len(ratios)} 个 step 中，计算 kernel 时间最慢 rank 与最快 rank 之比中位数 {ratio:.2f}"
            f"（阈值 {RANK_IMBALANCE_RATIO}）；{rank} 有 {times} 次最慢。",
            f"{phase} 各 rank 计算/通信 kernel 时间中位数（us）："
            + "；".join(
                f"{row['rank']} {row['compute_us_median']:.0f}/{row['comm_us_median']:.0f}"
                for row in rows
            )
            + "。快 rank 的通信时间更长，是在集合通信里等慢 rank。",
        ]
        recommendation: list[str] = []
        parallel = context.parallel
        if batch_mismatch:
            dp_attention = parallel is None or parallel.attn_dp > 1
            evidence.append(
                f"{batch_mismatch} 个对齐 step 中各 rank 的阶段或 batch 大小不同"
                + ("（DP attention 下各 DP rank 各自组 batch）。" if dp_attention else "。")
            )
            if dp_attention:
                recommendation.append(
                    "DP rank 之间负载不均：`--load-balance-method` 默认 auto（非 PD 与 PD decode 下为 round_robin），"
                    "长短请求混合时改为 `total_tokens` 或 `total_requests`；DP attention 下 decode CUDA graph "
                    "以各 DP rank 中最大的 batch 判断能否回放。"
                )
            else:
                recommendation.append(
                    "没有 DP attention 时各 TP rank 执行同一个 batch：形状不同说明这些 trace 不是同一次采集，"
                    "或某个 rank 丢了 step；重新用 /start_profile 同时采集所有 rank。"
                )
        if (
            context.arch is not None
            and context.arch.is_moe
            and parallel is not None
            and parallel.ep > 1
        ):
            recommendation.append(
                "EP 下专家负载不均会让持有热门专家的 rank 变慢：`--expert-distribution-recorder-mode` 统计分布后"
                "开启 `--enable-eplb`（配合 `--ep-num-redundant-experts` 冗余热门专家）。"
            )
        slow_idle = row_by_rank[rank]["idle_us_median"]
        others_idle = [row["idle_us_median"] for row in rows if row["rank"] != rank]
        if others_idle and slow_idle > 1.5 * max(others_idle) + 20:
            recommendation.append(
                f"{rank} 的 GPU 空闲明显更多：CPU 侧发射更慢，检查该 rank 进程的 CPU/NUMA 绑定与同机其它负载。"
            )
        if not recommendation or not batch_mismatch:
            recommendation.append(
                f"各 rank 形状一致仍有差异时，检查 {rank} 所在 GPU 的时钟/功耗/温度"
                "（`nvidia-smi -q -d CLOCK,PERFORMANCE,POWER,TEMPERATURE`）、NVLink 状态（`nvidia-smi nvlink -s`）"
                "与拓扑（`nvidia-smi topo -m`）。"
            )
        span = row_by_rank[rank]["span_us_median"] or 1.0
        findings.append(
            Finding(
                id=f"anomaly.rank_imbalance.{'decode' if decode_like else 'prefill'}",
                kind="anomaly",
                severity=severity_for_share(gain / span, high=0.10, medium=0.03),
                title=f"{phase} 阶段 rank 间计算时间不均（{rank} 最慢）",
                evidence=evidence,
                recommendation=recommendation,
                gain_ms_per_step=gain / 1000,
                gain_note="最慢 rank 降到各 rank 平均计算时间时每步可省的时间",
                scope=phase,
            )
        )
    return all_rows, findings


# ---------------------------------------------------------------- launch / host bound
def is_graph_candidate_eager(group: GroupSummary) -> bool:
    """An eager group SGLang could run as a CUDA graph: decode, or a small launch-bound prefill."""
    if group.graph_fraction is None or group.graph_fraction >= EAGER_GRAPH_FRACTION:
        return False
    if group.kernels_per_step < EAGER_MIN_KERNELS:
        return False
    small_prefill = (
        group.stage == "EXTEND"
        and group.extend_tokens is not None
        and group.extend_tokens <= PREFILL_GRAPH_MAX_TOKENS
        and group.idle_ratio >= 0.2
    )
    return group.is_decode_like or small_prefill


def detect_eager_steps(context: AnalysisContext) -> tuple[list[Finding], set[int]]:
    findings: list[Finding] = []
    handled: set[int] = set()
    hardware = context.hardware
    gap_us = hardware.latency.graph_kernel_gap_us if hardware else 1.0
    graph_groups = [
        group
        for group in context.primary_groups()
        if group.is_decode_like and group.graph_fraction is not None and group.graph_fraction >= 0.9
    ]
    for group in context.primary_groups():
        if not is_graph_candidate_eager(group):
            continue
        graph_estimate = group.busy_us + group.kernels_per_step * gap_us
        gain = max(group.step_us - graph_estimate, 0.0)
        share = gain / group.step_us if group.step_us else 0.0
        if share < 0.02:
            continue
        handled.add(id(group))
        evidence = [
            f"{group.label}：每步 {group.kernels_per_step:.0f} 个 kernel、{group.kernel_launches_per_step:.0f} 次 "
            f"cudaLaunchKernel，graph 发射占比 {group.graph_fraction:.0%}；GPU 忙 {group.busy_us / 1000:.2f} ms / "
            f"step {group.step_us / 1000:.2f} ms（空闲 {group.idle_ratio:.0%}）。",
        ]
        comparable = min(
            graph_groups, key=lambda item: abs(item.batch_size - group.batch_size), default=None
        )
        if comparable is not None and group.is_decode_like:
            evidence.append(
                f"同一 trace 中 {comparable.label} 走 CUDA graph，空闲仅 {comparable.idle_ratio:.0%}。"
            )
        recommendation: list[str] = []
        flags: dict[str, int] = {}
        if hardware is not None and not hardware.software.graph_capture:
            recommendation.append(
                f"{hardware.name} 的软件栈未声明支持 graph 捕获：这是平台适配缺口（见适配分析），"
                "在此之前只能靠减少 kernel 数（融合、torch.compile）降低发射开销。"
            )
        if group.is_decode_like:
            tp = context.parallel.tp if context.parallel else 1
            default_max_bs = sglang_memory_defaults(hardware, tp)[1] if hardware else None
            reasons = (
                "SGLang decode 默认 `--cuda-graph-backend-decode full`。回退 eager 的条件（decode_cuda_graph_runner."
                "can_run_graph）：batch 超过捕获上限（DP attention 下取各 DP rank 中最大的 batch）、请求带 embedding "
                "覆盖（replace_embeds）、开启 TBO 但该 batch 不能切分、投机解码每请求 token 数与捕获宽度不一致。"
            )
            recommendation.append(reasons)
            if default_max_bs is not None and group.batch_size > default_max_bs:
                flags["--cuda-graph-max-bs-decode"] = group.batch_size
                recommendation.append(
                    f"bs={group.batch_size} 超过 {hardware.name} 上 TP{tp} 的默认捕获上限 {default_max_bs}："
                    f"设置 `--cuda-graph-max-bs-decode {group.batch_size}` 或用 `--cuda-graph-bs-decode` 显式列出需要的档位"
                    "（每多一档都会多占显存，需同步复核 KV 容量）。"
                )
            else:
                recommendation.append(
                    "batch 未超过默认捕获上限：检查启动参数是否把 `--cuda-graph-backend-decode` 设成了 disabled/breakable、"
                    "是否显式设置了更小的 `--cuda-graph-max-bs-decode`/`--cuda-graph-bs-decode`，"
                    "并在服务器日志的 Decode batch 行确认 `cuda graph: False` 出现的条件。"
                )
        else:
            recommendation.append(
                f"小 prefill（{group.extend_tokens} token）在 CUDA 上默认可走 breakable prefill graph"
                "（`--cuda-graph-backend-prefill breakable`，按 token 数分档捕获）：确认没有被设成 disabled，"
                "且 `--cuda-graph-max-bs-prefill` 覆盖该 token 数。"
            )
        findings.append(
            Finding(
                id=f"anomaly.eager.{group.stage.lower()}.{group.batch_size}",
                kind="launch_bound",
                severity=severity_for_share(share),
                title=f"{group.label} 没有走 CUDA graph，kernel 发射开销明显",
                evidence=evidence,
                recommendation=recommendation,
                flags=flags,
                gain_ms_per_step=gain / 1000,
                gain_note=f"按 graph 模式每个 kernel {gap_us:g} us 间隔估算",
                scope=group.label,
                references=[source("decode_graph_fallback"), source("cuda_graph_defaults")],
            )
        )
    return findings, handled


def _sync_producers(context: AnalysisContext, step_id: int) -> Counter[str]:
    """Kernels whose results are copied to the host right before an idle gap."""
    producers: Counter[str] = Counter()
    kernels = context.kernels_by_step.get(step_id, [])
    previous: KernelEvent | None = None
    for position, kernel in enumerate(kernels):
        if kernel.kind == "memcpy" and (
            "DtoH" in kernel.name
            or "Device -> Pinned" in kernel.name
            or "Device -> Pageable" in kernel.name
        ):
            following = kernels[position + 1] if position + 1 < len(kernels) else None
            if (
                following is not None
                and following.start_us - kernel.end_us >= 10.0
                and previous is not None
            ):
                producers[short_kernel_name(previous.name)] += 1
        if kernel.kind == "kernel":
            previous = kernel
    return producers


def detect_host_sync(context: AnalysisContext) -> tuple[list[Finding], set[int]]:
    findings: list[Finding] = []
    handled: set[int] = set()
    for group in context.primary_groups():
        if group.step_us <= 0:
            continue
        share = group.host_sync_idle_us / group.step_us
        if share < HOST_SYNC_MIN_SHARE and group.sync_calls_per_step < 1:
            continue
        if share < HOST_SYNC_MIN_SHARE / 2:
            continue
        handled.add(id(group))
        producers = _sync_producers(context, context.representative_step_id(group))
        evidence = [
            f"{group.label}：每步 {group.sync_calls_per_step:.0f} 次同步类 runtime 调用"
            f"（cudaStreamSynchronize/cudaMemcpy 等）、{group.memcpy_calls_per_step:.0f} 次 memcpy；"
            f"DtoH 拷贝后的 GPU 空闲 {group.host_sync_idle_us / 1000:.2f} ms/step（{share:.1%}）。",
        ]
        if producers:
            evidence.append(
                "拷回主机之前的 kernel（数据来源）："
                + "，".join(f"{name}×{count}" for name, count in producers.most_common(4))
            )
        findings.append(
            Finding(
                id=f"anomaly.host_sync.{group.stage.lower()}.{group.batch_size}",
                kind="host_bound",
                severity=severity_for_share(share),
                title=f"{group.label} 的 forward 里有主机同步，GPU 每次都要排空等待 CPU",
                evidence=evidence,
                recommendation=[
                    "用 with_stack 采集（SGLANG_PROFILE_WITH_STACK 默认开启），在 CPU 时间线上找到这些 "
                    "cudaStreamSynchronize/cudaMemcpy 的 Python 调用栈；常见来源是 `.item()`/`.tolist()`/`.cpu()`、"
                    "布尔索引与 `torch.nonzero` 这类输出形状依赖数据的算子、以及按张量值走的 Python 分支。",
                    "把依赖数据的形状计算挪到调度侧的 CPU 元数据（ForwardBatch 里已有 *_cpu 镜像），或改成定长的设备端实现；"
                    "这类同步也会阻止该路径被 CUDA graph 捕获。",
                ],
                gain_ms_per_step=group.host_sync_idle_us / 1000,
                gain_note="只计 GPU 时间线上可见的等待，不含 CPU 侧同步耗时",
                scope=group.label,
            )
        )
    return findings, handled


def detect_idle_gaps(context: AnalysisContext, handled_groups: set[int]) -> list[Finding]:
    findings: list[Finding] = []
    for group in context.primary_groups():
        if id(group) in handled_groups or group.idle_ratio < IDLE_MIN_SHARE:
            continue
        gaps = "；".join(
            f"{gap.before} → {gap.after} 空 {gap.duration_us:.0f} us"
            for gap in group.largest_gaps[:3]
        )
        idle_us = group.step_us - group.busy_us
        findings.append(
            Finding(
                id=f"anomaly.idle.{group.stage.lower()}.{group.batch_size}",
                kind="host_bound",
                severity=severity_for_share(idle_us / group.step_us if group.step_us else 0.0),
                title=f"{group.label} 的 step 内 GPU 空闲 {group.idle_ratio:.0%}",
                evidence=[
                    f"GPU 忙 {group.busy_us / 1000:.2f} ms / step {group.step_us / 1000:.2f} ms，"
                    f"graph 发射占比 {group.graph_fraction if group.graph_fraction is not None else '未知'}。",
                    f"最大空洞：{gaps}"
                    if gaps
                    else "没有单个超过 10 us 的空洞，空闲由大量小间隔组成。",
                ],
                recommendation=[
                    "空洞两侧的 kernel 指出了等待点：跨流 event 等待、CPU 发射被阻塞（GIL/其它线程）或通信前的同步。"
                    "graph 模式下 kernel 应当首尾相接，出现空洞时检查是否有 graph 之外的 eager 段（breakable graph 的断点）。",
                ],
                gain_ms_per_step=idle_us / 1000,
                gain_note="上限：消除全部空闲",
                scope=group.label,
            )
        )
    return findings


def detect_inter_step_gaps(context: AnalysisContext) -> list[Finding]:
    findings: list[Finding] = []
    for group in context.primary_groups():
        if not group.is_decode_like or group.gap_to_next is None or group.gap_to_next.count < 2:
            continue
        gap = group.gap_to_next.median
        if gap < max(INTER_STEP_GAP_MIN_US, INTER_STEP_GAP_MIN_SHARE * group.step_us):
            continue
        findings.append(
            Finding(
                id=f"anomaly.inter_step_gap.{group.batch_size}",
                kind="host_bound",
                severity=severity_for_share(gap / (group.step_us + gap)),
                title=f"{group.label}：相邻 step 之间 GPU 空闲 {gap / 1000:.2f} ms",
                evidence=[
                    f"step 结束到下一个 step 第一个 kernel 的间隔中位数 {gap:.0f} us，P90 {group.gap_to_next.p90:.0f} us；"
                    f"step 本身 {group.step_us / 1000:.2f} ms。间隔中包含 step span 之外的采样 kernel。",
                ],
                recommendation=[
                    "overlap 调度会让 CPU 准备下一个 batch 时 GPU 继续执行：确认没有设置 `--disable-overlap-schedule`，"
                    "且启动日志里没有因其它特性自动关闭 overlap 的提示。",
                    "调度 CPU 开销大时可试 `--num-continuous-decode-steps 2~4`（每轮连续跑多步 decode，可能增加首 token 延迟），"
                    "并保证调度进程不与 tokenizer/detokenizer/HTTP 进程争用同一批 CPU 核。",
                ],
                gain_ms_per_step=gap / 1000,
                gain_note="上限：间隔完全被 overlap 掉",
                scope=group.label,
            )
        )
    return findings


# --------------------------------------------------------------- roofline deviation
def _bucket_recommendations(
    context: AnalysisContext, group: GroupSummary, bucket: str
) -> tuple[list[str], dict]:
    hardware, knowledge = context.hardware, context.knowledge
    flags: dict = {}
    if bucket == "attention":
        backends = list(hardware.software.attention_backends) if hardware else []
        default = hardware.software.default_attention_backend if hardware else None
        alternatives = [name for name in backends if name != default][:4]
        phase_flag = (
            "--decode-attention-backend" if group.is_decode_like else "--prefill-attention-backend"
        )
        lines = [
            f"对注意力后端做 A/B：`{phase_flag}` 可单独替换 {'decode' if group.is_decode_like else 'prefill'} 阶段后端"
            + (
                f"，{hardware.name} 上可选 {', '.join(alternatives)}（默认 {default}）"
                if alternatives
                else ""
            )
            + "；注意各后端的 page size 约束。",
            "确认 KV cache dtype 与后端匹配（FP8 KV 需要后端支持对应 kernel），以及 MLA 模型的 DCP/DPA 切分没有让每卡头数过小。",
        ]
        return lines, flags
    if bucket == "gemm":
        fp8 = context.cost_model is not None and context.cost_model.linear_compute_dtype in {
            "fp8",
            "int8",
        }
        flag = "--fp8-gemm-backend" if fp8 else "--bf16-gemm-backend"
        choices = [choice for choice in (knowledge.choices(flag) or []) if choice != "auto"]
        lines = [
            f"对 GEMM 后端做 A/B：`{flag}`（可选 {', '.join(choices[:6])}）。",
            "确认测量窗口已过 JIT/autotune 预热（DeepGEMM 首次遇到新形状会编译），并检查 GEMM 的 M（每卡 token 数）"
            "是否落在库调优过的区间。",
        ]
        return lines, flags
    if bucket == "moe":
        lines: list[str] = []
        triton = any("fused_moe_kernel" in kernel.name for kernel in group.top_kernels)
        if (
            triton
            and context.arch is not None
            and context.parallel is not None
            and hardware is not None
        ):
            key = triton_moe_config_key(context.arch, context.parallel, hardware)
            if key is not None:
                tuned = knowledge.moe_tuned_configs(
                    device=key.device,
                    experts=key.experts,
                    intermediate=key.intermediate,
                    dtype=key.dtype,
                    block_shape=list(key.block_shape) if key.block_shape else None,
                )
                versions = sorted({entry["triton"] for entry in tuned})
                if tuned:
                    lines.append(
                        f"SGLang 自带 `{key.file_name()}`（Triton {', '.join(versions)}）；若运行环境的 Triton 版本不在其中，"
                        "SGLang 会回退到其它版本的配置并告警，按当前环境重新调优更稳。"
                    )
                else:
                    lines.append(
                        f"SGLang 没有自带 `{key.file_name()}`，会使用默认 MoE kernel 配置（日志：Using default MoE kernel "
                        f"config）。在目标卡上运行 `{tuning_command(context.arch, context.parallel, key)}`，把生成的 JSON "
                        "放到 `SGLANG_MOE_CONFIG_DIR/configs/triton_<版本>/`。"
                    )
        runners = [
            name
            for name in (hardware.software.moe_runner_backends if hardware else [])
            if name != "auto"
        ]
        lines.append(
            f"对 MoE 后端做 A/B：`--moe-runner-backend`（{hardware.name if hardware else '该硬件'}上可选 {', '.join(runners)}）。"
            if runners
            else f"对 MoE 后端做 A/B：`--moe-runner-backend`；Triton MoE 调优脚本见 `{TUNING_SCRIPT}`。"
        )
        return lines, flags
    if bucket == "comm":
        return [
            "通信 kernel 的时长包含等待其它 rank 的时间：先看 rank 不均衡的结论，再判断链路本身。",
            "检查 GPU 互联（`nvidia-smi topo -m`）与 NCCL 选择的算法/协议（NCCL_DEBUG=INFO）；decode 小消息确认"
            "custom allreduce 没有被 `--disable-custom-all-reduce` 关掉，可 A/B `--enable-torch-symm-mem`/`--enable-symm-mem`。",
        ], flags
    return [
        "小算子总耗时远超带宽估算，说明 kernel 数量多、单个 kernel 太小：见融合建议（未融合的 add+norm、"
        "act+mul、elementwise 链）。",
    ], flags


def _comm_floor_us(context: AnalysisContext, group: GroupSummary) -> float | None:
    """Median over the group's steps of the smallest per-rank communication kernel time.

    Collectives finish together, so the rank that arrives last spends the least time inside them;
    its time approximates the transfer itself and the rest on other ranks is waiting.
    """
    if len(context.trace.ranks) < 2:
        return None
    floors: list[float] = []
    for step_id in group.step_ids:
        aligned = [
            step
            for step in context.aligned_steps(context.steps_by_id[step_id])
            if step.step_id in context.metrics_by_step
        ]
        if len(aligned) < 2:
            continue
        floors.append(
            min(
                context.metrics_by_step[step.step_id].by_category_us.get("comm", 0.0)
                for step in aligned
            )
        )
    return Distribution.of(floors).median if floors else None


def detect_bucket_deviation(context: AnalysisContext) -> list[Finding]:
    findings: list[Finding] = []
    for group in context.primary_groups():
        expectation = context.expectation(group)
        if expectation is None:
            continue
        measured = measured_buckets(group.by_category_us)
        for bucket in BUCKETS:
            expected = expectation.buckets_us.get(bucket, 0.0)
            actual = measured.get(bucket, 0.0)
            if expected <= 0:
                continue
            waiting_note = None
            if bucket == "comm":
                floor = _comm_floor_us(context, group)
                if floor is not None and floor < actual:
                    waiting_note = (
                        f"本 rank 通信 kernel {actual / 1000:.2f} ms/step，各 rank 同一 step 的最小值 "
                        f"{floor / 1000:.2f} ms；差值是等待其它 rank，已从比较中扣除（见 rank 不均衡结论）。"
                    )
                    actual = floor
            ratio = actual / expected
            excess = actual - expected
            if ratio < BUCKET_RATIO or excess < max(
                BUCKET_MIN_EXCESS_US, BUCKET_MIN_EXCESS_SHARE * group.step_us
            ):
                continue
            top = [
                f"{kernel.short_name} {kernel.time_per_step_us:.0f} us/step（{kernel.count_per_step:.0f} 次）"
                for kernel in group.top_kernels
                if bucket_of(kernel.category) == bucket
            ][:3]
            recommendation, flags = _bucket_recommendations(context, group, bucket)
            efficiency = context.hardware.efficiency if context.hardware else None
            assumption = (
                f"（模型假设 GEMM 效率 {efficiency.gemm_compute:.0%}、注意力 {efficiency.attention_compute:.0%}、"
                f"带宽 {efficiency.memory_bandwidth:.0%}、通信 {efficiency.intra_node_comm:.0%}）"
                if efficiency
                else ""
            )
            findings.append(
                Finding(
                    id=f"anomaly.roofline.{group.stage.lower()}.{group.batch_size}.{bucket}",
                    kind="kernel_efficiency",
                    severity=severity_for_share(excess / group.step_us if group.step_us else 0.0),
                    title=f"{group.label}：{bucket} 实测是屋脊估算的 {ratio:.1f}×",
                    evidence=[
                        f"实测 {actual / 1000:.2f} ms/step，按 step 注解重建的形状估算 {expected / 1000:.2f} ms{assumption}，"
                        f"形状来源：{expectation.shape_source}。",
                        *([waiting_note] if waiting_note else []),
                        ("主要 kernel：" + "；".join(top))
                        if top
                        else "该类没有进入 Top kernel 的条目。",
                    ],
                    recommendation=recommendation,
                    flags=flags,
                    gain_ms_per_step=excess / 1000,
                    gain_note="上限：降到屋脊估算（效率假设需用 sgl-agent calibrate 校准）",
                    scope=group.label,
                )
            )
    return findings
