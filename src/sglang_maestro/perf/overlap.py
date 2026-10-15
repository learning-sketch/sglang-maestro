"""Communication exposure, multi-stream overlap and L2 weight-prefetch headroom in a trace.

Exposed communication is collective time on the primary rank that no compute kernel on another
stream overlaps (``StepMetrics.overlapped_comm_us``); the cost model tells whether the step's
collectives are latency- or bandwidth-bound at its shape, which decides what can help.
Multi-stream use is read from the streams that run compute kernels in a step and compared with
the dual-stream paths SGLang has for the model. The L2 prefetch estimate treats every collective
of a decode step as a window in which HBM is mostly idle and asks how much of the next GEMM's
weights could be staged in L2 meanwhile, bounded by that GEMM's own compute time.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from sglang_maestro.findings import Finding, severity_for_share
from sglang_maestro.hardware.spec import bytes_per_element
from sglang_maestro.knowledge.facts import (
    CUSTOM_ALLREDUCE_MAX_BYTES_CUDA,
    CUSTOM_ALLREDUCE_WORLD_SIZES,
    DEEPSEEK_SHARED_EXPERTS_FUSION_ARCHITECTURES,
    DEEPSEEK_SHARED_EXPERTS_FUSION_QUARK_ROUTED_EXPERTS,
    DEEPSEEK_SHARED_EXPERTS_FUSION_ROUTED_EXPERTS,
    DEEPSEEK_V2_MODEL_ARCHITECTURES,
    GLM4_MOE_SHARED_EXPERTS_FUSION_ARCHITECTURE,
    NPU_SHARED_EXPERT_STREAM_ARCHITECTURES,
    SBO_UNSUPPORTED_SM_MAJOR,
    TBO_PREFILL_ONLY_ARCHITECTURES,
    TORCH_SYMM_MEM_SM90_WORLD_SIZES,
    TORCH_SYMM_MEM_SM100_WORLD_SIZES,
    source,
)
from sglang_maestro.model.config import ModelArch
from sglang_maestro.model.ops import OpTiming
from sglang_maestro.model.params import weight_precision
from sglang_maestro.perf.anomalies import EAGER_GRAPH_FRACTION, comm_floor_us
from sglang_maestro.perf.classify import COMPUTE_CATEGORIES
from sglang_maestro.perf.context import AnalysisContext
from sglang_maestro.perf.fusion import LATENCY_REGIME_FACTOR, compact_kernel_name, step_tokens
from sglang_maestro.perf.metrics import GroupSummary
from sglang_maestro.perf.trace import (
    KernelEvent,
    StepSpan,
    intersection_length,
    merge_intervals,
    total_length,
)

EXPOSED_COMM_MIN_SHARE = 0.05
DUAL_STREAM_MIN_SHARE = 0.01
PREFETCH_MIN_SHARE = 0.01
PREFETCH_L2_BUDGET_FRACTION = 0.5
"""Share of L2 staged weights may occupy; activations and KV lines need the rest (assumption)."""
PREFETCH_BANDWIDTH_SPARE = 0.8
"""Share of HBM bandwidth a prefetch can use while a collective runs (assumption)."""
PREFETCH_LOOKAHEAD_KERNELS = 4
"""How far after a collective the consuming GEMM may sit (norms, adds, quantization between)."""
PREFETCH_BARRIER_CATEGORIES = frozenset({"comm", "attention", "moe", "moe_routing"})
"""Kernels that end the search for a consumer: routed-expert weights are unknown before routing."""
MAX_EVIDENCE_GROUPS = 3
SHARED_EXPERT_OPS = frozenset({"moe.shared_gate_up", "moe.shared_act", "moe.shared_down"})
ROUTED_EXPERT_OPS = frozenset({"moe.experts_grouped_gemm"})
QUANT_EXCLUDE_KEYS = (
    "exclude",
    "exclude_layers",
    "exclude_modules",
    "ignore",
    "ignored_layers",
    "modules_to_not_convert",
)


def _platform(context: AnalysisContext) -> str | None:
    return context.hardware.software.platform if context.hardware else None


def _format_bytes(value: float) -> str:
    if value >= 1024**2:
        return f"{value / 1024**2:.1f} MiB"
    if value >= 1024:
        return f"{value / 1024:.0f} KiB"
    return f"{value:.0f} B"


# ------------------------------------------------------------------------- collectives
def collective_implementation(kernel: KernelEvent, library: str | None = None) -> str:
    """Which library ran a communication kernel, from its name (``library``: the device's CCL)."""
    lowered = kernel.name.lower()
    if kernel.fused:
        return "allreduce+norm 融合"
    if library and library.lower() not in {"nccl", "rccl"}:
        token = re.sub(r"[^a-z0-9]+", "_", library.lower()).strip("_")
        if token and token in re.sub(r"[^a-z0-9]+", "_", lowered):
            return library
    if "deep_ep" in lowered or "deepep" in lowered or "internode_ll" in lowered:
        return "DeepEP"
    if "mscclpp" in lowered:
        return "MSCCL++"
    if "cross_device_reduce" in lowered or "custom_all_reduce" in lowered:
        return "custom allreduce"
    if "multimem" in lowered or "symm" in lowered or "one_shot" in lowered or "two_shot" in lowered:
        return "symmetric memory"
    if "nccl" in lowered:
        return "NCCL"
    return "其它"


@dataclass(frozen=True)
class CollectiveStats:
    pattern: str
    implementation: str
    count: int
    total_us: float

    @property
    def mean_us(self) -> float:
        return self.total_us / self.count if self.count else 0.0


def step_collectives(
    kernels: list[KernelEvent], library: str | None = None
) -> list[CollectiveStats]:
    """Communication kernels of one step grouped by pattern and implementation, largest first."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for kernel in kernels:
        if kernel.category != "comm":
            continue
        key = (kernel.comm_pattern or "other", collective_implementation(kernel, library))
        counts[key] += 1
        totals[key] += kernel.duration_us
    stats = [
        CollectiveStats(pattern, implementation, counts[(pattern, implementation)], total)
        for (pattern, implementation), total in totals.items()
    ]
    stats.sort(key=lambda item: -item.total_us)
    return stats


@dataclass(frozen=True)
class ExpectedCollective:
    name: str
    pattern: str
    message_bytes: float
    per_call_us: float
    transfer_us: float
    calls_per_step: float

    @property
    def latency_bound(self) -> bool:
        """The fixed per-call cost outweighs the transfer itself."""
        return self.transfer_us < self.per_call_us - self.transfer_us


def expected_collectives(context: AnalysisContext, group: GroupSummary) -> list[ExpectedCollective]:
    """Collectives the cost model expects at the group's shape, largest total first."""
    expectation, hardware = context.expectation(group), context.hardware
    if expectation is None or hardware is None:
        return []
    latency = hardware.latency
    collectives: list[ExpectedCollective] = []
    for timing in expectation.timings:
        op = timing.op
        if op.kind != "comm" or timing.kernel_us <= 0:
            continue
        inter_node = (
            op.comm_group > hardware.devices_per_node and hardware.inter_node_bandwidth_gbps > 0
        )
        base = latency.inter_node_collective_base_us if inter_node else latency.collective_base_us
        collectives.append(
            ExpectedCollective(
                name=op.name,
                pattern=op.comm_pattern or "other",
                message_bytes=op.comm_bytes,
                per_call_us=timing.kernel_us,
                transfer_us=max(timing.kernel_us - base, 0.0),
                calls_per_step=op.layers,
            )
        )
    collectives.sort(key=lambda item: -item.per_call_us * item.calls_per_step)
    return collectives


def _describe_collectives(stats: list[CollectiveStats], limit: int = 3) -> str:
    return "；".join(
        f"{item.pattern} ×{item.count}（{item.implementation}，平均 {item.mean_us:.1f} us）"
        for item in stats[:limit]
    )


def _tbo_advice(
    context: AnalysisContext, group: GroupSummary, latency_bound: bool
) -> tuple[str, list[str]]:
    architecture, parallel = context.architecture, context.parallel
    if architecture in TBO_PREFILL_ONLY_ARCHITECTURES and group.is_decode_like:
        return (
            f"{architecture} 的 TBO 只实现了 prefill（decode 会抛 NotImplementedError），decode 的通信不能靠 TBO 掩盖。",
            [source("tbo_deepseek_v4")],
        )
    if parallel is not None and parallel.uses_a2a:
        caveat = (
            "decode 小 batch 时两个 micro-batch 各付一次通信延迟、次数翻倍，收益有限甚至变慢，以 A/B 为准。"
            if group.is_decode_like and latency_bound
            else ""
        )
        return (
            f"{architecture} 支持 two-batch overlap：A/B `--enable-two-batch-overlap`，把 batch 切成两个 micro-batch，"
            "一个的 EP dispatch/combine 与另一个的计算重叠（prefill 由 `--tbo-token-distribution-threshold` 决定按请求"
            "切两个 batch 还是把序列切成两个 chunk）。" + caveat,
            [source("tbo_strategies")],
        )
    if architecture in TBO_PREFILL_ONLY_ARCHITECTURES:
        if parallel is not None and parallel.attn_dp > 1:
            return (
                f"{architecture} 有不走 a2a 的 DP TBO 路径（只用于 prefill）：A/B `--enable-two-batch-overlap`，"
                "DP 的 all_gatherv / reduce_scatterv 与另一个 micro-batch 的计算重叠。",
                [source("tbo_deepseek_v4")],
            )
        return (
            f"{architecture} 不走 a2a 时，TBO 要求 DP attention（`--attn-dp-size`），当前是纯 TP。",
            [source("tbo_requires_dpa"), source("tbo_deepseek_v4")],
        )
    return (
        f"{architecture} 的 TBO 重叠的是 EP 的 dispatch/combine 与另一个 micro-batch 的计算，需要 "
        "`--moe-a2a-backend deepep` 等 EP a2a 后端（不走 a2a 时这些操作为空）；当前没有 a2a，TP 的 allreduce "
        "不在 TBO 的重叠范围内。是否改用 EP 见特性建议。",
        [source("tbo_strategies"), source("tbo_ep_dispatch")],
    )


def _exposed_comm_recommendation(
    context: AnalysisContext,
    group: GroupSummary,
    stats: list[CollectiveStats],
    expected: list[ExpectedCollective],
) -> tuple[list[str], list[str]]:
    knowledge, hardware, parallel = context.knowledge, context.hardware, context.parallel
    architecture = context.architecture
    platform = _platform(context)
    sm_major = context.sm_major
    tp = parallel.tp if parallel else None
    main = expected[0] if expected else None
    latency_bound = main.latency_bound if main else group.is_decode_like
    lines: list[str] = []
    references = [source("allreduce_dispatch")]
    if main is not None:
        if latency_bound:
            lines.append(
                f"按成本模型，本组最主要的集合通信 `{main.name}` 每次消息 {_format_bytes(main.message_bytes)}，"
                f"传输只需约 {main.transfer_us:.1f} us，其余 {main.per_call_us - main.transfer_us:.0f} us 是固定开销："
                "通信以延迟为主。换更高带宽的算法帮助不大，有效的是减少次数、降低单次延迟或把它藏到计算后面。"
            )
        else:
            lines.append(
                f"按成本模型，`{main.name}` 每次消息 {_format_bytes(main.message_bytes)}，传输约 "
                f"{main.transfer_us:.0f} us（单次的 {main.transfer_us / main.per_call_us:.0%}）：通信以带宽为主，"
                "集合通信算法与链路利用率决定耗时。"
            )
    nccl_allreduce = next(
        (item for item in stats if item.pattern == "allreduce" and item.implementation == "NCCL"),
        None,
    )
    allreduce_bytes = next(
        (item.message_bytes for item in expected if item.pattern == "allreduce"), None
    )
    if (
        platform == "cuda"
        and group.is_decode_like
        and nccl_allreduce is not None
        and tp in CUSTOM_ALLREDUCE_WORLD_SIZES
        and (allreduce_bytes is None or allreduce_bytes <= CUSTOM_ALLREDUCE_MAX_BYTES_CUDA)
    ):
        lines.append(
            f"decode 的 allreduce 走的是 NCCL kernel（每步 {nccl_allreduce.count} 次）：CUDA 上消息 ≤ 8 MiB、"
            f"world size ∈ {{{', '.join(map(str, CUSTOM_ALLREDUCE_WORLD_SIZES))}}} 时 SGLang 优先用 custom allreduce，"
            "前提是没有设置 `--disable-custom-all-reduce`，且 2 卡以上时 GPU 间 NVLink 全互联（`nvidia-smi topo -m`）。"
            "custom allreduce 没有生效时先排查这两点。"
        )
        references.append(source("custom_allreduce"))
    if architecture in knowledge.tbo_architectures:
        tbo_line, tbo_references = _tbo_advice(context, group, latency_bound)
        lines.append(tbo_line)
        references += tbo_references
    elif architecture:
        tbo_classes = ", ".join(knowledge.raw.get("tbo_layer_classes", []))
        lines.append(
            f"SGLang 没有为 {architecture} 实现 two-batch overlap（只覆盖 {tbo_classes or '少数 MoE decoder layer'}）；"
            "TP 的 allreduce 依赖前一个 GEMM 的结果，同一 batch 内无法与计算重叠。"
        )
    if (
        platform == "cuda"
        and sm_major == SBO_UNSUPPORTED_SM_MAJOR
        and parallel is not None
        and parallel.uses_a2a
    ):
        lines.append(
            "SBO（`--enable-single-batch-overlap`）在 SM90 上不可用：SGLang 初始化时直接报错。"
        )
        references.append(source("sbo_sm90"))
    if platform == "cuda":
        if latency_bound:
            symm_sizes = (
                TORCH_SYMM_MEM_SM90_WORLD_SIZES
                if sm_major == 9
                else TORCH_SYMM_MEM_SM100_WORLD_SIZES
                if sm_major is not None and sm_major >= 10
                else ()
            )
            if tp in symm_sizes:
                lines.append(
                    "可 A/B torch 对称内存 allreduce：`--enable-torch-symm-mem --disable-custom-all-reduce`"
                    f"（SM{sm_major}x 支持 world size {'/'.join(map(str, symm_sizes))}；custom allreduce 能处理的消息优先走"
                    " custom allreduce，只开前者不会生效），以 trace 中 allreduce kernel 的单次时长为准。"
                )
        else:
            nvls = (
                "`--enable-nccl-nvls`（NVSwitch 上由交换机做规约）、"
                if hardware is not None and hardware.intra_node_topology == "nvswitch"
                else ""
            )
            lines.append(
                f"大消息可 A/B {nvls}`--enable-symm-mem`（NCCL 对称内存）、`--enable-mscclpp`，以 trace 中通信 kernel 时长为准。"
            )
    single_tp = parallel is None or (parallel.attn_dp == 1 and not parallel.uses_a2a)
    if (
        not group.is_decode_like
        and architecture in knowledge.layernorm_sp_architectures
        and single_tp
    ):
        quant = context.arch.quant if context.arch is not None else None
        lines.append(
            "`--enable-layernorm-sp`（SGLang 只对 "
            + ", ".join(knowledge.layernorm_sp_architectures)
            + " 开放，只作用于 prefill）把 row-parallel 的 allreduce 拆成 reduce-scatter + all-gather，通信量不变；"
            "未量化、无 bias 的 BF16/FP16 线性层在 torch symm_mem 可用时走 fused matmul+reduce-scatter / "
            "all-gather+matmul，通信与 GEMM 重叠。"
            + (f"本模型是 {quant.method} 量化，线性层不满足这条快路径的条件。" if quant else "")
        )
        references += [source("layernorm_sp"), source("layernorm_sp_fused_matmul")]
    if hardware is not None and platform not in {"cuda", "rocm", "npu", "musa"}:
        lines.append(
            f"{hardware.name} 的集合通信经 torch.distributed（后端由平台插件的 `get_torch_distributed_backend_str` 给出）。"
            "custom allreduce 只在 CUDA/ROCm 上，NCCL/torch 对称内存与 FlashInfer allreduce 融合只在 CUDA 上；"
            "`DeviceMixin.get_communicator_class` 仍标记为 "
            "[Planned] 且没有被调用，要接入厂商的低延迟 allreduce 需要改 `GroupCoordinator`（见适配分析）。"
        )
        references += [
            source("platform_distributed_backend"),
            source("platform_communicator_planned"),
        ]
    lines.append(
        "减少每 token 通信次数的部署级手段（DP attention、更小的 TP + 多副本、PD 分离）见特性与部署建议。"
    )
    return lines, references


def detect_exposed_comm(context: AnalysisContext) -> list[Finding]:
    """Per phase: collective time that no compute overlaps, headlined by the costliest group."""
    findings: list[Finding] = []
    groups = [group for group in context.primary_groups() if group.step_us > 0]
    for decode_like in (True, False):
        candidates = [
            group
            for group in groups
            if group.is_decode_like == decode_like and group.exposed_comm_us > 0
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda group: -group.exposed_comm_us * group.steps)
        group = candidates[0]
        share = group.exposed_comm_us / group.step_us
        if share < EXPOSED_COMM_MIN_SHARE:
            continue
        step_id = context.representative_step_id(group)
        library = context.hardware.software.collective_library if context.hardware else None
        stats = step_collectives(context.kernels_by_step.get(step_id, []), library)
        expected = expected_collectives(context, group)
        phase = "decode" if decode_like else "prefill"
        evidence = [
            f"{group.label}（{group.steps} 个 step）：通信 kernel {group.comm_us / 1000:.2f} ms/step，其中 "
            f"{group.exposed_comm_us / 1000:.2f} ms 没有与计算重叠（step 的 {share:.0%}）。",
        ]
        if stats:
            evidence.append(
                f"代表 step #{context.steps_by_id[step_id].ordinal}：{_describe_collectives(stats)}。"
            )
        hidden_us = group.exposed_comm_us
        floor = comm_floor_us(context, group)
        if floor is not None and floor < group.comm_us:
            hidden_us = min(hidden_us, floor)
            evidence.append(
                f"各 rank 同一 step 通信时间的最小值为 {floor / 1000:.2f} ms，其余是等待较慢 rank（见 rank 不均衡结论），"
                "收益按最小值计。"
            )
        others = [
            f"{other.label} {other.exposed_comm_us / 1000:.2f} ms（{other.exposed_comm_us / other.step_us:.0%}）"
            for other in candidates[1:MAX_EVIDENCE_GROUPS]
        ]
        if others:
            evidence.append("同阶段其它形状暴露的通信：" + "；".join(others) + "。")
        recommendation, references = _exposed_comm_recommendation(context, group, stats, expected)
        findings.append(
            Finding(
                id=f"overlap.exposed_comm.{phase}",
                kind="comm_overlap",
                severity=severity_for_share(hidden_us / group.step_us),
                title=f"{phase}：{share:.0%} 的 step 时间花在没有被掩盖的通信上",
                evidence=evidence,
                recommendation=recommendation,
                gain_ms_per_step=hidden_us / 1000,
                gain_note="上限：通信全部被计算掩盖",
                scope=group.label,
                references=references,
            )
        )
    return findings


# -------------------------------------------------------------------------- multi-stream
@dataclass(frozen=True)
class StreamUsage:
    group: GroupSummary
    step: StepSpan
    compute_streams: int
    side_compute_us: float
    side_overlapped_us: float

    @property
    def side_overlap_ratio(self) -> float | None:
        return self.side_overlapped_us / self.side_compute_us if self.side_compute_us else None


def _is_compute(kernel: KernelEvent) -> bool:
    return kernel.kind == "kernel" and kernel.category in COMPUTE_CATEGORIES


def stream_usage(context: AnalysisContext, group: GroupSummary) -> StreamUsage:
    """Compute streams of the group's representative step and how much side-stream work overlaps."""
    step_id = context.representative_step_id(group)
    kernels = [kernel for kernel in context.kernels_by_step.get(step_id, []) if _is_compute(kernel)]
    busy: dict[tuple[int, int], float] = defaultdict(float)
    for kernel in kernels:
        busy[(kernel.device, kernel.stream)] += kernel.duration_us
    step = context.steps_by_id[step_id]
    if len(busy) < 2:
        return StreamUsage(group, step, len(busy), 0.0, 0.0)
    main = max(busy, key=lambda stream: busy[stream])
    main_intervals = merge_intervals(
        sorted(
            (kernel.start_us, kernel.end_us)
            for kernel in kernels
            if (kernel.device, kernel.stream) == main
        )
    )
    side_intervals = merge_intervals(
        sorted(
            (kernel.start_us, kernel.end_us)
            for kernel in kernels
            if (kernel.device, kernel.stream) != main
        )
    )
    return StreamUsage(
        group,
        step,
        len(busy),
        total_length(side_intervals),
        intersection_length(side_intervals, main_intervals),
    )


def _raw_quant_config(arch: ModelArch) -> dict[str, Any]:
    quant = arch.raw.get("quantization_config") or arch.raw.get("compression_config")
    return quant if isinstance(quant, dict) else {}


def _raw_quant_method(arch: ModelArch) -> str | None:
    method = _raw_quant_config(arch).get("quant_method")
    return str(method).lower() if method else None


def _shared_expert_left_unquantized(arch: ModelArch) -> bool:
    """The checkpoint keeps the shared-expert body out of quantization (``can_fuse_shared_expert``)."""
    quant = _raw_quant_config(arch)
    sections = [quant]
    if isinstance(quant.get("quantization"), dict):
        sections.append(quant["quantization"])
    return any(
        isinstance(name, str) and "shared_expert" in name and "shared_expert_gate" not in name
        for section in sections
        for key in QUANT_EXCLUDE_KEYS
        if isinstance(section.get(key), list)
        for name in section[key]
    )


def shared_experts_fusion_default(context: AnalysisContext) -> tuple[bool | None, str]:
    """Whether SGLang fuses the shared expert into the routed MoE kernel, and why.

    Mirrors ``shared_experts_fusion_disable_reason`` of the DeepSeek-V3 family and GLM-4.5 for what
    the agent can see; ``None`` when the model has no such rule or the outcome hinges on a launch
    flag (TBO/SBO, ``--enforce-``/``--disable-shared-experts-fusion``, the MoE runner backend).
    """
    arch, parallel = context.arch, context.parallel
    if arch is None or arch.moe is None or not arch.moe.num_shared_experts:
        return None, ""
    architecture = arch.primary_architecture
    platform = _platform(context)
    sm_major = context.sm_major
    method = _raw_quant_method(arch)
    expert_parallel = parallel is not None and parallel.ep > 1
    uses_a2a = parallel is not None and parallel.uses_a2a
    if architecture in DEEPSEEK_SHARED_EXPERTS_FUSION_ARCHITECTURES:
        experts = arch.moe.num_experts
        if method in {"quark", "modelopt"} and _shared_expert_left_unquantized(arch):
            return False, "checkpoint 把 shared expert 留在更高精度，无法并入量化的 routed expert"
        fp4_checkpoint = arch.quant is not None and arch.quant.weight_dtype == "fp4"
        if method == "modelopt" and fp4_checkpoint and sm_major == 9:
            return (
                None,
                "Hopper 上 modelopt_fp4 checkpoint 配 marlin MoE runner 时默认不融合，取决于 MoE runner",
            )
        if uses_a2a:
            return (
                False,
                "DeepEP 类 a2a 后端下默认不融合（`--enforce-shared-experts-fusion` 可强制）",
            )
        if arch.moe.num_shared_experts != 1 or experts not in {
            DEEPSEEK_SHARED_EXPERTS_FUSION_ROUTED_EXPERTS,
            DEEPSEEK_SHARED_EXPERTS_FUSION_QUARK_ROUTED_EXPERTS,
        }:
            return (
                False,
                "只对 256 个（或 Quark checkpoint 的 384 个）routed expert + 1 个 shared expert 融合",
            )
        if experts == DEEPSEEK_SHARED_EXPERTS_FUSION_QUARK_ROUTED_EXPERTS and method != "quark":
            return (
                False,
                "384 个 routed expert 时只有 Quark MXFP4 checkpoint 预先融合了 shared expert",
            )
        if platform is not None and platform not in {"cuda", "rocm", "musa"}:
            return False, "只在 CUDA SM≥80、ROCm gfx942+、MUSA 3.1+ 上融合"
        if platform == "cuda" and sm_major is not None and sm_major < 8:
            return False, "CUDA 需要 SM≥80"
        if expert_parallel and platform != "rocm":
            return False, "EP>1 时只在 ROCm gfx942+ 上融合"
        if method == "w4afp8" or (arch.quant is not None and arch.quant.weight_dtype == "int4"):
            return False, "W4A16/W4AFP8 的 routed 与 shared expert 量化方式不同"
        return True, f"{architecture}、{experts}+1 个专家、未开 EP/DeepEP（且未开 TBO/SBO）"
    if architecture in DEEPSEEK_V2_MODEL_ARCHITECTURES:
        return False, (
            f"{architecture} 继承的 fused_shared_experts_architecture 与 checkpoint 的架构名不一致，SGLang 不融合"
        )
    if architecture == GLM4_MOE_SHARED_EXPERTS_FUSION_ARCHITECTURE:
        if platform is not None and platform not in {"cuda", "rocm"}:
            return False, "只在 CUDA SM≥80、ROCm gfx942+ 上融合"
        if platform == "cuda" and sm_major is not None and sm_major < 8:
            return False, "CUDA 需要 SM≥80"
        if expert_parallel and platform != "rocm":
            return False, "EP>1 时只在 ROCm gfx942+ 上融合"
        if uses_a2a and parallel is not None and parallel.moe_a2a_backend in {"deepep", "mori"}:
            return False, "DeepEP/MORI a2a 下不融合"
        if method == "w4afp8":
            return False, "W4AFP8 的 routed 与 shared expert 量化方式不同"
        return True, f"{architecture}、未开 EP/DeepEP"
    return None, ""


@dataclass(frozen=True)
class DualStreamEstimate:
    shared_us: float
    routed_us: float
    gain_us: float


def _per_step(timings: list[OpTiming], attribute: str, scale: float) -> float:
    return sum(getattr(timing, attribute) * timing.op.layers for timing in timings) * scale


def dual_stream_estimate(
    context: AnalysisContext, group: GroupSummary
) -> DualStreamEstimate | None:
    """Shared vs routed expert time per step and what running them concurrently could save.

    Concurrent kernels still take at least the longer of the two and the combined HBM and compute
    time, so ``gain = shared + routed - max(shared, routed, Σmemory, Σcompute)``: what is left is
    the per-kernel fixed cost of the small shared-expert kernels.
    """
    expectation = context.expectation(group)
    if expectation is None:
        return None
    shared = [timing for timing in expectation.timings if timing.op.name in SHARED_EXPERT_OPS]
    routed = [timing for timing in expectation.timings if timing.op.name in ROUTED_EXPERT_OPS]
    if not shared or not routed:
        return None
    scale = expectation.scale
    shared_us = _per_step(shared, "kernel_us", scale)
    routed_us = _per_step(routed, "kernel_us", scale)
    both = shared + routed
    concurrent_floor = max(
        shared_us,
        routed_us,
        _per_step(both, "memory_us", scale),
        _per_step(both, "compute_us", scale),
    )
    return DualStreamEstimate(
        shared_us, routed_us, max(shared_us + routed_us - concurrent_floor, 0.0)
    )


def has_dual_stream_path(context: AnalysisContext) -> bool:
    architecture = context.architecture
    if architecture is None or context.arch is None or not context.arch.is_moe:
        return False
    return (
        architecture in context.knowledge.dual_stream_architectures
        or architecture in DEEPSEEK_V2_MODEL_ARCHITECTURES
    )


def _dual_stream_recommendation(
    context: AnalysisContext, group: GroupSummary, fusion_note: str
) -> tuple[list[str], dict[str, str], list[str]]:
    hardware = context.hardware
    architecture = context.architecture or ""
    platform = _platform(context)
    deepseek_family = architecture in DEEPSEEK_V2_MODEL_ARCHITECTURES
    lines: list[str] = []
    env: dict[str, str] = {}
    references: list[str] = []
    if group.graph_fraction is not None and group.graph_fraction < EAGER_GRAPH_FRACTION:
        lines.append(
            "本组没有走 CUDA graph（见“没有走 CUDA graph”一条）：双流路径只在 graph 捕获时生效，先解决 graph。"
        )
    if platform == "rocm" and deepseek_family:
        env["SGLANG_ROCM_USE_MULTI_STREAM"] = "1"
        lines.append(
            "ROCm 上设置 `SGLANG_ROCM_USE_MULTI_STREAM=1` 后 DeepSeek 系模型才创建 alt stream。"
        )
        references.append(source("multi_stream_env"))
    elif platform == "npu" and (
        deepseek_family or architecture in NPU_SHARED_EXPERT_STREAM_ARCHITECTURES
    ):
        env["SGLANG_NPU_USE_MULTI_STREAM"] = "1"
        lines.append(
            "NPU 上设置 `SGLANG_NPU_USE_MULTI_STREAM=1` 后 shared expert 才放到独立 stream 上。"
        )
        references.append(
            source("multi_stream_env") if deepseek_family else source("npu_shared_expert_stream")
        )
    elif (
        platform is not None and platform != "cuda" and not (deepseek_family and platform == "musa")
    ):
        device = hardware.name if hardware else platform
        lines.append(
            f"{architecture} 的模型文件只在 "
            + ("CUDA/MUSA（或设置了 ROCm/NPU 多流环境变量）" if deepseek_family else "CUDA")
            + f" 上创建 alt stream：{device} 上要让平台插件满足这个条件，或者在模型文件里按平台创建 side stream（适配项）。"
        )
        references.append(source("alt_stream_platforms"))
    if fusion_note:
        lines.append(fusion_note)
    if deepseek_family:
        lines.append(
            "DeepSeek 系在 breakable CUDA graph（`--cuda-graph-backend-decode breakable`）里不走 "
            "`forward_normal_dual_stream`；breakable/piecewise graph 内只有设置 `SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM=1` "
            "且 `--moe-runner-backend flashinfer_trtllm` 时才有另一条双流路径。默认的 full graph 不受影响。"
        )
        references.append(source("dual_stream_deepseek_breakable"))
    lines.append(
        "生效后 trace 中 shared expert 的 GEMM 应与 routed experts 的 kernel 在两个 stream 上重叠；若 profiler 把 "
        "graph 内的并行分支记在同一个 stream 上，用 Nsight Systems 确认。"
    )
    return lines, env, references


def detect_dual_stream(context: AnalysisContext) -> tuple[list[Finding], list[StreamUsage]]:
    """Decode steps that run on one stream although the model has a dual-stream MoE path.

    No finding when the MoE goes through an a2a backend (``forward_deepep``; TBO is the overlap
    there) or when SGLang fuses the shared expert into the routed kernel by default: the fused
    kernel already reads the shared expert's weights in the same pass, which is what a second
    stream would win back.
    """
    usages = [
        stream_usage(context, group) for group in context.primary_groups() if group.step_us > 0
    ]
    if not has_dual_stream_path(context):
        return [], usages
    parallel = context.parallel
    if parallel is not None and parallel.uses_a2a:
        return [], usages
    fused, fusion_reason = shared_experts_fusion_default(context)
    if fused:
        return [], usages
    candidates = [
        (usage, dual_stream_estimate(context, usage.group))
        for usage in usages
        if usage.group.is_decode_like and usage.compute_streams == 1
    ]
    if not candidates:
        return [], usages
    if all(estimate is None for _, estimate in candidates):
        headline, estimate = max(
            candidates, key=lambda item: item[0].group.steps * item[0].group.step_us
        )
    else:
        candidates = [item for item in candidates if item[1] is not None]
        candidates.sort(key=lambda item: -item[1].gain_us * item[0].group.steps)
        headline, estimate = candidates[0]
    group = headline.group
    share = estimate.gain_us / group.step_us if estimate is not None else None
    if share is not None and share < DUAL_STREAM_MIN_SHARE:
        return [], usages
    architecture = context.architecture
    deepseek_family = architecture in DEEPSEEK_V2_MODEL_ARCHITECTURES
    evidence = [
        f"{group.label}（代表 step #{headline.step.ordinal}）的计算 kernel 只出现在 1 个 stream 上。",
        f"{architecture} 的 MoE 块有 `forward_normal_dual_stream`：shared expert 在 alt stream 上与 routed experts "
        "并行。条件是 `get_is_capture_mode()` 为真（decode 的 CUDA graph 捕获，含 breakable graph；eager 的 prefill 不走）、"
        "alt stream 存在、不走 a2a"
        + (
            "，且不在 breakable graph 内、shared expert 没有被融合进 MoE kernel。"
            if deepseek_family
            else "。"
        ),
    ]
    fusion_note = ""
    if fused is False and fusion_reason:
        evidence.append(f"shared expert 没有被融合进 routed MoE kernel：{fusion_reason}。")
    elif fused is None and fusion_reason:
        fusion_note = f"shared expert 是否被融合进 routed MoE kernel 取决于启动参数：{fusion_reason}；融合时不走双流。"
    if estimate is not None:
        moe_layers = context.arch.num_moe_layers if context.arch else None
        evidence.append(
            f"成本模型：shared expert {estimate.shared_us / 1000:.2f} ms/step、routed experts "
            f"{estimate.routed_us / 1000:.2f} ms/step"
            + (f"（{moe_layers} 个 MoE 层）" if moe_layers else "")
            + f"；并行后仍受两者合计的 HBM 读取量限制，最多省 {estimate.gain_us / 1000:.2f} ms"
            "（主要是 shared expert 小 kernel 的固定开销）。"
        )
    others = [
        f"{usage.group.label} 约省 {other.gain_us / 1000:.2f} ms（{other.gain_us / usage.group.step_us:.1%}）"
        for usage, other in candidates[1:MAX_EVIDENCE_GROUPS]
        if other is not None
    ]
    if others:
        evidence.append("其它 decode 形状：" + "；".join(others) + "。")
    recommendation, env, references = _dual_stream_recommendation(context, group, fusion_note)
    findings = [
        Finding(
            id="overlap.dual_stream.decode",
            kind="multi_stream",
            severity=severity_for_share(share) if share is not None else "low",
            title=f"decode：{architecture} 的 shared expert 没有与 routed experts 并行",
            evidence=evidence,
            recommendation=recommendation,
            env=env,
            gain_ms_per_step=estimate.gain_us / 1000 if estimate is not None else None,
            gain_note="上限：按 HBM 与算力合计的下界估算，实际并发还受 SM 分配影响",
            scope=group.label,
            references=[
                source("dual_stream_deepseek"),
                source("dual_stream_capture_mode"),
                *references,
            ],
        )
    ]
    return findings, usages


# ------------------------------------------------------------------------ L2 prefetch
@dataclass(frozen=True)
class PrefetchWindow:
    window: str
    window_us: float
    consumer: str
    consumer_us: float
    staged_bytes: float
    saved_us: float


@dataclass
class GroupPrefetch:
    group: GroupSummary
    step: StepSpan
    tokens: int | None
    intensity: float
    ridge: float
    windows: list[PrefetchWindow] = field(default_factory=list)

    @property
    def saved_us(self) -> float:
        return sum(window.saved_us for window in self.windows)

    @property
    def memory_bound_tokens(self) -> float:
        """Tokens per GEMM below which the GEMM's weights take longer to read than to multiply."""
        return self.ridge * (self.tokens or 0) / self.intensity if self.intensity else 0.0


def gemm_intensity_and_ridge(context: AnalysisContext, tokens: int) -> tuple[float, float] | None:
    """Arithmetic intensity of a ``tokens``-row GEMM on the model's linear weights, and the ridge.

    For a GEMM with far fewer rows than weight columns the intensity is ``2·M / bytes_per_weight``;
    the ridge uses the achievable compute and bandwidth (efficiency assumptions).
    """
    hardware = context.hardware
    if hardware is None or tokens <= 0:
        return None
    if context.cost_model is not None:
        compute_dtype = context.cost_model.linear_compute_dtype
    else:
        compute_dtype = "bf16"
    if not hardware.supports_compute_dtype(compute_dtype):
        compute_dtype = "bf16" if hardware.supports_compute_dtype("bf16") else "fp16"
    if context.arch is not None:
        weight_bytes = weight_precision(context.arch).linear
    else:
        weight_bytes = bytes_per_element(compute_dtype)
    intensity = 2.0 * tokens / weight_bytes
    ridge = (
        hardware.peak_flops(compute_dtype)
        * hardware.efficiency.gemm_compute
        / (hardware.memory_bandwidth_bytes_per_s * hardware.efficiency.memory_bandwidth)
    )
    return intensity, ridge


def _mean(windows: list[PrefetchWindow], attribute: str) -> float:
    return sum(getattr(window, attribute) for window in windows) / len(windows) if windows else 0.0


def _consumer_after(events: list[KernelEvent], position: int) -> KernelEvent | None:
    for following in events[position + 1 : position + 1 + PREFETCH_LOOKAHEAD_KERNELS]:
        if following.category == "gemm":
            return following
        if following.category in PREFETCH_BARRIER_CATEGORIES:
            return None
    return None


def analyze_prefetch(context: AnalysisContext, group: GroupSummary) -> GroupPrefetch | None:
    """Weight bytes the decode step could stage in L2 during its collectives, and the time saved.

    Per window: ``staged = min(L2 budget, window × spare HBM bandwidth, consumer bytes)``; the
    consumer then takes ``max(compute, remaining HBM time, kernel floor)``. Reading staged weights
    from L2 is treated as free, so the result is an upper bound.
    """
    hardware = context.hardware
    if hardware is None or hardware.l2_cache_mb <= 0 or not group.is_decode_like:
        return None
    tokens = step_tokens(context, group)
    if tokens is None:
        return None
    bounds = gemm_intensity_and_ridge(context, tokens)
    if bounds is None:
        return None
    intensity, ridge = bounds
    step_id = context.representative_step_id(group)
    analysis = GroupPrefetch(group, context.steps_by_id[step_id], tokens, intensity, ridge)
    if intensity >= ridge:
        return analysis
    bandwidth = hardware.memory_bandwidth_bytes_per_s * hardware.efficiency.memory_bandwidth
    l2_budget = hardware.l2_cache_bytes * PREFETCH_L2_BUDGET_FRACTION
    floor_us = hardware.latency.min_kernel_us
    compute_fraction = intensity / ridge
    streams: dict[tuple[int, int], list[KernelEvent]] = defaultdict(list)
    for kernel in context.kernels_by_step.get(step_id, []):
        streams[(kernel.device, kernel.stream)].append(kernel)
    for events in streams.values():
        for position, kernel in enumerate(events):
            if kernel.category != "comm":
                continue
            consumer = _consumer_after(events, position)
            if consumer is None or consumer.duration_us <= LATENCY_REGIME_FACTOR * floor_us:
                continue
            consumer_bytes = consumer.duration_us * 1e-6 * bandwidth
            window_bytes = kernel.duration_us * 1e-6 * bandwidth * PREFETCH_BANDWIDTH_SPARE
            staged = min(l2_budget, window_bytes, consumer_bytes)
            remaining_us = (consumer_bytes - staged) / bandwidth * 1e6
            new_us = max(consumer.duration_us * compute_fraction, remaining_us, floor_us)
            saved = consumer.duration_us - new_us
            if saved <= 0:
                continue
            analysis.windows.append(
                PrefetchWindow(
                    window=compact_kernel_name(kernel.name),
                    window_us=kernel.duration_us,
                    consumer=compact_kernel_name(consumer.name),
                    consumer_us=consumer.duration_us,
                    staged_bytes=staged,
                    saved_us=saved,
                )
            )
    return analysis


def detect_prefetch_opportunities(
    context: AnalysisContext,
) -> tuple[list[Finding], list[GroupPrefetch]]:
    """L2 weight prefetch during decode collectives, headlined by the group that gains the most."""
    analyses = [
        analysis
        for group in context.primary_groups()
        if group.step_us > 0
        for analysis in [analyze_prefetch(context, group)]
        if analysis is not None
    ]
    hardware = context.hardware
    if hardware is None or not analyses:
        return [], analyses
    ranked = sorted(analyses, key=lambda item: -item.saved_us * item.group.steps)
    headline = ranked[0]
    group = headline.group
    share = headline.saved_us / group.step_us
    if share < PREFETCH_MIN_SHARE:
        return [], analyses
    windows = sorted(headline.windows, key=lambda item: -item.saved_us)
    pairs: dict[tuple[str, str], list[PrefetchWindow]] = defaultdict(list)
    for window in windows:
        pairs[(window.window, window.consumer)].append(window)
    top_pairs = sorted(pairs.items(), key=lambda item: -sum(w.saved_us for w in item[1]))[:2]
    pair_text = "；".join(
        f"{window} → {consumer} ×{len(items)}（通信平均 {_mean(items, 'window_us'):.0f} us，期间平均可预取 "
        f"{_format_bytes(_mean(items, 'staged_bytes'))}；GEMM 平均 {_mean(items, 'consumer_us'):.0f} us，"
        f"合计约省 {sum(item.saved_us for item in items):.0f} us）"
        for (window, consumer), items in top_pairs
    )
    evidence = [
        f"{group.label}（代表 step #{headline.step.ordinal}）有 {len(windows)} 个“通信 → GEMM”窗口：{pair_text}。",
        f"合计约省 {headline.saved_us:.0f} us/step（step {group.step_us / 1000:.2f} ms 的 {share:.1%}）。",
        f"每卡 M={headline.tokens} 个 token 时 GEMM 算术强度 {headline.intensity:.0f} FLOPs/byte，"
        f"{hardware.name} 的有效屋脊 {headline.ridge:.0f}：M 超过约 {headline.memory_bound_tokens:.0f} 后 GEMM "
        "转为计算主导，预取不再省时间。",
    ]
    others = [
        f"{item.group.label} 约省 {item.saved_us:.0f} us/step（M={item.tokens}）"
        for item in ranked[1:MAX_EVIDENCE_GROUPS]
    ]
    if others:
        evidence.append("其它 decode 形状：" + "；".join(others) + "。")
    evidence.append(
        f"估算方式：L2 预算 {hardware.l2_cache_mb * PREFETCH_L2_BUDGET_FRACTION:.0f} MB（L2 的 "
        f"{PREFETCH_L2_BUDGET_FRACTION:.0%}），通信期间可用 {PREFETCH_BANDWIDTH_SPARE:.0%} 的 HBM 带宽；从 L2 读预取的"
        "权重按不耗时计，结果是上限。"
    )
    cuda = _platform(context) == "cuda"
    if not hardware.software.l2_prefetch:
        mechanism = (
            f"{hardware.name} 的硬件描述没有声明 L2 预取指令（software.l2_prefetch=false）：先确认厂商 ISA/运行时"
            "是否提供 L2 预取或驻留控制，再按下面的方式接入。"
        )
    elif cuda:
        mechanism = (
            "在另一个 stream 上用 PTX `cp.async.bulk.prefetch.L2.global`（sm_90+）把下一个 GEMM 的权重预取进 L2，"
            "GEMM 随后从 L2 读取；SGLang 已有逐地址的 `prefetch.global.L2` 辅助函数（`prefetch_L2`），只用在 "
            "concat_mla 等 kernel 内部。"
        )
    else:
        mechanism = (
            "硬件描述声明了 L2 预取（software.l2_prefetch=true）：在另一个 stream 上用厂商 ISA 的 L2 预取指令把下一个 "
            "GEMM 的权重预取进 L2，GEMM 随后从 L2 读取；SGLang 的 `prefetch_L2` 是 CUDA PTX 写法，只能作参考。"
        )
    graph = "CUDA graph" if cuda else "graph"
    recommendation = [
        f"{hardware.name} 的 L2 为 {hardware.l2_cache_mb:g} MB、HBM {hardware.memory_bandwidth_gbps / 1000:.1f} TB/s："
        "decode 的集合通信主要是同步延迟，期间 HBM 基本空闲。" + mechanism,
        "SGLang 目前没有跨 kernel 的权重 L2 预取，属于框架/kernel 改动：在 allreduce 之前于 side stream 上发起预取，"
        f"{graph} 捕获时一并捕获 fork/join；或把预取放进"
        + (
            " custom allreduce kernel；也可以试 stream 的 access policy window（persisting L2），需先在该卡上确认可设置的 "
            "persisting L2 大小。"
            if cuda
            else "厂商的 allreduce kernel；运行时若提供 L2 驻留控制也可以用它固定下一层的权重。"
        ),
        "预取只是 hint，不影响正确性，但会与并发 kernel 争用 L2：改完用 A/B trace 确认 GEMM 时间下降，并用 "
        + (
            "Nsight Compute 看 L2 命中率（lts__t_sector_hit_rate）。"
            if cuda
            else "厂商 profiler 看 GEMM 的 L2 命中率。"
        ),
    ]
    findings = [
        Finding(
            id="overlap.l2_prefetch.decode",
            kind="weight_prefetch",
            severity=severity_for_share(share),
            title=f"decode：集合通信期间可把下一个 GEMM 的权重预取进 L2（{group.label}）",
            evidence=evidence,
            recommendation=recommendation,
            gain_ms_per_step=headline.saved_us / 1000,
            gain_note="上限：从 L2 读取按不耗时计，需 A/B 验证",
            scope=group.label,
            references=[source("l2_prefetch_helper")],
        )
    ]
    return findings, analyses
