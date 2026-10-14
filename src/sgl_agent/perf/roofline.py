"""Roofline view of a model on a device: per-operator bounds, batch sweeps and derived insights."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.model.config import ModelArch
from sgl_agent.model.ops import BatchShape, ExecutionConfig, StepCostModel, StepEstimate
from sgl_agent.model.parallel import MemoryKnobs, ParallelConfig, plan_memory
from sgl_agent.model.params import weight_precision

HIGHER_MEM_FRACTIONS = (0.90, 0.93, 0.95)
"""Explicit ``--mem-fraction-static`` values tried when the SGLang default leaves too little KV."""


@dataclass(frozen=True)
class OpRow:
    name: str
    kind: str
    bound: str
    time_ms: float
    share: float
    tflop: float
    gbytes: float
    arithmetic_intensity: float
    layers: float


@dataclass
class ScenarioResult:
    label: str
    phase: str
    batch_size: int
    new_tokens: int
    total_ms: float
    tokens_per_s_per_gpu: float
    by_kind_ms: dict[str, float]
    by_bound_ms: dict[str, float]
    rows: list[OpRow]


@dataclass
class RooflineReport:
    model: str
    hardware: str
    parallel: str
    linear_compute_dtype: str
    expert_compute_dtype: str
    ridge_points: dict[str, float]
    effective_ridge_points: dict[str, float]
    decode_gemm_compute_bound_tokens: float
    moe_compute_bound_global_tokens: float | None
    max_total_tokens: int
    memory: dict[str, Any]
    scenarios: list[ScenarioResult]
    batch_sweep: list[dict[str, float]]
    insights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rows(estimate: StepEstimate, limit: int) -> list[OpRow]:
    total = estimate.kernel_us or 1.0
    rows = []
    for timing in estimate.top_ops(limit):
        op = timing.op
        rows.append(
            OpRow(
                name=op.name,
                kind=op.kind,
                bound=timing.bound,
                time_ms=timing.total_us * estimate.scale / 1000,
                share=timing.total_us * estimate.scale / total,
                tflop=op.flops * op.layers / 1e12,
                gbytes=(op.bytes + op.comm_bytes) * op.layers / 1e9,
                arithmetic_intensity=op.arithmetic_intensity,
                layers=op.layers,
            )
        )
    return rows


def default_scenarios(
    context_len: int = 4096, prompt_len: int = 4096
) -> list[tuple[str, BatchShape]]:
    return [
        ("decode bs=1", BatchShape.decode(1, context_len)),
        ("decode bs=32", BatchShape.decode(32, context_len)),
        ("decode bs=128", BatchShape.decode(128, context_len)),
        (f"prefill 1×{prompt_len}", BatchShape.prefill(1, prompt_len)),
    ]


def analyze_roofline(
    arch: ModelArch,
    hardware: HardwareSpec,
    parallel: ParallelConfig,
    execution: ExecutionConfig | None = None,
    scenarios: list[tuple[str, BatchShape]] | None = None,
    sweep_context: int = 4096,
    top_ops: int = 12,
) -> RooflineReport:
    execution = execution or ExecutionConfig()
    errors = parallel.validate(arch)
    if errors:
        raise ValueError("; ".join(errors))
    model = StepCostModel(arch, hardware, parallel, execution)
    efficiency = hardware.efficiency
    ridge = {
        dtype: hardware.ridge_point(dtype)
        for dtype in ("bf16", "fp8", "int8")
        if hardware.supports_compute_dtype(dtype)
    }
    effective = {
        dtype: value * efficiency.gemm_compute / efficiency.memory_bandwidth
        for dtype, value in ridge.items()
    }
    precision = weight_precision(arch, execution.weight_dtype)
    linear_dtype = model.linear_compute_dtype
    decode_threshold = hardware.ridge_point(linear_dtype) * precision.linear / 2.0
    moe_threshold = None
    if arch.moe:
        expert_ridge = hardware.ridge_point(model.expert_compute_dtype) * precision.experts / 2.0
        moe_threshold = expert_ridge * arch.moe.num_experts / arch.moe.top_k
    memory = plan_memory(
        arch, hardware, parallel, weight_dtype=execution.weight_dtype, kv_dtype=execution.kv_dtype
    )
    alternatives = []
    for fraction in HIGHER_MEM_FRACTIONS:
        if fraction <= memory.mem_fraction_static:
            continue
        candidate = plan_memory(
            arch,
            hardware,
            parallel,
            weight_dtype=execution.weight_dtype,
            kv_dtype=execution.kv_dtype,
            knobs=MemoryKnobs(
                mem_fraction_static=fraction, chunked_prefill_size=4096, cuda_graph_max_bs=128
            ),
        )
        alternatives.append(
            {
                "mem_fraction_static": fraction,
                "chunked_prefill_size": 4096,
                "cuda_graph_max_bs_decode": 128,
                "max_total_tokens": candidate.max_total_tokens,
            }
        )

    results: list[ScenarioResult] = []
    for label, shape in scenarios or default_scenarios(sweep_context):
        estimate = model.estimate(shape)
        produced = (
            shape.num_seqs if shape.phase == "decode" else shape.new_tokens
        ) * parallel.attn_dp
        results.append(
            ScenarioResult(
                label=label,
                phase=shape.phase,
                batch_size=shape.num_seqs,
                new_tokens=shape.new_tokens,
                total_ms=estimate.total_us / 1000,
                tokens_per_s_per_gpu=produced / (estimate.total_us / 1e6) / parallel.num_gpus,
                by_kind_ms={kind: value / 1000 for kind, value in estimate.by_kind().items()},
                by_bound_ms={bound: value / 1000 for bound, value in estimate.by_bound().items()},
                rows=_rows(estimate, top_ops),
            )
        )

    sweep: list[dict[str, float]] = []
    batch = 1
    while batch <= 1024:
        step_ms = model.estimate(BatchShape.decode(batch, sweep_context)).total_us / 1000
        global_batch = batch * parallel.attn_dp
        sweep.append(
            {
                "batch_per_dp_rank": batch,
                "global_batch": global_batch,
                "step_ms": step_ms,
                "tokens_per_s_per_gpu": global_batch / (step_ms / 1000) / parallel.num_gpus,
                "kv_tokens_needed_per_rank": batch * sweep_context,
                "fits_kv": float(batch * sweep_context <= memory.max_total_tokens),
            }
        )
        batch *= 2

    report = RooflineReport(
        model=arch.name,
        hardware=hardware.name,
        parallel=parallel.describe(),
        linear_compute_dtype=linear_dtype,
        expert_compute_dtype=model.expert_compute_dtype,
        ridge_points=ridge,
        effective_ridge_points=effective,
        decode_gemm_compute_bound_tokens=decode_threshold,
        moe_compute_bound_global_tokens=moe_threshold,
        max_total_tokens=memory.max_total_tokens,
        memory={
            "weights_gb_per_gpu": memory.weights_bytes_per_gpu / 1e9,
            "kv_bytes_per_token_per_gpu": memory.kv_bytes_per_token_per_gpu,
            "mem_fraction_static": memory.mem_fraction_static,
            "mem_fraction_is_sglang_default": memory.mem_fraction_is_default,
            "kv_pool_gb_per_gpu": memory.kv_pool_bytes_per_gpu / 1e9,
            "chunked_prefill_size": memory.chunked_prefill_size,
            "cuda_graph_max_bs_decode": memory.cuda_graph_max_bs,
            "reserve_mb": memory.reserve_breakdown_mb,
            "higher_fraction_alternatives": alternatives,
        },
        scenarios=results,
        batch_sweep=sweep,
    )
    report.insights = _insights(arch, hardware, parallel, report, precision.linear)
    return report


def _insights(
    arch: ModelArch,
    hardware: HardwareSpec,
    parallel: ParallelConfig,
    report: RooflineReport,
    linear_bpp: float,
) -> list[str]:
    notes: list[str] = []
    ridge_text = "，".join(
        f"{dtype.upper()} {value:.0f}" for dtype, value in report.ridge_points.items()
    )
    notes.append(
        f"{hardware.name} 屋脊点（FLOPs/Byte）：{ridge_text}。线性层以 {report.linear_compute_dtype.upper()} 计算、"
        f"权重 {linear_bpp:.2f} B/参数时，decode GEMM 的 M（每卡 token 数）超过约 "
        f"{report.decode_gemm_compute_bound_tokens:.0f} 就转为算力受限；继续加大 batch 只增加延迟、不再摊薄权重读取。"
    )
    if report.moe_compute_bound_global_tokens:
        notes.append(
            f"MoE 专家 GEMM 需要每步约 {report.moe_compute_bound_global_tokens:.0f} 个 token（全局）才能让每个专家"
            "进入算力受限区；低于这个值时 decode 的主要代价是读取被激活专家的权重，提高并发或用 EP 聚合 token 收益最大。"
        )
    if report.linear_compute_dtype == "bf16" and linear_bpp < 2:
        notes.append(
            "当前是 weight-only 量化（反量化后 BF16 计算）：算力受限阈值随权重字节数一起降低，大 batch 下收益很小。"
        )
    if parallel.attn_dp > 1:
        notes.append(
            f"DP attention 下各场景的 batch/序列数指每个 DP rank；全局 batch = 该值 × {parallel.attn_dp}，"
            "MoE/稠密层看到的是全局 token 数。"
        )
    for scenario in report.scenarios:
        total = sum(scenario.by_kind_ms.values()) or 1.0
        top_kind, top_ms = next(iter(scenario.by_kind_ms.items()))
        bound_total = sum(scenario.by_bound_ms.values()) or 1.0
        dominant_bound = max(scenario.by_bound_ms.items(), key=lambda item: item[1])
        notes.append(
            f"{scenario.label}：单步 {scenario.total_ms:.2f} ms，最大头是 {top_kind}（{top_ms / total:.0%}），"
            f"{dominant_bound[0]} 受限时间占 {dominant_bound[1] / bound_total:.0%}，"
            f"每卡 {scenario.tokens_per_s_per_gpu:,.0f} token/s。"
        )
        comm_share = scenario.by_kind_ms.get("comm", 0.0) / total
        if comm_share > 0.15:
            notes.append(
                f"{scenario.label}：通信占 {comm_share:.0%}，值得做通算融合/overlap（见特性建议与 trace 分析）。"
            )
    decode_rows = [
        row for scenario in report.scenarios if scenario.phase == "decode" for row in scenario.rows
    ]
    mla_rows = [row for row in decode_rows if row.name == "attn.core_mla_decode"]
    if mla_rows and all(row.bound == "compute" for row in mla_rows):
        notes.append(
            f"MLA decode 注意力在该切分下已经算力受限（AI 约 {mla_rows[0].arithmetic_intensity:.0f} FLOPs/B，"
            f"高于 {hardware.name} 的有效屋脊点）。此时 DP attention / DCP 不减少每个 token 的注意力计算量，"
            "主要价值是成倍扩大 KV 容量（更大 batch）；只有在注意力仍是带宽受限的高算力卡上，它们才会直接加速注意力。"
        )
    reserve = report.memory["reserve_mb"]
    padding_mb = reserve.get("dp_attention_padding", 0.0)
    if report.memory["mem_fraction_is_sglang_default"] and padding_mb > 8 * 1024:
        notes.append(
            f"SGLang 默认 mem_fraction_static={report.memory['mem_fraction_static']:.3f}：DP attention 按 decode graph "
            f"max_bs={report.memory['cuda_graph_max_bs_decode']} 额外预留 {padding_mb / 1024:.1f} GiB/卡。"
            "按实际每 rank 并发显式设置 --cuda-graph-max-bs-decode（DeepEP low-latency 默认每 rank 最多 128 token）"
            "或直接给 --mem-fraction-static。"
        )
    fits = [entry for entry in report.batch_sweep if entry["fits_kv"]]
    context = report.batch_sweep[0]["kv_tokens_needed_per_rank"] if report.batch_sweep else 0
    if fits:
        best = max(fits, key=lambda entry: entry["tokens_per_s_per_gpu"])
        notes.append(
            f"KV 池可容纳 {report.max_total_tokens:,} token/DP rank；在 {context:.0f}"
            f" 上下文下，KV 能放下的最大扫描点为每 rank batch {int(fits[-1]['batch_per_dp_rank'])}，"
            f"吞吐最高点 batch {int(best['batch_per_dp_rank'])}（每卡 {best['tokens_per_s_per_gpu']:,.0f} token/s）。"
        )
    else:
        notes.append(
            "按当前显存规划 KV 池放不下扫描的最小 batch：需要更多卡、FP8 KV、DCP/DP attention 或更高 mem_fraction_static。"
        )
    small_pool = report.max_total_tokens < max(context * 32, 65536)
    better = [
        item
        for item in report.memory["higher_fraction_alternatives"]
        if item["max_total_tokens"] > report.max_total_tokens * 1.5
    ]
    if report.memory["mem_fraction_is_sglang_default"] and small_pool and better:
        options = "，".join(
            f"{item['mem_fraction_static']:.2f} → {item['max_total_tokens']:,} token"
            for item in better
        )
        notes.append(
            f"默认 mem_fraction_static={report.memory['mem_fraction_static']:.3f} 只给 KV 留下 "
            f"{report.memory['kv_pool_gb_per_gpu']:.1f} GB/卡。显式提高并配合 --chunked-prefill-size 4096、"
            f"--cuda-graph-max-bs-decode 128 压低激活/graph 预留：{options}。上线前用真实最长 prompt 压测确认不 OOM。"
        )
    return notes
