"""MoE expert parallelism (DeepEP, EPLB), batch overlap (TBO/SBO) and the hierarchical KV cache."""

from __future__ import annotations

import math
from typing import Any

from sgl_agent.features.base import FeatureAdvice, FeatureContext, pct, ratio
from sgl_agent.knowledge.facts import (
    DEEPEP_LOW_LATENCY_TOKENS_DEFAULT,
    DEEPEP_LOW_LATENCY_TOKENS_LIMIT,
    HICACHE_ALLOCATION_SLACK,
    HICACHE_DEFAULT_RATIO,
    HICACHE_HOST_MEMORY_FRACTION,
    HICACHE_HOST_RESERVE_BYTES,
    HICACHE_SELECTIVE_WRITE_THRESHOLD,
    HICACHE_TRANSFER_PLATFORMS,
    SBO_UNSUPPORTED_SM_MAJOR,
    TBO_PREFILL_ONLY_ARCHITECTURES,
    TBO_TOKEN_DISTRIBUTION_THRESHOLD,
    source,
)
from sgl_agent.model.ops import BatchShape, OpTiming, StepEstimate
from sgl_agent.model.parallel import ParallelConfig
from sgl_agent.model.params import count_params, weight_precision
from sgl_agent.workload import mean_length

GIB = 1024**3
DEEPEP_ASSUMED_NUM_SMS = 20
"""Assumed SMs the DeepEP normal kernels occupy: ``Buffer.num_sms`` class default of the DeepEP
library (deep_ep/buffer.py, not vendored in SGLang). ``--deepep-config`` pins the real value."""
A2A_OPS = frozenset({"moe.dispatch_a2a", "moe.combine_a2a"})
DP_MOE_COMM_OPS = frozenset({"moe.dp_allgather", "moe.reduce_scatter"})
RECOMMEND_GAIN = 1.15
OPTIONAL_GAIN = 1.03
EPLB_MAX_KV_LOSS = 0.2
"""Redundant expert slots that take more than this share of the KV pool are not worth it."""


# ---------------------------------------------------------------------- HiCache
def lru_hit_rates(probabilities: list[float], sizes: list[float], capacity: float) -> list[float]:
    """Per-item LRU hit probability for a cache of ``capacity`` tokens (Che's approximation).

    Item ``i`` is requested with probability ``probabilities[i]`` per request and occupies
    ``sizes[i]`` tokens; the characteristic time ``T`` solves ``sum size_i (1 - e^(-p_i T)) = C``.
    """
    if capacity >= sum(sizes):
        return [1.0] * len(sizes)
    if capacity <= 0:
        return [0.0] * len(sizes)

    def occupancy(horizon: float) -> float:
        return sum(
            size * -math.expm1(-p * horizon) for p, size in zip(probabilities, sizes, strict=True)
        )

    low, high = 0.0, 1.0
    while occupancy(high) < capacity and high < 1e15:
        high *= 2
    for _ in range(100):
        middle = (low + high) / 2
        if occupancy(middle) < capacity:
            low = middle
        else:
            high = middle
    horizon = (low + high) / 2
    return [-math.expm1(-p * horizon) for p in probabilities]


def lru_hit_rate(probabilities: list[float], sizes: list[float], capacity: float) -> float:
    rates = lru_hit_rates(probabilities, sizes, capacity)
    return sum(p * h for p, h in zip(probabilities, rates, strict=True))


def hicache_host_budget(ctx: FeatureContext, parallel: ParallelConfig) -> dict[str, float]:
    """Host bytes one rank may use for its pool (``auto_size_hicache`` with the default fraction)."""
    ranks = min(parallel.num_gpus, ctx.hardware.devices_per_node)
    free = max(ctx.host_memory_gb * GIB - HICACHE_HOST_RESERVE_BYTES, 0.0)
    budget = free / ranks * HICACHE_HOST_MEMORY_FRACTION * (1 - HICACHE_ALLOCATION_SLACK)
    return {"ranks_per_host": ranks, "budget_bytes_per_rank": budget}


def _hicache_adaptation(ctx: FeatureContext) -> list[str]:
    pool = "mla" if ctx.arch.attention.is_mla else "mha"
    platform = ctx.profile.platform if ctx.profile.in_tree else None
    if platform in HICACHE_TRANSFER_PLATFORMS[pool]:
        return []
    target = (
        "sglang.srt.mem_cache.pool_host.mla.MLATokenToKVPoolHost"
        if pool == "mla"
        else "sglang.srt.mem_cache.pool_host.mha.MHATokenToKVPoolHost"
    )
    platforms = "/".join(HICACHE_TRANSFER_PLATFORMS[pool])
    device_type = ctx.device_type
    return [
        f"host pool 的传输函数（transfer_kv_*）只在 {platforms} 上导入：`{target}` 的 "
        "load_to_device_per_layer / backup_from_device_all_layer 在这里引用未定义的名字，kernel 与 direct "
        "两种 I/O 后端都跑不起来",
        f'在通用插件（entry point 组 sglang.srt.plugins）里用 `HookRegistry.register("{target}'
        '.load_to_device_per_layer", fn, HookType.REPLACE)`（backup_from_device_all_layer 同理）换成设备原生实现：'
        "按 host/device 索引做逐层异步 H2D/D2H 拷贝（独立 stream + event）；固定 `--hicache-mem-layout layer_first "
        "--hicache-io-backend direct`，只需实现这一种布局",
        f"host 缓冲区分配走 `ALLOC_MEMORY_FUNCS[{device_type!r}]`：除 npu/musa/xpu 外默认是 cudaHostRegister。"
        f"插件需登记 `ALLOC_MEMORY_FUNCS[{device_type!r}] = alloc_with_pin_memory`，并让 torch 能为该设备分配 "
        "pinned host 内存（PrivateUse1 的 pinned allocator）",
        "capabilities.hicache_device_kernels 保持 False：它只控制写回 staging 的 JIT kernel",
    ]


def advise_hicache(ctx: FeatureContext) -> FeatureAdvice:
    key, title = "hicache", "分层 KV 缓存 HiCache（--enable-hierarchical-cache）"
    references = [
        source("hicache_auto_size"),
        source("hicache_ratio_default"),
        source("hicache_write_policy"),
        source("hicache_layout_rewrite"),
        source("hicache_host_pool_io"),
        source("hicache_oot_transfer"),
        source("hicache_host_alloc"),
        source("hicache_dcp"),
        source("router_dp_aware"),
        source("hicache_cache_hit_log"),
        source("pd_disaggregation_rules"),
    ]
    adaptation = _hicache_adaptation(ctx)
    base = ctx.parallel
    workload = ctx.workload
    prefix = workload.shared_prefix
    problems: list[str] = []
    if base.dcp > 1 and not ctx.arch.attention.is_mla:
        problems.append("HiCache + DCP 只支持 MLA 模型（MHA host pool 没有 DCP 索引转换）")
    if problems:
        return FeatureAdvice(
            key,
            title,
            "unsupported",
            problems[0],
            [],
            adaptation,
            [],
            {},
            {},
            {},
            problems,
            references,
            bool(adaptation),
        )
    plan = ctx.memory(base)
    budget = hicache_host_budget(ctx, base)
    device_bytes = plan.kv_pool_bytes_per_gpu
    auto_ratio = (
        min(HICACHE_DEFAULT_RATIO, budget["budget_bytes_per_rank"] / device_bytes)
        if device_bytes > 0
        else 0.0
    )
    common_tuning = [
        "PD 分离时 decode 实例默认关闭 radix cache：HiCache 放在 prefill 实例上",
        "上线后看 prefill 日志的 `#cached-token` 与 cache hit rate，核对实际命中率",
    ]
    if prefix is None:
        return FeatureAdvice(
            key=key,
            title=title,
            status="not_recommended",
            summary="负载没有可复用的前缀（random 数据集每条 prompt 互不相同），HiCache 只会增加写回流量",
            reasons=[
                "HiCache 把 radix cache 扩展到主机内存，收益全部来自前缀命中",
                "多轮对话 / agent（会话间隔长、历史被挤出显存）才是 HiCache 的主场：用 generated-shared-prefix 预设评估",
            ],
            adaptation=adaptation,
            tuning=common_tuning,
            estimate={"auto_ratio": round(auto_ratio, 3)},
            references=references,
            requires_adaptation=bool(adaptation),
        )
    if not plan.feasible or plan.kv_bytes_per_token_per_gpu <= 0:
        return FeatureAdvice(
            key,
            title,
            "not_recommended",
            "当前布局的 KV 池不可行",
            [],
            adaptation,
            [],
            {},
            {},
            {},
            [],
            references,
            bool(adaptation),
        )
    kv_bytes = plan.kv_bytes_per_token_per_gpu
    device_tokens = plan.max_total_tokens
    system = mean_length(prefix.system_prompt_len, workload.range_ratio)
    question = mean_length(prefix.question_len, workload.range_ratio)
    probabilities = prefix.group_probabilities()
    sizes = [system] * prefix.num_groups
    working_set = system * prefix.num_groups
    batch_total = ctx.operating_batch(base)
    batch_rank = math.ceil(batch_total / base.attn_dp)
    inflight = batch_rank * max(ctx.decode_context - system, 0.0)
    device_cache = max(device_tokens - inflight, 0.0)
    host_tokens = auto_ratio * device_tokens
    l2_cache = max(device_cache, host_tokens - inflight)
    pooled = base.attn_dp if base.attn_dp > 1 else 1

    def hit(capacity: float, routed: bool = False) -> float:
        return lru_hit_rate(probabilities, sizes, capacity * (pooled if routed else 1))

    h_dev, h_l2 = hit(device_cache), hit(l2_cache)
    prompt = max(int(round(system + question)), 1)
    t_hit = ctx.prefill_ms(base, prompt, prefix_len=int(system))
    t_miss = ctx.prefill_ms(base, prompt)
    link = ctx.hardware.host_link_bandwidth_gbps * 1e9 * ctx.hardware.efficiency.host_link
    load_ms = system * kv_bytes / link * 1000 if link > 0 else math.inf
    t_host = max(t_hit, load_ms) + load_ms / max(ctx.arch.num_layers, 1)
    ttft_without = h_dev * t_hit + (1 - h_dev) * t_miss
    ttft_with = h_dev * t_hit + (h_l2 - h_dev) * t_host + (1 - h_l2) * t_miss
    new_without = question + (1 - h_dev) * system
    new_with = question + (1 - h_l2) * system
    saving = 1 - new_with / new_without if new_without > 0 else 0.0
    gain = ratio(ttft_without, ttft_with)

    turns = prefix.num_turns
    tpot = ctx.decode_step_ms(base, batch_total)
    rate = workload.effective_request_rate
    if rate is None:
        rate = batch_total / max((ttft_with + workload.mean_output_len * tpot) / 1000, 1e-9)
    rate_rank = rate / base.attn_dp
    backup_bytes_s = rate_rank * (new_with + workload.mean_output_len) * kv_bytes
    load_bytes_s = rate_rank * (h_l2 - h_dev) * system * kv_bytes
    link_util = max(backup_bytes_s, load_bytes_s) / link if link > 0 else math.inf

    needed_ratio = (working_set + inflight) / device_tokens if device_tokens else math.inf
    max_ratio = budget["budget_bytes_per_rank"] / device_bytes if device_bytes > 0 else 0.0

    reasons = [
        f"共享前缀工作集 {prefix.num_groups} 组 × {system:.0f} = {working_set:,.0f} token；显存 KV 池每 rank "
        f"{device_tokens:,} token，扣掉在途请求（≈{batch_rank} 条 × 私有 {max(ctx.decode_context - system, 0):.0f} token）"
        f"后可缓存 {device_cache:,.0f} token",
        f"主机内存 {ctx.host_memory_gb:g} GiB、每机 {budget['ranks_per_host']} rank：auto-sizing 每 rank 最多 "
        f"{budget['budget_bytes_per_rank'] / GIB:.1f} GiB，ratio={auto_ratio:.2f}（默认 2.0，受主机内存约束会自动下调）"
        f"→ host 池 {host_tokens:,.0f} token",
        f"LRU 命中率（{prefix.distribution} 分布）：仅显存 {h_dev:.0%} → 加 HiCache {h_l2:.0%}；每请求 prefill "
        f"{new_without:.0f}→{new_with:.0f} token（{saving:.0%}）",
        f"命中 host 的前缀以 {ctx.hardware.host_link_bandwidth_gbps:g} GB/s 主机链路逐层加载 {load_ms:.1f} ms，"
        f"重算需要 {t_miss - t_hit:.1f} ms：平均 TTFT（无排队）{ttft_without:.0f}→{ttft_with:.0f} ms",
    ]
    if ctx.hardware.peak_flops("bf16") < 300e12:
        reasons.append(
            f"{ctx.hardware.name} 算力低（BF16 {ctx.hardware.peak_flops('bf16') / 1e12:.0f} TFLOPS）而主机链路不慢："
            "从主机加载比重算便宜得多，前缀复用型负载上 HiCache 收益比 H100 更大"
        )
    if turns > 1:
        reasons.append(
            "多轮对话的历史在下一轮立即复用（bench 的下一轮在上一轮结束时发出），基本命中显存；"
            "真实流量里会话间隔越长，历史越依赖 host 层"
        )
    if pooled > 1:
        routed = hit(l2_cache, routed=True)
        reasons.append(
            f"DP attention 下每个 DP rank 有独立的 radix cache：轮询分发时每个 rank 都要缓存全部前缀；"
            f"用 router `--dp-aware --policy cache_aware` 按前缀固定 DP rank，命中率 {h_l2:.0%}→{routed:.0%}，"
            f"所需 ratio 降到 ≈{(working_set / pooled + inflight) / device_tokens:.1f}"
        )
    tuning = [
        "`--enable-hierarchical-cache`；默认 `--hicache-write-policy write_through`（首次插入即写回 host），"
        f"一次性 prompt 多时用 `write_through_selective`（命中 {HICACHE_SELECTIVE_WRITE_THRESHOLD} 次才写回）",
        f"主机链路写回+加载占用约 {link_util:.0%}（单方向）"
        + ("：已接近上限，改用 write_through_selective 或降低并发" if link_util > 0.8 else ""),
        "显式 `--hicache-ratio` / `--hicache-size`（GB/rank）会关闭 auto-sizing；"
        "`--hicache-host-memory-fraction`（默认 0.8）控制多引擎共享主机内存",
        "需要更大容量时接 L3：`--hicache-storage-backend`（file / mooncake / hf3fs / nixl …），"
        "mooncake 存储会把 layer_first 布局改成 page_first",
        *common_tuning,
    ]
    flags: dict[str, Any] = {"--enable-hierarchical-cache": True}
    if needed_ratio > auto_ratio * 1.05 and h_l2 < 0.99:
        if needed_ratio <= max_ratio:
            value = math.ceil(needed_ratio * 2) / 2
            flags["--hicache-ratio"] = value
            tuning.insert(
                1,
                f"工作集需要 ratio≈{needed_ratio:.1f}：设 `--hicache-ratio {value:g}`（主机内存允许到 {max_ratio:.1f}）",
            )
        else:
            tuning.insert(
                1,
                f"工作集需要 ratio≈{needed_ratio:.1f}，但主机内存只够 {max_ratio:.1f}：加主机内存、用 L3 存储或按前缀路由",
            )
    if adaptation:
        flags["--hicache-mem-layout"] = "layer_first"
        flags["--hicache-io-backend"] = "direct"
    else:
        tuning.append(
            "CUDA 上保持默认 `--hicache-io-backend kernel --hicache-mem-layout page_first`；"
            "direct 后端配 page_first 会被改写成 page_first_direct"
        )
    if base.dcp > 1:
        tuning.append(
            "DCP>1：只支持 L1/L2（不能配 --hicache-storage-backend），投机解码只能是 DSPARK 或不用"
        )
    if h_l2 - h_dev < 0.02:
        status = "not_recommended"
        summary = f"显存 radix cache 已能装下工作集（命中率 {h_dev:.0%}），HiCache 无增益"
    elif gain >= RECOMMEND_GAIN or saving >= 0.2:
        status = "recommend"
        summary = f"前缀命中率 {h_dev:.0%}→{h_l2:.0%}，平均 TTFT {ttft_without:.0f}→{ttft_with:.0f} ms，prefill 量 -{saving:.0%}"
    elif gain >= OPTIONAL_GAIN or saving >= 0.05:
        status = "optional"
        summary = (
            f"前缀命中率 {h_dev:.0%}→{h_l2:.0%}，平均 TTFT {ttft_without:.0f}→{ttft_with:.0f} ms"
        )
    else:
        status = "not_recommended"
        summary = f"命中率只提高 {h_l2 - h_dev:.0%}，收益不明显"
    return FeatureAdvice(
        key=key,
        title=title,
        status=status,
        summary=summary,
        reasons=reasons,
        adaptation=adaptation,
        tuning=tuning,
        flags=flags,
        estimate={
            "working_set_tokens": round(working_set),
            "device_cache_tokens": round(device_cache),
            "host_pool_tokens": round(host_tokens),
            "auto_ratio": round(auto_ratio, 3),
            "max_ratio": round(max_ratio, 3),
            "needed_ratio": round(needed_ratio, 3),
            "hit_rate": {"device_only": round(h_dev, 4), "hicache": round(h_l2, 4)},
            "prefill_tokens_per_request": {
                "device_only": round(new_without, 1),
                "hicache": round(new_with, 1),
            },
            "mean_ttft_ms": {"device_only": round(ttft_without, 1), "hicache": round(ttft_with, 1)},
            "host_load_ms": round(load_ms, 2),
            "host_link_utilization": round(link_util, 3),
        },
        references=references,
        requires_adaptation=bool(adaptation),
    )


# ---------------------------------------------------------------------- expert parallel / DeepEP
def _deepep_adaptation(ctx: FeatureContext) -> list[str]:
    if ctx.has_a2a("deepep"):
        return []
    return [
        "DeepEP 是 CUDA 库（token_dispatcher/deepep.py 里 `from deep_ep import Buffer, Config`）。要在这里用 a2a："
        "提供与 deep_ep 同 API 的厂商实现（normal 的 dispatch/combine、low_latency_dispatch/low_latency_combine、"
        "两阶段 dispatch_a/dispatch_b），或在通用插件里用 HookRegistry 替换 "
        "`sglang.srt.layers.moe.fused_moe_triton.layer.create_moe_dispatcher` 返回自研 dispatcher",
        "不做适配时 EP 只能走 `--moe-a2a-backend none`：token all-gather 后各 rank 算本地专家，再 all-reduce / "
        "reduce-scatter，只依赖标准集合通信",
    ]


def advise_expert_parallel(ctx: FeatureContext) -> FeatureAdvice:
    key, title = "deepep", "专家并行 EP / DeepEP（--ep-size、--moe-a2a-backend deepep）"
    references = [
        source("a2a_ep_size"),
        source("a2a_dispatcher_factory"),
        source("deepep_buffer_import"),
        source("deepep_auto_mode"),
        source("deepep_normal_graph"),
        source("deepep_low_latency_tokens"),
        source("deepep_low_latency_cap"),
        source("deepep_sm_warning"),
        source("deepep_v2_overlap"),
    ]
    moe = ctx.arch.moe
    if moe is None:
        return FeatureAdvice(
            key,
            title,
            "unsupported",
            f"{ctx.arch.name} 是稠密模型，没有 MoE 层",
            references=references,
        )
    base = ctx.parallel
    adaptation = _deepep_adaptation(ctx)
    if base.tp == 1:
        return FeatureAdvice(
            key,
            title,
            "not_recommended",
            f"当前布局 {base.describe()} 只有 1 张卡，EP 需要多卡",
            references=references,
            adaptation=adaptation,
            requires_adaptation=bool(adaptation),
        )
    candidates: list[tuple[str, ParallelConfig]] = []
    rejected: list[str] = []

    def add(role: str, layout: ParallelConfig) -> None:
        if layout == base or any(layout == other for _, other in candidates):
            return
        errors = layout.validate(ctx.arch)
        if errors:
            rejected.append(f"{layout.describe()}：{errors[0]}")
            return
        candidates.append((role, layout))

    tp = base.tp
    add("ep_allgather", base.with_(ep=tp, moe_a2a="none"))
    add("ep_allgather_dpa", base.with_(attn_dp=tp, attn_cp=1, dcp=1, ep=tp, moe_a2a="none"))
    if ctx.has_a2a("deepep"):
        add("deepep", base.with_(ep=tp, moe_a2a="deepep"))
        add("deepep_dpa", base.with_(attn_dp=tp, attn_cp=1, dcp=1, ep=tp, moe_a2a="deepep"))
    operating = ctx.operating_batch(base)
    slo = ctx.workload.slo
    rows: list[dict[str, Any]] = []
    measured: list[tuple[str, ParallelConfig, dict[str, float]]] = []
    for role, layout in [("baseline", base), *candidates]:
        point = ctx.decode_point(layout)
        if point is None:
            rows.append({"role": role, "layout": layout.describe(), "feasible": False})
            continue
        values = {
            "decode_tok_s_per_gpu": point.tokens_per_s_per_gpu,
            "decode_batch_total": point.total_batch,
            "batch_per_rank": point.batch_per_rank,
            "tpot_ms": ctx.decode_step_ms(layout, operating),
            "ttft_ms": ctx.prefill_ms(layout, ctx.long_prompt),
        }
        rows.append(
            {
                "role": role,
                "layout": layout.describe(),
                **{name: round(value, 2) for name, value in values.items()},
            }
        )
        measured.append((role, layout, values))
    if not measured or measured[0][0] != "baseline":
        return FeatureAdvice(
            key,
            title,
            "not_recommended",
            "基线布局不可行，无法比较 EP 布局",
            rejected,
            adaptation,
            estimate={"layouts": rows},
            references=references,
            requires_adaptation=bool(adaptation),
        )

    def meets(values: dict[str, float]) -> bool:
        return (slo.ttft_ms is None or values["ttft_ms"] <= slo.ttft_ms) and (
            slo.tpot_ms is None or values["tpot_ms"] <= slo.tpot_ms
        )

    baseline_values = measured[0][2]
    best_role, best_layout, best_values = max(
        measured, key=lambda item: (meets(item[2]), item[2]["decode_tok_s_per_gpu"])
    )
    gain = ratio(best_values["decode_tok_s_per_gpu"], baseline_values["decode_tok_s_per_gpu"])
    reasons = [
        "H20 类芯片 decode 是带宽瓶颈：EP 与 TP-MoE 每步读的专家权重总量基本相同，差别在通信（a2a 只发 top-k 份 "
        "token，TP-MoE 要 all-reduce 全部 hidden）和 GEMM 形状（EP 下每个专家是完整矩阵）",
        f"负载工作点（总 batch≈{operating}）TPOT {baseline_values['tpot_ms']:.1f}→{best_values['tpot_ms']:.1f} ms，"
        f"最长 prompt TTFT {baseline_values['ttft_ms'] / 1000:.2f}→{best_values['ttft_ms'] / 1000:.2f} s",
        *rejected,
    ]
    if best_role == "baseline":
        status = "not_recommended"
        summary = f"当前布局 {base.describe()} 在这些 EP 候选里已经最好"
    else:
        status = (
            "recommend"
            if gain >= RECOMMEND_GAIN
            else "optional"
            if gain >= OPTIONAL_GAIN
            else "not_recommended"
        )
        summary = (
            f"{best_layout.describe()}：KV-bound decode 吞吐 {pct(gain)}/GPU"
            f"（{baseline_values['decode_tok_s_per_gpu']:.0f}→{best_values['decode_tok_s_per_gpu']:.0f} tok/s/GPU）"
        )
    flags: dict[str, Any] = {}
    env: dict[str, str] = {}
    tuning: list[str] = []
    if best_role != "baseline":
        flags["--ep-size"] = best_layout.ep
        if best_layout.attn_dp > 1:
            flags["--attn-dp-size"] = best_layout.attn_dp
    if best_layout.uses_a2a:
        flags["--moe-a2a-backend"] = "deepep"
        tokens = math.ceil(best_values["batch_per_rank"] / best_layout.attn_tp)
        tuning += [
            "a2a 后端会强制 ep_size = tp_size；`--deepep-mode auto`（默认）对含 EXTEND 的批次用 normal、纯 decode 用 "
            "low_latency。`normal` 会同时关闭 decode 与 prefill CUDA graph，混部时不要用",
            "PD 分离：prefill 实例 `--deepep-mode normal`，decode 实例 `--deepep-mode low_latency`",
            f"low_latency 每 rank 每步最多发 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK 个 token（默认 "
            f"{DEEPEP_LOW_LATENCY_TOKENS_DEFAULT}，SGLang 断言 ≤ {DEEPEP_LOW_LATENCY_TOKENS_LIMIT}；决定 RDMA 缓冲大小）：本布局"
            f"每 rank decode 约 {tokens} token（batch/attn_tp），投机解码还要乘以 draft token 数",
            "prefill 走 normal 模式且没开 TBO 时，DeepEP 占的 SM 少于总数一半会告警 `Only use N SMs for DeepEP "
            "communication`：用 `--deepep-config` 提高 normal_dispatch/normal_combine 的 num_sms（配置用 DeepEP 仓库 "
            "tests/ 下的调优脚本生成）",
            "`--moe-a2a-backend deepep_v2` 不支持 TBO/SBO，且只支持 deep_gemm MoE runner",
        ]
        if tokens > DEEPEP_LOW_LATENCY_TOKENS_DEFAULT:
            value = min(2 ** math.ceil(math.log2(tokens)), DEEPEP_LOW_LATENCY_TOKENS_LIMIT)
            env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = str(value)
            if tokens > DEEPEP_LOW_LATENCY_TOKENS_LIMIT:
                tuning.append(
                    f"每 rank {tokens} token 超过 {DEEPEP_LOW_LATENCY_TOKENS_LIMIT} 上限：用 `--cuda-graph-max-bs` / "
                    "`--max-running-requests` 压低每 rank batch"
                )
    elif best_layout.ep > 1:
        tuning.append(
            "`--ep-size` + `--moe-a2a-backend none`：每 rank 只放 1/ep 的专家，token 仍是 all-gather + all-reduce；"
            "开 EPLB 时 dispatch 算法自动为 dynamic（各 rank 必须选同一个副本）"
        )
    return FeatureAdvice(
        key=key,
        title=title,
        status=status,
        summary=summary,
        reasons=reasons,
        adaptation=adaptation,
        tuning=tuning,
        flags=flags,
        env=env,
        estimate={
            "layouts": rows,
            "best_layout": best_layout.describe(),
            "decode_gain_per_gpu": round(gain, 3),
            "operating_batch": operating,
        },
        references=references,
        requires_adaptation=bool(adaptation),
        layout=best_layout if best_role != "baseline" else None,
    )


# ---------------------------------------------------------------------- TBO / SBO
def deep_gemm_limited(timing: OpTiming) -> bool:
    """GEMMs that run on DeepGEMM (FP8 block GEMMs and grouped experts on SM90+)."""
    return timing.op.kind in {"gemm", "moe"} and timing.op.compute_dtype == "fp8"


def _split(
    estimate: StepEstimate, overlappable: frozenset[str], reserved_sm_share: float = 0.0
) -> tuple[float, float]:
    """``(compute_us, comm_us)``; DeepGEMM kernels lose ``reserved_sm_share`` of the SMs."""
    compute = comm = 0.0
    for timing in estimate.timings:
        if timing.op.name in overlappable:
            comm += timing.total_us
            continue
        time = timing.time_us
        if reserved_sm_share and deep_gemm_limited(timing):
            roofline = max(timing.compute_us, timing.memory_us)
            time += max(0.0, timing.compute_us / (1 - reserved_sm_share) - roofline)
        compute += time * timing.op.layers
    return compute * estimate.scale, comm * estimate.scale


def tbo_step_us(
    ctx: FeatureContext,
    layout: ParallelConfig,
    full: BatchShape,
    halves: tuple[BatchShape, BatchShape],
    overlappable: frozenset[str],
    *,
    reserved_sm_share: float = 0.0,
) -> tuple[float, float, float]:
    """``(serial_us, overlapped_us, hidden_comm_us)`` of one step with two-batch overlap.

    The two micro-batches alternate on the compute stream while the other one's dispatch/combine
    is in flight, so a step costs ``max(sum compute, sum comm)``; each micro-batch re-reads the
    weights, which the half-size estimates already include. Kernels other than DeepGEMM share
    SMs with DeepEP only while they overlap a transfer, which is not modelled.
    """
    model = ctx.cost(layout)
    serial = model.estimate(full)
    compute, comm = 0.0, 0.0
    for shape in halves:
        part_compute, part_comm = _split(model.estimate(shape), overlappable, reserved_sm_share)
        compute += part_compute
        comm += part_comm
    overlapped = max(compute, comm) + serial.step_overhead_us
    _, serial_comm = _split(serial, overlappable)
    return serial.total_us, overlapped, serial_comm


def advise_tbo(ctx: FeatureContext, layout: ParallelConfig | None = None) -> FeatureAdvice:
    key, title = "tbo", "Two-Batch Overlap / Single-Batch Overlap（--enable-two-batch-overlap）"
    references = [
        source("tbo_strategies"),
        source("tbo_ep_dispatch"),
        source("tbo_requires_dpa"),
        source("tbo_deepseek_v4"),
        source("tbo_prefill_sms"),
        source("tbo_two_chunk"),
        source("tbo_decode_graph"),
        source("breakable_prefill_graph"),
        source("deepep_v2_overlap"),
        source("sbo_sm90"),
    ]
    arch_name = ctx.arch.primary_architecture
    layout = layout or ctx.parallel
    sbo_note = (
        "SBO（--enable-single-batch-overlap）在 SM90 上直接报错（当前 sgl-deep-gemm wheel 不支持），H20 不可用"
        if ctx.is_cuda and ctx.sm_major == SBO_UNSUPPORTED_SM_MAJOR
        else "SBO 依赖 DeepGEMM/DeepEP 的 CUDA 实现，只在 CUDA（非 SM90）上可用"
        if not ctx.is_cuda
        else "SBO 与 TBO 二选一；它在单个 batch 内重叠 combine 与下一段计算"
    )
    if ctx.arch.moe is None:
        return FeatureAdvice(
            key,
            title,
            "unsupported",
            "TBO 只为 MoE decoder 层定义了算子拆分策略，稠密模型不适用",
            [sbo_note],
            references=references,
        )
    if arch_name not in ctx.knowledge.tbo_architectures:
        return FeatureAdvice(
            key,
            title,
            "unsupported",
            f"{arch_name} 没有 TBO 算子策略（OperationsStrategy.init_new_tbo 只支持 "
            f"{', '.join(ctx.knowledge.tbo_architectures)}）",
            [sbo_note],
            references=references,
        )
    if layout.tp == 1:
        return FeatureAdvice(
            key,
            title,
            "not_recommended",
            f"当前布局 {layout.describe()} 只有 1 张卡，没有 EP/DP 通信可隐藏",
            [sbo_note],
            references=references,
            layout=layout,
        )
    v4 = arch_name in TBO_PREFILL_ONLY_ARCHITECTURES
    given = layout
    switched = False
    if not layout.uses_a2a and not (v4 and layout.attn_dp > 1):
        variant = layout.with_(ep=layout.tp, moe_a2a="deepep")
        if (
            ctx.has_a2a("deepep")
            and not variant.validate(ctx.arch)
            and ctx.memory(variant).feasible
        ):
            layout, switched = variant, True
    if layout.uses_a2a:
        overlappable = A2A_OPS
        what = "EP dispatch/combine（DeepEP 异步通信）"
    elif v4 and layout.attn_dp > 1:
        overlappable = DP_MOE_COMM_OPS
        what = "DP all_gatherv / reduce_scatterv（DeepSeek-V4 非 EP 路径）"
    else:
        problem = (
            "moe_a2a_backend=none 时 TBO 必须开 DP attention（check_two_batch_overlap 报错）"
            if layout.attn_dp == 1
            else "TBO 只重叠 EP dispatch/combine，moe_a2a_backend=none 时没有可隐藏的通信"
        )
        reasons = [
            "TBO 需要 a2a 后端（DeepEP 的异步 dispatch/combine）提供可隐藏的通信",
            "设备没有声明 DeepEP：先完成 EP/DeepEP 建议里的 dispatcher 适配，再评估 TBO"
            if not ctx.has_a2a("deepep")
            else f"DeepEP 变体 {layout.with_(ep=layout.tp, moe_a2a='deepep').describe()} 在当前显存规划下不可行",
            sbo_note,
        ]
        return FeatureAdvice(
            key,
            title,
            "not_recommended",
            f"布局 {layout.describe()}：{problem}",
            reasons,
            references=references,
            problems=[problem] if layout.attn_dp == 1 else [],
            layout=layout,
        )
    total_sms = ctx.hardware.num_compute_units
    deep_gemm = ctx.is_cuda and (ctx.sm_major or 0) >= 9
    reserved = (
        min(DEEPEP_ASSUMED_NUM_SMS / total_sms, 0.5)
        if layout.uses_a2a and total_sms and deep_gemm
        else 0.0
    )
    chunk = ctx.per_rank_chunk(layout)
    half = max(chunk // 2, 1)
    prefill_serial, prefill_tbo, prefill_comm = tbo_step_us(
        ctx,
        layout,
        BatchShape.prefill(1, chunk, 0),
        (BatchShape.prefill(1, half, 0), BatchShape.prefill(1, half, half)),
        overlappable,
        reserved_sm_share=reserved,
    )
    prefill_gain = ratio(prefill_serial, prefill_tbo)
    rows = {
        "prefill": {
            "chunk_tokens": chunk,
            "serial_ms": round(prefill_serial / 1000, 3),
            "tbo_ms": round(prefill_tbo / 1000, 3),
            "hidden_comm_ms": round(prefill_comm / 1000, 3),
            "gain": round(prefill_gain, 3),
        }
    }
    decode_gain: float | None = None
    batch_rank = math.ceil(ctx.operating_batch(layout) / layout.attn_dp)
    if not v4 and batch_rank >= 2:
        context = ctx.decode_context
        first = batch_rank // 2
        decode_serial, decode_tbo, decode_comm = tbo_step_us(
            ctx,
            layout,
            BatchShape.decode(batch_rank, context),
            (BatchShape.decode(first, context), BatchShape.decode(batch_rank - first, context)),
            overlappable,
        )
        decode_gain = ratio(decode_serial, decode_tbo)
        rows["decode"] = {
            "batch_per_rank": batch_rank,
            "serial_ms": round(decode_serial / 1000, 3),
            "tbo_ms": round(decode_tbo / 1000, 3),
            "hidden_comm_ms": round(decode_comm / 1000, 3),
            "gain": round(decode_gain, 3),
        }
    reasons = [
        f"TBO 把批次拆成两个 micro-batch，一个算的时候另一个做 {what}；代价是每个 micro-batch 各读一遍权重，"
        "decode（带宽瓶颈）时这部分会翻倍",
        f"prefill（每 rank chunk {chunk}）：{prefill_serial / 1000:.1f}→{prefill_tbo / 1000:.1f} ms（{pct(prefill_gain)}），"
        f"可隐藏通信 {prefill_comm / 1000:.2f} ms"
        + (
            f"；prefill TBO 期间 DeepGEMM 只用 SM 总数减去 DeepEP 占用的部分（假设 "
            f"{DEEPEP_ASSUMED_NUM_SMS}/{total_sms}），FP8 GEMM 的计算时间相应变长"
            if reserved
            else ""
        ),
    ]
    if switched:
        reasons.insert(
            0,
            f"当前布局 {given.describe()} 没有 a2a 通信可隐藏，按 DeepEP 变体 {layout.describe()} 评估"
            "（显存/吞吐对比见 EP/DeepEP 建议）",
        )
    if decode_gain is not None:
        decode = rows["decode"]
        reasons.append(
            f"decode（每 rank {batch_rank}）：{decode['serial_ms']:.2f}→{decode['tbo_ms']:.2f} ms（{pct(decode_gain)}）"
        )
    elif v4:
        reasons.append("DeepSeek-V4 的 TBO 只在 EXTEND（prefill）阶段生效，decode 照常执行")
    else:
        reasons.append("负载工作点每 rank decode 不足 2 条请求，TBO 无法拆分 decode 批次")
    reasons.append(sbo_note)
    gains = [prefill_gain] + ([decode_gain] if decode_gain is not None else [])
    if all(g >= 1.05 for g in gains) and max(gains) >= 1.1:
        status = "recommend"
    elif prefill_gain >= 1.05 or (decode_gain is not None and decode_gain >= 1.05):
        status = "optional"
    else:
        status = "not_recommended"
    tuning = [
        "`--enable-two-batch-overlap`：decode 的 TBO 可以被 CUDA graph 捕获，但会关闭 breakable prefill CUDA graph",
        f"`--tbo-token-distribution-threshold`（默认 {TBO_TOKEN_DISTRIBUTION_THRESHOLD}）：按请求拆分后一半不到该比例时"
        "改为在一条序列内部拆成两块（two-chunk overlap），设 0 关闭",
        "deepep_v2 a2a 后端不支持 TBO/SBO",
    ]
    if decode_gain is not None and decode_gain < 1.0 <= prefill_gain:
        tuning.insert(1, "decode 变慢、prefill 变快：PD 分离时只在 prefill 实例开 TBO")
    summary = f"{layout.describe()}：prefill {pct(prefill_gain)}" + (
        f"，decode {pct(decode_gain)}" if decode_gain is not None else ""
    )
    flags: dict[str, Any] = {"--enable-two-batch-overlap": True}
    if switched:
        flags.update({"--ep-size": layout.ep, "--moe-a2a-backend": "deepep"})
    return FeatureAdvice(
        key=key,
        title=title,
        status=status,
        summary=summary,
        reasons=reasons,
        tuning=tuning,
        flags=flags,
        estimate={
            "layout": layout.describe(),
            "given_layout": given.describe(),
            **rows,
            "reserved_sm_share": round(reserved, 3),
        },
        references=references,
        layout=layout,
    )


# ---------------------------------------------------------------------- EPLB
def advise_eplb(ctx: FeatureContext, layout: ParallelConfig | None = None) -> FeatureAdvice:
    key, title = "eplb", "专家负载均衡 EPLB（--enable-eplb / --init-expert-location）"
    references = [
        source("eplb_rules"),
        source("eplb_divisibility"),
        source("expert_distribution"),
    ]
    moe = ctx.arch.moe
    if moe is None:
        return FeatureAdvice(key, title, "unsupported", "稠密模型没有专家", references=references)
    layout = layout or ctx.parallel
    if layout.ep <= 1:
        return FeatureAdvice(
            key,
            title,
            "not_recommended",
            f"EPLB 需要 ep_size > 1（当前 {layout.describe()}）",
            ["TP-MoE 下每个 rank 持有所有专家的 1/tp，不存在专家放置不均的问题"],
            references=references,
            layout=layout,
        )
    params = count_params(ctx.arch)
    expert_bytes = params.single_expert * weight_precision(ctx.arch, ctx.weight_dtype).experts
    redundant = layout.ep
    extra = redundant / layout.ep * expert_bytes * ctx.arch.num_moe_layers / layout.pp
    plan = ctx.memory(layout)
    lost = extra / plan.kv_bytes_per_token_per_gpu if plan.kv_bytes_per_token_per_gpu else 0.0
    share = lost / plan.max_total_tokens if plan.max_total_tokens else 0.0
    dispatch = "static" if layout.uses_a2a else "dynamic"
    reasons = [
        "最忙的 EP rank 决定每层 MoE 的耗时（prefill 的 grouped GEMM、a2a 的接收量）；decode 下各 rank 读的专家权重"
        "基本不变，收益主要在 prefill 和大 batch",
        f"{moe.num_experts} 个专家 / ep={layout.ep}：每 rank {moe.num_experts // layout.ep} 个；冗余 {redundant} 个后每 rank "
        f"{(moe.num_experts + redundant) // layout.ep} 个，KV 少 {lost:,.0f} token",
        "热点专家分布要用真实流量测；合成随机数据测不出真实的不均衡",
    ]
    if share > EPLB_MAX_KV_LOSS:
        status = "not_recommended"
        summary = (
            f"每 rank 1 个冗余专家槽多占 {extra / 1e9:.2f} GB，"
            + (
                f"超过整个 KV 池（{plan.kv_pool_bytes_per_gpu / 1e9:.2f} GB），放不下"
                if share >= 1
                else f"会吃掉 {share:.0%} 的 KV 池"
            )
            + "；先用 --init-expert-location 做无冗余的静态放置"
        )
        reasons.insert(
            0,
            f"KV 池本来就紧（每 rank {plan.max_total_tokens:,} token），冗余专家的收益抵不过并发下降",
        )
    else:
        status = "optional"
        summary = f"先录专家分布再决定；每 rank 加 1 个冗余专家槽约多 {extra / 1e9:.2f} GB 权重（KV -{share:.1%}）"
    return FeatureAdvice(
        key=key,
        title=title,
        status=status,
        summary=summary,
        reasons=reasons,
        tuning=[
            "1) 录分布：`--expert-distribution-recorder-mode stat`，压测时依次调用 `/start_expert_distribution_record`、"
            "`/stop_expert_distribution_record`、`/dump_expert_distribution_record`（落盘到 "
            "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR，默认 /tmp，文件名 expert_distribution_recorder_*.pt）",
            f"2) 静态放置：`--init-expert-location <dump.pt> --ep-num-redundant-experts {redundant}`"
            "（专家数 + 冗余数必须能被 ep_size 整除）",
            f"3) 在线重排：`--enable-eplb --ep-num-redundant-experts {redundant}`，每 `--eplb-rebalance-num-iterations`"
            f"（默认 1000）步重排；recorder 自动设为 stat，冗余副本选择算法自动为 {dispatch}",
        ],
        flags={"--ep-num-redundant-experts": redundant} if status == "optional" else {},
        estimate={
            "kv_loss_share": round(share, 4),
            "redundant_experts": redundant,
            "extra_weight_gb_per_gpu": round(extra / 1e9, 3),
            "kv_tokens_lost_per_gpu": round(lost),
            "dispatch_algorithm": dispatch,
        },
        references=references,
        layout=layout,
    )
