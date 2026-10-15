"""Speculative decoding: MTP/NextN and EAGLE/EAGLE3 drafts, NGRAM and DFLASH on one instance.

A chain draft of ``s`` steps costs ``s`` draft forwards, one target verify over ``s + 1`` tokens
per request and one draft extend; it emits the expected accept length per verify. On a chip
whose ridge point is as low as the H20's, verify turns compute-bound at small batches, so the
advisor reports the batch where speculation stops paying off next to the gain at the workload's
operating point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sglang_maestro.features.base import FeatureAdvice, FeatureContext, ratio
from sglang_maestro.knowledge.facts import (
    DEEPEP_LOW_LATENCY_TOKENS_DEFAULT,
    DEEPEP_LOW_LATENCY_TOKENS_LIMIT,
    DEEPSEEK_V3_MTP_DEFAULT_PARAMS,
    MTP_DRAFT_FROM_TARGET_ARCHITECTURES,
    NGRAM_DEFAULT_DRAFT_TOKENS,
    NGRAM_DEVICES,
    PAGED_TREE_SPECULATIVE_BACKENDS,
    SPECULATIVE_DEFAULT_MAX_RUNNING_REQUESTS,
    SPECULATIVE_WIDE_TREE_ARCHITECTURES,
    SPECULATIVE_WIDE_TREE_PARAMS,
    source,
)
from sglang_maestro.model.ops import BatchShape
from sglang_maestro.model.parallel import ParallelConfig

BATCH_GRID = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256)
CHAIN_STEPS = (1, 2, 3, 4, 5)
RECOMMEND_SPEEDUP = 1.2
OPTIONAL_SPEEDUP = 1.05


@dataclass(frozen=True)
class SpecPoint:
    """One decode iteration with and without speculation at ``batch_per_rank`` requests."""

    batch_per_rank: int
    num_steps: int
    topk: int
    draft_tokens: int
    accept_length: float
    decode_ms: float
    draft_ms: float
    verify_ms: float
    extend_ms: float

    @property
    def iteration_ms(self) -> float:
        return self.draft_ms + self.verify_ms + self.extend_ms

    @property
    def spec_tpot_ms(self) -> float:
        return self.iteration_ms / self.accept_length

    @property
    def speedup(self) -> float:
        return ratio(self.decode_ms, self.spec_tpot_ms)

    @property
    def break_even_accept_length(self) -> float:
        """Accept length at which a speculative iteration costs the same per token as decode."""
        return self.iteration_ms / self.decode_ms if self.decode_ms > 0 else math.inf

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_per_rank": self.batch_per_rank,
            "num_steps": self.num_steps,
            "topk": self.topk,
            "draft_tokens": self.draft_tokens,
            "accept_length": round(self.accept_length, 3),
            "decode_tpot_ms": round(self.decode_ms, 3),
            "spec_tpot_ms": round(self.spec_tpot_ms, 3),
            "draft_ms": round(self.draft_ms, 3),
            "verify_ms": round(self.verify_ms, 3),
            "extend_ms": round(self.extend_ms, 3),
            "speedup": round(self.speedup, 3),
            "break_even_accept_length": round(self.break_even_accept_length, 3),
        }


def spec_point(
    ctx: FeatureContext,
    layout: ParallelConfig,
    batch_per_rank: int,
    *,
    num_steps: int,
    topk: int,
    draft_tokens: int,
    context_len: float | None = None,
    accept_length: float | None = None,
    draft_layers: int = 1,
) -> SpecPoint:
    """Cost of one draft/verify/extend iteration; drafts run ``draft_layers`` decoder layers."""
    context = context_len or ctx.decode_context
    model = ctx.cost(layout)
    batch = max(int(batch_per_rank), 1)
    decode_ms = model.step_time_us(BatchShape.decode(batch, context)) / 1000
    draft_us = 0.0
    if draft_layers:
        per_step = model.draft_step_time_us(batch * max(topk, 1), context)
        draft_us = per_step * num_steps * draft_layers
    verify_ms = model.step_time_us(BatchShape.verify(batch, context, draft_tokens)) / 1000
    extend_us = (
        model.draft_step_time_us(batch * draft_tokens, context) * draft_layers
        if draft_layers
        else 0.0
    )
    if accept_length is None:
        accept_length = ctx.workload.accept_length(num_steps, topk)
    return SpecPoint(
        batch_per_rank=batch,
        num_steps=num_steps,
        topk=topk,
        draft_tokens=draft_tokens,
        accept_length=accept_length,
        decode_ms=decode_ms,
        draft_ms=draft_us / 1000,
        verify_ms=verify_ms,
        extend_ms=extend_us / 1000,
    )


def _auto_params(arch_name: str, algorithm: str) -> tuple[int, int, int]:
    if algorithm == "STANDALONE":
        return DEEPSEEK_V3_MTP_DEFAULT_PARAMS
    if arch_name in SPECULATIVE_WIDE_TREE_ARCHITECTURES:
        return SPECULATIVE_WIDE_TREE_PARAMS
    return DEEPSEEK_V3_MTP_DEFAULT_PARAMS


def _crossover_batch(points: list[SpecPoint]) -> int | None:
    """Smallest grid batch at which speculation is no faster than plain decode."""
    for point in points:
        if point.speedup < 1.0:
            return point.batch_per_rank
    return None


def _oot_adaptation(ctx: FeatureContext) -> list[str]:
    if not ctx.out_of_tree:
        return []
    lines = [
        "EAGLE/MTP 的辅助 kernel（sglang.kernels.ops.speculative.eagle 里的 fill_bonus_tokens、"
        "prepare_verify_commit_outputs、build_chain_tree 等）都是 @triton.jit，没有通用的 torch 回退："
        + (
            "设备支持 Triton，可以直接运行，需逐个验证数值"
            if ctx.hardware.software.supports_triton
            else "设备不支持 Triton，需要在通用插件里用 HookRegistry 把这些函数 REPLACE 成设备实现"
        ),
        "KV cache 位置分配有专门的平台钩子 `get_speculative_cache_locs_fn()`（默认 None → Triton 实现）",
        "插件的 attention backend 要支持 TARGET_VERIFY（每条请求多个 query token、按树/链的 causal mask）"
        "和 DRAFT_EXTEND 两种 forward mode，并能被 decode 图捕获（draft 多步各捕获一张图）",
        "spec_utils 里有 `@torch.compile(dynamic=True)` 的函数，只在 NPU/XPU 上关闭编译：其他设备会用平台编译后端，"
        "需要能编译或在插件里替换",
    ]
    return lines


def advise_speculative(ctx: FeatureContext, layout: ParallelConfig | None = None) -> FeatureAdvice:
    key, title = "speculative", "投机解码（MTP/NextN、EAGLE3、NGRAM、DFLASH）"
    references = [
        source("speculative_auto_params"),
        source("speculative_nextn_alias"),
        source("speculative_max_running"),
        source("speculative_eagle_rules"),
        source("speculative_ngram"),
        source("dflash_rules"),
        source("dflash_platform_gate"),
        source("speculative_cache_locs"),
        source("deepep_low_latency_tokens"),
        source("layernorm_sp_rules"),
        source("hicache_dcp"),
    ]
    layout = layout or ctx.parallel
    arch = ctx.arch
    arch_name = arch.primary_architecture
    mtp = arch_name in MTP_DRAFT_FROM_TARGET_ARCHITECTURES and arch.num_mtp_layers > 0
    algorithm = "EAGLE" if mtp else "EAGLE3"
    adaptation = _oot_adaptation(ctx)
    plan = ctx.memory(layout, include_mtp=mtp)
    base_plan = ctx.memory(layout)
    if not plan.feasible or not base_plan.feasible:
        return FeatureAdvice(
            key,
            title,
            "not_recommended",
            f"布局 {layout.describe()} 的 KV 池不可行",
            references=references,
            adaptation=adaptation,
            requires_adaptation=bool(adaptation),
            layout=layout,
        )
    draft_kv_share = arch.num_mtp_layers / arch.num_layers if mtp else 1 / arch.num_layers
    spec_kv_tokens = int(plan.max_total_tokens / (1 + draft_kv_share))
    context = ctx.decode_context
    kv_cap = max(int(spec_kv_tokens // max(context, 1.0)), 1)
    base_cap = max(int(base_plan.max_total_tokens // max(context, 1.0)), 1)
    default_cap = max(SPECULATIVE_DEFAULT_MAX_RUNNING_REQUESTS // layout.attn_dp, 1)
    operating_total = ctx.operating_batch(layout)
    operating = max(1, min(math.ceil(operating_total / layout.attn_dp), kv_cap))

    auto_steps, auto_topk, auto_tokens = _auto_params(arch_name, algorithm)
    auto = spec_point(
        ctx, layout, operating, num_steps=auto_steps, topk=auto_topk, draft_tokens=auto_tokens
    )
    chains = [
        spec_point(ctx, layout, operating, num_steps=steps, topk=1, draft_tokens=steps + 1)
        for steps in CHAIN_STEPS
    ]
    best = max(chains, key=lambda point: point.speedup)
    grid = [
        spec_point(
            ctx, layout, batch, num_steps=best.num_steps, topk=1, draft_tokens=best.draft_tokens
        )
        for batch in BATCH_GRID
        if batch <= kv_cap
    ]
    crossover = _crossover_batch(grid)
    rate_mode = (
        ctx.workload.effective_request_rate is not None or ctx.workload.max_concurrency is not None
    )
    base_peak = min(base_cap, 256)
    base_peak_tps = (
        base_peak / ctx.cost(layout).step_time_us(BatchShape.decode(base_peak, context)) * 1e6
    )
    spec_peak_batch = min(kv_cap, 256)
    spec_peak = spec_point(
        ctx,
        layout,
        spec_peak_batch,
        num_steps=best.num_steps,
        topk=1,
        draft_tokens=best.draft_tokens,
    )
    spec_peak_tps = spec_peak_batch * spec_peak.accept_length / spec_peak.iteration_ms * 1000
    peak_ratio = ratio(spec_peak_tps, base_peak_tps)

    draft_desc = (
        f"{arch_name} 自带 {arch.num_mtp_layers} 层 MTP（NextN），`--speculative-algorithm NEXTN` 会解析成 EAGLE，"
        "draft 默认就是目标 checkpoint，不用传 `--speculative-draft-model-path`"
        if mtp
        else f"{arch_name} 没有内置 MTP 层：EAGLE3 需要与目标模型配套训练的 draft 模型（`--speculative-draft-model-path`），"
        "没有 draft 模型时只能用 NGRAM"
    )
    reasons = [
        draft_desc,
        f"接受率按负载假设 {ctx.workload.spec_accept_rate:g}/token（上线后用 Decode 日志的 `accept len` 替换）："
        f"链式 {best.num_steps} 步期望接受长度 {best.accept_length:.2f}",
        f"负载工作点每 rank {operating} 条请求：decode {best.decode_ms:.2f} ms/token → 投机 "
        f"{best.spec_tpot_ms:.2f} ms/token（{best.speedup:.2f}×；draft {best.draft_ms:.2f} + verify "
        f"{best.verify_ms:.2f} + extend {best.extend_ms:.2f} ms/轮）",
        f"盈亏平衡接受长度 {best.break_even_accept_length:.2f}：实测 accept len 低于它就比不开更慢",
    ]
    if crossover is not None:
        reasons.append(
            f"{ctx.hardware.name} 的算力/带宽比低（BF16 拐点 ≈ {ctx.hardware.ridge_point('bf16'):.0f} FLOP/B），"
            f"verify 的 {best.draft_tokens}× token 很快变成算力瓶颈：每 rank batch ≥ {crossover} 时投机解码不再有收益"
        )
    reasons.append(
        f"吞吐上限：不开投机 {base_peak_tps * layout.attn_dp / layout.num_gpus:.0f} tok/s/GPU（每 rank {base_peak} 并发）→ "
        f"开 {spec_peak_tps * layout.attn_dp / layout.num_gpus:.0f} tok/s/GPU（{peak_ratio:.2f}×）"
    )
    if mtp:
        extra_gb = (plan.weights_bytes_per_gpu - base_plan.weights_bytes_per_gpu) / 1e9
        reasons.append(
            f"MTP 层权重每卡 +{extra_gb:.2f} GB、draft KV 按 {draft_kv_share:.1%} 估算：KV 池 "
            f"{base_plan.max_total_tokens:,}→{spec_kv_tokens:,} token/rank"
        )
    else:
        reasons.append("EAGLE3 draft 权重（单层 + LM head）未计入显存估算")

    tuning = [
        f"未显式设置时 SGLang 会把 `--max-running-requests` 重置为 {SPECULATIVE_DEFAULT_MAX_RUNNING_REQUESTS}"
        f"（全局，按 attn_dp 均分：每 rank {default_cap}）；高并发时显式设置 `--max-running-requests` 与 "
        "`--cuda-graph-max-bs`",
        f"不设 `--speculative-num-steps` 时 SGLang 自动选 (steps, topk, draft_tokens) = "
        f"({auto_steps}, {auto_topk}, {auto_tokens})：本负载下 {auto.speedup:.2f}×",
        "topk=1 时 draft_tokens 会被改成 steps+1；topk>1 且 page_size>1 只支持 "
        f"{list(PAGED_TREE_SPECULATIVE_BACKENDS)}，trtllm_mha 只支持 topk=1",
        "开启后 mixed chunked prefill 会被关闭；STANDALONE 不支持 DP attention；LayerNorm SP 与 EAGLE 系互斥",
    ]
    if layout.dcp > 1:
        tuning.append("DCP>1 且同时开 HiCache 时，投机解码只能是 DSPARK 或不用")
    env: dict[str, str] = {}
    if layout.uses_a2a:
        tokens = math.ceil(min(operating, default_cap) / layout.attn_tp) * best.draft_tokens
        tuning.append(
            f"DeepEP low_latency 每 rank 每步发送 batch/attn_tp × draft_tokens ≈ {tokens} token（默认上限 "
            f"{DEEPEP_LOW_LATENCY_TOKENS_DEFAULT}，断言 ≤ {DEEPEP_LOW_LATENCY_TOKENS_LIMIT}）"
        )
        if tokens > DEEPEP_LOW_LATENCY_TOKENS_DEFAULT:
            value = min(2 ** math.ceil(math.log2(tokens)), DEEPEP_LOW_LATENCY_TOKENS_LIMIT)
            env["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = str(value)

    ngram_ok = ctx.profile.in_tree and ctx.device_type in NGRAM_DEVICES
    ngram = spec_point(
        ctx,
        layout,
        operating,
        num_steps=0,
        topk=1,
        draft_tokens=NGRAM_DEFAULT_DRAFT_TOKENS,
        accept_length=1.0,
        draft_layers=0,
    )
    alternatives = [
        (
            f"NGRAM（无 draft 模型，从已生成内容里查 n-gram）：默认 {NGRAM_DEFAULT_DRAFT_TOKENS} 个 draft token，"
            f"verify {ngram.verify_ms:.2f} ms vs decode {ngram.decode_ms:.2f} ms，平均接受长度需 ≥ "
            f"{ngram.break_even_accept_length:.2f} 才划算；只适合输出大量复用输入的负载（代码改写、RAG 摘抄）"
            if ngram_ok
            else f"NGRAM 只支持 {list(NGRAM_DEVICES)} 设备（_handle_ngram 直接报错）"
        ),
        "DFLASH：需要 draft 模型，不支持 DP attention（NPU 除外）、只支持 pp_size=1，steps/topk 固定为 1；"
        "树外平台还要让 `supports_speculative_algorithm('DFLASH')` 返回 True",
    ]
    reasons += alternatives

    if best.speedup >= RECOMMEND_SPEEDUP and (rate_mode or peak_ratio >= 1.0):
        status = "recommend"
    elif best.speedup >= OPTIONAL_SPEEDUP:
        status = "optional"
    else:
        status = "not_recommended"
    if not rate_mode and peak_ratio < 1.0:
        reasons.insert(0, "负载是吞吐模式（不限速）：投机解码降低峰值吞吐，只在需要低 TPOT 时才开")
    flags: dict[str, Any] = {
        "--speculative-algorithm": "NEXTN" if mtp else "EAGLE3",
        "--speculative-num-steps": best.num_steps,
        "--speculative-eagle-topk": 1,
        "--speculative-num-draft-tokens": best.draft_tokens,
    }
    if not mtp:
        flags["--speculative-draft-model-path"] = "<eagle3-draft-model>"
    target_running = max(operating, 1) * layout.attn_dp
    if target_running > SPECULATIVE_DEFAULT_MAX_RUNNING_REQUESTS:
        flags["--max-running-requests"] = target_running
    summary = (
        f"{'MTP' if mtp else 'EAGLE3'} 链式 {best.num_steps} 步：工作点（每 rank {operating}）TPOT "
        f"{best.decode_ms:.1f}→{best.spec_tpot_ms:.1f} ms（{best.speedup:.2f}×）"
        + (f"，每 rank batch ≥ {crossover} 后无收益" if crossover is not None else "")
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
            "algorithm": flags["--speculative-algorithm"],
            "operating_batch_per_rank": operating,
            "best": best.to_dict(),
            "auto_params": auto.to_dict(),
            "chains": [point.to_dict() for point in chains],
            "batch_grid": [point.to_dict() for point in grid],
            "crossover_batch_per_rank": crossover,
            "peak_tok_s_per_gpu": {
                "decode": round(base_peak_tps * layout.attn_dp / layout.num_gpus, 1),
                "speculative": round(spec_peak_tps * layout.attn_dp / layout.num_gpus, 1),
            },
            "kv_tokens_per_rank": {
                "decode": base_plan.max_total_tokens,
                "speculative": spec_kv_tokens,
            },
        },
        references=references,
        requires_adaptation=bool(adaptation),
        layout=layout,
    )
