"""Attention-side parallel features: DP attention, prefill CP, decode CP and LayerNorm SP."""

from __future__ import annotations

from typing import Any

from sglang_maestro.features.base import (
    DecodePoint,
    FeatureAdvice,
    FeatureContext,
    pct,
    ratio,
)
from sglang_maestro.knowledge.facts import (
    DCP_ATTENTION_BACKENDS,
    DP_ATTENTION_CONSERVATIVENESS_FACTOR,
    NO_ATTN_CP_MODEL_TYPES,
    PREFILL_CP_HIP_ARCHITECTURES,
    PREFILL_CP_INTERLEAVE_ONLY_ARCHITECTURES,
    PREFILL_CP_MIN_TOKENS_FACTOR,
    PREFILL_CP_STRATEGY_BACKENDS,
    PREFILL_CP_ZIGZAG_ONLY_ARCHITECTURES,
    source,
)
from sglang_maestro.model.ops import BatchShape
from sglang_maestro.model.parallel import ParallelConfig
from sglang_maestro.model.params import count_params, weight_precision

RECOMMEND_GAIN = 1.15
OPTIONAL_GAIN = 1.03


def _powers_up_to(limit: int) -> list[int]:
    values, value = [], 2
    while value <= limit:
        values.append(value)
        value *= 2
    return values


def _row(
    ctx: FeatureContext, point: DecodePoint | None, parallel: ParallelConfig
) -> dict[str, Any]:
    plan = ctx.memory(parallel)
    row: dict[str, Any] = {
        "layout": parallel.describe(),
        "weights_gb_per_gpu": round(plan.weights_bytes_per_gpu / 1e9, 2),
        "kv_tokens_total": plan.max_total_tokens * parallel.attn_dp,
        "feasible": plan.feasible,
    }
    if point is not None:
        row.update(
            {
                "decode_batch_total": point.total_batch,
                "decode_step_ms": round(point.step_ms, 2),
                "decode_tok_s_per_gpu": round(point.tokens_per_s_per_gpu, 1),
            }
        )
    return row


def _tp1_advice(key: str, title: str, ctx: FeatureContext, what: str) -> FeatureAdvice:
    return FeatureAdvice(
        key=key,
        title=title,
        status="not_recommended",
        summary=f"当前布局 {ctx.parallel.describe()} 只有 1 张卡，{what}需要 TP > 1",
        reasons=[
            f"{ctx.arch.name} 在 {ctx.hardware.name} 上单卡即可放下（精度路线的最小布局）；"
            "吞吐扩展优先用多副本 + router（见 `sglang-maestro deploy`）",
            "如果确实需要单实例跨卡（长上下文的 KV 容量或 TTFT），用 `--tp N` 重新评估本特性",
        ],
    )


# ---------------------------------------------------------------------- DP attention
def _lm_head_gather_ms(ctx: FeatureContext, parallel: ParallelConfig, batch: int) -> float:
    estimate = ctx.cost(parallel).estimate(BatchShape.decode(batch, ctx.decode_context))
    return sum(t.total_us for t in estimate.timings if t.op.name == "lm_head.allgather") / 1000


def advise_dp_attention(ctx: FeatureContext) -> FeatureAdvice:
    key, title = "dp_attention", "DP Attention（--attn-dp-size）"
    references = [
        source("dp_attention"),
        source("dp_attention_resolution"),
        source("dp_attention_deprecated"),
        source("dp_lm_head"),
        source("dp_attention_collectives"),
        source("load_balance_default"),
    ]
    base = ctx.parallel
    if base.tp == 1:
        advice = _tp1_advice(key, title, ctx, "DP attention ")
        advice.references = references
        return advice
    attention = ctx.arch.attention
    baseline = ctx.decode_point(base)
    rows = [{"role": "baseline", **_row(ctx, baseline, base)}]
    best: tuple[DecodePoint, ParallelConfig] | None = None
    for attn_dp in _powers_up_to(base.tp // base.attn_cp):
        candidate = base.with_(attn_dp=attn_dp)
        if candidate.validate(ctx.arch):
            continue
        point = ctx.decode_point(candidate)
        rows.append({"role": "candidate", **_row(ctx, point, candidate)})
        if point is not None and (
            best is None or point.tokens_per_s_per_gpu > best[0].tokens_per_s_per_gpu
        ):
            best = (point, candidate)

    why_kv = (
        "MLA 的 latent KV 在纯 TP 下每张卡都存一份完整副本，DP attention 让每个 DP rank 只存自己请求的 KV，"
        "总 KV 容量按 attn_dp 倍增"
        if attention.is_mla
        else (
            f"KV 头只有 {attention.num_kv_heads} 个：attn_tp={base.attn_tp} 时每张卡会复制 KV 头，"
            "DP attention 把复制换成按请求切分"
            if attention.num_kv_heads < base.attn_tp
            else f"KV 头（{attention.num_kv_heads}）已经按 attn_tp={base.attn_tp} 切开，DP attention 不会增加 KV 容量"
        )
    )
    reasons = [
        why_kv,
        "代价：attention 权重只在 attn_tp=tp/attn_dp 内切分，每卡权重变多、KV 池变小",
    ]
    adaptation: list[str] = []
    if ctx.out_of_tree:
        adaptation.append(
            f"DP attention 在 TP 组上做 all-gather / reduce-scatter（`_dp_gather_via_all_gather`、"
            f"`_dp_gather_via_all_reduce`）：厂商通信库（{ctx.hardware.software.collective_library}）需要实现 "
            "all_gather_into_tensor、reduce_scatter_tensor、all_reduce，并能在 decode 图捕获里使用"
        )
        adaptation.append(
            "各 DP rank 的 batch 会被 pad 到相同 token 数后再做集合通信，图捕获的 bs 列表要覆盖 pad 后的大小"
        )

    if baseline is None or best is None:
        status = "not_recommended"
        summary = "当前显存规划下没有可行的 DP attention 布局"
        if best is None:
            reasons.append(
                f"所有 attn_dp 候选的 KV 池都装不下一条请求（≈{ctx.decode_context:,.0f} token）："
                "attention 权重只在 attn_tp 内切分，每卡权重变多"
            )
        return FeatureAdvice(
            key,
            title,
            status,
            summary,
            reasons,
            adaptation,
            [],
            {},
            {},
            {"layouts": rows},
            [],
            references,
            bool(adaptation),
        )

    point, candidate = best
    gain = ratio(point.tokens_per_s_per_gpu, baseline.tokens_per_s_per_gpu)
    kv_gain = ratio(
        ctx.memory(candidate).max_total_tokens * candidate.attn_dp,
        ctx.memory(base).max_total_tokens * base.attn_dp,
    )
    operating = ctx.operating_batch(base)
    tpot_base = ctx.decode_step_ms(base, operating)
    tpot_cand = ctx.decode_step_ms(candidate, operating)
    if gain >= RECOMMEND_GAIN:
        status = "recommend"
    elif gain >= OPTIONAL_GAIN:
        status = "optional"
    else:
        status = "not_recommended"
    summary = (
        f"{candidate.describe()}：KV-bound decode 吞吐 {pct(gain)}/GPU，KV 容量 ×{kv_gain:.2f}；"
        f"负载工作点（总 batch≈{operating}）TPOT {tpot_base:.1f}→{tpot_cand:.1f} ms"
    )
    if tpot_cand > tpot_base * 1.1:
        reasons.append(
            f"低并发时 DP attention 反而更慢：每个 rank 读更多 attention 权重（TPOT {tpot_base:.1f}→{tpot_cand:.1f} ms），"
            "适合高并发/吞吐场景"
        )
    base_chunk = ctx.memory(base).chunked_prefill_size
    per_rank = max(base_chunk // candidate.attn_dp, 1)
    keep_chunk = base_chunk * candidate.attn_dp
    keep_plan = ctx.memory(candidate, chunked_prefill_size=keep_chunk)
    tuning = [
        f"`--attn-dp-size {candidate.attn_dp}`（`--enable-dp-attention --dp-size N` 已弃用，SGLang 会改写成 "
        "`--attn-dp-size N`；`--dp-size` 是多副本，与 `--attn-dp-size` 同时 >1 会直接报错）",
        f"chunked prefill 会被除以 attn_dp：{base_chunk} → 每 rank {per_rank} token。要保持每 rank {base_chunk}，"
        f"设 `--chunked-prefill-size {keep_chunk}`；激活预留按全局 chunk 计算，KV 池会变成每 rank "
        f"{keep_plan.max_total_tokens:,} token（当前 {ctx.memory(candidate).max_total_tokens:,}）",
        f"schedule_conservativeness 会乘以 {DP_ATTENTION_CONSERVATIVENESS_FACTOR}（新请求的 decode 预留更激进）："
        "日志频繁出现 `KV cache pool is full. Retract requests.` 时调大 `--schedule-conservativeness`（如 2–3）",
        "`--load-balance-method`：auto 解析为 round_robin；输入长度差异大时用 `total_tokens` 让各 DP rank 的 token 数更均衡",
    ]
    lm_gather = _lm_head_gather_ms(ctx, candidate, point.batch_per_rank)
    params = count_params(ctx.arch)
    lm_bytes = params.lm_head * weight_precision(ctx.arch, ctx.weight_dtype).unquantized
    extra_gb = lm_bytes * (1 / candidate.attn_tp - 1 / candidate.tp) / 1e9
    tuning.append(
        f"`--enable-dp-lm-head`：LM head 只在 attention-TP 组内做词表并行，省掉跨 DP 组的 all-gather"
        f"（当前每步 {lm_gather:.2f} ms），代价是每卡多 {extra_gb:.2f} GB LM head 权重"
    )
    flags: dict[str, Any] = {"--attn-dp-size": candidate.attn_dp}
    if ctx.workload.range_ratio < 0.9 or ctx.workload.shared_prefix is not None:
        flags["--load-balance-method"] = "total_tokens"
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
            "layouts": rows,
            "best_layout": candidate.describe(),
            "decode_gain_per_gpu": round(gain, 3),
            "kv_capacity_gain": round(kv_gain, 3),
            "operating_batch": operating,
            "tpot_ms": {"baseline": round(tpot_base, 2), "candidate": round(tpot_cand, 2)},
        },
        references=references,
        requires_adaptation=bool(adaptation),
        layout=candidate,
    )


# ---------------------------------------------------------------------- prefill CP
def _cp_strategy(ctx: FeatureContext) -> str:
    return (
        "interleave"
        if ctx.arch.primary_architecture in PREFILL_CP_INTERLEAVE_ONLY_ARCHITECTURES
        else "zigzag"
    )


def _pick_backend(ctx: FeatureContext, allowed: tuple[str, ...]) -> tuple[str | None, bool]:
    """``(backend, needs_flag)``: the default backend if allowed, else the first declared one."""
    declared = ctx.hardware.software.attention_backends
    default = ctx.hardware.software.default_attention_backend
    if default in allowed:
        return default, False
    for backend in allowed:
        if backend in declared:
            return backend, True
    return None, False


def advise_prefill_cp(
    ctx: FeatureContext, reference: ParallelConfig | None = None
) -> FeatureAdvice:
    """Compare TTFT of the longest prompt with and without CP, against ``reference`` (DPA layout)."""
    key, title = "prefill_cp", "Prefill 上下文并行 CP（--enable-prefill-cp）"
    references = [
        source("prefill_cp_rules"),
        source("prefill_cp_platform"),
        source("cp_strategies"),
        source("prefill_cp_backends"),
        source("prefill_cp_can_apply"),
        source("attn_cp_rules"),
        source("breakable_prefill_graph"),
    ]
    arch_name = ctx.arch.primary_architecture
    strategy = _cp_strategy(ctx)
    problems: list[str] = []
    if ctx.arch.model_type in NO_ATTN_CP_MODEL_TYPES:
        problems.append(f"{ctx.arch.model_type} 不支持 --attn-cp-size > 1")
    if ctx.profile.platform == "musa":
        problems.append("MUSA 上的 prefill CP 已弃用（validate_prefill_cp_platform 直接报错）")
    if ctx.profile.platform == "rocm" and arch_name not in PREFILL_CP_HIP_ARCHITECTURES:
        problems.append(f"HIP 上 prefill CP 只支持 {sorted(PREFILL_CP_HIP_ARCHITECTURES)}")
    if arch_name in PREFILL_CP_ZIGZAG_ONLY_ARCHITECTURES:
        strategy = "zigzag"
    allowed = PREFILL_CP_STRATEGY_BACKENDS[strategy]
    adaptation: list[str] = []
    backend, needs_flag = (None, False)
    if ctx.out_of_tree:
        adaptation.append(
            f"CP 的注意力只在 {list(allowed)} 后端里实现（{strategy} 策略）：插件的 attention backend 需要实现同样的路径——"
            "按策略切分 query（zigzag：每条序列切 2×cp 块，rank i 持有第 i 和 2cp−1−i 块），在 CP 组内 all-gather KV，"
            "按全局位置做 causal mask，再把输出按原顺序还原"
        )
        adaptation.append(
            f"厂商通信库（{ctx.hardware.software.collective_library}）需要 CP 组上的 all_gather（KV 与 hidden states）"
        )
    else:
        backend, needs_flag = _pick_backend(ctx, allowed)
        if backend is None:
            problems.append(
                f"{ctx.hardware.name} 没有声明 {strategy} 所需的 attention backend {list(allowed)}"
            )
    if problems:
        return FeatureAdvice(
            key,
            title,
            "unsupported",
            problems[0],
            problems[1:],
            adaptation,
            [],
            {},
            {},
            {},
            problems,
            references,
            bool(adaptation),
        )
    base = ctx.parallel
    long_prompt = ctx.long_prompt
    slo = ctx.workload.slo.ttft_ms
    if base.tp == 1:
        advice = _tp1_advice(key, title, ctx, "prefill CP ")
        ttft = ctx.prefill_ms(base, long_prompt)
        advice.reasons.insert(
            0,
            f"单卡 prefill {long_prompt} token 约 {ttft / 1000:.1f} s"
            + (f"（SLO {slo / 1000:g} s）" if slo else ""),
        )
        advice.references = references
        advice.adaptation = adaptation
        advice.requires_adaptation = bool(adaptation)
        return advice
    references_layouts = [("baseline", base.with_(attn_cp=1))]
    if reference is not None and reference.attn_dp != base.attn_dp:
        references_layouts.append(("dp_attention", reference.with_(attn_cp=1, dcp=1)))
    rows: list[dict[str, Any]] = []
    measured: dict[ParallelConfig, tuple[float, float, int] | None] = {}

    def measure(parallel: ParallelConfig) -> tuple[float, float, int] | None:
        if parallel not in measured:
            plan = ctx.memory(parallel)
            if not plan.feasible:
                measured[parallel] = None
            else:
                point = ctx.decode_point(parallel)
                measured[parallel] = (
                    ctx.prefill_ms(parallel, long_prompt),
                    point.tokens_per_s_per_gpu if point else 0.0,
                    plan.max_total_tokens * parallel.attn_dp,
                )
        return measured[parallel]

    def row(role: str, parallel: ParallelConfig) -> dict[str, Any]:
        values = measure(parallel)
        if values is None:
            return {"role": role, "layout": parallel.describe(), "feasible": False}
        return {
            "role": role,
            "layout": parallel.describe(),
            "ttft_ms": round(values[0], 1),
            "decode_tok_s_per_gpu": round(values[1], 1),
            "kv_tokens_total": values[2],
        }

    Comparison = tuple[
        str, ParallelConfig, tuple[float, float, int], ParallelConfig, tuple[float, float, int]
    ]
    comparisons: list[Comparison] = []
    for role, ref in references_layouts:
        rows.append(row(role, ref))
        ref_values = measure(ref)
        if ref_values is None:
            continue
        options: list[tuple[ParallelConfig, tuple[float, float, int]]] = []
        for attn_cp in _powers_up_to(ref.tp // ref.attn_dp):
            candidate = ref.with_(attn_cp=attn_cp, dcp=1)
            if candidate.validate(ctx.arch):
                continue
            rows.append(row(f"{role}+cp", candidate))
            values = measure(candidate)
            if values is not None:
                options.append((candidate, values))
        if not options:
            continue
        meeting = [option for option in options if slo is not None and option[1][0] <= slo]
        if meeting:
            chosen = min(meeting, key=lambda option: option[0].attn_cp)
        else:
            chosen = min(options, key=lambda option: option[1][0])
        comparisons.append((role, ref, ref_values, chosen[0], chosen[1]))
    if not comparisons:
        return FeatureAdvice(
            key,
            title,
            "not_recommended",
            "没有显存可行的 CP 布局",
            [
                "CP 让 attention 只在 attn_tp=tp/(attn_dp·attn_cp) 内切分，复制后的 attention 权重放不下"
            ],
            adaptation,
            [],
            {},
            {},
            {"layouts": rows},
            [],
            references,
            bool(adaptation),
        )
    role, ref, ref_values, candidate, values = max(
        comparisons, key=lambda item: ratio(item[2][0], item[4][0])
    )
    ref_ttft, ref_tput, _ = ref_values
    ttft, tput, _ = values
    gain = ratio(ref_ttft, ttft)
    reasons = [
        f"最长 prompt {long_prompt} token：{ref.describe()} TTFT {ref_ttft / 1000:.2f} s → {candidate.describe()} "
        f"{ttft / 1000:.2f} s（{gain:.2f}×）" + (f"，SLO {slo / 1000:g} s" if slo else ""),
        "单节点上 CP 和 TP 都把一条 prompt 分到多张卡；CP 的价值在 attn_tp 很小（DP attention）时把长 prompt "
        "重新切开，并把每层 attention 的 TP all-reduce 换成 KV all-gather",
        f"decode 不走 CP（只有 EXTEND/MIXED 批次切分），且每个 rank 的注意力头变多：decode 吞吐 "
        f"{ref_tput:.0f}→{tput:.0f} tok/s/GPU",
    ]
    if role == "dp_attention":
        reasons.insert(
            1,
            f"DP attention 让每条 prompt 只在 attn_tp={ref.attn_tp} 张卡上做 attention；CP 保留 DP attention 的 KV 容量，"
            f"把长 prompt 切到 {candidate.attn_tp * candidate.attn_cp} 张卡",
        )
    for _other_role, other_ref, other_ref_values, other, other_values in comparisons:
        if other == candidate:
            continue
        reasons.append(
            f"对比 {other_ref.describe()}：{other.describe()} TTFT {other_ref_values[0] / 1000:.2f}→"
            f"{other_values[0] / 1000:.2f} s（{ratio(other_ref_values[0], other_values[0]):.2f}×）"
        )
    long_enough = long_prompt >= 8192
    slo_violated = slo is not None and ref_ttft > slo
    if (
        gain >= 1.3
        and (slo_violated or (slo is None and long_enough))
        and (slo is None or ttft <= slo)
    ):
        status = "recommend"
    elif gain >= 1.1 and long_enough:
        status = "optional"
    else:
        status = "not_recommended"
    tuning = [
        f"`--enable-prefill-cp --attn-cp-size {candidate.attn_cp} --cp-strategy {strategy}`（启用 prefill CP 时 "
        "`--cp-strategy` 必填；DeepSeek-V3.2 只能 interleave，MiMo-V2 只能 zigzag）",
        f"只有每条 extend 长度 ≥ {PREFILL_CP_MIN_TOKENS_FACTOR}×cp 的批次才切分；短请求照常走非 CP 路径",
        "tp 必须能被 attn_dp×attn_cp 整除；--dcp-size > 1 不能和 --enable-prefill-cp 同时用",
        "attn_cp>1 时 breakable prefill CUDA graph 会被关闭（除非是受支持的 CP+BCG 组合），prefill 走 eager",
        "PD 分离部署时只在 prefill 实例上开 CP；decode 实例用 DP attention / DCP 扩 KV 容量",
    ]
    flags: dict[str, Any] = {
        "--enable-prefill-cp": True,
        "--attn-cp-size": candidate.attn_cp,
        "--cp-strategy": strategy,
    }
    if candidate.attn_dp > 1:
        flags["--attn-dp-size"] = candidate.attn_dp
    if needs_flag and backend:
        flags["--attention-backend"] = backend
        tuning.append(
            f"默认 attention backend 不在 {list(allowed)} 中，需显式 `--attention-backend {backend}`"
        )
    return FeatureAdvice(
        key=key,
        title=title,
        status=status,
        summary=f"{candidate.describe()}（对比 {ref.describe()}）：最长 prompt TTFT "
        f"{ref_ttft / 1000:.2f}→{ttft / 1000:.2f} s",
        reasons=reasons,
        adaptation=adaptation,
        tuning=tuning,
        flags=flags,
        estimate={
            "layouts": rows,
            "reference_layout": ref.describe(),
            "best_layout": candidate.describe(),
            "ttft_gain": round(gain, 3),
            "long_prompt": long_prompt,
        },
        references=references,
        requires_adaptation=bool(adaptation),
        layout=candidate,
    )


# ---------------------------------------------------------------------- decode CP
def _dcp_comm_backend(ctx: FeatureContext) -> str:
    """``_dcp_comm_backend_default``: fi_a2a needs a Blackwell MNNVL domain, not modelled here."""
    if ctx.profile.in_tree and ctx.profile.platform in {"cuda", "rocm", "xpu"}:
        return "a2a"
    return "ag_rs"


def advise_dcp(ctx: FeatureContext) -> FeatureAdvice:
    key, title = "dcp", "Decode 上下文并行 DCP（--dcp-size）"
    references = [
        source("dcp_rules"),
        source("dcp_comm_backend_default"),
        source("dcp_kv_heads"),
        source("dcp_attention_backends"),
        source("hicache_dcp"),
        source("breakable_prefill_graph"),
    ]
    base = ctx.parallel
    attention = ctx.arch.attention
    if base.tp == 1:
        advice = _tp1_advice(key, title, ctx, "DCP ")
        advice.references = references
        return advice
    rows: list[dict[str, Any]] = []
    baseline = ctx.decode_point(base)
    rows.append({"role": "baseline", **_row(ctx, baseline, base)})
    candidates: list[tuple[DecodePoint, ParallelConfig]] = []
    rejected: list[str] = []
    for dcp in _powers_up_to(base.attn_tp):
        candidate = base.with_(dcp=dcp)
        errors = candidate.validate(ctx.arch)
        if errors:
            rejected.append(f"DCP{dcp}: {errors[0]}")
            continue
        point = ctx.decode_point(candidate)
        rows.append({"role": "candidate", **_row(ctx, point, candidate)})
        if point is not None:
            candidates.append((point, candidate))
    adaptation: list[str] = []
    if ctx.out_of_tree:
        adaptation.append(
            "插件的 attention backend 需要实现 DCP：KV 按 token 交错写到 DCP 组内各 rank（radix 树按 "
            "page_size×dcp 对齐），decode 时各 rank 对本地 KV 计算部分注意力并返回 LSE，再按 LSE 合并"
        )
        adaptation.append(
            f"通信后端会解析成 ag_rs（all-gather + reduce-scatter，非 CUDA/HIP/XPU 的默认），需要 "
            f"{ctx.hardware.software.collective_library} 支持这两个集合通信"
        )
    if not candidates or baseline is None:
        summary = "没有可行的 DCP 布局"
        reasons = rejected or ["DCP 布局的 KV 池不可行"]
        if not attention.is_mla and attention.num_kv_heads >= base.attn_tp:
            summary = (
                f"KV 头（{attention.num_kv_heads}）已经按 attn_tp={base.attn_tp} 切开，DCP 不适用"
            )
        return FeatureAdvice(
            key,
            title,
            "not_recommended",
            summary,
            reasons,
            adaptation,
            [],
            {},
            {},
            {"layouts": rows},
            [],
            references,
            bool(adaptation),
        )
    point, candidate = max(candidates, key=lambda item: item[0].tokens_per_s_per_gpu)
    gain = ratio(point.tokens_per_s_per_gpu, baseline.tokens_per_s_per_gpu)
    kv_gain = ratio(ctx.memory(candidate).max_total_tokens, ctx.memory(base).max_total_tokens)
    operating = ctx.operating_batch(base)
    tpot_base = ctx.decode_step_ms(base, operating)
    tpot_cand = ctx.decode_step_ms(candidate, operating)
    if gain >= RECOMMEND_GAIN:
        status = "recommend"
    elif gain >= OPTIONAL_GAIN or kv_gain >= 1.5:
        status = "optional"
    else:
        status = "not_recommended"
    why = (
        "MLA latent KV 在 attention-TP 组内是复制的，DCP 按 token 交错切分，每卡 KV 降为 1/dcp"
        if attention.is_mla
        else f"KV 头（{attention.num_kv_heads}）少于 attn_tp={base.attn_tp}，每卡复制 KV 头；DCP 把复制换成按 token 切分"
    )
    reasons = [
        why,
        f"KV-bound decode 吞吐 {pct(gain)}/GPU（{baseline.total_batch}→{point.total_batch} 并发），"
        f"工作点 TPOT {tpot_base:.1f}→{tpot_cand:.1f} ms",
        "DCP 不改变权重切分（不像 DP attention 会复制 attention 权重），代价是 decode 每层多一次部分结果交换",
        *rejected,
    ]
    comm = _dcp_comm_backend(ctx)
    tuning = [
        f"`--dcp-size {candidate.dcp}`：DCP 组嵌套在 attention-TP 组内（attn_tp={candidate.attn_tp} 必须能被 dcp 整除）",
        f"通信后端默认解析为 `{comm}`（fi_a2a 只在 Blackwell MNNVL 域内可用；`--dcp-replicate-q-proj` 只对 a2a/fi_a2a "
        "生效，用复制 q_proj 换掉按头的 Q all-gather）",
        "dcp>1 会关闭 breakable prefill CUDA graph；不能与 --enable-prefill-cp、--enable-kv-cache-sharding 同时用",
        "HiCache + DCP 只支持 MLA 模型的 L1/L2：不支持 --hicache-storage-backend（L3）、LMCache、HiSparse，"
        "投机解码只能不用或用 DSPARK",
        "PD 分离时 decode 实例的 DCP 需要 mooncake/nixl/mori/ascend/fake 传输后端",
    ]
    flags: dict[str, Any] = {"--dcp-size": candidate.dcp}
    if not ctx.out_of_tree:
        allowed = tuple(
            backend
            for backend in DCP_ATTENTION_BACKENDS
            if not (backend == "flashmla" and (ctx.kv_dtype or "").startswith("fp8"))
        )
        preferred = (
            ("flashmla", "flashinfer", "triton") if attention.is_mla else ("flashinfer", "triton")
        )
        ordered = tuple(b for b in preferred if b in allowed) + allowed
        backend, needs_flag = _pick_backend(ctx, ordered)
        if backend is None:
            return FeatureAdvice(
                key,
                title,
                "unsupported",
                f"{ctx.hardware.name} 没有声明支持 DCP 的 attention backend {list(DCP_ATTENTION_BACKENDS)}",
                reasons,
                adaptation,
                tuning,
                {},
                {},
                {"layouts": rows},
                [],
                references,
                False,
            )
        if needs_flag:
            flags["--attention-backend"] = backend
            tuning.append(
                f"默认 attention backend（{ctx.hardware.software.default_attention_backend}）没有 DCP 路径，"
                f"显式 `--attention-backend {backend}`"
                + (
                    "（FlashMLA 需要 `--page-size 64`，且不支持 DCP + FP8 KV）"
                    if backend == "flashmla"
                    else ""
                )
            )
            if backend == "flashmla":
                flags["--page-size"] = 64
    return FeatureAdvice(
        key=key,
        title=title,
        status=status,
        summary=f"{candidate.describe()}：KV 容量 ×{kv_gain:.2f}，KV-bound decode 吞吐 {pct(gain)}/GPU",
        reasons=reasons,
        adaptation=adaptation,
        tuning=tuning,
        flags=flags,
        estimate={
            "layouts": rows,
            "best_layout": candidate.describe(),
            "decode_gain_per_gpu": round(gain, 3),
            "kv_capacity_gain": round(kv_gain, 3),
            "comm_backend": comm,
            "tpot_ms": {"baseline": round(tpot_base, 2), "candidate": round(tpot_cand, 2)},
        },
        references=references,
        requires_adaptation=bool(adaptation),
        layout=candidate,
    )


# ---------------------------------------------------------------------- LayerNorm SP
def advise_layernorm_sp(ctx: FeatureContext) -> FeatureAdvice:
    key, title = "layernorm_sp", "LayerNorm 序列并行 SP（--enable-layernorm-sp）"
    references = [
        source("layernorm_sp"),
        source("layernorm_sp_rules"),
        source("layernorm_sp_fused_matmul"),
    ]
    supported = ctx.knowledge.layernorm_sp_architectures
    arch_name = ctx.arch.primary_architecture
    base = ctx.parallel
    if arch_name not in supported:
        return FeatureAdvice(
            key,
            title,
            "unsupported",
            f"只支持 {supported}，{arch_name} 没有 SP 实现",
            ["SGLang 在启动时校验 SP_SUPPORTED_ARCHITECTURES，其他架构直接报错"],
            references=references,
        )
    problems = []
    if base.tp <= 1:
        problems.append("需要 tp_size > 1（单卡没有可切的序列）")
    if base.attn_dp > 1:
        problems.append(
            "与 DP attention 不兼容：SP 在整个 TP 组切序列，DP attention 下各组是不同的序列"
        )
    if base.pp > 1:
        problems.append("与 PP 不兼容")
    if problems:
        status = "not_recommended" if base.tp <= 1 else "unsupported"
        return FeatureAdvice(
            key, title, status, problems[0], problems[1:], problems=problems, references=references
        )
    chunk = ctx.per_rank_chunk(base)
    estimate = ctx.cost(base).estimate(BatchShape.prefill(1, chunk, 0))
    norm_ms = (
        sum(t.total_us for t in estimate.timings if t.op.kind == "norm") * estimate.scale / 1000
    )
    comm_ms = estimate.by_kind().get("comm", 0.0) / 1000
    step_ms = estimate.total_us / 1000
    saved_ms = norm_ms * (1 - 1 / base.tp)
    activation_mb = chunk * ctx.arch.hidden_size * 2 * 2 * (1 - 1 / base.tp) / 2**20
    fused = (
        ctx.is_cuda
        and ctx.arch.quant is None
        and not ctx.arch.attention.has_qkv_bias
        and (ctx.weight_dtype is None or ctx.weight_dtype in {"bf16", "fp16"})
    )
    reasons = [
        f"prefill（chunk {chunk}）里 norm/residual 只在 1/{base.tp} 的 token 上做：约省 {saved_ms:.2f} ms/步"
        f"（步长 {step_ms:.1f} ms），残差/归一化激活少 {activation_mb:.0f} MB",
        "all-reduce 换成 reduce-scatter + all-gather，通信量不变；只在 EXTEND 阶段生效，decode 不受影响",
    ]
    if fused:
        reasons.append(
            f"线性层是未量化、无 bias 的 BF16/FP16：可走 torch symm_mem 的 fused_all_gather_matmul / "
            f"fused_matmul_reduce_scatter，把 prefill 的 TP 通信（{comm_ms:.2f} ms/步）与 GEMM 重叠"
        )
    else:
        reasons.append(
            "不满足 symm_mem 融合条件（需 CUDA、未量化、无 bias、BF16/FP16 权重），走普通 RS/AG"
        )
    adaptation = []
    if ctx.out_of_tree:
        adaptation.append(
            f"{ctx.hardware.software.collective_library} 需要 reduce_scatter_tensor / all_gather_into_tensor；"
            "symm_mem 融合路径只在 CUDA 上可用"
        )
    share = (saved_ms + (comm_ms * 0.5 if fused else 0.0)) / step_ms if step_ms else 0.0
    prefill_heavy = ctx.workload.mean_input_len >= 2 * ctx.workload.mean_output_len
    status = "recommend" if share >= 0.03 and prefill_heavy else "optional"
    return FeatureAdvice(
        key=key,
        title=title,
        status=status,
        summary=f"prefill 每步约省 {saved_ms:.2f} ms"
        + (f"，另有 {comm_ms:.2f} ms TP 通信可与 GEMM 重叠" if fused else ""),
        reasons=reasons,
        adaptation=adaptation,
        tuning=[
            "`--enable-layernorm-sp`；与 DP attention、投机解码（EAGLE/EAGLE3 的 aux hidden states 会被切开）、PP 互斥",
            "收益随 prefill token 数增长：长 prompt / 大 chunk 时更明显",
        ],
        flags={"--enable-layernorm-sp": True},
        estimate={
            "prefill_chunk": chunk,
            "norm_saved_ms_per_step": round(saved_ms, 3),
            "prefill_comm_ms_per_step": round(comm_ms, 3),
            "symm_mem_fused": fused,
        },
        references=references,
        requires_adaptation=bool(adaptation),
    )
