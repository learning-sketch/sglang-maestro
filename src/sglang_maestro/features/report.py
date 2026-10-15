"""Run every feature advisor for one model/device/workload and merge the winners into one launch.

Each advisor is measured against the baseline layout on its own. The combined plan then picks
the decode layout (DP attention, DCP and EP merged when they compose), re-evaluates the
layout-dependent features (HiCache, speculative decoding, TBO, LayerNorm SP) on it and drops
whatever SGLang would reject together.
"""

from __future__ import annotations

import dataclasses
import itertools
from dataclasses import dataclass, field
from typing import Any

from sglang_maestro.commands import (
    Command,
    launch_server_command,
    multi_node_flags,
    parallel_server_flags,
)
from sglang_maestro.features.base import FeatureAdvice, FeatureContext
from sglang_maestro.features.moe_cache import (
    advise_eplb,
    advise_expert_parallel,
    advise_hicache,
    advise_tbo,
)
from sglang_maestro.features.parallelism import (
    advise_dcp,
    advise_dp_attention,
    advise_layernorm_sp,
    advise_prefill_cp,
)
from sglang_maestro.features.speculative import advise_speculative
from sglang_maestro.markdown import STATUS_LABELS, bullets, code, flags_text, table
from sglang_maestro.model.parallel import ParallelConfig

FEATURE_KEYS = (
    "dp_attention",
    "prefill_cp",
    "dcp",
    "layernorm_sp",
    "hicache",
    "deepep",
    "tbo",
    "eplb",
    "speculative",
)
GOOD = ("recommend", "optional")
SERVER_PORT = 30000


@dataclass
class CombinedPlan:
    layout: ParallelConfig
    flags: dict[str, Any]
    env: dict[str, str]
    enabled: list[str]
    dropped: list[str]
    notes: list[str]
    candidates: list[dict[str, Any]]
    command: Command
    capacity_note: str | None = None
    """Set when one instance of this layout cannot carry the workload's arrival rate."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout.describe(),
            "flags": self.flags,
            "env": self.env,
            "enabled": self.enabled,
            "dropped": self.dropped,
            "notes": self.notes,
            "capacity_note": self.capacity_note,
            "candidates": self.candidates,
            "command": self.command.to_dict(),
        }


@dataclass
class FeatureReport:
    model: str
    hardware: str
    workload: str
    workload_description: str
    platform: str
    baseline: ParallelConfig
    base_flags: dict[str, Any]
    knowledge_commit: str | None
    advices: list[FeatureAdvice]
    combined: CombinedPlan
    assumptions: list[str] = field(default_factory=list)

    def advice(self, key: str) -> FeatureAdvice | None:
        return next((item for item in self.advices if item.key == key), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "hardware": self.hardware,
            "workload": self.workload,
            "workload_description": self.workload_description,
            "platform": self.platform,
            "baseline": self.baseline.describe(),
            "base_flags": self.base_flags,
            "knowledge_commit": self.knowledge_commit,
            "advices": [item.to_dict() for item in self.advices],
            "combined": self.combined.to_dict(),
            "assumptions": self.assumptions,
        }

    def to_markdown(self) -> str:
        return render_feature_report(self)


def server_flags(
    ctx: FeatureContext, layout: ParallelConfig, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """``launch_server`` flags for ``layout`` on top of the baseline precision route."""
    flags: dict[str, Any] = {"--model-path": ctx.model_path}
    if ctx.out_of_tree:
        flags["--device"] = ctx.device_type
    flags.update(parallel_server_flags(layout))
    flags.update({key: value for key, value in ctx.base_flags.items() if key != "--tp-size"})
    flags.update(extra or {})
    flags["--trust-remote-code"] = bool(ctx.arch.raw.get("auto_map"))
    flags.update(multi_node_flags(layout.num_gpus, ctx.hardware.devices_per_node))
    flags.setdefault("--host", "0.0.0.0")
    flags.setdefault("--port", SERVER_PORT)
    return flags


def _run_advisors(ctx: FeatureContext, keys: tuple[str, ...]) -> list[FeatureAdvice]:
    results: dict[str, FeatureAdvice] = {}

    def need(key: str) -> bool:
        return key in keys

    def good_layout(key: str) -> ParallelConfig | None:
        advice = results.get(key)
        return advice.layout if advice is not None and advice.status in GOOD else None

    if need("dp_attention") or need("prefill_cp"):
        results["dp_attention"] = advise_dp_attention(ctx)
    if need("prefill_cp"):
        results["prefill_cp"] = advise_prefill_cp(ctx, good_layout("dp_attention"))
    if need("dcp"):
        results["dcp"] = advise_dcp(ctx)
    if need("layernorm_sp"):
        results["layernorm_sp"] = advise_layernorm_sp(ctx)
    if need("hicache"):
        results["hicache"] = advise_hicache(ctx)
    if need("deepep") or need("tbo") or need("eplb"):
        results["deepep"] = advise_expert_parallel(ctx)
    if need("tbo"):
        results["tbo"] = advise_tbo(ctx, good_layout("deepep"))
    if need("eplb"):
        results["eplb"] = advise_eplb(ctx, good_layout("deepep"))
    if need("speculative"):
        results["speculative"] = advise_speculative(ctx)
    return [results[key] for key in FEATURE_KEYS if key in results and key in keys]


def _merge(base: ParallelConfig, parts: list[ParallelConfig]) -> ParallelConfig:
    layout = base
    for part in parts:
        changes: dict[str, Any] = {}
        if part.attn_dp != base.attn_dp:
            changes["attn_dp"] = part.attn_dp
        if part.dcp != base.dcp:
            changes["dcp"] = part.dcp
        if part.ep != base.ep or part.moe_a2a != base.moe_a2a:
            changes.update(ep=part.ep, moe_a2a=part.moe_a2a)
        layout = layout.with_(**changes)
    return layout


def _score(ctx: FeatureContext, layout: ParallelConfig) -> dict[str, Any] | None:
    point = ctx.decode_point(layout)
    if point is None:
        return None
    operating = ctx.operating_batch(layout)
    demand, cap = ctx.concurrency_demand(layout)
    overloaded = demand is not None and demand > cap
    tpot = ctx.decode_step_ms(layout, operating)
    ttft = ctx.prefill_ms(layout, ctx.long_prompt)
    slo = ctx.workload.slo
    meets = (
        not overloaded
        and (slo.ttft_ms is None or ttft <= slo.ttft_ms)
        and (slo.tpot_ms is None or tpot <= slo.tpot_ms)
    )
    return {
        "layout": layout.describe(),
        "decode_tok_s_per_gpu": round(point.tokens_per_s_per_gpu, 1),
        "kv_tokens_total": point.kv_tokens_per_rank * layout.attn_dp,
        "operating_batch": operating,
        "concurrency_demand": round(demand, 1) if demand is not None else None,
        "concurrency_capacity": cap,
        "overloaded": overloaded,
        "tpot_ms": round(tpot, 2),
        "long_prompt_ttft_ms": round(ttft, 1),
        "meets_slo": meets,
    }


def _pick_layout(
    ctx: FeatureContext, advices: dict[str, FeatureAdvice]
) -> tuple[ParallelConfig, list[dict[str, Any]]]:
    base = ctx.parallel
    parts = [
        advices[key].layout
        for key in ("dp_attention", "dcp", "deepep")
        if key in advices and advices[key].status in GOOD and advices[key].layout is not None
    ]
    layouts: list[ParallelConfig] = [base]
    for size in range(1, len(parts) + 1):
        for subset in itertools.combinations(parts, size):
            merged = _merge(base, list(subset))
            if merged not in layouts and not merged.validate(ctx.arch):
                layouts.append(merged)
    scored = []
    for layout in layouts:
        score = _score(ctx, layout)
        if score is not None:
            scored.append((layout, score))
    if not scored:
        return base, []
    best_layout, _ = max(
        scored, key=lambda item: (item[1]["meets_slo"], item[1]["decode_tok_s_per_gpu"])
    )
    baseline_score = next((score for layout, score in scored if layout == base), None)
    if baseline_score is not None:
        best_score = next(score for layout, score in scored if layout == best_layout)
        if best_score["decode_tok_s_per_gpu"] < baseline_score["decode_tok_s_per_gpu"] * 1.03 and (
            best_score["meets_slo"] == baseline_score["meets_slo"]
        ):
            best_layout = base
    return best_layout, [score for _, score in scored]


def combine(ctx: FeatureContext, advices: list[FeatureAdvice]) -> CombinedPlan:
    by_key = {item.key: item for item in advices}
    layout, candidates = _pick_layout(ctx, by_key)
    derived = dataclasses.replace(ctx, parallel=layout)
    flags: dict[str, Any] = {}
    env: dict[str, str] = {}
    enabled: list[str] = []
    dropped: list[str] = []
    notes: list[str] = []
    base = ctx.parallel
    chosen = next((row for row in candidates if row["layout"] == layout.describe()), None)
    capacity_note = None
    if chosen is not None and chosen["overloaded"]:
        capacity_note = (
            f"单个 {layout.describe()} 实例承载不了该到达率（并发需求 {chosen['concurrency_demand']:g} > "
            f"KV 可承载 {chosen['concurrency_capacity']}）：需要多副本或 PD 分离，见 `sglang-maestro deploy`"
        )

    if layout.attn_dp > base.attn_dp:
        enabled.append("dp_attention")
        flags.update(
            {k: v for k, v in by_key["dp_attention"].flags.items() if k == "--load-balance-method"}
        )
    if layout.dcp > base.dcp:
        enabled.append("dcp")
        flags.update({k: v for k, v in by_key["dcp"].flags.items() if k != "--dcp-size"})
    if layout.ep > base.ep:
        enabled.append("deepep")
        if "deepep" in by_key:
            env.update(by_key["deepep"].env)

    cp = by_key.get("prefill_cp")
    if cp is not None and cp.status == "recommend" and cp.layout is not None:
        if layout.dcp > 1:
            dropped.append(
                "prefill_cp：与 DCP 互斥（--dcp-size > 1 不能和 --enable-prefill-cp 同时用）"
            )
            notes.append(
                f"PD 分离部署时：prefill 实例用 CP（{cp.layout.describe()}），decode 实例用 {layout.describe()}"
            )
        else:
            with_cp = layout.with_(attn_cp=cp.layout.attn_cp)
            if not with_cp.validate(ctx.arch) and ctx.memory(with_cp).feasible:
                layout = with_cp
                derived = dataclasses.replace(ctx, parallel=layout)
                enabled.append("prefill_cp")
                flags.update(
                    {
                        k: v
                        for k, v in cp.flags.items()
                        if k not in {"--attn-cp-size", "--attn-dp-size"}
                    }
                )
            else:
                dropped.append(f"prefill_cp：{with_cp.describe()} 不是合法布局")

    hicache = advise_hicache(derived)
    if hicache.status == "recommend":
        enabled.append("hicache")
        flags.update(hicache.flags)
    elif "hicache" in by_key and by_key["hicache"].status == "recommend":
        dropped.append(
            f"hicache：在 {layout.describe()} 上重新评估为{STATUS_LABELS[hicache.status]}（{hicache.summary}）"
        )

    spec = advise_speculative(derived)
    if spec.status == "recommend":
        if layout.dcp > 1 and "hicache" in enabled:
            dropped.append("speculative：DCP + HiCache 时投机解码只能是 DSPARK 或不用")
        else:
            enabled.append("speculative")
            flags.update(spec.flags)
            env.update(spec.env)
            if spec.flags.get("--speculative-draft-model-path", "").startswith("<"):
                notes.append(
                    "把 `--speculative-draft-model-path` 换成与目标模型配套的 EAGLE3 draft 模型路径"
                )
    elif "speculative" in by_key and by_key["speculative"].status == "recommend":
        dropped.append(
            f"speculative：在 {layout.describe()} 上重新评估为{STATUS_LABELS[spec.status]}（{spec.summary}）"
        )

    if layout.uses_a2a and ctx.arch.moe is not None:
        tbo = advise_tbo(derived, layout)
        if tbo.status == "recommend" and tbo.layout == layout:
            enabled.append("tbo")
            flags.update({k: v for k, v in tbo.flags.items() if k == "--enable-two-batch-overlap"})
        elif tbo.status == "optional":
            notes.append(f"TBO 可选（{tbo.summary}），默认不开；PD 分离时可只在 prefill 实例上评估")

    sp = by_key.get("layernorm_sp")
    if sp is not None and sp.status == "recommend":
        if layout.attn_dp > 1:
            dropped.append("layernorm_sp：与 DP attention 互斥")
        elif "speculative" in enabled:
            dropped.append("layernorm_sp：与 EAGLE 系投机解码互斥（aux hidden states 会被切开）")
        else:
            enabled.append("layernorm_sp")
            flags.update(sp.flags)

    eplb = by_key.get("eplb")
    if eplb is not None and eplb.status in GOOD and layout.ep > 1:
        notes.append("EPLB 需要先在真实流量上录专家分布：" + eplb.tuning[0])

    launch = server_flags(ctx, layout, flags)
    command = launch_server_command(
        ctx.knowledge,
        launch,
        env=env,
        title=f"组合配置：{layout.describe()}",
        note="特性逐项对比基线后合并；合并后的效果请用对应压测命令验证",
    )
    return CombinedPlan(
        layout=layout,
        flags=launch,
        env=env,
        enabled=enabled,
        dropped=dropped,
        notes=notes,
        candidates=candidates,
        command=command,
        capacity_note=capacity_note,
    )


def advise_features(
    ctx: FeatureContext, *, only: list[str] | tuple[str, ...] | None = None
) -> FeatureReport:
    keys = tuple(only) if only else FEATURE_KEYS
    unknown = sorted(set(keys) - set(FEATURE_KEYS))
    if unknown:
        raise ValueError(f"unknown features {unknown}; known: {list(FEATURE_KEYS)}")
    advices = _run_advisors(ctx, keys)
    combined = combine(ctx, advices)
    platform = (
        f"{ctx.profile.platform}（树内）"
        if ctx.profile.in_tree
        else f"树外插件（device_type={ctx.device_type}）"
    )
    assumptions = [
        "所有耗时来自逐算子 roofline 成本模型（效率系数见硬件 YAML，可用 `sglang-maestro calibrate` 用实测日志校准）",
        "TTFT 指单条请求独占时的 prefill 时间（不含排队）；工作点并发按 Little 定律从到达率推出",
        f"投机解码接受率是负载 YAML 里的假设值（{ctx.workload.spec_accept_rate:g}），上线后以日志 accept len 为准",
    ]
    return FeatureReport(
        model=ctx.arch.name,
        hardware=ctx.hardware.name,
        workload=ctx.workload.name,
        workload_description=ctx.workload.describe(),
        platform=platform,
        baseline=ctx.parallel,
        base_flags=ctx.base_flags,
        knowledge_commit=ctx.knowledge.commit,
        advices=advices,
        combined=combined,
        assumptions=assumptions,
    )


def render_advice(advice: FeatureAdvice) -> str:
    status = STATUS_LABELS.get(advice.status, advice.status)
    parts = [f"### {advice.title} —— {status}", f"**结论**：{advice.summary}"]
    if advice.problems:
        parts.append("**会触发的启动错误**：\n" + bullets(advice.problems))
    if advice.reasons:
        parts.append("**依据**：\n" + bullets(advice.reasons))
    if advice.adaptation:
        parts.append("**本设备需要的适配**：\n" + bullets(advice.adaptation))
    if advice.tuning:
        parts.append("**调优要点**：\n" + bullets(advice.tuning))
    if advice.flags:
        parts.append(f"**参数**：`{flags_text(advice.flags)}`")
    if advice.env:
        parts.append("**环境变量**：`" + " ".join(f"{k}={v}" for k, v in advice.env.items()) + "`")
    layouts = advice.estimate.get("layouts")
    if isinstance(layouts, list) and layouts and isinstance(layouts[0], dict):
        columns = [key for key in layouts[0] if key != "role"]
        for row in layouts[1:]:
            columns += [key for key in row if key not in columns and key != "role"]
        parts.append(
            table(
                ["角色", *columns],
                [[row.get("role"), *(row.get(c) for c in columns)] for row in layouts],
            )
        )
    if advice.references:
        parts.append("<sub>源码依据：" + "；".join(advice.references) + "</sub>")
    return "\n\n".join(parts)


def render_feature_report(report: FeatureReport) -> str:
    combined = report.combined
    header = [
        f"## 框架特性适配与调优：{report.model} @ {report.hardware}",
        bullets(
            [
                f"负载：`{report.workload}` —— {report.workload_description}",
                f"平台：{report.platform}；基线布局 {report.baseline.describe()}"
                + (f"（`{flags_text(report.base_flags)}`）" if report.base_flags else ""),
                f"SGLang 快照：{report.knowledge_commit or '未知'}",
            ]
        ),
        "### 总览",
        table(
            ["特性", "结论", "摘要"],
            [
                [item.title, STATUS_LABELS.get(item.status, item.status), item.summary]
                for item in report.advices
            ],
        ),
        "### 组合配置",
        bullets(
            [
                f"布局：**{combined.layout.describe()}**",
                "启用："
                + ("、".join(combined.enabled) if combined.enabled else "无（基线已是最优）"),
                *(f"未启用 {item}" for item in combined.dropped),
                *combined.notes,
                combined.capacity_note or "",
            ]
        ),
    ]
    if combined.candidates:
        header.append(
            table(
                [
                    "候选布局",
                    "decode tok/s/GPU",
                    "KV token（总）",
                    "并发需求/可承载",
                    "TPOT ms",
                    "最长 prompt TTFT ms",
                    "满足 SLO",
                ],
                [
                    [
                        row["layout"],
                        row["decode_tok_s_per_gpu"],
                        row["kv_tokens_total"],
                        f"{row['concurrency_demand']:g}/{row['concurrency_capacity']}"
                        + ("（过载）" if row["overloaded"] else "")
                        if row["concurrency_demand"] is not None
                        else f"吞吐模式/{row['concurrency_capacity']}",
                        row["tpot_ms"],
                        row["long_prompt_ttft_ms"],
                        row["meets_slo"],
                    ]
                    for row in combined.candidates
                ],
            )
        )
    header.append(code(combined.command.shell()))
    if combined.command.problems:
        header.append("⚠ 参数校验：\n" + bullets(combined.command.problems))
    body = [render_advice(item) for item in report.advices]
    footer = ["### 假设", bullets(report.assumptions)]
    return "\n\n".join([*header, *body, *footer]) + "\n"
