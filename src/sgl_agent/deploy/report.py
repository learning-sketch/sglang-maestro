"""Markdown rendering of a deployment plan."""

from __future__ import annotations

from typing import Any

from sgl_agent.deploy.plan import LIMIT_LABELS, Candidate, DeployPlan, Pool
from sgl_agent.markdown import bullets, command_block, flags_text, table

ROLE_NAMES = {"colocated": "个实例", "prefill": "个 prefill 实例", "decode": "个 decode 实例"}


def _capacity(pool: Pool, unit: str) -> str:
    capacity = pool.capacity
    if capacity is None:
        return "—"
    text = f"{capacity.rate:.3g} {unit}" if capacity.rate > 0 else "无"
    if capacity.limit:
        text += f"（瓶颈：{LIMIT_LABELS[capacity.limit]}）"
    return text


def _label(candidate: Candidate) -> str:
    if candidate.feasible:
        return candidate.describe()
    return " + ".join(pool.spec.describe() for pool in candidate.pools.values()) + "（不可行）"


def _candidate_capacity(candidate: Candidate, unit: str) -> str:
    if not candidate.disaggregated:
        return _capacity(candidate.pools["colocated"], unit)
    return "；".join(f"{role} {_capacity(pool, unit)}" for role, pool in candidate.pools.items())


def _candidate_notes(candidate: Candidate) -> str:
    notes = list(candidate.notes)
    if candidate.retuned:
        notes.insert(0, "按单副本到达率重新调优")
    if candidate.added:
        notes.insert(0, f"整集群仿真后加了 {candidate.added} 个实例")
    if candidate.result is None and candidate.feasible:
        notes.append("未做整集群仿真（GPU 更多，不会被选中）")
    return "；".join(notes)


def _prediction(plan: DeployPlan, candidate: Candidate) -> str:
    result = candidate.result
    if result is None:
        return "没有整集群仿真结果"
    slo = plan.workload.slo
    percentile = slo.percentile
    parts = [
        f"P{percentile} TTFT {result.ttft_ms[percentile]:,.0f} ms",
        f"P{percentile} TPOT {result.tpot_ms[percentile]:.1f} ms",
        f"输出 {result.output_throughput:,.0f} tok/s（每卡 {result.output_throughput_per_gpu:,.0f}）",
    ]
    if slo.defined:
        parts.append(f"SLO 达成率 {result.slo_attainment:.1%}")
    if result.cache_hit_rate > 0:
        parts.append(f"前缀命中率 {result.cache_hit_rate:.0%}")
    if result.retractions:
        parts.append(f"撤回 {result.retractions} 次")
    if result.mean_transfer_ms is not None:
        parts.append(f"KV 传输平均 {result.mean_transfer_ms:.1f} ms")
    return "，".join(parts)


def _summary(plan: DeployPlan) -> list[str]:
    chosen = plan.chosen
    if chosen is None or not chosen.feasible:
        return ["**没有可行方案**：没有一个实例规模能放下模型并处理这个负载，见下面的瓶颈分析"]
    if not plan.workload.slo.defined:
        verdict = "输出吞吐最高"
    elif chosen.meets_slo:
        verdict = "满足 SLO"
    else:
        verdict = "**满足不了 SLO**（已是最接近的方案）"
    lines = [
        f"**{chosen.describe()}**：{chosen.num_gpus} 卡、{plan.num_nodes} 台机器、"
        f"{chosen.num_instances} 个实例，{verdict}",
        "预测（成本模型 + 仿真）：" + _prediction(plan, chosen),
    ]
    if plan.mode == "throughput":
        lines.append(
            "吞吐模式在 0 时刻发出全部请求（bench_serving 默认 `--request-rate inf`），TTFT 含排队时间"
        )
    return lines + plan.reasons


def _candidates_table(plan: DeployPlan) -> str:
    slo = plan.workload.slo
    percentile = slo.percentile
    rate_mode = plan.mode == "rate"
    headers = ["方案", "GPU"]
    if rate_mode:
        headers.append("单实例容量")
    headers += [f"P{percentile} TTFT ms", f"P{percentile} TPOT ms", "输出 tok/s/GPU"]
    if slo.defined:
        headers += ["SLO 达成率", "满足 SLO"]
    headers.append("说明")
    rows = []
    for candidate in plan.candidates:
        result = candidate.result
        row: list[Any] = [
            _label(candidate) + ("（选中）" if candidate is plan.chosen else ""),
            candidate.num_gpus if candidate.feasible else None,
        ]
        if rate_mode:
            row.append(_candidate_capacity(candidate, plan.rate_unit))
        row += [
            result.ttft_ms[percentile] if result else None,
            result.tpot_ms[percentile] if result else None,
            result.output_throughput_per_gpu if result else None,
        ]
        if slo.defined:
            row += [
                f"{result.slo_attainment:.1%}" if result else None,
                result.meets_slo if result else None,
            ]
        row.append(_candidate_notes(candidate))
        rows.append(row)
    return table(headers, rows)


def _pool_lines(pool: Pool) -> list[str]:
    spec = pool.spec
    facts = [
        f"每个 attention-DP rank 的 KV 池 {spec.kv_tokens_per_rank:,} token",
        f"最多 {spec.max_running_per_rank:,} 个运行请求/rank",
    ]
    if spec.chunk_per_rank:
        facts.append(f"chunked prefill {spec.chunk_per_rank * spec.attn_dp:,} token")
    if spec.speculative:
        facts.append(spec.speculative.describe())
    if spec.host_cache_tokens_per_rank:
        facts.append(f"HiCache 主机层 {spec.host_cache_tokens_per_rank:,} token/rank")
    if not spec.radix_cache:
        facts.append("无 radix cache")
    lines = [
        f"**{pool.count} {ROLE_NAMES[pool.role]}**：{spec.describe()}（每个 {spec.num_gpus} 卡）",
        "；".join(facts),
    ]
    if spec.features:
        lines.append("启用的特性：" + "、".join(spec.features))
    lines.append(f"参数：`{flags_text(spec.flags)}`")
    if spec.env:
        lines.append("环境变量：`" + " ".join(f"{k}={v}" for k, v in spec.env.items()) + "`")
    return lines


def _commands(plan: DeployPlan) -> list[str]:
    parts = []
    for node in plan.nodes:
        parts.append(f"#### 节点 {node.node}（用 {node.gpus_used}/{plan.devices_per_node} 卡）")
        parts += [command_block(command) for command in node.commands]
    if plan.router is not None:
        parts += ["#### 路由", command_block(plan.router)]
    return parts


def render_deploy_plan(plan: DeployPlan) -> str:
    workload = plan.workload
    mode = (
        f"按到达率 {workload.request_rate:g} {plan.rate_unit} 求满足 SLO 的最少 GPU"
        if plan.mode == "rate"
        else f"吞吐模式：在 {plan.budget_gpus} 卡内求最高输出吞吐"
    )
    parts = [
        f"## 部署方案：{plan.model} @ {plan.hardware}",
        bullets(
            [
                f"负载：`{workload.name}` —— {workload.describe()}",
                f"目标：{mode}",
                f"平台：{plan.platform}；GPU 预算 {plan.budget_gpus} 卡，每台机器 "
                f"{plan.devices_per_node} 卡",
                f"SGLang 快照：{plan.knowledge_commit or '未知'}",
            ]
        ),
        "### 结论",
        bullets(_summary(plan)),
    ]
    if plan.candidates:
        parts += ["### 候选对比", _candidates_table(plan)]
    chosen = plan.chosen
    if chosen is not None and chosen.feasible:
        parts.append("### 实例配置")
        parts += [bullets(_pool_lines(pool)) for pool in chosen.pools.values()]
        parts.append("### 部署命令")
        parts += _commands(plan)
        parts.append("### 上线与验证")
        parts.append("\n".join(f"{i}. {step}" for i, step in enumerate(plan.steps, 1)))
        parts += [command_block(command) for command in plan.checks]
        if plan.bench is not None:
            parts.append(command_block(plan.bench))
    if plan.bottlenecks:
        parts += ["### 瓶颈", bullets(plan.bottlenecks)]
    if plan.adaptation:
        parts += ["### 本设备需要的适配", bullets(plan.adaptation)]
    if plan.notes:
        parts += ["### 说明", bullets(plan.notes)]
    if plan.assumptions:
        parts += ["### 假设", bullets(plan.assumptions)]
    if plan.references:
        parts.append("<sub>源码依据：" + "；".join(plan.references) + "</sub>")
    return "\n\n".join(parts) + "\n"
