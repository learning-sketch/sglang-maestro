"""Markdown rendering of a scenario 1 adaptation report."""

from __future__ import annotations

from sgl_agent.adapt.analysis import AdaptationReport
from sgl_agent.adapt.bringup import BringupStep
from sgl_agent.adapt.operators import GAP_STATUSES
from sgl_agent.markdown import bullets, command_block, findings_section, flags_text, table

HOOK_LEVEL_LABELS = {
    "required": "必须实现",
    "conditional": "本配置需要",
    "recommended": "建议实现",
    "optional": "可选",
}
COMPILE_APPLIES_LABELS = {
    "yes": "会走平台编译后端",
    "conditional": "开启对应特性时才走",
    "no": "不受影响",
    "manual": "需要人工确认",
}


def _platform_line(report: AdaptationReport) -> str:
    profile = report.profile
    if profile.in_tree:
        return f"SGLang 内置平台 `{profile.platform}`（device_type={profile.device_type}）"
    parts = [
        f"树外插件平台（device_type=`{profile.device_type}`",
        f"fused op 分发键 `forward_{profile.dispatch_key}`"
        if profile.dispatch_key
        else "无平台 forward",
        f"编译后端 `{profile.compile_backend}`",
        f"通信后端 `{profile.distributed_backend}`（SGLang 默认 {profile.default_distributed_backend}）",
    ]
    return "，".join(parts) + "）"


def _summary(report: AdaptationReport) -> list[str]:
    inventory = report.inventory
    gaps = [item for item in inventory.requirements if item.status in GAP_STATUSES]
    blocking = [item for item in gaps if item.status in ("broken", "missing")]
    required = report.checklist.by_level("required")
    conditional = report.checklist.by_level("conditional")
    route = report.precision.recommended
    lines = []
    if not report.support.supported:
        lines.append(
            f"SGLang 没有 `{report.support.architecture}` 的实现：先用 `--model-impl transformers` 跑通，"
            "再把模型实现放进外部模型包"
        )
    else:
        lines.append(
            f"SGLang 已实现 `{report.support.architecture}`（`models/{report.support.model_file}.py`），"
            "模型代码不用改"
        )
    if report.profile.out_of_tree:
        lines.append(
            f"平台插件：必须实现 {len(required)} 个 hook，本配置另需 {len(conditional)} 个"
        )
    if blocking:
        names = "、".join(item.title.split("（")[0] for item in blocking[:4])
        more = "…" if len(blocking) > 4 else ""
        lines.append(
            f"算子：{len(gaps)} 类有缺口，其中 {len(blocking)} 类会直接阻塞（{names}{more}）"
        )
    elif gaps:
        lines.append(f"算子：{len(gaps)} 类回落到参考实现或有 host 同步，能跑但要替换")
    else:
        lines.append("算子：全部有可用实现")
    if route is not None:
        sizing = route.sizing
        lines.append(
            f"推荐精度路线：{route.title}"
            + (
                f"，最少 {sizing.parallel.describe()}（{sizing.parallel.num_gpus} 卡）"
                if sizing
                else ""
            )
            + f"；{route.accuracy}"
        )
    else:
        lines.append("没有一条精度路线放得下模型：见精度路线表里的阻碍")
    return lines


def _hooks_table(report: AdaptationReport) -> str:
    rows = [
        [
            item.name,
            HOOK_LEVEL_LABELS.get(item.level, item.level),
            item.owner,
            item.reason,
            item.how,
            item.default,
        ]
        for item in report.checklist.requirements
    ]
    return table(["hook", "级别", "位置", "为什么需要", "怎么做", "不实现时的默认行为"], rows)


def _operators_table(report: AdaptationReport) -> str:
    rows = []
    for item in report.inventory.requirements:
        rows.append(
            [
                item.title,
                item.status_label,
                item.resolved_path,
                f"{item.share_decode:.1%}" if item.share_decode is not None else None,
                f"{item.share_prefill:.1%}" if item.share_prefill is not None else None,
                item.how if item.status in GAP_STATUSES else "",
            ]
        )
    return table(["算子", "状态", "当前路径", "decode 占比", "prefill 占比", "需要做的"], rows)


def _precision_section(report: AdaptationReport) -> list[str]:
    plan = report.precision
    rows = []
    for route in plan.routes:
        sizing = route.sizing
        rows.append(
            [
                route.title + ("（推荐）" if route is plan.recommended else ""),
                route.available,
                route.accuracy,
                sizing.parallel.describe() if sizing else None,
                round(sizing.weights_gb_per_gpu, 1) if sizing else None,
                sizing.kv_tokens if sizing else None,
                "；".join(route.blockers) or "",
            ]
        )
    parts = [
        f"checkpoint 精度：`{plan.checkpoint}`；KV 池至少 {plan.min_kv_tokens:,} token 才算放得下",
        table(
            ["路线", "可用", "精度影响", "最小布局", "每卡权重 GB", "KV token", "阻碍"],
            rows,
        ),
    ]
    route = plan.recommended
    if route is not None:
        details = []
        if route.flags:
            details.append(f"参数：`{flags_text(route.flags)}`")
        if route.kernels_missing:
            details.append("缺的 kernel：" + "、".join(route.kernels_missing))
        details += list(route.steps)
        if details:
            parts.append(f"**推荐路线 {route.title}**：\n" + bullets(details))
    if plan.kv_options:
        parts.append(
            table(
                ["KV cache dtype", "参数", "KV token", "可用", "说明"],
                [
                    [option.dtype, option.flag, option.kv_tokens, option.available, option.note]
                    for option in plan.kv_options
                ],
            )
        )
    if plan.numerics:
        parts.append(
            "**数值检查**：\n" + bullets(f"{item.title}：{item.detail}" for item in plan.numerics)
        )
    return parts


def _compile_sites(report: AdaptationReport) -> str | None:
    relevant = [item for item in report.compile_sites if item.applies in ("yes", "conditional")]
    if not relevant:
        return None
    return table(
        ["函数", "位置", "影响", "原因"],
        [
            [item.function, item.location, COMPILE_APPLIES_LABELS[item.applies], item.reason]
            for item in relevant
        ],
    )


def _step_block(index: int, step: BringupStep) -> str:
    parts = [f"#### {index}. {step.title}", f"**目标**：{step.goal}"]
    if step.blocked_by:
        parts.append("**先要解决**：\n" + bullets(step.blocked_by))
    if step.actions:
        parts.append("\n".join(f"{i}. {action}" for i, action in enumerate(step.actions, 1)))
    parts += [command_block(command) for command in step.commands]
    if step.exit_criteria:
        parts.append("**完成标准**：\n" + bullets(step.exit_criteria))
    return "\n\n".join(parts)


def render_adaptation(report: AdaptationReport, *, findings_limit: int | None = None) -> str:
    parts = [
        f"## 模型适配：{report.model} @ {report.hardware}",
        bullets(
            [
                f"架构：`{report.support.architecture}`",
                f"平台：{_platform_line(report)}",
                f"布局：{report.parallel.describe()}（{report.parallel.num_gpus} 卡）",
                f"SGLang 快照：{report.knowledge_commit or '未知'}",
                *report.profile.problems,
            ]
        ),
        "### 结论",
        bullets(_summary(report)),
        "### 改动原则",
        bullets(f"**{item.title}**：{item.detail}" for item in report.plan.principles),
    ]
    if report.profile.out_of_tree:
        parts += ["### 平台插件 hook 清单", _hooks_table(report)]
        if report.checklist.missing_from_snapshot:
            parts.append(
                "快照里找不到的 hook（上游可能改名或删除）："
                + "、".join(report.checklist.missing_from_snapshot)
            )
    parts += ["### 算子清单", _operators_table(report)]
    if report.inventory.shares_note:
        parts.append(f"> {report.inventory.shares_note}")
    if report.inventory.unknown_vendor_ops:
        parts.append(
            "硬件 YAML 里声明了、但 sgl-agent 不认识的厂商算子："
            + "、".join(report.inventory.unknown_vendor_ops)
        )
    parts += ["### 精度路线", *_precision_section(report)]
    compile_table = _compile_sites(report)
    if compile_table is not None:
        parts += ["### import 时编译的函数", compile_table]
    parts.append("### 上手步骤")
    parts += [_step_block(i, step) for i, step in enumerate(report.plan.steps, 1)]
    parts.append(findings_section(report.findings, title="发现", limit=findings_limit))
    return "\n\n".join(parts) + "\n"
