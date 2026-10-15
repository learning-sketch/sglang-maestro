"""Scenario 1 entry point: how one model adapts to one device, as a report plus ranked findings.

The report answers, for this model, parallel layout and software stack: does SGLang implement
the architecture, which platform hooks the plugin needs, which kernels exist and which fall back
or break, which precision route fits in how many GPUs, and in which order to bring it up.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from sgl_agent.adapt.bringup import BringupPlan, bringup_plan
from sgl_agent.adapt.operators import (
    DECODE_SHAPE,
    GAP_STATUSES,
    OperatorInventory,
    operator_inventory,
)
from sgl_agent.adapt.platform import (
    CompileSiteImpact,
    HookChecklist,
    PlatformProfile,
    compile_site_impact,
    hook_checklist,
    resolve_platform,
)
from sgl_agent.adapt.precision import (
    DEFAULT_MIN_KV_TOKENS,
    PrecisionPlan,
    precision_plan,
)
from sgl_agent.findings import Finding, severity_for_share, sort_findings
from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.knowledge import SGLangKnowledge
from sgl_agent.knowledge.facts import source
from sgl_agent.model.config import ModelArch
from sgl_agent.model.ops import BatchShape, StepCostModel
from sgl_agent.model.parallel import ParallelConfig

GUARDED_ARCHITECTURE_PREFIXES = {"DeepseekV4": "the DeepSeek-V4 KV pool"}
"""Architectures whose default KV pool SGLang rejects on out-of-tree platforms."""


@dataclass(frozen=True)
class ArchitectureSupport:
    architecture: str
    supported: bool
    model_file: str | None


@dataclass
class AdaptationReport:
    model: str
    hardware: str
    knowledge_commit: str | None
    support: ArchitectureSupport
    profile: PlatformProfile
    parallel: ParallelConfig
    checklist: HookChecklist
    inventory: OperatorInventory
    compile_sites: list[CompileSiteImpact]
    precision: PrecisionPlan
    plan: BringupPlan
    guards: list[dict[str, Any]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "hardware": self.hardware,
            "knowledge_commit": self.knowledge_commit,
            "support": asdict(self.support),
            "profile": asdict(self.profile),
            "parallel": self.parallel.describe(),
            "hooks": [asdict(item) for item in self.checklist.requirements],
            "hooks_missing_from_snapshot": self.checklist.missing_from_snapshot,
            "operators": [
                {**asdict(item), "status_label": item.status_label}
                for item in self.inventory.requirements
            ],
            "operator_shares_note": self.inventory.shares_note,
            "compile_sites": [asdict(item) for item in self.compile_sites],
            "precision": asdict(self.precision),
            "principles": [asdict(item) for item in self.plan.principles],
            "steps": [
                {**asdict(step), "commands": [command.to_dict() for command in step.commands]}
                for step in self.plan.steps
            ],
            "guards": self.guards,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_markdown(self, *, findings_limit: int | None = None) -> str:
        from sgl_agent.adapt.report import render_adaptation

        return render_adaptation(self, findings_limit=findings_limit)


def _default_parallel(
    arch: ModelArch, hardware: HardwareSpec, precision: PrecisionPlan
) -> ParallelConfig:
    if precision.recommended is not None and precision.recommended.sizing is not None:
        return precision.recommended.sizing.parallel
    for tp in (hardware.devices_per_node, 4, 2, 1):
        candidate = ParallelConfig(tp=tp)
        if not candidate.validate(arch):
            return candidate
    return ParallelConfig(tp=1)


def _operator_findings(inventory: OperatorInventory) -> list[Finding]:
    findings = []
    for item in inventory.requirements:
        if item.status not in GAP_STATUSES:
            continue
        if item.status in ("broken", "missing"):
            severity = "high"
        elif item.status == "slow":
            severity = "high" if item.max_share >= 0.10 else "medium"
        else:
            severity = severity_for_share(item.max_share)
        evidence = [f"当前路径：{item.resolved_path}", item.impact]
        if item.share_decode is not None or item.share_prefill is not None:
            evidence.append(
                f"代价模型按理想 kernel 估算的耗时占比：decode {item.share_decode or 0:.1%}，"
                f"prefill {item.share_prefill or 0:.1%}"
            )
        recommendation = [item.how]
        if item.contract:
            recommendation.append(f"契约：{item.contract}")
        findings.append(
            Finding(
                id=f"adapt.op.{item.key}",
                kind="adaptation",
                severity=severity,
                title=f"{item.title}：{item.status_label}",
                evidence=evidence,
                recommendation=recommendation,
                gain_note=inventory.shares_note,
                scope=item.category,
                references=list(item.references),
            )
        )
    return findings


def _hook_findings(checklist: HookChecklist) -> list[Finding]:
    findings = []
    for level, severity in (("required", "high"), ("conditional", "medium")):
        items = checklist.by_level(level)
        if not items:
            continue
        findings.append(
            Finding(
                id=f"adapt.hooks.{level}",
                kind="adaptation",
                severity=severity,
                title=(
                    f"平台插件必须实现 {len(items)} 项"
                    if level == "required"
                    else f"本配置还需要 {len(items)} 项 hook"
                ),
                evidence=[f"{item.name}：{item.reason}" for item in items],
                recommendation=[f"{item.name}：{item.how}" for item in items],
                scope="platform",
                references=sorted({ref for item in items for ref in item.references}),
            )
        )
    if checklist.missing_from_snapshot:
        findings.append(
            Finding(
                id="adapt.hooks.snapshot_drift",
                kind="adaptation",
                severity="medium",
                title="规则引用的 hook 不在当前 SGLang 快照里",
                evidence=[", ".join(sorted(set(checklist.missing_from_snapshot)))],
                recommendation=[
                    "上游可能改名或删除了这些 hook：对照 platforms/interface.py 更新插件与 sgl-agent 规则"
                ],
                scope="platform",
                references=[source("platform_interface")],
            )
        )
    return findings


def _compile_findings(sites: list[CompileSiteImpact]) -> list[Finding]:
    manual = [site for site in sites if site.applies == "manual"]
    if not manual:
        return []
    return [
        Finding(
            id="adapt.compile.manual",
            kind="adaptation",
            severity="medium",
            title=f"{len(manual)} 个 import 时编译的函数无法自动判断是否在本模型路径上",
            evidence=[f"{site.function}（{site.location}）：{site.reason}" for site in manual],
            recommendation=[
                "确认这些函数是否被本模型调用；调用到的函数依赖 get_compile_backend() 返回的后端"
            ],
            scope="platform",
            references=[source("torch_compile_sites"), source("oot_compile_backend")],
        )
    ]


def _precision_findings(
    arch: ModelArch, hardware: HardwareSpec, precision: PrecisionPlan
) -> list[Finding]:
    findings = []
    recommended = precision.recommended
    table = []
    for route in precision.routes:
        sizing = route.sizing
        where = f"{route.min_gpus} 卡" if sizing else "放不下"
        status = "可用" if route.available else "设备不支持"
        missing = f"，缺 kernel：{'/'.join(route.kernels_missing)}" if route.kernels_missing else ""
        blockers = f"，{'；'.join(route.blockers)}" if route.blockers else ""
        table.append(f"{route.title}：{where}，{status}{missing}{blockers}")
    if recommended is None:
        findings.append(
            Finding(
                id="adapt.precision.none",
                kind="adaptation",
                severity="high",
                title="没有能在可用卡数内放下的精度路线",
                evidence=table,
                recommendation=["增大 --max-gpus 或补齐量化 kernel；也可降低最小 KV 容量要求"],
                scope="precision",
            )
        )
        return findings
    sizing = recommended.sizing
    flags = dict(recommended.flags)
    if sizing is not None:
        flags.update(sizing.flags)
    checkpoint = next((route for route in precision.routes if route.key == "checkpoint"), None)
    severity = "info" if recommended.key == "checkpoint" else "medium"
    if checkpoint is not None and not checkpoint.available:
        severity = "high"
    recommendation = [*recommended.steps]
    if sizing is not None and sizing.chunked_prefill_size:
        recommendation.append(
            f"默认 chunked prefill 预留的激活显存会把 KV 池压到 {precision.min_kv_tokens} token 以下；"
            f"--chunked-prefill-size {sizing.chunked_prefill_size} 后 KV 池约 {sizing.kv_tokens:,} token"
        )
    findings.append(
        Finding(
            id="adapt.precision.route",
            kind="adaptation",
            severity=severity,
            title=f"推荐精度路线：{recommended.title}（{recommended.min_gpus} 卡）",
            evidence=[f"checkpoint：{precision.checkpoint}", *table],
            recommendation=recommendation,
            flags=flags,
            scope="precision",
            references=list(recommended.references),
        )
    )
    fp8_kv = next((option for option in precision.kv_options if option.flag), None)
    default_kv = next((option for option in precision.kv_options if not option.flag), None)
    if fp8_kv is not None and default_kv is not None and fp8_kv.available:
        findings.append(
            Finding(
                id="adapt.precision.kv_fp8",
                kind="adaptation",
                severity="low",
                title=f"FP8 KV cache 可把 KV 池从 {default_kv.kv_tokens:,} 提到 {fp8_kv.kv_tokens:,} token",
                evidence=[fp8_kv.note],
                recommendation=["长上下文或高并发受 KV 容量限制时再打开，并评测长上下文精度"],
                flags={"--kv-cache-dtype": fp8_kv.flag},
                scope="precision",
            )
        )
    return findings


def _graph_finding(
    arch: ModelArch, hardware: HardwareSpec, profile: PlatformProfile, parallel: ParallelConfig
) -> Finding | None:
    if profile.in_tree or hardware.software.graph_capture:
        return None
    try:
        model = StepCostModel(arch, hardware, parallel)
        shape = BatchShape.decode(*DECODE_SHAPE)
        eager = model.estimate(shape, cuda_graph=False).total_us
        graph = model.estimate(shape, cuda_graph=True).total_us
        single = BatchShape.decode(1, DECODE_SHAPE[1])
        eager_single = model.estimate(single, cuda_graph=False).total_us
        graph_single = model.estimate(single, cuda_graph=True).total_us
    except (ValueError, ZeroDivisionError):
        return None
    gain_ms = (eager - graph) / 1000
    share = (eager - graph) / eager if eager else 0.0
    return Finding(
        id="adapt.graph.eager",
        kind="adaptation",
        severity=severity_for_share(share),
        title=f"设备不支持图捕获，decode 每步多付约 {gain_ms:.2f} ms launch 开销",
        evidence=[
            f"代价模型（kernel_launch_us={hardware.latency.kernel_launch_us}）：batch {DECODE_SHAPE[0]} 每步 "
            f"eager {eager / 1000:.2f} ms vs 图 {graph / 1000:.2f} ms；batch 1 eager {eager_single / 1000:.2f} ms "
            f"vs 图 {graph_single / 1000:.2f} ms",
        ],
        recommendation=[
            "支持图捕获后设置 capabilities.graph_capture=True 并实现 get_graph_runner_cls()",
            "在此之前减少每层 kernel 数：优先实现 fused add+RMSNorm、门控激活、RoPE+KV 写入这类融合算子",
        ],
        gain_ms_per_step=gain_ms,
        gain_note="按代价模型的 launch 延迟假设估算，需要用实测 trace 校准",
        scope="decode",
        references=[source("graph_capture_capability"), source("graph_runner_required")],
    )


def _guard_findings(
    arch: ModelArch, profile: PlatformProfile, guards: list[dict[str, Any]]
) -> list[Finding]:
    if profile.in_tree or not guards:
        return []
    findings = []
    for prefix, subsystem in GUARDED_ARCHITECTURE_PREFIXES.items():
        if not arch.primary_architecture.startswith(prefix):
            continue
        hit = next((guard for guard in guards if guard.get("subsystem") == subsystem), None)
        if hit is not None:
            findings.append(
                Finding(
                    id="adapt.guard.model",
                    kind="adaptation",
                    severity="high",
                    title=f"SGLang 在 out-of-tree 平台上拒绝 {subsystem}",
                    evidence=[f"{hit.get('location')}：reject_out_of_tree_path"],
                    recommendation=[
                        "需要上游为该 KV 池增加平台接缝，或在插件的 get_kv_pool_cls 里提供实现后推动放开该检查"
                    ],
                    scope="platform",
                    references=[source("oot_guards")],
                )
            )
    rejected = [guard for guard in guards if guard.get("kind") == "rejects"]
    if rejected:
        findings.append(
            Finding(
                id="adapt.guard.features",
                kind="adaptation",
                severity="info",
                title=f"{len(rejected)} 个子系统在 out-of-tree 平台上会直接报错",
                evidence=[
                    f"{guard.get('subsystem')}（{guard.get('location')}）" for guard in rejected
                ],
                recommendation=["部署时不要打开这些特性；特性建议（场景 3）会跳过它们"],
                scope="platform",
                references=[source("oot_guards")],
            )
        )
    return findings


def _misc_findings(
    arch: ModelArch, inventory: OperatorInventory, profile: PlatformProfile
) -> list[Finding]:
    findings = []
    for index, problem in enumerate(profile.problems):
        findings.append(
            Finding(
                id=f"adapt.platform.{index}",
                kind="adaptation",
                severity="high"
                if profile.dispatch_key_collides and "dispatch_key" in problem
                else "medium",
                title="平台配置问题",
                evidence=[problem],
                recommendation=["按提示修正硬件 YAML 的 software 部分或插件实现"],
                scope="platform",
                references=[source("platform_interface")],
            )
        )
    if inventory.unknown_vendor_ops:
        findings.append(
            Finding(
                id="adapt.yaml.unknown_ops",
                kind="adaptation",
                severity="low",
                title="硬件 YAML 的 fused_ops 有无法识别的名字",
                evidence=[", ".join(inventory.unknown_vendor_ops)],
                recommendation=["检查拼写：可用 sgl-agent 的算子名或 SGLang 的 BaseFusedOp 类名"],
                scope="platform",
            )
        )
    if arch.warnings:
        findings.append(
            Finding(
                id="adapt.model.warnings",
                kind="adaptation",
                severity="info",
                title="模型配置解析提示",
                evidence=list(arch.warnings),
                scope="model",
            )
        )
    return findings


def _support_finding(support: ArchitectureSupport) -> Finding | None:
    if support.supported:
        return None
    return Finding(
        id="adapt.architecture.unsupported",
        kind="adaptation",
        severity="high",
        title=f"SGLang 没有 {support.architecture} 的实现",
        evidence=["能力快照的 model_architectures 中没有该架构"],
        recommendation=[
            "先用 --model-impl transformers 跑通（Transformers 后端），再在 SGLANG_EXTERNAL_MODEL_PACKAGE "
            "指向的包里实现 SGLang 版本",
        ],
        flags={"--model-impl": "transformers"},
        scope="model",
        references=[source("transformers_fallback"), source("external_model_package")],
    )


def analyze_adaptation(
    arch: ModelArch,
    hardware: HardwareSpec,
    knowledge: SGLangKnowledge,
    *,
    parallel: ParallelConfig | None = None,
    model_path: str | None = None,
    hardware_ref: str | None = None,
    plugin_name: str | None = None,
    prefer: str = "accuracy",
    min_kv_tokens: int = DEFAULT_MIN_KV_TOKENS,
    max_gpus: int | None = None,
) -> AdaptationReport:
    """Scenario 1 for ``arch`` on ``hardware``; ``parallel`` defaults to the recommended route's layout."""
    profile = resolve_platform(hardware, knowledge)
    precision = precision_plan(
        arch, hardware, profile, min_kv_tokens=min_kv_tokens, max_gpus=max_gpus, prefer=prefer
    )
    if parallel is None:
        parallel = _default_parallel(arch, hardware, precision)
    else:
        errors = parallel.validate(arch)
        if errors:
            raise ValueError("；".join(errors))
    sites = compile_site_impact(arch, profile, knowledge, tp=parallel.tp)
    checklist = hook_checklist(
        arch, hardware, profile, knowledge, num_gpus=parallel.num_gpus, compile_sites=sites
    )
    inventory = operator_inventory(arch, hardware, profile, knowledge, parallel)
    support = ArchitectureSupport(
        architecture=arch.primary_architecture,
        supported=knowledge.is_architecture_supported(arch.primary_architecture),
        model_file=knowledge.model_file_for(arch.primary_architecture),
    )
    compile_required = any(
        item.name == "get_compile_backend" and item.level == "required"
        for item in checklist.requirements
    )
    plan = bringup_plan(
        arch,
        hardware,
        profile,
        knowledge,
        checklist=checklist,
        inventory=inventory,
        precision=precision,
        parallel=parallel,
        model_path=model_path,
        hardware_ref=hardware_ref,
        plugin_name=plugin_name,
        compile_required=compile_required,
    )
    guards = knowledge.oot_guards if profile.out_of_tree else []
    findings: list[Finding] = []
    support_finding = _support_finding(support)
    if support_finding is not None:
        findings.append(support_finding)
    findings += _misc_findings(arch, inventory, profile)
    findings += _operator_findings(inventory)
    findings += _hook_findings(checklist)
    findings += _compile_findings(sites)
    findings += _precision_findings(arch, hardware, precision)
    graph = _graph_finding(arch, hardware, profile, parallel)
    if graph is not None:
        findings.append(graph)
    findings += _guard_findings(arch, profile, guards)
    for problem in plan.command_problems:
        findings.append(
            Finding(
                id="adapt.plan.flags",
                kind="adaptation",
                severity="medium",
                title="生成的命令有参数问题",
                evidence=[problem],
                recommendation=["sgl-agent 的规则与当前 SGLang 快照不一致，请更新快照或规则"],
                scope="plan",
            )
        )
    return AdaptationReport(
        model=arch.name,
        hardware=hardware.name,
        knowledge_commit=knowledge.commit,
        support=support,
        profile=profile,
        parallel=parallel,
        checklist=checklist,
        inventory=inventory,
        compile_sites=sites,
        precision=precision,
        plan=plan,
        guards=guards,
        findings=sort_findings(findings),
    )
