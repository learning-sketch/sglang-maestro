"""Profiler-trace analysis pipeline (scenario 2).

``analyze_trace`` loads SGLang profiler traces, builds per-step metrics and step groups, rebuilds
each group's batch shape for the cost model and runs every detector: operator/step anomalies,
fusion and compute-communication fusion, exposed communication, dual-stream MoE and L2 weight
prefetch. Without a model, a hardware spec or a parallel layout the comparisons that need them
are skipped and the report says so.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sgl_agent.findings import Finding, sort_findings
from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.knowledge import SGLangKnowledge, load_knowledge
from sgl_agent.knowledge.facts import source
from sgl_agent.model.config import ModelArch
from sgl_agent.model.ops import Calibration, ExecutionConfig, StepCostModel
from sgl_agent.model.parallel import ParallelConfig
from sgl_agent.perf import anomalies, fusion, overlap
from sgl_agent.perf.classify import KernelClassifier
from sgl_agent.perf.context import AnalysisContext
from sgl_agent.perf.expected import GroupExpectation, expect_shape, shape_from_step
from sgl_agent.perf.metrics import GroupSummary, compute_step_metrics, summarize_groups
from sgl_agent.perf.trace import ParsedTrace, TraceDialect, load_trace

DEFAULT_SERVER_URL = "http://127.0.0.1:30000"
_TP_INDEX_PATTERN = re.compile(r"TP(\d+)")
_DEVICE_NAME_NOISE = re.compile(r"[\s_\-]+")

SYNTHETIC_WARNING = (
    "这是 sgl-agent 生成的合成 trace（synthetic=true）：kernel 时间来自成本模型和人为注入的问题，"
    "只用于演示和回归测试，不代表真实硬件表现。"
)


@dataclass
class TraceReport:
    sources: list[str]
    ranks: list[str]
    device_names: dict[str, str]
    synthetic: bool
    kernel_count: int
    step_count: int
    model: str | None
    hardware: str | None
    parallel: str | None
    groups: list[GroupSummary]
    rank_comparison: list[dict[str, Any]]
    findings: list[Finding]
    warnings: list[str] = field(default_factory=list)
    profiling_tips: list[str] = field(default_factory=list)

    @property
    def primary_rank(self) -> str | None:
        return self.ranks[0] if self.ranks else None

    def primary_groups(self) -> list[GroupSummary]:
        return [group for group in self.groups if group.rank == self.primary_rank]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": self.sources,
            "ranks": self.ranks,
            "device_names": self.device_names,
            "synthetic": self.synthetic,
            "kernel_count": self.kernel_count,
            "step_count": self.step_count,
            "model": self.model,
            "hardware": self.hardware,
            "parallel": self.parallel,
            "groups": [group.to_dict() for group in self.groups],
            "rank_comparison": self.rank_comparison,
            "findings": [finding.to_dict() for finding in self.findings],
            "warnings": self.warnings,
            "profiling_tips": self.profiling_tips,
        }

    def to_markdown(self, *, findings_limit: int | None = None) -> str:
        from sgl_agent.perf.report import render_trace

        return render_trace(self, findings_limit=findings_limit)


def load_dialect(path: str | Path) -> TraceDialect:
    """Read a trace dialect override from a JSON or YAML file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: a trace dialect must be a mapping of field name -> names")
    return TraceDialect.from_dict(payload)


def infer_tp_from_ranks(ranks: list[str]) -> int | None:
    """TP size when the trace holds every TP rank ``0..n-1`` (n > 1); ``None`` when unsure."""
    indices: set[int] = set()
    for rank in ranks:
        match = _TP_INDEX_PATTERN.search(rank)
        if match is None:
            return None
        indices.add(int(match.group(1)))
    if len(indices) > 1 and indices == set(range(len(indices))):
        return len(indices)
    return None


def _normalized_device(name: str) -> str:
    return _DEVICE_NAME_NOISE.sub("", name).lower()


def _device_warnings(trace: ParsedTrace, hardware: HardwareSpec | None) -> list[str]:
    if hardware is None or not trace.device_names:
        return []
    expected = {
        _normalized_device(name)
        for name in (hardware.software.torch_device_name, hardware.name, *hardware.aliases)
        if name
    }
    mismatched = sorted(
        {
            name
            for name in trace.device_names.values()
            if name
            and not any(
                candidate in _normalized_device(name) or _normalized_device(name) in candidate
                for candidate in expected
            )
        }
    )
    if not mismatched:
        return []
    return [
        f"trace 采集自 {'、'.join(mismatched)}，与硬件描述 {hardware.name} 不一致：期望值和阈值按 "
        f"{hardware.name} 计算，结论可能不适用；请换成采集机器对应的 --hardware。"
    ]


def _resolve_parallel(
    trace: ParsedTrace,
    arch: ModelArch | None,
    hardware: HardwareSpec | None,
    parallel: ParallelConfig | None,
) -> tuple[ParallelConfig | None, list[str]]:
    if parallel is not None or arch is None or hardware is None:
        return parallel, []
    if trace.world_size:
        inferred, basis = (
            trace.world_size,
            f"distributedInfo.world_size={trace.world_size}（假设 PP=1）",
        )
    else:
        inferred, basis = infer_tp_from_ranks(trace.ranks), "trace 中的 rank 文件"
    if inferred is None:
        return None, [
            "没有给并行配置，trace 里也没有 world_size、rank 文件不足以推断 TP 大小：跳过与成本模型的对比。"
            "请用 --tp（以及 --attn-dp/--ep/--dcp 等）给出采集时的启动参数。"
        ]
    return ParallelConfig(tp=inferred), [
        f"没有给并行配置：按 {basis} 推断为 TP{inferred}（纯 TP、无 DP attention/EP）；"
        "如果采集时用了 --attn-dp-size/--ep-size 等，请显式传入，否则期望值会有偏差。"
    ]


def _rank_warnings(trace: ParsedTrace, parallel: ParallelConfig | None) -> list[str]:
    if parallel is None:
        return []
    warnings: list[str] = []
    if trace.world_size:
        if trace.world_size != parallel.num_gpus:
            warnings.append(
                f"trace 的 distributedInfo.world_size={trace.world_size}，与并行配置 {parallel.describe()} 的 "
                f"{parallel.num_gpus} 卡不一致：检查 --tp/--pp 是否与采集时的启动参数相同。"
            )
    elif len(trace.ranks) > parallel.num_gpus:
        warnings.append(
            f"trace 里有 {len(trace.ranks)} 个 rank，多于并行配置 {parallel.describe()} 的 "
            f"{parallel.num_gpus} 卡：检查 --tp/--pp 是否与采集时一致，或是否混入了别的 profile 目录。"
        )
    return warnings


def _expectation_key(group: GroupSummary) -> tuple:
    return (
        group.stage,
        group.batch_size,
        group.extend_tokens,
        tuple(sorted(group.annotations.items())),
    )


def _attach_expectations(
    context: AnalysisContext, cost_model: StepCostModel, context_len_hint: int | None
) -> None:
    """Cost-model expectation per group; ranks running the same shape share one estimate."""
    cache: dict[tuple, GroupExpectation | str] = {}
    for group in context.groups:
        key = _expectation_key(group)
        entry = cache.get(key)
        if entry is None:
            shape, shape_source = shape_from_step(
                group.stage,
                group.batch_size,
                group.extend_tokens,
                group.annotations,
                context_len_hint,
            )
            entry = (
                expect_shape(cost_model, shape, shape_source) if shape is not None else shape_source
            )
            cache[key] = entry
        if isinstance(entry, str):
            group.expected = {"unavailable": entry}
            continue
        context.expectations[id(group)] = entry
        group.expected = entry.to_dict()


def _unique_ids(findings: list[Finding]) -> list[Finding]:
    seen: Counter[str] = Counter()
    for finding in findings:
        seen[finding.id] += 1
        if seen[finding.id] > 1:
            finding.id = f"{finding.id}#{seen[finding.id]}"
    return findings


def run_detectors(context: AnalysisContext) -> tuple[list[Finding], list[dict[str, Any]]]:
    """Every trace detector in dependency order; returns sorted findings and the rank table."""
    outlier_findings, explained_steps = anomalies.detect_kernel_outliers(context)
    findings = list(outlier_findings)
    findings += anomalies.detect_layer_outliers(context)
    findings += anomalies.detect_step_jitter(context, explained_steps)
    rank_rows, rank_findings = anomalies.compare_ranks(context)
    findings += rank_findings
    eager_findings, eager_groups = anomalies.detect_eager_steps(context)
    sync_findings, sync_groups = anomalies.detect_host_sync(context)
    findings += eager_findings + sync_findings
    findings += anomalies.detect_idle_gaps(context, eager_groups | sync_groups)
    findings += anomalies.detect_inter_step_gaps(context)
    findings += anomalies.detect_bucket_deviation(context)
    fusion_findings, _ = fusion.detect_fusion_opportunities(context)
    findings += fusion_findings
    findings += overlap.detect_exposed_comm(context)
    dual_stream_findings, _ = overlap.detect_dual_stream(context)
    prefetch_findings, _ = overlap.detect_prefetch_opportunities(context)
    findings += dual_stream_findings + prefetch_findings
    return _unique_ids(sort_findings(findings)), rank_rows


def profiling_tips(
    trace: ParsedTrace,
    hardware: HardwareSpec | None,
    parallel: ParallelConfig | None,
    server_url: str = DEFAULT_SERVER_URL,
) -> list[str]:
    """How to collect a trace that every detector can use, given what this one lacks."""
    platform = hardware.software.platform if hardware else "cuda"
    if platform in {"cuda", "rocm", "npu"}:
        device_activity = "GPU"
    elif platform == "xpu":
        device_activity = "XPU"
    else:
        device_activity = "<平台的 profiler activity>"
    body = json.dumps(
        {
            "output_dir": "/tmp/sgl-profile",
            "num_steps": 20,
            "activities": ["CPU", device_activity],
            "detailed_annotations": True,
        },
        ensure_ascii=False,
    )
    tips = [
        f"压测进入稳态后采集：`curl -X POST {server_url.rstrip('/')}/start_profile "
        f"-H 'Content-Type: application/json' -d '{body}'`。num_steps 到了自动停止，每个 rank 写一个 "
        "`{profile_id}-TP-{rank}[-DP-..][-EP-..].trace.json.gz`（不传 output_dir 时写到 "
        "SGLANG_TORCH_PROFILER_DIR，默认 /tmp）；想分别拿 prefill 和 decode 各 N 步，再加 "
        '`"profile_by_stage": true`。参考 ' + source("profile_request") + "。",
    ]
    if trace.has_step_spans and not trace.detailed_annotations:
        tips.append(
            '这份 trace 的 step span 没有详细注解：加 `"detailed_annotations": true`，每个 step 会带 '
            "Σq/Σkv 等聚合量，期望值按真实 batch 形状计算；否则 decode 只能按 --context-len 估算，"
            "EXTEND 只能按 toks 估算。"
        )
    if trace.kernels and not trace.has_runtime_events:
        tips.append(
            '这份 trace 没有 CPU 侧运行时事件：activities 里要包含 "CPU"。kernel 靠 correlation id 归到 step，'
            "kernel 发射/graph 发射/同步调用次数也来自 CPU 事件，缺了它“没走 CUDA graph”和“主机同步”两类检测会失效。"
        )
    if len(trace.ranks) == 1 and parallel is not None and parallel.num_gpus > 1:
        tips.append(
            f"只加载了 1 个 rank（并行配置共 {parallel.num_gpus} 卡）：把整个 profile 目录传进来，"
            "才能比较各 rank 的计算时间——慢的 rank 会让其它 rank 在集合通信里空等。"
        )
    if platform == "oot":
        tips.append(
            "out-of-tree 平台上 SGLang 的 profiler 会调用 `current_platform.get_torch_profiler_activity_str()`，"
            "DeviceMixin 的默认实现直接抛 NotImplementedError：平台插件必须实现它（返回 "
            "torch.profiler.ProfilerActivity 的成员名）以及 `get_torch_profiler_activity()`，/start_profile 才能用；"
            "activities 里填这个名字。参考 " + source("oot_profiler_activity") + "。"
        )
    if platform == "npu":
        tips.append(
            'NPU 上 activities 里的 "GPU" 会被 SGLang 映射到 torch_npu 的 NPU activity，trace 由 torch_npu 的 '
            "tensorboard_trace_handler 导出，目录结构和文件名与 `*-TP-<rank>.trace.json.gz` 不同："
            "把导出目录中的 Chrome trace JSON 文件直接传进来。参考 "
            + source("npu_profiler_export")
            + "。"
        )
    if platform != "cuda":
        tips.append(
            "非 CUDA 平台的 profiler 事件类别和运行时 API 名可能不同（例如 hipLaunchKernel、厂商 runtime 的发射函数）："
            "用 --dialect 指定一个 YAML/JSON，覆盖 kernel_categories、runtime_categories、kernel_launch_names、"
            "graph_launch_names、sync_names 等字段，否则 kernel 归属和发射统计会缺失。"
        )
    tips.append(
        "每改一项建议，用相同负载和 num_steps 再采一次，两份报告按 finding id 对比 gain，确认收益后再改下一项。"
    )
    return tips


def analyze_parsed_trace(
    trace: ParsedTrace,
    *,
    knowledge: SGLangKnowledge | None = None,
    hardware: HardwareSpec | None = None,
    arch: ModelArch | None = None,
    parallel: ParallelConfig | None = None,
    execution: ExecutionConfig | None = None,
    calibration: Calibration | None = None,
    context_len_hint: int | None = None,
    dialect: TraceDialect | None = None,
    server_url: str = DEFAULT_SERVER_URL,
) -> TraceReport:
    knowledge = knowledge or load_knowledge()
    dialect = dialect or TraceDialect()
    warnings = list(trace.warnings)
    if trace.synthetic:
        warnings.insert(0, SYNTHETIC_WARNING)
    parallel, parallel_notes = _resolve_parallel(trace, arch, hardware, parallel)
    warnings += parallel_notes
    cost_model: StepCostModel | None = None
    if arch is not None and hardware is not None and parallel is not None:
        errors = parallel.validate(arch)
        if errors:
            raise ValueError(
                f"并行配置 {parallel.describe()} 不适用于 {arch.name}：" + "；".join(errors)
            )
        cost_model = StepCostModel(
            arch, hardware, parallel, execution or ExecutionConfig(), calibration or Calibration()
        )
    else:
        missing = [
            label for label, value in (("--model", arch), ("--hardware", hardware)) if value is None
        ]
        if missing:
            warnings.append(
                f"没有提供 {' 和 '.join(missing)}：不与成本模型对比，期望值、偏差检测以及部分融合/重叠的收益估算会跳过。"
            )
    warnings += _device_warnings(trace, hardware)
    warnings += _rank_warnings(trace, parallel)

    kernels_by_step = trace.kernels_by_step()
    for kernels in kernels_by_step.values():
        kernels.sort(key=lambda kernel: kernel.start_us)
    metrics_by_step = {
        step.step_id: compute_step_metrics(step, kernels_by_step.get(step.step_id, []), dialect)
        for step in trace.steps
    }
    groups = summarize_groups(trace, metrics_by_step, kernels_by_step)
    findings: list[Finding] = []
    rank_rows: list[dict[str, Any]] = []
    if not trace.kernels:
        warnings.append(
            "trace 中没有 GPU kernel 事件：确认采集时 activities 包含设备侧活动，且采集期间有请求在跑。"
        )
    elif not groups:
        warnings.append(
            "trace 中的 kernel 都没有归属到任何 step：确认 trace 未被截断，且同时采集了 CPU 与设备活动。"
        )
    else:
        context = AnalysisContext(
            trace=trace,
            groups=groups,
            kernels_by_step=kernels_by_step,
            metrics_by_step=metrics_by_step,
            knowledge=knowledge,
            dialect=dialect,
            hardware=hardware,
            arch=arch,
            parallel=parallel,
            cost_model=cost_model,
        )
        if cost_model is not None:
            _attach_expectations(context, cost_model, context_len_hint)
        findings, rank_rows = run_detectors(context)

    return TraceReport(
        sources=list(trace.sources),
        ranks=list(trace.ranks),
        device_names=dict(trace.device_names),
        synthetic=trace.synthetic,
        kernel_count=len(trace.kernels),
        step_count=sum(1 for step in trace.steps if step.source != "none"),
        model=arch.name if arch else None,
        hardware=hardware.name if hardware else None,
        parallel=parallel.describe() if parallel else None,
        groups=groups,
        rank_comparison=rank_rows,
        findings=findings,
        warnings=warnings,
        profiling_tips=profiling_tips(trace, hardware, parallel, server_url),
    )


def analyze_trace(
    paths: str | Path | list[str | Path],
    *,
    knowledge: SGLangKnowledge | None = None,
    hardware: HardwareSpec | None = None,
    arch: ModelArch | None = None,
    parallel: ParallelConfig | None = None,
    execution: ExecutionConfig | None = None,
    calibration: Calibration | None = None,
    context_len_hint: int | None = None,
    dialect: TraceDialect | None = None,
    classifier: KernelClassifier | None = None,
    server_url: str = DEFAULT_SERVER_URL,
) -> TraceReport:
    """Load trace files (or a profile directory) and analyze them; see the module docstring."""
    trace = load_trace(paths, classifier=classifier, dialect=dialect)
    return analyze_parsed_trace(
        trace,
        knowledge=knowledge,
        hardware=hardware,
        arch=arch,
        parallel=parallel,
        execution=execution,
        calibration=calibration,
        context_len_hint=context_len_hint,
        dialect=dialect,
        server_url=server_url,
    )
