"""Markdown rendering of scenario 2 reports: roofline estimates, profiler traces and server logs."""

from __future__ import annotations

from typing import Any

from sglang_maestro.markdown import bullets, findings_section, table
from sglang_maestro.perf.analyze import TraceReport
from sglang_maestro.perf.expected import BUCKETS, bucket_of, measured_buckets
from sglang_maestro.perf.logs import LogReport
from sglang_maestro.perf.metrics import GroupSummary
from sglang_maestro.perf.roofline import RooflineReport, ScenarioResult

BOUND_LABELS = {"compute": "算力", "memory": "带宽", "comm": "通信", "launch": "发射"}
BUCKET_LABELS = {
    "attention": "attention",
    "gemm": "GEMM",
    "moe": "MoE",
    "comm": "通信",
    "small_ops": "小算子（norm/激活/RoPE/KV 写入/逐元素）",
}
TOP_KERNELS = 8
MAX_DECODE_SAMPLES = 8


def _ms(us: float | None) -> float | None:
    return None if us is None else us / 1000


# ---------------------------------------------------------------------- roofline
def _scenario_row(scenario: ScenarioResult) -> list[Any]:
    kinds = sum(scenario.by_kind_ms.values()) or 1.0
    bounds_total = sum(scenario.by_bound_ms.values()) or 1.0
    kind, kind_ms = max(scenario.by_kind_ms.items(), key=lambda item: item[1], default=("—", 0))
    bounds = "，".join(
        f"{BOUND_LABELS.get(bound, bound)} {value / bounds_total:.0%}"
        for bound, value in sorted(scenario.by_bound_ms.items(), key=lambda item: -item[1])
        if value > 0
    )
    return [
        scenario.label,
        scenario.total_ms,
        scenario.tokens_per_s_per_gpu,
        f"{kind}（{kind_ms / kinds:.0%}）",
        bounds,
    ]


def _ops_table(scenario: ScenarioResult) -> str:
    return table(
        ["算子", "类别", "受限于", "ms", "占比", "TFLOP", "GB", "算术强度"],
        [
            [
                row.name,
                row.kind,
                BOUND_LABELS.get(row.bound, row.bound),
                row.time_ms,
                f"{row.share:.1%}",
                row.tflop,
                row.gbytes,
                row.arithmetic_intensity,
            ]
            for row in scenario.rows
        ],
    )


def render_roofline(report: RooflineReport) -> str:
    memory = report.memory
    ridge = "，".join(f"{dtype.upper()} {value:g}" for dtype, value in report.ridge_points.items())
    parts = [
        f"## 算子性能（roofline）：{report.model} @ {report.hardware}",
        bullets(
            [
                f"布局：{report.parallel}；线性层按 {report.linear_compute_dtype} 计算"
                + (
                    f"，专家按 {report.expert_compute_dtype}"
                    if report.expert_compute_dtype != report.linear_compute_dtype
                    else ""
                ),
                f"屋脊点（FLOPs/Byte）：{ridge}",
                f"decode GEMM 每卡超过约 {report.decode_gemm_compute_bound_tokens:.0f} 个 token 转为算力受限"
                + (
                    f"；MoE 专家 GEMM 全局约 {report.moe_compute_bound_global_tokens:,.0f} token"
                    if report.moe_compute_bound_global_tokens
                    else ""
                ),
                f"KV 池：每个 DP rank {report.max_total_tokens:,} token",
            ]
        ),
        "### 结论",
        bullets(report.insights),
        "### 场景",
        table(
            ["场景", "单步 ms", "每卡 token/s", "最大头", "受限时间构成"],
            [_scenario_row(scenario) for scenario in report.scenarios],
        ),
    ]
    for scenario in report.scenarios:
        parts += [f"#### {scenario.label}：最耗时的算子", _ops_table(scenario)]
    if report.batch_sweep:
        parts += [
            "### decode batch 扫描",
            table(
                [
                    "每 rank batch",
                    "全局 batch",
                    "单步 ms",
                    "每卡 token/s",
                    "每 rank KV 需求",
                    "放得下",
                ],
                [
                    [
                        int(row["batch_per_dp_rank"]),
                        int(row["global_batch"]),
                        row["step_ms"],
                        row["tokens_per_s_per_gpu"],
                        int(row["kv_tokens_needed_per_rank"]),
                        bool(row["fits_kv"]),
                    ]
                    for row in report.batch_sweep
                ],
            ),
        ]
    if memory:
        lines = [
            f"每卡权重 {memory['weights_gb_per_gpu']:.1f} GB，KV 每 token 每卡 "
            f"{memory['kv_bytes_per_token_per_gpu']:,.0f} B，KV 池每卡 {memory['kv_pool_gb_per_gpu']:.1f} GB",
            f"mem_fraction_static {memory['mem_fraction_static']:g}"
            + (
                "（SGLang 按显存自动选的默认值）"
                if memory.get("mem_fraction_is_sglang_default")
                else ""
            ),
            f"chunked prefill {memory['chunked_prefill_size']}，decode CUDA graph 最大 batch "
            f"{memory['cuda_graph_max_bs_decode']}",
        ]
        reserve = memory.get("reserve_mb") or {}
        if reserve:
            lines.append(
                "预留显存（MB）："
                + "，".join(f"{key} {value:,.0f}" for key, value in reserve.items() if value)
            )
        for option in memory.get("higher_fraction_alternatives") or []:
            lines.append(
                f"`--mem-fraction-static {option['mem_fraction_static']}` + chunked prefill "
                f"{option['chunked_prefill_size']} + graph max bs {option['cuda_graph_max_bs_decode']}："
                f"KV 池 {option['max_total_tokens']:,} token"
            )
        parts += ["### 显存", bullets(lines)]
    parts.append(
        "> 时间来自逐算子 roofline 成本模型（硬件 YAML 的峰值 × 效率系数），用于找瓶颈和比较方案；"
        "绝对值以 profiler trace 和 `sglang-maestro calibrate` 校准后的结果为准。"
    )
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------- trace
def _group_row(group: GroupSummary) -> list[Any]:
    expected = group.expected or {}
    expected_us = expected.get("total_us")
    measured_us = group.span.mean
    return [
        group.label,
        group.steps,
        _ms(measured_us),
        f"{1 - group.idle_ratio:.0%}",
        group.kernels_per_step,
        group.kernel_launches_per_step,
        f"{group.graph_fraction:.0%}" if group.graph_fraction is not None else None,
        group.streams,
        _ms(group.comm_us),
        _ms(group.exposed_comm_us),
        _ms(expected_us),
        f"{measured_us / expected_us:.2f}" if expected_us else None,
    ]


def _breakdown(group: GroupSummary) -> str:
    expected = (group.expected or {}).get("buckets_us") or {}
    measured = measured_buckets(group.by_category_us)
    rows = []
    for bucket in BUCKETS:
        got, want = measured.get(bucket, 0.0), expected.get(bucket)
        if not got and not want:
            continue
        rows.append(
            [BUCKET_LABELS[bucket], _ms(got), _ms(want), f"{got / want:.2f}" if want else None]
        )
    other = sum(
        value for category, value in group.by_category_us.items() if bucket_of(category) is None
    )
    if other:
        rows.append(["其它（memcpy 等）", _ms(other), None, None])
    return table(["类别", "实测 ms/step", "模型预期 ms/step", "实测/预期"], rows)


def _kernels(group: GroupSummary) -> str:
    return table(
        ["kernel", "类别", "次数/step", "ms/step", "占比", "单次中位 us", "单次 P99 us"],
        [
            [
                kernel.short_name,
                kernel.category,
                kernel.count_per_step,
                _ms(kernel.time_per_step_us),
                f"{kernel.share:.1%}",
                kernel.durations.median,
                kernel.durations.p99,
            ]
            for kernel in group.top_kernels[:TOP_KERNELS]
        ],
    )


def render_trace(report: TraceReport, *, findings_limit: int | None = None) -> str:
    devices = "，".join(f"{rank} {name}" for rank, name in report.device_names.items())
    parts = [
        "## Profiler trace 分析" + (f"：{report.model}" if report.model else ""),
        bullets(
            [
                f"输入：{len(report.sources)} 个文件，rank {', '.join(report.ranks)}"
                + (f"（{devices}）" if devices else ""),
                f"{report.step_count} 个 step、{report.kernel_count:,} 个 kernel",
                f"成本模型对比：{report.hardware} · {report.parallel}"
                if report.hardware and report.parallel
                else "没有指定硬件/布局：不和成本模型对比",
            ]
        ),
    ]
    if report.warnings:
        parts.append("⚠ " + "\n⚠ ".join(report.warnings))
    groups = report.primary_groups()
    if groups:
        parts += [
            f"### step 分组（rank {report.primary_rank}）",
            table(
                [
                    "分组",
                    "step 数",
                    "平均 step ms",
                    "GPU 忙碌",
                    "kernel/step",
                    "发射/step",
                    "CUDA graph",
                    "stream 数",
                    "通信 ms",
                    "未掩盖通信 ms",
                    "模型预期 ms",
                    "实测/预期",
                ],
                [_group_row(group) for group in groups],
            ),
        ]

        def weight(group: GroupSummary) -> float:
            return group.span.mean * group.steps

        shown = [max(groups, key=weight)]
        decode = [group for group in groups if group.is_decode_like]
        if decode and max(decode, key=weight) is not shown[0]:
            shown.append(max(decode, key=weight))
        for group in shown:
            parts += [
                f"#### {group.label}：耗时构成",
                _breakdown(group),
                f"#### {group.label}：最耗时的 kernel",
                _kernels(group),
            ]
    if report.rank_comparison:
        parts += [
            "### rank 对比",
            table(
                ["阶段", "rank", "step 数", "计算 ms", "通信 ms", "step ms", "空闲 ms"],
                [
                    [
                        row["phase"],
                        row["rank"],
                        row["steps"],
                        _ms(row["compute_us_median"]),
                        _ms(row["comm_us_median"]),
                        _ms(row["span_us_median"]),
                        _ms(row["idle_us_median"]),
                    ]
                    for row in report.rank_comparison
                ],
            ),
        ]
    parts.append(
        findings_section(
            report.findings, title="发现（融合、通算重叠、多流、预取、异常）", limit=findings_limit
        )
    )
    if report.profiling_tips:
        parts += ["### 采集建议", bullets(report.profiling_tips)]
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------- logs
def _stat(label: str, summary: dict[str, Any] | None, unit: str = "") -> str | None:
    if not summary or not summary.get("count"):
        return None
    return (
        f"{label}：中位 {summary['median']:,.4g}{unit}，P90 {summary['p90']:,.4g}{unit}，"
        f"最大 {summary['max']:,.4g}{unit}"
    )


def render_logs(report: LogReport, *, findings_limit: int | None = None) -> str:
    parts = ["## Server 日志分析", bullets([f"输入：{', '.join(report.sources)}"])]
    if report.warnings:
        parts.append("⚠ " + "\n⚠ ".join(report.warnings))
    if report.startup:
        startup = report.startup
        parts += [
            "### 启动参数（日志里的实际值）",
            table(
                [
                    "max_total_num_tokens",
                    "chunked_prefill_size",
                    "max_prefill_tokens",
                    "max_running_requests",
                    "context_len",
                    "剩余显存 GB",
                ],
                [
                    [
                        startup.get("max_total_num_tokens"),
                        startup.get("chunked_prefill_size"),
                        startup.get("max_prefill_tokens"),
                        startup.get("max_running_requests"),
                        startup.get("context_len"),
                        startup.get("available_mem_gb"),
                    ]
                ],
            ),
        ]
    facts = []
    if report.weight_load:
        load = report.weight_load
        facts.append(
            f"权重加载 {load.get('elapsed_s')} s，每卡 {load.get('weights_gb')} GB"
            + (f"（{load.get('quantization')}）" if load.get("quantization") else "")
        )
    for item in report.kv_cache:
        size = item.get("kv_gb") or (item.get("k_gb") or 0) + (item.get("v_gb") or 0)
        facts.append(
            f"{item.get('name')}：{item.get('dtype')}，{item.get('tokens'):,} token，{size} GB"
        )
    for capture in report.graph_captures:
        sizes = capture.get("batch_sizes") or []
        facts.append(
            f"CUDA graph（{capture.get('role')} {capture.get('kind')}）：{len(sizes)} 个 batch size"
            + (f"，最大 {max(sizes)}" if sizes else "")
            + f"，用时 {capture.get('elapsed_s')} s，占显存 {capture.get('mem_gb')} GB"
        )
    if facts:
        parts += ["### 启动过程", bullets(facts)]
    prefill, decode = report.prefill, report.decode
    lines: list[str | None] = []
    if prefill.get("batches"):
        lines += [
            f"prefill：{prefill['batches']} 个 batch，前缀命中率 "
            f"{prefill.get('prefix_cache_hit_rate', 0):.1%}",
            _stat("每个 prefill batch 的新 token", prefill.get("new_tokens_per_batch")),
            _stat("prefill 时的排队请求", prefill.get("queue_requests")),
        ]
        if prefill.get("full_chunk_fraction") is not None:
            lines.append(f"满 chunk 的 prefill batch 占 {prefill['full_chunk_fraction']:.0%}")
    if decode.get("log_lines"):
        lines += [
            f"decode：{decode['log_lines']} 行日志",
            _stat("运行请求", decode.get("running_requests")),
            _stat("KV 占用率", decode.get("token_usage")),
            _stat("生成吞吐", decode.get("gen_throughput"), " token/s"),
            _stat("decode 时的排队请求", decode.get("queue_requests")),
            _stat("投机解码 accept len", decode.get("accept_length")),
        ]
        if decode.get("graph_fraction") is not None:
            lines.append(f"走 CUDA graph 的 decode 行占 {decode['graph_fraction']:.0%}")
    if report.retractions.get("events"):
        lines.append(
            f"KV 池满撤回 {report.retractions['events']} 次，共 "
            f"{report.retractions['retracted_requests']} 个请求"
        )
    if any(lines):
        parts += ["### 运行统计", bullets(line for line in lines if line)]
    scale = report.decode_scale
    if report.decode_samples:
        samples = report.decode_samples
        step = max(len(samples) // MAX_DECODE_SAMPLES, 1)
        parts += [
            "### decode step：实测与成本模型",
            (
                f"实测/模型中位数 **{scale:.2f}**（可作为 decode 校准系数，见 `sglang-maestro calibrate`）"
                if scale
                else "样本不足或没有成本模型，未计算比值"
            ),
            table(
                [
                    "运行请求",
                    "KV token",
                    "生成吞吐 token/s",
                    "实测 step ms",
                    "模型 step ms",
                    "实测/模型",
                ],
                [
                    [
                        sample.running_requests,
                        sample.kv_tokens,
                        sample.gen_throughput,
                        _ms(sample.measured_step_us),
                        _ms(sample.model_step_us),
                        sample.ratio,
                    ]
                    for sample in samples[::step][:MAX_DECODE_SAMPLES]
                ],
            ),
        ]
    if report.moe_default_configs:
        parts += [
            "### 缺少调优配置的 MoE kernel",
            bullets(f"`{path}`" for path in report.moe_default_configs),
        ]
    for bench in report.benchmarks:
        parts += [
            "### 压测结果",
            table(["指标", "值"], [[key, value] for key, value in bench.items()]),
        ]
    parts.append(findings_section(report.findings, title="发现", limit=findings_limit))
    return "\n\n".join(parts) + "\n"
