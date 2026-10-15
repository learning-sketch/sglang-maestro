"""Fit cost-model corrections from SGLang server logs and benchmark results.

Decode: a clean ``Decode batch`` interval (no prefill logged since the previous decode line)
measures one step as ``#running-req / gen throughput``; the cost model is evaluated at the
logged batch and KV size. Prefill: SGLang logs ``input throughput`` as the batch's tokens over the
time since the previous prefill log, so two consecutive ``Prefill batch`` lines of one stream
with a backlog (``#queue-req > 0``) and no decode line between them time the first batch (decode
steps are logged only every ``--decode-log-interval`` steps, so an unlogged one can inflate a
sample; the median absorbs a few). Each scale is the median measured/model ratio. A benchmark
record supplies the measured latencies, the speculative accept length and, through
``server_info``, the parallel layout; a deployment plan's prediction can be checked against it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.knowledge.facts import source
from sgl_agent.markdown import bullets, table
from sgl_agent.model.config import ModelArch
from sgl_agent.model.ops import BatchShape, Calibration, ExecutionConfig, StepCostModel
from sgl_agent.model.parallel import ParallelConfig
from sgl_agent.perf.logs import (
    DECODE_MODEL_MIN_SAMPLES,
    BatchLine,
    ParsedLog,
    attach_model_times,
    decode_samples,
    parse_logs,
)
from sgl_agent.perf.metrics import Distribution
from sgl_agent.workload import accept_rate_from_length

SCALE_RANGE = (0.25, 4.0)
"""Fitted scales outside this range point at a wrong layout or model rather than efficiency."""
PLAN_TOLERANCE = 0.20
PRINTED_BENCH_KEYS = {
    "successful_requests": "completed",
    "benchmark_duration_s": "duration",
    "request_throughput_req_per_s": "request_throughput",
    "input_token_throughput_tok_per_s": "input_throughput",
    "output_token_throughput_tok_per_s": "output_throughput",
}
"""Printed ``Serving Benchmark Result`` labels -> the keys of the ``--output-file`` JSONL record."""
BENCH_METRICS = (
    ("completed", "完成请求数"),
    ("request_throughput", "请求吞吐 req/s"),
    ("output_throughput", "输出吞吐 tok/s"),
    ("concurrency", "平均并发"),
    ("median_ttft_ms", "TTFT P50 ms"),
    ("p90_ttft_ms", "TTFT P90 ms"),
    ("p99_ttft_ms", "TTFT P99 ms"),
    ("median_tpot_ms", "TPOT P50 ms"),
    ("p90_tpot_ms", "TPOT P90 ms"),
    ("p99_tpot_ms", "TPOT P99 ms"),
    ("median_itl_ms", "ITL P50 ms"),
    ("accept_length", "投机接受长度"),
)
LAYOUT_KEYS = {
    "tp": "tp_size",
    "pp": "pp_size",
    "attn_dp": "attn_dp_size",
    "attn_cp": "attn_cp_size",
    "dcp": "dcp_size",
    "ep": "ep_size",
}


@dataclass
class PrefillSample:
    """One prefill batch timed by the next prefill log of the same stream."""

    ranks: str
    new_seqs: int
    new_tokens: int
    cached_tokens: int
    measured_step_us: float
    model_step_us: float | None = None

    @property
    def ratio(self) -> float | None:
        if not self.model_step_us:
            return None
        return self.measured_step_us / self.model_step_us


@dataclass
class ScaleFit:
    samples: int
    ratios: Distribution | None
    scale: float
    used: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        ratios = self.ratios
        return {
            "samples": self.samples,
            "median": ratios.median if ratios else None,
            "p90": ratios.p90 if ratios else None,
            "min": ratios.minimum if ratios else None,
            "max": ratios.maximum if ratios else None,
            "scale": self.scale,
            "used": self.used,
            "note": self.note,
        }


@dataclass
class PlanCheck:
    metric: str
    predicted: float
    measured: float

    @property
    def deviation(self) -> float:
        return (self.measured - self.predicted) / self.predicted if self.predicted else 0.0

    @property
    def within(self) -> bool:
        return abs(self.deviation) <= PLAN_TOLERANCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "predicted": self.predicted,
            "measured": self.measured,
            "deviation": round(self.deviation, 4),
            "within_tolerance": self.within,
        }


@dataclass
class CalibrationReport:
    model: str
    hardware: str
    layout: ParallelConfig
    layout_source: str
    sources: list[str]
    calibration: Calibration
    decode: ScaleFit
    prefill: ScaleFit
    bench: dict[str, Any] | None = None
    accept_rate: float | None = None
    plan_checks: list[PlanCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "hardware": self.hardware,
            "layout": self.layout.describe(),
            "layout_source": self.layout_source,
            "sources": self.sources,
            "calibration": calibration_to_dict(self.calibration),
            "decode": self.decode.to_dict(),
            "prefill": self.prefill.to_dict(),
            "bench": self.bench,
            "accept_rate": self.accept_rate,
            "plan_checks": [check.to_dict() for check in self.plan_checks],
            "notes": self.notes,
        }

    def to_markdown(self) -> str:
        return render_calibration(self)


# ---------------------------------------------------------------------- calibration files
def calibration_to_dict(calibration: Calibration) -> dict[str, Any]:
    return {
        "prefill_scale": round(calibration.prefill_scale, 4),
        "decode_scale": round(calibration.decode_scale, 4),
        "step_overhead_us": calibration.step_overhead_us,
        "source": calibration.source,
    }


def calibration_from_dict(payload: dict[str, Any]) -> Calibration:
    """Accept a bare calibration or a ``sgl-agent calibrate --json`` report."""
    data = payload.get("calibration", payload)
    known = {"prefill_scale", "decode_scale", "step_overhead_us", "source"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown calibration fields {unknown}; known: {sorted(known)}")
    calibration = Calibration(
        prefill_scale=float(data.get("prefill_scale", 1.0)),
        decode_scale=float(data.get("decode_scale", 1.0)),
        step_overhead_us=(
            float(data["step_overhead_us"]) if data.get("step_overhead_us") is not None else None
        ),
        source=str(data.get("source", "file")),
    )
    if min(calibration.prefill_scale, calibration.decode_scale) <= 0:
        raise ValueError("calibration scales must be > 0")
    return calibration


def load_calibration(path: str | Path) -> Calibration:
    return calibration_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------- samples
def prefill_samples(parsed: ParsedLog) -> list[PrefillSample]:
    samples: list[PrefillSample] = []
    previous: dict[tuple[str, str], BatchLine | None] = {}
    for line in parsed.batches:
        stream = (line.source, line.ranks)
        if line.kind == "decode":
            previous[stream] = None
            continue
        before = previous.get(stream)
        previous[stream] = line
        if before is None or not (before.number("#queue-req") or 0) > 0:
            continue
        throughput = line.number("input throughput (token/s)")
        tokens = (line.number("#new-token") or 0) + (line.number("#replay-token") or 0)
        seqs = before.number("#new-seq")
        new = before.number("#new-token")
        if not throughput or throughput <= 0 or not tokens or not seqs or not new:
            continue
        samples.append(
            PrefillSample(
                ranks=before.ranks,
                new_seqs=int(seqs),
                new_tokens=int(new),
                cached_tokens=int(before.number("#cached-token") or 0),
                measured_step_us=tokens / throughput * 1e6,
            )
        )
    return samples


def attach_prefill_times(samples: list[PrefillSample], cost_model: StepCostModel) -> None:
    """The batch as ``#new-seq`` equal chunks sharing ``#cached-token`` as prefix (eager)."""
    cache: dict[tuple[int, int, int], float] = {}
    for sample in samples:
        key = (sample.new_seqs, sample.new_tokens, sample.cached_tokens)
        if key not in cache:
            count = max(sample.new_seqs, 1)
            chunks = [(sample.new_tokens // count, sample.cached_tokens // count)] * count
            first = sample.new_tokens - chunks[0][0] * (count - 1)
            chunks[0] = (first, chunks[0][1])
            shape = BatchShape.chunks(chunks)
            cache[key] = cost_model.estimate(shape, cuda_graph=False).total_us
        sample.model_step_us = cache[key]


def _fit(name: str, ratios: list[float]) -> ScaleFit:
    if len(ratios) < DECODE_MODEL_MIN_SAMPLES:
        return ScaleFit(
            len(ratios),
            Distribution.of(ratios) if ratios else None,
            1.0,
            False,
            f"{name} 只有 {len(ratios)} 个可用样本（至少 {DECODE_MODEL_MIN_SAMPLES} 个），保持 1.0",
        )
    distribution = Distribution.of(ratios)
    low, high = SCALE_RANGE
    if not low <= distribution.median <= high:
        return ScaleFit(
            len(ratios),
            distribution,
            1.0,
            False,
            f"{name} 实测/模型中位数 {distribution.median:.2f} 超出 [{low:g}, {high:g}]："
            "多半是并行布局、模型或量化方式与日志不符，不采用",
        )
    return ScaleFit(
        len(ratios),
        distribution,
        distribution.median,
        True,
        f"{name} 系数取 {len(ratios)} 个样本实测/模型比值的中位数",
    )


# ---------------------------------------------------------------------- benchmark records
def _normalize_bench(record: dict[str, Any]) -> dict[str, Any]:
    return {PRINTED_BENCH_KEYS.get(key, key): value for key, value in record.items()}


def load_bench_records(path: str | Path) -> list[dict[str, Any]]:
    """``bench_serving --output-file`` JSONL records, or printed result blocks in a text log."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"benchmark file {target} does not exist")
    if target.suffix in {".jsonl", ".json"}:
        text = target.read_text(encoding="utf-8").strip()
        if not text:
            return []
        if target.suffix == ".json" or text.startswith("["):
            payload = json.loads(text)
            records = payload if isinstance(payload, list) else [payload]
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        return [_normalize_bench(record) for record in records if isinstance(record, dict)]
    return [_normalize_bench(record) for record in parse_logs(target).benchmarks]


def _find(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


def layout_from_server_info(info: dict[str, Any] | None) -> ParallelConfig | None:
    """The layout a server reported in ``/server_info`` (embedded in the bench record)."""
    if not isinstance(info, dict) or _find(info, "tp_size") is None:
        return None
    values: dict[str, int] = {}
    for name, key in LAYOUT_KEYS.items():
        value = _find(info, key)
        if isinstance(value, int) and value >= 1:
            values[name] = value
    return ParallelConfig(**values)


def _accept_rate(bench: dict[str, Any]) -> float | None:
    length = bench.get("accept_length")
    steps = _find(bench.get("server_info"), "speculative_num_steps")
    if not isinstance(length, (int, float)) or not isinstance(steps, int):
        return None
    try:
        return accept_rate_from_length(float(length), steps)
    except ValueError:
        return None


def check_plan(plan: dict[str, Any], bench: dict[str, Any]) -> list[PlanCheck]:
    """Measured benchmark metrics against a ``sgl-agent deploy --json`` prediction."""
    chosen = plan.get("chosen") or {}
    result = chosen.get("result") or {}
    pairs = []
    for percentile in ("p50", "p90", "p99"):
        bench_key = "median" if percentile == "p50" else percentile
        for metric in ("ttft", "tpot"):
            predicted = (result.get(f"{metric}_ms") or {}).get(percentile)
            pairs.append(
                (f"{percentile} {metric}", predicted, bench.get(f"{bench_key}_{metric}_ms"))
            )
    pairs.append(
        ("output_throughput", result.get("output_throughput"), bench.get("output_throughput"))
    )
    checks = []
    for metric, predicted, measured in pairs:
        if isinstance(predicted, (int, float)) and isinstance(measured, (int, float)) and predicted:
            checks.append(PlanCheck(metric, float(predicted), float(measured)))
    return checks


# ---------------------------------------------------------------------- fit
def fit_calibration(
    logs: str | Path | list[str | Path] | ParsedLog,
    arch: ModelArch,
    hardware: HardwareSpec,
    parallel: ParallelConfig | None = None,
    *,
    execution: ExecutionConfig | None = None,
    bench: str | Path | None = None,
    plan: dict[str, Any] | None = None,
) -> CalibrationReport:
    """Fit ``Calibration`` for ``arch`` on ``hardware`` from one server's logs (see module doc).

    ``parallel`` defaults to the layout in the benchmark record's ``server_info``.
    """
    parsed = logs if isinstance(logs, ParsedLog) else parse_logs(logs)
    notes: list[str] = []
    records = load_bench_records(bench) if bench is not None else list(parsed.benchmarks)
    record = _normalize_bench(records[-1]) if records else None
    if parsed.synthetic or (record is not None and record.get("synthetic")):
        notes.append(
            "输入是 sgl-agent 生成的合成日志/压测记录（不是实测）：拟合出的系数应当还原生成时设定的偏差，"
            "只用于演示和回归测试"
        )
    if len(records) > 1:
        notes.append(f"压测文件里有 {len(records)} 条记录，用最后一条")
    reported = layout_from_server_info(record.get("server_info") if record else None)
    if parallel is None and reported is None:
        raise ValueError(
            "需要并行布局：传 --tp 等参数，或用带 server_info 的 bench_serving --output-file 结果"
        )
    layout = parallel or reported
    assert layout is not None
    layout_source = "命令行参数" if parallel is not None else "压测记录的 server_info"
    if parallel is not None and reported is not None and reported != parallel:
        notes.append(
            f"压测记录里服务端的布局是 {reported.describe()}，与传入的 {parallel.describe()} 不同"
        )
    errors = layout.validate(arch)
    if errors:
        raise ValueError(f"并行配置 {layout.describe()} 不适用于 {arch.name}：" + "；".join(errors))

    cost = StepCostModel(arch, hardware, layout, execution or ExecutionConfig(), Calibration())
    decode = decode_samples(parsed)
    attach_model_times(decode, cost)
    prefill = prefill_samples(parsed)
    attach_prefill_times(prefill, cost)
    decode_fit = _fit("decode", [s.ratio for s in decode if s.ratio])
    prefill_fit = _fit("prefill", [s.ratio for s in prefill if s.ratio])
    if not parsed.batches:
        notes.append(
            "日志里没有 Prefill/Decode batch 行：确认是 launch_server 的日志（默认每 40 个 decode step 打一行）"
        )
    spec_lines = [line for line in parsed.batches if line.number("accept len")]
    if spec_lines:
        notes.append("开了投机解码的 decode 行不参与 decode 拟合（一步含 draft + verify）")
    used = [
        f"{name} {fit.samples} 个样本"
        for name, fit in (("decode", decode_fit), ("prefill", prefill_fit))
        if fit.used
    ]
    calibration = Calibration(
        prefill_scale=prefill_fit.scale,
        decode_scale=decode_fit.scale,
        source=f"sgl-agent calibrate（{layout.describe()}，{'、'.join(used) or '无可用样本'}）",
    )
    bench_view = None
    accept_rate = None
    checks: list[PlanCheck] = []
    if record is not None:
        bench_view = {key: record.get(key) for key, _ in BENCH_METRICS if key in record}
        bench_view["request_rate"] = record.get("request_rate")
        accept_rate = _accept_rate(record)
        if plan is not None:
            checks = check_plan(plan, record)
    elif plan is not None:
        notes.append("给了部署方案但没有压测结果，无法对比预测")
    return CalibrationReport(
        model=arch.name,
        hardware=hardware.name,
        layout=layout,
        layout_source=layout_source,
        sources=parsed.sources + ([str(bench)] if bench is not None else []),
        calibration=calibration,
        decode=decode_fit,
        prefill=prefill_fit,
        bench=bench_view,
        accept_rate=accept_rate,
        plan_checks=checks,
        notes=notes,
    )


# ---------------------------------------------------------------------- rendering
def _fit_row(name: str, fit: ScaleFit) -> list[Any]:
    ratios = fit.ratios
    return [
        name,
        fit.samples,
        ratios.median if ratios else None,
        ratios.p90 if ratios else None,
        fit.scale,
        fit.used,
    ]


def render_calibration(report: CalibrationReport) -> str:
    calibration = report.calibration
    parts = [
        f"## 成本模型校准：{report.model} @ {report.hardware}",
        bullets(
            [
                f"布局：{report.layout.describe()}（来自{report.layout_source}）",
                "输入：" + "、".join(f"`{item}`" for item in report.sources),
                f"结果：prefill_scale = **{calibration.prefill_scale:.3f}**，"
                f"decode_scale = **{calibration.decode_scale:.3f}**",
            ]
        ),
        "### 拟合",
        table(
            ["阶段", "样本数", "实测/模型 中位数", "P90", "采用的系数", "已采用"],
            [_fit_row("decode", report.decode), _fit_row("prefill", report.prefill)],
        ),
        bullets([report.decode.note, report.prefill.note]),
    ]
    if report.bench:
        parts += [
            "### 压测结果",
            table(
                ["指标", "实测"],
                [
                    [label, report.bench.get(key)]
                    for key, label in BENCH_METRICS
                    if key in report.bench
                ],
            ),
        ]
        if report.accept_rate is not None:
            parts.append(
                f"按实测接受长度反推的单 token 接受率约 {report.accept_rate:.2f}："
                "把负载 YAML 的 `spec_accept_rate` 改成这个值再跑 `sgl-agent features`/`deploy`"
            )
    if report.plan_checks:
        rows = [
            [
                check.metric,
                check.predicted,
                check.measured,
                f"{check.deviation:+.0%}",
                check.within,
            ]
            for check in report.plan_checks
        ]
        parts += [
            "### 与部署方案的预测对比",
            table(["指标", "预测", "实测", "偏差", f"在 ±{PLAN_TOLERANCE:.0%} 内"], rows),
        ]
        if not all(check.within for check in report.plan_checks):
            parts.append(
                "有指标偏差超过容差：带 `--calibration <本命令的 JSON 输出>` 重新跑 `sgl-agent deploy`；"
                "decode 系数正常但 TTFT 偏差大时，多半是排队（到达率或路由不均），看 #queue-req"
            )
    usage = [
        "把 `--json` 输出存成文件（或 `-o calibration.json`），"
        "`sgl-agent features|deploy|trace|logs --calibration <文件>` 会用这些系数",
        "decode 样本只取两条 Decode 日志之间没有 prefill 的间隔；step 时间 = #running-req ÷ gen throughput",
        "prefill 样本是有排队时相邻两条 Prefill 日志的间隔，可能混入没打日志的 decode step（偏大）",
    ]
    parts += ["### 说明", bullets(report.notes + usage)]
    parts.append(
        "<sub>源码依据："
        + "；".join([source("kv_pool_stats"), source("prefill_log_throughput")])
        + "</sub>"
    )
    return "\n\n".join(parts) + "\n"
