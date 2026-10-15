"""SGLang server logs and benchmark output: batch statistics, startup facts and findings.

Formats come from the SGLang commit in the knowledge snapshot:

* prefix ``[%Y-%m-%d %H:%M:%S[.mmm][ DP0][ PP0][ ATTN_CP0][ MOE_DP0][ TP0][ EP0]] ``
  (``managers/scheduler.py`` and ``utils/common.py::configure_logger``);
* ``Prefill batch, #new-seq: .., #new-token: .., #cached-token: .., token usage: .., #running-req: ..,
  #queue-req: .., #pending-token: .., cuda graph: .., input throughput (token/s): ..``;
* ``Decode batch, #running-req: .., #token: .., token usage: .., [accept len: .., accept rate: .., ]
  cuda graph: .., gen throughput (token/s): .., #queue-req: ..`` every ``--decode-log-interval``
  decode steps (``#token`` excludes evictable radix-cache tokens; ``cuda graph`` is the last batch's);
* ``max_total_num_tokens=.., chunked_prefill_size=.., ..., available_gpu_mem=.. GB`` at startup;
* ``KV cache pool is full. Retract requests. #retracted_reqs: ..``;
* ``Using default MoE kernel config. Performance might be sub-optimal! Config file not found at ..``;
* ``Capture target decode CUDA graph end. bs=[..], elapsed=.. s, mem usage=.. GB, avail mem=.. GB.``;
* the ``python -m sglang.benchmark.serving`` result block.
"""

from __future__ import annotations

import gzip
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sgl_agent.findings import Finding, sort_findings
from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.knowledge import SGLangKnowledge, load_knowledge
from sgl_agent.knowledge.facts import source
from sgl_agent.model.config import ModelArch
from sgl_agent.model.moe_tuning import TUNING_SCRIPT, triton_moe_config_key, tuning_command
from sgl_agent.model.ops import BatchShape, Calibration, ExecutionConfig, StepCostModel
from sgl_agent.model.parallel import ParallelConfig
from sgl_agent.perf.metrics import Distribution

SYNTHETIC_LOG_MARK = (
    "# synthetic SGLang server log written by sgl-agent from its cost model; not a measurement"
)
LOG_PREFIX_PATTERN = re.compile(
    r"\[(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)(?P<ranks>(?: [A-Z_]+\d+)*)\] "
)
BATCH_PATTERN = re.compile(r"(?P<kind>Prefill|Decode) batch(?: \[\d+\])?, (?P<fields>.*)$")
FIELD_PATTERN = re.compile(r"^(?P<key>[^:]+): (?P<value>\S+)$")
STARTUP_PATTERN = re.compile(
    r"max_total_num_tokens=(?P<max_total_num_tokens>\d+), "
    r"chunked_prefill_size=(?P<chunked_prefill_size>-?\d+), "
    r"max_prefill_tokens=(?P<max_prefill_tokens>\d+), "
    r"max_running_requests=(?P<max_running_requests>\d+), "
    r"context_len=(?P<context_len>\d+), "
    r"available_(?:gpu|cpu)_mem=(?P<available_mem_gb>[\d.]+) GB"
)
WEIGHT_PATTERN = re.compile(
    r"Load weight end\. elapsed=(?P<elapsed_s>[\d.]+) s, type=(?P<model_class>\w+), "
    r"(?:(?P<quantization>.*?), )?avail mem=(?P<available_mem_gb>[\d.]+) GB, "
    r"mem usage=(?P<weights_gb>[\d.]+) GB\."
)
KV_PATTERN = re.compile(
    r"(?P<name>[\w ]+?) (?:is allocated|VA upper bound)\. dtype: (?P<dtype>[\w.]+), "
    r"#tokens: (?P<tokens>\d+), (?:K size: (?P<k_gb>[\d.]+) GB, V size: (?P<v_gb>[\d.]+) GB|"
    r"KV size: (?P<kv_gb>[\d.]+) GB)"
)
GRAPH_PATTERN = re.compile(
    r"Capture (?P<role>\w+) (?P<kind>decode|verify|prefill) (?P<backend>.+?) end\. "
    r"(?:bs=\[(?P<batch_sizes>[^\]]*)\], )?elapsed=(?P<elapsed_s>[\d.]+) s, "
    r"mem usage=(?P<mem_gb>[\d.]+) GB, avail mem=(?P<available_mem_gb>[\d.]+) GB\."
)
RETRACT_PATTERN = re.compile(
    r"KV cache pool is full\. Retract requests\. #retracted_reqs: (?P<count>\d+)"
)
MOE_DEFAULT_PATTERN = re.compile(
    r"Using default MoE kernel config\. Performance might be sub-optimal! "
    r"Config file not found at (?P<path>.+?), you can create"
)
MOE_VERSION_FALLBACK_PATTERN = re.compile(
    r"Config file not found at (?P<path>.+?)\. Fallback to triton version (?P<version>[\d.]+) "
)
OVERLAP_DISABLED_PATTERN = re.compile(r"[Oo]verlap sched\w* (?:is disabled|is not implemented).*$")
BENCH_HEADER = "Serving Benchmark Result"
BENCH_LINE_PATTERN = re.compile(r"^(?P<label>[A-Za-z][^:]*?):\s+(?P<value>-?\d+(?:\.\d+)?)%?\s*$")

DECODE_MODEL_MIN_SAMPLES = 3
DECODE_SLOWER_RATIO = 1.5
DECODE_MUCH_SLOWER_RATIO = 2.5
DECODE_FASTER_RATIO = 0.67
GRAPH_OFF_MIN_FRACTION = 0.10
KV_BOUND_USAGE = 0.85
QUEUE_CAPPED_MIN_FRACTION = 0.2
QUEUE_CAPPED_MAX_USAGE = 0.7
SPEC_LOW_ACCEPT_LENGTH = 1.5
RETRACT_HIGH_EVENTS = 3
RETRACT_HIGH_REQUESTS = 10


def _to_value(text: str) -> float | bool | str:
    if text in {"True", "False"}:
        return text == "True"
    try:
        return float(text)
    except ValueError:
        return text


@dataclass
class BatchLine:
    kind: str
    source: str
    line_number: int
    time: datetime | None
    ranks: str
    fields: dict[str, float | bool | str]

    def number(self, key: str) -> float | None:
        value = self.fields.get(key)
        if isinstance(value, bool) or not isinstance(value, float):
            return None
        return value

    @property
    def graph(self) -> bool | None:
        """``cuda graph``/``npu graph``/``musa graph``/``cpu graph`` flag of the logged batch."""
        for key, value in self.fields.items():
            if key.endswith(" graph") and isinstance(value, bool):
                return value
        return None


@dataclass
class ParsedLog:
    sources: list[str] = field(default_factory=list)
    batches: list[BatchLine] = field(default_factory=list)
    startups: list[dict[str, Any]] = field(default_factory=list)
    weight_loads: list[dict[str, Any]] = field(default_factory=list)
    kv_allocations: list[dict[str, Any]] = field(default_factory=list)
    graph_captures: list[dict[str, Any]] = field(default_factory=list)
    retractions: list[dict[str, Any]] = field(default_factory=list)
    moe_default_configs: Counter[str] = field(default_factory=Counter)
    moe_version_fallbacks: Counter[str] = field(default_factory=Counter)
    """``<missing config path> → triton <version whose tuned config was used instead>``."""
    overlap_messages: Counter[str] = field(default_factory=Counter)
    benchmarks: list[dict[str, float]] = field(default_factory=list)
    total_lines: int = 0
    recognized_lines: int = 0
    synthetic: bool = False
    """A file starts with ``SYNTHETIC_LOG_MARK`` (written by ``sgl-agent synth-log``)."""


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _bench_key(label: str) -> str:
    text = label.strip().lower()
    for unit, replacement in (
        ("(ms)", " ms"),
        ("(s)", " s"),
        ("(tok/s)", " tok_per_s"),
        ("(req/s)", " req_per_s"),
    ):
        text = text.replace(unit, replacement)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _parse_time(text: str) -> datetime | None:
    for pattern in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    return None


def _numbers(match: re.Match[str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in match.groupdict().items():
        if value is None:
            continue
        try:
            values[key] = int(value)
        except ValueError:
            try:
                values[key] = float(value)
            except ValueError:
                values[key] = value
    return values


def parse_logs(paths: str | Path | list[str | Path]) -> ParsedLog:
    """Stream log files (plain or ``.gz``) and keep only the lines the analysis uses."""
    items = [paths] if isinstance(paths, (str, Path)) else list(paths)
    files: list[Path] = []
    for item in items:
        candidate = Path(item)
        if candidate.is_dir():
            files += sorted(
                child
                for child in candidate.iterdir()
                if child.is_file() and child.suffix in {".log", ".txt", ".gz", ".out"}
            )
        elif candidate.is_file():
            files.append(candidate)
        else:
            raise FileNotFoundError(f"log path {candidate} does not exist")
    if not files:
        raise FileNotFoundError(f"no log files found in {items}")

    parsed = ParsedLog(sources=[str(path) for path in files])
    for path in files:
        bench: dict[str, float] | None = None
        with _open_text(path) as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                parsed.total_lines += 1
                line = raw_line.rstrip("\n")
                if line_number == 1 and line == SYNTHETIC_LOG_MARK:
                    parsed.synthetic = True
                    continue
                if bench is not None:
                    stripped = line.strip()
                    if stripped and set(stripped) == {"="}:
                        if bench:
                            parsed.benchmarks.append(bench)
                            bench = None
                        continue
                    match = BENCH_LINE_PATTERN.match(stripped)
                    if match is not None:
                        bench[_bench_key(match.group("label"))] = float(match.group("value"))
                        continue
                if BENCH_HEADER in line:
                    if bench:
                        parsed.benchmarks.append(bench)
                    bench = {}
                    parsed.recognized_lines += 1
                    continue
                prefix = LOG_PREFIX_PATTERN.search(line)
                if prefix is not None:
                    message = line[prefix.end() :]
                    time = _parse_time(prefix.group("time"))
                    ranks = prefix.group("ranks").strip()
                else:
                    message, time, ranks = line, None, ""
                if _parse_message(parsed, message, str(path), line_number, time, ranks):
                    parsed.recognized_lines += 1
        if bench:
            parsed.benchmarks.append(bench)
    return parsed


def _parse_message(
    parsed: ParsedLog,
    message: str,
    source_path: str,
    line_number: int,
    time: datetime | None,
    ranks: str,
) -> bool:
    batch = BATCH_PATTERN.search(message)
    if batch is not None:
        fields: dict[str, float | bool | str] = {}
        for part in batch.group("fields").split(", "):
            item = FIELD_PATTERN.match(part.strip())
            if item is not None:
                fields[item.group("key").strip()] = _to_value(item.group("value"))
        parsed.batches.append(
            BatchLine(
                kind=batch.group("kind").lower(),
                source=source_path,
                line_number=line_number,
                time=time,
                ranks=ranks,
                fields=fields,
            )
        )
        return True
    for pattern, target in (
        (STARTUP_PATTERN, parsed.startups),
        (WEIGHT_PATTERN, parsed.weight_loads),
        (GRAPH_PATTERN, parsed.graph_captures),
        (KV_PATTERN, parsed.kv_allocations),
    ):
        match = pattern.search(message)
        if match is not None:
            values = _numbers(match)
            if "batch_sizes" in values:
                values["batch_sizes"] = [
                    int(size) for size in str(values["batch_sizes"]).split(",") if size.strip()
                ]
            values["ranks"] = ranks
            target.append(values)
            return True
    retract = RETRACT_PATTERN.search(message)
    if retract is not None:
        parsed.retractions.append(
            {"time": time, "ranks": ranks, "count": int(retract.group("count"))}
        )
        return True
    moe = MOE_DEFAULT_PATTERN.search(message)
    if moe is not None:
        parsed.moe_default_configs[moe.group("path")] += 1
        return True
    fallback = MOE_VERSION_FALLBACK_PATTERN.search(message)
    if fallback is not None:
        parsed.moe_version_fallbacks[
            f"{fallback.group('path')} → triton {fallback.group('version')}"
        ] += 1
        return True
    overlap = OVERLAP_DISABLED_PATTERN.search(message)
    if overlap is not None:
        parsed.overlap_messages[overlap.group(0).strip()] += 1
        return True
    return False


# ------------------------------------------------------------------------------ report
@dataclass
class DecodeSample:
    """One ``Decode batch`` interval without prefill in between (usable for calibration)."""

    ranks: str
    running_requests: int
    kv_tokens: int
    gen_throughput: float
    graph: bool | None
    measured_step_us: float
    model_step_us: float | None = None

    @property
    def ratio(self) -> float | None:
        if not self.model_step_us:
            return None
        return self.measured_step_us / self.model_step_us


@dataclass
class LogReport:
    sources: list[str]
    startup: dict[str, Any] | None
    weight_load: dict[str, Any] | None
    kv_cache: list[dict[str, Any]]
    graph_captures: list[dict[str, Any]]
    prefill: dict[str, Any]
    decode: dict[str, Any]
    retractions: dict[str, Any]
    moe_default_configs: list[str]
    moe_version_fallbacks: list[str]
    overlap_messages: list[str]
    benchmarks: list[dict[str, float]]
    decode_samples: list[DecodeSample]
    findings: list[Finding]
    warnings: list[str] = field(default_factory=list)

    @property
    def decode_scale(self) -> float | None:
        """Median measured/model decode step ratio over clean intervals (a calibration factor)."""
        ratios = sorted(sample.ratio for sample in self.decode_samples if sample.ratio)
        if len(ratios) < DECODE_MODEL_MIN_SAMPLES:
            return None
        return Distribution.of(ratios).median

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": self.sources,
            "startup": self.startup,
            "weight_load": self.weight_load,
            "kv_cache": self.kv_cache,
            "graph_captures": self.graph_captures,
            "prefill": self.prefill,
            "decode": self.decode,
            "retractions": self.retractions,
            "moe_default_configs": self.moe_default_configs,
            "moe_version_fallbacks": self.moe_version_fallbacks,
            "overlap_messages": self.overlap_messages,
            "benchmarks": self.benchmarks,
            "decode_scale": self.decode_scale,
            "decode_samples": [
                {
                    "ranks": sample.ranks,
                    "running_requests": sample.running_requests,
                    "kv_tokens": sample.kv_tokens,
                    "gen_throughput": sample.gen_throughput,
                    "graph": sample.graph,
                    "measured_step_us": sample.measured_step_us,
                    "model_step_us": sample.model_step_us,
                }
                for sample in self.decode_samples
            ],
            "findings": [finding.to_dict() for finding in self.findings],
            "warnings": self.warnings,
        }


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    distribution = Distribution.of(values)
    return {
        "count": distribution.count,
        "median": distribution.median,
        "p90": distribution.p90,
        "max": distribution.maximum,
        "mean": distribution.mean,
    }


def _column(lines: list[BatchLine], key: str) -> list[float]:
    return [value for line in lines if (value := line.number(key)) is not None]


def _prefill_summary(lines: list[BatchLine], chunked_prefill_size: int | None) -> dict[str, Any]:
    new_tokens = _column(lines, "#new-token")
    cached = _column(lines, "#cached-token")
    total_new, total_cached = sum(new_tokens), sum(cached)
    summary: dict[str, Any] = {
        "batches": len(lines),
        "new_tokens_per_batch": _summary(new_tokens),
        "cached_tokens_total": total_cached,
        "new_tokens_total": total_new,
        "prefix_cache_hit_rate": (
            total_cached / (total_cached + total_new) if total_cached + total_new > 0 else None
        ),
        "queue_requests": _summary(_column(lines, "#queue-req")),
        "token_usage": _summary(_column(lines, "token usage")),
    }
    if chunked_prefill_size and chunked_prefill_size > 0 and new_tokens:
        summary["full_chunk_fraction"] = sum(
            1 for value in new_tokens if value >= chunked_prefill_size
        ) / len(new_tokens)
    return summary


def _decode_summary(lines: list[BatchLine]) -> dict[str, Any]:
    graph_flags = [line.graph for line in lines if line.graph is not None]
    return {
        "log_lines": len(lines),
        "running_requests": _summary(_column(lines, "#running-req")),
        "kv_tokens": _summary(_column(lines, "#token")),
        "token_usage": _summary(_column(lines, "token usage")),
        "gen_throughput": _summary(_column(lines, "gen throughput (token/s)")),
        "queue_requests": _summary(_column(lines, "#queue-req")),
        "accept_length": _summary(_column(lines, "accept len")),
        "graph_fraction": (
            sum(1 for flag in graph_flags if flag) / len(graph_flags) if graph_flags else None
        ),
    }


def decode_samples(parsed: ParsedLog) -> list[DecodeSample]:
    """Decode intervals with no prefill batch logged in between, per log stream (file + ranks)."""
    samples: list[DecodeSample] = []
    state: dict[tuple[str, str], tuple[bool, bool]] = {}
    for line in parsed.batches:
        stream = (line.source, line.ranks)
        seen_decode, prefill_since = state.get(stream, (False, False))
        if line.kind == "prefill":
            state[stream] = (seen_decode, True)
            continue
        state[stream] = (True, False)
        if not seen_decode or prefill_since or line.number("accept len"):
            continue
        running = line.number("#running-req")
        throughput = line.number("gen throughput (token/s)")
        kv_tokens = line.number("#token")
        if not running or not throughput or throughput <= 0 or kv_tokens is None:
            continue
        samples.append(
            DecodeSample(
                ranks=line.ranks,
                running_requests=int(running),
                kv_tokens=int(kv_tokens),
                gen_throughput=throughput,
                graph=line.graph,
                measured_step_us=running / throughput * 1e6,
            )
        )
    return samples


def attach_model_times(samples: list[DecodeSample], cost_model: StepCostModel) -> None:
    cache: dict[tuple[int, int, bool | None], float] = {}
    for sample in samples:
        key = (sample.running_requests, sample.kv_tokens, sample.graph)
        if key not in cache:
            shape = BatchShape.decode_total(sample.running_requests, max(sample.kv_tokens, 1))
            cache[key] = cost_model.estimate(
                shape, cuda_graph=sample.graph if sample.graph is not None else None
            ).total_us
        sample.model_step_us = cache[key]


# ---------------------------------------------------------------------------- findings
def _kv_recommendations(knowledge: SGLangKnowledge, arch: ModelArch | None) -> list[str]:
    lines = [
        "提高 KV 容量：显式调大 `--mem-fraction-static`（先看启动日志里 available_gpu_mem 的余量）；"
        "`--kv-cache-dtype fp8_e4m3` 让 KV 容量约翻倍（上线前做精度评测）；",
        "或者控制准入：降低 `--max-running-requests`，或把 `--schedule-conservativeness` 调到 1.0 以上"
        "（帮助文本：频繁 retract 时用更大的值）。",
    ]
    if arch is not None and arch.attention.is_mla:
        lines.append(
            "MLA 模型在纯 TP 下每张卡都存一份完整的 latent KV：DP attention（`--attn-dp-size`）或 DCP（`--dcp-size`）"
            "可以按 DP/DCP 倍数扩大可容纳的 token 数，见特性建议与部署搜索。"
        )
    return [
        line
        for line in lines
        if all(knowledge.has_flag(flag) for flag in re.findall(r"`(--[\w-]+)", line))
    ]


def _retraction_finding(
    report_retractions: dict[str, Any], knowledge: SGLangKnowledge, arch: ModelArch | None
) -> Finding | None:
    events = report_retractions["events"]
    if not events:
        return None
    total = report_retractions["retracted_requests"]
    severity = (
        "high" if events >= RETRACT_HIGH_EVENTS or total >= RETRACT_HIGH_REQUESTS else "medium"
    )
    return Finding(
        id="log.kv.retract",
        kind="kv_capacity",
        severity=severity,
        title=f"KV cache 池满，decode 请求被 retract {events} 次（共 {total} 个请求）",
        evidence=[
            f"日志中出现 {events} 次“KV cache pool is full. Retract requests.”，累计 retract {total} 个请求；"
            "被 retract 的请求会释放 KV、回到等待队列，之后重新 prefill，表现为延迟尖刺和吞吐下降。",
        ],
        recommendation=_kv_recommendations(knowledge, arch),
        scope="decode",
        references=[source("retract")],
    )


def _graph_off_finding(
    decode_lines: list[BatchLine], captured_max_bs: int | None
) -> Finding | None:
    active = [line for line in decode_lines if (line.number("#running-req") or 0) > 0]
    flagged = [line for line in active if line.graph is not None]
    off = [line for line in flagged if line.graph is False]
    if not flagged or len(off) / len(flagged) < GRAPH_OFF_MIN_FRACTION:
        return None
    running_off = sorted(int(line.number("#running-req") or 0) for line in off)
    fraction = len(off) / len(flagged)
    evidence = [
        f"{len(flagged)} 条 Decode batch 日志里有 {len(off)} 条（{fraction:.0%}）graph 标记为 False；"
        f"这些行的 #running-req 为 {running_off[0]}–{running_off[-1]}（中位数 {running_off[len(running_off) // 2]}）。",
        "该标记只反映每个日志间隔最后一个 batch，是抽样值。",
    ]
    recommendation: list[str] = []
    flags: dict[str, Any] = {}
    if captured_max_bs is not None:
        evidence.append(f"启动日志里 decode graph 捕获的最大 batch 为 {captured_max_bs}。")
        if running_off[-1] > captured_max_bs:
            flags["--cuda-graph-max-bs-decode"] = running_off[-1]
            recommendation.append(
                f"运行中的 batch（最多 {running_off[-1]}）超过捕获上限 {captured_max_bs}：设置 "
                f"`--cuda-graph-max-bs-decode {running_off[-1]}`（多捕获的档位会占用显存，需复核 KV 容量）。"
            )
    if not recommendation:
        recommendation.append(
            "batch 没有超过捕获上限时，回退 eager 的其它条件见 decode_cuda_graph_runner.can_run_graph："
            "DP attention 下取各 DP rank 中最大的 batch、请求带 embedding 覆盖、开启 TBO 但该 batch 不能切分、"
            "投机解码每请求 token 数与捕获宽度不一致。"
        )
    recommendation.append(
        "用 /start_profile 采一段 trace，`sgl-agent trace` 会量化 eager decode 的发射开销。"
    )
    return Finding(
        id="log.decode.graph_off",
        kind="launch_bound",
        severity="medium" if fraction >= 0.3 else "low",
        title=f"{fraction:.0%} 的 decode 日志显示没有走 graph",
        evidence=evidence,
        recommendation=recommendation,
        flags=flags,
        scope="decode",
        references=[source("decode_graph_fallback")],
    )


def _queue_findings(
    decode_lines: list[BatchLine],
    startup: dict[str, Any] | None,
    knowledge: SGLangKnowledge,
    arch: ModelArch | None,
    has_retraction: bool,
) -> list[Finding]:
    waiting = [line for line in decode_lines if (line.number("#queue-req") or 0) > 0]
    if not waiting:
        return []
    findings: list[Finding] = []
    usage = Distribution.of(_column(waiting, "token usage"))
    max_running = startup.get("max_running_requests") if startup else None
    if max_running:
        capped = [
            line
            for line in waiting
            if (line.number("#running-req") or 0) >= max_running
            and (line.number("token usage") or 0.0) < QUEUE_CAPPED_MAX_USAGE
        ]
        if len(capped) / len(waiting) >= QUEUE_CAPPED_MIN_FRACTION:
            findings.append(
                Finding(
                    id="log.decode.max_running_capped",
                    kind="scheduling",
                    severity="medium",
                    title=f"请求在排队，但并发被 max_running_requests={max_running} 卡住，KV 还有余量",
                    evidence=[
                        f"{len(waiting)} 条有排队的 Decode 日志中，{len(capped)} 条 #running-req 已达到 {max_running}，"
                        f"而 token usage < {QUEUE_CAPPED_MAX_USAGE:.0%}。",
                    ],
                    recommendation=[
                        "在延迟目标允许时调大 `--max-running-requests`；decode graph 的默认捕获上限也受它限制，"
                        "需要同时确认 `--cuda-graph-max-bs-decode` 覆盖新的并发。用部署搜索（`sgl-agent deploy`）评估 "
                        "更大 batch 对 ITL 的影响。",
                    ],
                    scope="decode",
                )
            )
    if not has_retraction and usage.count and usage.median >= KV_BOUND_USAGE:
        findings.append(
            Finding(
                id="log.kv.capacity_bound",
                kind="kv_capacity",
                severity="medium",
                title=f"有请求排队时 KV 使用率中位数 {usage.median:.0%}：吞吐受 KV 容量限制",
                evidence=[
                    f"{len(waiting)} 条有排队的 Decode 日志，token usage 中位数 {usage.median:.0%}、P90 {usage.p90:.0%}。",
                ],
                recommendation=_kv_recommendations(knowledge, arch),
                scope="decode",
            )
        )
    return findings


def _spec_finding(decode_lines: list[BatchLine]) -> Finding | None:
    accept = Distribution.of(_column(decode_lines, "accept len"))
    if not accept.count or accept.median >= SPEC_LOW_ACCEPT_LENGTH:
        return None
    return Finding(
        id="log.spec.low_accept",
        kind="speculative",
        severity="medium",
        title=f"投机解码平均接受长度只有 {accept.median:.2f}",
        evidence=[
            f"{accept.count} 条 Decode 日志的 accept len 中位数 {accept.median:.2f}、最小 {accept.minimum:.2f}："
            "每次 verify 平均只多产出不到半个 token，而 verify 比普通 decode 多算 draft token，草稿模型本身也要时间。",
        ],
        recommendation=[
            "检查草稿模型是否与目标模型匹配（MTP/NextN 权重版本、EAGLE 头），调小 `--speculative-num-steps`/"
            "`--speculative-num-draft-tokens` 或 `--speculative-eagle-topk`；大 batch 下投机解码收益本来就小，"
            "用 `sgl-agent deploy` 对比开/关投机解码在目标负载下的结果。",
        ],
        scope="decode",
    )


def _moe_config_finding(
    paths: list[str],
    fallbacks: list[str],
    arch: ModelArch | None,
    hardware: HardwareSpec | None,
    parallel: ParallelConfig | None,
    knowledge: SGLangKnowledge,
) -> Finding | None:
    if not paths and not fallbacks:
        return None
    evidence: list[str] = []
    if paths:
        names = sorted({Path(path).name for path in paths})
        evidence.append(
            "SGLang 没找到 Triton fused MoE 的调优配置，用的是默认 tile 配置："
            + "；".join(names[:3])
            + (f" 等 {len(names)} 个" if len(names) > 3 else "")
            + "。"
        )
    if fallbacks:
        evidence.append(
            "当前 Triton 版本没有调优配置，借用了其它 Triton 版本的配置（性能可能不是最优）："
            + "；".join(sorted(fallbacks)[:3])
            + "。"
        )
    recommendation = [
        f"在目标卡上用当前的 Triton 版本运行 `{TUNING_SCRIPT}` 生成这些 JSON，放到 "
        "`$SGLANG_MOE_CONFIG_DIR/configs/triton_<版本号，点换成下划线>/`；不设该环境变量时目录是 SGLang 安装目录下的 "
        "`srt/layers/moe/moe_runner/triton_utils/configs/triton_<版本>/`。",
    ]
    if arch is not None and hardware is not None and parallel is not None:
        key = triton_moe_config_key(arch, parallel, hardware)
        if key is not None:
            recommendation.append(
                f"本配置对应的调优命令：`{tuning_command(arch, parallel, key)}`。"
            )
            device = key.device
            shipped = knowledge.moe_tuned_configs(
                device=device,
                experts=key.experts,
                intermediate=key.intermediate,
                dtype=key.dtype,
                block_shape=list(key.block_shape) if key.block_shape else None,
            )
            if not shipped:
                evidence.append(
                    f"知识快照里也没有 {device} 上 E={key.experts}、N={key.intermediate} 的现成配置。"
                )
    recommendation.append("调优前后各采一次 trace，比较 moe 类 kernel 的时间确认收益。")
    return Finding(
        id="log.moe.default_config",
        kind="kernel_tuning",
        severity="medium" if paths else "low",
        title=(
            "Fused MoE 在用默认 kernel 配置（未针对本卡调优）"
            if paths
            else "Fused MoE 借用了其它 Triton 版本的调优配置"
        ),
        evidence=evidence,
        recommendation=recommendation,
        scope="moe",
        references=[source("moe_config_dir")],
    )


def _decode_model_finding(samples: list[DecodeSample], cost_model_label: str) -> Finding | None:
    ratios = sorted(sample.ratio for sample in samples if sample.ratio)
    if len(ratios) < DECODE_MODEL_MIN_SAMPLES:
        return None
    ratio = Distribution.of(ratios)
    examples = sorted(samples, key=lambda sample: -(sample.ratio or 0.0))[:3]
    example_text = "；".join(
        f"bs={sample.running_requests}、KV {sample.kv_tokens} token：实测 {sample.measured_step_us / 1000:.2f} ms/step，"
        f"模型 {(sample.model_step_us or 0) / 1000:.2f} ms"
        for sample in examples
    )
    caveat = (
        "实测值 = #running-req ÷ gen throughput，假设日志间隔内 batch 不变；只用两条 Decode 日志之间没有 prefill 的间隔，"
        "#token 不含可驱逐的前缀缓存，共享前缀被多个请求重复读取时模型会低估注意力时间。"
    )
    if ratio.median >= DECODE_SLOWER_RATIO:
        return Finding(
            id="log.decode.slower_than_model",
            kind="performance_gap",
            severity="high" if ratio.median >= DECODE_MUCH_SLOWER_RATIO else "medium",
            title=f"decode 实测比成本模型慢 {ratio.median:.1f} 倍（{cost_model_label}）",
            evidence=[
                f"{ratio.count} 个干净的日志间隔，实测/模型 step 时间比中位数 {ratio.median:.2f}、P90 {ratio.p90:.2f}。",
                f"差距最大的间隔：{example_text}。",
                caveat,
            ],
            recommendation=[
                "用 /start_profile 采一段稳态 decode 的 trace 交给 `sgl-agent trace`，按 kernel 类别找出比屋脊点慢的部分"
                "（MoE 未调优、注意力后端、eager decode、通信暴露等）。",
                "如果 trace 确认各类 kernel 都正常，用 `sgl-agent calibrate` 把实测拟合进成本模型，再做部署搜索。",
            ],
            scope="decode",
            references=[source("kv_pool_stats")],
        )
    if ratio.median <= DECODE_FASTER_RATIO:
        return Finding(
            id="log.decode.faster_than_model",
            kind="calibration",
            severity="info",
            title=f"decode 实测比成本模型快（实测/模型 = {ratio.median:.2f}）：模型的效率假设偏保守",
            evidence=[f"{ratio.count} 个干净的日志间隔。", caveat],
            recommendation=["用 `sgl-agent calibrate` 拟合 decode 系数，部署搜索会更准。"],
            scope="decode",
        )
    return None


def analyze_logs(
    logs: str | Path | list[str | Path] | ParsedLog,
    *,
    arch: ModelArch | None = None,
    hardware: HardwareSpec | None = None,
    parallel: ParallelConfig | None = None,
    execution: ExecutionConfig | None = None,
    calibration: Calibration | None = None,
    knowledge: SGLangKnowledge | None = None,
) -> LogReport:
    parsed = logs if isinstance(logs, ParsedLog) else parse_logs(logs)
    knowledge = knowledge or load_knowledge()
    warnings: list[str] = []
    if parsed.synthetic:
        warnings.append(
            "这是 sgl-agent 生成的合成日志：耗时来自成本模型乘以设定的偏差系数，只用于演示和回归测试，"
            "不代表真实硬件表现。"
        )
    if parsed.recognized_lines == 0:
        warnings.append(
            "没有识别到 SGLang 的日志行：确认传入的是 sglang.launch_server 的输出（含 Prefill/Decode batch 行）"
            "或 sglang.benchmark.serving 的结果。"
        )
    if len(parsed.startups) > 1:
        warnings.append(
            f"日志里有 {len(parsed.startups)} 次启动记录：统计合并了所有启动，启动参数取最后一次。"
        )
    startup = parsed.startups[-1] if parsed.startups else None
    prefill_lines = [line for line in parsed.batches if line.kind == "prefill"]
    decode_lines = [line for line in parsed.batches if line.kind == "decode"]
    decode_captures = [
        capture
        for capture in parsed.graph_captures
        if capture.get("kind") in {"decode", "verify"} and capture.get("role") == "target"
    ]
    captured_max_bs = max(
        (max(capture["batch_sizes"]) for capture in decode_captures if capture.get("batch_sizes")),
        default=None,
    )
    retractions = {
        "events": len(parsed.retractions),
        "retracted_requests": sum(item["count"] for item in parsed.retractions),
    }

    samples = decode_samples(parsed)
    cost_model_label = ""
    if arch is not None and hardware is not None and parallel is not None and samples:
        errors = parallel.validate(arch)
        if errors:
            raise ValueError(
                f"并行配置 {parallel.describe()} 不适用于 {arch.name}：" + "；".join(errors)
            )
        cost_model = StepCostModel(
            arch, hardware, parallel, execution or ExecutionConfig(), calibration or Calibration()
        )
        attach_model_times(samples, cost_model)
        cost_model_label = f"{arch.name} · {hardware.name} · {parallel.describe()}"
    elif samples and arch is None:
        warnings.append("没有提供 --model/--hardware/并行配置：不把 decode 吞吐与成本模型对比。")

    findings = [
        finding
        for finding in (
            _retraction_finding(retractions, knowledge, arch),
            _graph_off_finding(decode_lines, captured_max_bs),
            _spec_finding(decode_lines),
            _moe_config_finding(
                list(parsed.moe_default_configs),
                list(parsed.moe_version_fallbacks),
                arch,
                hardware,
                parallel,
                knowledge,
            ),
            _decode_model_finding(samples, cost_model_label) if cost_model_label else None,
        )
        if finding is not None
    ]
    findings += _queue_findings(
        decode_lines, startup, knowledge, arch, has_retraction=bool(parsed.retractions)
    )
    if parsed.overlap_messages:
        findings.append(
            Finding(
                id="log.overlap_schedule.disabled",
                kind="scheduling",
                severity="info",
                title="启动日志提示 overlap 调度被关闭",
                evidence=[
                    f"{message}（×{count}）" for message, count in parsed.overlap_messages.items()
                ],
                recommendation=[
                    "overlap 调度让 CPU 准备下一个 batch 时 GPU 继续计算；被关闭时 decode 每步都会多出调度间隙。"
                    "确认是否是所用特性的硬性限制，必要时换用兼容 overlap 的配置。",
                ],
                scope="scheduler",
            )
        )
    return LogReport(
        sources=parsed.sources,
        startup=startup,
        weight_load=parsed.weight_loads[-1] if parsed.weight_loads else None,
        kv_cache=parsed.kv_allocations,
        graph_captures=parsed.graph_captures,
        prefill=_prefill_summary(
            prefill_lines, startup.get("chunked_prefill_size") if startup else None
        ),
        decode=_decode_summary(decode_lines),
        retractions=retractions,
        moe_default_configs=list(parsed.moe_default_configs),
        moe_version_fallbacks=list(parsed.moe_version_fallbacks),
        overlap_messages=list(parsed.overlap_messages),
        benchmarks=parsed.benchmarks,
        decode_samples=samples,
        findings=sort_findings(findings),
        warnings=warnings,
    )
