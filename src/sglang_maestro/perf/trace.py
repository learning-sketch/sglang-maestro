"""Parse PyTorch/Kineto Chrome traces written by SGLang's profiler into per-step kernel timelines.

SGLang wraps every ``ModelRunner.forward`` in a ``record_function`` span named
``step[{STAGE} bs={bs}[ toks={extend_tokens}][ {detailed aggregates}]]``
(``sglang/srt/utils/profile_utils.py::build_step_span_name``); graph replays that bypass it emit
their own span, e.g. ``step[DRAFT_LOOP raw_bs=.. bs=.. topk=..]``. Kernels are attributed to a step
through their CUDA correlation id: the runtime launch (``cudaLaunchKernel``/``cudaGraphLaunch``)
with the same id sits inside the step's CPU span. When CPU events are missing, the projected
``gpu_user_annotation`` spans are used instead.

Per-rank files follow ``{profile_id}-TP-{rank}[-DP-..][-PP-..][-EP-..].trace.json.gz``; merged
files (``merged-{profile_id}.trace.json.gz``) prefix every ``pid`` with a ``[TP00-DP00]`` label.
"""

from __future__ import annotations

import bisect
import gzip
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from sglang_maestro.perf.classify import KernelClassifier

STEP_NAME_PATTERN = re.compile(
    r"^step\[(?P<stage>[A-Z][A-Z0-9_]*)(?P<lead>(?: [a-z_]+=\d+)*?) bs=(?P<bs>\d+)"
    r"(?: toks=(?P<toks>\d+))?(?P<rest>[^\]]*)\]$"
)
ANNOTATION_PAIR_PATTERN = re.compile(r"\b([a-z_]+)=(-?\d+)")
RANK_FILE_PATTERN = re.compile(r"(TP|DP|PP|EP)-(\d+)")
MERGED_PID_PATTERN = re.compile(r"^\[(?P<label>[^\]]*)\] (?P<pid>.*)$")
MERGED_LABEL_PART_PATTERN = re.compile(r"(TP|DP|PP|EP)(\d+)")
RANK_ORDER = ("TP", "DP", "PP", "EP")
DECODE_LIKE_STAGES = frozenset({"DECODE", "VERIFY", "DRAFT", "DRAFT_LOOP"})
TRACE_SUFFIXES = (".trace.json.gz", ".trace.json", ".json.gz", ".json")


@dataclass(frozen=True)
class TraceDialect:
    """Event categories and runtime API names; override for non-CUDA profilers."""

    kernel_categories: frozenset[str] = frozenset({"kernel"})
    memcpy_categories: frozenset[str] = frozenset({"gpu_memcpy"})
    memset_categories: frozenset[str] = frozenset({"gpu_memset"})
    runtime_categories: frozenset[str] = frozenset({"cuda_runtime", "cuda_driver"})
    cpu_annotation_categories: frozenset[str] = frozenset({"user_annotation"})
    gpu_annotation_categories: frozenset[str] = frozenset({"gpu_user_annotation"})
    kernel_launch_names: frozenset[str] = frozenset(
        {
            "cudaLaunchKernel",
            "cudaLaunchKernelExC",
            "cudaLaunchCooperativeKernel",
            "cuLaunchKernel",
            "cuLaunchKernelEx",
            "cuLaunchCooperativeKernel",
        }
    )
    graph_launch_names: frozenset[str] = frozenset({"cudaGraphLaunch", "cuGraphLaunch"})
    sync_names: frozenset[str] = frozenset(
        {
            "cudaStreamSynchronize",
            "cudaDeviceSynchronize",
            "cudaEventSynchronize",
            "cudaMemcpy",
            "cuStreamSynchronize",
            "cuCtxSynchronize",
            "cuEventSynchronize",
        }
    )
    memcpy_names: frozenset[str] = frozenset(
        {
            "cudaMemcpyAsync",
            "cudaMemcpy",
            "cuMemcpyAsync",
            "cuMemcpyDtoHAsync_v2",
            "cuMemcpyHtoDAsync_v2",
        }
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TraceDialect:
        """Replace the listed fields (each a list of names); unlisted fields keep the CUDA defaults."""
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown trace dialect fields {unknown}; known: {sorted(known)}")
        overrides: dict[str, frozenset[str]] = {}
        for key, value in payload.items():
            names = [value] if isinstance(value, str) else value
            if not isinstance(names, (list, tuple, set, frozenset)) or not names:
                raise ValueError(f"trace dialect field {key!r} must be a non-empty list of names")
            overrides[key] = frozenset(str(name) for name in names)
        return cls(**overrides)


@dataclass(slots=True)
class KernelEvent:
    name: str
    category: str
    comm_pattern: str | None
    fused: tuple[str, ...]
    kind: str
    rank: str
    device: int
    stream: int
    start_us: float
    duration_us: float
    correlation: int | None
    grid: tuple[int, ...] | None
    block: tuple[int, ...] | None
    nbytes: int | None
    launch_api: str | None = None
    step_id: int | None = None

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us

    @property
    def signature(self) -> tuple[str, tuple[int, ...] | None, tuple[int, ...] | None]:
        return (self.name, self.grid, self.block)


@dataclass(slots=True)
class StepSpan:
    step_id: int
    rank: str
    ordinal: int
    name: str
    stage: str
    batch_size: int
    extend_tokens: int | None
    annotations: dict[str, int]
    source: str
    cpu_start_us: float | None = None
    cpu_end_us: float | None = None
    gpu_start_us: float | None = None
    gpu_end_us: float | None = None
    kernel_count: int = 0
    kernel_launches: int = 0
    graph_launches: int = 0
    sync_calls: int = 0
    memcpy_calls: int = 0

    @property
    def label(self) -> str:
        text = f"{self.stage} bs={self.batch_size}"
        if self.extend_tokens is not None:
            text += f" toks={self.extend_tokens}"
        return text

    @property
    def is_decode_like(self) -> bool:
        return self.stage in DECODE_LIKE_STAGES

    @property
    def gpu_span_us(self) -> float:
        if self.gpu_start_us is None or self.gpu_end_us is None:
            return 0.0
        return self.gpu_end_us - self.gpu_start_us


@dataclass
class ParsedTrace:
    sources: list[str]
    ranks: list[str]
    kernels: list[KernelEvent]
    steps: list[StepSpan]
    device_names: dict[str, str] = field(default_factory=dict)
    runtime_totals: dict[str, int] = field(default_factory=dict)
    has_runtime_events: bool = False
    has_step_spans: bool = False
    synthetic: bool = False
    warnings: list[str] = field(default_factory=list)
    world_size: int | None = None
    """``distributedInfo.world_size`` written by the torch profiler (TP × PP in SGLang)."""

    @property
    def primary_rank(self) -> str:
        return self.ranks[0] if self.ranks else "rank0"

    def kernels_by_step(self) -> dict[int, list[KernelEvent]]:
        grouped: dict[int, list[KernelEvent]] = defaultdict(list)
        for kernel in self.kernels:
            if kernel.step_id is not None:
                grouped[kernel.step_id].append(kernel)
        return grouped

    def steps_for_rank(self, rank: str) -> list[StepSpan]:
        return [step for step in self.steps if step.rank == rank]

    @property
    def detailed_annotations(self) -> bool:
        return any(step.annotations for step in self.steps)


def parse_step_name(name: str) -> tuple[str, int, int | None, dict[str, int]] | None:
    """``step[DECODE bs=32 g_sq=32 ...]`` -> (stage, batch size, extend tokens, aggregates)."""
    match = STEP_NAME_PATTERN.match(name)
    if match is None:
        return None
    toks = match.group("toks")
    pairs = f"{match.group('lead')} {match.group('rest')}"
    aggregates = {key: int(value) for key, value in ANNOTATION_PAIR_PATTERN.findall(pairs)}
    return (
        match.group("stage"),
        int(match.group("bs")),
        int(toks) if toks is not None else None,
        aggregates,
    )


def _format_rank(parts: dict[str, int]) -> str:
    return "-".join(f"{kind}{parts[kind]}" for kind in RANK_ORDER if kind in parts)


def rank_from_filename(path: str | Path) -> str | None:
    parts = {kind: int(value) for kind, value in RANK_FILE_PATTERN.findall(Path(path).name)}
    return _format_rank(parts) if parts else None


def _rank_from_merged_label(label: str) -> str:
    parts = {kind: int(value) for kind, value in MERGED_LABEL_PART_PATTERN.findall(label)}
    return _format_rank(parts) if parts else label or "unknown"


def _rank_sort_key(rank: str) -> tuple:
    parts = dict(MERGED_LABEL_PART_PATTERN.findall(rank))
    if not parts:
        return (1, rank)
    return (0, *(int(parts.get(kind, 0)) for kind in ("DP", "EP", "PP", "TP")))


def discover_trace_files(path: str | Path) -> list[Path]:
    """Expand a directory into trace files; per-rank files win over ``merged-*`` duplicates."""
    candidate = Path(path)
    if candidate.is_file():
        return [candidate]
    if not candidate.is_dir():
        raise FileNotFoundError(f"trace path {candidate} does not exist")
    files = sorted(
        item
        for item in candidate.iterdir()
        if item.is_file() and item.name.endswith(TRACE_SUFFIXES)
    )
    per_rank = [
        item for item in files if "TP-" in item.name and not item.name.startswith("merged-")
    ]
    chosen = per_rank or files
    if not chosen:
        raise FileNotFoundError(f"no *.trace.json[.gz] files under {candidate}")
    return chosen


def load_trace_payload(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: not a readable Chrome trace ({error})") from error
    if isinstance(payload, list):
        return {"traceEvents": payload}
    if not isinstance(payload, dict) or not isinstance(payload.get("traceEvents"), list):
        raise ValueError(f"{path}: expected a JSON object with a 'traceEvents' list")
    return payload


class _SpanIndex:
    """Innermost enclosing span lookup over spans sorted by start (supports shallow nesting)."""

    __slots__ = ("starts", "spans")
    MAX_WALK_BACK = 32

    def __init__(self, spans: list[tuple[float, float, int]]):
        spans.sort(key=lambda item: item[0])
        self.spans = spans
        self.starts = [item[0] for item in spans]

    def find(self, timestamp: float) -> int | None:
        position = bisect.bisect_right(self.starts, timestamp) - 1
        walked = 0
        while position >= 0 and walked < self.MAX_WALK_BACK:
            _, end, step_id = self.spans[position]
            if end >= timestamp:
                return step_id
            position -= 1
            walked += 1
        return None


@dataclass(slots=True)
class _RuntimeCall:
    name: str
    rank: str
    thread: tuple[Any, Any]
    start_us: float
    correlation: int | None


@dataclass(slots=True)
class _RawSpan:
    rank: str
    lane: tuple[Any, Any]
    start_us: float
    end_us: float
    name: str
    stage: str
    batch_size: int
    extend_tokens: int | None
    annotations: dict[str, int]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_dims(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, (list, tuple)) and value:
        return tuple(_as_int(item) for item in value)
    return None


def _merge_gpu_spans(spans: list[_RawSpan]) -> list[_RawSpan]:
    """Kineto may project one CPU span onto several streams; fold same-name overlapping copies."""
    spans.sort(key=lambda item: item.start_us)
    merged: list[_RawSpan] = []
    open_by_name: dict[str, _RawSpan] = {}
    for span in spans:
        current = open_by_name.get(span.name)
        if current is not None and span.start_us <= current.end_us:
            current.end_us = max(current.end_us, span.end_us)
            continue
        copy = _RawSpan(
            span.rank,
            span.lane,
            span.start_us,
            span.end_us,
            span.name,
            span.stage,
            span.batch_size,
            span.extend_tokens,
            span.annotations,
        )
        merged.append(copy)
        open_by_name[span.name] = copy
    return merged


def load_trace(
    paths: str | Path | list[str | Path],
    classifier: KernelClassifier | None = None,
    dialect: TraceDialect | None = None,
) -> ParsedTrace:
    """Load one or more trace files (or a profile directory) into a :class:`ParsedTrace`."""
    classifier = classifier or KernelClassifier()
    dialect = dialect or TraceDialect()
    if isinstance(paths, (str, Path)):
        files = discover_trace_files(paths)
    else:
        files = [file for item in paths for file in discover_trace_files(item)]

    kernels: list[KernelEvent] = []
    runtime_calls: list[_RuntimeCall] = []
    cpu_spans: list[_RawSpan] = []
    gpu_spans: list[_RawSpan] = []
    device_names: dict[str, str] = {}
    runtime_totals: dict[str, int] = defaultdict(int)
    warnings: list[str] = []
    synthetic = False
    world_sizes: set[int] = set()
    kernel_like = dialect.kernel_categories | dialect.memcpy_categories | dialect.memset_categories

    for file_index, path in enumerate(files):
        payload = load_trace_payload(path)
        synthetic = synthetic or bool(payload.get("synthetic"))
        distributed = payload.get("distributedInfo")
        if not isinstance(distributed, dict):
            distributed = {}
        if _as_int(distributed.get("world_size"), 0) > 0:
            world_sizes.add(_as_int(distributed["world_size"]))
        file_rank = rank_from_filename(path)
        if file_rank is None:
            file_rank = f"TP{distributed['rank']}" if "rank" in distributed else f"file{file_index}"
        properties = payload.get("deviceProperties") or []
        if properties and isinstance(properties, list) and isinstance(properties[0], dict):
            device_names.setdefault(file_rank, str(properties[0].get("name", "")))
        for event in payload["traceEvents"]:
            if not isinstance(event, dict) or event.get("ph") != "X":
                continue
            category = event.get("cat")
            if category is None:
                continue
            duration = event.get("dur")
            timestamp = event.get("ts")
            if duration is None or timestamp is None:
                continue
            rank = file_rank
            pid = event.get("pid")
            if isinstance(pid, str):
                match = MERGED_PID_PATTERN.match(pid)
                if match is not None:
                    rank = _rank_from_merged_label(match.group("label"))
                    pid = match.group("pid")
            name = event.get("name") or ""
            if category in kernel_like:
                args = event.get("args") or {}
                kernel_class = classifier.classify(name, category)
                if category in dialect.memcpy_categories:
                    kind = "memcpy"
                elif category in dialect.memset_categories:
                    kind = "memset"
                else:
                    kind = "kernel"
                correlation = args.get("correlation")
                kernels.append(
                    KernelEvent(
                        name=name,
                        category=kernel_class.category,
                        comm_pattern=kernel_class.comm_pattern,
                        fused=kernel_class.fused,
                        kind=kind,
                        rank=rank,
                        device=_as_int(args.get("device", pid)),
                        stream=_as_int(args.get("stream", event.get("tid"))),
                        start_us=float(timestamp),
                        duration_us=float(duration),
                        correlation=_as_int(correlation) if correlation is not None else None,
                        grid=_as_dims(args.get("grid")),
                        block=_as_dims(args.get("block")),
                        nbytes=_as_int(args["bytes"]) if "bytes" in args else None,
                    )
                )
            elif category in dialect.runtime_categories:
                args = event.get("args") or {}
                correlation = args.get("correlation")
                runtime_totals[name] += 1
                runtime_calls.append(
                    _RuntimeCall(
                        name,
                        rank,
                        (pid, event.get("tid")),
                        float(timestamp),
                        _as_int(correlation) if correlation is not None else None,
                    )
                )
            elif name.startswith("step[") and (
                category in dialect.cpu_annotation_categories
                or category in dialect.gpu_annotation_categories
            ):
                parsed = parse_step_name(name)
                if parsed is None:
                    continue
                stage, batch_size, extend_tokens, aggregates = parsed
                span = _RawSpan(
                    rank,
                    (pid, event.get("tid")),
                    float(timestamp),
                    float(timestamp) + float(duration),
                    name,
                    stage,
                    batch_size,
                    extend_tokens,
                    aggregates,
                )
                if category in dialect.cpu_annotation_categories:
                    cpu_spans.append(span)
                else:
                    gpu_spans.append(span)

    ranks = sorted(
        {kernel.rank for kernel in kernels} | {span.rank for span in cpu_spans}, key=_rank_sort_key
    )
    steps: list[StepSpan] = []
    thread_indexes: dict[tuple[str, Any, Any], _SpanIndex] = {}
    gpu_indexes: dict[tuple[str, int], _SpanIndex] = {}
    cpu_by_rank: dict[str, list[_RawSpan]] = defaultdict(list)
    gpu_by_rank: dict[str, list[_RawSpan]] = defaultdict(list)
    for span in cpu_spans:
        cpu_by_rank[span.rank].append(span)
    for span in gpu_spans:
        gpu_by_rank[span.rank].append(span)

    for rank in ranks:
        rank_cpu = sorted(cpu_by_rank.get(rank, []), key=lambda item: item.start_us)
        rank_gpu = _merge_gpu_spans(gpu_by_rank.get(rank, []))
        rank_steps: list[StepSpan] = []
        if rank_cpu:
            per_thread: dict[tuple[Any, Any], list[tuple[float, float, int]]] = defaultdict(list)
            for ordinal, span in enumerate(rank_cpu):
                step = StepSpan(
                    step_id=len(steps) + ordinal,
                    rank=rank,
                    ordinal=ordinal,
                    name=span.name,
                    stage=span.stage,
                    batch_size=span.batch_size,
                    extend_tokens=span.extend_tokens,
                    annotations=span.annotations,
                    source="cpu",
                    cpu_start_us=span.start_us,
                    cpu_end_us=span.end_us,
                )
                rank_steps.append(step)
                per_thread[span.lane].append((span.start_us, span.end_us, step.step_id))
            for lane, entries in per_thread.items():
                thread_indexes[(rank, *lane)] = _SpanIndex(entries)
            queues: dict[str, deque[int]] = defaultdict(deque)
            for step in rank_steps:
                queues[step.name].append(step.step_id)
            per_device: dict[int, list[tuple[float, float, int]]] = defaultdict(list)
            for span in rank_gpu:
                queue = queues.get(span.name)
                if queue:
                    per_device[_as_int(span.lane[0])].append(
                        (span.start_us, span.end_us, queue.popleft())
                    )
            for device, entries in per_device.items():
                gpu_indexes[(rank, device)] = _SpanIndex(entries)
        elif rank_gpu:
            per_device = defaultdict(list)
            for ordinal, span in enumerate(rank_gpu):
                step = StepSpan(
                    step_id=len(steps) + ordinal,
                    rank=rank,
                    ordinal=ordinal,
                    name=span.name,
                    stage=span.stage,
                    batch_size=span.batch_size,
                    extend_tokens=span.extend_tokens,
                    annotations=span.annotations,
                    source="gpu",
                    gpu_start_us=span.start_us,
                    gpu_end_us=span.end_us,
                )
                rank_steps.append(step)
                per_device[_as_int(span.lane[0])].append((span.start_us, span.end_us, step.step_id))
            for device, entries in per_device.items():
                gpu_indexes[(rank, device)] = _SpanIndex(entries)
        steps.extend(rank_steps)

    has_step_spans = bool(steps)
    if not has_step_spans and kernels:
        warnings.append(
            "trace 中没有 step[...] span：按整段 trace 统计。请用 SGLang 自带的 /start_profile 采集"
            "（torch profiler 激活时每个 ModelRunner.forward 都会打 step span）。"
        )
        for ordinal, rank in enumerate(ranks):
            steps.append(
                StepSpan(
                    step_id=ordinal,
                    rank=rank,
                    ordinal=0,
                    name="step[ALL bs=0]",
                    stage="ALL",
                    batch_size=0,
                    extend_tokens=None,
                    annotations={},
                    source="none",
                )
            )

    by_id = {step.step_id: step for step in steps}
    launch_info: dict[tuple[str, int], tuple[str, int | None]] = {}
    for call in runtime_calls:
        index = thread_indexes.get((call.rank, *call.thread))
        step_id = index.find(call.start_us) if index is not None else None
        if step_id is not None:
            step = by_id[step_id]
            if call.name in dialect.kernel_launch_names:
                step.kernel_launches += 1
            elif call.name in dialect.graph_launch_names:
                step.graph_launches += 1
            if call.name in dialect.sync_names:
                step.sync_calls += 1
            if call.name in dialect.memcpy_names:
                step.memcpy_calls += 1
        if call.correlation is not None:
            launch_info[(call.rank, call.correlation)] = (call.name, step_id)

    fallback_steps = {step.rank: step.step_id for step in steps if step.source == "none"}
    kernel_extents: dict[int, list[float]] = {}
    unattributed = 0
    for kernel in kernels:
        info = (
            launch_info.get((kernel.rank, kernel.correlation))
            if kernel.correlation is not None
            else None
        )
        if info is not None:
            kernel.launch_api, kernel.step_id = info
        else:
            index = gpu_indexes.get((kernel.rank, kernel.device))
            if index is not None:
                kernel.step_id = index.find(kernel.start_us)
            elif kernel.rank in fallback_steps:
                kernel.step_id = fallback_steps[kernel.rank]
            if kernel.correlation is not None and runtime_calls:
                unattributed += 1
        if kernel.step_id is None:
            continue
        by_id[kernel.step_id].kernel_count += 1
        end = kernel.start_us + kernel.duration_us
        extent = kernel_extents.get(kernel.step_id)
        if extent is None:
            kernel_extents[kernel.step_id] = [kernel.start_us, end]
        else:
            if kernel.start_us < extent[0]:
                extent[0] = kernel.start_us
            if end > extent[1]:
                extent[1] = end
    for step_id, (start, end) in kernel_extents.items():
        by_id[step_id].gpu_start_us = start
        by_id[step_id].gpu_end_us = end

    if runtime_calls and unattributed > 0.2 * max(len(kernels), 1):
        warnings.append(
            f"{unattributed} 个 kernel 的 correlation 找不到对应的 runtime 调用，"
            "可能是 trace 被截断或采集时没有开启 CPU activity。"
        )
    if len(world_sizes) > 1:
        warnings.append(
            f"trace 文件的 distributedInfo.world_size 不一致（{sorted(world_sizes)}）：可能混入了不同部署的 profile。"
        )
    kernels.sort(key=lambda item: (_rank_sort_key(item.rank), item.start_us))
    steps.sort(key=lambda item: (_rank_sort_key(item.rank), item.ordinal))
    return ParsedTrace(
        sources=[str(path) for path in files],
        ranks=ranks,
        kernels=kernels,
        steps=steps,
        device_names=device_names,
        runtime_totals=dict(runtime_totals),
        has_runtime_events=bool(runtime_calls),
        has_step_spans=has_step_spans,
        synthetic=synthetic,
        warnings=warnings,
        world_size=max(world_sizes) if world_sizes else None,
    )


# ---------------------------------------------------------------------- interval helpers
def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of intervals; the input must be sorted by start."""
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def total_length(merged: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merged)


def intersection_length(
    first: list[tuple[float, float]], second: list[tuple[float, float]]
) -> float:
    """Overlap between two merged (disjoint, sorted) interval lists."""
    overlap = 0.0
    first_index = second_index = 0
    while first_index < len(first) and second_index < len(second):
        start = max(first[first_index][0], second[second_index][0])
        end = min(first[first_index][1], second[second_index][1])
        if end > start:
            overlap += end - start
        if first[first_index][1] < second[second_index][1]:
            first_index += 1
        else:
            second_index += 1
    return overlap


_WRAPPER_TEMPLATES = frozenset(
    {
        "cutlass::device_kernel",
        "cutlass::Kernel",
        "cutlass::Kernel2",
        "flash::enable_sm90_or_later",
        "flash::enable_sm80_to_sm89",
        "enable_sm90_or_later",
    }
)
_FUNCTOR_PATTERN = re.compile(r"(\w*Functor\w*|\w+_kernel\w*)")


def _head_and_template(text: str) -> tuple[str, str | None]:
    """Identifier before the first top-level ``<``/``(`` and the text of its template arguments."""
    head: list[str] = []
    for position, character in enumerate(text):
        if character == "(":
            break
        if character == "<":
            depth = 0
            for closing in range(position, len(text)):
                if text[closing] == "<":
                    depth += 1
                elif text[closing] == ">":
                    depth -= 1
                    if depth == 0:
                        return "".join(head).strip(), text[position + 1 : closing]
            return "".join(head).strip(), text[position + 1 :]
        head.append(character)
    return "".join(head).strip(), None


def short_kernel_name(name: str, max_length: int = 72) -> str:
    """Readable kernel name: no ``void``/parameters, wrapper templates unwrapped, ATen functor kept."""
    text = name.strip().replace("(anonymous namespace)::", "")
    if text.startswith("void "):
        text = text[5:]
    head, template = _head_and_template(text)
    for _ in range(4):
        if template is None or head not in _WRAPPER_TEMPLATES:
            break
        head, template = _head_and_template(template.strip())
    cleaned = head or text
    if template and cleaned.endswith("elementwise_kernel"):
        functor = _FUNCTOR_PATTERN.search(template)
        if functor:
            cleaned = f"{cleaned}<{functor.group(1)}>"
    return cleaned if len(cleaned) <= max_length else cleaned[: max_length - 1] + "…"
