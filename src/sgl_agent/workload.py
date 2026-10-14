"""Serving workloads in ``sglang.benchmark.serving`` terms: lengths, arrivals, prefix sharing, SLOs.

A workload describes the traffic a deployment must serve. It uses the benchmark's own semantics
(``random`` and ``generated-shared-prefix`` datasets) so every plan the agent produces comes with
the exact benchmark command that checks it on the cluster.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from sgl_agent.commands import Command, tool_command
from sgl_agent.knowledge.facts import BENCH_SERVING_MODULE, source

SLO_PERCENTILES = (50, 90, 95, 99)
GROUP_DISTRIBUTIONS = ("uniform", "zipf")
MULTI_TURN_BENCH_BACKEND = "sglang-oai-chat"
"""bench_serving replays multi-turn rows only on chat backends (``MULTI_TURN_BACKENDS``)."""
RANDOM_RANGE_RATIO_DEFAULT = 0.0
GSP_RANGE_RATIO_DEFAULT = 1.0
"""``--random-range-ratio`` defaults to 0.0 (lengths uniform in [1, L]); ``--gsp-range-ratio`` to 1.0."""


def length_bounds(full_len: int, range_ratio: float) -> tuple[int, int]:
    """Inclusive bounds ``compute_random_lens`` draws from: ``[max(int(L*r), 1), L]``."""
    if full_len <= 0:
        return 0, 0
    return max(int(full_len * range_ratio), 1), full_len


def mean_length(full_len: int, range_ratio: float) -> float:
    low, high = length_bounds(full_len, range_ratio)
    return (low + high) / 2


def _span(full_len: int, range_ratio: float) -> str:
    low, high = length_bounds(full_len, range_ratio)
    return str(high) if low == high else f"{low}–{high}"


def sample_length(rng: random.Random, full_len: int, range_ratio: float) -> int:
    low, high = length_bounds(full_len, range_ratio)
    return rng.randint(low, high) if high > 0 else 0


def expected_accept_length(accept_rate: float, num_steps: int, topk: int = 1) -> float:
    """Tokens one verify step emits: ``sum(a^i, i=0..steps)`` for a chain of i.i.d. drafts.

    ``topk > 1`` is approximated by a per-depth acceptance of ``1 - (1 - a)^topk`` (any of the
    ``topk`` candidates may match); real trees share the draft-token budget, so this is an upper
    bound. Replace ``accept_rate`` with the value measured from the server log (``accept len``).
    """
    if not 0 < accept_rate < 1:
        raise ValueError("accept_rate must be in (0, 1)")
    if num_steps < 0:
        raise ValueError("num_steps must be >= 0")
    alpha = 1 - (1 - accept_rate) ** max(topk, 1)
    return (1 - alpha ** (num_steps + 1)) / (1 - alpha)


def accept_rate_from_length(accept_length: float, num_steps: int) -> float:
    """Invert :func:`expected_accept_length` (chain drafts) to recover the per-token rate."""
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    upper = num_steps + 1
    if not 1 < accept_length < upper:
        raise ValueError(f"accept length must be in (1, {upper}) for {num_steps} steps")
    low, high = 1e-6, 1 - 1e-9
    for _ in range(100):
        middle = (low + high) / 2
        if expected_accept_length(middle, num_steps) < accept_length:
            low = middle
        else:
            high = middle
    return (low + high) / 2


@dataclass(frozen=True)
class SharedPrefix:
    """``generated-shared-prefix``: groups share a system prompt, each request adds a question."""

    num_groups: int = 64
    prompts_per_group: int = 16
    system_prompt_len: int = 2048
    question_len: int = 128
    num_turns: int = 1
    distribution: str = "uniform"
    zipf_alpha: float | None = None

    def __post_init__(self) -> None:
        for name in ("num_groups", "prompts_per_group", "system_prompt_len", "num_turns"):
            if getattr(self, name) < 1:
                raise ValueError(f"shared_prefix.{name} must be >= 1")
        if self.question_len < 0:
            raise ValueError("shared_prefix.question_len must be >= 0")
        if self.distribution not in GROUP_DISTRIBUTIONS:
            raise ValueError(f"shared_prefix.distribution must be one of {GROUP_DISTRIBUTIONS}")
        if self.distribution == "zipf" and (self.zipf_alpha is None or self.zipf_alpha <= 0):
            raise ValueError("shared_prefix.zipf_alpha must be > 0 for the zipf distribution")

    @property
    def num_rows(self) -> int:
        return self.num_groups * self.prompts_per_group

    def group_probabilities(self) -> list[float]:
        """Share of rows per group (``_zipf_group_probs``: weight 1/rank^alpha, group 0 hottest)."""
        if self.distribution == "uniform":
            return [1 / self.num_groups] * self.num_groups
        assert self.zipf_alpha is not None
        weights = [1 / (rank**self.zipf_alpha) for rank in range(1, self.num_groups + 1)]
        total = sum(weights)
        return [weight / total for weight in weights]


@dataclass(frozen=True)
class SLO:
    """Latency targets at a percentile across requests (bench_serving reports p50/p90/p95/p99)."""

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    percentile: int = 90

    def __post_init__(self) -> None:
        if self.percentile not in SLO_PERCENTILES:
            raise ValueError(f"slo.percentile must be one of {SLO_PERCENTILES}")
        for name in ("ttft_ms", "tpot_ms"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"slo.{name} must be > 0")

    @property
    def defined(self) -> bool:
        return self.ttft_ms is not None or self.tpot_ms is not None

    def describe(self) -> str:
        if not self.defined:
            return "无延迟 SLO（只看吞吐）"
        parts = []
        if self.ttft_ms is not None:
            parts.append(f"P{self.percentile} TTFT ≤ {self.ttft_ms:g} ms")
        if self.tpot_ms is not None:
            parts.append(f"P{self.percentile} TPOT ≤ {self.tpot_ms:g} ms")
        return "，".join(parts)


@dataclass(frozen=True)
class Request:
    """One benchmark row. Follow-up turns arrive when the previous turn finishes."""

    arrival_s: float
    input_len: int
    output_len: int
    group: int | None = None
    shared_len: int = 0
    """Leading tokens shared with every request of ``group`` (the system prompt)."""
    followup_questions: tuple[int, ...] = ()


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    input_len: int
    output_len: int
    range_ratio: float = RANDOM_RANGE_RATIO_DEFAULT
    request_rate: float | None = None
    """Target arrival rate in req/s (Poisson); None means "as fast as the server takes them"."""
    max_concurrency: int | None = None
    num_prompts: int = 1000
    shared_prefix: SharedPrefix | None = None
    slo: SLO = field(default_factory=SLO)
    spec_accept_rate: float = 0.7
    """Per-token draft acceptance for speculative decoding: an assumption until measured."""
    seed: int = 1

    def __post_init__(self) -> None:
        if self.input_len < 1 or self.output_len < 1:
            raise ValueError("input_len and output_len must be >= 1")
        if not 0 <= self.range_ratio <= 1:
            raise ValueError("range_ratio must be in [0, 1]")
        if self.request_rate is not None and self.request_rate <= 0:
            raise ValueError("request_rate must be > 0 (omit it for an unbounded rate)")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        if self.num_prompts < 1:
            raise ValueError("num_prompts must be >= 1")
        if not 0 < self.spec_accept_rate < 1:
            raise ValueError("spec_accept_rate must be in (0, 1)")
        if self.shared_prefix is not None:
            expected = self.shared_prefix.system_prompt_len + self.shared_prefix.question_len
            if self.input_len != expected:
                raise ValueError(
                    f"input_len ({self.input_len}) must equal system_prompt_len + question_len "
                    f"({expected}) for a shared-prefix workload"
                )

    @property
    def dataset_name(self) -> str:
        return "generated-shared-prefix" if self.shared_prefix else "random"

    @property
    def num_requests(self) -> int:
        """Requests the benchmark sends (each multi-turn round is one request)."""
        if self.shared_prefix:
            return self.shared_prefix.num_rows * self.shared_prefix.num_turns
        return self.num_prompts

    @property
    def mean_output_len(self) -> float:
        return mean_length(self.output_len, self.range_ratio)

    @property
    def mean_input_len(self) -> float:
        """Mean prompt length per request, multi-turn history included."""
        prefix = self.shared_prefix
        if prefix is None:
            return mean_length(self.input_len, self.range_ratio)
        system = mean_length(prefix.system_prompt_len, self.range_ratio)
        question = mean_length(prefix.question_len, self.range_ratio)
        turns = prefix.num_turns
        # turn t carries the system prompt, t+1 questions and t previous answers.
        history = sum((t + 1) * question + t * self.mean_output_len for t in range(turns)) / turns
        return system + history

    @property
    def mean_new_input_len(self) -> float:
        """Mean prompt tokens a request must prefill when every shared prefix hits the cache."""
        prefix = self.shared_prefix
        if prefix is None:
            return self.mean_input_len
        question = mean_length(prefix.question_len, self.range_ratio)
        system = mean_length(prefix.system_prompt_len, self.range_ratio)
        # each group's system prompt is prefilled once; later turns reuse the cached history.
        return question + system * prefix.num_groups / self.num_requests

    @property
    def prefix_reuse_ratio(self) -> float:
        """Fraction of prompt tokens a warm radix cache can serve (shared prefix + history)."""
        return max(0.0, 1 - self.mean_new_input_len / self.mean_input_len)

    def accept_length(self, num_steps: int, topk: int = 1) -> float:
        return expected_accept_length(self.spec_accept_rate, num_steps, topk)

    def with_(self, **changes: Any) -> Workload:
        return replace(self, **changes)

    def describe(self) -> str:
        prefix = self.shared_prefix
        if prefix:
            shape = (
                f"{prefix.num_groups} 组×{prefix.prompts_per_group} 条共享前缀，系统提示 "
                f"{prefix.system_prompt_len} + 问题 {prefix.question_len} token"
                + (f"，{prefix.num_turns} 轮对话" if prefix.num_turns > 1 else "")
                + (f"，zipf α={prefix.zipf_alpha:g}" if prefix.distribution == "zipf" else "")
            )
        else:
            shape = f"输入 {_span(self.input_len, self.range_ratio)} token"
        unit = "会话/s（每个会话的各轮依次发送）" if prefix and prefix.num_turns > 1 else "req/s"
        rate = (
            f"{self.request_rate:g} {unit}（泊松到达）"
            if self.request_rate
            else "不限速（吞吐模式）"
        )
        concurrency = f"，并发上限 {self.max_concurrency}" if self.max_concurrency else ""
        output = _span(self.output_len, self.range_ratio)
        return f"{shape}，输出 {output} token，{rate}{concurrency}；{self.slo.describe()}"


def _arrivals(rng: random.Random, count: int, rate: float | None) -> list[float]:
    """bench_serving's default arrivals: Poisson (burstiness 1.0), or all at t=0 for rate=inf."""
    if rate is None:
        return [0.0] * count
    now, times = 0.0, []
    for _ in range(count):
        times.append(now)
        now += rng.expovariate(rate)
    return times


def sample_requests(
    workload: Workload, *, num: int | None = None, seed: int | None = None
) -> list[Request]:
    """Draw benchmark rows with the dataset's semantics (lengths, group assignment, turns)."""
    rng = random.Random(workload.seed if seed is None else seed)
    prefix = workload.shared_prefix
    ratio = workload.range_ratio
    if prefix is None:
        count = num or workload.num_prompts
        arrivals = _arrivals(rng, count, workload.request_rate)
        return [
            Request(
                arrival_s=arrival,
                input_len=sample_length(rng, workload.input_len, ratio),
                output_len=sample_length(rng, workload.output_len, ratio),
            )
            for arrival in arrivals
        ]
    system_lens = [
        sample_length(rng, prefix.system_prompt_len, ratio) for _ in range(prefix.num_groups)
    ]
    rows = num or prefix.num_rows
    if prefix.distribution == "uniform":
        groups = [index % prefix.num_groups for index in range(rows)]
        rng.shuffle(groups)
    else:
        groups = rng.choices(range(prefix.num_groups), weights=prefix.group_probabilities(), k=rows)
    arrivals = _arrivals(rng, rows, workload.request_rate)
    requests: list[Request] = []
    for arrival, group in zip(arrivals, groups, strict=True):
        questions = [
            sample_length(rng, prefix.question_len, ratio) for _ in range(prefix.num_turns)
        ]
        system = system_lens[group]
        requests.append(
            Request(
                arrival_s=arrival,
                input_len=system + questions[0],
                output_len=sample_length(rng, workload.output_len, ratio),
                group=group,
                shared_len=system,
                followup_questions=tuple(questions[1:]),
            )
        )
    return requests


def bench_command(
    workload: Workload,
    *,
    model_path: str,
    base_url: str = "http://127.0.0.1:30000",
    request_rate: float | None = None,
    num_prompts: int | None = None,
    output_file: str | None = None,
    title: str | None = None,
) -> Command:
    """``python3 -m sglang.benchmark.serving`` that replays this workload against a server."""
    rate = workload.request_rate if request_rate is None else request_rate
    prefix = workload.shared_prefix
    backend = MULTI_TURN_BENCH_BACKEND if prefix and prefix.num_turns > 1 else "sglang"
    flags: dict[str, Any] = {
        "--backend": backend,
        "--base-url": base_url,
        "--model": model_path,
        "--dataset-name": workload.dataset_name,
    }
    if prefix is None:
        flags.update(
            {
                "--random-input-len": workload.input_len,
                "--random-output-len": workload.output_len,
                "--random-range-ratio": workload.range_ratio,
                "--num-prompts": num_prompts or workload.num_prompts,
            }
        )
    else:
        flags.update(
            {
                "--gsp-num-groups": prefix.num_groups,
                "--gsp-prompts-per-group": prefix.prompts_per_group,
                "--gsp-system-prompt-len": prefix.system_prompt_len,
                "--gsp-question-len": prefix.question_len,
                "--gsp-output-len": workload.output_len,
                "--gsp-range-ratio": workload.range_ratio,
                "--gsp-num-turns": prefix.num_turns if prefix.num_turns > 1 else None,
                "--gsp-group-distribution": prefix.distribution
                if prefix.distribution != "uniform"
                else None,
                "--gsp-zipf-alpha": prefix.zipf_alpha if prefix.distribution == "zipf" else None,
            }
        )
    flags.update(
        {
            "--request-rate": rate,
            "--max-concurrency": workload.max_concurrency,
            "--seed": workload.seed,
            "--output-file": output_file,
        }
    )
    note = f"负载 {workload.name}：{workload.describe()}"
    if prefix is not None and num_prompts:
        note += "；generated-shared-prefix 的请求数由 组数×每组条数 决定，--num-prompts 不生效"
    return tool_command(
        BENCH_SERVING_MODULE,
        flags,
        title=title or f"压测 {workload.name}",
        note=note,
    )


def workload_from_dict(payload: dict[str, Any]) -> Workload:
    data = dict(payload)
    known = {item for item in Workload.__dataclass_fields__}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown workload fields {unknown}; known: {sorted(known)}")
    prefix_payload = data.pop("shared_prefix", None)
    slo_payload = data.pop("slo", None) or {}
    prefix = SharedPrefix(**prefix_payload) if prefix_payload else None
    if prefix is not None:
        data.setdefault("input_len", prefix.system_prompt_len + prefix.question_len)
        data.setdefault("range_ratio", GSP_RANGE_RATIO_DEFAULT)
    return Workload(shared_prefix=prefix, slo=SLO(**slo_payload), **data)


def _workload_files() -> list[Path]:
    directory = resources.files("sgl_agent") / "data" / "workloads"
    return sorted(Path(str(item)) for item in directory.iterdir() if item.name.endswith(".yaml"))


def list_workloads() -> dict[str, str]:
    """Bundled workloads: file stem -> description."""
    listing: dict[str, str] = {}
    for path in _workload_files():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        listing[path.stem] = payload.get("description", "")
    return listing


def load_workload(name_or_path: str) -> Workload:
    """Load a bundled workload by name (``chat``) or a YAML file."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return workload_from_dict(yaml.safe_load(candidate.read_text(encoding="utf-8")))
    for path in _workload_files():
        if path.stem == name_or_path.strip().lower():
            return workload_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    raise FileNotFoundError(
        f"workload {name_or_path!r} not found; bundled: {sorted(list_workloads())} or pass a YAML path"
    )


WORKLOAD_REFERENCES = (
    source("bench_random_lengths"),
    source("bench_gsp"),
    source("bench_metrics"),
)
