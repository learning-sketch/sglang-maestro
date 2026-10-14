"""Shared context and metrics for the framework-feature advisors.

Every advisor answers three questions for one model on one device under one workload: can the
feature run here (SGLang's own validation rules plus what an out-of-tree platform must provide),
what does it buy (KV capacity, decode throughput, TTFT, TPOT from the step cost model) and how
to tune it (flags with values derived from the same numbers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sgl_agent.adapt.platform import PlatformProfile, resolve_platform
from sgl_agent.adapt.precision import precision_plan
from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.knowledge import SGLangKnowledge
from sgl_agent.model.config import ModelArch
from sgl_agent.model.ops import BatchShape, Calibration, ExecutionConfig, StepCostModel
from sgl_agent.model.parallel import (
    MemoryKnobs,
    MemoryPlan,
    ParallelConfig,
    per_rank_chunked_prefill_size,
    plan_memory,
)
from sgl_agent.workload import Workload

STATUSES = ("recommend", "optional", "not_recommended", "unsupported")
MAX_DECODE_BATCH_PER_RANK = 256
"""Upper bound on the per-rank decode batch the comparisons consider (decode graph range)."""
DEFAULT_HOST_MEMORY_GB = 1024.0
"""Host DRAM per node assumed for HiCache sizing when the user does not pass one."""


@dataclass
class FeatureAdvice:
    key: str
    title: str
    status: str
    summary: str
    reasons: list[str] = field(default_factory=list)
    adaptation: list[str] = field(default_factory=list)
    """What the SGLang platform plugin (or the deployment) must provide for this feature here."""
    tuning: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    estimate: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    """SGLang startup errors this feature would hit with the given layout."""
    references: list[str] = field(default_factory=list)
    requires_adaptation: bool = False
    layout: ParallelConfig | None = field(default=None, repr=False)
    """Instance layout the advice settles on (lets later advisors build on it)."""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; known: {STATUSES}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "summary": self.summary,
            "reasons": self.reasons,
            "adaptation": self.adaptation,
            "tuning": self.tuning,
            "flags": self.flags,
            "env": self.env,
            "estimate": self.estimate,
            "problems": self.problems,
            "references": self.references,
            "requires_adaptation": self.requires_adaptation,
            "layout": self.layout.describe() if self.layout else None,
        }


@dataclass(frozen=True)
class DecodePoint:
    """Steady decode at a given per-rank batch on one instance."""

    parallel: ParallelConfig
    batch_per_rank: int
    context_len: float
    step_ms: float
    kv_tokens_per_rank: int
    kv_bound: bool

    @property
    def total_batch(self) -> int:
        return self.batch_per_rank * self.parallel.attn_dp

    @property
    def tokens_per_s(self) -> float:
        return self.total_batch / self.step_ms * 1000 if self.step_ms > 0 else 0.0

    @property
    def tokens_per_s_per_gpu(self) -> float:
        return self.tokens_per_s / self.parallel.num_gpus

    def to_dict(self) -> dict[str, Any]:
        return {
            "layout": self.parallel.describe(),
            "batch_per_rank": self.batch_per_rank,
            "total_batch": self.total_batch,
            "context_len": round(self.context_len),
            "step_ms": round(self.step_ms, 3),
            "tokens_per_s": round(self.tokens_per_s, 1),
            "tokens_per_s_per_gpu": round(self.tokens_per_s_per_gpu, 1),
            "kv_tokens_per_rank": self.kv_tokens_per_rank,
            "kv_bound": self.kv_bound,
        }


@dataclass
class FeatureContext:
    arch: ModelArch
    hardware: HardwareSpec
    knowledge: SGLangKnowledge
    profile: PlatformProfile
    parallel: ParallelConfig
    workload: Workload
    model_path: str
    weight_dtype: str | None = None
    kv_dtype: str | None = None
    chunked_prefill_size: int | None = None
    """``--chunked-prefill-size`` the precision route needs (None keeps SGLang's default)."""
    base_flags: dict[str, Any] = field(default_factory=dict)
    """Server flags of the baseline launch (precision route flags, chunk size)."""
    host_memory_gb: float = DEFAULT_HOST_MEMORY_GB
    calibration: Calibration = field(default_factory=Calibration)
    _costs: dict[ParallelConfig, StepCostModel] = field(default_factory=dict, repr=False)
    _plans: dict[tuple, MemoryPlan] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ platform
    @property
    def is_cuda(self) -> bool:
        return self.profile.in_tree and self.profile.platform == "cuda"

    @property
    def out_of_tree(self) -> bool:
        return self.profile.out_of_tree

    @property
    def device_type(self) -> str:
        return self.profile.device_type

    @property
    def sm_major(self) -> int | None:
        capability = self.hardware.software.compute_capability
        if not capability:
            return None
        try:
            return int(str(capability).split(".")[0])
        except ValueError:
            return None

    def has_a2a(self, backend: str) -> bool:
        return backend in self.hardware.software.moe_a2a_backends

    # ------------------------------------------------------------------ models
    def cost(self, parallel: ParallelConfig) -> StepCostModel:
        model = self._costs.get(parallel)
        if model is None:
            model = StepCostModel(
                self.arch,
                self.hardware,
                parallel,
                ExecutionConfig(weight_dtype=self.weight_dtype, kv_dtype=self.kv_dtype),
                self.calibration,
            )
            self._costs[parallel] = model
        return model

    def memory(
        self,
        parallel: ParallelConfig,
        *,
        chunked_prefill_size: int | None = None,
        include_mtp: bool = False,
        role: str = "colocated",
        max_running_requests: int | None = None,
        speculative_num_draft_tokens: int | None = None,
    ) -> MemoryPlan:
        chunk = (
            chunked_prefill_size if chunked_prefill_size is not None else self.chunked_prefill_size
        )
        key = (
            parallel,
            chunk,
            include_mtp,
            role,
            max_running_requests,
            speculative_num_draft_tokens,
        )
        plan = self._plans.get(key)
        if plan is None:
            plan = plan_memory(
                self.arch,
                self.hardware,
                parallel,
                weight_dtype=self.weight_dtype,
                kv_dtype=self.kv_dtype,
                knobs=MemoryKnobs(
                    chunked_prefill_size=chunk,
                    role=role,
                    max_running_requests=max_running_requests,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                ),
                include_mtp=include_mtp,
            )
            self._plans[key] = plan
        return plan

    # ------------------------------------------------------------------ workload metrics
    @property
    def decode_context(self) -> float:
        """Mean tokens a running request holds while decoding (prompt + half the answer)."""
        return self.workload.mean_input_len + self.workload.mean_output_len / 2

    @property
    def long_prompt(self) -> int:
        """Longest prompt the workload sends (TTFT worst case)."""
        prefix = self.workload.shared_prefix
        if prefix is None:
            return self.workload.input_len
        return prefix.system_prompt_len + prefix.question_len

    def per_rank_chunk(
        self, parallel: ParallelConfig, chunked_prefill_size: int | None = None
    ) -> int:
        plan = self.memory(parallel, chunked_prefill_size=chunked_prefill_size)
        return per_rank_chunked_prefill_size(plan.chunked_prefill_size, parallel)

    def decode_point(
        self,
        parallel: ParallelConfig,
        *,
        batch_per_rank: int | None = None,
        context_len: float | None = None,
        plan: MemoryPlan | None = None,
    ) -> DecodePoint | None:
        """Decode at ``batch_per_rank`` (default: the largest batch the KV pool holds)."""
        context = context_len or self.decode_context
        plan = plan or self.memory(parallel)
        if not plan.feasible:
            return None
        kv_cap = int(plan.max_total_tokens // max(context, 1.0))
        if kv_cap < 1:
            return None
        if batch_per_rank is None:
            batch = min(kv_cap, MAX_DECODE_BATCH_PER_RANK)
        else:
            batch = max(1, min(batch_per_rank, kv_cap))
        step = self.cost(parallel).step_time_us(BatchShape.decode(batch, context)) / 1000
        return DecodePoint(
            parallel=parallel,
            batch_per_rank=batch,
            context_len=context,
            step_ms=step,
            kv_tokens_per_rank=plan.max_total_tokens,
            kv_bound=kv_cap <= MAX_DECODE_BATCH_PER_RANK,
        )

    def decode_step_ms(
        self, parallel: ParallelConfig, total_batch: int, context_len: float | None = None
    ) -> float:
        per_rank = max(1, math.ceil(total_batch / parallel.attn_dp))
        shape = BatchShape.decode(per_rank, context_len or self.decode_context)
        return self.cost(parallel).step_time_us(shape) / 1000

    def prefill_ms(
        self,
        parallel: ParallelConfig,
        prompt_len: int,
        *,
        prefix_len: int = 0,
        chunk: int | None = None,
    ) -> float:
        """One request prefilled alone in chunks of the per-rank chunk size (TTFT without queueing)."""
        chunk = chunk or self.per_rank_chunk(parallel)
        if chunk <= 0:
            chunk = prompt_len
        model = self.cost(parallel)
        done, total_us = prefix_len, 0.0
        remaining = max(prompt_len - prefix_len, 1)
        while remaining > 0:
            size = min(chunk, remaining)
            total_us += model.step_time_us(BatchShape.prefill(1, size, done))
            done += size
            remaining -= size
        return total_us / 1000

    def concurrency_demand(self, parallel: ParallelConfig) -> tuple[float | None, int]:
        """``(demand, capacity)``: Little's-law concurrency of the arrival rate and the KV cap.

        ``demand`` is None in throughput mode (no arrival rate); a demand above the capacity means
        requests queue without bound on one instance.
        """
        point = self.decode_point(parallel)
        if point is None:
            return None, 0
        cap = point.total_batch
        rate = self.workload.effective_request_rate
        if rate is None:
            return None, cap
        ttft_s = self.prefill_ms(parallel, int(self.workload.mean_input_len)) / 1000
        demand = 1.0
        for _ in range(30):
            tpot_s = self.decode_step_ms(parallel, min(max(math.ceil(demand), 1), cap)) / 1000
            updated = rate * (ttft_s + self.workload.mean_output_len * tpot_s)
            if abs(updated - demand) < 0.5:
                demand = updated
                break
            demand = updated
        return demand, cap

    def operating_batch(self, parallel: ParallelConfig) -> int:
        """Concurrency the workload's arrival rate produces (Little's law), capped by KV capacity."""
        demand, cap = self.concurrency_demand(parallel)
        if cap == 0:
            return 1
        limit = self.workload.max_concurrency
        if demand is None:
            return min(cap, limit) if limit else cap
        batch = int(min(max(math.ceil(demand), 1), cap))
        return min(batch, limit) if limit else batch


def build_context(
    arch: ModelArch,
    hardware: HardwareSpec,
    knowledge: SGLangKnowledge,
    workload: Workload,
    *,
    model_path: str | None = None,
    parallel: ParallelConfig | None = None,
    kv_dtype: str | None = None,
    host_memory_gb: float = DEFAULT_HOST_MEMORY_GB,
    calibration: Calibration | None = None,
) -> FeatureContext:
    """Baseline from ``parallel`` or from the recommended precision route of scenario 1."""
    profile = resolve_platform(hardware, knowledge)
    plan = precision_plan(arch, hardware, profile)
    route = plan.recommended
    weight_dtype = route.weight_override if route else None
    base_flags: dict[str, Any] = dict(route.flags) if route else {}
    chunk = None
    if parallel is None:
        if route is None or route.sizing is None:
            raise ValueError(
                f"{arch.name} 在 {hardware.name} 上没有可用的精度路线（见 sgl-agent adapt），"
                "请用 --tp 指定布局"
            )
        parallel = route.sizing.parallel
        chunk = route.sizing.chunked_prefill_size
    elif route is not None and route.sizing is not None and route.sizing.parallel == parallel:
        chunk = route.sizing.chunked_prefill_size
    if chunk is not None:
        base_flags["--chunked-prefill-size"] = chunk
    if host_memory_gb <= 0:
        raise ValueError("host_memory_gb must be > 0")
    return FeatureContext(
        arch=arch,
        hardware=hardware,
        knowledge=knowledge,
        profile=profile,
        parallel=parallel,
        workload=workload,
        model_path=model_path or arch.hf_repo or arch.name,
        weight_dtype=weight_dtype,
        kv_dtype=kv_dtype,
        chunked_prefill_size=chunk,
        base_flags=base_flags,
        host_memory_gb=host_memory_gb,
        calibration=calibration or Calibration(),
    )


def ratio(new: float, old: float) -> float:
    return new / old if old > 0 else math.inf


def pct(value: float) -> str:
    return f"{(value - 1) * 100:+.0f}%"
