"""SGLang server instances as the deployment planner sees them.

An instance is one ``launch_server`` process group: a role (colocated, PD prefill or PD decode),
a parallel layout, the launch flags the feature advisors settled on, and what the simulator
needs from it: KV tokens per attention-DP rank, chunk size, running-request cap, speculative
settings, prefix-cache tiers, and step times from the cost model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from sgl_agent.features.base import FeatureContext
from sgl_agent.features.moe_cache import hicache_host_budget
from sgl_agent.features.speculative import spec_point
from sgl_agent.knowledge.facts import (
    CLIP_MAX_NEW_TOKENS,
    DEFAULT_MAX_PREFILL_TOKENS,
    HICACHE_DEFAULT_RATIO,
    MAX_RUNNING_REQUESTS_CEIL,
    MAX_RUNNING_REQUESTS_FLOOR,
    MTP_DRAFT_FROM_TARGET_ARCHITECTURES,
    PD_DECODE_EXTRA_SLOTS_MAX_BATCH,
    SPECULATIVE_DEFAULT_MAX_RUNNING_REQUESTS,
)
from sgl_agent.model.ops import BatchShape
from sgl_agent.model.parallel import ParallelConfig, per_rank_chunked_prefill_size

ROLES = ("colocated", "prefill", "decode")
PLACEMENT_FLAGS = frozenset(
    {
        "--host",
        "--port",
        "--base-gpu-id",
        "--nnodes",
        "--node-rank",
        "--dist-init-addr",
        "--disaggregation-mode",
        "--disaggregation-bootstrap-port",
        "--disaggregation-transfer-backend",
        "--disaggregation-ib-device",
    }
)
"""Flags that depend on where an instance runs; the planner sets them per placement."""
DEFAULT_CONTEXT_LEN = 4096


@dataclass(frozen=True)
class SpecConfig:
    algorithm: str
    num_steps: int
    topk: int
    draft_tokens: int
    accept_length: float
    draft_from_target: bool
    """MTP/NextN layers shipped in the target checkpoint (no separate draft model)."""

    def describe(self) -> str:
        return f"{self.algorithm} {self.num_steps} 步（接受长度 {self.accept_length:.2f}）"


@dataclass
class InstanceSpec:
    role: str
    layout: ParallelConfig
    flags: dict[str, Any]
    """Launch flags without the placement ones (``PLACEMENT_FLAGS``)."""
    env: dict[str, str]
    features: list[str]
    kv_tokens_per_rank: int
    chunk_per_rank: int
    max_prefill_tokens: int
    max_running_per_rank: int
    request_slots_per_rank: int
    """Requests a rank may hold: running ones plus, on a PD decode server, in-transfer ones."""
    cuda_graph_max_bs: int
    speculative: SpecConfig | None
    mixed_chunk: bool
    """Decode runs inside prefill batches (``--enable-mixed-chunk``, colocated, no speculation)."""
    radix_cache: bool
    host_cache_tokens_per_rank: int
    kv_bytes_per_token_per_gpu: float
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}; known: {ROLES}")

    @property
    def num_gpus(self) -> int:
        return self.layout.num_gpus

    @property
    def attn_dp(self) -> int:
        return self.layout.attn_dp

    def describe(self) -> str:
        extras = [self.speculative.algorithm] if self.speculative else []
        if self.host_cache_tokens_per_rank:
            extras.append("HiCache")
        if self.mixed_chunk:
            extras.append(f"mixed chunk {self.chunk_per_rank * self.attn_dp}")
        return self.layout.describe() + (f" + {' + '.join(extras)}" if extras else "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "layout": self.layout.describe(),
            "num_gpus": self.num_gpus,
            "flags": self.flags,
            "env": self.env,
            "features": self.features,
            "kv_tokens_per_rank": self.kv_tokens_per_rank,
            "chunk_per_rank": self.chunk_per_rank,
            "max_running_per_rank": self.max_running_per_rank,
            "request_slots_per_rank": self.request_slots_per_rank,
            "speculative": self.speculative.describe() if self.speculative else None,
            "mixed_chunk": self.mixed_chunk,
            "radix_cache": self.radix_cache,
            "host_cache_tokens_per_rank": self.host_cache_tokens_per_rank,
            "notes": self.notes,
        }


def _spec_config(ctx: FeatureContext, flags: dict[str, Any]) -> SpecConfig | None:
    algorithm = flags.get("--speculative-algorithm")
    if not algorithm:
        return None
    steps = int(flags.get("--speculative-num-steps", 3))
    topk = int(flags.get("--speculative-eagle-topk", 1))
    tokens = int(flags.get("--speculative-num-draft-tokens", steps + 1))
    from_target = (
        algorithm in {"NEXTN", "EAGLE"}
        and ctx.arch.primary_architecture in MTP_DRAFT_FROM_TARGET_ARCHITECTURES
        and ctx.arch.num_mtp_layers > 0
    )
    return SpecConfig(
        algorithm=str(algorithm),
        num_steps=steps,
        topk=topk,
        draft_tokens=tokens,
        accept_length=ctx.workload.accept_length(steps, topk),
        draft_from_target=from_target,
    )


def _max_running_per_rank(
    ctx: FeatureContext, flags: dict[str, Any], kv_tokens: int, layout: ParallelConfig, spec: bool
) -> int:
    """SGLang's per-DP-worker request cap (``resolve_max_num_reqs``)."""
    explicit = flags.get("--max-running-requests")
    if explicit is None and spec:
        explicit = SPECULATIVE_DEFAULT_MAX_RUNNING_REQUESTS
    if explicit is not None:
        return max(1, min(int(explicit) // layout.attn_dp, kv_tokens // 2))
    context_len = ctx.arch.max_position_embeddings or DEFAULT_CONTEXT_LEN
    estimated = int(kv_tokens / context_len * 512)
    estimated = max(min(estimated, MAX_RUNNING_REQUESTS_CEIL), MAX_RUNNING_REQUESTS_FLOOR)
    return max(1, min(estimated, kv_tokens // 2))


def _request_slots(flags: dict[str, Any], role: str, running: int, attn_dp: int) -> int:
    if role != "decode":
        return running
    extra = flags.get("--disaggregation-decode-extra-slots")
    if extra is None:
        explicit = flags.get("--max-running-requests")
        share = int(explicit) // attn_dp if explicit is not None else None
        extra = 2 * share if share is not None and share <= PD_DECODE_EXTRA_SLOTS_MAX_BATCH else 0
    return running + int(extra)


def build_instance(
    ctx: FeatureContext,
    layout: ParallelConfig,
    flags: dict[str, Any],
    *,
    role: str = "colocated",
    env: dict[str, str] | None = None,
    features: list[str] | None = None,
    notes: list[str] | None = None,
) -> InstanceSpec:
    flags = {key: value for key, value in flags.items() if key not in PLACEMENT_FLAGS}
    spec = _spec_config(ctx, flags)
    chunk = flags.get("--chunked-prefill-size")
    plan = ctx.memory(
        layout,
        chunked_prefill_size=int(chunk) if chunk is not None else None,
        role=role,
        include_mtp=bool(spec and spec.draft_from_target),
        max_running_requests=flags.get("--max-running-requests"),
        speculative_num_draft_tokens=spec.draft_tokens if spec else None,
    )
    kv_tokens = plan.max_total_tokens
    if spec is not None:
        arch = ctx.arch
        share = arch.num_mtp_layers if spec.draft_from_target else 1
        kv_tokens = int(kv_tokens / (1 + share / arch.num_layers))
    radix = role != "decode" and not flags.get("--disable-radix-cache")
    host_tokens = 0
    if radix and flags.get("--enable-hierarchical-cache") and plan.kv_pool_bytes_per_gpu > 0:
        ratio = flags.get("--hicache-ratio")
        if ratio is None:
            budget = hicache_host_budget(ctx, layout)["budget_bytes_per_rank"]
            ratio = min(HICACHE_DEFAULT_RATIO, budget / plan.kv_pool_bytes_per_gpu)
        host_tokens = int(float(ratio) * kv_tokens)
    running = _max_running_per_rank(ctx, flags, kv_tokens, layout, spec is not None)
    chunk_per_rank = per_rank_chunked_prefill_size(plan.chunked_prefill_size, layout)
    return InstanceSpec(
        role=role,
        layout=layout,
        flags=flags,
        env=dict(env or {}),
        features=list(features or []),
        kv_tokens_per_rank=kv_tokens,
        chunk_per_rank=chunk_per_rank,
        max_prefill_tokens=int(flags.get("--max-prefill-tokens", DEFAULT_MAX_PREFILL_TOKENS)),
        max_running_per_rank=running,
        request_slots_per_rank=_request_slots(flags, role, running, layout.attn_dp),
        cuda_graph_max_bs=plan.cuda_graph_max_bs,
        speculative=spec,
        mixed_chunk=bool(flags.get("--enable-mixed-chunk"))
        and role == "colocated"
        and chunk_per_rank > 0
        and spec is None,
        radix_cache=radix,
        host_cache_tokens_per_rank=host_tokens,
        kv_bytes_per_token_per_gpu=plan.kv_bytes_per_token_per_gpu,
        notes=list(notes or []),
    )


def _bucket(value: float) -> int:
    return int(math.log(max(value, 1.0)) / math.log(1.02))


class InstanceModel:
    """Step times of one instance's attention-DP rank, memoized for the simulator."""

    def __init__(self, ctx: FeatureContext, spec: InstanceSpec):
        self.ctx = ctx
        self.spec = spec
        self.cost = ctx.cost(spec.layout)
        self._decode: dict[tuple[int, int], float] = {}
        self._prefill: dict[tuple, float] = {}
        hardware = ctx.hardware
        link = hardware.host_link_bandwidth_gbps * 1e9 * hardware.efficiency.host_link
        self._host_bytes_per_us = link / 1e6 if link > 0 else 0.0
        self.tokens_per_step = spec.speculative.accept_length if spec.speculative else 1.0
        layers = ctx.arch.num_layers
        draft_layers = 1 if spec.speculative else 0
        self._prefill_factor = 1 + draft_layers / layers
        self._layers = layers

    @property
    def context_len(self) -> int:
        return self.ctx.arch.max_position_embeddings or 0

    @property
    def max_req_len(self) -> int:
        """SGLang's ``max_req_len``: prompt plus output tokens one request may reach."""
        limit = self.spec.kv_tokens_per_rank - 1
        if self.context_len:
            limit = min(limit, self.context_len - 1)
        return max(limit, 0)

    def prefill_us(
        self,
        chunks: list[tuple[int, int]],
        host_tokens: int = 0,
        decode: tuple[int, float] | None = None,
    ) -> float:
        """One prefill batch of ``(new_tokens, prefix_len)`` chunks plus host-cache loads;
        ``decode=(batch, sum_context)`` mixes the running batch in (``--enable-mixed-chunk``)."""
        shape = BatchShape.chunks(chunks)
        if decode is not None:
            shape = shape.merged_with(BatchShape.decode_total(*decode))
        key = (
            shape.phase,
            shape.num_seqs,
            _bucket(shape.new_tokens),
            _bucket(shape.sum_kv),
            _bucket(shape.sum_q_kv),
        )
        compute = self._prefill.get(key)
        if compute is None:
            compute = self.cost.step_time_us(shape) * self._prefill_factor
            if len(self._prefill) > 50_000:
                self._prefill.clear()
            self._prefill[key] = compute
        if host_tokens <= 0 or self._host_bytes_per_us <= 0:
            return compute
        load = host_tokens * self.spec.kv_bytes_per_token_per_gpu / self._host_bytes_per_us
        return max(compute, load) + load / self._layers

    def decode_us(self, batch: int, sum_context: float) -> float:
        """One decode step (or one draft/verify/extend iteration) for ``batch`` requests."""
        key = (batch, _bucket(sum_context))
        cached = self._decode.get(key)
        if cached is not None:
            return cached
        spec = self.spec.speculative
        context = sum_context / max(batch, 1)
        if spec is not None:
            point = spec_point(
                self.ctx,
                self.spec.layout,
                batch,
                num_steps=spec.num_steps,
                topk=spec.topk,
                draft_tokens=spec.draft_tokens,
                context_len=context,
                accept_length=spec.accept_length,
            )
            value = point.iteration_ms * 1000
        else:
            shape = BatchShape.decode_total(batch, sum_context)
            if batch <= self.spec.cuda_graph_max_bs:
                value = self.cost.step_time_us(shape)
            else:
                value = self.cost.estimate(shape, cuda_graph=False).total_us
        self._decode[key] = value
        return value

    def admission_reserve(self, remaining_output: float) -> float:
        """Decode tokens a new request reserves when it is admitted (colocated scheduler)."""
        if self.spec.role != "colocated":
            return 0.0
        return min(max(remaining_output, 0.0), CLIP_MAX_NEW_TOKENS)
