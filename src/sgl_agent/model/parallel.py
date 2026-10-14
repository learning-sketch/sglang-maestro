"""Parallel layouts as SGLang resolves them, and the per-GPU memory they imply."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from sgl_agent.hardware.spec import HardwareSpec, bytes_per_element, normalize_dtype
from sgl_agent.model.config import ModelArch
from sgl_agent.model.params import count_params, weight_precision

MIB = 1024 * 1024

DEEPEP_LOW_LATENCY_DEFAULT_TOKENS_PER_RANK = 128
"""``SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`` default; DeepEP asserts it is ≤ 1024."""


@dataclass(frozen=True)
class ParallelConfig:
    """One SGLang server instance.

    ``tp`` is the TP world (``--tp-size``); attention is laid out as
    ``tp = attn_dp * attn_cp * attn_tp`` (``--attn-dp-size``, ``--attn-cp-size``), DCP stripes
    KV tokens inside one attention-TP group (``--dcp-size``) and MoE experts are split by ``--ep-size``.
    ``moe_a2a`` is ``--moe-a2a-backend``; ``auto`` means DeepEP whenever ``ep > 1``.
    """

    tp: int = 1
    attn_dp: int = 1
    attn_cp: int = 1
    dcp: int = 1
    ep: int = 1
    pp: int = 1
    dense_mlp_tp: int | None = None
    moe_a2a: str = "auto"

    @property
    def num_gpus(self) -> int:
        return self.tp * self.pp

    @property
    def attn_tp(self) -> int:
        return max(1, self.tp // (self.attn_dp * self.attn_cp))

    @property
    def moe_tp(self) -> int:
        return max(1, self.tp // self.ep)

    @property
    def effective_dense_mlp_tp(self) -> int:
        return self.dense_mlp_tp or self.tp

    @property
    def moe_a2a_backend(self) -> str:
        if self.moe_a2a == "auto":
            return "deepep" if self.ep > 1 else "none"
        return self.moe_a2a

    @property
    def uses_a2a(self) -> bool:
        """Tokens are dispatched to expert owners (DeepEP-style) instead of gathered and reduced."""
        return self.moe_a2a_backend != "none" and self.ep > 1

    def with_(self, **changes: int | str | None) -> ParallelConfig:
        return replace(self, **changes)

    def describe(self) -> str:
        parts = [f"TP{self.tp}"]
        if self.attn_dp > 1:
            parts.append(f"DPA{self.attn_dp}")
        if self.attn_cp > 1:
            parts.append(f"CP{self.attn_cp}")
        if self.dcp > 1:
            parts.append(f"DCP{self.dcp}")
        if self.ep > 1:
            parts.append(f"EP{self.ep}" + (f"({self.moe_a2a_backend})" if self.uses_a2a else ""))
        if self.pp > 1:
            parts.append(f"PP{self.pp}")
        return "·".join(parts)

    def validate(self, arch: ModelArch) -> list[str]:
        errors: list[str] = []
        attention = arch.attention
        for name, value in (
            ("tp", self.tp),
            ("attn_dp", self.attn_dp),
            ("attn_cp", self.attn_cp),
            ("dcp", self.dcp),
            ("ep", self.ep),
            ("pp", self.pp),
        ):
            if value < 1:
                errors.append(f"{name} 必须 ≥ 1")
        if errors:
            return errors
        if self.tp % (self.attn_dp * self.attn_cp) != 0:
            errors.append(
                f"tp={self.tp} 必须能被 attn_dp×attn_cp={self.attn_dp * self.attn_cp} 整除"
            )
        if self.dcp > 1:
            if self.attn_tp % self.dcp != 0:
                errors.append(
                    f"DCP 组必须嵌套在 attention-TP 组内：attn_tp={self.attn_tp} 不能被 dcp={self.dcp} 整除"
                )
            elif not attention.is_mla and attention.num_kv_heads > self.attn_tp // self.dcp:
                errors.append(
                    f"MHA/GQA 的 DCP 要求同一 DCP 组的 rank 持有相同 KV 头：kv_heads={attention.num_kv_heads} "
                    f"必须 ≤ attn_tp/dcp={self.attn_tp // self.dcp}（否则 KV 会被写坏）"
                )
        if attention.num_heads % self.attn_tp != 0:
            errors.append(f"注意力头数 {attention.num_heads} 不能被 attn_tp={self.attn_tp} 整除")
        if not attention.is_mla:
            kv_heads = attention.num_kv_heads
            if self.attn_tp > kv_heads and self.attn_tp % kv_heads != 0:
                errors.append(
                    f"attn_tp={self.attn_tp} > kv_heads={kv_heads} 时必须是其整数倍（KV 头复制）"
                )
            if self.attn_tp <= kv_heads and kv_heads % self.attn_tp != 0:
                errors.append(f"kv_heads={kv_heads} 不能被 attn_tp={self.attn_tp} 整除")
        if self.ep > 1:
            if arch.moe is None:
                errors.append("EP 只适用于 MoE 模型")
            elif arch.moe.num_experts % self.ep != 0:
                errors.append(f"专家数 {arch.moe.num_experts} 不能被 ep={self.ep} 整除")
            if self.tp % self.ep != 0:
                errors.append(f"tp={self.tp} 必须能被 ep={self.ep} 整除")
        if self.moe_a2a_backend != "none":
            if arch.moe is None:
                errors.append(f"--moe-a2a-backend {self.moe_a2a_backend} 只适用于 MoE 模型")
            elif self.ep != self.tp:
                errors.append(
                    f"--moe-a2a-backend {self.moe_a2a_backend} 会被 SGLang 强制成 ep_size=tp_size={self.tp}，"
                    f"请把 ep 设为 {self.tp}"
                )
        if (
            arch.moe is not None
            and self.moe_tp > 1
            and arch.moe.expert_intermediate_size % self.moe_tp != 0
        ):
            errors.append(
                f"专家中间维 {arch.moe.expert_intermediate_size} 不能被 moe_tp={self.moe_tp} 整除"
            )
        if self.pp > arch.num_layers:
            errors.append("pp 不能超过层数")
        return errors


@dataclass(frozen=True)
class MemoryKnobs:
    """SGLang settings that decide ``mem_fraction_static``; ``None`` means "SGLang's default"."""

    mem_fraction_static: float | None = None
    chunked_prefill_size: int | None = None
    cuda_graph_max_bs: int | None = None
    max_running_requests: int | None = None
    speculative_num_draft_tokens: int | None = None
    role: str = "colocated"
    decode_cuda_graph: bool = True
    prefill_cuda_graph: bool = True
    max_prefill_tokens: int = 16384

    def __post_init__(self) -> None:
        if self.role not in {"colocated", "prefill", "decode"}:
            raise ValueError(f"role must be colocated/prefill/decode, got {self.role!r}")
        if self.mem_fraction_static is not None and not 0 < self.mem_fraction_static < 1:
            raise ValueError("mem_fraction_static must be in (0, 1)")


@dataclass(frozen=True)
class MemoryFractionEstimate:
    mem_fraction_static: float
    reserved_mb: float
    chunked_prefill_size: int
    cuda_graph_max_bs: int
    breakdown_mb: dict[str, float]


@dataclass(frozen=True)
class MemoryPlan:
    weights_bytes_per_gpu: float
    kv_bytes_per_token_per_gpu: float
    mem_fraction_static: float
    mem_fraction_is_default: bool
    reserved_bytes_per_gpu: float
    kv_pool_bytes_per_gpu: float
    max_total_tokens: int
    chunked_prefill_size: int
    cuda_graph_max_bs: int
    breakdown: dict[str, float]
    reserve_breakdown_mb: dict[str, float] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.max_total_tokens > 0


def kv_heads_per_rank(arch: ModelArch, attn_tp: int, dcp: int = 1) -> int:
    """``ModelConfig.get_num_kv_heads``: DCP ranks share heads, so heads shard over attn_tp/dcp."""
    return max(1, arch.attention.num_kv_heads // max(attn_tp // dcp, 1))


def shared_expert_tp(parallel: ParallelConfig) -> int:
    """a2a backends replicate the shared expert on every rank; otherwise it is TP-sharded."""
    return 1 if parallel.uses_a2a else parallel.tp


def per_rank_chunked_prefill_size(chunked_prefill_size: int, parallel: ParallelConfig) -> int:
    """SGLang divides ``--chunked-prefill-size`` by the attention-DP size after sizing memory."""
    if chunked_prefill_size <= 0:
        return chunked_prefill_size
    return max(chunked_prefill_size // parallel.attn_dp, 1)


def kv_bytes_per_token_per_gpu(
    arch: ModelArch, parallel: ParallelConfig, kv_dtype: str | None = None
) -> float:
    """KV-cache bytes one token occupies on one GPU, averaged over layers (sliding layers excluded)."""
    element = bytes_per_element(kv_dtype or arch.torch_dtype)
    attention = arch.attention
    layers_per_stage = arch.num_layers / parallel.pp
    full_layers = layers_per_stage * (1 - attention.num_sliding_layers / arch.num_layers)
    if attention.is_mla:
        per_layer = attention.mla_latent_dim * element / parallel.dcp
        if attention.sparse_indexer and attention.index_head_dim:
            per_layer += attention.index_head_dim * 1.0 + 4.0
        return full_layers * per_layer
    per_layer = (
        2
        * kv_heads_per_rank(arch, parallel.attn_tp, parallel.dcp)
        * attention.head_dim
        * element
        / parallel.dcp
    )
    return full_layers * per_layer


def weights_bytes_per_gpu(
    arch: ModelArch,
    parallel: ParallelConfig,
    weight_dtype: str | None = None,
    include_mtp: bool = False,
) -> dict[str, float]:
    params = count_params(arch)
    precision = weight_precision(arch, weight_dtype)
    attention = arch.attention
    layers = arch.num_layers / parallel.pp
    moe_layers = arch.num_moe_layers / parallel.pp
    dense_layers = arch.num_dense_layers / parallel.pp
    attn_tp = parallel.attn_tp

    sharded = params.attention_sharded_per_layer
    if not attention.is_mla and attn_tp > attention.num_kv_heads:
        kv_share = params.attention_kv_proj_per_layer * (1 / attention.num_kv_heads - 1 / attn_tp)
        sharded_per_gpu = sharded / attn_tp + kv_share
    else:
        sharded_per_gpu = sharded / attn_tp
    attention_bytes = layers * (
        params.attention_replicated_per_layer * precision.unquantized
        if not attention.is_mla
        else params.attention_replicated_per_layer * precision.linear
    )
    attention_bytes += layers * sharded_per_gpu * precision.linear
    dense_bytes = (
        dense_layers
        * params.dense_mlp_per_layer
        * precision.linear
        / parallel.effective_dense_mlp_tp
    )
    experts_bytes = moe_layers * params.routed_experts_per_layer * precision.experts / parallel.tp
    shared_bytes = (
        moe_layers * params.shared_experts_per_layer * precision.linear / shared_expert_tp(parallel)
    )
    router_bytes = moe_layers * params.router_per_layer * precision.unquantized
    vocab_bytes = (params.embedding + params.lm_head) * precision.unquantized / parallel.tp
    norm_bytes = params.norms_total * precision.unquantized / parallel.pp
    breakdown = {
        "attention": attention_bytes,
        "dense_mlp": dense_bytes,
        "routed_experts": experts_bytes,
        "shared_experts": shared_bytes,
        "router": router_bytes,
        "embedding_lm_head": vocab_bytes,
        "norms": norm_bytes,
    }
    if include_mtp and params.mtp_module:
        breakdown["mtp_draft"] = params.mtp_module * precision.linear / parallel.tp
    return breakdown


def sglang_memory_defaults(hardware: HardwareSpec, tp: int) -> tuple[int, int]:
    """``(chunked_prefill_size, decode cuda-graph max_bs)`` SGLang picks from the GPU memory size."""
    gpu_mem_mb = hardware.memory_capacity_bytes / MIB
    if gpu_mem_mb < 20 * 1024:
        return 2048, 8
    if gpu_mem_mb < 35 * 1024:
        return 4096, 48 if tp < 4 else 160
    if gpu_mem_mb < 60 * 1024:
        return 4096, 32 if tp < 4 else 160
    if gpu_mem_mb < 160 * 1024:
        return 8192, 256 if tp < 4 else 512
    return 16384, 512


def prefill_graph_capture_sizes(max_bs: int) -> list[int]:
    """Token buckets SGLang captures for the (breakable) prefill CUDA graph."""
    sizes = (
        list(range(4, 33, 4))
        + list(range(48, 257, 16))
        + list(range(288, 513, 32))
        + list(range(576, 1025, 64))
        + list(range(1280, 4097, 256))
        + list(range(4608, max_bs + 1, 512))
    )
    return [size for size in sizes if size <= max_bs]


def estimate_mem_fraction_static(
    hardware: HardwareSpec,
    parallel: ParallelConfig,
    *,
    is_mla: bool,
    knobs: MemoryKnobs | None = None,
) -> MemoryFractionEstimate:
    """Mirror of SGLang's ``handle_gpu_memory_settings`` (``arg_groups/memory_hook.py``).

    Covers the default path (post-capture KV sizing is opt-in via an env var) on CUDA; VLM and
    XPU adjustments are not modelled.
    """
    knobs = knobs or MemoryKnobs()
    default_chunk, default_max_bs = sglang_memory_defaults(hardware, parallel.tp)
    chunked = default_chunk if knobs.chunked_prefill_size is None else knobs.chunked_prefill_size
    max_bs = knobs.cuda_graph_max_bs
    if max_bs is None:
        max_bs = default_max_bs
        if knobs.max_running_requests is not None and max_bs > knobs.max_running_requests:
            max_bs = knobs.max_running_requests
    role = knobs.role
    if role == "decode":
        running = knobs.max_running_requests or max_bs or 1
        activation_tokens = max(running * (knobs.speculative_num_draft_tokens or 1), 2048)
    elif chunked > 0:
        activation_tokens = max(chunked, 2048)
    else:
        activation_tokens = max(knobs.max_prefill_tokens, 2048)

    breakdown = {
        "constant": 512.0,
        "activations": activation_tokens * 1.5,
        "parallel_slack": parallel.tp * parallel.pp / 8 * 1024,
        "decode_graph": 0.0,
        "dp_attention_padding": 0.0,
        "prefill_graph": 0.0,
    }
    if role != "prefill" and knobs.decode_cuda_graph:
        breakdown["decode_graph"] = max_bs * 2.0
    if parallel.attn_dp > 1 and role != "prefill":
        padding = max_bs * parallel.attn_dp * 3.0
        if max_bs > 300:
            padding += max_bs * parallel.attn_dp * 1.5
        breakdown["dp_attention_padding"] = padding
    if role != "decode" and knobs.prefill_cuda_graph:
        prefill_graph = 1.5 * 1024 if is_mla else len(prefill_graph_capture_sizes(chunked)) * 8.0
        if parallel.moe_a2a_backend == "deepep":
            prefill_graph += 1024
        breakdown["prefill_graph"] = prefill_graph
    reserved = sum(breakdown.values())
    gpu_mem_mb = hardware.memory_capacity_bytes / MIB
    if gpu_mem_mb > 60 * 1024 and reserved < 10 * 1024:
        breakdown["floor_10gib"] = 10 * 1024 - reserved
        reserved = 10 * 1024
    if role != "prefill" and knobs.decode_cuda_graph and parallel.moe_a2a_backend == "deepep":
        breakdown["deepep_buffers"] = 2 * 1024
        reserved += 2 * 1024
    return MemoryFractionEstimate(
        mem_fraction_static=round((gpu_mem_mb - reserved) / gpu_mem_mb, 3),
        reserved_mb=reserved,
        chunked_prefill_size=chunked,
        cuda_graph_max_bs=max_bs,
        breakdown_mb=breakdown,
    )


def plan_memory(
    arch: ModelArch,
    hardware: HardwareSpec,
    parallel: ParallelConfig,
    *,
    weight_dtype: str | None = None,
    kv_dtype: str | None = None,
    knobs: MemoryKnobs | None = None,
    include_mtp: bool = False,
) -> MemoryPlan:
    if kv_dtype:
        normalize_dtype(kv_dtype)
    knobs = knobs or MemoryKnobs()
    estimate = estimate_mem_fraction_static(
        hardware, parallel, is_mla=arch.attention.is_mla, knobs=knobs
    )
    fraction = knobs.mem_fraction_static or estimate.mem_fraction_static
    breakdown = weights_bytes_per_gpu(arch, parallel, weight_dtype, include_mtp=include_mtp)
    weights = sum(breakdown.values())
    capacity = hardware.memory_capacity_bytes
    kv_pool = capacity * fraction - weights
    per_token = kv_bytes_per_token_per_gpu(arch, parallel, kv_dtype)
    max_tokens = int(kv_pool // per_token) if kv_pool > 0 and per_token > 0 else 0
    return MemoryPlan(
        weights_bytes_per_gpu=weights,
        kv_bytes_per_token_per_gpu=per_token,
        mem_fraction_static=fraction,
        mem_fraction_is_default=knobs.mem_fraction_static is None,
        reserved_bytes_per_gpu=capacity * (1 - fraction),
        kv_pool_bytes_per_gpu=max(kv_pool, 0.0),
        max_total_tokens=max_tokens,
        chunked_prefill_size=estimate.chunked_prefill_size,
        cuda_graph_max_bs=estimate.cuda_graph_max_bs,
        breakdown=breakdown,
        reserve_breakdown_mb=estimate.breakdown_mb,
    )


def enumerate_parallel_configs(
    arch: ModelArch,
    num_gpus: int,
    *,
    allow_dpa: bool = True,
    allow_dcp: bool = True,
    allow_ep: bool = True,
) -> list[ParallelConfig]:
    """Valid single-instance layouts on ``num_gpus`` GPUs (PP=1), powers of two only."""
    if num_gpus < 1 or num_gpus & (num_gpus - 1):
        return []
    options: list[ParallelConfig] = []
    powers = [2**i for i in range(int(math.log2(num_gpus)) + 1)]
    expert_layouts = [(1, "none")]
    if allow_ep and arch.moe and num_gpus > 1:
        expert_layouts += [(num_gpus, "none"), (num_gpus, "deepep")]
    for attn_dp in powers if allow_dpa else [1]:
        for dcp in powers if allow_dcp else [1]:
            for ep, a2a in expert_layouts:
                candidate = ParallelConfig(
                    tp=num_gpus, attn_dp=attn_dp, dcp=dcp, ep=ep, moe_a2a=a2a
                )
                if candidate in options or candidate.validate(arch):
                    continue
                options.append(candidate)
    return options
