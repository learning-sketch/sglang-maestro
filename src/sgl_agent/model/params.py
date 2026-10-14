"""Parameter counts and weight bytes derived from a :class:`ModelArch`."""

from __future__ import annotations

from dataclasses import dataclass

from sgl_agent.hardware.spec import bytes_per_element, normalize_dtype
from sgl_agent.model.config import ModelArch


@dataclass(frozen=True)
class ParamBreakdown:
    embedding: int
    lm_head: int
    attention_replicated_per_layer: int
    attention_sharded_per_layer: int
    attention_kv_proj_per_layer: int
    dense_mlp_per_layer: int
    router_per_layer: int
    routed_experts_per_layer: int
    single_expert: int
    shared_experts_per_layer: int
    norms_per_layer: int
    final_norm: int
    num_layers: int
    num_dense_layers: int
    num_moe_layers: int
    top_k: int
    mtp_module: int

    @property
    def attention_per_layer(self) -> int:
        return self.attention_replicated_per_layer + self.attention_sharded_per_layer

    @property
    def attention_total(self) -> int:
        return self.attention_per_layer * self.num_layers

    @property
    def dense_mlp_total(self) -> int:
        return self.dense_mlp_per_layer * self.num_dense_layers

    @property
    def routed_experts_total(self) -> int:
        return self.routed_experts_per_layer * self.num_moe_layers

    @property
    def shared_experts_total(self) -> int:
        return self.shared_experts_per_layer * self.num_moe_layers

    @property
    def router_total(self) -> int:
        return self.router_per_layer * self.num_moe_layers

    @property
    def norms_total(self) -> int:
        return self.norms_per_layer * self.num_layers + self.final_norm

    @property
    def total(self) -> int:
        return (
            self.embedding
            + self.lm_head
            + self.attention_total
            + self.dense_mlp_total
            + self.router_total
            + self.routed_experts_total
            + self.shared_experts_total
            + self.norms_total
        )

    @property
    def active(self) -> int:
        active_experts = self.single_expert * self.top_k * self.num_moe_layers
        return (
            self.embedding
            + self.lm_head
            + self.attention_total
            + self.dense_mlp_total
            + self.router_total
            + active_experts
            + self.shared_experts_total
            + self.norms_total
        )


def _attention_params(arch: ModelArch) -> tuple[int, int, int]:
    """Return (replicated, tensor-parallel-sharded, k/v projection subset of sharded) per layer."""
    attention = arch.attention
    hidden = arch.hidden_size
    if attention.is_mla:
        heads = attention.num_heads
        nope = int(attention.qk_nope_head_dim or 0)
        rope = int(attention.qk_rope_head_dim or 0)
        kv_lora = int(attention.kv_lora_rank or 0)
        v_dim = attention.v_head_dim
        replicated = hidden * (kv_lora + rope) + kv_lora
        if attention.q_lora_rank:
            q_lora = int(attention.q_lora_rank)
            replicated += hidden * q_lora + q_lora
            sharded = q_lora * heads * (nope + rope)
        else:
            sharded = hidden * heads * (nope + rope)
        sharded += kv_lora * heads * (nope + v_dim) + heads * v_dim * hidden
        if attention.sparse_indexer and attention.index_n_heads and attention.index_head_dim:
            q_input = int(attention.q_lora_rank or hidden)
            replicated += (
                q_input * attention.index_n_heads * attention.index_head_dim
                + hidden * attention.index_head_dim
                + 2 * attention.index_head_dim
                + hidden * attention.index_n_heads
            )
        return replicated, sharded, 0
    head_dim = attention.head_dim
    q_proj = hidden * attention.num_heads * head_dim
    kv_proj = 2 * hidden * attention.num_kv_heads * head_dim
    o_proj = attention.num_heads * head_dim * hidden
    bias = (
        (attention.num_heads + 2 * attention.num_kv_heads) * head_dim
        if attention.has_qkv_bias
        else 0
    )
    qk_norm = 2 * head_dim if attention.has_qk_norm else 0
    return qk_norm, q_proj + kv_proj + o_proj + bias, kv_proj


def count_params(arch: ModelArch) -> ParamBreakdown:
    hidden = arch.hidden_size
    mlp_matrices = 3 if arch.gated_mlp else 2
    replicated, sharded, kv_proj = _attention_params(arch)
    embedding = arch.vocab_size * hidden
    lm_head = 0 if arch.tie_word_embeddings else arch.vocab_size * hidden
    moe = arch.moe
    single_expert = mlp_matrices * hidden * moe.expert_intermediate_size if moe else 0
    routed = single_expert * moe.num_experts if moe else 0
    shared = mlp_matrices * hidden * moe.shared_intermediate_size if moe else 0
    router = (
        (hidden * moe.num_experts + (moe.num_experts if moe.has_router_bias else 0)) if moe else 0
    )
    norm_width = 2 if arch.norm_type == "layernorm" else 1
    norms_per_layer = 2 * hidden * norm_width
    mtp_module = 0
    if arch.num_mtp_layers:
        decoder_layer = replicated + sharded + norms_per_layer
        decoder_layer += (
            (router + routed + shared)
            if moe
            else mlp_matrices * hidden * arch.dense_intermediate_size
        )
        mtp_module = decoder_layer + 2 * hidden * hidden + 3 * hidden
    return ParamBreakdown(
        embedding=embedding,
        lm_head=lm_head,
        attention_replicated_per_layer=replicated,
        attention_sharded_per_layer=sharded,
        attention_kv_proj_per_layer=kv_proj,
        dense_mlp_per_layer=mlp_matrices * hidden * arch.dense_intermediate_size,
        router_per_layer=router,
        routed_experts_per_layer=routed,
        single_expert=single_expert,
        shared_experts_per_layer=shared,
        norms_per_layer=norms_per_layer,
        final_norm=hidden * norm_width,
        num_layers=arch.num_layers,
        num_dense_layers=arch.num_dense_layers,
        num_moe_layers=arch.num_moe_layers,
        top_k=moe.top_k if moe else 0,
        mtp_module=mtp_module,
    )


@dataclass(frozen=True)
class WeightPrecision:
    """Bytes per parameter for each weight family once loaded by SGLang."""

    linear: float
    experts: float
    unquantized: float
    linear_dtype: str
    experts_dtype: str

    @property
    def describe(self) -> str:
        if self.linear_dtype == self.experts_dtype:
            return self.linear_dtype
        return f"linear={self.linear_dtype}, experts={self.experts_dtype}"


def weight_precision(arch: ModelArch, override_dtype: str | None = None) -> WeightPrecision:
    """Embeddings, norms and routers stay in the activation dtype; linear layers follow the checkpoint."""
    unquantized = bytes_per_element(arch.torch_dtype)
    activation_dtype = arch.activation_dtype
    if override_dtype:
        dtype = normalize_dtype(override_dtype)
        per_param = bytes_per_element(dtype)
        if dtype == "fp8":
            per_param += 4.0 / (128 * 128)
        elif dtype == "int4":
            per_param += 2.0 / 128
        return WeightPrecision(per_param, per_param, unquantized, dtype, dtype)
    quant = arch.quant
    if quant is None:
        return WeightPrecision(
            unquantized, unquantized, unquantized, activation_dtype, activation_dtype
        )
    quantized = quant.weight_bytes_per_param()
    if quant.applies_to == "experts_only":
        return WeightPrecision(
            unquantized, quantized, unquantized, activation_dtype, quant.weight_dtype
        )
    return WeightPrecision(
        quantized, quantized, unquantized, quant.weight_dtype, quant.weight_dtype
    )


def total_weight_bytes(arch: ModelArch, override_dtype: str | None = None) -> float:
    params = count_params(arch)
    precision = weight_precision(arch, override_dtype)
    linear = params.attention_total + params.dense_mlp_total + params.shared_experts_total
    return (
        linear * precision.linear
        + params.routed_experts_total * precision.experts
        + (params.embedding + params.lm_head + params.norms_total + params.router_total)
        * precision.unquantized
    )
