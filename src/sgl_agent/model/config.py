"""Parse a Hugging Face ``config.json`` into the architecture facts the agent reasons about."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from sgl_agent.hardware.spec import bytes_per_element, normalize_dtype

QK_NORM_MODEL_TYPES = frozenset(
    {"qwen3", "qwen3_moe", "qwen3_next", "olmo2", "olmoe", "gemma3", "gemma3_text", "glm4_moe"}
)
QKV_BIAS_MODEL_TYPES = frozenset({"qwen2", "qwen2_moe", "qwen2_vl", "qwen2_5_vl"})
NON_GATED_MLP_MODEL_TYPES = frozenset(
    {"gpt2", "gpt_bigcode", "gpt_neox", "falcon", "phi", "starcoder2", "bloom", "opt", "mpt"}
)
LAYERNORM_MODEL_TYPES = frozenset(
    {"gpt2", "gpt_bigcode", "gpt_neox", "falcon", "phi", "starcoder2", "bloom", "opt"}
)


def _first(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


@dataclass(frozen=True)
class AttentionArch:
    kind: str
    num_heads: int
    num_kv_heads: int
    head_dim: int
    v_head_dim: int
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    has_qk_norm: bool = False
    has_qkv_bias: bool = False
    sliding_window: int | None = None
    num_sliding_layers: int = 0
    sparse_indexer: bool = False
    index_n_heads: int | None = None
    index_head_dim: int | None = None
    rope_scaling_type: str | None = None

    @property
    def is_mla(self) -> bool:
        return self.kind == "MLA"

    @property
    def qk_head_dim(self) -> int:
        if self.is_mla:
            return int(self.qk_nope_head_dim or 0) + int(self.qk_rope_head_dim or 0)
        return self.head_dim

    @property
    def mla_latent_dim(self) -> int:
        """Per-token, per-layer MLA cache width: compressed KV plus the decoupled RoPE key."""
        return int(self.kv_lora_rank or 0) + int(self.qk_rope_head_dim or 0)


@dataclass(frozen=True)
class MoEArch:
    num_experts: int
    top_k: int
    expert_intermediate_size: int
    num_shared_experts: int
    shared_intermediate_size: int
    moe_layer_indices: tuple[int, ...]
    routing: str
    num_expert_groups: int | None = None
    topk_groups: int | None = None
    has_router_bias: bool = False


@dataclass(frozen=True)
class QuantArch:
    method: str
    weight_dtype: str
    activation_dtype: str | None
    block_shape: tuple[int, int] | None
    group_size: int | None
    applies_to: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def weight_bytes_per_param(self) -> float:
        """Bytes per quantized weight including scale overhead."""
        base = bytes_per_element(self.weight_dtype)
        if self.block_shape:
            return base + 4.0 / (self.block_shape[0] * self.block_shape[1])
        if self.group_size:
            return base + 2.0 / self.group_size
        return base


@dataclass(frozen=True)
class ModelArch:
    name: str
    architectures: tuple[str, ...]
    model_type: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    tie_word_embeddings: bool
    dense_intermediate_size: int
    gated_mlp: bool
    hidden_act: str
    norm_type: str
    attention: AttentionArch
    moe: MoEArch | None
    quant: QuantArch | None
    torch_dtype: str
    max_position_embeddings: int | None
    num_mtp_layers: int
    hf_repo: str | None = None
    warnings: tuple[str, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_moe(self) -> bool:
        return self.moe is not None

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe.moe_layer_indices) if self.moe else 0

    @property
    def num_dense_layers(self) -> int:
        return self.num_layers - self.num_moe_layers

    @property
    def activation_dtype(self) -> str:
        return normalize_dtype(self.torch_dtype)

    @property
    def primary_architecture(self) -> str:
        return self.architectures[0] if self.architectures else self.model_type


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    """VLM configs nest the language model under ``text_config``; merge it over the top level."""
    nested = config.get("text_config")
    if isinstance(nested, dict) and (
        "hidden_size" not in config or "num_hidden_layers" not in config
    ):
        merged = {**config, **nested}
        merged["architectures"] = config.get("architectures") or nested.get("architectures")
        return merged
    return config


def _parse_attention(config: dict[str, Any], model_type: str, num_layers: int) -> AttentionArch:
    hidden = int(config["hidden_size"])
    num_heads = int(config["num_attention_heads"])
    rope_scaling = config.get("rope_scaling") or {}
    rope_type = (
        (rope_scaling.get("type") or rope_scaling.get("rope_type")) if rope_scaling else None
    )
    if config.get("kv_lora_rank"):
        return AttentionArch(
            kind="MLA",
            num_heads=num_heads,
            num_kv_heads=num_heads,
            head_dim=int(config["qk_nope_head_dim"]) + int(config["qk_rope_head_dim"]),
            v_head_dim=int(config["v_head_dim"]),
            q_lora_rank=config.get("q_lora_rank"),
            kv_lora_rank=int(config["kv_lora_rank"]),
            qk_nope_head_dim=int(config["qk_nope_head_dim"]),
            qk_rope_head_dim=int(config["qk_rope_head_dim"]),
            sparse_indexer="index_topk" in config,
            index_n_heads=config.get("index_n_heads"),
            index_head_dim=config.get("index_head_dim"),
            rope_scaling_type=rope_type,
        )
    num_kv_heads = int(_first(config, "num_key_value_heads", "num_kv_heads", default=num_heads))
    head_dim = int(_first(config, "head_dim", default=hidden // num_heads))
    if num_kv_heads == num_heads:
        kind = "MHA"
    elif num_kv_heads == 1:
        kind = "MQA"
    else:
        kind = "GQA"

    sliding_window = config.get("sliding_window")
    num_sliding_layers = 0
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        num_sliding_layers = sum(1 for item in layer_types if "sliding" in str(item))
    elif config.get("use_sliding_window") is False:
        sliding_window = None
    elif sliding_window and config.get("sliding_window_pattern"):
        pattern = int(config["sliding_window_pattern"])
        num_sliding_layers = num_layers - num_layers // pattern if pattern > 0 else 0
    elif sliding_window:
        num_sliding_layers = num_layers
    if not num_sliding_layers:
        sliding_window = None

    has_qk_norm = model_type in QK_NORM_MODEL_TYPES or bool(
        _first(config, "use_qk_norm", "qk_layernorm", "qk_norm", default=False)
    )
    has_qkv_bias = bool(config.get("attention_bias")) or model_type in QKV_BIAS_MODEL_TYPES
    return AttentionArch(
        kind=kind,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        v_head_dim=head_dim,
        has_qk_norm=has_qk_norm,
        has_qkv_bias=has_qkv_bias,
        sliding_window=sliding_window,
        num_sliding_layers=num_sliding_layers,
        rope_scaling_type=rope_type,
    )


def _moe_layer_indices(config: dict[str, Any], num_layers: int) -> tuple[int, ...]:
    if "first_k_dense_replace" in config or "moe_layer_freq" in config:
        first_dense = int(config.get("first_k_dense_replace") or 0)
        frequency = int(config.get("moe_layer_freq") or 1)
        return tuple(i for i in range(num_layers) if i >= first_dense and i % frequency == 0)
    if "decoder_sparse_step" in config or "mlp_only_layers" in config:
        step = int(config.get("decoder_sparse_step") or 1)
        mlp_only = set(config.get("mlp_only_layers") or [])
        return tuple(i for i in range(num_layers) if i not in mlp_only and (i + 1) % step == 0)
    if "interleave_moe_layer_step" in config:
        step = int(config.get("interleave_moe_layer_step") or 1)
        return tuple(i for i in range(num_layers) if (i + 1) % step == 0)
    return tuple(range(num_layers))


def _parse_moe(config: dict[str, Any], model_type: str, num_layers: int) -> MoEArch | None:
    num_experts = _first(
        config, "n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts"
    )
    if not num_experts or int(num_experts) <= 1:
        return None
    top_k = int(
        _first(config, "num_experts_per_tok", "moe_topk", "experts_per_token", "top_k", default=1)
    )
    expert_intermediate = int(_first(config, "moe_intermediate_size", "intermediate_size"))
    num_shared = int(_first(config, "n_shared_experts", "num_shared_experts", default=0))
    shared_intermediate = num_shared * expert_intermediate
    if not shared_intermediate and config.get("shared_expert_intermediate_size"):
        num_shared = 1
        shared_intermediate = int(config["shared_expert_intermediate_size"])
    if model_type.startswith("llama4") and not shared_intermediate:
        num_shared = 1
        shared_intermediate = expert_intermediate
    scoring = str(
        config.get("scoring_func") or ("sigmoid" if model_type.startswith("llama4") else "softmax")
    )
    groups = config.get("n_group")
    routing = f"{scoring}_grouped_topk" if groups and int(groups) > 1 else f"{scoring}_topk"
    return MoEArch(
        num_experts=int(num_experts),
        top_k=top_k,
        expert_intermediate_size=expert_intermediate,
        num_shared_experts=num_shared,
        shared_intermediate_size=shared_intermediate,
        moe_layer_indices=_moe_layer_indices(config, num_layers),
        routing=routing,
        num_expert_groups=int(groups) if groups else None,
        topk_groups=config.get("topk_group"),
        has_router_bias=config.get("topk_method") == "noaux_tc",
    )


def _parse_quant(config: dict[str, Any], warnings: list[str]) -> QuantArch | None:
    quant = config.get("quantization_config") or config.get("compression_config")
    if not isinstance(quant, dict):
        return None
    method = str(quant.get("quant_method") or quant.get("quant_algo") or "unknown").lower()
    if method == "fp8":
        block = quant.get("weight_block_size")
        dynamic = quant.get("activation_scheme", "dynamic") == "dynamic"
        return QuantArch(
            method="fp8",
            weight_dtype="fp8",
            activation_dtype="fp8",
            block_shape=tuple(block) if block else None,
            group_size=None,
            applies_to="all_linear",
            raw={**quant, "dynamic_activation": dynamic},
        )
    if method in {"awq", "gptq", "awq_marlin", "gptq_marlin", "auto-round"}:
        bits = int(quant.get("bits") or quant.get("w_bit") or 4)
        return QuantArch(
            method=method,
            weight_dtype="int4" if bits == 4 else "int8",
            activation_dtype=None,
            block_shape=None,
            group_size=quant.get("group_size") or quant.get("q_group_size"),
            applies_to="all_linear",
            raw=quant,
        )
    if method == "mxfp4":
        return QuantArch(
            method="mxfp4",
            weight_dtype="fp4",
            activation_dtype=None,
            block_shape=None,
            group_size=32,
            applies_to="experts_only",
            raw=quant,
        )
    if method in {"modelopt", "modelopt_fp8", "modelopt_fp4"}:
        algo = str(
            quant.get("quant_algo") or quant.get("quantization", {}).get("quant_algo", "")
        ).upper()
        dtype = "fp4" if "FP4" in algo else "fp8"
        return QuantArch(
            method=method,
            weight_dtype=dtype,
            activation_dtype=dtype,
            block_shape=None,
            group_size=16 if dtype == "fp4" else None,
            applies_to="all_linear",
            raw=quant,
        )
    if method == "compressed-tensors":
        groups = quant.get("config_groups") or {}
        first_group = next(iter(groups.values()), {}) if isinstance(groups, dict) else {}
        weights = first_group.get("weights") or {}
        activations = first_group.get("input_activations") or {}
        weight_bits = int(weights.get("num_bits") or 8)
        weight_type = str(weights.get("type") or "int")
        if weight_type == "float":
            weight_dtype = "fp8" if weight_bits == 8 else "fp4"
        else:
            weight_dtype = "int8" if weight_bits == 8 else "int4"
        activation_dtype = None
        if activations:
            activation_bits = int(activations.get("num_bits") or 8)
            activation_type = str(activations.get("type") or "int")
            if activation_type == "float":
                activation_dtype = "fp8" if activation_bits == 8 else "fp4"
            else:
                activation_dtype = "int8"
        return QuantArch(
            method=method,
            weight_dtype=weight_dtype,
            activation_dtype=activation_dtype,
            block_shape=None,
            group_size=weights.get("group_size"),
            applies_to="all_linear",
            raw=quant,
        )
    warnings.append(f"未识别的量化方法 {method!r}，按 BF16 权重估算")
    return None


def parse_hf_config(
    config: dict[str, Any], name: str = "model", hf_repo: str | None = None
) -> ModelArch:
    warnings: list[str] = []
    config = _text_config(config)
    model_type = str(config.get("model_type") or "unknown")
    missing = [
        key
        for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size")
        if config.get(key) is None
    ]
    if missing:
        raise ValueError(f"config.json is missing required keys {missing}")
    num_layers = int(config["num_hidden_layers"])
    attention = _parse_attention(config, model_type, num_layers)
    moe = _parse_moe(config, model_type, num_layers)
    quant = _parse_quant(config, warnings)
    if moe is not None and "moe_intermediate_size" in config:
        dense_intermediate = int(config.get("intermediate_size") or 0)
    else:
        dense_intermediate = int(
            _first(config, "intermediate_size_mlp", "intermediate_size", "ffn_dim", default=0)
        )
    if not dense_intermediate and (moe is None or len(moe.moe_layer_indices) < num_layers):
        dense_intermediate = 4 * int(config["hidden_size"])
        warnings.append("未找到 intermediate_size，按 4×hidden_size 估算稠密 MLP")
    torch_dtype = str(_first(config, "torch_dtype", "dtype", default="bfloat16"))
    try:
        normalize_dtype(torch_dtype)
    except ValueError:
        warnings.append(f"未知 torch_dtype {torch_dtype!r}，按 bfloat16 处理")
        torch_dtype = "bfloat16"
    if (
        model_type not in QK_NORM_MODEL_TYPES | QKV_BIAS_MODEL_TYPES | NON_GATED_MLP_MODEL_TYPES
        and not (
            model_type.startswith(
                ("llama", "mistral", "mixtral", "deepseek", "qwen", "glm", "kimi", "gpt_oss")
            )
        )
    ):
        warnings.append(f"model_type={model_type} 未内置结构规则，按 gated-MLP + RMSNorm 估算")
    return ModelArch(
        name=name,
        architectures=tuple(config.get("architectures") or ()),
        model_type=model_type,
        hidden_size=int(config["hidden_size"]),
        num_layers=num_layers,
        vocab_size=int(config["vocab_size"]),
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        dense_intermediate_size=dense_intermediate,
        gated_mlp=model_type not in NON_GATED_MLP_MODEL_TYPES,
        hidden_act=str(_first(config, "hidden_act", "activation_function", default="silu")),
        norm_type="layernorm" if model_type in LAYERNORM_MODEL_TYPES else "rmsnorm",
        attention=attention,
        moe=moe,
        quant=quant,
        torch_dtype=torch_dtype,
        max_position_embeddings=config.get("max_position_embeddings"),
        num_mtp_layers=int(config.get("num_nextn_predict_layers") or 0),
        hf_repo=hf_repo,
        warnings=tuple(warnings),
        raw=config,
    )


def _bundled_index() -> dict[str, dict[str, str]]:
    index_path = resources.files("sgl_agent") / "data" / "models" / "index.yaml"
    return yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}


def list_bundled_models() -> dict[str, str]:
    return {alias: entry["hf_repo"] for alias, entry in _bundled_index().items()}


def load_model(name_or_path: str) -> ModelArch:
    """Load from a ``config.json`` path, a model directory, or a bundled alias such as ``deepseek-v3``."""
    candidate = Path(name_or_path)
    if candidate.is_dir():
        candidate = candidate / "config.json"
    if candidate.is_file():
        config = json.loads(candidate.read_text(encoding="utf-8"))
        name = candidate.parent.name if candidate.name == "config.json" else candidate.stem
        return parse_hf_config(config, name=name, hf_repo=config.get("_name_or_path"))
    index = _bundled_index()
    entry = index.get(name_or_path.lower())
    if entry is None:
        raise FileNotFoundError(
            f"model {name_or_path!r} not found: pass a config.json/model directory or one of {sorted(index)}"
        )
    path = resources.files("sgl_agent") / "data" / "models" / entry["file"]
    config = json.loads(path.read_text(encoding="utf-8"))
    return parse_hf_config(config, name=name_or_path.lower(), hf_repo=entry.get("hf_repo"))
