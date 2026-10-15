"""The RoPE class SGLang's ``get_rope`` factory builds for a model config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_maestro.knowledge.facts import DEEPSEEK_V2_MODEL_ARCHITECTURES
from sglang_maestro.model.config import ModelArch


@dataclass(frozen=True)
class RopeChoice:
    class_name: str | None
    """None when SGLang raises ``Unknown RoPE scaling type``."""
    scaling_type: str
    rewritten: bool = False
    """DeepSeek MLA attention rewrote ``rope_type`` to ``deepseek_yarn`` before the factory."""


def _scaling_dict(raw: dict[str, Any]) -> dict[str, Any] | None:
    rope_scaling = raw.get("rope_scaling")
    if isinstance(rope_scaling, dict) and rope_scaling:
        return dict(rope_scaling)
    return None


def rope_class_for(arch: ModelArch) -> RopeChoice:
    """Mirror of ``layers/rotary_embedding/factory.py::get_rope``.

    The key ``rope_type`` wins over ``type`` by presence, even when its value is empty.
    """
    raw = arch.raw
    if raw.get("dual_chunk_attention_config"):
        return RopeChoice("DualChunkRotaryEmbedding", "dual_chunk")
    rope_scaling = _scaling_dict(raw)
    rewritten = False
    if (
        rope_scaling is not None
        and arch.attention.is_mla
        and arch.primary_architecture in DEEPSEEK_V2_MODEL_ARCHITECTURES
    ):
        rope_scaling["rope_type"] = "deepseek_yarn"
        rewritten = True
    if rope_scaling is None:
        return RopeChoice("RotaryEmbedding", "none")
    if "rope_type" in rope_scaling:
        scaling_type = str(rope_scaling["rope_type"])
    elif "type" in rope_scaling:
        scaling_type = str(rope_scaling["type"])
    else:
        return RopeChoice(None, "unknown", rewritten)
    mrope = "mrope_section" in rope_scaling
    video = bool(rope_scaling.get("video_rope", False))
    if scaling_type == "llama3":
        name = "Llama3RotaryEmbedding"
    elif scaling_type == "default":
        if mrope and video:
            name = "BailingMRotaryEmbedding"
        elif mrope:
            name = "MRotaryEmbedding"
        elif rope_scaling.get("use_fope", False):
            name = "FourierRotaryEmbedding"
        else:
            name = "RotaryEmbedding"
    elif scaling_type == "linear":
        name = "LinearScalingRotaryEmbedding"
    elif scaling_type == "dynamic":
        name = (
            "DynamicNTKAlphaRotaryEmbedding"
            if "alpha" in rope_scaling
            else "DynamicNTKScalingRotaryEmbedding"
        )
    elif scaling_type == "yarn":
        if mrope and video:
            name = "BailingMRotaryEmbedding"
        elif mrope:
            name = "YaRNScalingMRotaryEmbedding"
        else:
            name = "YaRNScalingRotaryEmbedding"
    elif scaling_type == "deepseek_yarn":
        name = "DeepseekScalingRotaryEmbedding"
    elif scaling_type == "longrope":
        name = "Phi3LongRoPEScaledRotaryEmbedding"
    elif scaling_type == "proportional":
        name = "Gemma4RotaryEmbedding"
    else:
        name = None
    return RopeChoice(name, scaling_type, rewritten)
