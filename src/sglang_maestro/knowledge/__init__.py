"""Access to the bundled SGLang capability snapshot (flags, backends, architectures, tuned configs).

Set ``SGLANG_MAESTRO_KNOWLEDGE=/path/to/snapshot.json`` to use a snapshot generated from the SGLang
version you actually deploy (``sglang-maestro sync-knowledge --sglang-src <checkout>``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

BUNDLED_SNAPSHOT = Path(__file__).with_name("sglang_capabilities.json")


@dataclass(frozen=True)
class SGLangKnowledge:
    raw: dict[str, Any]
    source_path: str
    _alias_to_flag: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source_path: str = "<memory>") -> SGLangKnowledge:
        alias_to_flag: dict[str, str] = {}
        for flag, metadata in raw.get("server_flags", {}).items():
            alias_to_flag[flag] = flag
            for alias in metadata.get("aliases") or []:
                alias_to_flag.setdefault(alias, flag)
        return cls(raw=raw, source_path=source_path, _alias_to_flag=alias_to_flag)

    @property
    def commit(self) -> str | None:
        return self.raw.get("sglang_commit")

    @property
    def commit_date(self) -> str | None:
        return self.raw.get("sglang_commit_date")

    def canonical_flag(self, flag: str) -> str | None:
        return self._alias_to_flag.get(flag)

    def flag(self, flag: str) -> dict[str, Any] | None:
        canonical = self.canonical_flag(flag)
        return self.raw["server_flags"].get(canonical) if canonical else None

    def has_flag(self, flag: str) -> bool:
        return self.canonical_flag(flag) is not None

    def is_deprecated(self, flag: str) -> bool:
        metadata = self.flag(flag)
        return bool(metadata and metadata.get("deprecated"))

    def choices(self, flag: str) -> list[Any] | None:
        metadata = self.flag(flag)
        return metadata.get("choices") if metadata else None

    def choice_constant(self, name: str) -> list[Any]:
        return list(self.raw.get("choices", {}).get(name, []))

    def is_architecture_supported(self, architecture: str) -> bool:
        return architecture in self.raw.get("model_architectures", {})

    def model_file_for(self, architecture: str) -> str | None:
        return self.raw.get("model_architectures", {}).get(architecture)

    @property
    def layernorm_sp_architectures(self) -> list[str]:
        return list(self.raw.get("layernorm_sp_supported_architectures", []))

    @property
    def allreduce_fusion_auto_architectures(self) -> list[str]:
        """Architectures for which SGLang auto-enables FlashInfer allreduce+RMSNorm fusion."""
        return list(self.raw.get("allreduce_fusion_auto_architectures", []))

    @property
    def fused_qk_norm_rope_architectures(self) -> list[str]:
        """Architectures whose model file honours ``--enable-fused-qk-norm-rope``."""
        return list(self.raw.get("fused_qk_norm_rope_architectures", []))

    @property
    def fused_rope_kv_store_architectures(self) -> list[str]:
        """Architectures that write K/V inside the RoPE kernel (``fused_set_kv_buffer_arg``)."""
        return list(self.raw.get("fused_rope_kv_store_architectures", []))

    @property
    def tbo_architectures(self) -> list[str]:
        """Architectures built from a decoder layer that two-batch overlap has a strategy for."""
        return list(self.raw.get("tbo_architectures", []))

    @property
    def dual_stream_architectures(self) -> list[str]:
        """Architectures whose MoE block has a ``forward_normal_dual_stream`` path."""
        return list(self.raw.get("dual_stream_architectures", []))

    @property
    def layer_boundary_architectures(self) -> list[str]:
        """Architectures whose residual add + norm go through ``layers/layer_boundary``.

        That path hosts the allreduce + residual + RMSNorm fusion, so the FlashInfer fusion
        flag only changes kernels for these models.
        """
        return list(self.raw.get("layer_boundary_architectures", []))

    @property
    def dwdp_architectures(self) -> list[str]:
        """Architectures with a ``forward_dwdp`` path (``--dwdp-size``)."""
        return list(self.raw.get("dwdp_architectures", []))

    @property
    def fused_op_dispatch(self) -> dict[str, Any]:
        """``backend_methods``, ``platform_methods`` and ``default_priority`` of ``BaseFusedOp``."""
        return dict(self.raw.get("fused_op_dispatch", {}))

    @property
    def fused_op_classes(self) -> dict[str, dict[str, Any]]:
        return self.raw.get("fused_op_classes", {})

    def fused_op_class(self, qualified_name: str) -> dict[str, Any] | None:
        return self.fused_op_classes.get(qualified_name)

    def find_fused_op_classes(self, class_name: str) -> list[str]:
        """Qualified names of fused-op classes whose ``__name__`` is ``class_name``."""
        return [
            qualified
            for qualified in self.fused_op_classes
            if qualified.rsplit(".", 1)[-1] == class_name
        ]

    @property
    def platform_hooks(self) -> dict[str, dict[str, Any]]:
        """``DeviceMixin``/``SRTPlatform`` methods with their default behaviour and call sites."""
        return self.raw.get("platform_interface", {}).get("hooks", {})

    def platform_hook(self, name: str) -> dict[str, Any] | None:
        return self.platform_hooks.get(name)

    @property
    def platform_capabilities(self) -> dict[str, dict[str, Any]]:
        """``PlatformCapabilities`` fields with their defaults and the code that reads them."""
        return self.raw.get("platform_interface", {}).get("capabilities", {})

    @property
    def platform_attributes(self) -> dict[str, dict[str, Any]]:
        return self.raw.get("platform_interface", {}).get("attributes", {})

    @property
    def platform_compile_sites(self) -> list[dict[str, Any]]:
        """Functions decorated with ``torch.compile(backend=get_compiler_backend())`` at import time."""
        return list(self.raw.get("platform_compile_sites", []))

    @property
    def oot_guards(self) -> list[dict[str, Any]]:
        """Calls that raise on out-of-tree platforms: missing hook overrides and unported subsystems."""
        return list(self.raw.get("oot_guards", []))

    def validate_flags(self, flag_values: dict[str, Any]) -> list[str]:
        """Return human-readable problems for flags that do not exist, are deprecated or use bad choices."""
        problems: list[str] = []
        for flag, value in flag_values.items():
            metadata = self.flag(flag)
            if metadata is None:
                problems.append(f"{flag} 不在 SGLang@{(self.commit or '?')[:8]} 的参数列表中")
                continue
            if metadata.get("deprecated"):
                replacement = metadata.get("replacement")
                problems.append(
                    f"{flag} 已弃用" + (f"，请改用 {replacement}" if replacement else "")
                )
            choices = metadata.get("choices")
            if choices and value is not None and value is not True:
                values = value if isinstance(value, (list, tuple)) else [value]
                for single_value in values:
                    if single_value not in choices and str(single_value) not in map(str, choices):
                        problems.append(f"{flag}={single_value} 不是合法取值 {choices}")
        return problems

    def moe_tuned_configs(
        self,
        *,
        device: str,
        experts: int,
        intermediate: int,
        dtype: str | None = None,
        block_shape: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        device_key = device.replace(" ", "_")
        return [
            entry
            for entry in self.raw.get("moe_triton_tuned_configs", [])
            if entry["device"] == device_key
            and entry["E"] == experts
            and entry["N"] == intermediate
            and entry.get("dtype") == dtype
            and (entry.get("block_shape") or None) == (block_shape or None)
        ]


@lru_cache(maxsize=4)
def _load(path: str) -> SGLangKnowledge:
    with open(path, encoding="utf-8") as handle:
        return SGLangKnowledge.from_dict(json.load(handle), source_path=path)


def load_knowledge(path: str | os.PathLike[str] | None = None) -> SGLangKnowledge:
    chosen = path or os.environ.get("SGLANG_MAESTRO_KNOWLEDGE") or BUNDLED_SNAPSHOT
    return _load(str(Path(chosen).resolve()))
