"""Which Triton fused-MoE tuned config SGLang looks up for a model/layout, and how to tune it.

Mirrors ``layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py::get_config_file_name``:
``E`` and ``N`` come from the local ``w2`` shape ``(E, hidden, N)``, i.e. local experts (including
shared experts fused into the routed kernel) and the expert intermediate size per MoE-TP rank.
Missing files make SGLang log "Using default MoE kernel config. Performance might be sub-optimal!".
"""

from __future__ import annotations

from dataclasses import dataclass

from sglang_maestro.hardware.spec import HardwareSpec
from sglang_maestro.model.config import ModelArch
from sglang_maestro.model.parallel import ParallelConfig

TUNING_SCRIPT = "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py"
TUNING_DTYPES = ("fp8_w8a8", "int8_w8a16", "int8_w8a8", "int4_w4a16")


@dataclass(frozen=True)
class MoEConfigKey:
    experts: int
    intermediate: int
    dtype: str | None
    block_shape: tuple[int, int] | None
    device: str
    fused_shared_experts: int = 0
    shared_fusion_known: bool = False
    """False when SGLang's fusion decision for this architecture is not modelled."""

    def file_name(self) -> str:
        dtype = f",dtype={self.dtype}" if self.dtype else ""
        block = (
            f",block_shape=[{self.block_shape[0]}, {self.block_shape[1]}]"
            if self.block_shape and all(self.block_shape)
            else ""
        )
        device = self.device.replace(" ", "_")
        return f"E={self.experts},N={self.intermediate},device_name={device}{dtype}{block}.json"


def triton_dtype_selector(arch: ModelArch, weight_dtype: str | None = None) -> str | None:
    """``get_config_dtype_str``: the ``dtype=`` part of the file name (None for BF16/FP16)."""
    if weight_dtype:
        return "fp8_w8a8" if weight_dtype.lower().startswith("fp8") else None
    quant = arch.quant
    if quant is None:
        return None
    if quant.weight_dtype == "fp8" and quant.activation_dtype == "fp8":
        return "fp8_w8a8"
    if quant.weight_dtype == "int8":
        return "int8_w8a8" if quant.activation_dtype == "int8" else "int8_w8a16"
    if quant.weight_dtype == "int4" and quant.activation_dtype is None:
        return "int4_w4a16"
    return None


def deepseek_shared_experts_fused(
    arch: ModelArch,
    parallel: ParallelConfig,
    hardware: HardwareSpec,
    *,
    two_batch_overlap: bool = False,
) -> bool | None:
    """SGLang's default shared-expert fusion decision for ``DeepseekV3ForCausalLM``.

    Follows ``DeepseekV2ForCausalLM.shared_experts_fusion_disable_reason`` without
    ``--enforce-shared-experts-fusion``; returns None for architectures it does not model.
    """
    if arch.primary_architecture != "DeepseekV3ForCausalLM" or arch.moe is None:
        return None
    moe = arch.moe
    quant = arch.quant
    if moe.num_experts not in (256, 384) or moe.num_shared_experts != 1:
        return False
    if moe.num_experts == 384 and (quant is None or quant.method != "quark"):
        return False
    if two_batch_overlap or parallel.moe_a2a_backend != "none":
        return False
    software = hardware.software
    if software.platform == "cuda":
        try:
            capability = tuple(int(part) for part in str(software.compute_capability).split("."))
        except ValueError:
            return None
        if capability < (8, 0) or parallel.ep > 1:
            return False
    elif software.platform != "rocm":
        return None
    return not (quant is not None and quant.weight_dtype == "int4")


def triton_moe_config_key(
    arch: ModelArch,
    parallel: ParallelConfig,
    hardware: HardwareSpec,
    *,
    weight_dtype: str | None = None,
    fused_shared_experts: int | None = None,
) -> MoEConfigKey | None:
    if arch.moe is None:
        return None
    known = fused_shared_experts is not None
    if fused_shared_experts is None:
        fused = deepseek_shared_experts_fused(arch, parallel, hardware)
        known = fused is not None
        fused_shared_experts = arch.moe.num_shared_experts if fused else 0
    block = None
    if arch.quant is not None and arch.quant.block_shape and not weight_dtype:
        block = (int(arch.quant.block_shape[0]), int(arch.quant.block_shape[1]))
    return MoEConfigKey(
        experts=arch.moe.num_experts // max(parallel.ep, 1) + fused_shared_experts,
        intermediate=arch.moe.expert_intermediate_size // max(parallel.moe_tp, 1),
        dtype=triton_dtype_selector(arch, weight_dtype),
        block_shape=block,
        device=hardware.software.torch_device_name or hardware.name,
        fused_shared_experts=fused_shared_experts,
        shared_fusion_known=known,
    )


def tuning_command(
    arch: ModelArch, parallel: ParallelConfig, key: MoEConfigKey, model_path: str | None = None
) -> str:
    """Command that writes the missing config (run on the target GPU, then copy the JSON)."""
    model = model_path or arch.hf_repo or arch.name
    parts = [
        f"python {TUNING_SCRIPT}",
        f"--model {model}",
        f"--tp-size {parallel.tp}",
    ]
    if parallel.ep > 1:
        parts.append(f"--ep-size {parallel.ep}")
    if key.dtype in TUNING_DTYPES:
        parts.append(f"--dtype {key.dtype}")
    if (
        key.shared_fusion_known
        and key.fused_shared_experts == 0
        and arch.moe is not None
        and arch.moe.num_shared_experts
    ):
        parts.append("--disable-shared-experts-fusion")
    parts.append("--tune")
    return " ".join(parts)
