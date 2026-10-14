"""Accelerator description: datasheet numbers, software stack capabilities and model assumptions."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

DTYPE_ALIASES: dict[str, str] = {
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "fp16": "fp16",
    "float32": "fp32",
    "fp32": "fp32",
    "tf32": "tf32",
    "fp8": "fp8",
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8",
    "float8_e4m3fn": "fp8",
    "e4m3": "fp8",
    "int8": "int8",
    "w8a8_int8": "int8",
    "fp4": "fp4",
    "nvfp4": "fp4",
    "mxfp4": "fp4",
    "int4": "int4",
}

BYTES_PER_ELEMENT: dict[str, float] = {
    "fp32": 4.0,
    "tf32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp4": 0.5,
    "int4": 0.5,
}


def normalize_dtype(name: str) -> str:
    key = name.strip().lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(f"unknown dtype {name!r}; known: {sorted(set(DTYPE_ALIASES))}")
    return DTYPE_ALIASES[key]


def bytes_per_element(dtype: str) -> float:
    return BYTES_PER_ELEMENT[normalize_dtype(dtype)]


@dataclass(frozen=True)
class SoftwareStack:
    """What the SGLang stack can run on this device (drives adaptation gap analysis)."""

    platform: str = "cuda"
    torch_device_name: str = ""
    compute_capability: str | None = None
    gemm_dtypes: tuple[str, ...] = ("bf16", "fp16")
    kv_cache_dtypes: tuple[str, ...] = ("bf16",)
    attention_backends: tuple[str, ...] = ()
    default_attention_backend: str | None = None
    moe_runner_backends: tuple[str, ...] = ()
    moe_a2a_backends: tuple[str, ...] = ("none",)
    fused_ops: tuple[str, ...] = ()
    collective_library: str = "nccl"
    supports_triton: bool = False
    graph_capture: bool = False
    piecewise_graph: bool = False
    hicache_device_kernels: bool = False
    custom_allreduce: bool = False
    allreduce_fusion: bool = False
    torch_compile: bool = False
    multi_stream: bool = True
    l2_prefetch: bool = False
    rdma: bool = False
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class EfficiencyAssumptions:
    """Achievable fraction of peak; defaults are assumptions to be replaced by calibration."""

    gemm_compute: float = 0.70
    attention_compute: float = 0.55
    memory_bandwidth: float = 0.80
    intra_node_comm: float = 0.70
    inter_node_comm: float = 0.70
    host_link: float = 0.75


@dataclass(frozen=True)
class LatencyAssumptions:
    """Fixed costs in microseconds; defaults are assumptions to be replaced by calibration."""

    kernel_launch_us: float = 4.0
    graph_kernel_gap_us: float = 1.0
    collective_base_us: float = 10.0
    inter_node_collective_base_us: float = 25.0
    step_overhead_us: float = 400.0


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    vendor: str
    architecture: str
    memory_capacity_gb: float
    memory_bandwidth_gbps: float
    l2_cache_mb: float
    compute_tflops: dict[str, float]
    intra_node_bandwidth_gbps: float
    inter_node_bandwidth_gbps: float
    host_link_bandwidth_gbps: float
    devices_per_node: int = 8
    intra_node_topology: str = "switch"
    marketed_memory_gb: float | None = None
    num_compute_units: int | None = None
    power_watts: float | None = None
    aliases: tuple[str, ...] = ()
    software: SoftwareStack = field(default_factory=SoftwareStack)
    efficiency: EfficiencyAssumptions = field(default_factory=EfficiencyAssumptions)
    latency: LatencyAssumptions = field(default_factory=LatencyAssumptions)
    sources: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def memory_capacity_bytes(self) -> float:
        return self.memory_capacity_gb * 1e9

    @property
    def memory_bandwidth_bytes_per_s(self) -> float:
        return self.memory_bandwidth_gbps * 1e9

    @property
    def l2_cache_bytes(self) -> float:
        return self.l2_cache_mb * 1e6

    def supports_compute_dtype(self, dtype: str) -> bool:
        return self.compute_tflops.get(normalize_dtype(dtype), 0.0) > 0.0

    def peak_flops(self, dtype: str) -> float:
        """Dense peak in FLOP/s; raises if the device has no native unit for ``dtype``."""
        normalized = normalize_dtype(dtype)
        tflops = self.compute_tflops.get(normalized, 0.0)
        if tflops <= 0.0:
            raise ValueError(f"{self.name} has no native {normalized} compute")
        return tflops * 1e12

    def ridge_point(self, dtype: str) -> float:
        """Arithmetic intensity (FLOPs/byte) where compute and HBM time are equal."""
        return self.peak_flops(dtype) / self.memory_bandwidth_bytes_per_s

    def compute_dtype_for(self, weight_dtype: str, activation_dtype: str) -> str:
        """Datatype the tensor cores run at for a weight/activation pair.

        W8A8 runs at FP8/INT8 rate only when both operands are 8-bit and the unit exists;
        weight-only quantization (e.g. W4A16, W8A16) dequantizes and runs at the activation rate.
        """
        weight = normalize_dtype(weight_dtype)
        activation = normalize_dtype(activation_dtype)
        if weight == activation and self.supports_compute_dtype(weight):
            return weight
        if self.supports_compute_dtype(activation):
            return activation
        return "bf16"


def _tuple_fields(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    known = {item.name: item for item in fields(cls)}
    converted: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in known:
            raise ValueError(f"unknown field {key!r} for {cls.__name__}")
        converted[key] = tuple(value) if isinstance(value, list) else value
    return converted


def hardware_from_dict(payload: dict[str, Any]) -> HardwareSpec:
    memory = payload.get("memory", {})
    interconnect = payload.get("interconnect", {})
    intra = interconnect.get("intra_node", {})
    inter = interconnect.get("inter_node", {})
    host = interconnect.get("host_link", {})
    compute = {normalize_dtype(k): float(v) for k, v in payload.get("compute_tflops", {}).items()}
    if not compute.get("bf16") and not compute.get("fp16"):
        raise ValueError("compute_tflops must provide at least bf16 or fp16")
    if memory.get("capacity_mib"):
        capacity_gb = float(memory["capacity_mib"]) * 1024 * 1024 / 1e9
    else:
        capacity_gb = float(memory.get("capacity_gb") or 0.0)
    required = {
        "memory.capacity_mib|capacity_gb": capacity_gb,
        "memory.bandwidth_gbps": memory.get("bandwidth_gbps"),
        "interconnect.intra_node.bandwidth_gbps": intra.get("bandwidth_gbps"),
    }
    missing = [key for key, value in required.items() if value in (None, 0)]
    if missing:
        raise ValueError(f"hardware spec is missing {missing}")
    return HardwareSpec(
        name=payload["name"],
        vendor=payload.get("vendor", "unknown"),
        architecture=payload.get("architecture", "unknown"),
        memory_capacity_gb=capacity_gb,
        marketed_memory_gb=memory.get("marketed_capacity_gb"),
        memory_bandwidth_gbps=float(memory["bandwidth_gbps"]),
        l2_cache_mb=float(memory.get("l2_cache_mb", 0.0)),
        compute_tflops=compute,
        intra_node_bandwidth_gbps=float(intra["bandwidth_gbps"]),
        inter_node_bandwidth_gbps=float(inter.get("bandwidth_gbps", 0.0)),
        host_link_bandwidth_gbps=float(host.get("bandwidth_gbps", 0.0)),
        devices_per_node=int(intra.get("devices_per_node", 8)),
        intra_node_topology=str(intra.get("topology", "switch")),
        num_compute_units=payload.get("num_compute_units"),
        power_watts=payload.get("power_watts"),
        aliases=tuple(payload.get("aliases", [])),
        software=SoftwareStack(**_tuple_fields(SoftwareStack, payload.get("software", {}))),
        efficiency=EfficiencyAssumptions(
            **_tuple_fields(EfficiencyAssumptions, payload.get("efficiency", {}))
        ),
        latency=LatencyAssumptions(**_tuple_fields(LatencyAssumptions, payload.get("latency", {}))),
        sources=tuple(payload.get("sources", [])),
        notes=tuple(payload.get("notes", [])),
    )


def _preset_files() -> list[Path]:
    directory = resources.files("sgl_agent") / "data" / "hardware"
    return sorted(Path(str(item)) for item in directory.iterdir() if item.name.endswith(".yaml"))


def list_presets() -> dict[str, list[str]]:
    presets: dict[str, list[str]] = {}
    for path in _preset_files():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        presets[path.stem] = [payload["name"], *payload.get("aliases", [])]
    return presets


def load_hardware(name_or_path: str) -> HardwareSpec:
    """Load a preset by file stem/alias (``h20``) or a YAML file path."""
    candidate = Path(name_or_path)
    if candidate.suffix in {".yaml", ".yml"} and candidate.exists():
        return hardware_from_dict(yaml.safe_load(candidate.read_text(encoding="utf-8")))
    wanted = name_or_path.strip().lower()
    for path in _preset_files():
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        names = {
            path.stem.lower(),
            payload["name"].lower(),
            *map(str.lower, payload.get("aliases", [])),
        }
        if wanted in names:
            return hardware_from_dict(payload)
    raise FileNotFoundError(
        f"hardware {name_or_path!r} not found; presets: {sorted(list_presets())} or pass a YAML path"
    )
