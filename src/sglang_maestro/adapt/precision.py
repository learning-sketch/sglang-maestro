"""Precision routes for running a checkpoint on a device, and the numerics a port must preserve.

A route is "which bytes sit in HBM and which dtype the GEMMs run at": keep the checkpoint,
dequantize to BF16, requantize to W8A8 INT8, keep FP8 weights but run W8A16, or W4A16. Each
route is sized (smallest GPU count that leaves a usable KV pool), timed with the step cost
model and checked against the kernels the device declares.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from sglang_maestro.adapt.platform import PlatformProfile
from sglang_maestro.hardware.spec import HardwareSpec, normalize_dtype
from sglang_maestro.knowledge.facts import DEEPSEEK_V2_MODEL_ARCHITECTURES, source
from sglang_maestro.model.config import ModelArch
from sglang_maestro.model.ops import BatchShape, ExecutionConfig, StepCostModel
from sglang_maestro.model.parallel import MemoryKnobs, ParallelConfig, plan_memory
from sglang_maestro.model.params import weight_precision

DEFAULT_MIN_KV_TOKENS = 32768
"""A route is servable on N GPUs when the KV pool holds at least this many tokens."""
MAX_NODES = 4
_GPU_COUNTS = (1, 2, 4, 8, 16, 32, 64)
CHUNKED_PREFILL_LADDER = (None, 4096, 2048)
"""SGLang reserves activation memory in proportion to the chunk; smaller chunks free KV pool."""


@dataclass(frozen=True)
class RouteSizing:
    parallel: ParallelConfig
    weights_gb_per_gpu: float
    kv_tokens: int
    decode_step_ms: float | None
    prefill_step_ms: float | None
    chunked_prefill_size: int | None = None
    """Set when SGLang's default chunk leaves too little KV pool on this many GPUs."""

    @property
    def flags(self) -> dict[str, Any]:
        flags: dict[str, Any] = {"--tp-size": self.parallel.tp}
        if self.chunked_prefill_size is not None:
            flags["--chunked-prefill-size"] = self.chunked_prefill_size
        return flags


@dataclass(frozen=True)
class PrecisionRoute:
    key: str
    title: str
    weight_override: str | None
    """``ExecutionConfig.weight_dtype`` for the cost model; None keeps the checkpoint's precision."""
    compute_dtype: str
    available: bool
    """The device has the arithmetic this route needs."""
    blockers: tuple[str, ...]
    kernels: tuple[str, ...]
    kernels_missing: tuple[str, ...]
    accuracy: str
    steps: tuple[str, ...]
    flags: dict[str, Any] = field(default_factory=dict)
    sizing: RouteSizing | None = None
    references: tuple[str, ...] = ()

    @property
    def min_gpus(self) -> int | None:
        return self.sizing.parallel.num_gpus if self.sizing else None


@dataclass(frozen=True)
class KvCacheOption:
    dtype: str
    flag: str | None
    kv_tokens: int
    available: bool
    note: str


@dataclass(frozen=True)
class NumericsCheck:
    title: str
    detail: str
    references: tuple[str, ...] = ()


@dataclass
class PrecisionPlan:
    checkpoint: str
    routes: list[PrecisionRoute]
    recommended: PrecisionRoute | None
    kv_options: list[KvCacheOption]
    numerics: list[NumericsCheck]
    min_kv_tokens: int


def _declared(hardware: HardwareSpec) -> set[str]:
    return {name.lower() for name in hardware.software.fused_ops}


def _gemm_dtypes(hardware: HardwareSpec) -> set[str]:
    dtypes: set[str] = set()
    for name in hardware.software.gemm_dtypes:
        try:
            dtypes.add(normalize_dtype(name))
        except ValueError:
            continue
    return dtypes


def _has_unit(hardware: HardwareSpec, dtype: str) -> bool:
    return hardware.supports_compute_dtype(dtype) and dtype in _gemm_dtypes(hardware)


def size_route(
    arch: ModelArch,
    hardware: HardwareSpec,
    weight_override: str | None,
    *,
    min_kv_tokens: int = DEFAULT_MIN_KV_TOKENS,
    max_gpus: int | None = None,
) -> RouteSizing | None:
    """Smallest pure-TP layout whose KV pool holds ``min_kv_tokens``; None if none fits.

    On each GPU count SGLang's default chunked-prefill size is tried first, then smaller chunks.
    """
    limit = max_gpus or hardware.devices_per_node * MAX_NODES
    for count in _GPU_COUNTS:
        if count > limit:
            break
        parallel = ParallelConfig(tp=count)
        if parallel.validate(arch):
            continue
        chosen = None
        for chunk in CHUNKED_PREFILL_LADDER:
            plan = plan_memory(
                arch,
                hardware,
                parallel,
                weight_dtype=weight_override,
                knobs=MemoryKnobs(chunked_prefill_size=chunk),
            )
            if plan.max_total_tokens >= min_kv_tokens:
                chosen = (chunk, plan)
                break
        if chosen is None:
            continue
        chunk, plan = chosen
        decode_ms = prefill_ms = None
        try:
            model = StepCostModel(
                arch, hardware, parallel, ExecutionConfig(weight_dtype=weight_override)
            )
            decode_ms = model.estimate(BatchShape.decode(32, 4096)).total_us / 1000
            prefill_ms = model.estimate(BatchShape.prefill(1, 4096)).total_us / 1000
        except ValueError:
            pass
        return RouteSizing(
            parallel=parallel,
            weights_gb_per_gpu=plan.weights_bytes_per_gpu / 1e9,
            kv_tokens=plan.max_total_tokens,
            decode_step_ms=decode_ms,
            prefill_step_ms=prefill_ms,
            chunked_prefill_size=chunk,
        )
    return None


def _checkpoint_dtype(arch: ModelArch) -> str:
    precision = weight_precision(arch)
    if precision.linear_dtype == precision.experts_dtype:
        return precision.linear_dtype
    return f"{precision.linear_dtype}+experts {precision.experts_dtype}"


def _routes(
    arch: ModelArch, hardware: HardwareSpec, profile: PlatformProfile
) -> list[PrecisionRoute]:
    quant = arch.quant
    declared = _declared(hardware)
    activation = arch.activation_dtype
    in_tree = profile.in_tree
    is_deepseek = arch.primary_architecture in DEEPSEEK_V2_MODEL_ARCHITECTURES
    routes: list[PrecisionRoute] = []

    def missing(*names: str) -> tuple[str, ...]:
        if in_tree:
            return ()
        return tuple(name for name in names if name not in declared)

    if quant is None:
        available = hardware.supports_compute_dtype(activation)
        routes.append(
            PrecisionRoute(
                key="checkpoint",
                title=f"保持 checkpoint 精度（{activation}）",
                weight_override=None,
                compute_dtype=activation,
                available=available,
                blockers=() if available else (f"设备没有 {activation} 计算单元",),
                kernels=(),
                kernels_missing=(),
                accuracy="基准精度",
                steps=("直接加载",),
                references=(source("unquantized_linear"),),
            )
        )
    else:
        weight = quant.weight_dtype
        if weight in ("fp8", "int8"):
            unit = _has_unit(hardware, weight)
            kernels = (
                ("fp8_blockwise_gemm", "per_token_group_quant_fp8", "fused_moe")
                if weight == "fp8" and quant.block_shape
                else (f"{weight}_gemm",) + (("fused_moe",) if arch.moe else ())
            )
            routes.append(
                PrecisionRoute(
                    key="checkpoint",
                    title=f"保持 checkpoint 精度（{quant.method}，W8A8 {weight}）",
                    weight_override=None,
                    compute_dtype=weight if unit else activation,
                    available=unit,
                    blockers=()
                    if unit
                    else (f"设备没有 {weight} 计算单元或 YAML gemm_dtypes 未声明",),
                    kernels=kernels,
                    kernels_missing=missing(*kernels),
                    accuracy="与 checkpoint 一致",
                    steps=(
                        "直接加载（量化参数从 config.json 的 quantization_config 自动识别）",
                        *(
                            ()
                            if in_tree
                            else (
                                f"插件 get_quantization_config({quant.method!r}) 返回厂商 "
                                "QuantizationConfig",
                            )
                        ),
                    ),
                    references=(source("oot_quantization_hook"),),
                )
            )
        else:
            kernels = ("w4a16_gemm",) + (("fused_moe",) if arch.moe else ())
            routes.append(
                PrecisionRoute(
                    key="checkpoint",
                    title=f"保持 checkpoint 精度（{quant.method}，{weight} 权重）",
                    weight_override=None,
                    compute_dtype=activation,
                    available=True,
                    blockers=(),
                    kernels=kernels,
                    kernels_missing=missing(*kernels),
                    accuracy="与 checkpoint 一致",
                    steps=("直接加载",),
                    references=(source("oot_quantization_hook"),),
                )
            )
        moe_kernels = ("fused_moe",) if arch.moe else ()
        routes.append(
            PrecisionRoute(
                key="bf16",
                title="反量化为 BF16",
                weight_override="bf16",
                compute_dtype="bf16" if hardware.supports_compute_dtype("bf16") else "fp16",
                available=hardware.supports_compute_dtype("bf16")
                or hardware.supports_compute_dtype("fp16"),
                blockers=(),
                kernels=moe_kernels,
                kernels_missing=missing(*moe_kernels),
                accuracy=(
                    "近似无损：fp8×scale 舍入到 bf16，单个权重相对误差 ≤ 2^-9"
                    if weight == "fp8"
                    else "与 checkpoint 的反量化值一致"
                ),
                steps=(
                    (
                        "用 DeepSeek-V3 仓库的 inference/fp8_cast_bf16.py 把 FP8 权重转成 BF16"
                        if is_deepseek and weight == "fp8"
                        else "离线把量化权重反量化成 BF16"
                    ),
                    "删除 config.json 里的 quantization_config，否则 SGLang 仍按 block 量化切分权重并报 "
                    "output_partition_size 不能被 block_n 整除",
                    "权重翻倍：需要的卡数通常也翻倍（见 min_gpus）",
                ),
                flags={"--dtype": "bfloat16"},
                references=(source("fp8_to_bf16_precedent"), source("bf16_remove_quant_config")),
            )
        )
        if weight == "fp8":
            routes.append(
                PrecisionRoute(
                    key="fp8_w8a16",
                    title="保留 FP8 权重，GEMM 内反量化（W8A16）",
                    weight_override="fp8",
                    compute_dtype=activation,
                    available=True,
                    blockers=(),
                    kernels=("fp8_w8a16_gemm", *moe_kernels),
                    kernels_missing=missing("fp8_w8a16_gemm", *moe_kernels),
                    accuracy="与 checkpoint 一致（激活保持 bf16，比 W8A8 更准）",
                    steps=(
                        "checkpoint 不变；插件的 QuantizationConfig 给 Linear/FusedMoE 返回 W8A16 方法："
                        "按 block scale 把 FP8 权重在 kernel 内反量化为 bf16 再做 MMA",
                        "显存与 FP8 相同，算力按 bf16 计：decode（带宽受限）几乎不受影响，prefill 变慢",
                    ),
                    references=(source("oot_quantization_hook"),),
                )
            )
    if (quant is None or quant.weight_dtype != "int8") and _has_unit(hardware, "int8"):
        routes.append(
            PrecisionRoute(
                key="w8a8_int8",
                title="重量化为 W8A8 INT8（权重 per-channel，激活 per-token 动态）",
                weight_override="int8",
                compute_dtype="int8",
                available=True,
                blockers=(),
                kernels=("int8_gemm",) + (("fused_moe",) if arch.moe else ()),
                kernels_missing=missing("int8_gemm", *(("fused_moe",) if arch.moe else ())),
                accuracy="有精度风险：必须跑 gsm8k/mmlu 与原精度对比（社区有 DeepSeek-R1 channel INT8 先例）",
                steps=(
                    *(
                        ("先把 FP8 权重反量化成 BF16",)
                        if quant and quant.weight_dtype == "fp8"
                        else ()
                    ),
                    "用 per-channel 对称量化生成 INT8 权重（可配合 SmoothQuant），保存为 w8a8_int8 格式",
                    "启动时加 --quantization w8a8_int8；out-of-tree 平台由插件的 "
                    "get_quantization_config('w8a8_int8') 提供 INT8 GEMM 与激活量化",
                ),
                flags={"--quantization": "w8a8_int8"},
                references=(
                    source("int8_checkpoint_precedent"),
                    source("w8a8_int8_cuda_only"),
                    source("oot_quantization_hook"),
                ),
            )
        )
    if quant is None or quant.weight_dtype not in ("int4", "fp4"):
        kernels = ("w4a16_gemm",) + (("fused_moe",) if arch.moe else ())
        routes.append(
            PrecisionRoute(
                key="w4a16",
                title="W4A16（AWQ/GPTQ int4 权重）",
                weight_override="int4",
                compute_dtype=activation,
                available=True,
                blockers=(),
                kernels=kernels,
                kernels_missing=missing(*kernels),
                accuracy="精度风险最高：必须评测；MoE 专家通常比 attention 更耐受",
                steps=(
                    "用 AWQ/GPTQ 量化（group_size 128），MoE 模型启动时用 --quantization moe_wna16",
                    "decode 读权重的字节减半，适合显存/带宽受限的小 batch 场景",
                ),
                flags={"--quantization": "moe_wna16" if arch.moe else "awq"},
                references=(source("oot_quantization_hook"),),
            )
        )
    return routes


_ACCURACY_RISK = {"checkpoint": 0, "fp8_w8a16": 0, "bf16": 1, "w8a8_int8": 2, "w4a16": 3}
LOSSLESS_ROUTES = frozenset({"checkpoint", "fp8_w8a16", "bf16"})
PREFERENCES = ("accuracy", "cost")


def _rank(route: PrecisionRoute, prefer: str) -> tuple:
    gpus = route.min_gpus or 10**6
    risk = _ACCURACY_RISK[route.key]
    missing = len(route.kernels_missing)
    if prefer == "cost":
        return (not route.available, gpus, missing, risk)
    return (not route.available, route.key not in LOSSLESS_ROUTES, gpus, missing, risk)


def _kv_options(
    arch: ModelArch, hardware: HardwareSpec, route: PrecisionRoute | None
) -> list[KvCacheOption]:
    if route is None or route.sizing is None:
        return []
    parallel = route.sizing.parallel
    supported = set()
    for name in hardware.software.kv_cache_dtypes:
        try:
            supported.add(normalize_dtype(name))
        except ValueError:
            continue
    options = []
    knobs = MemoryKnobs(chunked_prefill_size=route.sizing.chunked_prefill_size)
    for dtype, flag in (("bf16", None), ("fp8", "fp8_e4m3")):
        plan = plan_memory(
            arch,
            hardware,
            parallel,
            weight_dtype=route.weight_override,
            kv_dtype=dtype,
            knobs=knobs,
        )
        available = dtype in supported or dtype == normalize_dtype(arch.torch_dtype)
        note = (
            "默认（--kv-cache-dtype auto）"
            if flag is None
            else "KV 容量约翻倍；注意力 kernel 必须直接读 FP8 KV（含 MLA latent），并评测长上下文精度"
        )
        if not available:
            note = f"YAML kv_cache_dtypes={list(hardware.software.kv_cache_dtypes)} 不含 {dtype}"
        options.append(KvCacheOption(dtype, flag, plan.max_total_tokens, available, note))
    return options


def _numerics(arch: ModelArch, profile: PlatformProfile) -> list[NumericsCheck]:
    checks = [
        NumericsCheck(
            "RMSNorm 在 fp32 里累加",
            "SGLang 的参考实现先把 x（以及 residual 相加）转成 fp32 再求方差；厂商 kernel 必须同样"
            "在 fp32 里做 add+norm，并把更新后的 residual 以原 dtype 返回",
            (source("rmsnorm_fp32"),),
        ),
        NumericsCheck(
            "路由权重 fp32、专家 id int32",
            "TopK 输出 topk_weights 为 float32、topk_ids 为 int32；MoE kernel 按该 dtype 读取",
            (source("topk_output_dtypes"),),
        ),
    ]
    if profile.out_of_tree:
        checks.append(
            NumericsCheck(
                "RoPE cos/sin cache 精度",
                "离开 CUDA/HIP/XPU 时 SGLang 把 cos/sin cache 转成模型 dtype（bf16）；长上下文先用 "
                "SGLANG_ROPE_CACHE_FP32=1 保持 fp32，再评估是否能关掉",
                (source("rope_cache_dtype"),),
            )
        )
    if arch.attention.is_mla and arch.raw.get("rope_scaling"):
        checks.append(
            NumericsCheck(
                "MLA softmax scale 含 YaRN mscale",
                "DeepSeek MLA 的 layer.scaling 已乘上 YaRN mscale；注意力后端必须使用传入的 "
                "scaling，不能自己按 1/sqrt(head_dim) 重算",
                (source("mla_mscale"),),
            )
        )
    checks.append(
        NumericsCheck(
            "逐层对齐与端到端验收",
            "SGLANG_FORCE_FUSED_OP_BACKEND=torch 让所有 BaseFusedOp 走 forward_native，用来二分是哪个厂商"
            "算子引入误差；DUMPER_ENABLE=1 分别在参考 GPU 与目标设备上 dump 张量，再用 python -m "
            "sglang.srt.debug_utils.dump_comparator --baseline-path <gpu> --target-path <dev> 对比；"
            "model_truncator --keep-num-layers 2 生成小模型快速迭代；最后用 python -m "
            "sglang.test.run_eval --eval-name gsm8k 验收（该脚本已弃用但仍支持 gsm8k，mmlu 等走 sgl-eval）",
            (
                source("fused_op_force_backend"),
                source("dumper"),
                source("dump_comparator"),
                source("model_truncator"),
                source("run_eval"),
            ),
        )
    )
    return checks


def precision_plan(
    arch: ModelArch,
    hardware: HardwareSpec,
    profile: PlatformProfile,
    *,
    min_kv_tokens: int = DEFAULT_MIN_KV_TOKENS,
    max_gpus: int | None = None,
    prefer: str = "accuracy",
) -> PrecisionPlan:
    """Rank precision routes; ``accuracy`` keeps lossless routes first, ``cost`` minimizes GPUs."""
    if prefer not in PREFERENCES:
        raise ValueError(f"prefer must be one of {PREFERENCES}, got {prefer!r}")
    sized = []
    for route in _routes(arch, hardware, profile):
        sizing = size_route(
            arch,
            hardware,
            route.weight_override,
            min_kv_tokens=min_kv_tokens,
            max_gpus=max_gpus,
        )
        blockers = route.blockers
        if sizing is None:
            limit = max_gpus or hardware.devices_per_node * MAX_NODES
            blockers = (*blockers, f"{limit} 卡以内放不下权重 + {min_kv_tokens} token 的 KV")
        sized.append(replace(route, sizing=sizing, blockers=blockers))
    sized.sort(key=lambda route: _rank(route, prefer))
    usable = [route for route in sized if route.available and route.sizing is not None]
    recommended = usable[0] if usable else None
    return PrecisionPlan(
        checkpoint=_checkpoint_dtype(arch),
        routes=sized,
        recommended=recommended,
        kv_options=_kv_options(arch, hardware, recommended),
        numerics=_numerics(arch, profile),
        min_kv_tokens=min_kv_tokens,
    )
