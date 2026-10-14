"""Synthetic Kineto-format traces built from the step cost model, with injectable performance issues.

They let the trace analyzer run end to end without an accelerator (tests, demos, CI). Kernel names
are representative of what SGLang launches on Hopper (FA3, DeepGEMM/cuBLAS, Triton fused MoE,
custom all-reduce, NCCL); collectives take the device's ``collective_library`` name instead when it
is not NCCL/RCCL, and custom all-reduce only appears when the device declares it. Every payload
carries ``"synthetic": true`` and reports flag it; these traces are not measurements.
"""

from __future__ import annotations

import gzip
import json
import random
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.model.config import ModelArch
from sgl_agent.model.ops import ACTIVATION_BYTES, BatchShape, ExecutionConfig, OpCost, StepCostModel
from sgl_agent.model.parallel import ParallelConfig

EAGER_LAUNCH_CPU_US = 20.0
"""Python dispatch plus cudaLaunchKernel per kernel when a step runs eagerly (assumption)."""
GRAPH_LAUNCH_CPU_US = 12.0
HOST_SYNC_CPU_US = 60.0
STEP_PREPARE_CPU_US = 40.0
SCHEDULER_CPU_US = 150.0
OVERLAP_LOOKAHEAD_US = 400.0
TRACE_BASE_US = 1_000_000.0
MAIN_STREAM = 7

_FA3 = (
    "void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<"
    "flash::CollectiveMainloopFwdSm90<2, cute::tuple<cute::C<1>, cute::C<1>, cute::C<1>>, 128, "
    "cutlass::bfloat16_t, float, cutlass::arch::Sm90>>>>(flash::FlashAttnFwdSm90<int>::Params)"
)
_DEEP_GEMM = (
    "void deep_gemm::sm90_fp8_gemm_1d2d_impl<0u, 128u, 128u, 1u, 4u>(float*, int*, unsigned int, "
    "unsigned int, unsigned int, CUtensorMap, CUtensorMap, CUtensorMap, CUtensorMap)"
)
_CUBLAS_BF16 = "nvjet_tst_64x8_64x16_4x1_v_bz_TNT"
_FUSED_ADD_RMSNORM = (
    "void flashinfer::norm::FusedAddRMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, "
    "__nv_bfloat16*, __nv_bfloat16*, unsigned int, unsigned int, unsigned int, float, float)"
)
_RMSNORM = (
    "void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat16>(__nv_bfloat16*, __nv_bfloat16*, "
    "__nv_bfloat16*, unsigned int, unsigned int, unsigned int, unsigned int, float, float)"
)
_ADD = (
    "void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<c10::BFloat16>, "
    "at::detail::Array<char*, 3> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, "
    "at::detail::Array<char*, 3>)"
)
_MUL = (
    "void at::native::vectorized_elementwise_kernel<4, at::native::binary_internal::MulFunctor<float>, "
    "at::detail::Array<char*, 3> >(int, at::native::binary_internal::MulFunctor<float>, "
    "at::detail::Array<char*, 3>)"
)
_SILU = (
    "void at::native::vectorized_elementwise_kernel<4, at::native::(anonymous namespace)::"
    "silu_kernel(at::TensorIteratorBase&)::{lambda()#1}, at::detail::Array<char*, 2> >(int, "
    "at::detail::Array<char*, 2>)"
)
_CAST = (
    "void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda("
    "at::TensorIteratorBase&)::{lambda()#3}, at::detail::Array<char*, 2> >(int, at::detail::Array<char*, 2>)"
)
_CAT = (
    "void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::"
    "OpaqueType<2u>, unsigned int, 2, 128, 1>(at::native::(anonymous namespace)::OpaqueType<2u>*, "
    "at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::"
    "OpaqueType<2u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<"
    "unsigned int, 4u>, int, unsigned int)"
)
_ACT_AND_MUL = "void flashinfer::activation::act_and_mul_kernel<__nv_bfloat16, &silu>(__nv_bfloat16*, __nv_bfloat16 const*, int)"
_ROPE = (
    "void flashinfer::BatchQKApplyRotaryPosIdsCosSinCacheKernel<true, 128u, 8u, 16u, __nv_bfloat16, long>("
    "__nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, long*, unsigned int)"
)
_QUANT = (
    "void per_token_group_quant_8bit_kernel<__nv_bfloat16, c10::Float8_e4m3fn, false>(__nv_bfloat16 const*, "
    "void*, float*, int, int, int, float, float, float, int, int)"
)
_CUSTOM_ALLREDUCE = (
    "void sglang::cross_device_reduce_1stage<__nv_bfloat16, 8>(sglang::RankData*, sglang::RankSignals, "
    "sglang::Signal*, __nv_bfloat16*, int, int)"
)
_NCCL_ALLREDUCE = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL128(ncclDevKernelArgsStorage<4096ul>)"
_NCCL_ALLGATHER = "ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
_NCCL_REDUCE_SCATTER = (
    "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
)
_NCCL_SENDRECV = "ncclDevKernel_SendRecv(ncclDevKernelArgsStorage<4096ul>)"
_DEEPEP_DISPATCH = (
    "void deep_ep::internode_ll::dispatch<true, 8, 7168>(void*, float*, int*, long*, void*)"
)
_DEEPEP_COMBINE = (
    "void deep_ep::internode_ll::combine<8, 7168>(void*, int*, long const*, float const*)"
)
_SAMPLING = (
    "void flashinfer::sampling::TopKTopPSamplingFromProbKernel<1024u, (cub::BlockScanAlgorithm)2, "
    "(cub::BlockReduceAlgorithm)2, 4u, true, float, int>(float*, int*, int*, float*, float*, unsigned int)"
)
_EMBED = (
    "void at::native::(anonymous namespace)::indexSelectLargeIndex<c10::BFloat16, long, unsigned int, 2, 2, -2, true>("
    "at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, long, long)"
)
_MEMCPY_DTOH = "Memcpy DtoH (Device -> Pinned)"
CUSTOM_ALLREDUCE_MAX_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class SyntheticStep:
    stage: str
    batch_size: int
    context_len: int
    new_tokens_per_seq: int = 1

    def __post_init__(self) -> None:
        if self.stage not in {"EXTEND", "DECODE"}:
            raise ValueError("synthetic steps support EXTEND and DECODE")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")

    def shape(self) -> BatchShape:
        if self.stage == "DECODE":
            return BatchShape.decode(self.batch_size, self.context_len)
        return BatchShape.prefill(self.batch_size, self.new_tokens_per_seq, self.context_len)


@dataclass(frozen=True)
class SyntheticIssues:
    """Problems injected so the analyzer has something real to find."""

    unfused_add_rmsnorm: bool = True
    unfused_silu_mul: bool = False
    elementwise_chain: bool = True
    host_sync_in_prefill: bool = True
    attention_slowdown_decode_ordinal: int | None = 3
    attention_slowdown_factor: float = 3.0
    attention_slowdown_layers: tuple[int, int] = (10, 14)
    eager_decode_batch: int | None = 48
    rank_compute_skew: float = 0.0

    @staticmethod
    def none() -> SyntheticIssues:
        return SyntheticIssues(
            unfused_add_rmsnorm=False,
            unfused_silu_mul=False,
            elementwise_chain=False,
            host_sync_in_prefill=False,
            attention_slowdown_decode_ordinal=None,
            eager_decode_batch=None,
        )


def default_synthetic_steps(context_len: int = 4096, prompt_len: int = 4096) -> list[SyntheticStep]:
    return [
        SyntheticStep("EXTEND", 1, 0, prompt_len),
        *[SyntheticStep("DECODE", 32, context_len) for _ in range(6)],
        *[SyntheticStep("DECODE", 48, context_len) for _ in range(2)],
    ]


@dataclass(frozen=True)
class _Kernel:
    name: str
    duration_us: float
    comm: bool = False
    kind: str = "kernel"
    host_sync: bool = False
    nbytes: int | None = None
    attention: bool = False
    grid_key: str | None = None
    """Distinguishes launches of one kernel with different shapes (real traces differ in grid)."""


def _stable_grid(key: str) -> list[int]:
    return [zlib.crc32(key.encode()) % 240 + 16, 1, 1]


def step_span_name(step: SyntheticStep, detailed: bool = True) -> str:
    """Same layout as ``sglang/srt/utils/profile_utils.py::build_step_span_name``."""
    batch = step.batch_size
    if step.stage == "DECODE":
        base = f"step[DECODE bs={batch}"
        if detailed:
            kv = batch * step.context_len
            base += f" g_sq={batch} g_sqsq={batch} g_sqsk={kv} g_sk={kv}"
        return base + "]"
    query = step.new_tokens_per_seq
    base = f"step[EXTEND bs={batch} toks={batch * query}"
    if detailed:
        kv = step.context_len + query
        base += (
            f" c_sq={batch * query} c_sqsq={batch * query * query}"
            f" c_sqsk={batch * query * kv} c_sk={batch * kv}"
        )
    return base + "]"


class _KernelPlanner:
    def __init__(self, model: StepCostModel, issues: SyntheticIssues):
        self.model = model
        self.issues = issues
        self.arch = model.arch
        self.hardware = model.hardware
        self.floor_us = model.hardware.latency.min_kernel_us
        self.bandwidth = (
            model.hardware.memory_bandwidth_bytes_per_s * model.hardware.efficiency.memory_bandwidth
        )

    def _duration(self, op: OpCost) -> float:
        return self.model.time_op(op, cuda_graph=True).kernel_us

    def _moe_activation_us(self, tokens: float) -> float:
        """act_and_mul between the two grouped GEMMs, on the routed (token, expert) rows."""
        moe, parallel = self.arch.moe, self.model.parallel
        assert moe is not None
        group_tokens = tokens * parallel.attn_cp * parallel.attn_dp
        rows = group_tokens * moe.top_k / parallel.ep
        width = moe.expert_intermediate_size / parallel.moe_tp
        moved = rows * width * 3 * ACTIVATION_BYTES
        return max(self.floor_us, moved / self.bandwidth * 1e6)

    def _gemm_name(self, op: OpCost) -> str:
        return _DEEP_GEMM if op.compute_dtype in {"fp8", "int8"} else _CUBLAS_BF16

    def _comm_name(self, op: OpCost) -> str:
        software = self.hardware.software
        library = software.collective_library.lower()
        if library not in {"nccl", "rccl"}:
            prefix = re.sub(r"[^a-z0-9]+", "_", library).strip("_") or "ccl"
            return f"{prefix}_{op.comm_pattern or 'alltoall'}_kernel"
        if op.comm_pattern == "allreduce":
            return (
                _CUSTOM_ALLREDUCE
                if software.custom_allreduce and op.comm_bytes <= CUSTOM_ALLREDUCE_MAX_BYTES
                else _NCCL_ALLREDUCE
            )
        if op.comm_pattern == "allgather":
            return _NCCL_ALLGATHER
        if op.comm_pattern == "reducescatter":
            return _NCCL_REDUCE_SCATTER
        if "dispatch" in op.name:
            return _DEEPEP_DISPATCH
        if "combine" in op.name:
            return _DEEPEP_COMBINE
        return _NCCL_SENDRECV

    def _chain(self, tokens: float) -> list[_Kernel]:
        duration = max(self.floor_us, tokens * self.arch.hidden_size * 4.0 / self.bandwidth * 1e6)
        return [_Kernel(_CAST, duration), _Kernel(_MUL, duration), _Kernel(_CAT, duration)]

    def expand(self, op: OpCost, layer: int, stage: str, tokens: float) -> list[_Kernel]:
        duration = self._duration(op)
        kernels = max(op.kernels, 1)
        per_kernel = max(duration / kernels, self.floor_us)
        issues = self.issues
        if op.kind == "comm":
            return [_Kernel(self._comm_name(op), max(duration, self.floor_us), comm=True)]
        if op.kind == "norm":
            residual = op.name in {"attn.post_norm", "final_norm"} or (
                op.name == "attn.input_norm" and layer > 0
            )
            if residual and issues.unfused_add_rmsnorm:
                return [
                    _Kernel(_ADD, max(per_kernel * 0.55, self.floor_us)),
                    _Kernel(_RMSNORM, max(per_kernel * 0.75, self.floor_us)),
                ]
            name = _FUSED_ADD_RMSNORM if residual else _RMSNORM
            return [_Kernel(name, per_kernel) for _ in range(kernels)]
        if op.kind == "quant":
            return [_Kernel(_QUANT, per_kernel)]
        if op.kind == "gemm":
            return [
                _Kernel(self._gemm_name(op), per_kernel, grid_key=op.name) for _ in range(kernels)
            ]
        if op.kind == "rope":
            return [_Kernel(_ROPE, per_kernel)]
        if op.kind == "kv_store":
            name = (
                "set_mla_kv_buffer_kernel" if self.arch.attention.is_mla else "store_kvcache_kernel"
            )
            return [_Kernel(name, per_kernel)]
        if op.kind == "attention":
            name = "sparse_mla_indexer_fwd_kernel" if op.name == "attn.dsa_indexer" else _FA3
            result = [_Kernel(name, per_kernel, attention=True) for _ in range(kernels)]
            if issues.elementwise_chain and op.name != "attn.dsa_indexer":
                result += self._chain(tokens)
            return result
        if op.kind == "moe_routing":
            if op.name == "moe.topk":
                grouped = self.arch.moe is not None and "grouped" in self.arch.moe.routing
                name = (
                    "void moe_fused_gate_kernel<__nv_bfloat16, 256, 8, 4>(void*, void*, float*, int*, int, int)"
                    if grouped
                    else "void topkGatingSoftmax<8, 128, 4, 16>(float const*, bool const*, float*, int, int*, int, int, int)"
                )
                result = [_Kernel(name, per_kernel)]
                if stage == "EXTEND" and issues.host_sync_in_prefill:
                    result.append(
                        _Kernel(_MEMCPY_DTOH, 2.0, kind="memcpy", host_sync=True, nbytes=1024)
                    )
                return result
            if op.name == "moe.align":
                return [
                    _Kernel(
                        "void moe_align_block_size_kernel<int>(int*, int*, int*, int*, int, int, int, int, unsigned long)",
                        per_kernel,
                    )
                ]
            return [
                _Kernel(
                    "void moe_sum_reduce_kernel<__nv_bfloat16, 8>(__nv_bfloat16 const*, __nv_bfloat16*, int, int, int)",
                    per_kernel,
                )
            ]
        if op.kind == "moe":
            activation = self._moe_activation_us(tokens)
            grouped_gemm = max(duration - activation, 2 * self.floor_us)
            gate_up_share = 2 / 3 if self.arch.gated_mlp else 1 / 2
            return [
                _Kernel("fused_moe_kernel", grouped_gemm * gate_up_share, grid_key="moe.gate_up"),
                _Kernel(_ACT_AND_MUL, activation, grid_key="moe.act"),
                _Kernel(
                    "fused_moe_kernel", grouped_gemm * (1 - gate_up_share), grid_key="moe.down"
                ),
            ]
        if op.kind == "activation":
            if issues.unfused_silu_mul:
                return [
                    _Kernel(_SILU, max(per_kernel * 0.7, self.floor_us), grid_key=op.name),
                    _Kernel(_MUL, max(per_kernel * 0.7, self.floor_us), grid_key=op.name),
                ]
            result = [_Kernel(_ACT_AND_MUL, per_kernel, grid_key=op.name)]
            if (
                stage == "EXTEND"
                and issues.host_sync_in_prefill
                and self.arch.moe is None
                and op.name == "mlp.act_and_mul"
            ):
                result.append(
                    _Kernel(_MEMCPY_DTOH, 2.0, kind="memcpy", host_sync=True, nbytes=1024)
                )
            return result
        if op.kind == "sampling":
            return [_Kernel(_SAMPLING, max(duration, self.floor_us))]
        if op.name == "embed.lookup":
            return [_Kernel(_EMBED, per_kernel)]
        return [_Kernel(_ADD, per_kernel) for _ in range(kernels)]

    def step_kernels(
        self, step: SyntheticStep, decode_ordinal: int | None
    ) -> tuple[list[_Kernel], list[_Kernel]]:
        """Kernels inside the step span and the sampling kernels launched after it."""
        model, arch, issues = self.model, self.arch, self.issues
        shape = step.shape()
        tokens = shape.new_tokens / model.parallel.attn_cp
        head = model.head_ops(shape)
        head_by_name = {op.name: op for op in head}
        kernels: list[_Kernel] = []
        for name in ("embed.lookup", "embed.allreduce"):
            if name in head_by_name:
                kernels += self.expand(head_by_name[name], -1, step.stage, tokens)
        moe_layers = set(arch.moe.moe_layer_indices) if arch.moe else set()
        layers = max(arch.num_layers // model.parallel.pp, 1)
        slow_layers = range(*issues.attention_slowdown_layers)
        slow_step = (
            decode_ordinal is not None
            and issues.attention_slowdown_decode_ordinal is not None
            and decode_ordinal == issues.attention_slowdown_decode_ordinal
        )
        per_layer_cache: dict[bool, list[OpCost]] = {}
        for layer in range(layers):
            is_moe = layer in moe_layers
            ops = per_layer_cache.get(is_moe)
            if ops is None:
                ops = model.layer_ops(shape, moe_layer=is_moe)
                per_layer_cache[is_moe] = ops
            for op in ops:
                for kernel in self.expand(op, layer, step.stage, tokens):
                    if slow_step and kernel.attention and layer in slow_layers:
                        kernel = _Kernel(
                            kernel.name,
                            kernel.duration_us * issues.attention_slowdown_factor,
                            attention=True,
                        )
                    kernels.append(kernel)
        for name in ("final_norm", "lm_head", "lm_head.allgather"):
            if name in head_by_name:
                kernels += self.expand(head_by_name[name], 1, step.stage, tokens)
        after = (
            self.expand(head_by_name["sampling"], -1, step.stage, tokens)
            if "sampling" in head_by_name
            else []
        )
        return kernels, after


@dataclass
class _RankState:
    rank: int
    pid: int
    device: int
    cpu_us: float = TRACE_BASE_US
    gpu_us: float = TRACE_BASE_US
    correlation: int = 1000
    events: list[dict[str, Any]] = field(default_factory=list)
    rng: random.Random = field(default_factory=random.Random)

    def next_correlation(self) -> int:
        self.correlation += 1
        return self.correlation

    def runtime(self, name: str, start: float, duration: float, correlation: int | None) -> None:
        args: dict[str, Any] = {}
        if correlation is not None:
            args["correlation"] = correlation
        self.events.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": name,
                "pid": self.pid,
                "tid": self.pid,
                "ts": round(start, 3),
                "dur": round(duration, 3),
                "args": args,
            }
        )

    def kernel(self, kernel: _Kernel, start: float, duration: float, correlation: int) -> None:
        args: dict[str, Any] = {
            "device": self.device,
            "stream": MAIN_STREAM,
            "correlation": correlation,
        }
        if kernel.kind == "kernel":
            args["grid"] = _stable_grid(kernel.grid_key or kernel.name)
            args["block"] = [128, 1, 1]
        if kernel.nbytes is not None:
            args["bytes"] = kernel.nbytes
        self.events.append(
            {
                "ph": "X",
                "cat": "kernel" if kernel.kind == "kernel" else "gpu_memcpy",
                "name": kernel.name,
                "pid": self.device,
                "tid": MAIN_STREAM,
                "ts": round(start, 3),
                "dur": round(max(duration, 0.5), 3),
                "args": args,
            }
        )


def generate_synthetic_traces(
    arch: ModelArch,
    hardware: HardwareSpec,
    parallel: ParallelConfig,
    *,
    steps: list[SyntheticStep] | None = None,
    issues: SyntheticIssues | None = None,
    num_ranks: int = 1,
    execution: ExecutionConfig | None = None,
    detailed_annotations: bool = True,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """One Kineto-style payload per rank; collectives finish together across ranks."""
    if num_ranks < 1:
        raise ValueError("num_ranks must be >= 1")
    issues = issues or SyntheticIssues()
    steps = steps or default_synthetic_steps()
    model = StepCostModel(arch, hardware, parallel, execution or ExecutionConfig())
    planner = _KernelPlanner(model, issues)
    states = [
        _RankState(
            rank=rank,
            pid=20000 + rank,
            device=rank % max(hardware.devices_per_node, 1),
            rng=random.Random(seed * 1000 + rank),
        )
        for rank in range(num_ranks)
    ]
    skew = [
        1.0 + issues.rank_compute_skew * (rank / (num_ranks - 1) if num_ranks > 1 else 0.0)
        for rank in range(num_ranks)
    ]
    decode_ordinal = -1
    for step in steps:
        if step.stage == "DECODE":
            decode_ordinal += 1
        in_step, after_step = planner.step_kernels(
            step, decode_ordinal if step.stage == "DECODE" else None
        )
        graph = step.stage == "DECODE" and step.batch_size != issues.eager_decode_batch
        name = step_span_name(step, detailed_annotations)
        span_start = {}
        graph_correlation = {}
        first_kernel_start: dict[int, float] = {}
        for state in states:
            span_start[state.rank] = state.cpu_us
            state.cpu_us += STEP_PREPARE_CPU_US
            if graph:
                correlation = state.next_correlation()
                state.runtime("cudaGraphLaunch", state.cpu_us, GRAPH_LAUNCH_CPU_US, correlation)
                state.cpu_us += GRAPH_LAUNCH_CPU_US
                state.gpu_us = max(state.gpu_us, state.cpu_us + 3.0)
                graph_correlation[state.rank] = correlation
        gap = hardware.latency.graph_kernel_gap_us if graph else 1.0
        for kernel in in_step:
            starts: dict[int, tuple[float, int]] = {}
            for state in states:
                if graph:
                    correlation = graph_correlation[state.rank]
                    ready = state.gpu_us + gap
                else:
                    correlation = state.next_correlation()
                    api = "cudaMemcpyAsync" if kernel.kind == "memcpy" else "cudaLaunchKernel"
                    state.runtime(api, state.cpu_us, EAGER_LAUNCH_CPU_US * 0.5, correlation)
                    ready = max(state.gpu_us + gap, state.cpu_us + EAGER_LAUNCH_CPU_US * 0.5 + 2.0)
                    state.cpu_us += EAGER_LAUNCH_CPU_US
                starts[state.rank] = (ready, correlation)
                first_kernel_start.setdefault(state.rank, ready)
            if kernel.comm:
                finish = max(start for start, _ in starts.values()) + kernel.duration_us * (
                    1.0 + states[0].rng.uniform(-0.03, 0.03)
                )
                for state in states:
                    start, correlation = starts[state.rank]
                    state.kernel(kernel, start, finish - start, correlation)
                    state.gpu_us = finish
            else:
                for state in states:
                    start, correlation = starts[state.rank]
                    duration = (
                        kernel.duration_us
                        * skew[state.rank]
                        * (1.0 + state.rng.uniform(-0.03, 0.03))
                    )
                    state.kernel(kernel, start, duration, correlation)
                    state.gpu_us = start + duration
            if kernel.host_sync:
                for state in states:
                    call_start = state.cpu_us
                    state.cpu_us = max(state.cpu_us, state.gpu_us) + 5.0
                    state.runtime(
                        "cudaStreamSynchronize", call_start, state.cpu_us - call_start, None
                    )
                    state.cpu_us += HOST_SYNC_CPU_US
        for state in states:
            span_end = state.cpu_us + 8.0
            state.events.append(
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": name,
                    "pid": state.pid,
                    "tid": state.pid,
                    "ts": round(span_start[state.rank], 3),
                    "dur": round(span_end - span_start[state.rank], 3),
                    "args": {},
                }
            )
            state.events.append(
                {
                    "ph": "X",
                    "cat": "gpu_user_annotation",
                    "name": name,
                    "pid": state.device,
                    "tid": MAIN_STREAM,
                    "ts": round(first_kernel_start[state.rank], 3),
                    "dur": round(state.gpu_us - first_kernel_start[state.rank], 3),
                    "args": {},
                }
            )
            state.cpu_us = span_end + 20.0
            for kernel in after_step:
                correlation = state.next_correlation()
                state.runtime(
                    "cudaLaunchKernel", state.cpu_us, EAGER_LAUNCH_CPU_US * 0.5, correlation
                )
                start = max(state.gpu_us + 1.0, state.cpu_us + EAGER_LAUNCH_CPU_US * 0.5 + 2.0)
                state.cpu_us += EAGER_LAUNCH_CPU_US
                state.kernel(kernel, start, kernel.duration_us, correlation)
                state.gpu_us = start + kernel.duration_us
            state.cpu_us = max(state.cpu_us + SCHEDULER_CPU_US, state.gpu_us - OVERLAP_LOOKAHEAD_US)

    payloads = []
    for state in states:
        payloads.append(
            {
                "schemaVersion": 1,
                "synthetic": True,
                "deviceProperties": [
                    {
                        "id": state.device,
                        "name": hardware.software.torch_device_name or hardware.name,
                        "totalGlobalMem": int(hardware.memory_capacity_bytes),
                        "numSms": hardware.num_compute_units or 0,
                    }
                ],
                "distributedInfo": {
                    "backend": hardware.software.collective_library,
                    "rank": state.rank,
                    "world_size": parallel.num_gpus,
                },
                "traceEvents": sorted(state.events, key=lambda event: event["ts"]),
            }
        )
    return payloads


def write_synthetic_traces(
    payloads: list[dict[str, Any]], output_dir: str | Path, profile_id: str = "synthetic"
) -> list[Path]:
    """Write ``{profile_id}-TP-{rank}.trace.json.gz`` files, the naming SGLang's profiler uses."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for payload in payloads:
        rank = payload["distributedInfo"]["rank"]
        path = directory / f"{profile_id}-TP-{rank}.trace.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle, separators=(",", ":"))
        written.append(path)
    return written
