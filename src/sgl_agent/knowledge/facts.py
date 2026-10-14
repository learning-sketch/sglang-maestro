"""SGLang behaviours the analyzers rely on, each pinned to the source that implements it.

The values were read from the SGLang commit recorded in the bundled snapshot. Paths are relative
to the SGLang repository root; re-check them after syncing a different SGLang version.
"""

from __future__ import annotations

FLASHINFER_ALLREDUCE_FUSION_MAX_TOKENS = 2048
"""``FUSE_ALLREDUCE_MAX_BATCH_SIZE``: the fused allreduce+RMSNorm path only runs up to this many tokens."""

FLASHINFER_ALLREDUCE_FUSION_SM_MAJORS = (9, 10)
"""FlashInfer allreduce fusion runs on SM90 and SM100-class GPUs only."""

SBO_UNSUPPORTED_SM_MAJOR = 9
"""``--enable-single-batch-overlap`` raises ``ValueError`` on SM90 with the current DeepGEMM wheel."""

FUSED_MOE_SUM_ALLREDUCE_MIN_TOP_K = 3
"""``--enable-fused-moe-sum-all-reduce`` only applies when ``topk > 2`` (Triton MoE runner)."""

TORCH_COMPILE_MAX_BS_DEFAULT = 32
"""``--torch-compile-max-bs`` default: torch.compile only covers decode batches up to this size."""

FUSED_QK_NORM_ROPE_HEAD_DIMS = (64, 128, 256)
"""Head dims the JIT fused QK-norm + RoPE kernel supports (BF16 only, no MRoPE)."""

FUSED_INPLACE_QKNORM_HEAD_DIMS = (64, 128, 256, 512, 1024)
"""Head dims the JIT in-place QK-norm kernel (q and k in one launch) supports."""

CUSTOM_ALLREDUCE_MAX_BYTES_CUDA = 8192 * 1024
"""``CustomAllreduce._MAX_CAR_SIZE`` on CUDA: larger allreduce messages leave the custom kernel."""

CUSTOM_ALLREDUCE_WORLD_SIZES = (2, 4, 6, 8)
"""World sizes the custom allreduce supports; beyond 2 GPUs it also needs full NVLink."""

TORCH_SYMM_MEM_SM90_WORLD_SIZES = (4, 6, 8)
TORCH_SYMM_MEM_SM100_WORLD_SIZES = (6, 8)
"""``--enable-torch-symm-mem`` world sizes; ``GroupCoordinator.all_reduce`` still prefers MSCCL++
and custom allreduce when they accept the message."""

DEEPSEEK_SHARED_EXPERTS_FUSION_ARCHITECTURES = frozenset(
    {"DeepseekV3ForCausalLM", "DeepseekV3ForCausalLMNextN", "GlmMoeDsaForCausalLM"}
)
"""``hf_config.architectures[0]`` must equal the class's ``fused_shared_experts_architecture``;
``DeepseekV2ForCausalLM``/``DeepseekV32ForCausalLM`` inherit ``"DeepseekV3ForCausalLM"`` and never match."""
DEEPSEEK_SHARED_EXPERTS_FUSION_ROUTED_EXPERTS = 256
DEEPSEEK_SHARED_EXPERTS_FUSION_QUARK_ROUTED_EXPERTS = 384
"""Kimi-K2.5 layout: fused only for the Quark MXFP4 checkpoint, which stores the shared expert pre-fused."""

GLM4_MOE_SHARED_EXPERTS_FUSION_ARCHITECTURE = "Glm4MoeForCausalLM"

DEEPSEEK_V2_MODEL_ARCHITECTURES = frozenset(
    {
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLMNextN",
        "GlmMoeDsaForCausalLM",
        "MistralLarge3ForCausalLM",
    }
)
"""Subclasses of ``DeepseekV2ForCausalLM``: they run ``DeepseekV2MoE`` and its dual-stream rules.

Their alt stream exists on CUDA and MUSA, and on ROCm/NPU with ``SGLANG_ROCM_USE_MULTI_STREAM`` /
``SGLANG_NPU_USE_MULTI_STREAM``; every other dual-stream model file creates it on CUDA only.
"""

NPU_SHARED_EXPERT_STREAM_ARCHITECTURES = frozenset({"Qwen2MoeForCausalLM", "Glm4MoeForCausalLM"})
"""Models with a separate NPU path that runs the shared expert on its own stream."""

TBO_PREFILL_ONLY_ARCHITECTURES = frozenset({"DeepseekV4ForCausalLM", "DeepseekV4ForCausalLMNextN"})
"""DeepSeek-V4 TBO raises ``NotImplementedError`` outside EXTEND; it is also the only strategy
with a ``moe_a2a_backend none`` branch (DP all_gatherv/reduce_scatterv overlapped). The other
strategies overlap EP dispatch/combine, which are no-ops unless ``ep_size > 1``."""

DEFAULT_DECODE_GRAPH_BACKEND = "full"
DEFAULT_PREFILL_GRAPH_BACKEND_CUDA = "breakable"

SOURCES: dict[str, str] = {
    "step_span": "python/sglang/srt/utils/profile_utils.py::build_step_span_name",
    "step_annotations": (
        "python/sglang/srt/model_executor/step_span_utils.py::build_detailed_annotation_suffix"
    ),
    "profile_request": "python/sglang/srt/managers/io_struct.py::ProfileReq",
    "server_log": "python/sglang/srt/managers/scheduler_components/metrics_reporter.py",
    "cuda_graph_defaults": "python/sglang/srt/model_executor/cuda_graph_config.py::CudaGraphConfig",
    "decode_graph_fallback": (
        "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py::can_run_graph"
    ),
    "fused_add_rmsnorm": "python/sglang/srt/layers/layernorm.py::RMSNorm.forward_cuda",
    "rmsnorm_native_fallback": (
        "python/sglang/srt/layers/layernorm.py::RMSNorm.forward_cuda (batch-invariant mode with "
        "a residual, or variance_size_override -> forward_native)"
    ),
    "batch_invariant_mode": (
        "python/sglang/srt/model_executor/model_runner.py::maybe_enable_batch_invariant_mode "
        "(--enable-deterministic-inference)"
    ),
    "npu_add_rmsnorm": "python/sglang/srt/layers/layernorm.py::RMSNorm.forward_npu",
    "layer_boundary": "python/sglang/srt/layers/layer_boundary/residual/add_norm.py",
    "silu_and_mul": "python/sglang/srt/layers/activation.py::SiluAndMul",
    "allreduce_fusion": (
        "python/sglang/srt/layers/layer_boundary/residual/add_norm.py::flashinfer_ar_fusion_applies"
    ),
    "allreduce_fusion_auto": (
        "python/sglang/srt/arg_groups/overrides.py::_flashinfer_allreduce_fusion_auto_enable"
    ),
    "allreduce_fusion_enforce_disable": (
        "python/sglang/srt/arg_groups/overrides.py::_enforce_disable_allreduce_fusion"
    ),
    "allreduce_fusion_boundary": (
        "python/sglang/srt/layers/layer_boundary/fusions/allreduce.py::fused_ffn_input"
    ),
    "allreduce_fusion_deterministic": (
        "python/sglang/srt/arg_groups/model_hook.py::handle_model_specific_adjustments "
        "(--enable-deterministic-inference -> enforce-disable)"
    ),
    "allreduce_fusion_unavailable": (
        "python/sglang/srt/layers/flashinfer_comm_fusion.py (allreduce_fusion API is not "
        "available -> standard implementation)"
    ),
    "allreduce_fusion_norm": (
        "python/sglang/srt/layers/layernorm.py::RMSNorm.forward_with_allreduce_fusion"
    ),
    "norm_quant_fusion": (
        "python/sglang/srt/layers/layer_boundary/residual/add_norm.py::_norm_quant_kernel"
    ),
    "norm_static_quant_fusion": "python/sglang/srt/layers/layernorm.py::RMSNorm.fuse_input_quant",
    "act_quant_fusion": (
        "python/sglang/kernels/ops/moe/ep_moe_kernels.py::silu_and_mul_masked_post_quant_fwd"
    ),
    "group_quant_fused_silu": (
        "python/sglang/kernels/ops/quantization/per_token_group_quant.py::per_token_group_quant "
        "(fuse_silu_and_mul=True)"
    ),
    "deep_gemm_masked_act_quant": (
        "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py (masked layout: "
        "per_token_group_quant(fuse_silu_and_mul=True))"
    ),
    "qk_norm": "python/sglang/srt/models/utils.py::apply_qk_norm",
    "fused_inplace_qknorm": "python/sglang/kernels/ops/layernorm/norm.py::can_use_fused_inplace_qknorm",
    "fused_qk_norm_rope_kernel": (
        "python/sglang/kernels/ops/attention/fused_qknorm_rope.py::can_use_fused_qk_norm_rope"
    ),
    "fused_qk_norm_rope": "python/sglang/srt/arg_groups/fields/exec_.py (--enable-fused-qk-norm-rope)",
    "fused_qk_norm_rope_model": "python/sglang/srt/models/qwen3_moe.py (use_fused_qk_norm_rope)",
    "fused_set_kv_buffer": "python/sglang/srt/models/utils.py::enable_fused_set_kv_buffer",
    "rope_fused_kv_platforms": (
        "python/sglang/srt/layers/rotary_embedding/base.py (native/NPU/CPU paths assert "
        "fused_set_kv_buffer_arg is None)"
    ),
    "mla_rope_cache_rocm": (
        "python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py"
        "::_fused_rope_cat_and_cache"
    ),
    "fused_moe_sum_allreduce": (
        "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py (use_fused_moe_sum_all_reduce)"
    ),
    "torch_compile_bs": (
        "python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py "
        "(compile_bs = capture_bs <= --torch-compile-max-bs)"
    ),
    "sbo_sm90": "python/sglang/srt/layers/moe/utils.py (sbo_enabled on SM90 -> ValueError)",
    "tbo_requires_dpa": "python/sglang/srt/arg_groups/validation_hook.py::check_two_batch_overlap",
    "tbo_strategies": (
        "python/sglang/srt/batch_overlap/operations_strategy.py::OperationsStrategy.init_new_tbo"
    ),
    "tbo_ep_dispatch": (
        "python/sglang/srt/models/{deepseek_v2,qwen3_moe,mimo_v2}.py::*MoE.op_dispatch_a "
        "(dispatch/combine only when ep_size > 1)"
    ),
    "tbo_deepseek_v4": (
        "python/sglang/srt/batch_overlap/operations_strategy.py"
        "::_compute_moe_deepseek_v4_layer_operations_strategy_tbo (EXTEND only)"
    ),
    "dual_stream_deepseek": "python/sglang/srt/models/deepseek_v2.py::DeepseekV2MoE.forward",
    "dual_stream_capture_mode": (
        "python/sglang/srt/model_executor/runner_utils/capture_mode.py::get_is_capture_mode "
        "(graph capture or breakable CUDA graph)"
    ),
    "dual_stream_deepseek_breakable": (
        "python/sglang/srt/models/deepseek_v2.py::DeepseekV2MoE._can_dual_stream_graph "
        "(SGLANG_ENABLE_PCG_DSV2_DUAL_STREAM, flashinfer_trtllm MoE runner)"
    ),
    "shared_experts_fusion": (
        "python/sglang/srt/models/deepseek_v2.py::DeepseekV2ForCausalLM.shared_experts_fusion_disable_reason"
    ),
    "shared_experts_fusion_glm4": (
        "python/sglang/srt/models/glm4_moe.py::Glm4MoeForCausalLM.shared_experts_fusion_disable_reason"
    ),
    "alt_stream_platforms": (
        "python/sglang/srt/models/deepseek_v2.py::DeepseekV2Model.__init__ (alt_stream); other "
        'dual-stream models: get_stream("alt") if _is_cuda else None'
    ),
    "multi_stream_env": (
        "python/sglang/srt/environ.py (SGLANG_ROCM_USE_MULTI_STREAM, SGLANG_NPU_USE_MULTI_STREAM)"
    ),
    "npu_shared_expert_stream": (
        "python/sglang/srt/models/qwen2_moe.py, glm4_moe.py (is_npu() and SGLANG_NPU_USE_MULTI_STREAM)"
    ),
    "custom_allreduce": (
        "python/sglang/srt/distributed/device_communicators/custom_all_reduce.py"
        "::CustomAllreduce.should_custom_ar"
    ),
    "allreduce_dispatch": "python/sglang/srt/distributed/parallel_state.py::GroupCoordinator.all_reduce",
    "layernorm_sp": "python/sglang/srt/layers/layernorm_sp.py (SP_SUPPORTED_ARCHITECTURES, EXTEND only)",
    "layernorm_sp_fused_matmul": (
        "python/sglang/srt/layers/layernorm_sp.py::sp_fused_matmul_eligible "
        "(torch symm_mem fused_matmul_reduce_scatter / fused_all_gather_matmul)"
    ),
    "platform_distributed_backend": (
        "python/sglang/srt/platforms/device_mixin.py::DeviceMixin.get_torch_distributed_backend_str"
    ),
    "platform_communicator_planned": (
        "python/sglang/srt/platforms/device_mixin.py::DeviceMixin.get_communicator_class "
        "([Planned], not called)"
    ),
    "l2_prefetch_helper": "python/sglang/kernels/aot/csrc/elementwise/utils.cuh::prefetch_L2",
}


def source(key: str) -> str:
    """``SGLang@<path>`` reference for a behaviour key (raises on unknown keys to catch typos)."""
    return f"SGLang: {SOURCES[key]}"
