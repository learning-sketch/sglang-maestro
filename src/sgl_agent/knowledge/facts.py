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
    "norm_quant_fusion": (
        "python/sglang/srt/layers/layer_boundary/residual/add_norm.py::_norm_quant_kernel"
    ),
    "norm_static_quant_fusion": "python/sglang/srt/layers/layernorm.py::RMSNorm.fuse_input_quant",
    "act_quant_fusion": (
        "python/sglang/kernels/ops/moe/ep_moe_kernels.py::silu_and_mul_masked_post_quant_fwd"
    ),
    "qk_norm": "python/sglang/srt/models/utils.py::apply_qk_norm",
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
    "dual_stream_deepseek": "python/sglang/srt/models/deepseek_v2.py::DeepseekV2MoE.forward",
    "shared_experts_fusion": (
        "python/sglang/srt/models/deepseek_v2.py::DeepseekV2ForCausalLM.shared_experts_fusion_disable_reason"
    ),
    "alt_stream_platforms": "python/sglang/srt/models/deepseek_v2.py::DeepseekV2Model.__init__ (alt_stream)",
}


def source(key: str) -> str:
    """``SGLang@<path>`` reference for a behaviour key (raises on unknown keys to catch typos)."""
    return f"SGLang: {SOURCES[key]}"
