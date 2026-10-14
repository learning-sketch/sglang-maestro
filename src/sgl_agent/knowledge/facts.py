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

DEVICE_TO_DISTRIBUTED_BACKEND = {
    "cuda": "nccl",
    "xpu": "xccl",
    "hpu": "hccl",
    "cpu": "gloo",
    "npu": "hccl",
    "musa": "mccl",
}
"""``DeviceMixin.get_torch_distributed_backend_str`` default: looked up by ``device_type``, any
other type gets ``gloo`` (NPU switches to ``zbal`` when ``SGLANG_ZBAL_LOCAL_MEM_SIZE > 0``)."""

IN_TREE_DEVICE_TYPES = {
    "cuda": "cuda",
    "rocm": "cuda",
    "npu": "npu",
    "xpu": "xpu",
    "musa": "musa",
    "cpu": "cpu",
    "mps": "mps",
}
"""``device_type`` of each in-tree platform (ROCm keeps ``torch.device("cuda")``)."""

IN_TREE_DISPATCH_KEYS = {
    "cuda": "cuda",
    "rocm": "hip",
    "npu": "npu",
    "xpu": "xpu",
    "musa": "musa",
    "cpu": "cpu",
}
"""Fused-op ``forward_<key>`` each in-tree platform selects; CPU only with AMX, MPS has none."""

IN_TREE_COMPILE_BACKENDS = {"mps": "eager", "npu": "torchair"}
"""``get_compiler_backend()`` on in-tree platforms that do not use ``inductor``."""

DEFAULT_OOT_DISPATCH_KEY = "native"
DEFAULT_OOT_COMPILE_BACKEND = "inductor"

MOE_NATIVE_ACTIVATIONS = ("silu", "gelu", "situ")
"""Activations ``moe_forward_native`` implements; anything else raises."""

DEEPSEEK_V3_MTP_DEFAULT_PARAMS = (3, 1, 4)
"""``(num_steps, eagle_topk, num_draft_tokens)`` SGLang picks for DeepSeek-V3 MTP when unset."""

SPECULATIVE_WIDE_TREE_ARCHITECTURES = frozenset(
    {"LlamaForCausalLM", "Grok1ForCausalLM", "Grok1VForCausalLM"}
)
SPECULATIVE_WIDE_TREE_PARAMS = (5, 4, 8)
"""``_auto_choose_speculative_params``: these architectures get (5, 4, 8), every other one and
STANDALONE get ``DEEPSEEK_V3_MTP_DEFAULT_PARAMS``."""

MTP_DRAFT_FROM_TARGET_ARCHITECTURES = frozenset(
    {
        "DeepseekV32ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV4ForCausalLM",
        "Glm4MoeForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "GlmMoeDsaForCausalLM",
        "BailingMoeForCausalLM",
        "BailingMoeV2ForCausalLM",
        "BailingMoeV2_5ForCausalLM",
        "MistralLarge3ForCausalLM",
        "PixtralForConditionalGeneration",
        "HYV3ForCausalLM",
        "HYV4ForCausalLM",
        "Qwen4ExpForConditionalGeneration",
    }
)
"""EAGLE-family drafts default to the target checkpoint (its MTP/NextN layers) for these."""

SPECULATIVE_DEFAULT_MAX_RUNNING_REQUESTS = 48
"""EAGLE/EAGLE3/NEXTN/STANDALONE, NGRAM and DFLASH reset ``--max-running-requests`` to 48 unless
it is given explicitly."""
NGRAM_DEVICES = ("cuda", "cpu", "xpu")
NGRAM_DEFAULT_DRAFT_TOKENS = 12
PAGED_TREE_SPECULATIVE_BACKENDS = ("flashinfer", "fa3", "triton")
"""``speculative_eagle_topk > 1`` with ``page_size > 1`` only runs on these attention backends."""

PREFILL_CP_INTERLEAVE_ONLY_ARCHITECTURES = frozenset({"DeepseekV32ForCausalLM"})
PREFILL_CP_ZIGZAG_ONLY_ARCHITECTURES = frozenset({"MiMoV2ForCausalLM", "MiMoV2FlashForCausalLM"})
PREFILL_CP_HIP_ARCHITECTURES = frozenset({"DeepseekV4ForCausalLM"})
NO_ATTN_CP_MODEL_TYPES = frozenset({"nemotron_h", "nemotron_h_puzzle"})
PREFILL_CP_STRATEGY_BACKENDS = {"zigzag": ("fa3", "fa4", "trtllm_mha"), "interleave": ("dsa",)}
"""Attention backends each CP strategy runs on (``get_supported_attention_backend``; fa3/fa4 are
``flashattention_backend``). No other backend calls into the CP strategy."""
PREFILL_CP_MIN_TOKENS_FACTOR = 2
"""zigzag ``can_apply``: a batch uses CP only if every extend length is at least 2 x cp."""
DCP_ATTENTION_BACKENDS = (
    "flashmla",
    "cutedsl_mla",
    "trtllm_mla",
    "tokenspeed_mla",
    "flashinfer",
    "triton",
    "dsa",
    "aiter",
    "intel_xpu",
)
"""Attention backends that read ``attn_dcp_size`` (fa3 does not); FlashMLA rejects DCP + FP8 KV."""
BREAKABLE_PREFILL_GRAPH_A2A_BACKENDS = ("none", "deepep", "megamoe", "flashinfer")
"""a2a backends the breakable prefill CUDA graph is validated with; DCP > 1, TBO and CP > 1
(outside the supported prefill-CP path) also turn the prefill graph off."""

DP_ATTENTION_CONSERVATIVENESS_FACTOR = 0.3
"""With attention DP, ``schedule_conservativeness`` is multiplied by this factor and
``--chunked-prefill-size`` is divided by the attention-DP size."""

CLIP_MAX_NEW_TOKENS = 4096
"""``SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`` default: admission reserves at most this many decode
tokens per request (new requests in full, running ones scaled by the new-token ratio)."""
INIT_NEW_TOKEN_RATIO = 0.7
MIN_NEW_TOKEN_RATIO_FACTOR = 0.14
NEW_TOKEN_RATIO_DECAY_STEPS = 600
RETRACT_DECODE_STEPS = 20
"""``NewTokenRatioTracker``: starts at 0.7 x schedule_conservativeness and decays linearly to
init x 0.14 over 600 decode steps. A retraction sets it to ``min(1, (decoded + 20 x running) /
(sum of max_new_tokens + 1))`` over the requests left running; an idle scheduler resets it."""

HICACHE_DEFAULT_RATIO = 2.0
HICACHE_BUFFER_ONLY_RATIO = 1.2
HICACHE_HOST_MEMORY_FRACTION = 0.8
HICACHE_HOST_RESERVE_BYTES = 10 * 1024**3
HICACHE_ALLOCATION_SLACK = 0.05
"""Auto-sizing: host pool = min(ratio, fraction x (host free - 10 GiB) / ranks per host x 0.95 /
device KV bytes) x device KV bytes; an explicit ratio or size turns auto-sizing off."""
HICACHE_SELECTIVE_WRITE_THRESHOLD = 2
"""``write_through`` backs a radix node up to host on its first insertion, ``write_through_selective``
only once it has been hit twice; ``write_back`` copies on eviction."""
HICACHE_TRANSFER_PLATFORMS = {"mha": ("cuda", "rocm", "npu"), "mla": ("cuda", "rocm", "xpu", "npu")}
"""Platforms whose host-pool transfer functions are imported (``sgl_kernel.kvcacheio`` on CUDA/HIP,
plus XPU for the MLA pool; ``sgl_kernel_npu.kvcacheio`` on NPU); elsewhere both the kernel and the
direct I/O paths reference undefined names."""

DEEPEP_LOW_LATENCY_TOKENS_DEFAULT = 128
DEEPEP_LOW_LATENCY_TOKENS_LIMIT = 1024
"""``SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`` default and the assert on it; it sizes the
low-latency RDMA buffer, so each rank's decode tokens (draft tokens included) must fit."""
TBO_TOKEN_DISTRIBUTION_THRESHOLD = 0.48
"""``--tbo-token-distribution-threshold``: an EXTEND batch whose sequence split leaves one half with
less than this share of tokens is split inside a sequence (two-chunk overlap); 0 disables that."""

DISAGGREGATION_BOOTSTRAP_PORT = 8998
"""``--disaggregation-bootstrap-port`` default; the router's ``--prefill URL PORT`` must match."""
PD_RESERVED_DECODE_TOKENS = 512
"""``--num-reserved-decode-tokens``: a PD decode server admits a transferred request only while
``free KV - 512 x active requests`` still holds its prompt plus 512 tokens."""
PD_DECODE_EXTRA_SLOTS_MAX_BATCH = 32
"""Unset ``--disaggregation-decode-extra-slots``: a decode worker whose ``--max-running-requests``
share is at most 32 gets twice that many extra request slots for in-transfer KV, otherwise none;
the running batch itself never exceeds ``max_running_requests``."""
PD_TRANSFER_BACKENDS = ("mooncake", "nixl", "ascend", "mori", "mooncake_tcp")
"""Real ``--disaggregation-transfer-backend`` choices (``fake`` moves no KV); ``mooncake`` is the
default and ``mooncake_tcp`` forces the TCP transport (``MC_FORCE_TCP=1``, no RDMA)."""
PD_DECODE_DCP_TRANSFER_BACKENDS = ("mooncake", "nixl", "mori", "ascend")
"""Transfer backends that relayout KV for a DCP decode server (``fake`` aside)."""

DEFAULT_MAX_PREFILL_TOKENS = 16384
"""``--max-prefill-tokens``: extend tokens one prefill batch may carry (the chunk size caps it)."""
MIXED_CHUNK_SPECULATIVE_ALGORITHMS = ("EAGLE", "EAGLE3", "DFLASH", "DSPARK")
"""``--enable-mixed-chunk`` (needs chunked prefill) runs the decode batch inside each prefill
batch: chunk and input budgets shrink by the running batch size and every running request
advances one token per mixed step. These speculative algorithms keep it (running requests fall
back to a plain one-token decode in a mixed step); others turn it off."""
MAX_REQ_INPUT_MARGIN = 5
"""``max_req_len = min(context_len - 1, KV tokens - 1)``; prompts of ``max_req_len - 5`` tokens or
more are aborted and ``max_new_tokens`` is clamped to ``max_req_len - input - 1``."""
MAX_RUNNING_REQUESTS_FLOOR = 2048
MAX_RUNNING_REQUESTS_CEIL = 4096
"""Unset ``--max-running-requests``: per DP worker ``min(clamp(KV tokens / context_len x 512,
2048, 4096), KV tokens // 2)``; set: ``min(value // attn_dp, KV tokens // 2)``."""

ZMQ_TCP_PORT_DELTA = 233
DP_ATTENTION_HANDSHAKE_PORT_DELTA = 13
"""Ports an attention-DP server binds besides ``--port``: the worker-port handshake at ``--port + 13``
and, on one node without ``--dist-init-addr``, dist init at ``--port + 233`` plus six ZMQ ports
up to ``--port + 239`` (with ``--dist-init-addr`` they follow its port: ``+1 .. +6`` and ``+13``).
Instances sharing a machine need ``--port`` and ``--dist-init-addr`` values whose ranges stay apart."""
ROUTER_DEFAULT_PORT = 30000
ROUTER_DEFAULT_POLICY = "cache_aware"
ROUTER_CACHE_THRESHOLD = 0.3
ROUTER_BALANCE_ABS_THRESHOLD = 64
ROUTER_BALANCE_REL_THRESHOLD = 1.5
"""``cache_aware`` routes to the worker with the longest prefix match when the match ratio exceeds
0.3, else to the least loaded one; it balances by load once ``max - min > 64`` and
``max > 1.5 x min``."""

PLATFORM_PLUGIN_GROUP = "sglang.srt.platforms"
GENERAL_PLUGIN_GROUP = "sglang.srt.plugins"

APPLY_ROTARY_POS_EMB_MODEL_FILES = frozenset({"gemma3_causal", "glm_image_vl"})
"""Model files that call the compiled ``apply_rotary_pos_emb`` directly (besides
``FourierRotaryEmbedding`` and the ``normal`` vision RoPE)."""

LAUNCH_SERVER_MODULE = "sglang.launch_server"
BENCH_SERVING_MODULE = "sglang.benchmark.serving"
ROUTER_MODULE = "sglang_router.launch_router"
RUN_EVAL_MODULE = "sglang.test.run_eval"
MODEL_TRUNCATOR_MODULE = "sglang.srt.debug_utils.model_truncator"
DUMP_COMPARATOR_MODULE = "sglang.srt.debug_utils.dump_comparator"

TOOL_FLAGS: dict[str, frozenset[str]] = {
    BENCH_SERVING_MODULE: frozenset(
        {
            "--backend",
            "--base-url",
            "--host",
            "--port",
            "--ready-check-timeout-sec",
            "--dataset-name",
            "--dataset-path",
            "--dataset-offset",
            "--model",
            "--served-model-name",
            "--tokenizer",
            "--num-prompts",
            "--sharegpt-output-len",
            "--sharegpt-context-len",
            "--random-input-len",
            "--random-output-len",
            "--random-range-ratio",
            "--request-rate",
            "--use-trace-timestamps",
            "--max-concurrency",
            "--output-file",
            "--output-details",
            "--disable-tqdm",
            "--disable-stream",
            "--seed",
            "--disable-ignore-eos",
            "--temperature",
            "--top-p",
            "--extra-request-body",
            "--apply-chat-template",
            "--profile",
            "--profile-activities",
            "--profile-num-steps",
            "--profile-output-dir",
            "--pd-separated",
            "--profile-prefill-url",
            "--profile-decode-url",
            "--flush-cache",
            "--warmup-requests",
            "--tokenize-prompt",
            "--gsp-num-groups",
            "--gsp-prompts-per-group",
            "--gsp-system-prompt-len",
            "--gsp-question-len",
            "--gsp-output-len",
            "--gsp-range-ratio",
            "--gsp-num-turns",
            "--gsp-ordered",
            "--gsp-group-distribution",
            "--gsp-zipf-alpha",
            "--gsp-send-routing-key",
            "--gsp-fast-prepare",
            "--tag",
            "--header",
        }
    ),
    ROUTER_MODULE: frozenset(
        {
            "--host",
            "--port",
            "--worker-urls",
            "--policy",
            "--prefill-policy",
            "--decode-policy",
            "--cache-threshold",
            "--balance-abs-threshold",
            "--balance-rel-threshold",
            "--dp-aware",
            "--pd-disaggregation",
            "--prefill",
            "--decode",
            "--worker-startup-timeout-secs",
            "--model-path",
            "--tokenizer-path",
            "--max-concurrent-requests",
            "--queue-size",
            "--queue-timeout-secs",
            "--request-timeout-secs",
            "--prometheus-host",
            "--prometheus-port",
            "--log-level",
            "--health-check-interval-secs",
            "--disable-retries",
            "--disable-circuit-breaker",
        }
    ),
    RUN_EVAL_MODULE: frozenset(
        {
            "--base-url",
            "--host",
            "--port",
            "--model",
            "--repeat",
            "--eval-name",
            "--api",
            "--num-examples",
            "--num-threads",
            "--max-tokens",
            "--temperature",
            "--top-p",
            "--top-k",
            "--min-p",
            "--chat-template-kwargs",
            "--reasoning-effort",
            "--thinking-mode",
            "--num-shots",
            "--gsm8k-data-path",
        }
    ),
    MODEL_TRUNCATOR_MODULE: frozenset({"--input", "--output", "--keep-num-layers"}),
    DUMP_COMPARATOR_MODULE: frozenset(
        {
            "--baseline-path",
            "--target-path",
            "--start-step",
            "--end-step",
            "--diff-threshold",
            "--filter",
        }
    ),
}
"""argparse flags of the SGLang tools sgl-agent renders commands for (a subset for the router
and benchmark: the ones the generated commands may use)."""

TOOL_CHOICES: dict[tuple[str, str], tuple[str, ...]] = {
    (BENCH_SERVING_MODULE, "--backend"): (
        "sglang",
        "sglang-native",
        "sglang-oai",
        "sglang-oai-chat",
        "sglang-embedding",
        "vllm",
        "vllm-chat",
        "vllm-embedding",
        "lmdeploy",
        "lmdeploy-chat",
        "trt",
        "gserver",
        "truss",
    ),
    (BENCH_SERVING_MODULE, "--dataset-name"): (
        "agentic-trace",
        "sharegpt",
        "custom",
        "openai",
        "random",
        "random-ids",
        "generated-shared-prefix",
        "mmmu",
        "image",
        "mooncake",
        "longbench_v2",
        "speed-bench",
    ),
    (ROUTER_MODULE, "--policy"): (
        "random",
        "round_robin",
        "cache_aware",
        "power_of_two",
        "bucket",
        "manual",
        "consistent_hashing",
        "prefix_hash",
    ),
    (BENCH_SERVING_MODULE, "--gsp-group-distribution"): ("uniform", "zipf"),
    (RUN_EVAL_MODULE, "--api"): ("chat", "completion", "generate"),
}
TOOL_CHOICES[(ROUTER_MODULE, "--prefill-policy")] = TOOL_CHOICES[(ROUTER_MODULE, "--policy")]
TOOL_CHOICES[(ROUTER_MODULE, "--decode-policy")] = TOOL_CHOICES[(ROUTER_MODULE, "--policy")]

SOURCES: dict[str, str] = {
    "step_span": "python/sglang/srt/utils/profile_utils.py::build_step_span_name",
    "step_annotations": (
        "python/sglang/srt/model_executor/step_span_utils.py::build_detailed_annotation_suffix"
    ),
    "profile_request": "python/sglang/srt/managers/io_struct.py::ProfileReq",
    "profiler_output": (
        "python/sglang/srt/managers/scheduler_components/profiler_manager.py "
        "(SGLANG_TORCH_PROFILER_DIR, {profile_id}-TP-{rank}[-DP-..][-PP-..][-EP-..].trace.json.gz)"
    ),
    "oot_profiler_activity": (
        "python/sglang/srt/managers/scheduler_components/profiler_manager.py (out-of-tree platforms: "
        "current_platform.get_torch_profiler_activity_str(); the DeviceMixin default raises "
        "NotImplementedError)"
    ),
    "npu_profiler_export": (
        "python/sglang/srt/managers/scheduler_components/profiler_manager.py (NPU: "
        "ProfilerActivity.CUDA patched to torch_npu NPU; tensorboard_trace_handler instead of "
        "export_chrome_trace)"
    ),
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
    "moe_config_dir": (
        "python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py"
        "::get_moe_configs (SGLANG_MOE_CONFIG_DIR, configs/triton_<version>/)"
    ),
    "kv_pool_stats": (
        "python/sglang/srt/managers/scheduler_components/pool_stats_observer.py::_get_token_info "
        "(#token = max_total_num_tokens - available - evictable)"
    ),
    "retract": "python/sglang/srt/managers/scheduler.py (KV cache pool is full. Retract requests.)",
    "platform_plugin_discovery": (
        "python/sglang/srt/platforms/__init__.py::_resolve_platform (entry point group "
        "sglang.srt.platforms; SGLANG_PLATFORM selects one plugin, otherwise plugin errors are "
        "only logged and the base platform is used)"
    ),
    "general_plugin_hooks": (
        "python/sglang/srt/plugins/hook_registry.py::HookRegistry.register (entry point group "
        "sglang.srt.plugins; BEFORE/AFTER/AROUND/REPLACE)"
    ),
    "platform_interface": (
        "python/sglang/srt/platforms/interface.py::SRTPlatform, "
        "python/sglang/srt/platforms/device_mixin.py::DeviceMixin"
    ),
    "oot_device_fallback": (
        "python/sglang/srt/utils/common.py::get_device (falls back to current_platform.get_device)"
    ),
    "oot_memory_hooks": (
        "python/sglang/srt/utils/common.py::get_available_gpu_memory (out-of-tree: "
        "get_device_total_memory - get_current_memory_usage)"
    ),
    "pinned_h2d": "python/sglang/srt/utils/common.py::async_h2d (pinned staging for H2D copies)",
    "distributed_backend_fallback": (
        "python/sglang/srt/distributed/parallel_state.py::get_default_distributed_backend, "
        "python/sglang/srt/platforms/device_mixin.py::_DEVICE_TO_DISTRIBUTED_BACKEND"
    ),
    "oot_default_attention_backend": (
        "python/sglang/srt/arg_groups/model_override_base.py::get_default_attn_backend"
    ),
    "attention_registry": (
        "python/sglang/srt/layers/attention/attention_registry.py::register_attention_backend"
    ),
    "mla_backend_handler": (
        "python/sglang/srt/models/deepseek_common/attention_backend_handler.py"
        "::AttentionBackendRegistry (unknown backend names use the triton handler)"
    ),
    "graph_capture_capability": (
        "python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py "
        "(decode graphs are captured only when capabilities.graph_capture)"
    ),
    "graph_runner_required": (
        "python/sglang/srt/model_executor/model_runner_components/cuda_graph_setup.py "
        "(require_out_of_tree_impl(hook='get_graph_runner_cls()'))"
    ),
    "full_graph_backend": (
        "python/sglang/srt/model_executor/runner_backend/utils.py (out-of-tree: "
        "get_full_graph_backend_cls(), else FullCudaGraphBackend with torch.cuda.CUDAGraph)"
    ),
    "oot_guards": (
        "python/sglang/srt/platforms/interface.py::require_out_of_tree_impl, reject_out_of_tree_path"
    ),
    "fused_op_dispatch": (
        "python/sglang/kernels/fused_op.py::BaseFusedOp (register_oot_forward on the exact class, "
        "then forward_<dispatch key>, then forward_native)"
    ),
    "fused_op_force_backend": (
        "python/sglang/kernels/fused_op.py::get_fused_op_backend (SGLANG_FORCE_FUSED_OP_BACKEND)"
    ),
    "oot_compile_backend": (
        "python/sglang/srt/utils/common.py::get_compiler_backend (out-of-tree: "
        "current_platform.get_compile_backend())"
    ),
    "oot_init_backend": (
        "python/sglang/srt/model_executor/model_runner.py (module level: "
        "current_platform.init_backend() on out-of-tree platforms)"
    ),
    "oot_quantization_hook": (
        "python/sglang/srt/layers/quantization/__init__.py::get_quantization_config "
        "(out-of-tree: current_platform.get_quantization_config(name))"
    ),
    "kv_pool_hook": (
        "python/sglang/srt/mem_cache/kv_cache_configurator.py::_resolve_kv_pool_class "
        "(current_platform.get_kv_pool_cls(kind=mha|mla|dsa))"
    ),
    "paged_allocator_triton": (
        "python/sglang/srt/mem_cache/allocator/paged.py (use_triton_kernels = "
        "capabilities.supports_triton)"
    ),
    "speculative_cache_locs": (
        "python/sglang/kernels/ops/speculative/cache_locs.py::assign_extend_cache_locs_func "
        "(out-of-tree: get_speculative_cache_locs_fn(), else Triton)"
    ),
    "dflash_platform_gate": (
        "python/sglang/srt/arg_groups/speculative_hook.py::_handle_dflash "
        "(supports_speculative_algorithm is consulted for DFLASH only)"
    ),
    "speculative_auto_params": (
        "python/sglang/srt/arg_groups/speculative_hook.py::_auto_choose_speculative_params"
    ),
    "speculative_nextn_alias": (
        "python/sglang/srt/arg_groups/speculative_hook.py (NEXTN resolves to EAGLE; DeepSeek-V3 "
        "drafts default to the target checkpoint's MTP layers)"
    ),
    "torch_compile_sites": (
        "python/sglang/srt/layers/vocab_parallel_embedding.py::get_masked_input_and_mask, "
        "python/sglang/srt/layers/moe/topk.py (@torch.compile(backend=get_compiler_backend()))"
    ),
    "oot_grouped_topk": (
        "python/sglang/srt/layers/moe/topk.py::biased_grouped_topk_gpu (non-CUDA/HIP: "
        "biased_grouped_topk_impl under torch.compile)"
    ),
    "oot_softmax_topk": "python/sglang/srt/layers/moe/topk.py::fused_topk_torch_native",
    "moe_native_fallback": (
        "python/sglang/srt/layers/quantization/unquant.py::UnquantizedFusedMoEMethod, "
        "python/sglang/srt/layers/moe/fused_moe_native.py::moe_forward_native (per-expert loop, "
        "tokens_per_expert.cpu() host sync)"
    ),
    "moe_triton_fused_func": (
        "python/sglang/srt/layers/moe/moe_runner/triton.py (register_fused_func('none', 'triton') "
        "bypasses register_moe_runner_core)"
    ),
    "w8a8_int8_cuda_only": (
        "python/sglang/srt/layers/quantization/w8a8_int8.py::W8A8Int8LinearMethod "
        "(int8_scaled_mm imported on CUDA only, per_token_quant_int8 is Triton)"
    ),
    "unquantized_linear": (
        "python/sglang/srt/layers/quantization/unquant.py::UnquantizedLinearMethod (F.linear)"
    ),
    "fp8_to_bf16_precedent": (
        "benchmark/deepseek_v3/README.md (A100 without FP8: fp8_cast_bf16.py, then 4x8 GPUs TP32)"
    ),
    "bf16_remove_quant_config": (
        "benchmark/deepseek_v3/README.md (Troubleshooting: a converted BF16 checkpoint must drop "
        "quantization_config, else 'output_partition_size = 576 is not divisible by block_n = 128')"
    ),
    "int8_checkpoint_precedent": (
        "benchmark/deepseek_v3/README.md (meituan/DeepSeek-R1-Channel-INT8 --quantization w8a8_int8)"
    ),
    "rope_cache_dtype": (
        "python/sglang/srt/layers/rotary_embedding/base.py::RotaryEmbedding.__init__ (cos/sin "
        "cache cast to the model dtype off CUDA/HIP/XPU unless SGLANG_ROPE_CACHE_FP32=1)"
    ),
    "rope_factory": "python/sglang/srt/layers/rotary_embedding/factory.py::get_rope",
    "deepseek_yarn_rewrite": (
        "python/sglang/srt/models/deepseek_v2.py (rope_scaling['rope_type'] = 'deepseek_yarn')"
    ),
    "external_model_package": (
        "python/sglang/srt/models/registry.py (SGLANG_EXTERNAL_MODEL_PACKAGE: every module of the "
        "package that defines EntryClass registers it under the class __name__, which must equal "
        "config.json architectures[0]; the in-tree models are then loaded lazily)"
    ),
    "fused_op_resolution_cache": (
        "python/sglang/kernels/fused_op.py::BaseFusedOp.forward (_forward_method is resolved on "
        "the first call and cached per instance; registrations made later do not reach it)"
    ),
    "fused_op_trace": (
        "python/sglang/kernels/fused_op.py::enable_fused_op_trace / get_fused_op_trace (backend "
        "label: 'torch' for forward_native, the key for forward_<key>, else the function name)"
    ),
    "server_ready_log": (
        "python/sglang/srt/entrypoints/http_server.py (The server is fired up and ready to roll!)"
    ),
    "model_truncator_layers": (
        "python/sglang/srt/debug_utils/model_truncator.py (keeps model.layers.<i> for i < "
        "--keep-num-layers and rewrites num_hidden_layers)"
    ),
    "platform_resolution": (
        "python/sglang/srt/platforms/__init__.py (current_platform is resolved lazily on first "
        "access)"
    ),
    "dumper_non_intrusive": (
        "python/sglang/srt/model_executor/model_runner.py (dumper.register_non_intrusive_dumper "
        "hooks module outputs when DUMPER_ENABLE=1; DUMPER_DIR defaults to /tmp/dumper)"
    ),
    "transformers_fallback": (
        "python/sglang/srt/model_loader/utils.py::get_model_architecture (--model-impl auto: "
        "resolve_transformers_arch when no SGLang implementation exists)"
    ),
    "dumper": "python/sglang/srt/debug_utils/dumper.py (DUMPER_ENABLE=1, DUMPER_DIR)",
    "dump_comparator": "python/sglang/srt/debug_utils/dump_comparator.py",
    "model_truncator": "python/sglang/srt/debug_utils/model_truncator.py",
    "run_eval": (
        "python/sglang/test/run_eval.py (deprecated but still scores gsm8k/mgsm_en/humaneval; "
        "mmlu/gpqa/aime go through sgl-eval: python/sglang/test/sgl_eval_utils.py::run_sgl_eval)"
    ),
    "rmsnorm_fp32": (
        "python/sglang/srt/layers/layernorm.py::RMSNorm.forward_native (x.to(torch.float32) "
        "before the residual add and the variance)"
    ),
    "mla_mscale": (
        "python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA.__init__ "
        "(self.scaling = compute_mla_mscale_scaling(rope_scaling, scaling))"
    ),
    "topk_output_dtypes": (
        "python/sglang/srt/layers/moe/topk.py::fused_topk_torch_native (topk_weights float32, "
        "topk_ids int32)"
    ),
    "hicache_transfer_kernels": (
        "python/sglang/srt/mem_cache/pool_host/mha.py, mla.py, memory_pool_host.py "
        "(transfer kernels imported from sgl_kernel.kvcacheio on CUDA/HIP[/XPU], "
        "sgl_kernel_npu on NPU)"
    ),
    "hicache_layout_io": (
        "python/sglang/srt/arg_groups/hicache_hook.py::resolve_layout_io_compatibility"
    ),
    "hicache_dcp": "python/sglang/srt/arg_groups/hicache_hook.py::resolve_hicache_dcp_compatibility",
    "layernorm_sp_rules": "python/sglang/srt/arg_groups/layernorm_sp_hook.py::validate_layernorm_sp",
    "dcp_rules": "python/sglang/srt/arg_groups/parallel_hook.py::handle_decode_context_parallelism",
    "attn_cp_rules": "python/sglang/srt/arg_groups/parallel_hook.py (attn_cp_size divisibility)",
    "dwdp_rules": "python/sglang/srt/arg_groups/parallel_hook.py (dwdp_size == tp_size)",
    "dp_attention": "python/sglang/srt/arg_groups/fields/parallel.py (--attn-dp-size)",
    "deepep_low_latency_tokens": (
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK (python/sglang/srt/environ.py)"
    ),
    "benchmark_serving": "python/sglang/benchmark/serving.py",
    "pd_router": (
        "sgl-model-gateway/bindings/python/src/sglang_router/router_args.py "
        "(--pd-disaggregation, --prefill URL [BOOTSTRAP_PORT], --decode URL, --policy)"
    ),
    "launch_server": "python/sglang/launch_server.py",
    "kv_pool_budget": (
        "python/sglang/srt/mem_cache/kv_cache_configurator.py::_profile_available_bytes "
        "(free memory - pre_model_load_memory * (1 - mem_fraction_static))"
    ),
    "attention_backend_choices": (
        "python/sglang/srt/arg_groups/choices.py (add_attention_backend_choices, "
        "add_quantization_method_choices, add_moe_runner_backend_choices, ...; extend before "
        "ServerArgs is constructed)"
    ),
    "general_plugin_load_order": (
        "python/sglang/launch_server.py (load_plugins() before prepare_server_args); "
        "scheduler.py, data_parallel_controller.py, detokenizer_manager.py reload them"
    ),
    "server_device_default": (
        "python/sglang/srt/arg_groups/serving_hook.py (device=get_device(), then device.split(':'))"
    ),
    "sampler_backend": (
        "python/sglang/srt/layers/sampler.py::register_sampler_backend, "
        "python/sglang/srt/arg_groups/overrides.py::_sampling_backend_default "
        "(flashinfer when available, else pytorch)"
    ),
    "moe_runner_core": (
        "python/sglang/srt/layers/moe/moe_runner/runner.py::register_moe_runner_core"
    ),
    "pinned_batch_copies": (
        "python/sglang/srt/managers/schedule_batch.py, "
        "python/sglang/srt/model_executor/forward_batch_info.py "
        "(pin_memory=is_pin_memory_available(device))"
    ),
    "kv_store_fallback": (
        "python/sglang/srt/mem_cache/memory_pool.py::_set_kv_buffer_impl (store_cache on "
        "CUDA/HIP, XPU and AMX CPU; otherwise k_cache[loc] = k; v_cache[loc] = v)"
    ),
    "prefill_graph_default": (
        "python/sglang/srt/model_executor/cuda_graph_config.py (prefill backend: breakable on "
        "CUDA, disabled elsewhere)"
    ),
    "oot_forward_override": (
        "python/sglang/kernels/fused_op.py::BaseFusedOp.forward (subclasses that override "
        "forward() without calling dispatch bypass register_oot_forward)"
    ),
    "oot_dispatch_order": (
        "python/sglang/kernels/fused_op.py::BaseFusedOp._resolve_forward_method (out-of-tree: "
        "registry[type(self)] -> forward_<key> found through the MRO -> forward_native; the "
        "triton/jit/aot/flashinfer backends are never auto-selected)"
    ),
    "rmsnorm_contract": (
        "python/sglang/srt/layers/layernorm.py::RMSNorm.forward_native(x, residual, "
        "post_residual_addition, quant_linear) -> out | (out, residual)"
    ),
    "add_norm_plain": (
        "python/sglang/srt/layers/layer_boundary/residual/add_norm.py"
        "::_update_and_read_residual_plain (norm(hidden_states, residual, post_residual_addition))"
    ),
    "rope_contract": (
        "python/sglang/srt/layers/rotary_embedding/base.py::RotaryEmbedding.forward_native"
        "(positions, query, key, offsets, fused_set_kv_buffer_arg) -> (query, key)"
    ),
    "deepseek_rope_interleave": (
        "python/sglang/srt/models/deepseek_v2.py::DeepseekV2AttentionMLA.__init__ "
        "(is_neox_style = not config.rope_interleave, default True: GPT-J interleaved)"
    ),
    "mla_kv_buffer": (
        "python/sglang/kernels/ops/kvcache/mla_buffer.py (capabilities.supports_triton False -> "
        "set_mla_kv_buffer_naive)"
    ),
    "compute_position": (
        "python/sglang/srt/model_executor/forward_batch_info.py::compute_position "
        "(support_triton() False -> compute_position_torch: Python loop over device tensors)"
    ),
    "topk_native_asserts": (
        "python/sglang/srt/layers/moe/topk.py::select_experts (torch_native non-grouped path: "
        "fused_topk_native asserts num_token_non_padded and expert_location_dispatch_info are None)"
    ),
    "num_token_non_padded": (
        "python/sglang/srt/model_executor/forward_batch_info.py::enable_num_token_non_padded "
        "(ep_size > 1 or a customized a2a backend), ForwardBatch.moe_num_token_non_padded "
        "(None only with attention DP > 1 and a2a none)"
    ),
    "prefill_cp_rules": (
        "python/sglang/srt/arg_groups/parallel_hook.py::handle_context_parallelism "
        "(--cp-strategy required, per-model strategy limits, tp % (attn_dp x attn_cp) == 0)"
    ),
    "prefill_cp_platform": (
        "python/sglang/srt/arg_groups/parallel_hook.py::validate_prefill_cp_platform"
    ),
    "cp_strategies": (
        "python/sglang/srt/layers/cp/zigzag.py (2 x cp blocks, rank i owns i and 2cp-1-i), "
        "python/sglang/srt/layers/cp/interleave.py (rank i owns every cp-th token)"
    ),
    "dp_attention_resolution": (
        "python/sglang/srt/arg_groups/parallel_hook.py::handle_data_parallelism "
        "(schedule_conservativeness x 0.3, chunked_prefill_size // attn_dp_size, "
        "--dp-size with --attn-dp-size raises)"
    ),
    "dp_attention_deprecated": (
        "python/sglang/srt/arg_groups/parallel_hook.py::handle_deprecated_dp_attention "
        "(--enable-dp-attention --dp-size N -> --attn-dp-size N)"
    ),
    "dp_lm_head": "python/sglang/srt/arg_groups/overrides.py::_dp_lm_head_validation",
    "dp_attention_collectives": (
        "python/sglang/srt/layers/dp_attention.py (_dp_gather_via_all_gather, "
        "_dp_gather_via_all_reduce)"
    ),
    "dcp_comm_backend_default": (
        "python/sglang/srt/arg_groups/overrides.py::_dcp_comm_backend_default (fi_a2a on a "
        "supported MNNVL domain, a2a on CUDA/HIP/XPU, otherwise ag_rs)"
    ),
    "a2a_ep_size": (
        "python/sglang/srt/arg_groups/overrides.py::_a2a_ep_size (a2a backends set ep_size = tp_size)"
    ),
    "speculative_max_running": (
        "python/sglang/srt/arg_groups/speculative_hook.py::_handle_eagle_family, _handle_ngram, "
        "_handle_dflash (max_running_requests defaults to 48)"
    ),
    "speculative_eagle_rules": (
        "python/sglang/srt/arg_groups/speculative_hook.py::_handle_eagle_family (STANDALONE "
        "rejects attention DP, trtllm_mha needs topk 1, paged topk > 1 needs flashinfer/fa3/"
        "triton, mixed chunk off)"
    ),
    "speculative_ngram": "python/sglang/srt/arg_groups/speculative_hook.py::_handle_ngram",
    "deepep_normal_graph": (
        "python/sglang/srt/arg_groups/moe_hook.py::handle_a2a_moe (deepep_mode normal disables "
        "the decode and prefill graphs)"
    ),
    "deepep_auto_mode": (
        "python/sglang/srt/layers/moe/utils.py::DeepEPMode.resolve (auto: normal for batches "
        "with EXTEND, low_latency otherwise)"
    ),
    "deepep_v2_overlap": (
        "python/sglang/srt/arg_groups/moe_hook.py::handle_a2a_moe (deepep_v2 rejects TBO/SBO)"
    ),
    "a2a_dispatcher_factory": (
        "python/sglang/srt/layers/moe/fused_moe_triton/layer.py::create_moe_dispatcher "
        "(deepep/mooncake/mori/nixl/pplx -> MaybeTboDeepEPDispatcher, else NotImplementedError)"
    ),
    "deepep_buffer_import": (
        "python/sglang/srt/layers/moe/token_dispatcher/deepep.py (from deep_ep import Buffer, "
        "Config; two-phase dispatch_a/dispatch_b/combine_a/combine_b)"
    ),
    "hicache_auto_size": (
        "python/sglang/srt/mem_cache/hicache_auto_size.py::auto_size_hicache, "
        "python/sglang/srt/mem_cache/pool_host/base.py::host_memory_budget_bytes"
    ),
    "hicache_ratio_default": (
        "python/sglang/srt/arg_groups/hicache_hook.py::handle_hicache_ratio_default"
    ),
    "hicache_buffer_only": (
        "python/sglang/srt/arg_groups/hicache_hook.py::validate_hicache_host_memory_mode"
    ),
    "hicache_host_pool_io": (
        "python/sglang/srt/mem_cache/pool_host/mha.py::MHATokenToKVPoolHost, "
        "python/sglang/srt/mem_cache/pool_host/mla.py::MLATokenToKVPoolHost "
        "(load_to_device_per_layer, backup_from_device_all_layer)"
    ),
    "breakable_prefill_graph": (
        "python/sglang/srt/arg_groups/cuda_graph_hook.py::disable_breakable_cudagraph_if_incompatible"
    ),
    "pd_disaggregation_rules": (
        "python/sglang/srt/arg_groups/pd_disaggregation_hook.py::handle_pd_disaggregation "
        "(decode servers disable the radix cache by default)"
    ),
    "admission_reserve": (
        "python/sglang/srt/managers/schedule_policy.py::PrefillAdder (new requests reserve "
        "extend + min(max_new_tokens, CLIP_MAX_NEW_TOKENS); running ones x new_token_ratio)"
    ),
    "new_token_ratio": (
        "python/sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py"
        "::NewTokenRatioTracker"
    ),
    "prefill_log_throughput": (
        "python/sglang/srt/managers/scheduler_components/metrics_reporter.py::report_prefill_stats "
        "(input throughput = new tokens / time since the previous prefill log)"
    ),
    "bench_random_lengths": (
        "python/sglang/benchmark/datasets/common.py::compute_random_lens (uniform integers in "
        "[max(int(L x range_ratio), 1), L])"
    ),
    "bench_gsp": "python/sglang/benchmark/datasets/generated_shared_prefix.py",
    "bench_metrics": (
        "python/sglang/benchmark/serving.py (mean/median/p90/p95/p99 TTFT, TPOT, ITL, E2E)"
    ),
    "expert_distribution": (
        "python/sglang/srt/eplb/expert_distribution.py (--expert-distribution-recorder-mode)"
    ),
    "prefill_cp_backends": (
        "python/sglang/srt/layers/cp/zigzag.py, interleave.py::get_supported_attention_backend "
        "(CP attention in flashattention_backend.py, trtllm_mha_backend.py, dsa_backend.py)"
    ),
    "prefill_cp_can_apply": (
        "python/sglang/srt/layers/cp/zigzag.py::can_apply (every extend length >= 2 x cp, "
        "EXTEND/MIXED only)"
    ),
    "dcp_kv_heads": (
        "python/sglang/srt/configs/model_config.py::get_num_kv_heads (heads shard over tp // dcp), "
        "python/sglang/srt/arg_groups/attention_hook.py::_check_xpu_dcp (every rank of a DCP "
        "group must hold the same KV heads)"
    ),
    "dcp_attention_backends": (
        "python/sglang/srt/layers/attention/{flashmla,cutedsl_mla,trtllm_mla,tokenspeed_mla,"
        "flashinfer,flashinfer_mla,triton,dsa,aiter,xpu}_backend.py (attn_dcp_size)"
    ),
    "load_balance_default": (
        "python/sglang/srt/arg_groups/serving_hook.py::handle_load_balance_method (auto: "
        "round_robin, follow_bootstrap_room on PD prefill)"
    ),
    "hicache_write_policy": (
        "python/sglang/srt/mem_cache/unified_radix_cache.py (write_through_threshold: 1 for "
        "write_through, 2 otherwise), python/sglang/srt/managers/cache_controller.py"
    ),
    "hicache_oot_transfer": (
        "python/sglang/srt/mem_cache/pool_host/mha.py, mla.py (transfer_kv_* imported only under "
        "is_cuda()/is_hip() (mla.py also is_xpu()) from sgl_kernel.kvcacheio and is_npu() from "
        "sgl_kernel_npu)"
    ),
    "hicache_host_alloc": (
        "python/sglang/srt/mem_cache/pool_host/common.py::ALLOC_MEMORY_FUNCS (npu/musa/xpu -> "
        "alloc_with_pin_memory; any other device type -> alloc_with_host_register, which calls "
        "torch.cuda.cudart().cudaHostRegister)"
    ),
    "hicache_layout_rewrite": (
        "python/sglang/srt/arg_groups/hicache_hook.py::resolve_layout_io_compatibility (page_first "
        "+ direct -> page_first_direct; page_first_direct + kernel -> direct)"
    ),
    "hicache_cache_hit_log": (
        "python/sglang/srt/managers/scheduler_components/metrics_reporter.py (#cached-token, "
        "cache hit rate)"
    ),
    "router_dp_aware": (
        "sgl-model-gateway/bindings/python/src/sglang_router/router_args.py (--dp-aware), "
        "python/sglang/srt/managers/data_parallel_controller.py (routed_dp_rank)"
    ),
    "deepep_sm_warning": (
        "python/sglang/srt/layers/moe/token_dispatcher/deepep.py (normal/auto mode without TBO "
        "warns when DeepEPConfig.num_sms < SMs // 2; --deepep-config normal_dispatch.num_sms)"
    ),
    "deepep_low_latency_cap": (
        "python/sglang/srt/layers/moe/token_dispatcher/deepep.py "
        "(num_max_dispatch_tokens_per_rank <= 1024)"
    ),
    "tbo_prefill_sms": (
        "python/sglang/srt/batch_overlap/operations_strategy.py::_compute_moe_deepseek_blog_prefill "
        "(deep_gemm_num_sms = SMs - DeepEPConfig.num_sms)"
    ),
    "tbo_two_chunk": (
        "python/sglang/srt/batch_overlap/two_batch_overlap.py::_is_two_chunk_split_enabled"
    ),
    "tbo_decode_graph": (
        "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py (tbo_plugin capture "
        "and replay)"
    ),
    "eplb_rules": (
        "python/sglang/srt/arg_groups/parallel_hook.py::handle_eplb_and_dispatch (recorder mode "
        "stat, dispatch static with a2a / dynamic without, ep_size > 1)"
    ),
    "eplb_divisibility": (
        "python/sglang/srt/eplb/expert_location.py::ExpertLocationMetadata.num_local_physical_experts "
        "((experts + redundant) % ep_size == 0)"
    ),
    "dflash_rules": (
        "python/sglang/srt/arg_groups/speculative_hook.py::_handle_dflash (draft model required, "
        "no attention DP except NPU, pp_size 1, steps/topk forced to 1)"
    ),
    "scheduler_prefill_first": (
        "python/sglang/srt/managers/scheduler.py::get_next_batch_to_run (a prefill batch runs "
        "first whenever one can be built; decode otherwise)"
    ),
    "mixed_chunk": (
        "python/sglang/srt/managers/scheduler.py::get_new_batch_prefill (mixed-style chunked "
        "prefill, PrefillAdder gets running_bs), schedule_policy.py::PrefillAdder.__init__ "
        "(budgets minus num_mixed_decode_tokens), speculative/spec_info.py::supports_mixed_chunk"
    ),
    "chunked_prefill_single": (
        "python/sglang/srt/managers/schedule_policy.py::PrefillAdder._select_prefill_admission "
        "(only one unfinished chunked request; the request over the chunk budget is truncated)"
    ),
    "max_num_reqs_default": (
        "python/sglang/srt/mem_cache/kv_cache_configurator.py::resolve_max_num_reqs"
    ),
    "max_req_len": (
        "python/sglang/srt/managers/tp_worker.py::get_worker_info (max_req_len, max_req_input_len "
        "= max_req_len - 5), scheduler.py (max_new_tokens clamped to max_req_len - input - 1), "
        "tokenizer_manager.py (input + max_new_tokens > context_len is rejected)"
    ),
    "pd_kv_layout_match": (
        "python/sglang/srt/disaggregation/common/conn.py (prefill and decode must use the same "
        "--page-size and --kv-cache-dtype; decode DCP needs an MLA KV pool and prefill CP 1)"
    ),
    "pd_decode_prealloc": (
        "python/sglang/srt/disaggregation/decode.py::DecodePreallocQueue (reserves "
        "num_reserved_decode_tokens per active request; the first token comes from prefill)"
    ),
    "pd_decode_slots": (
        "python/sglang/srt/arg_groups/pd_disaggregation_hook.py::handle_pd_disaggregation "
        "(extra decode request slots), python/sglang/srt/disaggregation/decode.py"
        "::_get_new_prebuilt_batch (running batch capped at max_running_requests)"
    ),
    "pd_decode_retraction": (
        "python/sglang/srt/mem_cache/kv_cache_builder.py (decode retraction keeps a host_pool or "
        "cpu_tensor KV backup and resumes), python/sglang/srt/managers/scheduler.py"
        "::_add_request_to_queue (colocated retractions rejoin the back of the waiting queue)"
    ),
    "pd_transfer_backend": (
        "python/sglang/srt/arg_groups/pd_disaggregation_hook.py::handle_pd_disaggregation "
        "(mooncake_tcp -> mooncake + MC_FORCE_TCP; decode DCP transfer backends)"
    ),
    "pd_speculative_both_roles": (
        "python/sglang/srt/disaggregation/prefill.py (EAGLE prefill sends draft hidden states and "
        "top-k), test/registered/disaggregation/test_disaggregation_dsv4.py (same EAGLE flags on "
        "both roles)"
    ),
    "pd_router_bootstrap": (
        "sgl-model-gateway/bindings/python/src/sglang_router/router_args.py (--prefill URL "
        "[BOOTSTRAP_PORT], omitted means none: decode falls back to its own bootstrap port)"
    ),
    "router_cache_aware": "sgl-model-gateway/src/policies/cache_aware.rs::select_worker",
    "multi_instance_ports": (
        "python/sglang/srt/server_args.py::PortArgs.init_new (attention DP: ZMQ ports from "
        "--port + 233; nccl_port random), python/sglang/srt/managers/data_parallel_controller.py"
        "::_broadcast_worker_ports (handshake at port + 13), "
        "python/sglang/srt/arg_groups/fields/device.py (--base-gpu-id for several instances on "
        "one machine)"
    ),
    "dp_size_vs_attn_dp": (
        "python/sglang/srt/arg_groups/parallel_hook.py::handle_data_parallelism (--dp-size > 1 "
        "with --attn-dp-size > 1 raises)"
    ),
    "device_selection": (
        "python/sglang/srt/managers/data_parallel_controller.py (gpu_id = base_gpu_id + local "
        "tp_rank x gpu_id_step), python/sglang/srt/model_executor/model_runner.py "
        "(torch.get_device_module(device).set_device(gpu_id))"
    ),
    "pd_ascend_transfer": (
        "python/sglang/srt/disaggregation/ascend/conn.py (the ascend backend reuses the Mooncake "
        "KV manager, sender and receiver with AscendTransferEngine)"
    ),
}


def source(key: str) -> str:
    """``SGLang@<path>`` reference for a behaviour key (raises on unknown keys to catch typos)."""
    return f"SGLang: {SOURCES[key]}"
