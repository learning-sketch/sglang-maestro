"""Map GPU kernel names to operator categories.

Rules are ordered regexes; the first match wins. The defaults cover kernel names produced by
SGLang's own kernels (sgl-kernel / JIT / Triton), FlashInfer, FlashAttention-3, FlashMLA,
DeepGEMM, cuBLAS/CUTLASS, NCCL, DeepEP and PyTorch ATen. Kernel names on other accelerators
differ, so ``KernelClassifier.with_rules_file`` prepends vendor rules from a YAML file::

    - {category: attention, pattern: "my_npu_paged_attention"}
    - {category: comm, pattern: "hccl_allreduce", comm_pattern: allreduce}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATEGORIES = (
    "comm",
    "attention",
    "moe",
    "moe_routing",
    "gemm",
    "norm",
    "activation",
    "rope",
    "kv_store",
    "quant",
    "sampling",
    "spec",
    "hicache",
    "copy",
    "elementwise",
    "memcpy",
    "memset",
    "other",
)

COMPUTE_CATEGORIES = frozenset(CATEGORIES) - {"comm", "memcpy", "memset", "hicache"}


@dataclass(frozen=True)
class KernelRule:
    category: str
    pattern: re.Pattern[str]
    comm_pattern: str | None = None
    fused: tuple[str, ...] = ()

    @staticmethod
    def make(
        category: str,
        pattern: str,
        comm_pattern: str | None = None,
        fused: tuple[str, ...] | list[str] = (),
    ) -> KernelRule:
        if category not in CATEGORIES:
            raise ValueError(f"unknown kernel category {category!r}; known: {CATEGORIES}")
        return KernelRule(category, re.compile(pattern, re.IGNORECASE), comm_pattern, tuple(fused))


@dataclass(frozen=True)
class KernelClass:
    category: str
    comm_pattern: str | None = None
    fused: tuple[str, ...] = ()


_R = KernelRule.make

DEFAULT_RULES: tuple[KernelRule, ...] = (
    # Fused collectives first: they would otherwise match the norm/allreduce rules separately.
    _R(
        "comm",
        r"allreduce_fusion|all_reduce_fusion|lamport|fused_allreduce|allreduce.*norm|all_reduce_mhc",
        "allreduce",
        ("allreduce", "add", "rmsnorm"),
    ),
    _R(
        "comm",
        r"deep_?ep|internode_ll|intranode::|low_latency_(dispatch|combine)|mori.*(dispatch|combine)",
        "alltoall",
    ),
    _R("comm", r"nccl.*all_?reduce|nccl.*AllReduce", "allreduce"),
    _R("comm", r"nccl.*all_?gather|nccl.*AllGather", "allgather"),
    _R("comm", r"nccl.*reduce_?scatter|nccl.*ReduceScatter", "reducescatter"),
    _R("comm", r"nccl.*(all_?to_?all|SendRecv|Sendrecv|send|recv)", "alltoall"),
    _R("comm", r"nccl.*broadcast", "broadcast"),
    _R("comm", r"nccl", None),
    _R(
        "comm",
        r"cross_device_reduce|custom_all_reduce|one_?shot.*reduce|two_?shot.*reduce|mscclpp|"
        r"symm.*(reduce|allreduce)|(^|[^a-z])all_?reduce",
        "allreduce",
    ),
    _R("comm", r"all_?gather", "allgather"),
    _R("comm", r"reduce_?scatter", "reducescatter"),
    _R("hicache", r"hicache|transfer_kernel_impl|transfer_page_head|load_cache_to_device_buffer"),
    # Fused prologue kernels before the generic attention/norm/rope rules.
    _R(
        "rope",
        r"fused_?qk_?norm_?rope|qknorm_rope|fused_norm_rope|fusedQKNormRope",
        None,
        ("qk_norm", "rope"),
    ),
    _R(
        "rope",
        r"fused_rope_store|rope.*(kv|cache)_?store|rope_and_cache",
        None,
        ("rope", "kv_store"),
    ),
    _R(
        "kv_store",
        r"store_kvcache|set_mla_kv|store_flashmla_cache|store_indexer_cache|reshape_and_cache|"
        r"kv_cache_store|set_kv_buffer|qkv_kv_cache|store_kv_index",
    ),
    _R(
        "moe_routing",
        r"moe_align_block_size|topkGating|moeTopK|moe_fused_gate|grouped_topk|topk_softmax|"
        r"moe_sum|moeFinalize|finalizeKernel|finalizeDeepSeek|count_and_sort_expert|moeSoftmax|moeSigmoid|"
        r"shuffleRows|moe_permute|permuteKernel|ep_scatter|ep_gather|hash_topk",
    ),
    _R(
        "moe",
        r"fused_moe|moe_gemm|MoeGemm|grouped_gemm|groupedGemm|m_grouped|cutlass_moe|ep_moe|"
        r"marlin_moe|moe_wna16|fused_experts|expert|moe_vec|trtllm.*moe|fp8_block_scale_moe",
    ),
    # FlashInfer prefixes every kernel (norm, sampling, rope, ...) with "flashinfer::", so attention
    # needs specific names rather than "flash".
    _R(
        "attention",
        r"flash_fwd|flash_bwd|flash_attn|FlashAttn|flash_mla|flashmla|flash_decoding|fmha|"
        r"BatchPrefillWith|BatchDecodeWith|BatchMLAPaged|PersistentVariableLength|merge_attn_states|"
        r"merge_state|xqa|decode_attention|_fwd_kernel_stage|_fwd_grouped_kernel|paged_attention|attn_fwd|"
        r"splitkv|mla_decode|mla_prefill|sparse_mla|cutlass_mla|flex_attention|attention",
    ),
    _R(
        "activation",
        r"silu_and_mul.*quant|silu_mul.*quant|act_and_mul.*quant",
        None,
        ("act", "mul", "quant"),
    ),
    _R("norm", r"rms_?norm.*quant|norm.*fp8_quant", None, ("rmsnorm", "quant")),
    _R(
        "quant",
        r"per_token_group_quant|per_token_quant|per_tensor_quant|quant_fp8|cvt_fp8|scaled_fp8_quant|"
        r"fp8_quant|group_quant|quantize|Quant",
    ),
    _R(
        "gemm",
        r"gemm|Gemm|GEMM|cublas|cutlass|xmma|nvjet|wgmma|matmul|gemv|deep_gemm|fp8_scaled_mm|"
        r"Marlin|marlin|splitKreduce|sm90_|sm80_|ampere_|hopper_",
    ),
    _R("norm", r"fused_add_rms_?norm|FusedAddRMSNorm", None, ("add", "rmsnorm")),
    _R(
        "norm",
        r"rms_?norm|RMSNorm|layer_?norm|LayerNorm|qknorm|qk_norm|norm_kernel|rmsnorm_cta|rmsnorm_warp",
    ),
    _R(
        "activation",
        r"act_and_mul|silu_and_mul|gelu_and_mul|SiluAndMul|GeluAndMul|swiglu",
        None,
        ("act", "mul"),
    ),
    _R("activation", r"activation|silu|gelu|act_kernel|act_only"),
    _R("rope", r"rotary|rope|BatchQKApplyRotary"),
    _R("spec", r"VerifyTree|build_tree|TreeSpeculative|eagle|draft_extend|speculative"),
    _R(
        "sampling",
        r"sampling|Sampling|TopKTopP|top_k_top_p|top_p|min_p|argmax|multinomial|"
        r"LogitsBitmask|topk_kernel|softmax.*sample",
    ),
    _R(
        "copy",
        r"CatArrayBatchedCopy|copy_kernel|memcpy_kernel|cat_kernel|cat_pad|index_select|indexSelect|"
        r"gather_scatter|index_elementwise|index_put|scatter_gather|direct_copy|fused_metadata_copy",
    ),
    _R(
        "elementwise",
        r"elementwise_kernel|vectorized_elementwise|unrolled_elementwise|reduce_kernel|"
        r"fill_kernel|FillFunctor|add_kernel|mul_kernel|where_kernel|CUDAFunctor|BinaryFunctor|"
        r"triton_poi|triton_red|triton_per|_to_copy|cast",
    ),
)


@dataclass
class KernelClassifier:
    rules: tuple[KernelRule, ...] = DEFAULT_RULES
    _cache: dict[str, KernelClass] = field(default_factory=dict, repr=False)

    def classify(self, name: str, event_category: str | None = None) -> KernelClass:
        """Classify one kernel; ``event_category`` is the trace ``cat`` (memcpy/memset come from it)."""
        if event_category:
            lowered = event_category.lower()
            if lowered == "gpu_memcpy":
                return KernelClass("memcpy")
            if lowered == "gpu_memset":
                return KernelClass("memset")
        cached = self._cache.get(name)
        if cached is not None:
            return cached
        result = KernelClass("other")
        if name.startswith("Memcpy"):
            result = KernelClass("memcpy")
        elif name.startswith("Memset"):
            result = KernelClass("memset")
        else:
            for rule in self.rules:
                if rule.pattern.search(name):
                    result = KernelClass(rule.category, rule.comm_pattern, rule.fused)
                    break
        self._cache[name] = result
        return result

    def with_rules(self, extra_rules: list[KernelRule]) -> KernelClassifier:
        return KernelClassifier(tuple(extra_rules) + self.rules)

    def with_rules_file(self, path: str | Path) -> KernelClassifier:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a list of {{category, pattern}} rules")
        extra = [
            KernelRule.make(
                item["category"], item["pattern"], item.get("comm_pattern"), item.get("fused") or ()
            )
            for item in payload
        ]
        return self.with_rules(extra)
