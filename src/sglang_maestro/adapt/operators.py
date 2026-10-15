"""Which kernels a model reaches in SGLang, and who provides each of them on the device.

For every operator family the model executes, the inventory records the SGLang entry point that
dispatches it, the exact classes a plugin has to register, the path SGLang resolves on this
platform (vendor kernel, in-tree kernel, pure-torch fallback or nothing), what that path costs,
and the share of step time the cost model gives the operator, so gaps can be ranked.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from sglang_maestro.adapt.platform import PlatformProfile
from sglang_maestro.adapt.rope import rope_class_for
from sglang_maestro.hardware.spec import HardwareSpec
from sglang_maestro.knowledge import SGLangKnowledge
from sglang_maestro.knowledge.facts import (
    DEEPSEEK_V2_MODEL_ARCHITECTURES,
    DEFAULT_OOT_DISPATCH_KEY,
    MOE_NATIVE_ACTIVATIONS,
    source,
)
from sglang_maestro.model.config import ModelArch
from sglang_maestro.model.ops import BatchShape, StepCostModel
from sglang_maestro.model.parallel import ParallelConfig

KNOWN_FUSED_OPS = frozenset(
    {
        "rmsnorm",
        "fused_add_rmsnorm",
        "layernorm",
        "silu_and_mul",
        "gelu_and_mul",
        "rope",
        "qk_norm",
        "kv_cache_store",
        "topk_softmax",
        "grouped_topk",
        "moe_align",
        "fused_moe",
        "moe_sum",
        "per_token_group_quant_fp8",
        "fp8_blockwise_gemm",
        "fp8_gemm",
        "fp8_w8a16_gemm",
        "int8_gemm",
        "w4a16_gemm",
        "sampling",
        "custom_allreduce",
        "allreduce_rmsnorm_fusion",
        "attention_mha_prefill",
        "attention_mha_decode",
        "attention_mla_prefill",
        "attention_mla_decode",
        "attention_sliding_window",
        "hicache_transfer",
        "positions",
        "dsa_indexer",
    }
)
"""Vocabulary of ``software.fused_ops`` in hardware YAML; SGLang class names are accepted too."""

STATUSES = ("broken", "missing", "slow", "native", "vendor", "in_tree")
STATUS_LABELS = {
    "broken": "当前配置会报错",
    "missing": "该平台没有可用实现",
    "slow": "能跑，但有 host 同步/逐项循环",
    "native": "回落 torch 参考实现",
    "vendor": "厂商实现",
    "in_tree": "SGLang 内置实现",
}
_STATUS_RANK = {name: index for index, name in enumerate(STATUSES)}
GAP_STATUSES = ("broken", "missing", "slow", "native")
"""Statuses that need work before the operator runs well on the device."""

ACTIVATION_CLASSES = {
    "silu": "SiluAndMul",
    "swish": "SiluAndMul",
    "swiglu": "SiluAndMul",
    "gelu": "GeluAndMul",
    "gelu_new": "GeluAndMul",
    "gelu_fast": "GeluAndMul",
    "gelu_pytorch_tanh": "GeluAndMul",
    "relu2": "ReLU2",
    "quick_gelu": "QuickGELU",
    "xielu": "XIELU",
}
ACTIVATION_VENDOR_OPS = {"SiluAndMul": "silu_and_mul", "GeluAndMul": "gelu_and_mul"}

DECODE_SHAPE = (32, 4096)
"""(batch, context) of the decode step used for cost shares."""
PREFILL_SHAPE = (1, 4096)
"""(sequences, new tokens) of the prefill step used for cost shares."""


@dataclass(frozen=True)
class OperatorRequirement:
    key: str
    category: str
    title: str
    sglang_entry: str
    """Where SGLang dispatches the operator (fused-op class, registry, quant method, ...)."""
    status: str
    resolved_path: str
    """What runs on this platform today."""
    impact: str
    how: str
    contract: str = ""
    """Signature/semantics a vendor implementation has to honour."""
    op_classes: tuple[str, ...] = ()
    """Exact classes for ``BaseFusedOp.register_oot_forward``; subclasses are not covered."""
    vendor_ops: tuple[str, ...] = ()
    share_decode: float | None = None
    share_prefill: float | None = None
    references: tuple[str, ...] = ()

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def max_share(self) -> float:
        return max(self.share_decode or 0.0, self.share_prefill or 0.0)


@dataclass
class OperatorInventory:
    requirements: list[OperatorRequirement] = field(default_factory=list)
    unknown_vendor_ops: list[str] = field(default_factory=list)
    shares_note: str = ""

    def by_status(self, *statuses: str) -> list[OperatorRequirement]:
        return [item for item in self.requirements if item.status in statuses]

    def get(self, key: str) -> OperatorRequirement | None:
        return next((item for item in self.requirements if item.key == key), None)


def _cost_key(name: str, kind: str, comm_pattern: str | None) -> str:
    if name == "moe.topk":
        return "topk"
    if kind == "comm":
        return "moe_a2a" if comm_pattern == "alltoall" else "collectives"
    if kind == "moe" or name in ("moe.align", "moe.sum"):
        return "moe_experts"
    if name == "lm_head":
        return "lm_head"
    if name == "embed.lookup":
        return "embedding"
    return {
        "norm": "norm",
        "rope": "rope",
        "activation": "activation",
        "attention": "attention",
        "kv_store": "kv_cache_store",
        "gemm": "linear",
        "quant": "linear",
        "sampling": "sampling",
    }.get(kind, "other")


def _cost_shares(
    arch: ModelArch, hardware: HardwareSpec, parallel: ParallelConfig
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-family share of kernel time for a representative decode and prefill step."""
    try:
        model = StepCostModel(arch, hardware, parallel)
        steps = (
            model.estimate(BatchShape.decode(*DECODE_SHAPE)),
            model.estimate(BatchShape.prefill(*PREFILL_SHAPE)),
        )
    except (ValueError, ZeroDivisionError):
        return {}, {}
    shares: list[dict[str, float]] = []
    for step in steps:
        totals: dict[str, float] = {}
        for timing in step.timings:
            key = _cost_key(timing.op.name, timing.op.kind, timing.op.comm_pattern)
            totals[key] = totals.get(key, 0.0) + timing.total_us
        overall = sum(totals.values()) or 1.0
        shares.append({key: value / overall for key, value in totals.items()})
    return shares[0], shares[1]


class _Context:
    """Lookups shared by the per-operator rules."""

    def __init__(
        self,
        arch: ModelArch,
        hardware: HardwareSpec,
        profile: PlatformProfile,
        knowledge: SGLangKnowledge,
        parallel: ParallelConfig,
    ) -> None:
        self.arch = arch
        self.hardware = hardware
        self.software = hardware.software
        self.profile = profile
        self.knowledge = knowledge
        self.parallel = parallel
        self.declared = {name.lower() for name in self.software.fused_ops}
        self.key = profile.dispatch_key or DEFAULT_OOT_DISPATCH_KEY

    def has(self, *names: str) -> bool:
        return all(name.lower() in self.declared for name in names)

    def has_any(self, *names: str) -> bool:
        return any(name.lower() in self.declared for name in names)

    def fused_op_record(self, class_name: str) -> dict | None:
        candidates = self.knowledge.find_fused_op_classes(class_name)
        preferred = [name for name in candidates if name.startswith("sglang.srt.layers.")]
        chosen = (preferred or candidates or [None])[0]
        return self.knowledge.fused_op_class(chosen) if chosen else None

    def fused_op_path(self, class_name: str) -> tuple[str, str]:
        """``(method, note)`` that ``BaseFusedOp`` resolves for ``class_name`` when nothing is registered."""
        record = self.fused_op_record(class_name)
        if record is None:
            return "forward_native", f"快照里没有 {class_name}，按 forward_native 处理"
        override = record.get("forward_override") or {}
        if override and not override.get("calls_dispatch", True):
            return "forward", f"{class_name}.forward() 不经过分发，注册表与 forward_<key> 都不生效"
        effective = set(record.get("effective_forwards") or ())
        stubs = set(record.get("stub_forwards") or ())
        if self.profile.out_of_tree:
            method = f"forward_{self.key}"
            if method in effective and method not in stubs:
                return method, f"{class_name} 已有 {method}"
            if "forward_native" in stubs:
                return "forward_native", f"{class_name}.forward_native 是占位实现"
            return "forward_native", ""
        methods = self.knowledge.fused_op_dispatch.get("platform_methods", {}).get(self.key, [])
        for method in methods:
            if method in effective and method not in stubs:
                return method, ""
        return "forward_native", ""

    def fused_op_status(self, class_name: str, *vendor_ops: str) -> tuple[str, str, str]:
        """``(status, resolved_path, note)`` for a fused op the YAML may declare as vendor-provided."""
        method, note = self.fused_op_path(class_name)
        if method == "forward":
            return "broken", f"{class_name}.forward（绕过分发）", note
        if self.profile.out_of_tree and vendor_ops and self.has_any(*vendor_ops, class_name):
            return "vendor", f"register_oot_forward({class_name}) / forward_{self.key}", note
        if method == "forward_native":
            return "native", f"{class_name}.forward_native（纯 torch）", note
        if self.profile.out_of_tree:
            return "vendor", f"{class_name}.{method}", note
        return "in_tree", f"{class_name}.{method}", note


def _norm_requirement(ctx: _Context) -> OperatorRequirement:
    arch = ctx.arch
    if arch.model_type.startswith("gemma"):
        class_name = "GemmaRMSNorm"
    elif arch.norm_type == "layernorm":
        class_name = "LayerNorm"
    else:
        class_name = "RMSNorm"
    vendor_ops = ("layernorm",) if class_name == "LayerNorm" else ("rmsnorm", "fused_add_rmsnorm")
    status, path, note = ctx.fused_op_status(class_name, *vendor_ops)
    attention = arch.attention
    widths = [f"hidden={arch.hidden_size}"]
    if attention.is_mla:
        if attention.q_lora_rank:
            widths.append(f"q_a_layernorm={attention.q_lora_rank}")
        widths.append(f"kv_a_layernorm={attention.kv_lora_rank}")
    if attention.has_qk_norm:
        widths.append(f"q/k norm=head_dim {attention.head_dim}（逐头）")
    per_layer = 2 + (2 if attention.is_mla else 0) + (2 if attention.has_qk_norm else 0)
    if status == "native":
        impact = (
            f"每层约 {per_layer} 次 norm，forward_native 先转 fp32 再做 pow/mean/rsqrt/mul，"
            "每次 4-6 个 elementwise kernel、2-3 倍 HBM 读写；残差相加也拆成独立 kernel"
        )
    else:
        impact = f"每层约 {per_layer} 次 norm；残差路径必须在同一个 kernel 里完成 add+norm"
    if ctx.has("rmsnorm") and not ctx.has("fused_add_rmsnorm") and class_name != "LayerNorm":
        impact += "；YAML 只声明了 rmsnorm，没有 fused_add_rmsnorm：残差相加会多一个 kernel"
    if note:
        impact += f"（{note}）"
    return OperatorRequirement(
        key="norm",
        category="norm",
        title=f"{class_name}（含 fused add+norm{'、QK norm' if attention.has_qk_norm else ''}）",
        sglang_entry=f"BaseFusedOp {class_name}",
        status=status,
        resolved_path=path,
        impact=impact,
        how=(
            f"BaseFusedOp.register_oot_forward({class_name}, fn, '{ctx.key}')，或在 {class_name} 上"
            f"定义 forward_{ctx.key}（子类经 MRO 继承）。宽度：{'，'.join(widths)}"
        ),
        contract=(
            "fn(self, x, residual=None, post_residual_addition=None, quant_linear=None)：residual "
            "为 None 时返回 norm(x)；否则先 residual = x + residual (+ post_residual_addition)，"
            "返回 (norm(residual), residual)；fp32 累加、eps=self.variance_epsilon、权重 "
            "self.weight；layer_boundary 的残差路径也调用同一个模块"
            if class_name != "LayerNorm"
            else "fn(self, x)：F.layer_norm 语义（weight/bias 可选）"
        ),
        op_classes=(class_name,),
        vendor_ops=vendor_ops,
        references=(
            source("fused_op_dispatch"),
            source("oot_dispatch_order"),
            source("rmsnorm_contract"),
            source("add_norm_plain"),
        ),
    )


def _rope_requirement(ctx: _Context) -> OperatorRequirement:
    arch = ctx.arch
    attention = arch.attention
    choice = rope_class_for(arch)
    references = [source("rope_factory"), source("rope_contract"), source("rope_cache_dtype")]
    if choice.rewritten:
        references.append(source("deepseek_yarn_rewrite"))
    if choice.class_name is None:
        return OperatorRequirement(
            key="rope",
            category="rope",
            title=f"RoPE（rope_type={choice.scaling_type}）",
            sglang_entry="layers/rotary_embedding/factory.py::get_rope",
            status="broken",
            resolved_path="ValueError: Unknown RoPE scaling type",
            impact="get_rope 不认识该 scaling 类型，模型构造时直接报错",
            how="在 checkpoint config 里改成 SGLang 支持的 rope_type，或用通用插件替换 get_rope",
            references=tuple(references),
        )
    class_name = choice.class_name
    status, path, note = ctx.fused_op_status(class_name, "rope")
    interleaved = attention.is_mla and arch.primary_architecture in DEEPSEEK_V2_MODEL_ARCHITECTURES
    if interleaved:
        interleaved = bool(arch.raw.get("rope_interleave", True))
        references.append(source("deepseek_rope_interleave"))
    rotary_dim = attention.qk_rope_head_dim if attention.is_mla else attention.head_dim
    partial = arch.raw.get("partial_rotary_factor")
    if partial and not attention.is_mla:
        rotary_dim = int(rotary_dim * float(partial))
    details = [
        f"rotary_dim={rotary_dim}",
        f"head_size={attention.qk_head_dim if attention.is_mla else attention.head_dim}",
        "is_neox_style=False（GPT-J 交错）" if interleaved else "is_neox_style=True",
    ]
    if choice.rewritten:
        details.append("DeepSeek MLA 把 rope_scaling 改写为 deepseek_yarn")
    impact = (
        "forward_native 用 index_select + 拼接实现，每层 6-10 个小 kernel"
        if status == "native"
        else "每层一次（q 和 k 一起）"
    )
    if ctx.profile.out_of_tree:
        impact += (
            "；离开 CUDA/HIP/XPU 时 cos/sin cache 会被转成模型 dtype（bf16），长位置精度下降，"
            "可设 SGLANG_ROPE_CACHE_FP32=1"
        )
    if note:
        impact += f"（{note}）"
    how = (
        f"register_oot_forward({class_name}, fn, '{ctx.key}')——注册表按精确类匹配，只注册 "
        f"RotaryEmbedding 不会覆盖 {class_name}；或者在 RotaryEmbedding 上定义 forward_{ctx.key}，"
        "子类经 MRO 继承（CUDA 上子类同样复用基类的 forward_cuda）"
        if class_name != "RotaryEmbedding"
        else f"register_oot_forward(RotaryEmbedding, fn, '{ctx.key}') 或定义 forward_{ctx.key}"
    )
    scaling = "无 scaling" if choice.scaling_type == "none" else choice.scaling_type
    return OperatorRequirement(
        key="rope",
        category="rope",
        title=f"RoPE：{class_name}（{scaling}）",
        sglang_entry=f"BaseFusedOp {class_name}",
        status=status,
        resolved_path=path,
        impact=impact,
        how=how + "。" + "，".join(details),
        contract=(
            "fn(self, positions, query, key, offsets=None, fused_set_kv_buffer_arg=None) -> "
            "(query, key)：用 self.cos_sin_cache[positions] 原地或返回旋转后的 q/k；只旋转前 "
            "rotary_dim 维；fused_set_kv_buffer_arg 非 None 时要同时写 KV（native 路径断言它为 None）"
        ),
        op_classes=(class_name,),
        vendor_ops=("rope",),
        references=tuple(references),
    )


def _activation_requirement(ctx: _Context) -> OperatorRequirement | None:
    arch = ctx.arch
    has_dense = arch.num_dense_layers > 0 or (arch.moe and arch.moe.shared_intermediate_size)
    if not has_dense:
        return None
    activation = arch.hidden_act.lower()
    class_name = ACTIVATION_CLASSES.get(activation)
    if class_name is None:
        return OperatorRequirement(
            key="activation",
            category="activation",
            title=f"激活函数 {activation}",
            sglang_entry="models/<model>.py",
            status="missing",
            resolved_path="未知",
            impact=f"sglang-maestro 没有 {activation} 到 SGLang 激活类的映射，请人工确认模型文件",
            how="确认模型文件里实例化的激活类，再按 BaseFusedOp 规则注册",
        )
    vendor_op = ACTIVATION_VENDOR_OPS.get(class_name)
    status, path, note = ctx.fused_op_status(class_name, *((vendor_op,) if vendor_op else ()))
    users = []
    if arch.num_dense_layers:
        users.append(f"{arch.num_dense_layers} 个稠密 MLP 层")
    if arch.moe and arch.moe.shared_intermediate_size:
        users.append("共享专家")
    if arch.moe:
        users.append("路由专家的激活在 MoE kernel 内")
    impact = (
        "forward_native 拆成切片 + 激活 + 乘法 3 个 kernel，多读写一遍中间激活"
        if status == "native"
        else "门控激活（gate、up 拼在最后一维）"
    )
    if status == "broken":
        impact = f"{note}：只能整体替换模型里的激活模块"
    elif note:
        impact += f"（{note}）"
    return OperatorRequirement(
        key="activation",
        category="activation",
        title=f"{class_name}（{'；'.join(users)}）",
        sglang_entry=f"BaseFusedOp {class_name}",
        status=status,
        resolved_path=path,
        impact=impact,
        how=f"register_oot_forward({class_name}, fn, '{ctx.key}') 或定义 forward_{ctx.key}",
        contract=(
            "fn(self, x)：x 最后一维 = 2d，前 d 维是 gate，后 d 维是 up，返回 act(gate) * up"
            if arch.gated_mlp
            else "fn(self, x)：逐元素激活"
        ),
        op_classes=(class_name,),
        vendor_ops=(vendor_op,) if vendor_op else (),
        references=(source("silu_and_mul"), source("oot_dispatch_order")),
    )


def _attention_requirement(ctx: _Context) -> OperatorRequirement:
    arch, software = ctx.arch, ctx.software
    attention = arch.attention
    heads = attention.num_heads // max(ctx.parallel.attn_tp, 1)
    if attention.is_mla:
        needed = ("attention_mla_prefill", "attention_mla_decode")
        title = "MLA 注意力（prefill 走 MHA 形态，decode 走吸收后的 latent 形态）"
        contract = (
            f"extend/无前缀 prefill：q,k [T, {heads}, {attention.qk_head_dim}]，v [T, {heads}, "
            f"{attention.v_head_dim}]，causal；带前缀的 extend 与 decode：q [B, {heads}, "
            f"{attention.mla_latent_dim}]（{attention.kv_lora_rank} latent + "
            f"{attention.qk_rope_head_dim} rope），KV cache 每 token 一个 {attention.mla_latent_dim} "
            f"维 latent（单 KV 头），输出取前 {attention.kv_lora_rank} 维；softmax scale 含 YaRN mscale"
        )
    else:
        needed = ("attention_mha_prefill", "attention_mha_decode")
        title = f"{attention.kind} 注意力"
        kv_heads = max(1, attention.num_kv_heads // max(ctx.parallel.attn_tp, 1))
        contract = (
            f"q [T, {heads}, {attention.head_dim}]，KV cache [tokens, {kv_heads}, "
            f"{attention.head_dim}]（按 page_size 分页），prefill causal + 前缀，decode 每请求 1 个 query"
        )
    if attention.sliding_window:
        needed += ("attention_sliding_window",)
        title += f"，{attention.num_sliding_layers} 层滑动窗口 {attention.sliding_window}"
    if attention.sparse_indexer:
        title += "，DSA 稀疏（indexer top-k）"
    backends = tuple(software.attention_backends)
    declared = software.default_attention_backend or (backends[0] if backends else None)
    references = (
        source("attention_registry"),
        source("oot_default_attention_backend"),
        *((source("mla_backend_handler"),) if attention.is_mla else ()),
    )
    missing_ops = [name for name in needed if not ctx.has(name)]
    if ctx.profile.in_tree:
        status = "in_tree" if declared else "missing"
        path = f"--attention-backend {declared}" if declared else "未声明可用后端"
        impact = "SGLang 内置后端" if declared else "YAML 没有声明该平台可用的注意力后端"
    elif declared:
        status = "vendor"
        path = f"注册的后端 {declared!r}"
        impact = "注意力是 decode 的主要耗时之一，长上下文时占比更高"
        if missing_ops:
            impact += f"；YAML fused_ops 缺少 {'/'.join(missing_ops)}，确认后端覆盖了这些形态"
    else:
        status = "missing"
        path = "无（get_default_attention_backend 默认抛异常）"
        impact = "out-of-tree 平台没有可用的注意力后端，服务无法启动"
    return OperatorRequirement(
        key="attention",
        category="attention",
        title=title,
        sglang_entry="AttentionBackend 注册表（--attention-backend）",
        status=status,
        resolved_path=path,
        impact=impact,
        how=(
            '@register_attention_backend("<name>") 注册 AttentionBackend 子类，实现 '
            "init_forward_metadata / forward_extend / forward_decode（图捕获还要 "
            "init_cuda_graph_state / init_forward_metadata_capture_cuda_graph / "
            "init_forward_metadata_replay_cuda_graph）；get_default_attention_backend 返回该名字"
            + (
                "；MLA 模型再用 AttentionBackendRegistry.register 指定每个 batch 走哪种形态"
                if attention.is_mla
                else ""
            )
        ),
        contract=contract,
        vendor_ops=needed,
        references=references,
    )


def _kv_store_requirement(ctx: _Context) -> OperatorRequirement:
    attention = ctx.arch.attention
    title = "KV cache 写入" + ("（MLA latent + rope 拼接写入）" if attention.is_mla else "")
    if ctx.profile.in_tree:
        return OperatorRequirement(
            key="kv_cache_store",
            category="kv_cache",
            title=title,
            sglang_entry="mem_cache/memory_pool.py::set_kv_buffer",
            status="in_tree",
            resolved_path="store_cache / Triton kernel",
            impact="内置实现",
            how="无需改动",
            vendor_ops=("kv_cache_store",),
            references=(source("kv_store_fallback"),),
        )
    if ctx.has("kv_cache_store"):
        status, path = "vendor", "厂商 KV pool（get_kv_pool_cls）或注意力后端内写入"
    else:
        status = "native"
        path = (
            "set_mla_kv_buffer_naive（torch 索引写）"
            if attention.is_mla and not ctx.software.supports_triton
            else "k_cache[loc] = k; v_cache[loc] = v（torch index_put）"
        )
    return OperatorRequirement(
        key="kv_cache_store",
        category="kv_cache",
        title=title,
        sglang_entry="mem_cache/memory_pool.py::set_kv_buffer",
        status=status,
        resolved_path=path,
        impact=(
            "torch 索引写是正确且可图捕获的，但每层多 1-2 个 scatter kernel；注意力后端若在 kernel "
            "里直接写 KV 可以省掉"
        ),
        how="需要特殊布局时用 get_kv_pool_cls 返回子类并重写 set_kv_buffer / _create_buffers",
        vendor_ops=("kv_cache_store",),
        references=(source("kv_store_fallback"), source("mla_kv_buffer"), source("kv_pool_hook")),
    )


def _positions_requirement(ctx: _Context) -> OperatorRequirement | None:
    if ctx.software.supports_triton:
        return None
    status = "vendor" if ctx.has("positions") else "slow"
    return OperatorRequirement(
        key="positions",
        category="runtime",
        title="prefill 位置编号（compute_position）",
        sglang_entry="model_executor/forward_batch_info.py::compute_position",
        status=status,
        resolved_path="compute_position_torch" if status == "slow" else "插件替换的实现",
        impact=(
            "support_triton() 为 False 时 compute_position_torch 在 Python 里遍历设备张量 "
            "extend_prefix_lens/extend_seq_lens：每个请求一次 D2H 同步加一次 arange，prefill "
            "batch 请求越多越慢，也无法进入 prefill 图"
        ),
        how=(
            "用通用插件（sglang.srt.plugins）的 HookRegistry REPLACE 掉 compute_position_torch，"
            "换成无同步的向量化版本：start = cumsum(seq_lens) - seq_lens；positions = "
            "arange(N) + repeat_interleave(prefix_lens - start, seq_lens, output_size=N)"
            "（N = extend_seq_lens_sum，由调用方传入，不需要同步）"
        ),
        contract="返回 (positions[int64, N], extend_start_loc[同 seq_lens dtype])",
        vendor_ops=("positions",),
        references=(source("compute_position"), source("general_plugin_hooks")),
    )


def _topk_requirement(ctx: _Context) -> OperatorRequirement | None:
    moe = ctx.arch.moe
    if moe is None:
        return None
    parallel = ctx.parallel
    grouped = bool(moe.num_expert_groups and moe.num_expert_groups > 1)
    vendor_ops = ("grouped_topk",) if grouped else ("topk_softmax",)
    status, path, note = ctx.fused_op_status("TopK", *vendor_ops)
    references = [source("oot_dispatch_order"), source("topk_native_asserts")]
    scoring = moe.routing.split("_", 1)[0]
    if grouped:
        title = (
            f"路由 TopK：{scoring} 分组 top-k（{moe.num_experts} 专家 / {moe.num_expert_groups} 组 "
            f"选 {moe.topk_groups} 组 / top-{moe.top_k}{'，带 correction bias' if moe.has_router_bias else ''}）"
        )
        impact = (
            "forward_native → select_experts → biased_grouped_topk：非 CUDA 平台用 "
            "torch.compile(backend=get_compile_backend()) 编译的 biased_grouped_topk_impl，"
            "编译后端不可用时这里先失败"
            if status == "native"
            else "每个 MoE 层一次"
        )
        references.append(source("oot_grouped_topk"))
    else:
        title = f"路由 TopK：{scoring} top-{moe.top_k}（{moe.num_experts} 专家）"
        impact = "forward_native → fused_topk_native（softmax + topk + 归一化，3-4 个 kernel）"
        references.append(source("oot_softmax_topk"))
        passes_padding = parallel.ep > 1 and (parallel.attn_dp == 1 or parallel.uses_a2a)
        if status == "native" and passes_padding:
            status = "broken"
            impact = (
                f"{parallel.describe()} 时 SGLang 会给 MoE 传 num_token_non_padded，而 "
                "fused_topk_native 断言它为 None：第一个 batch 就 AssertionError"
            )
            references.append(source("num_token_non_padded"))
        elif status == "native":
            impact += "；ep>1（attention DP=1）或启用 EPLB 时该路径会断言失败，届时必须有厂商 TopK"
    if note:
        impact += f"（{note}）"
    return OperatorRequirement(
        key="topk",
        category="moe",
        title=title,
        sglang_entry="BaseFusedOp TopK（layers/moe/topk.py）",
        status=status,
        resolved_path=path,
        impact=impact,
        how=f"register_oot_forward(TopK, fn, '{ctx.key}')，内部按 self.topk_config 实现路由",
        contract=(
            "fn(self, hidden_states, router_logits, *, num_token_non_padded=None, "
            "expert_location_dispatch_info=None, dynamic_expert_bias=None, "
            "router_logits_partials=None) -> TopKOutput(topk_weights[fp32], topk_ids[int32], "
            "router_logits)；padding 行（≥ num_token_non_padded）的 topk_ids 置 -1"
        ),
        op_classes=("TopK",),
        vendor_ops=vendor_ops,
        references=tuple(references),
    )


def _moe_requirement(ctx: _Context) -> OperatorRequirement | None:
    arch, software = ctx.arch, ctx.software
    moe = arch.moe
    if moe is None:
        return None
    parallel = ctx.parallel
    local_experts = moe.num_experts // max(parallel.ep, 1)
    intermediate = moe.expert_intermediate_size // max(parallel.moe_tp, 1)
    shape = (
        f"每卡 {local_experts} 个专家，每个 gate_up [{2 * intermediate}, {arch.hidden_size}]、"
        f"down [{arch.hidden_size}, {intermediate}]，top-{moe.top_k}"
    )
    quant = arch.quant
    references = [source("moe_native_fallback")]
    if quant is None or quant.applies_to == "none":
        status, path, _ = ctx.fused_op_status("UnquantizedFusedMoEMethod", "fused_moe")
        if status == "native":
            status = "slow"
            impact = (
                "moe_forward_native 逐专家循环，并对 tokens_per_expert 调用 .cpu()：每层一次 host "
                f"同步、{local_experts} 次小 GEMM；graph capture 中的 host 同步会让捕获失败，"
                "decode 只能 eager"
            )
            if arch.hidden_act.lower() not in MOE_NATIVE_ACTIVATIONS:
                status = "broken"
                impact = (
                    f"moe_forward_native 只实现 {'/'.join(MOE_NATIVE_ACTIVATIONS)}，"
                    f"{arch.hidden_act} 会直接报错"
                )
        else:
            impact = "MoE 是 MoE 模型最大的耗时项"
        how = (
            f"register_oot_forward(UnquantizedFusedMoEMethod, fn, '{ctx.key}') 实现 grouped GEMM，"
            "或用 register_moe_runner_core 注册厂商 runner（仅非 ('none','triton') 组合生效）"
        )
        sglang_entry = "BaseFusedOp UnquantizedFusedMoEMethod"
        op_classes: tuple[str, ...] = ("UnquantizedFusedMoEMethod",)
        references.append(source("moe_runner_core"))
    else:
        sglang_entry = f"{quant.method} 的 FusedMoEMethod（MoeRunner）"
        op_classes = ()
        references += [source("moe_triton_fused_func"), source("oot_quantization_hook")]
        if ctx.profile.in_tree:
            status, path = "in_tree", "Triton fused_moe（或 DeepGEMM/Marlin runner）"
            impact = "检查该形状是否有调优过的 Triton 配置（场景 2 的 moe_tuning）"
        elif ctx.has("fused_moe"):
            status, path = (
                "vendor",
                f"get_quantization_config({quant.method!r}) 返回的厂商 MoE 方法",
            )
            impact = "MoE 是 MoE 模型最大的耗时项"
        elif software.supports_triton:
            status, path = "native", "Triton fused_moe（厂商 Triton 后端编译）"
            impact = (
                f"需要厂商 Triton 支持 {quant.weight_dtype} 的 tl.dot，且没有该设备的调优配置，"
                "性能通常远低于手写 kernel"
            )
        else:
            status, path = "missing", "无（量化 MoE 只有 Triton runner）"
            impact = (
                f"{quant.method} 量化 MoE 只有 Triton runner，('none','triton') 组合不经过 "
                "register_moe_runner_core，supports_triton=False 的设备上无法运行"
            )
        how = (
            f"get_quantization_config({quant.method!r}) 返回厂商 QuantizationConfig，它的 "
            "get_quant_method(FusedMoE) 返回厂商 FusedMoEMethodBase（create_weights / "
            "process_weights_after_loading / apply）"
        )
    return OperatorRequirement(
        key="moe_experts",
        category="moe",
        title=f"路由专家 grouped GEMM（{'未量化' if quant is None else quant.method}）",
        sglang_entry=sglang_entry,
        status=status,
        resolved_path=path,
        impact=impact,
        how=how + "。" + shape,
        contract=(
            "apply(layer, dispatch_output) -> combine_input：按 topk_ids/topk_weights 计算 "
            "sum_k w_k · down(act(gate_up(x)))；activation 由 layer.moe_runner_config 给出"
        ),
        op_classes=op_classes,
        vendor_ops=("fused_moe", "moe_align", "moe_sum"),
        references=tuple(references),
    )


def _a2a_requirement(ctx: _Context) -> OperatorRequirement | None:
    parallel = ctx.parallel
    if ctx.arch.moe is None or not parallel.uses_a2a:
        return None
    backend = parallel.moe_a2a_backend
    available = backend in ctx.software.moe_a2a_backends
    return OperatorRequirement(
        key="moe_a2a",
        category="communication",
        title=f"EP all-to-all（--moe-a2a-backend {backend}）",
        sglang_entry="layers/moe/token_dispatcher",
        status=("in_tree" if ctx.profile.in_tree else "vendor") if available else "missing",
        resolved_path=backend if available else "无",
        impact=(
            "dispatch/combine 每个 MoE 层两次 all-to-all"
            if available
            else f"YAML moe_a2a_backends={list(ctx.software.moe_a2a_backends)} 不含 {backend}"
        ),
        how="改用 --moe-a2a-backend none（EP 退化为 allgather+reduce）或实现厂商 token dispatcher",
        references=(source("deepep_low_latency_tokens"),),
    )


def _linear_requirement(ctx: _Context) -> OperatorRequirement:
    arch, hardware, software = ctx.arch, ctx.hardware, ctx.software
    quant = arch.quant
    gemm_dtypes = {name.lower() for name in software.gemm_dtypes}
    if quant is None:
        dtype = arch.activation_dtype
        supported = dtype in gemm_dtypes
        return OperatorRequirement(
            key="linear",
            category="gemm",
            title=f"Linear GEMM（{dtype}，QKV/O/MLP/共享专家/router）",
            sglang_entry="UnquantizedLinearMethod.apply → F.linear",
            status=("in_tree" if ctx.profile.in_tree else "vendor") if supported else "missing",
            resolved_path="F.linear（设备 ATen matmul）"
            if supported
            else "设备不支持该 dtype 的 GEMM",
            impact="GEMM 吃掉 prefill 的大部分时间；decode 小 batch 是权重读带宽受限",
            how="确认厂商 ATen matmul 对这些形状（M 很小的 decode GEMM）有高效 kernel",
            references=(source("unquantized_linear"),),
        )
    weight = quant.weight_dtype
    native_unit = hardware.supports_compute_dtype(weight) if weight in ("fp8", "int8") else True
    if weight == "fp8":
        vendor_ops: tuple[str, ...] = (
            ("fp8_blockwise_gemm", "per_token_group_quant_fp8")
            if quant.block_shape
            else ("fp8_gemm",)
        )
    elif weight == "int8":
        vendor_ops = ("int8_gemm",)
    else:
        vendor_ops = ("w4a16_gemm",)
    block = f"，block {list(quant.block_shape)}" if quant.block_shape else ""
    references = [source("oot_quantization_hook")]
    if weight == "int8":
        references.append(source("w8a8_int8_cuda_only"))
    if ctx.profile.in_tree:
        status, path = "in_tree", f"{quant.method} LinearMethod（内置 kernel）"
        impact = "内置实现"
    elif not native_unit:
        status, path = "missing", f"设备没有 {weight} 计算单元"
        impact = (
            f"checkpoint 是 {weight}，设备没有对应 GEMM：需要改走反量化/重量化路线（见精度部分）"
        )
    elif ctx.has(*vendor_ops):
        status, path = (
            "vendor",
            f"get_quantization_config({quant.method!r}) 返回的厂商 LinearMethod",
        )
        impact = "确认 per-token/per-group 激活量化与 GEMM 融合，避免单独的量化 kernel"
    else:
        status, path = "missing", "SGLang 的实现依赖 CUDA/Triton kernel"
        impact = f"{quant.method} 的 LinearMethod 在该平台没有 kernel"
    return OperatorRequirement(
        key="linear",
        category="gemm",
        title=f"Linear GEMM（{quant.method}：{weight} 权重{block}）",
        sglang_entry=f"{quant.method} LinearMethod.apply",
        status=status,
        resolved_path=path,
        impact=impact,
        how=(
            f"get_quantization_config({quant.method!r}) 返回厂商 QuantizationConfig，"
            "get_quant_method(LinearBase) 返回厂商 LinearMethod；post_load_model 里做权重重排"
        ),
        vendor_ops=vendor_ops,
        references=tuple(references),
    )


def _simple_requirements(ctx: _Context) -> list[OperatorRequirement]:
    arch, software, parallel = ctx.arch, ctx.software, ctx.parallel
    in_tree = ctx.profile.in_tree
    requirements = [
        OperatorRequirement(
            key="embedding",
            category="elementwise",
            title=f"词表 Embedding（vocab {arch.vocab_size}，按 TP 切分）",
            sglang_entry="VocabParallelEmbedding → F.embedding",
            status="in_tree" if in_tree else "vendor",
            resolved_path="F.embedding"
            + (" + get_masked_input_and_mask（torch.compile）" if parallel.tp > 1 else ""),
            impact=(
                "TP>1 时越界 token 先由 import 时编译的 get_masked_input_and_mask 置零，依赖编译后端"
                if parallel.tp > 1
                else "单卡时只是 F.embedding"
            ),
            how="确认 get_compile_backend() 能编译该函数，或返回 'eager'",
            references=(source("torch_compile_sites"), source("oot_compile_backend")),
        ),
        OperatorRequirement(
            key="lm_head",
            category="gemm",
            title=f"LM head（{arch.hidden_size} → {arch.vocab_size}，TP 切分后 all-gather logits）",
            sglang_entry="ParallelLMHead + LogitsProcessor",
            status="in_tree" if in_tree else "vendor",
            resolved_path="F.linear（未量化）",
            impact="decode 每步读一遍 LM head 权重；logits all-gather 的消息随 batch 线性增长",
            how="无需单独适配，确认 GEMM 效率即可",
            references=(source("unquantized_linear"),),
        ),
    ]
    sampler_vendor = ctx.has("sampling")
    requirements.append(
        OperatorRequirement(
            key="sampling",
            category="sampling",
            title="采样（temperature/top-p/top-k/min-p）",
            sglang_entry="Sampler（--sampling-backend）",
            status="in_tree" if in_tree else ("vendor" if sampler_vendor else "native"),
            resolved_path=(
                "flashinfer / pytorch"
                if in_tree
                else (
                    "register_sampler_backend 注册的厂商实现" if sampler_vendor else "pytorch 后端"
                )
            ),
            impact=(
                "pytorch 后端对整个词表做 softmax + 排序（top-p），大 batch 时每步多几毫秒"
                if not in_tree and not sampler_vendor
                else "每步一次"
            ),
            how="register_sampler_backend 注册厂商采样，并在 apply_server_args_defaults 里设为默认",
            vendor_ops=("sampling",),
            references=(source("sampler_backend"),),
        )
    )
    if parallel.tp > 1:
        collective = software.collective_library
        usable = bool(collective) and collective != "gloo"
        status = ("in_tree" if in_tree else "vendor") if usable else "missing"
        extra = []
        if not in_tree:
            extra.append(
                "custom allreduce 与 FlashInfer allreduce+RMSNorm 融合只在 CUDA/ROCm 上存在"
            )
        requirements.append(
            OperatorRequirement(
                key="collectives",
                category="communication",
                title=f"TP 集合通信（{parallel.describe()}：all-reduce / all-gather）",
                sglang_entry="GroupCoordinator（torch.distributed 后端）",
                status=status,
                resolved_path=f"torch.distributed 后端 {collective!r}" if usable else "gloo（CPU）",
                impact=(
                    "decode 每层两次小消息 all-reduce，延迟下限由通信库决定"
                    + ("；" + "；".join(extra) if extra else "")
                ),
                how="get_torch_distributed_backend_str 返回厂商后端名，并用 torch.distributed.Backend.register_backend 注册",
                vendor_ops=("custom_allreduce", "allreduce_rmsnorm_fusion"),
                references=(source("allreduce_dispatch"), source("platform_distributed_backend")),
            )
        )
    if arch.attention.sparse_indexer:
        status, path, note = ctx.fused_op_status("Indexer", "dsa_indexer")
        requirements.append(
            OperatorRequirement(
                key="dsa_indexer",
                category="attention",
                title="DSA indexer（逐 token 选 top-k KV）",
                sglang_entry="BaseFusedOp Indexer（layers/attention/dsa）",
                status=status if status != "native" else "missing",
                resolved_path=path,
                impact="稀疏注意力的 indexer 依赖 FP8 GEMM 与专用 kernel，forward_native 只用于参考",
                how=f"register_oot_forward(Indexer, fn, '{ctx.key}') 并提供稀疏注意力后端",
                op_classes=("Indexer",),
                vendor_ops=("dsa_indexer",),
                references=(source("oot_dispatch_order"),),
            )
        )
    return requirements


def operator_inventory(
    arch: ModelArch,
    hardware: HardwareSpec,
    profile: PlatformProfile,
    knowledge: SGLangKnowledge,
    parallel: ParallelConfig | None = None,
) -> OperatorInventory:
    """Every operator family ``arch`` runs, how SGLang resolves it on ``hardware`` and what is missing."""
    parallel = parallel or ParallelConfig(tp=1)
    ctx = _Context(arch, hardware, profile, knowledge, parallel)
    candidates: list[OperatorRequirement | None] = [
        _attention_requirement(ctx),
        _linear_requirement(ctx),
        _moe_requirement(ctx),
        _topk_requirement(ctx),
        _a2a_requirement(ctx),
        _norm_requirement(ctx),
        _rope_requirement(ctx),
        _activation_requirement(ctx),
        _kv_store_requirement(ctx),
        _positions_requirement(ctx),
        *_simple_requirements(ctx),
    ]
    decode, prefill = _cost_shares(arch, hardware, parallel)
    requirements = [
        replace(item, share_decode=decode.get(item.key), share_prefill=prefill.get(item.key))
        for item in candidates
        if item is not None
    ]
    requirements.sort(key=lambda item: (_STATUS_RANK[item.status], -item.max_share))
    class_names = {qualified.rsplit(".", 1)[-1].lower() for qualified in knowledge.fused_op_classes}
    unknown = [
        name
        for name in hardware.software.fused_ops
        if name.lower() not in KNOWN_FUSED_OPS and name.lower() not in class_names
    ]
    return OperatorInventory(
        requirements=requirements,
        unknown_vendor_ops=unknown,
        shares_note=(
            f"占比来自代价模型：decode batch {DECODE_SHAPE[0]} × 上下文 {DECODE_SHAPE[1]}，"
            f"prefill {PREFILL_SHAPE[0]} × {PREFILL_SHAPE[1]} token，{parallel.describe()}，"
            "按理想 kernel 估算（回落实现会更慢）"
        ),
    )
