"""Per-GPU operator cost model for one SGLang forward step.

Each layer is decomposed into the kernels SGLang launches (GEMMs, attention, MoE, norms,
elementwise, collectives) with FLOPs, HBM bytes and link bytes; time is the roofline
``max(compute, memory)`` per kernel plus launch/graph gaps, and an alpha-beta model for
collectives. Efficiency factors come from the hardware spec and can be calibrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from sglang_maestro.hardware.spec import HardwareSpec, bytes_per_element, normalize_dtype
from sglang_maestro.model.config import ModelArch
from sglang_maestro.model.parallel import ParallelConfig, kv_heads_per_rank, shared_expert_tp
from sglang_maestro.model.params import count_params, weight_precision

ACTIVATION_BYTES = 2.0


@dataclass(frozen=True)
class BatchShape:
    """Work in one forward step on one attention-DP rank."""

    num_seqs: int
    new_tokens: int
    sum_kv: float
    sum_q_kv: float
    logits_tokens: int
    phase: str
    prefix_tokens: float = 0.0

    @staticmethod
    def decode(batch_size: int, context_len: float) -> BatchShape:
        batch_size = max(int(batch_size), 1)
        total = batch_size * max(float(context_len), 1.0)
        return BatchShape(batch_size, batch_size, total, total, batch_size, "decode")

    @staticmethod
    def decode_total(batch_size: int, sum_context: float) -> BatchShape:
        batch_size = max(int(batch_size), 1)
        total = max(float(sum_context), float(batch_size))
        return BatchShape(batch_size, batch_size, total, total, batch_size, "decode")

    @staticmethod
    def verify(batch_size: int, context_len: float, tokens_per_seq: int) -> BatchShape:
        """Speculative target-verify step: ``tokens_per_seq`` query tokens per request."""
        batch_size = max(int(batch_size), 1)
        tokens = batch_size * tokens_per_seq
        total_kv = batch_size * (max(float(context_len), 1.0) + tokens_per_seq)
        return BatchShape(batch_size, tokens, total_kv, total_kv * tokens_per_seq, tokens, "decode")

    @staticmethod
    def prefill(num_seqs: int, new_tokens_per_seq: int, prefix_len: int = 0) -> BatchShape:
        num_seqs = max(int(num_seqs), 1)
        query = max(int(new_tokens_per_seq), 1)
        prefix = max(int(prefix_len), 0)
        pairs = query * prefix + query * (query + 1) / 2
        return BatchShape(
            num_seqs,
            num_seqs * query,
            float(num_seqs * (prefix + query)),
            float(num_seqs * pairs),
            num_seqs,
            "prefill",
            prefix_tokens=float(num_seqs * prefix),
        )

    @staticmethod
    def chunks(chunks: list[tuple[int, int]]) -> BatchShape:
        """A prefill batch made of ``(new_tokens, prefix_len)`` chunks."""
        if not chunks:
            raise ValueError("empty prefill batch")
        new_tokens = sum(max(q, 1) for q, _ in chunks)
        sum_kv = float(sum(max(q, 1) + max(p, 0) for q, p in chunks))
        pairs = float(
            sum(max(q, 1) * max(p, 0) + max(q, 1) * (max(q, 1) + 1) / 2 for q, p in chunks)
        )
        prefix = float(sum(max(p, 0) for _, p in chunks))
        return BatchShape(len(chunks), new_tokens, sum_kv, pairs, len(chunks), "prefill", prefix)

    def merged_with(self, other: BatchShape) -> BatchShape:
        return BatchShape(
            self.num_seqs + other.num_seqs,
            self.new_tokens + other.new_tokens,
            self.sum_kv + other.sum_kv,
            self.sum_q_kv + other.sum_q_kv,
            self.logits_tokens + other.logits_tokens,
            "mixed" if self.phase != other.phase else self.phase,
            self.prefix_tokens + other.prefix_tokens,
        )


@dataclass(frozen=True)
class OpCost:
    name: str
    kind: str
    flops: float = 0.0
    bytes: float = 0.0
    weight_bytes: float = 0.0
    compute_dtype: str = "bf16"
    comm_bytes: float = 0.0
    comm_pattern: str | None = None
    comm_group: int = 1
    kernels: int = 1
    layers: float = 1.0

    @property
    def arithmetic_intensity(self) -> float:
        return self.flops / self.bytes if self.bytes else math.inf


@dataclass(frozen=True)
class OpTiming:
    op: OpCost
    compute_us: float
    memory_us: float
    comm_us: float
    overhead_us: float
    time_us: float
    bound: str
    kernel_us: float = 0.0
    """GPU-side duration of the op's kernels for one layer, without launch gaps."""

    @property
    def total_us(self) -> float:
        return self.time_us * self.op.layers


@dataclass
class StepEstimate:
    shape: BatchShape
    timings: list[OpTiming]
    step_overhead_us: float
    scale: float = 1.0

    @property
    def kernel_us(self) -> float:
        return sum(item.total_us for item in self.timings) * self.scale

    @property
    def total_us(self) -> float:
        return self.kernel_us + self.step_overhead_us

    def by_kind(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for item in self.timings:
            totals[item.op.kind] = totals.get(item.op.kind, 0.0) + item.total_us * self.scale
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    def by_bound(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for item in self.timings:
            totals[item.bound] = totals.get(item.bound, 0.0) + item.total_us * self.scale
        return totals

    def top_ops(self, limit: int = 10) -> list[OpTiming]:
        return sorted(self.timings, key=lambda item: -item.total_us)[:limit]


@dataclass(frozen=True)
class ExecutionConfig:
    weight_dtype: str | None = None
    kv_dtype: str | None = None
    cuda_graph: bool = True
    fp8_dispatch: bool = True


@dataclass(frozen=True)
class Calibration:
    """Multiplicative corrections fitted from measurements (``sglang-maestro calibrate``)."""

    prefill_scale: float = 1.0
    decode_scale: float = 1.0
    step_overhead_us: float | None = None
    source: str = "default"


@dataclass
class StepCostModel:
    arch: ModelArch
    hardware: HardwareSpec
    parallel: ParallelConfig
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    calibration: Calibration = field(default_factory=Calibration)

    def __post_init__(self) -> None:
        arch = self.arch
        self._params = count_params(arch)
        self._precision = weight_precision(arch, self.execution.weight_dtype)
        activation = arch.activation_dtype
        quant = arch.quant
        if self.execution.weight_dtype:
            activation_quant = normalize_dtype(self.execution.weight_dtype)
            if activation_quant not in {"fp8", "int8"}:
                activation_quant = activation
        elif quant is not None and quant.activation_dtype:
            activation_quant = quant.activation_dtype
        else:
            activation_quant = activation
        self._linear_dtype = self.hardware.compute_dtype_for(
            self._precision.linear_dtype, activation_quant
        )
        self._expert_dtype = self.hardware.compute_dtype_for(
            self._precision.experts_dtype, activation_quant
        )
        if not self.hardware.supports_compute_dtype(activation):
            activation = "bf16" if self.hardware.supports_compute_dtype("bf16") else "fp16"
        self._activation_dtype = activation
        self._quantize_activations = self._linear_dtype in {"fp8", "int8"}
        self._kv_bytes = bytes_per_element(self.execution.kv_dtype or arch.torch_dtype)
        self._cache: dict[tuple, float] = {}

    @property
    def linear_compute_dtype(self) -> str:
        return self._linear_dtype

    @property
    def expert_compute_dtype(self) -> str:
        return self._expert_dtype

    # ------------------------------------------------------------------ op builders
    def _gemm(
        self,
        name: str,
        m: float,
        n: float,
        k: float,
        *,
        weight_bpp: float,
        dtype: str,
        layers: float,
        kind: str = "gemm",
        kernels: int = 1,
    ) -> OpCost:
        input_bytes = 1.0 if dtype in {"fp8", "int8"} else ACTIVATION_BYTES
        weight_bytes = n * k * weight_bpp
        return OpCost(
            name=name,
            kind=kind,
            flops=2.0 * m * n * k,
            bytes=weight_bytes + m * k * input_bytes + m * n * ACTIVATION_BYTES,
            weight_bytes=weight_bytes,
            compute_dtype=dtype,
            kernels=kernels,
            layers=layers,
        )

    @staticmethod
    def _elementwise(
        name: str,
        elements: float,
        bytes_per_element_moved: float,
        layers: float,
        kind: str = "elementwise",
        kernels: int = 1,
        flops_per_element: float = 4.0,
    ) -> OpCost:
        return OpCost(
            name=name,
            kind=kind,
            flops=elements * flops_per_element,
            bytes=elements * bytes_per_element_moved,
            kernels=kernels,
            layers=layers,
        )

    @staticmethod
    def _comm(
        name: str, payload_bytes: float, pattern: str, group: int, layers: float
    ) -> OpCost | None:
        if group <= 1 or payload_bytes <= 0:
            return None
        return OpCost(
            name=name,
            kind="comm",
            comm_bytes=payload_bytes,
            comm_pattern=pattern,
            comm_group=group,
            layers=layers,
            kernels=1,
        )

    def _quant_op(self, name: str, elements: float, layers: float) -> OpCost | None:
        if not self._quantize_activations:
            return None
        return self._elementwise(name, elements, ACTIVATION_BYTES + 1.0, layers, kind="quant")

    # ------------------------------------------------------------------ attention
    def _attention_ops(self, shape: BatchShape, layers: float) -> list[OpCost]:
        arch, parallel = self.arch, self.parallel
        attention = arch.attention
        hidden = arch.hidden_size
        attn_tp, dcp = parallel.attn_tp, parallel.dcp
        # prefill CP splits EXTEND/MIXED batches only; decode runs the whole batch on every CP rank.
        cp = parallel.attn_cp if shape.phase != "decode" else 1
        tokens = shape.new_tokens / cp
        heads = attention.num_heads / attn_tp
        linear_bpp = self._precision.linear
        ops: list[OpCost | None] = [
            self._elementwise(
                "attn.input_norm", tokens * hidden, 3 * ACTIVATION_BYTES, layers, kind="norm"
            ),
            self._quant_op("attn.input_quant", tokens * hidden, layers),
        ]
        kv_store_bytes_per_token: float
        if attention.is_mla:
            kv_lora = int(attention.kv_lora_rank or 0)
            rope = int(attention.qk_rope_head_dim or 0)
            nope = int(attention.qk_nope_head_dim or 0)
            v_dim = attention.v_head_dim
            q_lora = int(attention.q_lora_rank or 0)
            a_width = (q_lora if q_lora else 0) + kv_lora + rope
            ops.append(
                self._gemm(
                    "attn.qkv_a_proj",
                    tokens,
                    a_width,
                    hidden,
                    weight_bpp=linear_bpp,
                    dtype=self._linear_dtype,
                    layers=layers,
                )
            )
            ops.append(
                self._elementwise(
                    "attn.qkv_a_norm",
                    tokens * (q_lora + kv_lora),
                    3 * ACTIVATION_BYTES,
                    layers,
                    kind="norm",
                    kernels=2 if q_lora else 1,
                )
            )
            q_in = q_lora if q_lora else hidden
            ops.append(
                self._gemm(
                    "attn.q_b_proj",
                    tokens,
                    heads * (nope + rope),
                    q_in,
                    weight_bpp=linear_bpp,
                    dtype=self._linear_dtype,
                    layers=layers,
                )
            )
            ops.append(
                self._elementwise(
                    "attn.rope",
                    tokens * (heads + 1) * rope,
                    2 * ACTIVATION_BYTES,
                    layers,
                    kind="rope",
                )
            )
            kv_store_bytes_per_token = attention.mla_latent_dim * self._kv_bytes / dcp
            unquant = self._precision.unquantized
            if shape.phase == "decode":
                ops.append(
                    self._gemm(
                        "attn.w_kc_absorb",
                        tokens,
                        heads * kv_lora,
                        nope,
                        weight_bpp=unquant,
                        dtype=self._activation_dtype,
                        layers=layers,
                    )
                )
                ops.append(
                    self._elementwise(
                        "attn.kv_store",
                        tokens * attention.mla_latent_dim,
                        ACTIVATION_BYTES + self._kv_bytes,
                        layers,
                        kind="kv_store",
                    )
                )
                score_dim = 2 * kv_lora + rope
                ops.append(
                    OpCost(
                        name="attn.core_mla_decode",
                        kind="attention",
                        flops=2.0 * heads * score_dim * shape.sum_q_kv,
                        bytes=shape.sum_kv / dcp * attention.mla_latent_dim * self._kv_bytes
                        + tokens * heads * dcp * (kv_lora + rope) * ACTIVATION_BYTES * 2,
                        compute_dtype=self._activation_dtype,
                        layers=layers,
                    )
                )
                if dcp > 1:
                    ops.append(
                        self._comm(
                            "attn.dcp_partial_a2a",
                            tokens * heads * dcp * (kv_lora * ACTIVATION_BYTES + 4.0),
                            "alltoall",
                            dcp,
                            layers,
                        )
                    )
                ops.append(
                    self._gemm(
                        "attn.w_vc_absorb",
                        tokens,
                        heads * v_dim,
                        kv_lora,
                        weight_bpp=unquant,
                        dtype=self._activation_dtype,
                        layers=layers,
                    )
                )
            else:
                expanded_tokens = shape.sum_kv
                ops.append(
                    self._gemm(
                        "attn.kv_b_proj",
                        expanded_tokens,
                        heads * (nope + v_dim),
                        kv_lora,
                        weight_bpp=linear_bpp,
                        dtype=self._linear_dtype,
                        layers=layers,
                    )
                )
                ops.append(
                    self._elementwise(
                        "attn.kv_store",
                        tokens * attention.mla_latent_dim,
                        ACTIVATION_BYTES + self._kv_bytes,
                        layers,
                        kind="kv_store",
                    )
                )
                qk_dim = nope + rope
                pairs = shape.sum_q_kv / cp
                if attention.sparse_indexer:
                    pairs = min(pairs, tokens * 2048.0)
                ops.append(
                    OpCost(
                        name="attn.core_mla_prefill",
                        kind="attention",
                        flops=2.0 * heads * (qk_dim + v_dim) * pairs,
                        bytes=expanded_tokens * heads * (qk_dim + v_dim) * ACTIVATION_BYTES
                        + tokens * heads * (qk_dim + v_dim) * ACTIVATION_BYTES,
                        compute_dtype=self._activation_dtype,
                        layers=layers,
                    )
                )
            if attention.sparse_indexer and attention.index_n_heads and attention.index_head_dim:
                ops.append(
                    OpCost(
                        name="attn.dsa_indexer",
                        kind="attention",
                        flops=2.0
                        * attention.index_n_heads
                        * attention.index_head_dim
                        * shape.sum_q_kv
                        / cp,
                        bytes=shape.sum_kv * attention.index_head_dim * 1.0,
                        compute_dtype="fp8"
                        if self.hardware.supports_compute_dtype("fp8")
                        else self._activation_dtype,
                        layers=layers,
                    )
                )
            out_width = heads * v_dim
        else:
            head_dim = attention.head_dim
            kv_heads = kv_heads_per_rank(arch, attn_tp, dcp)
            qkv_width = (heads + 2 * kv_heads) * head_dim
            ops.append(
                self._gemm(
                    "attn.qkv_proj",
                    tokens,
                    qkv_width,
                    hidden,
                    weight_bpp=linear_bpp,
                    dtype=self._linear_dtype,
                    layers=layers,
                )
            )
            if attention.has_qk_norm:
                ops.append(
                    self._elementwise(
                        "attn.qk_norm",
                        tokens * (heads + kv_heads) * head_dim,
                        2 * ACTIVATION_BYTES,
                        layers,
                        kind="norm",
                        kernels=2,
                    )
                )
            ops.append(
                self._elementwise(
                    "attn.rope",
                    tokens * (heads + kv_heads) * head_dim,
                    2 * ACTIVATION_BYTES,
                    layers,
                    kind="rope",
                )
            )
            kv_store_bytes_per_token = 2 * kv_heads * head_dim * self._kv_bytes / dcp
            ops.append(
                self._elementwise(
                    "attn.kv_store",
                    tokens * 2 * kv_heads * head_dim / dcp,
                    ACTIVATION_BYTES + self._kv_bytes,
                    layers,
                    kind="kv_store",
                )
            )
            full_layers = layers * (1 - attention.num_sliding_layers / arch.num_layers)
            sliding_layers = layers - full_layers
            for label, layer_count, window in (
                ("full", full_layers, None),
                ("sliding", sliding_layers, attention.sliding_window),
            ):
                if layer_count <= 0:
                    continue
                sum_kv, pairs = shape.sum_kv, shape.sum_q_kv / cp
                if window:
                    sum_kv = min(sum_kv, shape.num_seqs * float(window))
                    pairs = min(pairs, tokens * float(window))
                # under DCP each rank attends dcp x the query heads over 1/dcp of the tokens.
                ops.append(
                    OpCost(
                        name=f"attn.core_{shape.phase}_{label}",
                        kind="attention",
                        flops=2.0 * heads * 2 * head_dim * pairs,
                        bytes=sum_kv / dcp * 2 * kv_heads * head_dim * self._kv_bytes
                        + tokens * heads * dcp * head_dim * ACTIVATION_BYTES * 2,
                        compute_dtype=self._activation_dtype,
                        layers=layer_count,
                    )
                )
            if dcp > 1 and shape.phase == "decode":
                ops.append(
                    self._comm(
                        "attn.dcp_partial_a2a",
                        tokens * heads * dcp * (head_dim * ACTIVATION_BYTES + 4.0),
                        "alltoall",
                        dcp,
                        layers,
                    )
                )
            out_width = heads * head_dim
        if cp > 1:
            ops.append(
                self._comm(
                    "attn.cp_kv_allgather",
                    shape.new_tokens * kv_store_bytes_per_token,
                    "allgather",
                    cp,
                    layers,
                )
            )
        ops.append(self._quant_op("attn.o_quant", tokens * out_width, layers))
        ops.append(
            self._gemm(
                "attn.o_proj",
                tokens,
                hidden,
                out_width,
                weight_bpp=linear_bpp,
                dtype=self._linear_dtype,
                layers=layers,
            )
        )
        ops.append(
            self._comm(
                "attn.tp_allreduce",
                tokens * hidden * ACTIVATION_BYTES,
                "allreduce",
                attn_tp,
                layers,
            )
        )
        ops.append(
            self._elementwise(
                "attn.post_norm", tokens * hidden, 3 * ACTIVATION_BYTES, layers, kind="norm"
            )
        )
        return [op for op in ops if op is not None]

    # ------------------------------------------------------------------ FFN
    def _gathered_tokens(self, shape: BatchShape) -> float:
        return shape.new_tokens * self.parallel.attn_dp

    def _cp(self, shape: BatchShape) -> int:
        return self.parallel.attn_cp if shape.phase != "decode" else 1

    def _dense_mlp_ops(self, shape: BatchShape, layers: float) -> list[OpCost]:
        arch, parallel = self.arch, self.parallel
        hidden = arch.hidden_size
        mlp_tp = parallel.effective_dense_mlp_tp
        cp = self._cp(shape)
        group = parallel.attn_dp * cp
        regrouped = group > 1 and mlp_tp > parallel.attn_tp
        tokens = self._gathered_tokens(shape) if regrouped else shape.new_tokens / cp
        intermediate = arch.dense_intermediate_size / mlp_tp
        matrices_up = 2 if arch.gated_mlp else 1
        ops: list[OpCost | None] = []
        if regrouped:
            ops.append(
                self._comm(
                    "mlp.dp_allgather",
                    tokens * hidden * ACTIVATION_BYTES,
                    "allgather",
                    group,
                    layers,
                )
            )
        ops += [
            self._quant_op("mlp.input_quant", tokens * hidden, layers),
            self._gemm(
                "mlp.gate_up_proj",
                tokens,
                matrices_up * intermediate,
                hidden,
                weight_bpp=self._precision.linear,
                dtype=self._linear_dtype,
                layers=layers,
            ),
            self._elementwise(
                "mlp.act_and_mul",
                tokens * intermediate,
                3 * ACTIVATION_BYTES,
                layers,
                kind="activation",
            ),
            self._quant_op("mlp.down_quant", tokens * intermediate, layers),
            self._gemm(
                "mlp.down_proj",
                tokens,
                hidden,
                intermediate,
                weight_bpp=self._precision.linear,
                dtype=self._linear_dtype,
                layers=layers,
            ),
        ]
        if regrouped:
            ops.append(
                self._comm(
                    "mlp.reduce_scatter",
                    tokens * hidden * ACTIVATION_BYTES,
                    "reducescatter",
                    mlp_tp,
                    layers,
                )
            )
        else:
            ops.append(
                self._comm(
                    "mlp.tp_allreduce",
                    tokens * hidden * ACTIVATION_BYTES,
                    "allreduce",
                    mlp_tp,
                    layers,
                )
            )
        return [op for op in ops if op is not None]

    def _moe_ops(self, shape: BatchShape, layers: float) -> list[OpCost]:
        arch, parallel = self.arch, self.parallel
        moe = arch.moe
        assert moe is not None
        hidden = arch.hidden_size
        matrices = 3 if arch.gated_mlp else 2
        group_tokens = self._gathered_tokens(shape)
        top_k, experts = moe.top_k, moe.num_experts
        ops: list[OpCost | None] = []
        expert_bpp = self._precision.experts
        hit_probability = 1.0 - (1.0 - top_k / experts) ** max(group_tokens, 0.0)
        uses_a2a = parallel.uses_a2a
        dp_group = parallel.attn_dp * self._cp(shape)
        # a2a backends route each rank's own tokens; otherwise every rank sees the gathered batch.
        local_tokens = group_tokens / parallel.tp if uses_a2a else group_tokens
        assignments = group_tokens * top_k / parallel.ep
        intermediate = moe.expert_intermediate_size / parallel.moe_tp
        activated = experts / parallel.ep * hit_probability
        ops.append(
            self._gemm(
                "moe.router",
                local_tokens,
                experts,
                hidden,
                weight_bpp=self._precision.unquantized,
                dtype=self._activation_dtype,
                layers=layers,
            )
        )
        ops.append(
            self._elementwise("moe.topk", local_tokens * experts, 8.0, layers, kind="moe_routing")
        )
        ops.append(self._quant_op("moe.input_quant", local_tokens * hidden, layers))
        if uses_a2a:
            dispatch_bytes = (
                1.0
                if self.execution.fp8_dispatch and self._quantize_activations
                else ACTIVATION_BYTES
            )
            ops.append(
                self._comm(
                    "moe.dispatch_a2a",
                    local_tokens * top_k * hidden * dispatch_bytes,
                    "alltoall",
                    parallel.ep,
                    layers,
                )
            )
        else:
            if dp_group > 1:
                ops.append(
                    self._comm(
                        "moe.dp_allgather",
                        group_tokens * hidden * ACTIVATION_BYTES,
                        "allgather",
                        dp_group,
                        layers,
                    )
                )
            ops.append(
                self._elementwise(
                    "moe.align", local_tokens * top_k, 16.0, layers, kind="moe_routing"
                )
            )
        expert_weight_bytes = activated * matrices * hidden * intermediate * expert_bpp
        expert_flops = 2.0 * assignments * matrices * hidden * intermediate
        ops.append(
            OpCost(
                name="moe.experts_grouped_gemm",
                kind="moe",
                flops=expert_flops,
                bytes=expert_weight_bytes
                + assignments * (2 * hidden + (matrices + 1) * intermediate) * ACTIVATION_BYTES,
                weight_bytes=expert_weight_bytes,
                compute_dtype=self._expert_dtype,
                kernels=3,
                layers=layers,
            )
        )
        if moe.shared_intermediate_size:
            shared_tp = shared_expert_tp(parallel)
            shared_tokens = local_tokens
            shared_intermediate = moe.shared_intermediate_size / shared_tp
            ops.append(
                self._gemm(
                    "moe.shared_gate_up",
                    shared_tokens,
                    (matrices - 1) * shared_intermediate,
                    hidden,
                    weight_bpp=self._precision.linear,
                    dtype=self._linear_dtype,
                    layers=layers,
                )
            )
            ops.append(
                self._elementwise(
                    "moe.shared_act",
                    shared_tokens * shared_intermediate,
                    3 * ACTIVATION_BYTES,
                    layers,
                    kind="activation",
                )
            )
            ops.append(
                self._gemm(
                    "moe.shared_down",
                    shared_tokens,
                    hidden,
                    shared_intermediate,
                    weight_bpp=self._precision.linear,
                    dtype=self._linear_dtype,
                    layers=layers,
                )
            )
        if uses_a2a:
            ops.append(
                self._comm(
                    "moe.combine_a2a",
                    local_tokens * top_k * hidden * ACTIVATION_BYTES,
                    "alltoall",
                    parallel.ep,
                    layers,
                )
            )
        else:
            ops.append(
                self._elementwise(
                    "moe.sum",
                    local_tokens * top_k * hidden,
                    ACTIVATION_BYTES,
                    layers,
                    kind="moe_routing",
                )
            )
            if dp_group > 1:
                ops.append(
                    self._comm(
                        "moe.reduce_scatter",
                        group_tokens * hidden * ACTIVATION_BYTES,
                        "reducescatter",
                        parallel.tp,
                        layers,
                    )
                )
            else:
                ops.append(
                    self._comm(
                        "moe.tp_allreduce",
                        local_tokens * hidden * ACTIVATION_BYTES,
                        "allreduce",
                        parallel.tp,
                        layers,
                    )
                )
        return [op for op in ops if op is not None]

    def _head_ops(self, shape: BatchShape) -> list[OpCost]:
        arch, parallel = self.arch, self.parallel
        hidden = arch.hidden_size
        vocab_local = arch.vocab_size / parallel.tp
        logits_tokens = shape.logits_tokens * parallel.attn_dp
        tokens = shape.new_tokens / self._cp(shape)
        ops: list[OpCost | None] = [
            self._elementwise(
                "embed.lookup", tokens * hidden, 2 * ACTIVATION_BYTES, 1.0, kind="elementwise"
            ),
            self._comm(
                "embed.allreduce",
                tokens * hidden * ACTIVATION_BYTES,
                "allreduce",
                parallel.attn_tp,
                1.0,
            ),
            self._elementwise(
                "final_norm", tokens * hidden, 3 * ACTIVATION_BYTES, 1.0, kind="norm"
            ),
            self._gemm(
                "lm_head",
                logits_tokens,
                vocab_local,
                hidden,
                weight_bpp=self._precision.unquantized,
                dtype=self._activation_dtype,
                layers=1.0,
            ),
            self._comm(
                "lm_head.allgather",
                logits_tokens * arch.vocab_size * ACTIVATION_BYTES,
                "allgather",
                parallel.tp,
                1.0,
            ),
            self._elementwise(
                "sampling",
                shape.logits_tokens * arch.vocab_size,
                8.0,
                1.0,
                kind="sampling",
                kernels=3,
            ),
        ]
        return [op for op in ops if op is not None]

    # ------------------------------------------------------------------ public API
    def ops(self, shape: BatchShape) -> list[OpCost]:
        arch = self.arch
        layers_scale = 1.0 / self.parallel.pp
        dense_layers = arch.num_dense_layers * layers_scale
        moe_layers = arch.num_moe_layers * layers_scale
        all_layers = arch.num_layers * layers_scale
        result = self._attention_ops(shape, all_layers)
        if dense_layers > 0:
            result += self._dense_mlp_ops(shape, dense_layers)
        if moe_layers > 0:
            result += self._moe_ops(shape, moe_layers)
        result += self._head_ops(shape)
        return result

    def head_ops(self, shape: BatchShape) -> list[OpCost]:
        """Ops that run once per step: embedding, final norm, LM head and sampling."""
        return self._head_ops(shape)

    def layer_ops(self, shape: BatchShape, moe_layer: bool | None = None) -> list[OpCost]:
        """Ops of a single decoder layer (used for MTP draft cost and weight-prefetch analysis)."""
        use_moe = self.arch.is_moe if moe_layer is None else moe_layer and self.arch.is_moe
        result = self._attention_ops(shape, 1.0)
        result += self._moe_ops(shape, 1.0) if use_moe else self._dense_mlp_ops(shape, 1.0)
        return result

    def time_op(self, op: OpCost, cuda_graph: bool | None = None) -> OpTiming:
        hardware = self.hardware
        latency, efficiency = hardware.latency, hardware.efficiency
        use_graph = self.execution.cuda_graph if cuda_graph is None else cuda_graph
        if op.kind == "comm":
            group = op.comm_group
            if group <= 1 or op.comm_bytes <= 0:
                return OpTiming(op, 0.0, 0.0, 0.0, 0.0, 0.0, "comm")
            factor = (
                2.0 * (group - 1) / group if op.comm_pattern == "allreduce" else (group - 1) / group
            )
            moved = op.comm_bytes * factor
            per_node = hardware.devices_per_node
            intra_bw = hardware.intra_node_bandwidth_gbps * 1e9 * efficiency.intra_node_comm
            if group > per_node and hardware.inter_node_bandwidth_gbps > 0:
                inter_bw = hardware.inter_node_bandwidth_gbps * 1e9 * efficiency.inter_node_comm
                intra_share = (per_node - 1) / (group - 1)
                transfer_s = max(
                    moved * intra_share / intra_bw, moved * (1 - intra_share) / inter_bw
                )
                base = latency.inter_node_collective_base_us
            else:
                transfer_s = moved / intra_bw
                base = latency.collective_base_us
            comm_us = base + transfer_s * 1e6
            return OpTiming(op, 0.0, 0.0, comm_us, 0.0, comm_us, "comm", kernel_us=comm_us)
        dtype = op.compute_dtype if hardware.supports_compute_dtype(op.compute_dtype) else "bf16"
        compute_efficiency = (
            efficiency.attention_compute if op.kind == "attention" else efficiency.gemm_compute
        )
        compute_us = op.flops / (hardware.peak_flops(dtype) * compute_efficiency) * 1e6
        memory_us = (
            op.bytes / (hardware.memory_bandwidth_bytes_per_s * efficiency.memory_bandwidth) * 1e6
        )
        roofline_us = max(compute_us, memory_us)
        kernels = max(op.kernels, 1)
        kernel_us = max(roofline_us, kernels * latency.min_kernel_us)
        if use_graph:
            time_us = kernel_us + kernels * latency.graph_kernel_gap_us
        else:
            time_us = kernels * max(
                kernel_us / kernels + latency.graph_kernel_gap_us, latency.kernel_launch_us
            )
        overhead_us = time_us - roofline_us
        if overhead_us > roofline_us:
            bound = "launch"
        else:
            bound = "compute" if compute_us >= memory_us else "memory"
        return OpTiming(
            op, compute_us, memory_us, 0.0, overhead_us, time_us, bound, kernel_us=kernel_us
        )

    def _scale_for(self, shape: BatchShape) -> float:
        if shape.phase == "decode":
            return self.calibration.decode_scale
        if shape.phase == "prefill":
            return self.calibration.prefill_scale
        return 0.5 * (self.calibration.decode_scale + self.calibration.prefill_scale)

    def estimate(self, shape: BatchShape, cuda_graph: bool | None = None) -> StepEstimate:
        if cuda_graph is None:
            cuda_graph = self.execution.cuda_graph and shape.phase == "decode"
        timings = [self.time_op(op, cuda_graph) for op in self.ops(shape)]
        overhead = self.calibration.step_overhead_us
        if overhead is None:
            overhead = self.hardware.latency.step_overhead_us
        return StepEstimate(shape, timings, overhead, self._scale_for(shape))

    def step_time_us(self, shape: BatchShape) -> float:
        """Memoized total step time; shapes are bucketed to 1% so simulators hit the cache."""
        key = (
            shape.phase,
            shape.num_seqs,
            shape.new_tokens,
            _bucket(shape.sum_kv),
            _bucket(shape.sum_q_kv),
            shape.logits_tokens,
        )
        cached = self._cache.get(key)
        if cached is None:
            cached = self.estimate(shape).total_us
            if len(self._cache) > 200_000:
                self._cache.clear()
            self._cache[key] = cached
        return cached

    def draft_step_time_us(self, batch_size: int, context_len: float) -> float:
        """One MTP/NextN draft forward: a single decoder layer plus the shared LM head."""
        shape = BatchShape.decode(batch_size, context_len)
        ops = self.layer_ops(shape) + self._head_ops(shape)
        cuda_graph = self.execution.cuda_graph
        return (
            sum(self.time_op(op, cuda_graph).total_us for op in ops) * self.calibration.decode_scale
        )


def _bucket(value: float) -> int:
    if value <= 64:
        return int(value)
    exponent = math.floor(math.log(value, 1.01))
    return exponent + 1_000_000
