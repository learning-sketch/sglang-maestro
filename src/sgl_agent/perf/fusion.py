"""Fusion opportunities in a trace: neighbouring kernels that SGLang can (or could) run as one.

The representative step of every primary-rank shape group (the step closest to the group median)
is scanned per stream in launch order. A pattern names the kernel that survives the fusion and
the kernels it removes. Patterns are tried in priority order and a kernel is removed at most once,
so the savings of different findings add up; survivors may be shared (add → RMSNorm → quant yields
an add+norm and a norm+quant match on the same norm kernel). Any other event on the stream between
two kernels (a kernel of another kind, a memcpy, a memset) breaks the adjacency.

Saving per removed kernel: a kernel at the latency floor (≤ ``LATENCY_REGIME_FACTOR`` × the
device's minimum kernel time) disappears; a longer one saves the share of its memory traffic the
fused kernel avoids (``FusionPattern.traffic_saved``). Idle time between the matched kernels on the
stream disappears as well, capped per gap by ``gap_credit_cap_us``. The numbers rank the
opportunities; an A/B trace confirms them.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

from sgl_agent.findings import Finding, severity_for_share
from sgl_agent.hardware.spec import LatencyAssumptions
from sgl_agent.knowledge.facts import (
    FLASHINFER_ALLREDUCE_FUSION_MAX_TOKENS,
    FLASHINFER_ALLREDUCE_FUSION_SM_MAJORS,
    FUSED_INPLACE_QKNORM_HEAD_DIMS,
    FUSED_MOE_SUM_ALLREDUCE_MIN_TOP_K,
    FUSED_QK_NORM_ROPE_HEAD_DIMS,
    TORCH_COMPILE_MAX_BS_DEFAULT,
    source,
)
from sgl_agent.perf.anomalies import is_graph_candidate_eager
from sgl_agent.perf.context import AnalysisContext
from sgl_agent.perf.metrics import GroupSummary
from sgl_agent.perf.trace import KernelEvent, StepSpan, short_kernel_name

LATENCY_REGIME_FACTOR = 2.5
MAX_GAP_CREDIT_US = 30.0
FUSION_MIN_SAVED_US = 10.0
FUSION_MIN_SHARE = 0.005
CHAIN_MIN_KERNELS = 3
CHAIN_CATEGORIES = frozenset({"elementwise", "copy"})
MAX_EVIDENCE_GROUPS = 3
COMPOSITION_NAME_LENGTH = 56

_ADD_NAME = re.compile(
    r"CUDAFunctor_add|AddFunctor|functor_add|(?<![A-Za-z])add_kernel|triton_poi_fused_add_\d+\b",
    re.IGNORECASE,
)
_MUL_NAME = re.compile(
    r"MulFunctor|functor_mul|(?<![A-Za-z])mul_kernel|triton_poi_fused_mul_\d+\b", re.IGNORECASE
)
_QK_NORM_NAME = re.compile(r"qk_?norm", re.IGNORECASE)


# ------------------------------------------------------------------------- kernel predicates
def _is_add(kernel: KernelEvent) -> bool:
    return kernel.category == "elementwise" and _ADD_NAME.search(kernel.name) is not None


def _is_mul(kernel: KernelEvent) -> bool:
    return kernel.category == "elementwise" and _MUL_NAME.search(kernel.name) is not None


def _is_plain_norm(kernel: KernelEvent) -> bool:
    return kernel.category == "norm" and not kernel.fused


def _is_plain_rmsnorm(kernel: KernelEvent) -> bool:
    return (
        _is_plain_norm(kernel)
        and "rms" in kernel.name.lower()
        and _QK_NORM_NAME.search(kernel.name) is None
    )


def _is_fused_add_rmsnorm(kernel: KernelEvent) -> bool:
    return kernel.category == "norm" and kernel.fused == ("add", "rmsnorm")


def _is_plain_activation(kernel: KernelEvent) -> bool:
    return kernel.category == "activation" and not kernel.fused and "mul" not in kernel.name.lower()


def _is_act_and_mul(kernel: KernelEvent) -> bool:
    return (
        kernel.category == "activation"
        and "act" in kernel.fused
        and "mul" in kernel.fused
        and "quant" not in kernel.fused
    )


def _is_quant(kernel: KernelEvent) -> bool:
    return kernel.category == "quant"


def _is_plain_rope(kernel: KernelEvent) -> bool:
    return kernel.category == "rope" and not kernel.fused


def _is_kv_store(kernel: KernelEvent) -> bool:
    return kernel.category == "kv_store"


def _is_unfused_allreduce(kernel: KernelEvent) -> bool:
    return kernel.category == "comm" and kernel.comm_pattern == "allreduce" and not kernel.fused


def _is_triton_moe(kernel: KernelEvent) -> bool:
    return kernel.category == "moe" and "fused_moe" in kernel.name.lower()


def _is_moe_sum(kernel: KernelEvent) -> bool:
    return kernel.category == "moe_routing" and "moe_sum" in kernel.name.lower()


def _is_chain_member(kernel: KernelEvent) -> bool:
    return kernel.kind == "kernel" and kernel.category in CHAIN_CATEGORIES


def quant_flavor(name: str) -> str:
    """``group`` / ``per_token`` / ``per_tensor`` / ``unknown`` from a quantization kernel name."""
    lowered = name.lower()
    if "group" in lowered:
        return "group"
    if "per_token" in lowered:
        return "per_token"
    if "per_tensor" in lowered or "static" in lowered or "scaled_fp8_quant" in lowered:
        return "per_tensor"
    return "unknown"


def compact_kernel_name(name: str) -> str:
    """``short_kernel_name`` without the namespace, e.g. ``vectorized_elementwise_kernel<MulFunctor>``."""
    short = short_kernel_name(name, max_length=1024)
    head, bracket, template = short.partition("<")
    compact = head.rsplit("::", 1)[-1] + bracket + template
    if len(compact) <= COMPOSITION_NAME_LENGTH:
        return compact
    return compact[: COMPOSITION_NAME_LENGTH - 1] + "…"


def _activation_kind(name: str) -> str:
    lowered = name.lower()
    if "silu" in lowered or "swiglu" in lowered:
        return "silu"
    if "gelu" in lowered:
        return "gelu"
    return "other"


# ---------------------------------------------------------------------------------- matching
@dataclass
class ScanState:
    """Per-stream scan state plus the model facts that gate some patterns."""

    allreduce_fusion_applies: bool
    qk_norm_model: bool | None
    moe_sum_fusable: bool | None
    removed: set[int] = field(default_factory=set)

    def free(self, *indices: int) -> bool:
        return not any(index in self.removed for index in indices)


MatchResult = tuple[int, tuple[int, ...], str | None]
"""``(survivor index, removed indices, detail)``."""


def _match_allreduce_add_norm(
    events: list[KernelEvent], position: int, state: ScanState
) -> MatchResult | None:
    if not state.allreduce_fusion_applies or not _is_unfused_allreduce(events[position]):
        return None
    following = position + 1
    if following >= len(events) or not state.free(position, following):
        return None
    if _is_fused_add_rmsnorm(events[following]):
        return position, (following,), "fused_add_rmsnorm"
    after = following + 1
    if (
        after < len(events)
        and _is_add(events[following])
        and _is_plain_rmsnorm(events[after])
        and state.free(after)
    ):
        return position, (following, after), "add_rmsnorm"
    return None


def _match_qk_norm_rope(
    events: list[KernelEvent], position: int, state: ScanState
) -> MatchResult | None:
    first = events[position]
    if not _is_plain_norm(first) or not state.free(position) or state.qk_norm_model is False:
        return None
    named_qk = _QK_NORM_NAME.search(first.name) is not None
    if state.qk_norm_model is None and not named_qk:
        return None
    norms = [position]
    following = position + 1
    if following < len(events) and _is_plain_norm(events[following]) and state.free(following):
        norms.append(following)
        following += 1
    if following < len(events) and _is_plain_rope(events[following]) and state.free(following):
        detail = "inplace_qknorm" if len(norms) == 1 and named_qk else "separate_norms"
        return following, tuple(norms), detail
    return None


def _match_rope_kv_store(
    events: list[KernelEvent], position: int, state: ScanState
) -> MatchResult | None:
    following = position + 1
    if (
        following < len(events)
        and _is_plain_rope(events[position])
        and _is_kv_store(events[following])
        and state.free(position, following)
    ):
        return position, (following,), None
    return None


def _match_add_norm(
    events: list[KernelEvent], position: int, state: ScanState
) -> MatchResult | None:
    following = position + 1
    if (
        following < len(events)
        and _is_add(events[position])
        and _is_plain_rmsnorm(events[following])
        and state.free(position, following)
    ):
        return following, (position,), None
    return None


def _match_silu_mul(
    events: list[KernelEvent], position: int, state: ScanState
) -> MatchResult | None:
    following = position + 1
    if (
        following < len(events)
        and _is_plain_activation(events[position])
        and _is_mul(events[following])
        and state.free(position, following)
    ):
        return following, (position,), _activation_kind(events[position].name)
    return None


def _match_act_quant(
    events: list[KernelEvent], position: int, state: ScanState
) -> MatchResult | None:
    following = position + 1
    if (
        following >= len(events)
        or not _is_act_and_mul(events[position])
        or not _is_quant(events[following])
        or not state.free(position, following)
    ):
        return None
    neighbours = [
        events[index] for index in (position - 1, following + 1) if 0 <= index < len(events)
    ]
    scope = "moe" if any(kernel.category == "moe" for kernel in neighbours) else "dense"
    activation = _activation_kind(events[position].name)
    return position, (following,), f"{scope}|{quant_flavor(events[following].name)}|{activation}"


def _match_norm_quant(
    events: list[KernelEvent], position: int, state: ScanState
) -> MatchResult | None:
    following = position + 1
    norm = events[position]
    if (
        following < len(events)
        and (_is_plain_rmsnorm(norm) or _is_fused_add_rmsnorm(norm))
        and _is_quant(events[following])
        and state.free(position, following)
    ):
        return position, (following,), quant_flavor(events[following].name)
    return None


def _match_moe_sum(
    events: list[KernelEvent], position: int, state: ScanState
) -> MatchResult | None:
    if state.moe_sum_fusable is False:
        return None
    following = position + 1
    if (
        following < len(events)
        and _is_triton_moe(events[position])
        and _is_moe_sum(events[following])
        and state.free(position, following)
    ):
        return position, (following,), None
    return None


def _match_elementwise_chain(
    events: list[KernelEvent], position: int, state: ScanState
) -> MatchResult | None:
    if not _is_chain_member(events[position]) or not state.free(position):
        return None
    end = position
    while end + 1 < len(events) and _is_chain_member(events[end + 1]) and state.free(end + 1):
        end += 1
    if end - position + 1 < CHAIN_MIN_KERNELS:
        return None
    members = range(position, end + 1)
    survivor = max(members, key=lambda index: events[index].duration_us)
    return survivor, tuple(index for index in members if index != survivor), None


@dataclass(frozen=True)
class FusionPattern:
    key: str
    title: str
    kind: str
    traffic_saved: float
    """Share of a removed (bandwidth-bound) kernel's time the fused kernel avoids."""
    matcher: Callable[[list[KernelEvent], int, ScanState], MatchResult | None]


PATTERNS: tuple[FusionPattern, ...] = (
    # allreduce out (1) + add (3) + norm (2) -> fused AR reads residual, writes residual and output (3).
    FusionPattern(
        "allreduce_add_norm",
        "allreduce 之后的残差 add + RMSNorm 没有与通信融合",
        "comm_fusion",
        0.6,
        _match_allreduce_add_norm,
    ),
    # Both norms read and write q/k once more than the rope that follows.
    FusionPattern(
        "qk_norm_rope", "QK norm 与 RoPE 是分开的 kernel", "fusion", 1.0, _match_qk_norm_rope
    ),
    # The fused rope writes k straight to the cache: the separate store re-reads k (half its traffic).
    FusionPattern("rope_kv_store", "RoPE 之后单独写 KV cache", "fusion", 0.5, _match_rope_kv_store),
    # add (3) + norm (2) -> fused_add_rmsnorm (4): one of the add's three passes.
    FusionPattern(
        "add_norm", "残差 add 与 RMSNorm 是分开的 kernel", "fusion", 1 / 3, _match_add_norm
    ),
    FusionPattern("silu_mul", "激活与逐元素乘是分开的 kernel", "fusion", 1.0, _match_silu_mul),
    FusionPattern("act_quant", "act_and_mul 之后单独做激活量化", "fusion", 1.0, _match_act_quant),
    FusionPattern("norm_quant", "RMSNorm 之后单独做激活量化", "fusion", 1.0, _match_norm_quant),
    FusionPattern("moe_sum", "MoE down GEMM 之后单独做 top-k 求和", "fusion", 1.0, _match_moe_sum),
    FusionPattern(
        "elementwise_chain",
        "连续的逐元素/拷贝 kernel 链",
        "fusion",
        1.0,
        _match_elementwise_chain,
    ),
)
PATTERN_BY_KEY = {pattern.key: pattern for pattern in PATTERNS}


@dataclass(frozen=True)
class FusionMatch:
    pattern: str
    survivor: int
    removed: tuple[int, ...]
    detail: str | None
    removed_us: float
    saved_us: float
    composition: str
    preceding: str | None


def _saved_us(
    events: list[KernelEvent],
    members: list[int],
    removed: tuple[int, ...],
    factor: float,
    floor_us: float,
    gap_cap_us: float,
) -> tuple[float, float]:
    removed_us = saved = 0.0
    for index in removed:
        duration = events[index].duration_us
        removed_us += duration
        saved += duration if duration <= LATENCY_REGIME_FACTOR * floor_us else duration * factor
    for previous, following in pairwise(members):
        gap = events[following].start_us - events[previous].end_us
        if gap > 0:
            saved += min(gap, gap_cap_us)
    return removed_us, saved


def match_stream(
    events: list[KernelEvent],
    state: ScanState,
    floor_us: float,
    gap_cap_us: float = MAX_GAP_CREDIT_US,
) -> list[FusionMatch]:
    """All pattern matches on one stream's events (sorted by start time)."""
    matches: list[FusionMatch] = []
    for pattern in PATTERNS:
        position = 0
        while position < len(events):
            result = pattern.matcher(events, position, state)
            if result is None:
                position += 1
                continue
            survivor, removed, detail = result
            members = sorted((survivor, *removed))
            state.removed.update(removed)
            removed_us, saved = _saved_us(
                events, members, removed, pattern.traffic_saved, floor_us, gap_cap_us
            )
            matches.append(
                FusionMatch(
                    pattern=pattern.key,
                    survivor=survivor,
                    removed=removed,
                    detail=detail,
                    removed_us=removed_us,
                    saved_us=saved,
                    composition=" → ".join(
                        compact_kernel_name(events[index].name) for index in members
                    ),
                    preceding=(
                        compact_kernel_name(events[members[0] - 1].name) if members[0] > 0 else None
                    ),
                )
            )
            position = members[-1] + 1
    return matches


# ------------------------------------------------------------------------------- per group
@dataclass
class PatternTotals:
    count: int = 0
    removed_us: float = 0.0
    saved_us: float = 0.0
    compositions: Counter[str] = field(default_factory=Counter)
    details: Counter[str] = field(default_factory=Counter)
    preceding: Counter[str] = field(default_factory=Counter)

    def add(self, match: FusionMatch) -> None:
        self.count += 1
        self.removed_us += match.removed_us
        self.saved_us += match.saved_us
        self.compositions[match.composition] += 1
        if match.detail:
            self.details[match.detail] += 1
        if match.preceding:
            self.preceding[match.preceding] += 1


@dataclass
class GroupFusion:
    group: GroupSummary
    step: StepSpan
    tokens: int | None
    allreduce_fusion_applies: bool
    gap_cap_us: float
    by_pattern: dict[str, PatternTotals]


def gap_credit_cap_us(context: AnalysisContext, group: GroupSummary) -> float:
    """Idle time a removed launch may save between fused kernels.

    In an eager step the gaps are launch and dispatch overhead. When the step could run as a CUDA
    graph that overhead belongs to the eager-step finding, so fusion is credited only with the
    graph-mode gap; large eager prefills keep the full cap.
    """
    hardware = context.hardware
    if is_graph_candidate_eager(group) and (hardware is None or hardware.software.graph_capture):
        return (
            hardware.latency.graph_kernel_gap_us
            if hardware
            else LatencyAssumptions().graph_kernel_gap_us
        )
    return MAX_GAP_CREDIT_US


def step_tokens(context: AnalysisContext, group: GroupSummary) -> int | None:
    """Tokens one attention-TP rank processes in the step (what the AR fusion threshold checks)."""
    annotations = group.annotations
    if "g_sq" in annotations or "c_sq" in annotations:
        tokens = annotations.get("g_sq", 0) + annotations.get("c_sq", 0)
    elif group.extend_tokens is not None:
        tokens = group.extend_tokens
    else:
        tokens = group.batch_size
    if not tokens:
        return None
    attn_cp = context.parallel.attn_cp if context.parallel else 1
    return max(tokens // max(attn_cp, 1), 1)


def _allreduce_fusion_applies(context: AnalysisContext, tokens: int | None) -> bool:
    """The fused path takes the batch: TP > 1, no DP attention, at most 2048 tokens."""
    if context.uses_dp_attention():
        return False
    if context.parallel is not None and context.parallel.tp <= 1:
        return False
    return tokens is not None and 0 < tokens <= FLASHINFER_ALLREDUCE_FUSION_MAX_TOKENS


def _moe_sum_fusable(context: AnalysisContext) -> bool | None:
    arch = context.arch
    if arch is None:
        return None
    if arch.moe is None or arch.moe.top_k < FUSED_MOE_SUM_ALLREDUCE_MIN_TOP_K:
        return False
    quant = arch.quant
    weight_only_int = (
        quant is not None
        and quant.activation_dtype is None
        and quant.weight_dtype in {"int8", "int4"}
    )
    return not weight_only_int


def analyze_group(context: AnalysisContext, group: GroupSummary) -> GroupFusion:
    step_id = context.representative_step_id(group)
    floor_us = (
        context.hardware.latency.min_kernel_us
        if context.hardware
        else LatencyAssumptions().min_kernel_us
    )
    tokens = step_tokens(context, group)
    applies = _allreduce_fusion_applies(context, tokens)
    qk_norm_model = context.arch.attention.has_qk_norm if context.arch else None
    moe_sum_fusable = _moe_sum_fusable(context)
    gap_cap = gap_credit_cap_us(context, group)
    streams: dict[tuple[int, int], list[KernelEvent]] = defaultdict(list)
    for kernel in context.kernels_by_step.get(step_id, []):
        streams[(kernel.device, kernel.stream)].append(kernel)
    by_pattern: dict[str, PatternTotals] = defaultdict(PatternTotals)
    for events in streams.values():
        state = ScanState(applies, qk_norm_model, moe_sum_fusable)
        for match in match_stream(events, state, floor_us, gap_cap):
            by_pattern[match.pattern].add(match)
    return GroupFusion(
        group=group,
        step=context.steps_by_id[step_id],
        tokens=tokens,
        allreduce_fusion_applies=applies,
        gap_cap_us=gap_cap,
        by_pattern=dict(by_pattern),
    )


# ---------------------------------------------------------------------------- recommendations
Recommendation = tuple[list[str], dict[str, Any], list[str]]


def _platform(context: AnalysisContext) -> str | None:
    return context.hardware.software.platform if context.hardware else None


def _recommend_allreduce(
    context: AnalysisContext, fusion: GroupFusion, totals: PatternTotals
) -> Recommendation:
    knowledge, hardware, parallel = context.knowledge, context.hardware, context.parallel
    architecture = context.architecture
    references = [source("allreduce_fusion"), source("allreduce_fusion_boundary")]
    platform = _platform(context)
    if platform == "rocm" and knowledge.has_flag("--enable-aiter-allreduce-fusion"):
        return (
            [
                "ROCm 上对应的是 aiter 的 allreduce + RMSNorm 融合：设置 `SGLANG_USE_AITER=1` 并加 "
                "`--enable-aiter-allreduce-fusion`（可用 `--disable-aiter-allreduce-fusion-in-prefill` 只在 decode 使用）。"
            ],
            {"--enable-aiter-allreduce-fusion": True},
            references,
        )
    if hardware is not None and (
        platform != "cuda" or context.sm_major not in FLASHINFER_ALLREDUCE_FUSION_SM_MAJORS
    ):
        return (
            [
                f"SGLang 的 allreduce + 残差 add + RMSNorm 融合只有 CUDA SM90/SM100（FlashInfer）与 ROCm（aiter）实现，"
                f"{hardware.name} 上没有可用的融合 kernel。这是平台适配项：在平台插件里实现融合通信 kernel（reduce 之后在同一 "
                "kernel 内完成残差加与 RMSNorm），通过 `RMSNorm.forward_with_allreduce_fusion` 接入 layer_boundary 的融合点。"
            ],
            {},
            [*references, source("allreduce_fusion_norm")],
        )
    lines: list[str] = []
    flags: dict[str, Any] = {}
    auto = architecture in knowledge.allreduce_fusion_auto_architectures
    a2a_none = parallel is None or parallel.moe_a2a_backend == "none"
    if auto and a2a_none:
        lines.append(
            f"SGLang 对 {architecture} 会自动开启 FlashInfer AllReduce 融合（SM90/SM100、TP>1、未开 DP attention、"
            "moe_a2a_backend=none，SM90 上还要求单机；启动日志有 `Auto-enabling FlashInfer AllReduce Fusion`）。trace 中仍是"
            "独立的 allreduce + add + RMSNorm，说明融合被关掉或不可用：检查是否设置了 "
            "`--enforce-disable-flashinfer-allreduce-fusion`、是否开启了 `--enable-deterministic-inference`（会强制关闭该融合）、"
            "日志中有没有 `allreduce_fusion API is not available`（flashinfer 版本缺少 comm 融合接口），以及是否为多机部署。"
        )
        references += [
            source("allreduce_fusion_auto"),
            source("allreduce_fusion_enforce_disable"),
            source("allreduce_fusion_deterministic"),
            source("allreduce_fusion_unavailable"),
        ]
    elif architecture is None or architecture in knowledge.layer_boundary_architectures:
        flags["--flashinfer-allreduce-fusion-backend"] = "auto"
        lines.append(
            "设置 `--flashinfer-allreduce-fusion-backend auto` 打开 allreduce + 残差 add + RMSNorm 融合"
            + (
                f"（{architecture} 的残差路径走 layer_boundary，该参数对它生效）"
                if architecture
                else "（要求模型文件的残差路径走 layer_boundary）"
            )
            + f"。只有每步 token 数 ≤ {FLASHINFER_ALLREDUCE_FUSION_MAX_TOKENS} 时走融合路径；开启后做精度回归并 A/B 延迟。"
        )
    else:
        lines.append(
            f"{architecture} 的模型文件没有使用 layer_boundary，`--flashinfer-allreduce-fusion-backend` 不会改变它的 kernel："
            "把 decoder layer 的残差 add + norm 改为 layer_boundary 写法（参考 llama.py、qwen2.py），或在 allreduce 之后"
            "直接调用 `RMSNorm.forward_with_allreduce_fusion`。"
        )
        references.append(source("allreduce_fusion_norm"))
    if fusion.tokens is not None:
        lines.append(
            f"本组每个 rank 每步 {fusion.tokens} 个 token（阈值 {FLASHINFER_ALLREDUCE_FUSION_MAX_TOKENS}）；更大的 prefill "
            "batch 不走融合路径，这类 step 的 add + RMSNorm 见“残差 add 与 RMSNorm 是分开的 kernel”。"
        )
    return lines, flags, references


def _recommend_add_norm(
    context: AnalysisContext, fusion: GroupFusion, totals: PatternTotals
) -> Recommendation:
    lines = [
        "把残差交给 RMSNorm（`norm(hidden_states, residual)`）：CUDA 上一个 `fused_add_rmsnorm` kernel 同时完成残差加与归一化；"
        "改用 layer_boundary 写法的模型由框架处理残差。",
        "已经传了 residual 仍看到分开的 add 时：`--enable-deterministic-inference` 打开的 batch invariant 模式下，带 residual 的 "
        "RMSNorm 回退到 PyTorch 原生实现；设置了 variance_size_override 的 RMSNorm 同样走原生实现。",
    ]
    references = [
        source("fused_add_rmsnorm"),
        source("rmsnorm_native_fallback"),
        source("batch_invariant_mode"),
    ]
    platform = _platform(context)
    if context.hardware is not None and platform not in {"cuda", "rocm"}:
        lines.append(
            f"{context.hardware.name} 的平台插件要为 RMSNorm 提供带残差的 `forward_<platform>` 实现（NPU 上是 "
            "`torch_npu.npu_add_rms_norm`），否则会回到原生实现。"
        )
        references.append(source("npu_add_rmsnorm"))
    return lines, {}, references


def _recommend_silu_mul(
    context: AnalysisContext, fusion: GroupFusion, totals: PatternTotals
) -> Recommendation:
    return (
        [
            "用 `SiluAndMul`/`GeluAndMul` 替代分开的激活与逐元素乘：gate/up 投影合并成一个 `MergedColumnParallelLinear`，"
            "输出 [gate, up] 交给一个 act_and_mul kernel，省掉激活结果的一次写回与读取。"
        ],
        {},
        [source("silu_and_mul")],
    )


def _head_dims(values: tuple[int, ...]) -> str:
    return "{" + ", ".join(map(str, values)) + "}"


def _recommend_qk_norm_rope(
    context: AnalysisContext, fusion: GroupFusion, totals: PatternTotals
) -> Recommendation:
    knowledge, arch, architecture = context.knowledge, context.arch, context.architecture
    head_dim = arch.attention.head_dim if arch else None
    lines: list[str] = []
    if totals.details.get("separate_norms"):
        lines.append(
            "q 与 k 的 RMSNorm 是分开的 kernel：CUDA 上 `apply_qk_norm` 默认用 JIT `fused_inplace_qknorm` 一次处理 q 和 k，"
            "条件是 q/k 的 eps 相同、未开启确定性推理、prefill 图编译器不是 inductor、head_dim ∈ "
            f"{_head_dims(FUSED_INPLACE_QKNORM_HEAD_DIMS)}；模型代码逐个调用 q_norm/k_norm 时改为调用 `apply_qk_norm`。"
        )
    supported_dim = head_dim is None or head_dim in FUSED_QK_NORM_ROPE_HEAD_DIMS
    requirement = f"要求 head_dim ∈ {_head_dims(FUSED_QK_NORM_ROPE_HEAD_DIMS)}、非 MRoPE、BF16"
    if not supported_dim:
        lines.append(
            f"head_dim={head_dim} 不在 QK norm + RoPE 融合 kernel 支持的范围内（{requirement}），需要先扩展 kernel。"
        )
    elif architecture in knowledge.fused_qk_norm_rope_architectures:
        lines.append(
            f"{architecture} 支持 `--enable-fused-qk-norm-rope`：QK norm 与 RoPE 合成一个 kernel（{requirement}）。"
            "但走这条路径时 KV cache 改由注意力后端写入（`_used_fused_qk_norm_rope_last_call` 让 save_kv_cache 为真），"
            "RoPE + KV 写入的融合随之失效；默认路径（`fused_inplace_qknorm` + 带 KV 写入的 RoPE）同样是每层两个 kernel，"
            "两种组合要 A/B 比较。"
        )
    else:
        listed = ", ".join(knowledge.fused_qk_norm_rope_architectures) or "（快照中没有）"
        lines.append(
            f"SGLang 只在 {listed} 的模型文件里读取 `--enable-fused-qk-norm-rope`；{architecture or '该模型'} 需要参照 "
            f"qwen3_moe.py 的 `use_fused_qk_norm_rope` 分支接入 `fused_qk_norm_rope`（{requirement}）。"
        )
    references = [
        source("qk_norm"),
        source("fused_inplace_qknorm"),
        source("fused_qk_norm_rope"),
        source("fused_qk_norm_rope_kernel"),
        source("fused_qk_norm_rope_model"),
    ]
    return lines, {}, references


_FUSED_KV_CONDITIONS = "CUDA、BF16 KV cache、KV pool 不是 SWA、未启用 context parallel、没有 DCP"


def _recommend_rope_kv_store(
    context: AnalysisContext, fusion: GroupFusion, totals: PatternTotals
) -> Recommendation:
    knowledge, arch, architecture = context.knowledge, context.arch, context.architecture
    lines: list[str] = []
    references = [source("fused_set_kv_buffer")]
    platform = _platform(context)
    if context.hardware is not None and platform not in {"cuda", "rocm"}:
        lines.append(
            "非 CUDA/ROCm 平台的 RoPE 实现（native/NPU/CPU 路径）断言 `fused_set_kv_buffer_arg is None`："
            "平台插件需要先提供带 KV 写入的 RoPE kernel。"
        )
        references.append(source("rope_fused_kv_platforms"))
    if arch is not None and arch.attention.is_mla:
        lines.append(
            "MLA 的 latent KV 由注意力后端写入（`set_mla_kv_buffer`），RoPE 是单独的 kernel；SGLang 只在 ROCm 路径把 RoPE、"
            "拼接与写 cache 合成一个 kernel（`_fused_rope_cat_and_cache`）。CUDA 上合并属于框架改动：实现 RoPE + concat + "
            "写 latent cache 的 kernel，并让注意力后端跳过 save_kv_cache。"
        )
        references.append(source("mla_rope_cache_rocm"))
    elif architecture in knowledge.fused_rope_kv_store_architectures:
        lines.append(
            f"{architecture} 的模型文件支持在 RoPE kernel 里直接写 KV cache（`enable_fused_set_kv_buffer`），条件："
            f"{_FUSED_KV_CONDITIONS}。trace 中仍是独立的 KV 写入：检查 `--kv-cache-dtype` 是否为 FP8（FP8 KV 省显存与注意力"
            "带宽，通常比这一处融合更值，不要为此切回 BF16）以及是否开启了 CP/DCP。"
            + (
                "另外，开启 `--enable-fused-qk-norm-rope` 时 KV 改由注意力后端写入，这条融合也会失效。"
                if architecture in knowledge.fused_qk_norm_rope_architectures
                else ""
            )
        )
    else:
        lines.append(
            "参照 qwen3_moe.py、gpt_oss.py 接入：RoPE 调用时传 `fused_set_kv_buffer_arg=create_fused_set_kv_buffer_arg(...)`，"
            "注意力调用用 `save_kv_cache=not enable_fused_set_kv_buffer(forward_batch)` 跳过后端写入；生效条件："
            f"{_FUSED_KV_CONDITIONS}。"
        )
    return lines, {}, references


def _recommend_act_quant(
    context: AnalysisContext, fusion: GroupFusion, totals: PatternTotals
) -> Recommendation:
    parsed = [detail.split("|") for detail in totals.details]
    scopes = {parts[0] for parts in parsed}
    flavors = {parts[1] for parts in parsed}
    activations = {parts[2] for parts in parsed}
    lines: list[str] = []
    where = "、".join(
        sorted("MoE 专家中间激活" if scope == "moe" else "dense MLP" for scope in scopes)
    )
    if flavors == {"group"} and activations == {"silu"}:
        lines.append(
            f"{where}在 act_and_mul 之后单独做 FP8 per-token-group 量化。SGLang 的 `per_token_group_quant` 支持 "
            "`fuse_silu_and_mul=True`（输入 [T, 2·hidden]，直接输出量化后的 [T, hidden] 与 scale），可以把这两个 kernel "
            "换成一次调用；DeepGEMM MoE runner 的 masked 布局（DeepEP low-latency）默认已这样做，Triton fused_moe 与普通 "
            "dense MLP 路径没有。属于框架改动，改完做精度回归。"
        )
    else:
        lines.append(
            f"{where}在 act_and_mul 之后单独做量化（量化方式：{'、'.join(sorted(flavors))}；激活：{'、'.join(sorted(activations))}）。"
            "SGLang 现有的融合实现（`per_token_group_quant(fuse_silu_and_mul=True)`、`silu_and_mul_masked_post_quant_fwd`）"
            "只覆盖 SiLU + FP8 group 量化，其它组合需要新写 kernel。"
        )
    references = [
        source("group_quant_fused_silu"),
        source("deep_gemm_masked_act_quant"),
        source("act_quant_fusion"),
    ]
    return lines, {}, references


def _recommend_norm_quant(
    context: AnalysisContext, fusion: GroupFusion, totals: PatternTotals
) -> Recommendation:
    flavors = set(totals.details)
    lines: list[str] = []
    if "group" in flavors or "per_token" in flavors:
        kind = "、".join(
            name
            for flavor, name in (("group", "per-token-group"), ("per_token", "per-token"))
            if flavor in flavors
        )
        lines.append(
            f"RMSNorm 之后单独做 {kind} 量化：SGLang 在 CUDA 上没有 norm + 动态量化的融合（layer_boundary 的 "
            "`_norm_quant_kernel` 只在 ROCm/aiter 上选融合 kernel，CUDA 走 `_update_and_read_residual_plain`）。属于框架缺口："
            "实现 RMSNorm（含残差加）+ 量化的 kernel，并在 `_norm_quant_kernel` 中为 CUDA 分派。"
        )
    if "per_tensor" in flavors:
        lines.append(
            "静态 per-tensor FP8 量化可由 `RMSNorm.fuse_input_quant(linear)` 融合进 RMSNorm（flashinfer `rmsnorm_quant`/"
            "`fused_add_rmsnorm_quant`），条件是 checkpoint 提供标量 input_scale；llama.py、qwen2.py、ministral3.py 已接入，"
            "其它模型照此接入。动态 per-tensor 量化要先求 absmax，不能这样融合。"
        )
    if "unknown" in flavors or not lines:
        lines.append(
            "无法从 kernel 名判断量化方式：确认这是静态 per-tensor（可用 `RMSNorm.fuse_input_quant`）还是动态量化"
            "（CUDA 上需要新写 norm + 量化 kernel）。"
        )
    return lines, {}, [source("norm_quant_fusion"), source("norm_static_quant_fusion")]


def _recommend_moe_sum(
    context: AnalysisContext, fusion: GroupFusion, totals: PatternTotals
) -> Recommendation:
    top_k = context.arch.moe.top_k if context.arch and context.arch.moe else None
    lines = [
        "Triton MoE 在 down GEMM 之后单独用 `moe_sum` 把 top-k 个专家输出相加：`--enable-fused-moe-sum-all-reduce` 让 down "
        "GEMM 直接原子累加到输出，省掉这个 kernel"
        + (
            f"（top_k={top_k}）"
            if top_k
            else f"（要求 top_k ≥ {FUSED_MOE_SUM_ALLREDUCE_MIN_TOP_K}）"
        )
        + "。它与 TP allreduce 无关；原子累加的求和顺序不固定，结果不再逐位确定，开启后做精度回归。"
    ]
    return lines, {"--enable-fused-moe-sum-all-reduce": True}, [source("fused_moe_sum_allreduce")]


def _recommend_chain(
    context: AnalysisContext, fusion: GroupFusion, totals: PatternTotals
) -> Recommendation:
    group = fusion.group
    preceding = "，".join(f"{name}×{count}" for name, count in totals.preceding.most_common(2))
    lines = [
        "这些是 PyTorch 的逐元素/拷贝 kernel（`.to()` 类型转换、`.contiguous()`、`torch.cat`、逐元素加乘），每个都把中间结果写回显存。"
        + (f"链之前的 kernel 是 {preceding}，据此在模型代码里定位；" if preceding else "")
        + "先去掉多余的类型转换与拷贝（例如 cat 之后马上 split、transpose 之后 contiguous）。"
    ]
    if group.is_decode_like:
        if group.batch_size <= TORCH_COMPILE_MAX_BS_DEFAULT:
            lines.append(
                "decode 可 A/B `--enable-torch-compile`：只编译 batch ≤ `--torch-compile-max-bs`（默认 "
                f"{TORCH_COMPILE_MAX_BS_DEFAULT}）的 decode CUDA graph，由 inductor 合并这些逐元素 kernel；启动时间变长，"
                "SGLang 将其标为实验特性。"
            )
        else:
            lines.append(
                f"本组 bs={group.batch_size} 超过 `--torch-compile-max-bs` 的默认值 {TORCH_COMPILE_MAX_BS_DEFAULT}：要让 "
                "`--enable-torch-compile` 覆盖它需同时调大该参数（编译时间与显存都会增加）。"
            )
    else:
        lines.append(
            "torch.compile 只作用于 decode CUDA graph 的捕获档位，prefill 中的这类链需要在模型代码里合并：写成一个 "
            "Triton/JIT kernel，或复用 SGLang 已有的融合算子。"
        )
    return lines, {}, [source("torch_compile_bs")]


_RECOMMENDERS: dict[
    str, Callable[[AnalysisContext, GroupFusion, PatternTotals], Recommendation]
] = {
    "allreduce_add_norm": _recommend_allreduce,
    "qk_norm_rope": _recommend_qk_norm_rope,
    "rope_kv_store": _recommend_rope_kv_store,
    "add_norm": _recommend_add_norm,
    "silu_mul": _recommend_silu_mul,
    "act_quant": _recommend_act_quant,
    "norm_quant": _recommend_norm_quant,
    "moe_sum": _recommend_moe_sum,
    "elementwise_chain": _recommend_chain,
}


# ---------------------------------------------------------------------------------- findings
def _share(totals: PatternTotals, group: GroupSummary) -> float:
    return totals.saved_us / group.step_us if group.step_us > 0 else 0.0


def detect_fusion_opportunities(
    context: AnalysisContext,
) -> tuple[list[Finding], list[GroupFusion]]:
    """One finding per (pattern, phase), headlined by the group where it costs the most in total."""
    analyses = [
        analyze_group(context, group) for group in context.primary_groups() if group.step_us > 0
    ]
    floor_us = (
        context.hardware.latency.min_kernel_us
        if context.hardware
        else LatencyAssumptions().min_kernel_us
    )
    findings: list[Finding] = []
    for pattern in PATTERNS:
        for decode_like in (True, False):
            candidates = [
                analysis
                for analysis in analyses
                if analysis.group.is_decode_like == decode_like
                and pattern.key in analysis.by_pattern
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda analysis: (
                    -analysis.by_pattern[pattern.key].saved_us * analysis.group.steps
                )
            )
            headline = candidates[0]
            group, totals = headline.group, headline.by_pattern[pattern.key]
            share = _share(totals, group)
            if totals.saved_us < max(FUSION_MIN_SAVED_US, FUSION_MIN_SHARE * group.step_us):
                continue
            phase = "decode" if decode_like else "prefill"
            compositions = "；".join(
                f"{composition} ×{count}"
                for composition, count in totals.compositions.most_common(2)
            )
            evidence = [
                f"{group.label}（{group.steps} 个 step，代表 step #{headline.step.ordinal}）每步 {totals.count} 处：{compositions}。",
                f"被融合掉的 kernel 合计 {totals.removed_us:.0f} us/step，估算可省 {totals.saved_us:.0f} us/step"
                f"（step {group.step_us / 1000:.2f} ms 的 {share:.1%}）。",
            ]
            others = [
                f"{analysis.group.label} {analysis.by_pattern[pattern.key].count} 处、省 "
                f"{analysis.by_pattern[pattern.key].saved_us:.0f} us/step"
                f"（{_share(analysis.by_pattern[pattern.key], analysis.group):.1%}）"
                for analysis in candidates[1:MAX_EVIDENCE_GROUPS]
            ]
            if others:
                evidence.append("同阶段其它形状：" + "；".join(others) + "。")
            gap_note = f"每处最多 {headline.gap_cap_us:g} us"
            if headline.gap_cap_us < MAX_GAP_CREDIT_US:
                gap_note += "；该组没走 CUDA graph，发射开销计入“没有走 CUDA graph”一条，这里只按 graph 模式的间隙计"
            evidence.append(
                f"估算方式：≤{LATENCY_REGIME_FACTOR * floor_us:.1f} us 的 kernel 按整段省掉（延迟主导），更长的按融合后省掉的"
                f"访存比例 {pattern.traffic_saved:.0%} 计，另加被融合 kernel 之间的空隙（{gap_note}）。"
            )
            recommendation, flags, references = _RECOMMENDERS[pattern.key](
                context, headline, totals
            )
            findings.append(
                Finding(
                    id=f"fusion.{pattern.key}.{phase}",
                    kind=pattern.kind,
                    severity=severity_for_share(share),
                    title=f"{phase}：{pattern.title}（每步 {totals.count} 处）",
                    evidence=evidence,
                    recommendation=recommendation,
                    flags=flags,
                    gain_ms_per_step=totals.saved_us / 1000,
                    gain_note="估算值：以开启融合后的 A/B trace 为准",
                    scope=group.label,
                    references=references,
                )
            )
    return findings, analyses
