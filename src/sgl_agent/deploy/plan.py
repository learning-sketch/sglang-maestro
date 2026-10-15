"""Scenario 5: size, place and launch a deployment that serves a workload within its SLO.

Every candidate instance size starts from the tuned launch scenario 3 produces for it. With an
arrival rate, the simulator finds the highest rate one instance sustains within the SLO, the
replica count follows from it and the whole cluster is simulated again behind the router at the
workload's rate (adding instances while it misses). When TPOT limits an instance, mixed chunked
prefill with smaller chunks is tried. PD disaggregation sizes the prefill pool on TTFT and the
decode pool on TPOT and wins when it needs clearly fewer GPUs or colocated serving cannot meet
the SLO. Without an arrival rate (throughput mode) the GPU budget is filled with the instance
size that produces the most output tokens per GPU. The chosen cluster is placed on nodes and
rendered as launch, router, health-check and benchmark commands.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any

from sgl_agent.commands import (
    DIST_INIT_PORT,
    Command,
    Repeat,
    launch_server_command,
    parallel_server_flags,
    shell_command,
    tool_command,
)
from sgl_agent.deploy.instance import InstanceModel, InstanceSpec, build_instance
from sgl_agent.deploy.simulate import PD_TRANSFER_OVERHEAD_US, ClusterSim, SimResult
from sgl_agent.features.base import DEFAULT_HOST_MEMORY_GB, FeatureContext, build_context
from sgl_agent.features.report import (
    GOOD,
    SERVER_PORT,
    CombinedPlan,
    FeatureReport,
    advise_features,
    combine,
)
from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.knowledge import SGLangKnowledge
from sgl_agent.knowledge.facts import (
    DISAGGREGATION_BOOTSTRAP_PORT,
    PD_TRANSFER_BACKENDS,
    ROUTER_DEFAULT_POLICY,
    ROUTER_MODULE,
    source,
)
from sgl_agent.model.config import ModelArch
from sgl_agent.model.ops import Calibration
from sgl_agent.model.parallel import ParallelConfig
from sgl_agent.workload import (
    SLO,
    Request,
    Workload,
    bench_command,
    length_bounds,
    mean_length,
    sample_requests,
)

MIXED_CHUNK_SIZES = (1024, 512, 256)
"""Per-rank chunk sizes tried with ``--enable-mixed-chunk`` when TPOT limits an instance."""
CAPACITY_SAMPLES = 240
VERIFY_SAMPLES = 1200
STEADY_STATE_FACTOR = 3
"""A capacity probe sends at least this many times the requests in flight (Little's law)."""
MAX_PROBE_SAMPLES = 3000
BISECTION_STEPS = 6
EXPAND_STEPS = 4
VERIFY_TOP = 2
"""Colocated candidates (fewest GPUs first) simulated as a whole cluster."""
MAX_ADDED = 3
MAX_SIZES = 4
PD_MIN_SAVING = 0.10
RATE_BUDGET_NODES = 8
ROUTER_PORT = 8000
PORT_STRIDE = 1000
"""Gap between the ``--port`` of instances sharing a node: attention DP also binds ``--port + 13``
and ``--port + 233 .. + 239``."""
BENCH_OUTPUT = "bench_result.jsonl"
PARALLEL_FLAGS = frozenset(
    {
        "--tp-size",
        "--pp-size",
        "--attn-dp-size",
        "--attn-cp-size",
        "--dcp-size",
        "--ep-size",
        "--moe-a2a-backend",
        "--moe-dense-tp-size",
    }
)
PREFILL_CP_FLAGS = ("--enable-prefill-cp", "--attn-cp-size", "--cp-strategy")
DP_ATTENTION_FLAGS = ("--load-balance-method",)
PD_MATCHED_FLAGS = ("--page-size", "--kv-cache-dtype")
"""Flags prefill and decode servers must agree on (the KV receiver checks them at bootstrap)."""
LIMIT_LABELS = {
    "ttft": "TTFT 超 SLO",
    "tpot": "TPOT 超 SLO",
    "overload": "排队发散",
    "rejected": "请求被拒",
}


def rate_unit(workload: Workload) -> str:
    """Unit of arrival rates: multi-turn workloads arrive as sessions."""
    prefix = workload.shared_prefix
    return "会话/s" if prefix is not None and prefix.num_turns > 1 else "req/s"


@dataclass(frozen=True)
class DeployOptions:
    max_gpus: int | None = None
    """GPU budget; default one node in throughput mode and eight nodes with an arrival rate."""
    sizes: tuple[int, ...] | None = None
    """GPUs per instance to consider (default: powers of two from the smallest that fits)."""
    allow_pd: bool = True
    allow_mixed_chunk: bool = True
    transfer_backend: str = "mooncake"
    ib_device: str | None = None
    capacity_samples: int = CAPACITY_SAMPLES
    verify_samples: int = VERIFY_SAMPLES
    seed: int = 1

    def __post_init__(self) -> None:
        if self.max_gpus is not None and self.max_gpus < 1:
            raise ValueError("max_gpus must be >= 1")
        if self.sizes is not None and any(size < 1 for size in self.sizes):
            raise ValueError("instance sizes must be >= 1")
        if self.transfer_backend not in PD_TRANSFER_BACKENDS:
            raise ValueError(
                f"unknown transfer backend {self.transfer_backend!r}; known: {PD_TRANSFER_BACKENDS}"
            )
        if min(self.capacity_samples, self.verify_samples) < 20:
            raise ValueError("capacity_samples and verify_samples must be >= 20")


@dataclass
class Capacity:
    """Highest arrival rate (workload units) one instance serves within the SLO."""

    rate: float
    saturation: float
    """Completion rate when every request is sent at once (upper bound of ``rate``)."""
    result: SimResult | None
    """Simulation at ``rate``."""
    limit: str | None
    """What failed just above ``rate`` (``LIMIT_LABELS`` keys); None if saturation still passes."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rate": round(self.rate, 4),
            "saturation": round(self.saturation, 4),
            "limit": self.limit,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class Pool:
    """``count`` identical instances of one role."""

    role: str
    model: InstanceModel = field(repr=False)
    count: int
    capacity: Capacity | None = None

    @property
    def spec(self) -> InstanceSpec:
        return self.model.spec

    @property
    def num_gpus(self) -> int:
        return self.spec.num_gpus * self.count

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "count": self.count,
            "num_gpus": self.num_gpus,
            "capacity": self.capacity.to_dict() if self.capacity else None,
            "instance": self.spec.to_dict(),
        }


@dataclass
class Candidate:
    pools: dict[str, Pool]
    policy: str = ROUTER_DEFAULT_POLICY
    dp_aware: bool = False
    result: SimResult | None = None
    """The whole cluster at the workload's rate (or every request at once in throughput mode)."""
    retuned: bool = False
    """Features were re-advised for the per-replica arrival rate."""
    added: int = 0
    """Instances added after the cluster simulation missed the SLO."""
    notes: list[str] = field(default_factory=list)

    @property
    def disaggregated(self) -> bool:
        return "prefill" in self.pools

    @property
    def feasible(self) -> bool:
        return all(pool.count > 0 for pool in self.pools.values())

    @property
    def num_gpus(self) -> int:
        return sum(pool.num_gpus for pool in self.pools.values())

    @property
    def num_instances(self) -> int:
        return sum(pool.count for pool in self.pools.values())

    @property
    def meets_slo(self) -> bool:
        return self.result is not None and self.result.meets_slo

    def describe(self) -> str:
        parts = []
        for role, pool in self.pools.items():
            name = "" if role == "colocated" else f"{role} "
            parts.append(f"{pool.count}×{name}{pool.spec.describe()}")
        text = " + ".join(parts)
        return f"PD：{text}" if self.disaggregated else text

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.describe(),
            "disaggregated": self.disaggregated,
            "feasible": self.feasible,
            "num_gpus": self.num_gpus,
            "num_instances": self.num_instances,
            "policy": self.policy,
            "dp_aware": self.dp_aware,
            "retuned": self.retuned,
            "added": self.added,
            "meets_slo": self.meets_slo,
            "pools": {role: pool.to_dict() for role, pool in self.pools.items()},
            "result": self.result.to_dict() if self.result else None,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Placement:
    role: str
    index: int
    nodes: tuple[int, ...]
    gpus_per_node: int
    base_gpu_id: int
    port: int
    bootstrap_port: int | None = None

    @property
    def host(self) -> str:
        return f"<node{self.nodes[0]}_ip>"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def gpus(self) -> str:
        last = self.base_gpu_id + self.gpus_per_node - 1
        span = (
            f"GPU {self.base_gpu_id}"
            if last == self.base_gpu_id
            else f"GPU {self.base_gpu_id}–{last}"
        )
        if len(self.nodes) == 1:
            return f"节点 {self.nodes[0]} {span}"
        return f"节点 {self.nodes[0]}–{self.nodes[-1]} 每台 {span}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "index": self.index,
            "nodes": list(self.nodes),
            "gpus_per_node": self.gpus_per_node,
            "base_gpu_id": self.base_gpu_id,
            "port": self.port,
            "bootstrap_port": self.bootstrap_port,
            "url": self.url,
        }


@dataclass
class NodePlan:
    node: int
    gpus_used: int
    commands: list[Command]

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "gpus_used": self.gpus_used,
            "commands": [command.to_dict() for command in self.commands],
        }


@dataclass
class DeployPlan:
    model: str
    hardware: str
    workload: Workload
    platform: str
    mode: str
    """``rate`` (arrival rate given: fewest GPUs within the SLO) or ``throughput``."""
    budget_gpus: int
    devices_per_node: int
    knowledge_commit: str | None
    chosen: Candidate | None
    candidates: list[Candidate]
    placements: list[Placement]
    nodes: list[NodePlan]
    router: Command | None
    bench: Command | None
    checks: list[Command]
    steps: list[str]
    reasons: list[str] = field(default_factory=list)
    bottlenecks: list[str] = field(default_factory=list)
    adaptation: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    @property
    def rate_unit(self) -> str:
        return rate_unit(self.workload)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "hardware": self.hardware,
            "workload": self.workload.name,
            "workload_description": self.workload.describe(),
            "platform": self.platform,
            "mode": self.mode,
            "rate_unit": self.rate_unit,
            "budget_gpus": self.budget_gpus,
            "devices_per_node": self.devices_per_node,
            "num_nodes": self.num_nodes,
            "knowledge_commit": self.knowledge_commit,
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "placements": [placement.to_dict() for placement in self.placements],
            "nodes": [node.to_dict() for node in self.nodes],
            "router": self.router.to_dict() if self.router else None,
            "bench": self.bench.to_dict() if self.bench else None,
            "checks": [command.to_dict() for command in self.checks],
            "steps": self.steps,
            "reasons": self.reasons,
            "bottlenecks": self.bottlenecks,
            "adaptation": self.adaptation,
            "notes": self.notes,
            "assumptions": self.assumptions,
            "references": self.references,
        }

    def to_markdown(self) -> str:
        from sgl_agent.deploy.report import render_deploy_plan

        return render_deploy_plan(self)


# ---------------------------------------------------------------------- PD roles
def pd_layouts(
    arch: ModelArch, layout: ParallelConfig
) -> tuple[ParallelConfig, ParallelConfig] | None:
    """Prefill and decode layouts derived from a tuned colocated layout.

    Prefill CP stays on the prefill side and DCP on the decode side. A DCP decode server only
    takes KV from DCP-1 prefill ranks for MLA pools, and only when prefill CP is 1.
    """
    decode = layout.with_(attn_cp=1)
    if decode.dcp > 1 and not arch.attention.is_mla:
        decode = decode.with_(dcp=1)
    if decode.validate(arch):
        decode = decode.with_(dcp=1)
        if decode.validate(arch):
            return None
    prefill = layout.with_(dcp=1)
    if decode.dcp > 1:
        prefill = prefill.with_(attn_cp=1)
    if prefill.validate(arch):
        return None
    return prefill, decode


def relayout_flags(flags: dict[str, Any], layout: ParallelConfig) -> dict[str, Any]:
    """``flags`` with the parallel flags of ``layout`` (CP and DP-attention flags dropped when it
    has no CP or DP attention)."""
    head = {key: flags[key] for key in ("--model-path", "--device") if key in flags}
    rest = {
        key: value
        for key, value in flags.items()
        if key not in head
        and key not in PARALLEL_FLAGS
        and not (layout.attn_cp == 1 and key in PREFILL_CP_FLAGS)
        and not (layout.attn_dp == 1 and key in DP_ATTENTION_FLAGS)
    }
    return {**head, **parallel_server_flags(layout), **rest}


def role_flags(flags: dict[str, Any], role: str, layout: ParallelConfig) -> dict[str, Any]:
    """Launch flags of a PD server: no mixed chunk; no radix cache tiers on decode."""
    kept = {key: value for key, value in flags.items() if key != "--enable-mixed-chunk"}
    if role == "decode":
        kept = {
            key: value
            for key, value in kept.items()
            if key != "--enable-hierarchical-cache" and not key.startswith("--hicache-")
        }
    return relayout_flags(kept, layout)


def match_decode(prefill: dict[str, Any], decode: dict[str, Any]) -> dict[str, Any]:
    """Prefill flags with the decode server's speculative, page-size and KV-dtype settings."""
    kept = {
        key: value
        for key, value in prefill.items()
        if not key.startswith("--speculative-") and key not in PD_MATCHED_FLAGS
    }
    kept.update(
        {
            key: value
            for key, value in decode.items()
            if key.startswith("--speculative-") or key in PD_MATCHED_FLAGS
        }
    )
    return kept


# ---------------------------------------------------------------------- placement
def _node_span(size: int, devices_per_node: int) -> int:
    nodes = -(-size // devices_per_node)
    while size % nodes or size // nodes > devices_per_node:
        nodes += 1
    return nodes


def place(pools: list[Pool], devices_per_node: int) -> list[Placement]:
    """First-fit decreasing onto nodes; instances sharing a node get distinct GPUs and ports."""
    order = {"prefill": 0, "decode": 1, "colocated": 2}
    free: list[int] = []
    slots: list[int] = []
    placements: list[Placement] = []
    for pool in sorted(pools, key=lambda item: (-item.spec.num_gpus, order[item.role])):
        size = pool.spec.num_gpus
        for index in range(pool.count):
            if size > devices_per_node:
                span = _node_span(size, devices_per_node)
                first = len(free)
                free += [0] * span
                slots += [1] * span
                bootstrap = DISAGGREGATION_BOOTSTRAP_PORT if pool.role == "prefill" else None
                placements.append(
                    Placement(
                        pool.role,
                        index,
                        tuple(range(first, first + span)),
                        size // span,
                        0,
                        SERVER_PORT,
                        bootstrap,
                    )
                )
                continue
            node = next((i for i, left in enumerate(free) if left >= size), None)
            if node is None:
                node = len(free)
                free.append(devices_per_node)
                slots.append(0)
            base = devices_per_node - free[node]
            free[node] -= size
            slot = slots[node]
            slots[node] += 1
            placements.append(
                Placement(
                    pool.role,
                    index,
                    (node,),
                    size,
                    base,
                    SERVER_PORT + PORT_STRIDE * slot,
                    DISAGGREGATION_BOOTSTRAP_PORT + slot if pool.role == "prefill" else None,
                )
            )
    return placements


def launch_commands(
    knowledge: SGLangKnowledge,
    pool: Pool,
    placement: Placement,
    options: DeployOptions,
) -> list[tuple[int, Command]]:
    """One ``launch_server`` command per node the instance occupies."""
    spec = pool.spec
    flags = dict(spec.flags)
    flags["--host"] = "0.0.0.0"
    flags["--port"] = placement.port
    if placement.base_gpu_id:
        flags["--base-gpu-id"] = placement.base_gpu_id
    if pool.role != "colocated":
        flags["--disaggregation-mode"] = pool.role
        flags["--disaggregation-transfer-backend"] = options.transfer_backend
        if placement.bootstrap_port is not None:
            flags["--disaggregation-bootstrap-port"] = placement.bootstrap_port
        if options.ib_device:
            flags["--disaggregation-ib-device"] = options.ib_device
    span = len(placement.nodes)
    commands = []
    for rank, node in enumerate(placement.nodes):
        node_flags = dict(flags)
        title = f"{pool.role} 实例 {placement.index}（{spec.layout.describe()}）"
        if span > 1:
            node_flags.update(
                {
                    "--nnodes": span,
                    "--node-rank": rank,
                    "--dist-init-addr": f"{placement.host}:{DIST_INIT_PORT}",
                }
            )
            title += f" node-rank {rank}/{span}"
        commands.append(
            (
                node,
                launch_server_command(
                    knowledge,
                    node_flags,
                    env=spec.env,
                    title=f"节点 {node}：{title}",
                    note=placement.gpus(),
                ),
            )
        )
    return commands


def router_command(candidate: Candidate, placements: list[Placement]) -> Command | None:
    """``sglang_router.launch_router`` in front of the workers (none for a single instance)."""
    if not candidate.disaggregated and candidate.num_instances <= 1:
        return None
    flags: dict[str, Any] = {"--host": "0.0.0.0", "--port": ROUTER_PORT}
    if candidate.disaggregated:
        prefill = [p for p in placements if p.role == "prefill"]
        decode = [p for p in placements if p.role == "decode"]
        flags["--pd-disaggregation"] = True
        flags["--prefill"] = Repeat(tuple(f"{p.url} {p.bootstrap_port}" for p in prefill))
        flags["--decode"] = Repeat(tuple(p.url for p in decode))
        flags["--prefill-policy"] = candidate.policy
        flags["--decode-policy"] = "round_robin"
    else:
        flags["--worker-urls"] = [p.url for p in placements]
        flags["--policy"] = candidate.policy
    if candidate.dp_aware:
        flags["--dp-aware"] = True
    note = (
        "PD 模式：每个 prefill 地址后面跟它的 --disaggregation-bootstrap-port"
        if candidate.disaggregated
        else "cache_aware：前缀匹配率 > 0.3 时发往匹配最长的实例，否则发往负载最低的实例"
    )
    return tool_command(ROUTER_MODULE, flags, title="启动 sglang_router", note=note)


# ---------------------------------------------------------------------- search
def _limit(result: SimResult, slo: SLO) -> str:
    """Why ``result`` misses ``slo``: rejected requests, TTFT, TPOT or an unbounded queue."""
    if result.failed:
        return "rejected"
    percentile = slo.percentile
    if slo.ttft_ms is not None and result.ttft_ms[percentile] > slo.ttft_ms:
        return "ttft"
    if slo.tpot_ms is not None and result.tpot_ms[percentile] > slo.tpot_ms:
        return "tpot"
    return "overload"


@dataclass
class _Option:
    model: InstanceModel
    capacity: Capacity
    notes: list[str] = field(default_factory=list)


class _Planner:
    def __init__(
        self,
        arch: ModelArch,
        hardware: HardwareSpec,
        knowledge: SGLangKnowledge,
        workload: Workload,
        options: DeployOptions,
        *,
        model_path: str | None,
        kv_dtype: str | None,
        host_memory_gb: float,
        calibration: Calibration | None,
    ):
        self.arch = arch
        self.hardware = hardware
        self.knowledge = knowledge
        self.workload = workload
        self.options = options
        self._context_args: dict[str, Any] = {
            "model_path": model_path,
            "kv_dtype": kv_dtype,
            "host_memory_gb": host_memory_gb,
            "calibration": calibration,
        }
        self.base = build_context(arch, hardware, knowledge, workload, **self._context_args)
        self.rate = workload.request_rate
        self.slo = workload.slo
        prefix = workload.shared_prefix
        self.turns = prefix.num_turns if prefix is not None else 1
        devices = hardware.devices_per_node
        minimum = self.base.parallel.num_gpus
        default_budget = devices * (RATE_BUDGET_NODES if self.rate is not None else 1)
        self.budget = options.max_gpus or max(default_budget, minimum)
        if self.budget < minimum:
            raise ValueError(
                f"GPU 预算 {self.budget} 放不下最小实例 {self.base.parallel.describe()}（{minimum} 卡）"
            )
        efficiency = hardware.efficiency.inter_node_comm
        self.transfer_bytes_per_us = hardware.inter_node_bandwidth_gbps * 1e9 * efficiency / 1e6
        rows = prefix.num_rows if prefix is not None else workload.num_prompts
        self.verify_rows = min(rows, options.verify_samples)
        # mean KV a running request holds over its life; shared system prompts sit in the radix
        # cache once per group, not once per request.
        shared = mean_length(prefix.system_prompt_len, workload.range_ratio) if prefix else 0.0
        self.kv_per_request = max(
            workload.mean_input_len - shared + workload.mean_output_len / 2, 1.0
        )
        self.notes: list[str] = []
        self.reasons: list[str] = []
        self._tuned: dict[tuple[int, float | None], tuple[FeatureContext, FeatureReport]] = {}
        self._variants: dict[tuple[int, float | None], list[tuple[str, CombinedPlan]]] = {}
        self._samples: dict[tuple[float | None, int], list[Request]] = {}
        self._latency: dict[InstanceModel, float] = {}
        """Mean end-to-end seconds of each instance's last passing probe (sizes the next one)."""
        self.context_dropped = (0, 0)
        self.sizes = self._sizes()
        self.pd_evaluated = False

    # ------------------------------------------------------------------ inputs
    def _sizes(self) -> list[int]:
        minimum = self.base.parallel.num_gpus
        if self.options.sizes:
            wanted = sorted(set(self.options.sizes))
        else:
            top = min(self.budget, max(2 * self.hardware.devices_per_node, minimum))
            wanted = [minimum * 2**i for i in range(MAX_SIZES) if minimum * 2**i <= top]
        sizes = []
        for size in wanted:
            if size < minimum:
                self.notes.append(f"{size} 卡的实例放不下权重和最小 KV 池，跳过")
                continue
            if size > self.budget:
                self.notes.append(f"{size} 卡的实例超出 GPU 预算 {self.budget}，跳过")
                continue
            problems = ParallelConfig(tp=size).validate(self.arch)
            if problems:
                self.notes.append(f"TP{size} 不是合法布局（{problems[0]}），跳过")
                continue
            sizes.append(size)
        if not sizes:
            raise ValueError("没有可评估的实例规模：" + "；".join(self.notes))
        return sizes

    def _requests(self, rate: float | None, num: int) -> list[Request]:
        key = (rate, num)
        cached = self._samples.get(key)
        if cached is not None:
            return cached
        requests = sample_requests(
            self.workload.with_(request_rate=rate), num=num, seed=self.options.seed
        )
        limit = self.arch.max_position_embeddings or 0
        if limit:
            kept = [r for r in requests if r.input_len + r.output_len <= limit]
            if len(kept) < len(requests) and num == self.verify_rows:
                self.context_dropped = (len(requests) - len(kept), len(requests))
            requests = kept
        if rate is None or rate == self.rate:
            self._samples[key] = requests
        return requests

    def tuned(self, size: int, rate: float | None) -> tuple[FeatureContext, CombinedPlan]:
        """Scenario 3's combined launch for a ``size``-GPU instance at arrival ``rate``."""
        key = (size, rate)
        if key not in self._tuned:
            ctx = build_context(
                self.arch,
                self.hardware,
                self.knowledge,
                self.workload.with_(request_rate=rate),
                parallel=ParallelConfig(tp=size),
                **self._context_args,
            )
            self._tuned[key] = (ctx, advise_features(ctx))
        ctx, report = self._tuned[key]
        return ctx, report.combined

    def ttft_variants(self, size: int, rate: float | None) -> list[tuple[str, CombinedPlan]]:
        """Launches with a shorter prefill than the tuned one, for TTFT-limited instances.

        Scenario 3 judges TTFT on one prompt alone and picks the layout with the best decode
        throughput; under load a P90 TTFT target also pays for queueing behind other prefills.
        Prefill CP (with DCP off, they are exclusive) splits a prompt over more ranks, and
        dropping DP attention runs each prompt's attention over the whole TP group again.
        """
        key = (size, rate)
        if key not in self._variants:
            ctx, combined = self.tuned(size, rate)
            advices = self._tuned[key][1].advices
            statuses = {item.key: item.status for item in advices}
            wanted: list[tuple[str, dict[str, str]]] = []
            if statuses.get("prefill_cp") in GOOD and "prefill_cp" not in combined.enabled:
                wanted.append(("prefill CP", {"prefill_cp": "recommend", "dcp": "not_recommended"}))
            if "dp_attention" in combined.enabled:
                wanted.append(("不开 DP attention", {"dp_attention": "not_recommended"}))
            variants = []
            for label, changes in wanted:
                changed = [
                    replace(item, status=changes[item.key]) if item.key in changes else item
                    for item in advices
                ]
                variant = combine(ctx, changed)
                if variant.layout != combined.layout and (
                    "prefill_cp" not in changes or "prefill_cp" in variant.enabled
                ):
                    variants.append((label, variant))
            self._variants[key] = variants
        return self._variants[key]

    def colocated_model(
        self, ctx: FeatureContext, combined: CombinedPlan, extra: dict[str, Any] | None = None
    ) -> InstanceModel | None:
        spec = build_instance(
            ctx,
            combined.layout,
            {**combined.flags, **(extra or {})},
            env=combined.env,
            features=list(combined.enabled),
            notes=list(combined.notes),
        )
        return InstanceModel(ctx, spec) if spec.kv_tokens_per_rank > 0 else None

    def role_model(
        self,
        ctx: FeatureContext,
        combined: CombinedPlan,
        role: str,
        decode: InstanceSpec | None = None,
    ) -> InstanceModel | None:
        layouts = pd_layouts(self.arch, combined.layout)
        if layouts is None:
            return None
        layout = layouts[0] if role == "prefill" else layouts[1]
        flags = role_flags(combined.flags, role, layout)
        if decode is not None:
            flags = match_decode(flags, decode.flags)
        features = [
            name
            for name in combined.enabled
            if not (
                (name == "dcp" and layout.dcp == 1)
                or (name == "prefill_cp" and layout.attn_cp == 1)
                or (name == "hicache" and role == "decode")
                or (name == "speculative" and "--speculative-algorithm" not in flags)
            )
        ]
        if "--speculative-algorithm" in flags and "speculative" not in features:
            features.append("speculative")
        spec = build_instance(ctx, layout, flags, role=role, env=combined.env, features=features)
        return InstanceModel(ctx, spec) if spec.kv_tokens_per_rank > 0 else None

    # ------------------------------------------------------------------ simulation
    def role_slo(self, role: str) -> SLO:
        """A prefill pool answers for TTFT only, a decode pool for TPOT only."""
        if role == "prefill":
            return SLO(ttft_ms=self.slo.ttft_ms, percentile=self.slo.percentile)
        if role == "decode":
            return SLO(tpot_ms=self.slo.tpot_ms, percentile=self.slo.percentile)
        return self.slo

    def in_flight(self, rate: float, result: SimResult) -> float:
        """Requests in the system at ``rate`` (Little's law with the run's mean latency)."""
        return rate * self.turns * result.mean_e2e_ms / 1000

    def sustainable(self, model: InstanceModel, rate: float, result: SimResult) -> bool:
        """The requests holding KV at ``rate`` fit the running-request cap and the KV pool.

        A run shorter than the time to fill the pool can pass while the steady state would
        queue without bound; this catches it without simulating the steady state.
        """
        spec = model.spec
        if spec.role == "prefill" or result.completed == 0:
            return True
        holding = rate * self.turns * max(result.mean_e2e_ms - result.mean_ttft_ms, 0.0) / 1000
        fits = min(spec.max_running_per_rank, spec.kv_tokens_per_rank / self.kv_per_request)
        return holding <= spec.attn_dp * fits

    def passes(self, model: InstanceModel, rate: float, result: SimResult) -> bool:
        return result.meets_slo and self.sustainable(model, rate, result)

    def probe(self, model: InstanceModel, rate: float | None) -> SimResult:
        """One instance alone at ``rate`` (every request at once when None).

        A passing Poisson run is lengthened until it sends ``STEADY_STATE_FACTOR`` times the
        requests in flight, so long outputs reach the steady state instead of finishing while
        the KV pool is still filling.
        """
        num = self.options.capacity_samples
        latency = self._latency.get(model)
        if rate is not None and latency is not None:
            needed = math.ceil(STEADY_STATE_FACTOR * rate * self.turns * latency)
            num = max(num, min(needed, MAX_PROBE_SAMPLES))
        result = self._run(model, rate, num)
        while rate is not None and num < MAX_PROBE_SAMPLES and self.passes(model, rate, result):
            needed = math.ceil(STEADY_STATE_FACTOR * self.in_flight(rate, result))
            if needed <= num:
                break
            num = min(needed, MAX_PROBE_SAMPLES)
            result = self._run(model, rate, num)
        if rate is not None and self.passes(model, rate, result):
            self._latency[model] = result.mean_e2e_ms / 1000
        return result

    def _run(self, model: InstanceModel, rate: float | None, num: int) -> SimResult:
        requests = self._requests(rate, num)
        role = model.spec.role
        common: dict[str, Any] = {"slo": self.role_slo(role), "seed": self.options.seed}
        if role == "prefill":
            sim = ClusterSim(
                prefill=[model],
                decode=[],
                transfer_bytes_per_us=self.transfer_bytes_per_us,
                **common,
            )
        elif role == "decode":
            sim = ClusterSim(prefill=[], decode=[model], **common)
        else:
            sim = ClusterSim(colocated=[model], **common)
        return sim.run(requests)

    def capacity(self, model: InstanceModel) -> Capacity:
        """Bisect the arrival rate, starting from the saturation completion rate.

        Sending everything at once ends in a tail of long requests decoding alone, so the
        completion rate can undershoot what Poisson arrivals sustain: double while it passes.
        """
        slo = self.role_slo(model.spec.role)
        saturated = self.probe(model, None)
        top = saturated.request_throughput / self.turns
        if saturated.completed == 0 or top <= 0:
            return Capacity(0.0, 0.0, None, _limit(saturated, slo))
        result = self.probe(model, top)
        low, high, best, failure = 0.0, top, None, result
        if self.passes(model, top, result):
            low, best = top, result
            for _ in range(EXPAND_STEPS):
                result = self.probe(model, low * 2)
                if not self.passes(model, low * 2, result):
                    high, failure = low * 2, result
                    break
                low, best = low * 2, result
            else:
                return Capacity(low, top, best, None)
        for _ in range(BISECTION_STEPS):
            middle = (low + high) / 2
            result = self.probe(model, middle)
            if self.passes(model, middle, result):
                low, best = middle, result
            else:
                high, failure = middle, result
        return Capacity(low, top, best, _limit(failure, slo))

    def simulate(self, candidate: Candidate) -> SimResult:
        requests = self._requests(self.rate, self.verify_rows)
        common: dict[str, Any] = {
            "dp_aware": candidate.dp_aware,
            "max_concurrency": self.workload.max_concurrency,
            "slo": self.slo,
            "seed": self.options.seed,
        }
        if candidate.disaggregated:
            prefill, decode = candidate.pools["prefill"], candidate.pools["decode"]
            sim = ClusterSim(
                prefill=[prefill.model] * prefill.count,
                decode=[decode.model] * decode.count,
                prefill_policy=candidate.policy,
                decode_policy="round_robin",
                transfer_bytes_per_us=self.transfer_bytes_per_us,
                **common,
            )
        else:
            pool = candidate.pools["colocated"]
            sim = ClusterSim(colocated=[pool.model] * pool.count, policy=candidate.policy, **common)
        return sim.run(requests)

    # ------------------------------------------------------------------ colocated
    def _dp_aware(self, spec: InstanceSpec) -> bool:
        return spec.attn_dp > 1 and self.workload.shared_prefix is not None

    def _evaluate(
        self, ctx: FeatureContext, combined: CombinedPlan, extra: dict[str, Any] | None = None
    ) -> _Option | None:
        model = self.colocated_model(ctx, combined, extra)
        if model is None:
            return None
        return _Option(model, self.capacity(model))

    def _mixed_worth_trying(self, spec: InstanceSpec, limit: str | None) -> bool:
        return (
            self.options.allow_mixed_chunk
            and self.slo.tpot_ms is not None
            and spec.speculative is None
            and spec.chunk_per_rank > 0
            and limit == "tpot"
        )

    def _mixed_variants(self, spec: InstanceSpec) -> list[dict[str, Any]]:
        return [
            {"--enable-mixed-chunk": True, "--chunked-prefill-size": chunk * spec.attn_dp}
            for chunk in MIXED_CHUNK_SIZES
            if chunk < spec.chunk_per_rank
        ]

    def _best_option(self, size: int, rate: float) -> _Option | None:
        """The tuned instance, its shorter-prefill variants when TTFT limits it, then mixed chunk."""
        ctx, combined = self.tuned(size, rate)
        default = self._evaluate(ctx, combined)
        if default is None or default.capacity.limit == "ttft":
            tuned = default
            unit = rate_unit(self.workload)
            for label, variant in self.ttft_variants(size, rate):
                option = self._evaluate(ctx, variant)
                if option is None or (
                    default is not None and option.capacity.rate <= default.capacity.rate
                ):
                    continue
                if tuned is not None:
                    option.notes.append(
                        f"TTFT 限制了调优配置 {tuned.model.spec.describe()}（单实例 "
                        f"{tuned.capacity.rate:.3g} {unit}）：{label}（{variant.layout.describe()}）"
                        f"后单实例 {option.capacity.rate:.3g} {unit}"
                    )
                default, combined = option, variant
        if default is None or not self._mixed_worth_trying(
            default.model.spec, default.capacity.limit
        ):
            return default
        best = previous = default
        for extra in self._mixed_variants(default.model.spec):
            option = self._evaluate(ctx, combined, extra)
            if option is None:
                continue
            option.notes = list(default.notes)
            if option.capacity.rate > best.capacity.rate:
                best = option
            if option.capacity.rate < previous.capacity.rate:
                break
            previous = option
        return best

    def colocated_candidates(self) -> list[Candidate]:
        assert self.rate is not None
        candidates = []
        for size in self.sizes:
            _, combined = self.tuned(size, self.rate)
            option = self._best_option(size, self.rate)
            if option is None:
                self.notes.append(f"TP{size}：调优后的配置放不下 KV 池，跳过")
                continue
            retuned = False
            if option.capacity.rate > 0:
                replicas = math.ceil(self.rate / option.capacity.rate - 1e-9)
                if replicas > 1:
                    share_rate = self.rate / replicas
                    _, share = self.tuned(size, share_rate)
                    if (share.layout, share.flags) != (combined.layout, combined.flags):
                        alternative = self._best_option(size, share_rate)
                        if (
                            alternative is not None
                            and alternative.capacity.rate > option.capacity.rate
                        ):
                            option, retuned = alternative, True
            capacity = option.capacity
            count = math.ceil(self.rate / capacity.rate - 1e-9) if capacity.rate > 0 else 0
            candidate = Candidate(
                pools={"colocated": Pool("colocated", option.model, count, capacity)},
                dp_aware=self._dp_aware(option.model.spec),
                retuned=retuned,
                notes=list(option.notes),
            )
            if count == 0:
                candidate.notes.append(
                    f"单实例在任何到达率下都满足不了 SLO（{LIMIT_LABELS[capacity.limit or 'overload']}）"
                )
            elif candidate.num_gpus > self.budget:
                candidate.notes.append(f"需要 {candidate.num_gpus} 卡，超出 GPU 预算 {self.budget}")
            candidates.append(candidate)
        return candidates

    def _bottleneck_pool(self, candidate: Candidate) -> Pool:
        if not candidate.disaggregated:
            return candidate.pools["colocated"]
        assert candidate.result is not None
        prefill, decode = candidate.pools["prefill"], candidate.pools["decode"]
        if _limit(candidate.result, self.slo) == "tpot":
            return decode

        def headroom(pool: Pool) -> float:
            rate = pool.capacity.rate if pool.capacity else 0.0
            return pool.count * rate

        return decode if headroom(decode) < headroom(prefill) else prefill

    def verify(self, candidate: Candidate) -> None:
        """Simulate the whole cluster; add instances (at most ``MAX_ADDED``) while it misses."""
        if not candidate.feasible:
            return
        for attempt in range(MAX_ADDED + 1):
            candidate.result = self.simulate(candidate)
            if candidate.result.meets_slo or self.rate is None or attempt == MAX_ADDED:
                return
            pool = self._bottleneck_pool(candidate)
            if candidate.num_gpus + pool.spec.num_gpus > self.budget:
                candidate.notes.append(f"再加一个 {pool.role} 实例就超出 GPU 预算 {self.budget}")
                return
            pool.count += 1
            candidate.added += 1

    # ------------------------------------------------------------------ PD
    def pd_candidate(self, colocated: Candidate | None) -> Candidate | None:
        assert self.rate is not None
        if not self.options.allow_pd:
            return None
        if not self.hardware.software.rdma:
            self.notes.append(
                "硬件 YAML 没有声明 RDMA（software.rdma: false）：PD 的 KV 只能走 TCP（mooncake_tcp），不评估 PD"
            )
            return None
        smallest = min(self.sizes)
        if self.budget < 2 * smallest:
            return None
        if colocated is not None and colocated.meets_slo and colocated.num_gpus <= 2 * smallest:
            self.notes.append(
                f"colocated 只需 {colocated.num_gpus} 卡，PD 至少要一个 prefill + 一个 decode 实例，不评估 PD"
            )
            return None
        self.pd_evaluated = True
        best: dict[str, tuple[InstanceModel, Capacity] | None] = {"prefill": None, "decode": None}
        seen: set[str] = set()

        def consider(role: str, model: InstanceModel | None) -> None:
            if model is None:
                return
            key = json.dumps([role, model.spec.flags, model.spec.env], sort_keys=True, default=str)
            if key in seen:
                return
            seen.add(key)
            capacity = self.capacity(model)
            if capacity.rate <= 0:
                return
            current = best[role]
            per_gpu = capacity.rate / model.spec.num_gpus
            if current is None or per_gpu > current[1].rate / current[0].spec.num_gpus:
                best[role] = (model, capacity)

        for size in self.sizes:
            ctx, combined = self.tuned(size, self.rate)
            consider("decode", self.role_model(ctx, combined, "decode"))
        if best["decode"] is None:
            self.notes.append("PD：没有一个 decode 实例规模能满足 TPOT SLO")
            return None
        decode_model, decode_capacity = best["decode"]
        for size in self.sizes:
            ctx, combined = self.tuned(size, self.rate)
            for plan in [
                combined,
                *(variant for _, variant in self.ttft_variants(size, self.rate)),
            ]:
                model = self.role_model(ctx, plan, "prefill", decode_model.spec)
                if (
                    model is not None
                    and model.spec.layout.attn_cp > 1
                    and decode_model.spec.layout.dcp > 1
                ):
                    continue
                consider("prefill", model)
        if best["prefill"] is None:
            self.notes.append("PD：没有一个 prefill 实例规模能满足 TTFT SLO")
            return None
        prefill_model, prefill_capacity = best["prefill"]
        pools = {
            "prefill": Pool(
                "prefill",
                prefill_model,
                math.ceil(self.rate / prefill_capacity.rate - 1e-9),
                prefill_capacity,
            ),
            "decode": Pool(
                "decode",
                decode_model,
                math.ceil(self.rate / decode_capacity.rate - 1e-9),
                decode_capacity,
            ),
        }
        return Candidate(pools=pools, dp_aware=self._dp_aware(prefill_model.spec))

    # ------------------------------------------------------------------ throughput mode
    def throughput_candidates(self) -> list[Candidate]:
        candidates = []
        for size in self.sizes:
            ctx, combined = self.tuned(size, None)
            model = self.colocated_model(ctx, combined)
            if model is None:
                self.notes.append(f"TP{size}：调优后的配置放不下 KV 池，跳过")
                continue
            count = self.budget // size
            candidate = Candidate(
                pools={"colocated": Pool("colocated", model, count)},
                dp_aware=self._dp_aware(model.spec),
            )
            candidate.result = self.simulate(candidate)
            if not candidate.meets_slo and self._mixed_worth_trying(
                model.spec, _limit(candidate.result, self.slo)
            ):
                for extra in self._mixed_variants(model.spec):
                    mixed = self.colocated_model(ctx, combined, extra)
                    if mixed is None:
                        continue
                    variant = Candidate(
                        pools={"colocated": Pool("colocated", mixed, count)},
                        dp_aware=candidate.dp_aware,
                    )
                    variant.result = self.simulate(variant)
                    if variant.meets_slo:
                        candidate = variant
                        break
            candidates.append(candidate)
        return candidates

    # ------------------------------------------------------------------ choice
    def choose(self, candidates: list[Candidate]) -> Candidate | None:
        percentile = self.slo.percentile
        simulated = [c for c in candidates if c.result is not None]
        if self.rate is None:
            if not simulated:
                return None
            return max(
                simulated,
                key=lambda c: (c.meets_slo, c.result.output_throughput if c.result else 0.0),
            )
        good = [c for c in simulated if c.meets_slo and not c.disaggregated]
        colocated = (
            min(
                good,
                key=lambda c: (
                    c.num_gpus,
                    -(c.result.slo_attainment if c.result else 0.0),
                    c.result.ttft_ms[percentile] if c.result else math.inf,
                ),
            )
            if good
            else None
        )
        pd = next((c for c in simulated if c.disaggregated), None)
        if pd is not None and pd.meets_slo:
            if colocated is None:
                self.reasons.append("colocated 配置都满足不了 SLO，PD 分离可以")
                return pd
            if pd.num_gpus <= colocated.num_gpus * (1 - PD_MIN_SAVING):
                self.reasons.append(
                    f"PD 分离只需 {pd.num_gpus} 卡，比最省的 colocated（{colocated.num_gpus} 卡）少 "
                    f"{1 - pd.num_gpus / colocated.num_gpus:.0%}"
                )
                return pd
            self.reasons.append(
                f"PD 分离要 {pd.num_gpus} 卡，比 colocated 的 {colocated.num_gpus} 卡省不到 "
                f"{PD_MIN_SAVING:.0%}，选更简单的 colocated"
            )
        if colocated is not None:
            return colocated
        if not simulated:
            return None
        return max(
            simulated,
            key=lambda c: (c.result.slo_attainment if c.result else 0.0, -c.num_gpus),
        )

    def _slo_prompt(self) -> tuple[int, int]:
        """Prompt length at the SLO percentile and the part of it a shared prefix covers."""
        prefix = self.workload.shared_prefix
        if prefix is None:
            low, high = length_bounds(self.workload.input_len, self.workload.range_ratio)
            return int(low + (high - low) * self.slo.percentile / 100), 0
        return prefix.system_prompt_len + prefix.question_len, prefix.system_prompt_len

    def capacity_bottlenecks(self, chosen: Candidate) -> list[str]:
        """Why the instances of a plan that meets the SLO still carry so little each."""
        slo = self.slo
        prompt, cached = self._slo_prompt()
        lines = []
        for pool in chosen.pools.values():
            capacity = pool.capacity
            if capacity is None or capacity.limit != "ttft" or slo.ttft_ms is None:
                continue
            ttft = pool.model.ctx.prefill_ms(pool.spec.layout, prompt, prefix_len=cached)
            lines.append(
                f"{pool.spec.describe()} 单实例只能承载 {capacity.rate:.3g} "
                f"{rate_unit(self.workload)}（瓶颈：TTFT）："
                f"P{slo.percentile} 长度的 prompt（{prompt} token）单独 prefill 就要 {ttft / 1000:.1f} s，"
                f"占 TTFT SLO {slo.ttft_ms / 1000:g} s 的 {ttft / slo.ttft_ms:.0%}；排在另一条 prefill "
                f"后面的请求就会超时，所以实例大部分时间必须空闲。要省卡只能放宽 TTFT SLO、"
                "缩短 prompt（例如截断召回的上下文）或给 prefill 用算力更高的卡"
            )
        return lines

    def bottlenecks(self) -> list[str]:
        """Lower bounds that no replica count can fix (one request alone on an idle instance)."""
        slo = self.slo
        percentile = slo.percentile
        prompt, cached = self._slo_prompt()
        lines = []
        for size in self.sizes:
            ctx, combined = self.tuned(size, self.rate)
            layout = combined.layout
            if slo.ttft_ms is not None:
                ttft = ctx.prefill_ms(layout, prompt, prefix_len=cached)
                if ttft > slo.ttft_ms:
                    lines.append(
                        f"{layout.describe()}：P{percentile} 长度的 prompt（{prompt} token"
                        + (f"，前缀 {cached} 已命中" if cached else "")
                        + f"）单独 prefill 就要 {ttft:.0f} ms，超过 TTFT SLO {slo.ttft_ms:g} ms"
                    )
            if slo.tpot_ms is not None:
                step = ctx.decode_step_ms(layout, 1)
                if step > slo.tpot_ms:
                    lines.append(
                        f"{layout.describe()}：batch 1 的一个 decode step 就要 {step:.1f} ms，"
                        f"超过 TPOT SLO {slo.tpot_ms:g} ms"
                    )
        if not lines:
            lines.append(
                "单请求的 prefill 和 decode step 都在 SLO 以内：瓶颈是排队，"
                "放宽 GPU 预算（--max-gpus）或降低到达率"
            )
        return lines

    # ------------------------------------------------------------------ output
    def plan(self) -> DeployPlan:
        if self.rate is None:
            candidates = self.throughput_candidates()
        else:
            candidates = self.colocated_candidates()
            shortlist = sorted(
                (c for c in candidates if c.feasible),
                key=lambda c: (
                    c.num_gpus,
                    -(c.pools["colocated"].capacity.rate if c.pools["colocated"].capacity else 0),
                ),
            )
            for candidate in shortlist[:VERIFY_TOP]:
                self.verify(candidate)
            verified = [c for c in shortlist if c.meets_slo]
            pd = self.pd_candidate(min(verified, key=lambda c: c.num_gpus) if verified else None)
            if pd is not None:
                self.verify(pd)
                candidates.append(pd)
        chosen = self.choose(candidates)
        return self._render(chosen, candidates)

    def _render(self, chosen: Candidate | None, candidates: list[Candidate]) -> DeployPlan:
        ctx = self.base
        platform = (
            f"{ctx.profile.platform}（树内）"
            if ctx.profile.in_tree
            else f"树外插件（device_type={ctx.device_type}）"
        )
        placements: list[Placement] = []
        nodes: dict[int, NodePlan] = {}
        router = bench = None
        checks: list[Command] = []
        bottlenecks: list[str] = []
        notes = list(self.notes)
        if chosen is not None and chosen.feasible:
            pools = list(chosen.pools.values())
            placements = place(pools, self.hardware.devices_per_node)
            by_role = {pool.role: pool for pool in pools}
            for placement in placements:
                pool = by_role[placement.role]
                for node, command in launch_commands(self.knowledge, pool, placement, self.options):
                    plan = nodes.setdefault(node, NodePlan(node, 0, []))
                    plan.commands.append(command)
                    plan.gpus_used += placement.gpus_per_node
            router = router_command(chosen, placements)
            front = f"http://<router_ip>:{ROUTER_PORT}" if router else placements[0].url
            bench = bench_command(
                self.workload,
                model_path=ctx.model_path,
                base_url=front,
                output_file=BENCH_OUTPUT,
                title="压测（与规划相同的负载）",
            )
            checks = self._checks(placements, front)
            if chosen.num_gpus > self.budget:
                notes.append(f"所选方案需要 {chosen.num_gpus} 卡，超出 GPU 预算 {self.budget}")
            if not chosen.meets_slo:
                bottlenecks = self.bottlenecks()
            elif chosen.num_gpus > self.budget:
                bottlenecks = self.capacity_bottlenecks(chosen)
            notes += self._layout_notes(chosen, placements)
        elif self.rate is not None:
            bottlenecks = self.bottlenecks()
        dropped, total = self.context_dropped
        if dropped:
            notes.append(
                f"负载里约 {dropped / total:.1%} 的请求输入+输出超过模型上下文长度 "
                f"{self.arch.max_position_embeddings}，SGLang tokenizer 会直接拒绝；规划时已剔除这些请求"
            )
        return DeployPlan(
            model=self.arch.name,
            hardware=self.hardware.name,
            workload=self.workload,
            platform=platform,
            mode="throughput" if self.rate is None else "rate",
            budget_gpus=self.budget,
            devices_per_node=self.hardware.devices_per_node,
            knowledge_commit=self.knowledge.commit,
            chosen=chosen,
            candidates=candidates,
            placements=placements,
            nodes=[nodes[key] for key in sorted(nodes)],
            router=router,
            bench=bench,
            checks=checks,
            steps=self._steps(chosen, router is not None),
            reasons=self.reasons,
            bottlenecks=bottlenecks,
            adaptation=self._adaptation(chosen, placements),
            notes=notes,
            assumptions=self._assumptions(chosen),
            references=self._references(chosen),
        )

    def _checks(self, placements: list[Placement], front: str) -> list[Command]:
        urls = " ".join(p.url for p in placements)
        wait = (
            f"for u in {urls}; do until curl -sf $u/health >/dev/null; do sleep 10; done; "
            "echo ready $u; done"
        )
        payload = {"text": "Hello", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}
        return [
            shell_command("等所有实例的 /health 返回 200", ["bash", "-c", wait]),
            shell_command(
                "冒烟请求",
                [
                    "curl",
                    "-s",
                    f"{front}/generate",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    json.dumps(payload, separators=(",", ":")),
                ],
            ),
        ]

    def _steps(self, chosen: Candidate | None, has_router: bool) -> list[str]:
        if chosen is None or not chosen.feasible:
            return []
        steps = [
            "把命令里的 `<nodeN_ip>` 换成第 N 台机器的 IP；跨机实例的各 node-rank 命令要在对应机器上同时启动",
            "等所有实例的 `/health` 返回 200（大模型加载权重和抓 CUDA graph 可能要几分钟）",
        ]
        if has_router:
            steps.append("启动 sglang_router（可以跑在任意一台能访问所有实例的机器上，CPU 即可）")
        steps += [
            "发一条冒烟请求确认链路通",
            f"跑压测命令，对比 P{self.slo.percentile} TTFT/TPOT 和吞吐与本方案的预测",
            "偏差超过 20% 时：用 `sgl-agent calibrate --log <server.log> --bench "
            f"{BENCH_OUTPUT}` 校准成本模型，再带 `--calibration` 重新跑 `sgl-agent deploy`",
        ]
        return steps

    def _layout_notes(self, chosen: Candidate, placements: list[Placement]) -> list[str]:
        notes = []
        pool = chosen.pools.get("colocated")
        if (
            pool is not None
            and pool.count > 1
            and pool.spec.attn_dp == 1
            and pool.num_gpus <= self.hardware.devices_per_node
        ):
            notes.append(
                f"单机内也可以用一个 `--dp-size {pool.count}` 的 launch_server 起 {pool.count} 个副本"
                "（内置 data parallel 调度，不需要 router）；`--dp-size` 不能和 `--attn-dp-size` 同时用。"
                "这里用独立实例 + sglang_router，便于跨机扩展和滚动重启"
            )
        per_node: dict[int, int] = {}
        for placement in placements:
            if len(placement.nodes) == 1:
                per_node[placement.nodes[0]] = per_node.get(placement.nodes[0], 0) + 1
        if any(count > 1 for count in per_node.values()):
            notes.append(
                f"同一台机器上的多个实例用 `--base-gpu-id` 分开 GPU，`--port` 相隔 {PORT_STRIDE}"
                "（DP attention 还会占用 --port+13 和 --port+233..239）"
            )
        for role, pool in chosen.pools.items():
            for line in pool.spec.notes:
                notes.append(f"{role}：{line}")
        if chosen.disaggregated:
            notes.append(
                "prefill 和 decode 必须用相同的 `--page-size` 和 `--kv-cache-dtype`，投机解码参数也要一致；"
                "decode 默认关闭 radix cache（所以不开 HiCache）"
            )
        return notes

    def _adaptation(self, chosen: Candidate | None, placements: list[Placement]) -> list[str]:
        if not self.base.out_of_tree:
            return []
        lines = []
        if chosen is not None and any(p.base_gpu_id for p in placements):
            lines.append(
                "多实例同机：SGLang 用 `torch.get_device_module(device).set_device(base_gpu_id + 本机 tp_rank)` "
                "选卡，设备的 PyTorch 后端要注册 device module 并支持按序号 set_device"
            )
        if (chosen is not None and chosen.disaggregated) or self.pd_evaluated:
            lines.append(
                "PD 的 KV 传输：mooncake/nixl 要能把该设备显存注册给 RDMA 网卡（相当于 GPUDirect RDMA）；"
                "做不到时参照 SGLang 的 `ascend` 后端（复用 Mooncake 的 KV manager，换成设备自己的 transfer engine）"
                "实现一个传输后端"
            )
        return lines

    def _assumptions(self, chosen: Candidate | None) -> list[str]:
        lines = [
            "所有耗时来自逐算子 roofline 成本模型（效率系数见硬件 YAML），上线前用实测日志校准（`sgl-agent calibrate`）",
            "调度按 SGLang 默认行为仿真：prefill 优先、FCFS、一次只切一个 chunked 请求、按 new-token ratio 预留 KV、"
            "装不下时按生成 token 最少的先撤回",
            "单实例容量 = 仿真中满足 SLO、且在途请求（Little 定律）装得进 KV 池的最高泊松到达率；"
            f"每次探测至少 {self.options.capacity_samples} 条请求，并加长到在途请求数的 "
            f"{STEADY_STATE_FACTOR} 倍（最多 {MAX_PROBE_SAMPLES} 条）以进入稳态，二分 {BISECTION_STEPS} 次；"
            f"整集群再按压测的请求数（{self.verify_rows} 条）仿真确认",
            "router 的 cache_aware 按共享前缀组和会话近似（真实 router 在文本前缀树上匹配）",
        ]
        if self.pd_evaluated or (chosen is not None and chosen.disaggregated):
            lines.append(
                f"PD 的 KV 传输按每卡 {self.hardware.inter_node_bandwidth_gbps:g} GB/s × 效率 "
                f"{self.hardware.efficiency.inter_node_comm:g} 计，每个请求另加 "
                f"{PD_TRANSFER_OVERHEAD_US / 1000:g} ms 握手开销（假设）"
            )
        if chosen is not None and any(p.spec.speculative for p in chosen.pools.values()):
            lines.append(
                f"投机解码接受率是负载 YAML 里的假设值（{self.workload.spec_accept_rate:g}），"
                "上线后以日志 accept len 为准"
            )
        return lines

    def _references(self, chosen: Candidate | None) -> list[str]:
        keys = ["scheduler_prefill_first", "chunked_prefill_single", "max_req_len", "retract"]
        keys += ["router_cache_aware", "multi_instance_ports"]
        if chosen is not None and any(p.spec.mixed_chunk for p in chosen.pools.values()):
            keys.append("mixed_chunk")
        if chosen is not None and chosen.disaggregated:
            keys += ["pd_decode_prealloc", "pd_decode_slots", "pd_kv_layout_match"]
            keys += ["pd_speculative_both_roles", "pd_router_bootstrap", "pd_transfer_backend"]
        if self.base.out_of_tree:
            keys.append("device_selection")
            if chosen is not None and chosen.disaggregated:
                keys.append("pd_ascend_transfer")
        return [source(key) for key in keys]


def plan_deployment(
    arch: ModelArch,
    hardware: HardwareSpec,
    knowledge: SGLangKnowledge,
    workload: Workload,
    *,
    options: DeployOptions | None = None,
    model_path: str | None = None,
    kv_dtype: str | None = None,
    host_memory_gb: float = DEFAULT_HOST_MEMORY_GB,
    calibration: Calibration | None = None,
) -> DeployPlan:
    """Plan the deployment of ``arch`` on ``hardware`` for ``workload`` (see the module doc)."""
    planner = _Planner(
        arch,
        hardware,
        knowledge,
        workload,
        options or DeployOptions(),
        model_path=model_path,
        kv_dtype=kv_dtype,
        host_memory_gb=host_memory_gb,
        calibration=calibration,
    )
    return planner.plan()
