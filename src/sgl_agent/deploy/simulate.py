"""Discrete-event simulation of a serving cluster: router, SGLang schedulers and PD transfers.

Each instance steps like SGLang's scheduler: a prefill batch whenever one can be built (FCFS,
one chunked request at a time, admitted against the KV reservation), otherwise one decode step
for every running request; with ``--enable-mixed-chunk`` the running batch decodes inside the
prefill batch instead. When the next decode step does not fit, the requests with the fewest
generated tokens are retracted and the new-token ratio is re-estimated. Attention-DP ranks keep
their own queues and radix caches and step together, so a step lasts as long as the slowest
rank. In PD mode the decode server reserves KV before the prefill server may start a request,
the prefill server streams the KV chunk by chunk and the first token reaches the client from the
decode server. Step times come from the cost model: calibrate before trusting absolute numbers.
"""

from __future__ import annotations

import heapq
import itertools
import math
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from sgl_agent.deploy.instance import InstanceModel
from sgl_agent.knowledge.facts import (
    CLIP_MAX_NEW_TOKENS,
    DP_ATTENTION_CONSERVATIVENESS_FACTOR,
    INIT_NEW_TOKEN_RATIO,
    MAX_REQ_INPUT_MARGIN,
    MIN_NEW_TOKEN_RATIO_FACTOR,
    NEW_TOKEN_RATIO_DECAY_STEPS,
    PD_RESERVED_DECODE_TOKENS,
    RETRACT_DECODE_STEPS,
    ROUTER_BALANCE_ABS_THRESHOLD,
    ROUTER_BALANCE_REL_THRESHOLD,
    ROUTER_CACHE_THRESHOLD,
)
from sgl_agent.workload import SLO, Request

ROUTER_POLICIES = ("round_robin", "cache_aware")
PD_TRANSFER_OVERHEAD_US = 500.0
"""Per-request KV transfer setup (bootstrap handshake, completion polling): an assumption."""
MAX_EVENTS = 5_000_000
PERCENTILES = (50, 90, 95, 99)
REJECTION_LABELS = {
    "prompt": "prompt 超过 max_req_input_len（KV 池或上下文长度装不下）",
    "context": "输入 + 输出超过模型上下文长度，tokenizer 直接拒绝",
    "decode_kv": "prompt 加预留 decode token 超过 decode 实例单个 rank 的 KV 池",
}


@dataclass(eq=False)
class SimRequest:
    rid: int
    sent: float
    input_len: int
    output_len: int
    group: int | None = None
    shared_len: int = 0
    session: int | None = None
    history: int = 0
    """Tokens of earlier turns (prompts and answers) after the shared prefix."""
    followups: tuple[int, ...] = ()
    generated: float = 0.0
    target: int = 0
    """Prompt tokens to have KV for at this (re)admission: the prompt plus regenerated output."""
    computed: int = 0
    hit: int = 0
    host_hit: int = 0
    private: int = 0
    decode_reserved: int = 0
    """KV tokens the PD decode server preallocated (the prefill side keeps its own ``private``)."""
    keys: tuple = ()
    reserve: float = 0.0
    first_token: float | None = None
    finish: float | None = None
    prefill_start: float | None = None
    worker: int = -1
    rank: int = -1
    decode_worker: int = -1
    decode_rank: int = -1
    epoch: int = 0
    join_step: int = 0
    gen_at_join: float = 0.0
    private_at_join: float = 0.0
    done_step: int = 0
    decoding: bool = False
    resume: bool = False
    retractions: int = 0
    transfer_us: float = 0.0
    prompt_tokens_total: int = 0

    def __post_init__(self) -> None:
        if self.target == 0:
            self.target = self.input_len


class PrefixCache:
    """Reusable prefixes on one rank: shared system prompts and conversation histories.

    One-off tokens of finished requests also stay in SGLang's radix tree, but nobody reuses
    them and they are evicted first, so they are treated as free space. With a host tier
    (HiCache write-through) every inserted prefix is also backed up on host.
    """

    def __init__(self, host_capacity: int = 0):
        self.device: OrderedDict[tuple, int] = OrderedDict()
        self.host: OrderedDict[tuple, int] = OrderedDict()
        self.locks: dict[tuple, int] = {}
        self.device_tokens = 0
        self.locked_tokens = 0
        self.host_capacity = host_capacity
        self.host_tokens = 0

    def lookup(
        self, chain: list[tuple[tuple, int]]
    ) -> tuple[list[tuple[tuple, int, bool]], int, int]:
        """Longest cached prefix of ``chain``: ``(hits, device tokens, host tokens)``."""
        hits: list[tuple[tuple, int, bool]] = []
        device = host = 0
        for key, needed in chain:
            if key in self.device:
                size = min(self.device[key], needed)
                hits.append((key, size, False))
                device += size
            elif key in self.host:
                size = min(self.host[key], needed)
                hits.append((key, size, True))
                host += size
            else:
                break
            if size < needed:
                break
        return hits, device, host

    def acquire(self, hits: list[tuple[tuple, int, bool]]) -> tuple:
        for key, size, from_host in hits:
            if from_host and key not in self.device:
                self._device_put(key, size)
            else:
                self.device.move_to_end(key)
            self._lock(key)
        return tuple(key for key, _, _ in hits)

    def pinned_tokens(self, hits: list[tuple[tuple, int, bool]]) -> int:
        """Evictable device tokens that locking ``hits`` would take out of the free budget."""
        return sum(
            self.device[key]
            for key, _, from_host in hits
            if not from_host and key in self.device and key not in self.locks
        )

    def release(self, keys: tuple) -> None:
        for key in keys:
            count = self.locks.get(key, 0) - 1
            if count <= 0:
                self.locks.pop(key, None)
                if key in self.device:
                    self.locked_tokens -= self.device[key]
            else:
                self.locks[key] = count

    def insert(self, key: tuple, size: int, *, lock: bool = False) -> tuple:
        if size <= 0:
            return ()
        old = self.device.pop(key, None)
        if old is not None:
            self.device_tokens -= old
            if key in self.locks:
                self.locked_tokens -= old
        self._device_put(key, size)
        if self.host_capacity:
            if key in self.host:
                self.host_tokens -= self.host.pop(key)
            self.host[key] = size
            self.host_tokens += size
            while self.host_tokens > self.host_capacity and self.host:
                _, dropped = self.host.popitem(last=False)
                self.host_tokens -= dropped
        if lock:
            self._lock(key)
            return (key,)
        return ()

    def evict(self, needed: float, protect: frozenset = frozenset()) -> float:
        """Drop least recently used unlocked prefixes until ``needed`` tokens are free."""
        freed = 0.0
        victims = []
        for key, size in self.device.items():
            if freed >= needed:
                break
            if key in self.locks or key in protect:
                continue
            victims.append(key)
            freed += size
        for key in victims:
            self.device_tokens -= self.device.pop(key)
        return freed

    def _device_put(self, key: tuple, size: int) -> None:
        self.device[key] = size
        self.device_tokens += size
        if key in self.locks:
            self.locked_tokens += size

    def _lock(self, key: tuple) -> None:
        count = self.locks.get(key, 0)
        if count == 0 and key in self.device:
            self.locked_tokens += self.device[key]
        self.locks[key] = count + 1


def _chain(req: SimRequest) -> list[tuple[tuple, int]]:
    chain: list[tuple[tuple, int]] = []
    if req.group is not None and req.shared_len > 0:
        chain.append((("g", req.group), req.shared_len))
        if req.session is not None and req.history > 0:
            chain.append((("s", req.session), req.history))
    return chain


@dataclass
class _PrefillPlan:
    entries: list[tuple[SimRequest, int]]
    duration_us: float
    decode_batch: int = 0
    """Running requests mixed into this batch (``--enable-mixed-chunk``)."""


@dataclass
class _DecodePlan:
    batch: int
    duration_us: float


class _Rank:
    """One attention-DP rank: its scheduler queues, KV pool and radix cache.

    Decoding requests advance ``tokens_per_step`` tokens per step, so per-request state is kept
    relative to the step it joined at and the batch totals are aggregate sums.
    """

    def __init__(self, instance: _Instance, index: int):
        self.instance = instance
        self.index = index
        spec = instance.model.spec
        self.tps = instance.model.tokens_per_step
        self.capacity = spec.kv_tokens_per_rank
        self.cache = PrefixCache(spec.host_cache_tokens_per_rank) if spec.radix_cache else None
        self.waiting: deque[SimRequest] = deque()
        self.chunked: SimRequest | None = None
        self.prefill_private = 0.0
        self.prefill_reserve = 0.0
        self.prefilling = 0
        self.step = 0
        self.decay_steps = 0
        """Pure decode steps: mixed steps advance requests but do not decay the ratio."""
        self.done_heap: list[tuple[int, int, int, SimRequest]] = []
        self.retract_heap: list[tuple[float, int, int, int, SimRequest]] = []
        self.n = 0
        self.s_ctx = 0.0
        self.s_priv = 0.0
        self.s_rem = 0.0
        self.s_gen = 0.0
        self.s_out = 0
        conservativeness = DP_ATTENTION_CONSERVATIVENESS_FACTOR if spec.attn_dp > 1 else 1.0
        self.ratio_init = min(INIT_NEW_TOKEN_RATIO * conservativeness, 1.0)
        self.ratio_min = min(self.ratio_init * MIN_NEW_TOKEN_RATIO_FACTOR, 1.0)
        self.ratio_decay = (self.ratio_init - self.ratio_min) / NEW_TOKEN_RATIO_DECAY_STEPS
        self.ratio_value = self.ratio_init
        self.ratio_mark = 0
        self.prealloc: deque[SimRequest] = deque()
        self.pending = 0
        self.pending_tokens = 0.0
        self.ready: deque[SimRequest] = deque()
        self.nic_free = 0.0
        self.decode_steps = 0
        self.decode_batch_sum = 0

    # ------------------------------------------------------------------ accounting
    def decode_private(self) -> float:
        return self.s_priv + self.n * self.tps * self.step

    def context_sum(self) -> float:
        return self.s_ctx + self.n * self.tps * self.step

    def new_token_ratio(self) -> float:
        steps = self.decay_steps - self.ratio_mark
        if steps <= 0:
            return self.ratio_value
        return max(self.ratio_value - self.ratio_decay * steps, self.ratio_min)

    def reset_ratio(self) -> None:
        self.ratio_value, self.ratio_mark = self.ratio_init, self.decay_steps

    def ratio_after_retract(self) -> None:
        decoded = self.s_gen + self.n * self.tps * self.step
        estimate = (decoded + RETRACT_DECODE_STEPS * self.n) / (self.s_out + 1)
        self.ratio_value, self.ratio_mark = min(1.0, estimate), self.decay_steps

    def running_reserve(self) -> float:
        remaining = max(self.s_rem - self.n * self.tps * self.step, 0.0)
        return remaining * self.new_token_ratio() + self.prefill_reserve

    def used(self) -> float:
        cached = self.cache.device_tokens if self.cache else 0
        return self.decode_private() + self.prefill_private + self.pending_tokens + cached

    def load_tokens(self) -> float:
        return self.context_sum() + sum(req.target for req in self.waiting)

    def active(self) -> int:
        return self.n + self.prefilling + len(self.waiting) + self.pending

    def fits(self, total: float, alloc: float, protect: frozenset) -> bool:
        """PrefillAdder's check: ``total`` tokens against free + evictable minus reservations."""
        locked = self.cache.locked_tokens if self.cache else 0
        committed = self.decode_private() + self.prefill_private + self.pending_tokens + locked
        if committed + self.running_reserve() + total > self.capacity:
            return False
        shortfall = self.used() + alloc - self.capacity
        if shortfall > 0 and self.cache:
            self.cache.evict(shortfall, protect)
        return True

    # ------------------------------------------------------------------ decode set
    def join_decode(self, req: SimRequest) -> bool:
        """Add ``req`` to the decode batch; False when it has nothing left to generate."""
        tps = self.tps
        remaining = req.output_len - req.generated
        steps = max(math.ceil(remaining / tps - 1e-9), 0)
        if steps == 0:
            return False
        offset = tps * self.step
        req.epoch += 1
        req.join_step = self.step
        req.gen_at_join = req.generated
        req.private_at_join = req.private
        req.done_step = self.step + steps
        req.decoding = True
        heapq.heappush(self.done_heap, (req.done_step, req.rid, req.epoch, req))
        heapq.heappush(
            self.retract_heap, (req.generated - offset, -req.input_len, req.rid, req.epoch, req)
        )
        self.n += 1
        self.s_ctx += req.input_len + req.generated - offset
        self.s_priv += req.private - offset
        self.s_rem += min(remaining, CLIP_MAX_NEW_TOKENS) + offset
        self.s_gen += req.generated - offset
        self.s_out += req.output_len
        return True

    def leave_decode(self, req: SimRequest) -> None:
        tps = self.tps
        offset = tps * req.join_step
        steps = self.step - req.join_step
        self.n -= 1
        if self.n == 0:
            self.s_ctx = self.s_priv = self.s_rem = self.s_gen = 0.0
            self.s_out = 0
        else:
            self.s_ctx -= req.input_len + req.gen_at_join - offset
            self.s_priv -= req.private_at_join - offset
            self.s_rem -= min(req.output_len - req.gen_at_join, CLIP_MAX_NEW_TOKENS) + offset
            self.s_gen -= req.gen_at_join - offset
            self.s_out -= req.output_len
        req.generated = min(float(req.output_len), req.gen_at_join + tps * steps)
        req.private = int(req.private_at_join + tps * steps)
        req.decoding = False

    def pop_finished(self) -> list[SimRequest]:
        done: list[SimRequest] = []
        heap = self.done_heap
        while heap and heap[0][0] <= self.step:
            _, _, epoch, req = heapq.heappop(heap)
            if req.decoding and req.epoch == epoch:
                self.leave_decode(req)
                done.append(req)
        limit = 4 * max(self.n, 16)
        if len(heap) > limit:
            self.done_heap = [e for e in heap if e[3].decoding and e[3].epoch == e[2]]
            heapq.heapify(self.done_heap)
        if len(self.retract_heap) > limit:
            self.retract_heap = [
                e for e in self.retract_heap if e[4].decoding and e[4].epoch == e[3]
            ]
            heapq.heapify(self.retract_heap)
        return done

    def retract_one(self) -> SimRequest | None:
        """``retract_decode`` order: fewest generated tokens first, longer prompts first."""
        while self.retract_heap:
            *_, epoch, req = heapq.heappop(self.retract_heap)
            if req.decoding and req.epoch == epoch:
                self.leave_decode(req)
                req.retractions += 1
                return req
        return None


class _Instance:
    def __init__(self, sim: ClusterSim, index: int, model: InstanceModel):
        self.sim = sim
        self.index = index
        self.model = model
        self.role = model.spec.role
        self.ranks = [_Rank(self, i) for i in range(model.spec.attn_dp)]
        self.busy = False
        self.plans: list[_PrefillPlan | _DecodePlan | None] = []
        self._rr = 0
        method = str(model.spec.flags.get("--load-balance-method", "auto"))
        if method == "auto":
            method = "follow_bootstrap_room" if self.role == "prefill" else "round_robin"
        self.balance = method

    def pick_rank(self, req: SimRequest) -> int:
        count = len(self.ranks)
        if count == 1:
            return 0
        if self.balance == "follow_bootstrap_room":
            return self.sim.rng.randrange(count)
        if self.balance == "total_requests":
            return min(range(count), key=lambda i: (self.ranks[i].active(), i))
        if self.balance == "total_tokens":
            return min(range(count), key=lambda i: (self.ranks[i].load_tokens(), i))
        rank = self._rr % count
        self._rr += 1
        return rank


class _Router:
    """``sglang_router`` policies over workers (instances, or DP ranks with ``--dp-aware``)."""

    def __init__(self, targets: list[tuple[int, int | None]], policy: str, rng: random.Random):
        if policy not in ROUTER_POLICIES:
            raise ValueError(f"unknown router policy {policy!r}; known: {ROUTER_POLICIES}")
        self.targets = targets
        self.index = {target: i for i, target in enumerate(targets)}
        self.policy = policy
        self.rng = rng
        self.load = [0] * len(targets)
        self.owner: dict[tuple, int] = {}
        self._rr = 0

    def pick(self, req: SimRequest) -> int:
        count = len(self.targets)
        if count == 1:
            choice = 0
        elif self.policy == "round_robin":
            choice = self._rr % count
            self._rr += 1
        else:
            choice = self._cache_aware(req)
        self.load[choice] += 1
        for key, _ in _chain(req):
            self.owner[key] = choice
        return choice

    def done(self, choice: int) -> None:
        self.load[choice] = max(self.load[choice] - 1, 0)

    def _least_loaded(self) -> int:
        low = min(self.load)
        return self.rng.choice([i for i, value in enumerate(self.load) if value == low])

    def _cache_aware(self, req: SimRequest) -> int:
        high, low = max(self.load), min(self.load)
        if high - low > ROUTER_BALANCE_ABS_THRESHOLD and high > low * ROUTER_BALANCE_REL_THRESHOLD:
            return self._least_loaded()
        matched, owner = 0, None
        for key, size in _chain(req):
            if key not in self.owner:
                break
            matched += size
            owner = self.owner[key]
        if owner is not None and matched / max(req.input_len, 1) > ROUTER_CACHE_THRESHOLD:
            return owner
        return self._least_loaded()


@dataclass
class SimResult:
    num_requests: int
    completed: int
    failed: int
    duration_s: float
    request_throughput: float
    output_throughput: float
    input_throughput: float
    ttft_ms: dict[int, float]
    tpot_ms: dict[int, float]
    e2e_ms: dict[int, float]
    mean_ttft_ms: float
    mean_tpot_ms: float
    mean_e2e_ms: float
    slo_attainment: float
    goodput_rps: float
    meets_slo: bool
    overloaded: bool
    cache_hit_rate: float
    mean_decode_batch_per_rank: float
    retractions: int
    mean_transfer_ms: float | None
    num_gpus: int
    rejections: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def output_throughput_per_gpu(self) -> float:
        return self.output_throughput / self.num_gpus if self.num_gpus else 0.0

    def to_dict(self) -> dict[str, Any]:
        transfer = self.mean_transfer_ms
        return {
            "num_requests": self.num_requests,
            "completed": self.completed,
            "failed": self.failed,
            "duration_s": round(self.duration_s, 2),
            "request_throughput": round(self.request_throughput, 3),
            "output_throughput": round(self.output_throughput, 1),
            "output_throughput_per_gpu": round(self.output_throughput_per_gpu, 1),
            "input_throughput": round(self.input_throughput, 1),
            "ttft_ms": {f"p{k}": round(v, 1) for k, v in self.ttft_ms.items()},
            "tpot_ms": {f"p{k}": round(v, 2) for k, v in self.tpot_ms.items()},
            "e2e_ms": {f"p{k}": round(v, 1) for k, v in self.e2e_ms.items()},
            "mean_ttft_ms": round(self.mean_ttft_ms, 1),
            "mean_tpot_ms": round(self.mean_tpot_ms, 2),
            "mean_e2e_ms": round(self.mean_e2e_ms, 1),
            "slo_attainment": round(self.slo_attainment, 4),
            "goodput_rps": round(self.goodput_rps, 3),
            "meets_slo": self.meets_slo,
            "overloaded": self.overloaded,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "mean_decode_batch_per_rank": round(self.mean_decode_batch_per_rank, 1),
            "retractions": self.retractions,
            "mean_transfer_ms": round(transfer, 2) if transfer is not None else None,
            "num_gpus": self.num_gpus,
            "rejections": self.rejections,
            "notes": self.notes,
        }


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (``numpy.percentile``'s default)."""
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class ClusterSim:
    """Colocated instances behind a router, or PD prefill and decode pools behind a PD router.

    ``prefill=[]`` with decode instances simulates the decode pool alone (KV arrives with the
    request); ``decode=[]`` with prefill instances simulates the prefill pool alone (requests
    end when their KV has been shipped). A simulator runs one request list.
    """

    def __init__(
        self,
        *,
        colocated: list[InstanceModel] | None = None,
        prefill: list[InstanceModel] | None = None,
        decode: list[InstanceModel] | None = None,
        policy: str = "round_robin",
        prefill_policy: str | None = None,
        decode_policy: str = "round_robin",
        dp_aware: bool = False,
        transfer_bytes_per_us: float = 0.0,
        max_concurrency: int | None = None,
        slo: SLO | None = None,
        seed: int = 1,
    ):
        self.rng = random.Random(seed)
        self.pd = colocated is None
        if self.pd and not (prefill or decode):
            raise ValueError("need colocated instances or at least one PD pool")
        if not self.pd and (prefill or decode):
            raise ValueError("colocated instances cannot be mixed with PD pools")
        self.colocated = [_Instance(self, i, m) for i, m in enumerate(colocated or [])]
        self.prefill = [_Instance(self, i, m) for i, m in enumerate(prefill or [])]
        self.decode = [_Instance(self, i, m) for i, m in enumerate(decode or [])]
        self.transfer_bytes_per_us = transfer_bytes_per_us
        self.max_concurrency = max_concurrency
        self.slo = slo or SLO()
        self.dp_aware = dp_aware
        front = self.colocated or self.prefill
        self.front_router = (
            _Router(self._targets(front), prefill_policy or policy, self.rng) if front else None
        )
        self.decode_router = (
            _Router([(i, None) for i in range(len(self.decode))], decode_policy, self.rng)
            if self.decode
            else None
        )
        models = [inst.model for inst in self.colocated + self.prefill + self.decode]
        self._max_req_len = min(model.max_req_len for model in models)
        contexts = [model.context_len for model in models if model.context_len]
        self._context_len = min(contexts) if contexts else 0
        self._decode_kv = min(
            (inst.model.spec.kv_tokens_per_rank for inst in self.decode), default=math.inf
        )
        self._events: list[tuple[float, int, str, Any]] = []
        self._seq = itertools.count()
        self._inflight = 0
        self._client: deque[SimRequest] = deque()
        self._finished: list[SimRequest] = []
        self._requests: list[SimRequest] = []
        self._rejections: dict[str, int] = {}
        self._rid = itertools.count()
        self._sessions = itertools.count()
        self._truncated = False
        self.now = 0.0

    def _targets(self, instances: list[_Instance]) -> list[tuple[int, int | None]]:
        if self.dp_aware:
            return [(inst.index, rank.index) for inst in instances for rank in inst.ranks]
        return [(inst.index, None) for inst in instances]

    # ------------------------------------------------------------------ driver
    def run(self, requests: list[Request]) -> SimResult:
        for request in requests:
            session = next(self._sessions) if request.followup_questions else None
            req = SimRequest(
                rid=next(self._rid),
                sent=request.arrival_s * 1e6,
                input_len=request.input_len,
                output_len=max(request.output_len, 1),
                group=request.group,
                shared_len=request.shared_len,
                session=session,
                followups=request.followup_questions,
            )
            self._push(req.sent, "arrival", req)
        handled = 0
        while self._events:
            time, _, kind, payload = heapq.heappop(self._events)
            self.now = time
            if kind == "arrival":
                self._arrive(payload)
            elif kind == "step":
                self._step_end(payload)
            else:
                self._transfer_done(payload)
            handled += 1
            if handled > MAX_EVENTS:
                self._truncated = True
                break
        return self._result()

    def _push(self, time: float, kind: str, payload: Any) -> None:
        heapq.heappush(self._events, (time, next(self._seq), kind, payload))

    # ------------------------------------------------------------------ requests
    def _rejection(self, req: SimRequest) -> str | None:
        """SGLang's length checks; clamps ``output_len`` the way the scheduler clamps it."""
        if self._context_len and req.input_len + req.output_len > self._context_len:
            return "context"
        if req.input_len >= self._max_req_len - MAX_REQ_INPUT_MARGIN:
            return "prompt"
        if req.input_len + 2 * PD_RESERVED_DECODE_TOKENS > self._decode_kv:
            return "decode_kv"
        req.output_len = max(1, min(req.output_len, self._max_req_len - req.input_len - 1))
        return None

    def _arrive(self, req: SimRequest) -> None:
        if self.max_concurrency is not None and self._inflight >= self.max_concurrency:
            self._client.append(req)
            return
        req.sent = self.now
        req.prompt_tokens_total = req.input_len
        self._requests.append(req)
        reason = self._rejection(req)
        if reason is not None:
            self._rejections[reason] = self._rejections.get(reason, 0) + 1
            self._release_client()
            return
        self._inflight += 1
        front = self.colocated or self.prefill
        if self.front_router is not None:
            worker, rank = self.front_router.targets[self.front_router.pick(req)]
            req.worker = worker
            req.rank = front[worker].pick_rank(req) if rank is None else rank
        if self.decode_router is not None:
            req.decode_worker = self.decode_router.pick(req)
            decode = self.decode[req.decode_worker]
            req.decode_rank = decode.pick_rank(req)
            decode.ranks[req.decode_rank].prealloc.append(req)
            self._prealloc(decode)
            return
        instance = front[req.worker]
        instance.ranks[req.rank].waiting.append(req)
        self._kick(instance)

    def _release_client(self) -> None:
        if self._client and (self.max_concurrency is None or self._inflight < self.max_concurrency):
            self._push(self.now, "arrival", self._client.popleft())

    def _finish(self, req: SimRequest) -> None:
        req.finish = self.now
        self._finished.append(req)
        self._inflight -= 1
        if self.decode_router is not None:
            self.decode_router.done(req.decode_worker)
        elif not self.pd and self.front_router is not None:
            self.front_router.done(self._front_choice(req))
        if req.followups:
            nxt = SimRequest(
                rid=next(self._rid),
                sent=self.now,
                input_len=req.input_len + req.output_len + req.followups[0],
                output_len=req.output_len,
                group=req.group,
                shared_len=req.shared_len,
                session=req.session,
                history=req.input_len + req.output_len - req.shared_len,
                followups=req.followups[1:],
            )
            self._push(self.now, "arrival", nxt)
        self._release_client()

    def _front_choice(self, req: SimRequest) -> int:
        router = self.front_router
        assert router is not None
        return router.index[(req.worker, req.rank if self.dp_aware else None)]

    # ------------------------------------------------------------------ PD decode admission
    def _prealloc(self, instance: _Instance) -> None:
        slots = instance.model.spec.request_slots_per_rank
        for rank in instance.ranks:
            while rank.prealloc and rank.n + rank.pending < slots:
                req = rank.prealloc[0]
                tokens = req.input_len + (int(req.generated) if req.resume else 0)
                active = rank.n + rank.pending
                free = rank.capacity - rank.used() - PD_RESERVED_DECODE_TOKENS * active
                if free < tokens + PD_RESERVED_DECODE_TOKENS and active > 0:
                    break
                rank.prealloc.popleft()
                rank.pending += 1
                rank.pending_tokens += tokens
                req.decode_reserved = tokens
                if req.resume or not self.prefill:
                    rank.ready.append(req)
                    self._kick(instance)
                else:
                    prefill = self.prefill[req.worker]
                    prefill.ranks[req.rank].waiting.append(req)
                    self._kick(prefill)

    # ------------------------------------------------------------------ stepping
    def _kick(self, instance: _Instance) -> None:
        if instance.busy:
            return
        plans = [self._plan(instance, rank) for rank in instance.ranks]
        durations = [plan.duration_us for plan in plans if plan is not None]
        if not durations:
            return
        instance.busy = True
        instance.plans = plans
        self._push(self.now + max(durations), "step", instance)

    def _plan(self, instance: _Instance, rank: _Rank) -> _PrefillPlan | _DecodePlan | None:
        plan: _PrefillPlan | _DecodePlan | None = None
        if instance.role != "decode":
            plan = self._plan_prefill(instance, rank)
        if plan is None and instance.role != "prefill":
            plan = self._plan_decode(instance, rank)
        if plan is None and rank.n == 0 and not rank.waiting:
            rank.reset_ratio()
        return plan

    def _plan_prefill(self, instance: _Instance, rank: _Rank) -> _PrefillPlan | None:
        model, spec = instance.model, instance.model.spec
        budget = spec.chunk_per_rank if spec.chunk_per_rank > 0 else math.inf
        input_budget = spec.max_prefill_tokens
        if spec.mixed_chunk and rank.n:
            budget -= rank.n
            input_budget -= rank.n
            if budget <= 0:
                return None
        entries: list[tuple[SimRequest, int]] = []
        chunks: list[tuple[int, int]] = []
        host_tokens = 0
        chunk_open = False
        if rank.chunked is not None:
            req = rank.chunked
            remaining = req.target - req.computed
            take = int(min(remaining, budget))
            entries.append((req, take))
            chunks.append((take, req.computed))
            budget -= take
            input_budget -= take
            chunk_open = take < remaining
            if not chunk_open:
                rank.chunked = None
        while rank.waiting and budget > 0 and input_budget > 0:
            if rank.n + rank.prefilling + 1 > spec.max_running_per_rank:
                break
            req = rank.waiting[0]
            hits: list[tuple[tuple, int, bool]] = []
            device = host = 0
            if rank.cache is not None:
                hits, device, host = rank.cache.lookup(_chain(req))
            hit = min(device + host, req.target - 1)
            if hit < device + host:
                host = max(hit - device, 0)
            extend = req.target - hit
            if extend > input_budget and entries:
                break
            reserve = model.admission_reserve(req.output_len - req.generated)
            protect = frozenset(key for key, _, _ in hits)
            pinned = rank.cache.pinned_tokens(hits) if rank.cache is not None else 0
            if not rank.fits(extend + host + reserve + pinned, extend + host, protect):
                break
            take = int(min(extend, budget))
            if take < extend and chunk_open:
                break
            rank.waiting.popleft()
            if rank.cache is not None:
                req.keys = rank.cache.acquire(hits)
            req.hit, req.host_hit, req.computed = hit, host, hit
            req.private = extend
            req.reserve = reserve
            req.prefill_start = self.now
            rank.prefill_private += extend
            rank.prefill_reserve += reserve
            rank.prefilling += 1
            entries.append((req, take))
            chunks.append((take, req.computed))
            host_tokens += host
            budget -= take
            input_budget -= take
            if take < extend:
                rank.chunked = req
                break
        if not entries:
            return None
        if spec.mixed_chunk and rank.n and rank.used() + rank.n <= rank.capacity:
            decode = (rank.n, rank.context_sum())
            return _PrefillPlan(entries, model.prefill_us(chunks, host_tokens, decode), rank.n)
        return _PrefillPlan(entries, model.prefill_us(chunks, host_tokens))

    def _plan_decode(self, instance: _Instance, rank: _Rank) -> _DecodePlan | None:
        spec = instance.model.spec
        while rank.ready and rank.n < spec.max_running_per_rank:
            req = rank.ready.popleft()
            rank.pending -= 1
            rank.pending_tokens -= req.decode_reserved
            req.private = req.decode_reserved
            if req.first_token is None:
                req.first_token = self.now
                req.generated = max(req.generated, 1.0)
            req.resume = False
            if not rank.join_decode(req):
                self._finish_decode_side(rank, req)
        if rank.n == 0:
            return None
        over = rank.used() + rank.n * rank.tps - rank.capacity
        if over > 0 and rank.cache is not None:
            over -= rank.cache.evict(over)
        if over > 0 and rank.n > 1:
            while rank.n > 1:
                req = rank.retract_one()
                if req is None:
                    break
                over -= req.private
                self._retracted(instance, rank, req)
                if over <= 0:
                    break
            rank.ratio_after_retract()
        return _DecodePlan(rank.n, instance.model.decode_us(rank.n, rank.context_sum()))

    def _retracted(self, instance: _Instance, rank: _Rank, req: SimRequest) -> None:
        if rank.cache is not None:
            rank.cache.release(req.keys)
        req.keys = ()
        req.private = 0
        if instance.role == "decode":
            req.resume = True
            rank.prealloc.appendleft(req)
            return
        req.target = req.input_len + int(req.generated)
        req.computed = 0
        rank.waiting.append(req)

    def _step_end(self, instance: _Instance) -> None:
        plans, instance.plans = instance.plans, []
        instance.busy = False
        for rank, plan in zip(instance.ranks, plans, strict=True):
            if plan is None:
                continue
            decoded = plan.batch if isinstance(plan, _DecodePlan) else plan.decode_batch
            if decoded:
                rank.step += 1
                rank.decode_steps += 1
                rank.decode_batch_sum += decoded
                if isinstance(plan, _DecodePlan):
                    rank.decay_steps += 1
                for req in rank.pop_finished():
                    self._finish_decode_side(rank, req)
            if isinstance(plan, _PrefillPlan):
                self._apply_prefill(instance, rank, plan)
        if instance.role == "decode":
            self._prealloc(instance)
        self._kick(instance)

    def _apply_prefill(self, instance: _Instance, rank: _Rank, plan: _PrefillPlan) -> None:
        for req, take in plan.entries:
            req.computed += take
            if req.computed < req.target:
                continue
            if rank.chunked is req:
                rank.chunked = None
            rank.prefilling -= 1
            rank.prefill_reserve -= req.reserve
            req.reserve = 0.0
            cache = rank.cache
            if cache is not None and req.group is not None and req.shared_len > req.hit:
                req.keys += cache.insert(("g", req.group), req.shared_len, lock=True)
                moved = min(req.shared_len - req.hit, req.private)
                req.private -= moved
                rank.prefill_private -= moved
            if instance.role == "prefill":
                self._start_transfer(instance, rank, req)
                continue
            rank.prefill_private -= req.private
            if req.first_token is None:
                req.first_token = self.now
            req.generated += 1
            if not rank.join_decode(req):
                self._finish_decode_side(rank, req)

    def _finish_decode_side(self, rank: _Rank, req: SimRequest) -> None:
        if rank.cache is not None:
            rank.cache.release(req.keys)
            if req.followups and req.session is not None:
                history = req.input_len + req.output_len - req.shared_len
                rank.cache.insert(("s", req.session), history)
        req.keys = ()
        self._finish(req)

    # ------------------------------------------------------------------ PD transfer
    def _start_transfer(self, instance: _Instance, rank: _Rank, req: SimRequest) -> None:
        """KV leaves chunk by chunk; the last chunk's pages go out after the final forward."""
        spec = instance.model.spec
        done = self.now
        if self.transfer_bytes_per_us > 0:
            per_token_us = spec.kv_bytes_per_token_per_gpu / self.transfer_bytes_per_us
            chunk = spec.chunk_per_rank if spec.chunk_per_rank > 0 else req.target
            last = max(min(chunk, req.target - req.hit), 1)
            start = max(req.prefill_start or self.now, rank.nic_free)
            done = max(self.now + last * per_token_us, start + req.input_len * per_token_us)
            rank.nic_free = done
        done += PD_TRANSFER_OVERHEAD_US
        req.transfer_us = done - self.now
        self._push(done, "transfer", (instance, rank, req))

    def _transfer_done(self, payload: tuple[_Instance, _Rank, SimRequest]) -> None:
        instance, rank, req = payload
        rank.prefill_private -= req.private
        if rank.cache is not None:
            rank.cache.release(req.keys)
            if req.followups and req.session is not None:
                rank.cache.insert(("s", req.session), req.input_len - req.shared_len)
        req.keys = ()
        if self.front_router is not None:
            self.front_router.done(self._front_choice(req))
        if self.decode:
            decode = self.decode[req.decode_worker]
            decode.ranks[req.decode_rank].ready.append(req)
            self._kick(decode)
        else:
            req.first_token = self.now
            req.generated = 1.0
            self._finish(req)
        self._kick(instance)

    # ------------------------------------------------------------------ metrics
    @property
    def decode_present(self) -> bool:
        return bool(self.colocated or self.decode)

    def _result(self) -> SimResult:
        done = [req for req in self._finished if req.first_token is not None]
        sent = len(self._requests) + len(self._client)
        failed = sent - len(done)
        instances = self.colocated + self.prefill + self.decode
        gpus = sum(inst.model.spec.num_gpus for inst in instances)
        notes = [
            f"{count} 个请求被拒绝：{REJECTION_LABELS[reason]}"
            for reason, count in sorted(self._rejections.items())
        ]
        if self._truncated:
            notes.append(f"事件数超过 {MAX_EVENTS}，仿真提前结束")
        if not done:
            return SimResult(
                num_requests=sent,
                completed=0,
                failed=failed,
                duration_s=0.0,
                request_throughput=0.0,
                output_throughput=0.0,
                input_throughput=0.0,
                ttft_ms=dict.fromkeys(PERCENTILES, math.inf),
                tpot_ms=dict.fromkeys(PERCENTILES, math.inf),
                e2e_ms=dict.fromkeys(PERCENTILES, math.inf),
                mean_ttft_ms=math.inf,
                mean_tpot_ms=math.inf,
                mean_e2e_ms=math.inf,
                slo_attainment=0.0,
                goodput_rps=0.0,
                meets_slo=False,
                overloaded=True,
                cache_hit_rate=0.0,
                mean_decode_batch_per_rank=0.0,
                retractions=0,
                mean_transfer_ms=None,
                num_gpus=gpus,
                rejections=dict(self._rejections),
                notes=notes or ["没有请求完成：KV 池装不下单条请求或配置无效"],
            )
        start = min(req.sent for req in done)
        end = max(req.finish or req.first_token or 0.0 for req in done)
        duration_s = max((end - start) / 1e6, 1e-9)
        ttft = [(req.first_token - req.sent) / 1000 for req in done]
        with_tpot = self.decode_present
        tpot_of = {
            req.rid: (req.finish - req.first_token) / 1000 / (req.output_len - 1)
            for req in done
            if with_tpot and req.output_len > 1 and req.finish is not None
        }
        tpot = list(tpot_of.values())
        e2e = [((req.finish or req.first_token) - req.sent) / 1000 for req in done]
        slo = self.slo
        check_ttft = slo.ttft_ms is not None and bool(self.colocated or self.prefill)
        good = 0
        for req, first in zip(done, ttft, strict=True):
            ok = not check_ttft or first <= slo.ttft_ms
            per_token = tpot_of.get(req.rid)
            if ok and slo.tpot_ms is not None and per_token is not None:
                ok = per_token <= slo.tpot_ms
            good += ok
        ttft_p = {p: percentile(ttft, p) for p in PERCENTILES}
        tpot_p = (
            {p: percentile(tpot, p) for p in PERCENTILES}
            if tpot
            else dict.fromkeys(PERCENTILES, 0.0)
        )
        overloaded = self._overloaded(done, ttft)
        meets = failed == 0 and not overloaded and not self._truncated
        if check_ttft:
            meets = meets and ttft_p[slo.percentile] <= slo.ttft_ms
        if slo.tpot_ms is not None and tpot:
            meets = meets and tpot_p[slo.percentile] <= slo.tpot_ms
        prompt = sum(req.prompt_tokens_total for req in done)
        hits = sum(req.hit for req in done)
        ranks = [rank for inst in instances for rank in inst.ranks]
        steps = sum(rank.decode_steps for rank in ranks)
        transfers = [req.transfer_us / 1000 for req in done if req.transfer_us > 0]
        return SimResult(
            num_requests=sent,
            completed=len(done),
            failed=failed,
            duration_s=duration_s,
            request_throughput=len(done) / duration_s,
            output_throughput=sum(req.output_len for req in done) / duration_s,
            input_throughput=prompt / duration_s,
            ttft_ms=ttft_p,
            tpot_ms=tpot_p,
            e2e_ms={p: percentile(e2e, p) for p in PERCENTILES},
            mean_ttft_ms=sum(ttft) / len(ttft),
            mean_tpot_ms=sum(tpot) / len(tpot) if tpot else 0.0,
            mean_e2e_ms=sum(e2e) / len(e2e),
            slo_attainment=good / len(done),
            goodput_rps=good / duration_s,
            meets_slo=meets,
            overloaded=overloaded,
            cache_hit_rate=hits / prompt if prompt else 0.0,
            mean_decode_batch_per_rank=(
                sum(rank.decode_batch_sum for rank in ranks) / steps if steps else 0.0
            ),
            retractions=sum(req.retractions for req in done),
            mean_transfer_ms=sum(transfers) / len(transfers) if transfers else None,
            num_gpus=gpus,
            rejections=dict(self._rejections),
            notes=notes,
        )

    def _overloaded(self, done: list[SimRequest], ttft: list[float]) -> bool:
        """TTFT keeps growing over the run: the queue never drains at this arrival rate.

        Not defined when every request is sent at once (throughput mode).
        """
        if len(done) < 20 or max(r.sent for r in done) - min(r.sent for r in done) <= 0:
            return False
        ordered = [value for _, value in sorted(zip((r.sent for r in done), ttft, strict=True))]
        quarter = max(len(ordered) // 4, 1)
        head = sum(ordered[:quarter]) / quarter
        tail = sum(ordered[-quarter:]) / quarter
        return tail - head > 1000 and tail > 2 * max(head, 1.0)
