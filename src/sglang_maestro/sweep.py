"""Sweep the bundled model × hardware × workload matrix and fingerprint the decisions.

Scenario 1 and 2 depend only on the model and device; scenario 3 and 5 also take the workload.
Fingerprints keep decisions and key numbers (floats at four significant digits), not prose, so a
baseline diff points at the field that moved. Invariants 1–5 run on every deployment plan.
Determinism (invariant 6) and monotonicity (invariant 7, report only) are exercised by the caller:
a second pass of the whole matrix would blow the time budget.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sglang_maestro.adapt.analysis import analyze_adaptation
from sglang_maestro.deploy.plan import (
    CAPACITY_SAMPLES,
    VERIFY_SAMPLES,
    DeployOptions,
    DeployPlan,
    launch_commands,
    plan_deployment,
)
from sglang_maestro.features.report import advise_features
from sglang_maestro.hardware.spec import list_presets
from sglang_maestro.markdown import bullets, table
from sglang_maestro.model.config import list_bundled_models
from sglang_maestro.perf.roofline import RooflineReport, analyze_roofline
from sglang_maestro.target import Target
from sglang_maestro.workload import list_workloads, load_workload

BASELINE_CAPACITY_SAMPLES = 40
"""Sample count the checked-in deployment baseline was generated with."""
BASELINE_VERIFY_SAMPLES = 120
BASELINE_SEED = 1
SIG_DIGITS = 4
DIFF_LINE_LIMIT = 80
SCENARIOS = ("adapt", "roofline", "features", "deploy")

INVARIANTS: dict[str, str] = {
    "commands": "每个节点的启动命令以及 router、bench、checks 的 problems 都为空",
    "placement": "同一节点上各实例的 GPU 区间不重叠、不超过 devices_per_node，且 port 与 bootstrap_port 不重复",
    "budget": "选出的方案超过 GPU 预算时，notes 里要有超预算说明",
    "slo": "有选出的方案时 meets_slo 为真",
    "no_plan": "没选出方案时 notes 或 bottlenecks 非空，且不能和「任何到达率下都满足不了 SLO」自相矛盾",
    "determinism": "同一组合跑两次的结果一致，且 --jobs 1 和 --jobs 2 的结果一致",
    "monotonicity": "到达率翻倍时选出的卡数不应变少（只报告，不算违反）",
}

_IMPOSSIBLE_AT_ANY_RATE = "任何到达率下都满足不了"
_QUEUE_BOTTLENECK = "瓶颈是排队"
_OVER_BUDGET = "超出 GPU 预算"


@dataclass(frozen=True)
class KnownViolation:
    combo: str
    id: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"combo": self.combo, "id": self.id, "reason": self.reason}


KNOWN_VIOLATIONS: tuple[KnownViolation, ...] = (
    KnownViolation(
        combo="qwen2.5-72b-instruct/template-h20-class/shared-prefix-agent",
        id="no_plan",
        reason=(
            "基线参数（capacity_samples=40、verify_samples=120、seed=1）下选不出方案，"
            "把 --request-rate 降到 0.001 也一样。bottlenecks 却写「单请求的 prefill 和 decode step "
            "都在 SLO 以内：瓶颈是排队，放宽 GPU 预算（--max-gpus）或降低到达率」，"
            "和候选「单实例在任何到达率下都满足不了 SLO」矛盾。"
            "默认参数（240/1200）下这个组合是 7×TP8、56 卡、满足 SLO。"
            "deploy/plan.py 的 _Planner.capacity() 只在 0 和饱和吞吐之间二分 "
            "BISECTION_STEPS（6）次；bottlenecks() 用 _slo_prompt() 估下限时假设共享前缀总能命中。"
            "推测（未验证）：小样本里第一次访问某个前缀组的冷请求占比高，P90 TTFT 被整段 prefill 撑高。"
            "规划器这次不改。"
        ),
    ),
)


def baseline_deploy_options() -> DeployOptions:
    return DeployOptions(
        capacity_samples=BASELINE_CAPACITY_SAMPLES,
        verify_samples=BASELINE_VERIFY_SAMPLES,
        seed=BASELINE_SEED,
    )


def bundled_names() -> tuple[list[str], list[str], list[str]]:
    return sorted(list_bundled_models()), sorted(list_presets()), sorted(list_workloads())


def combo_key(model: str, hardware: str, workload: str) -> str:
    return f"{model}/{hardware}/{workload}"


def sigfigs(value: float, digits: int = SIG_DIGITS) -> float | None:
    """Four significant digits, or None when the value is not finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(value)
    if not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    return float(f"{float(value):.{digits}g}")


def _canon(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        return sigfigs(value)
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {str(key): _canon(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canon(item) for item in value]
    raise TypeError(f"fingerprint value {value!r} ({type(value).__name__}) is not JSON-safe")


@dataclass(frozen=True)
class SweepTask:
    model: str
    hardware: str
    workload: str
    capacity_samples: int
    verify_samples: int
    seed: int
    scenarios: tuple[str, ...]
    request_rate: float | None = None

    @property
    def key(self) -> str:
        return combo_key(self.model, self.hardware, self.workload)


@dataclass
class SweepCell:
    key: str
    seconds: float
    fingerprints: dict[str, Any]
    violations: list[dict[str, str]]
    summary: dict[str, Any]
    notes: list[str] = field(default_factory=list)
    bottlenecks: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "seconds": self.seconds,
            "fingerprints": self.fingerprints,
            "violations": self.violations,
            "summary": self.summary,
            "notes": self.notes,
            "bottlenecks": self.bottlenecks,
            "error": self.error,
        }


@dataclass
class MonotonicityRow:
    key: str
    rate: float
    doubled_rate: float
    gpus: int | None
    doubled_gpus: int | None

    @property
    def decreased(self) -> bool:
        return self.gpus is not None and (
            self.doubled_gpus is None or self.doubled_gpus < self.gpus
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "rate": self.rate,
            "doubled_rate": self.doubled_rate,
            "gpus": self.gpus,
            "doubled_gpus": self.doubled_gpus,
            "decreased": self.decreased,
        }


@dataclass
class SweepResult:
    cells: dict[str, SweepCell]
    options: dict[str, Any]
    monotonicity: list[MonotonicityRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "options": self.options,
            "invariants": INVARIANTS,
            "known_violations": [item.to_dict() for item in KNOWN_VIOLATIONS],
            "cells": {key: self.cells[key].to_dict() for key in sorted(self.cells)},
            "monotonicity": [row.to_dict() for row in self.monotonicity],
            "unexpected_violations": self.unexpected_violations(),
            "missing_known": [item.to_dict() for item in self.missing_known()],
        }

    def violations(self) -> list[dict[str, str]]:
        found = []
        for key in sorted(self.cells):
            found.extend(self.cells[key].violations)
        return found

    def unexpected_violations(self) -> list[dict[str, str]]:
        return [item for item in self.violations() if not _is_known(item)]

    def missing_known(self) -> list[KnownViolation]:
        """Known violations that this run should have reproduced and did not."""
        if not _is_baseline_options(self.options):
            return []
        present = {(item["combo"], item["id"]) for item in self.violations()}
        return [
            item
            for item in KNOWN_VIOLATIONS
            if item.combo in self.cells and (item.combo, item.id) not in present
        ]

    @property
    def ok(self) -> bool:
        return not self.unexpected_violations() and not self.missing_known()


def _is_known(violation: dict[str, str]) -> bool:
    return any(
        item.combo == violation["combo"] and item.id == violation["id"] for item in KNOWN_VIOLATIONS
    )


def _is_baseline_options(options: dict[str, Any]) -> bool:
    return (
        options.get("capacity_samples") == BASELINE_CAPACITY_SAMPLES
        and options.get("verify_samples") == BASELINE_VERIFY_SAMPLES
        and options.get("seed") == BASELINE_SEED
        and options.get("max_gpus") is None
        and options.get("allow_pd", True) is True
        and options.get("allow_mixed_chunk", True) is True
    )


def _violation(plan_key: str, invariant_id: str, detail: str) -> dict[str, str]:
    return {"combo": plan_key, "id": invariant_id, "detail": detail}


def no_plan_detail(
    notes: Sequence[str], bottlenecks: Sequence[str], candidate_notes: Sequence[Sequence[str]]
) -> str | None:
    """Invariant 5. ``None`` means a plan was not required to explain itself here."""
    if not notes and not bottlenecks:
        return "没选出方案，但 notes 和 bottlenecks 都是空的"
    marked = candidate_notes and all(
        any(_IMPOSSIBLE_AT_ANY_RATE in note for note in notes_of_one)
        for notes_of_one in candidate_notes
    )
    if marked and bottlenecks and all(_QUEUE_BOTTLENECK in line for line in bottlenecks):
        return "所有候选都标着「单实例在任何到达率下都满足不了 SLO」，但 bottlenecks 只说瓶颈是排队"
    return None


def _check_commands(plan: DeployPlan, key: str) -> list[dict[str, str]]:
    found = []
    for node in plan.nodes:
        for command in node.commands:
            if command.problems:
                found.append(
                    _violation(
                        key, "commands", f"节点 {node.node} 的启动命令有参数问题：{command.title}"
                    )
                )
    for command in (plan.router, plan.bench, *plan.checks):
        if command is not None and command.problems:
            found.append(_violation(key, "commands", f"{command.title} 有参数问题"))
    return found


def _check_placement(plan: DeployPlan, key: str) -> list[dict[str, str]]:
    by_node: dict[int, list[Any]] = defaultdict(list)
    for placement in plan.placements:
        for node in placement.nodes:
            by_node[node].append(placement)
    found = []
    limit = plan.devices_per_node
    for node, items in sorted(by_node.items()):
        intervals = sorted(
            (item.base_gpu_id, item.base_gpu_id + item.gpus_per_node) for item in items
        )
        for start, end in intervals:
            if start < 0 or end > limit:
                found.append(
                    _violation(
                        key,
                        "placement",
                        f"节点 {node} 的 GPU 区间 [{start}, {end}) 超出 [0, {limit})",
                    )
                )
        for (start, end), (nxt, _) in zip(intervals, intervals[1:], strict=False):
            if nxt < end:
                found.append(
                    _violation(
                        key, "placement", f"节点 {node} 的 GPU 区间 [{start}, {end}) 与后面的重叠"
                    )
                )
        ports = [item.port for item in items]
        if len(ports) != len(set(ports)):
            found.append(_violation(key, "placement", f"节点 {node} 的 port 重复：{ports}"))
        bootstraps = [item.bootstrap_port for item in items if item.bootstrap_port is not None]
        if len(bootstraps) != len(set(bootstraps)):
            found.append(
                _violation(key, "placement", f"节点 {node} 的 bootstrap_port 重复：{bootstraps}")
            )
    return found


def _check_budget(plan: DeployPlan, key: str) -> list[dict[str, str]]:
    chosen = plan.chosen
    if chosen is None or chosen.num_gpus <= plan.budget_gpus:
        return []
    if any(_OVER_BUDGET in note for note in plan.notes):
        return []
    return [
        _violation(
            key,
            "budget",
            f"选用了 {chosen.num_gpus} 卡，预算 {plan.budget_gpus}，notes 里没有超预算说明",
        )
    ]


def _check_slo(plan: DeployPlan, key: str) -> list[dict[str, str]]:
    if plan.chosen is not None and not plan.chosen.meets_slo:
        return [_violation(key, "slo", "选出了方案，但 meets_slo 为假")]
    return []


def _check_no_plan(plan: DeployPlan, key: str) -> list[dict[str, str]]:
    if plan.chosen is not None:
        return []
    detail = no_plan_detail(
        plan.notes, plan.bottlenecks, [candidate.notes for candidate in plan.candidates]
    )
    if detail is None:
        return []
    return [_violation(key, "no_plan", detail)]


def check_plan(plan: DeployPlan, key: str) -> list[dict[str, str]]:
    """Invariants 1–5 for one deployment plan."""
    found: list[dict[str, str]] = []
    found += _check_commands(plan, key)
    found += _check_placement(plan, key)
    found += _check_budget(plan, key)
    found += _check_slo(plan, key)
    found += _check_no_plan(plan, key)
    return found


def _adapt_fingerprint(target: Target, hardware_ref: str) -> dict[str, Any]:
    report = analyze_adaptation(
        target.arch,
        target.hardware,
        target.knowledge,
        parallel=target.parallel,
        model_path=target.model_path,
        hardware_ref=hardware_ref,
    )
    precision = report.precision
    return _canon(
        {
            "support": {
                "supported": report.support.supported,
                "model_file": report.support.model_file,
            },
            "parallel": report.parallel.describe(),
            "precision": {
                "recommended": precision.recommended.key if precision.recommended else None,
                "routes": [[route.key, route.available] for route in precision.routes],
            },
            "hooks": [[item.name, item.level] for item in report.checklist.requirements],
            "operators": [[item.key, item.status] for item in report.inventory.requirements],
            "findings": [[item.id, item.severity] for item in report.findings],
        }
    )


def _roofline_fingerprint(target: Target) -> dict[str, Any]:
    report: RooflineReport = analyze_roofline(
        target.arch,
        target.hardware,
        target.layout(),
        target.execution(None),
    )
    return _canon(
        {
            "scenarios": [
                {
                    "label": item.label,
                    "phase": item.phase,
                    "batch_size": item.batch_size,
                    "total_ms": item.total_ms,
                    "tokens_per_s_per_gpu": item.tokens_per_s_per_gpu,
                }
                for item in report.scenarios
            ]
        }
    )


def _features_fingerprint(target: Target) -> dict[str, Any]:
    report = advise_features(target.feature_context())
    return _canon(
        {
            "advices": [[item.key, item.status] for item in report.advices],
            "combined": {
                "layout": report.combined.layout.describe(),
                "enabled": list(report.combined.enabled),
                "flags": report.combined.flags,
            },
        }
    )


def _deploy_fingerprint(plan: DeployPlan, knowledge: Any, options: DeployOptions) -> dict[str, Any]:
    chosen = plan.chosen

    def chosen_body() -> dict[str, Any] | None:
        if chosen is None:
            return None
        result = chosen.result
        return {
            "description": chosen.describe(),
            "num_gpus": chosen.num_gpus,
            "num_instances": chosen.num_instances,
            "disaggregated": chosen.disaggregated,
            "policy": chosen.policy,
            "meets_slo": chosen.meets_slo,
            "pools": [
                {
                    "role": pool.role,
                    "count": pool.count,
                    "capacity": None
                    if pool.capacity is None
                    else {"rate": pool.capacity.rate, "limit": pool.capacity.limit},
                }
                for _role, pool in sorted(chosen.pools.items())
            ],
            "result": None
            if result is None
            else {
                "ttft_ms": {"p90": result.ttft_ms.get(90)},
                "tpot_ms": {"p90": result.tpot_ms.get(90)},
                "goodput_rps": result.goodput_rps,
                "slo_attainment": result.slo_attainment,
            },
        }

    first: dict[str, Any] = {}
    if chosen is not None:
        for placement in plan.placements:
            current = first.get(placement.role)
            if current is None or placement.index < current.index:
                first[placement.role] = placement
    pools = {pool.role: pool for pool in chosen.pools.values()} if chosen is not None else {}
    commands: dict[str, list[str]] = {}
    for role, placement in sorted(first.items()):
        launched = launch_commands(knowledge, pools[role], placement, options)
        _node, command = min(launched, key=lambda item: item[0])
        commands[role] = command.argv
    return _canon(
        {
            "mode": plan.mode,
            "budget_gpus": plan.budget_gpus,
            "chosen": chosen_body(),
            "candidates": [
                {
                    "description": candidate.describe(),
                    "num_gpus": candidate.num_gpus,
                    "feasible": candidate.feasible,
                    "meets_slo": candidate.meets_slo,
                }
                for candidate in plan.candidates
            ],
            "commands": commands,
        }
    )


def _summary(plan: DeployPlan) -> dict[str, Any]:
    chosen = plan.chosen
    rates: list[str] = []
    limits: list[str] = []
    if chosen is not None:
        for role, pool in chosen.pools.items():
            prefix = "" if role == "colocated" else f"{role} "
            if pool.capacity is None:
                rates.append(f"{prefix}—")
                limits.append(f"{prefix}—")
                continue
            rates.append(f"{prefix}{pool.capacity.rate:.4g}")
            limits.append(f"{prefix}{pool.capacity.limit or '未饱和'}")
    result = chosen.result if chosen is not None else None
    return {
        "chosen": None if chosen is None else chosen.describe(),
        "num_gpus": None if chosen is None else chosen.num_gpus,
        "over_budget": bool(chosen is not None and chosen.num_gpus > plan.budget_gpus),
        "meets_slo": None if chosen is None else chosen.meets_slo,
        "capacity": "；".join(rates),
        "limit": "；".join(limits),
        "ttft_p90": None if result is None else sigfigs(result.ttft_ms.get(90, math.nan)),
        "tpot_p90": None if result is None else sigfigs(result.tpot_ms.get(90, math.nan)),
    }


def _run_task(task: SweepTask) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        target = Target.load(
            task.model,
            task.hardware,
            task.workload,
            request_rate=task.request_rate,
        )
        fingerprints: dict[str, Any] = {}
        violations: list[dict[str, str]] = []
        summary: dict[str, Any] = {}
        notes: list[str] = []
        bottlenecks: list[str] = []
        if "adapt" in task.scenarios:
            fingerprints["adapt"] = _adapt_fingerprint(target, task.hardware)
        if "roofline" in task.scenarios:
            fingerprints["roofline"] = _roofline_fingerprint(target)
        if "features" in task.scenarios:
            fingerprints["features"] = _features_fingerprint(target)
        if "deploy" in task.scenarios:
            options = DeployOptions(
                capacity_samples=task.capacity_samples,
                verify_samples=task.verify_samples,
                seed=task.seed,
            )
            plan = plan_deployment(
                target.arch,
                target.hardware,
                target.knowledge,
                target.require_workload(),
                options=options,
                model_path=target.model_path,
                kv_dtype=target.kv_dtype,
                host_memory_gb=target.host_memory_gb,
                calibration=target.calibration,
            )
            fingerprints["deploy"] = _deploy_fingerprint(plan, target.knowledge, options)
            violations = check_plan(plan, task.key)
            summary = _summary(plan)
            notes = list(plan.notes)
            bottlenecks = list(plan.bottlenecks)
        cell = SweepCell(
            key=task.key,
            seconds=round(time.perf_counter() - started, 3),
            fingerprints=fingerprints,
            violations=violations,
            summary=summary,
            notes=notes,
            bottlenecks=bottlenecks,
        )
        return cell.to_dict()
    except Exception as error:  # one bad combo must not kill the pool
        return SweepCell(
            key=task.key,
            seconds=round(time.perf_counter() - started, 3),
            fingerprints={},
            violations=[_violation(task.key, "commands", f"组合运行失败：{error}")],
            summary={},
            error=f"{type(error).__name__}: {error}",
        ).to_dict()


def _cell_from_dict(payload: dict[str, Any]) -> SweepCell:
    return SweepCell(
        key=payload["key"],
        seconds=payload["seconds"],
        fingerprints=payload["fingerprints"],
        violations=payload["violations"],
        summary=payload["summary"],
        notes=payload["notes"],
        bottlenecks=payload["bottlenecks"],
        error=payload["error"],
    )


def _execute(tasks: Sequence[SweepTask], jobs: int) -> list[SweepCell]:
    if jobs < 1:
        raise ValueError("--jobs 至少为 1")
    if jobs == 1 or len(tasks) <= 1:
        return [_cell_from_dict(_run_task(task)) for task in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        payloads = pool.map(_run_task, tasks)
    # Pool completion order must not leak into the result; callers key by combo.
    return [_cell_from_dict(payload) for payload in payloads]


def _tasks(
    models: Sequence[str],
    hardware: Sequence[str],
    workloads: Sequence[str],
    *,
    capacity_samples: int,
    verify_samples: int,
    seed: int,
    scenarios: tuple[str, ...],
    request_rate: float | None = None,
) -> list[SweepTask]:
    return [
        SweepTask(
            model,
            device,
            workload,
            capacity_samples,
            verify_samples,
            seed,
            scenarios,
            request_rate,
        )
        for model in models
        for device in hardware
        for workload in workloads
    ]


def _validate_names(
    models: Sequence[str], hardware: Sequence[str], workloads: Sequence[str]
) -> None:
    known_models, known_hardware, known_workloads = bundled_names()
    unknown_models = sorted(set(models) - set(known_models))
    unknown_hardware = sorted(set(hardware) - set(known_hardware))
    unknown_workloads = sorted(set(workloads) - set(known_workloads))
    if unknown_models or unknown_hardware or unknown_workloads:
        raise ValueError(
            "未知名称："
            + "，".join(
                part
                for part in (
                    f"模型 {unknown_models}" if unknown_models else "",
                    f"硬件 {unknown_hardware}" if unknown_hardware else "",
                    f"负载 {unknown_workloads}" if unknown_workloads else "",
                )
                if part
            )
        )


def _monotonicity(
    cells: dict[str, SweepCell], template: SweepTask, jobs: int
) -> list[MonotonicityRow]:
    ranked = sorted(cells, key=lambda key: (cells[key].seconds, key))
    picked: list[tuple[str, float]] = []
    for key in ranked:
        model, _device, workload_name = key.split("/")
        workload = load_workload(workload_name)
        if workload.request_rate:
            picked.append((key, workload.request_rate))
        if len(picked) == 2:
            break
    rows = []
    for key, rate in picked:
        model, device, workload_name = key.split("/")
        doubled = _execute(
            [
                SweepTask(
                    model,
                    device,
                    workload_name,
                    template.capacity_samples,
                    template.verify_samples,
                    template.seed,
                    ("deploy",),
                    request_rate=rate * 2,
                )
            ],
            jobs=1,
        )[0]
        current = cells[key].fingerprints.get("deploy") or {}
        other = doubled.fingerprints.get("deploy") or {}
        chosen = current.get("chosen") or {}
        doubled_chosen = other.get("chosen") or {}
        rows.append(
            MonotonicityRow(
                key=key,
                rate=rate,
                doubled_rate=rate * 2,
                gpus=chosen.get("num_gpus"),
                doubled_gpus=doubled_chosen.get("num_gpus"),
            )
        )
    return rows


def sweep(
    models: Sequence[str] | None = None,
    hardware: Sequence[str] | None = None,
    workloads: Sequence[str] | None = None,
    *,
    combos: Sequence[str] | None = None,
    capacity_samples: int = CAPACITY_SAMPLES,
    verify_samples: int = VERIFY_SAMPLES,
    seed: int = 1,
    jobs: int = 1,
    scenarios: tuple[str, ...] = SCENARIOS,
    check_monotonicity: bool = False,
) -> SweepResult:
    """Run the matrix. ``combos`` is ``model/hardware/workload`` and overrides the three lists."""
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"未知场景 {sorted(unknown)}")
    if combos is not None:
        tasks = []
        for combo in combos:
            model, device, workload = combo.split("/")
            tasks.append(
                SweepTask(
                    model, device, workload, capacity_samples, verify_samples, seed, scenarios
                )
            )
        _validate_names(
            [task.model for task in tasks],
            [task.hardware for task in tasks],
            [task.workload for task in tasks],
        )
    else:
        default_models, default_hardware, default_workloads = bundled_names()
        models = list(models) if models is not None else default_models
        hardware = list(hardware) if hardware is not None else default_hardware
        workloads = list(workloads) if workloads is not None else default_workloads
        _validate_names(models, hardware, workloads)
        tasks = _tasks(
            models,
            hardware,
            workloads,
            capacity_samples=capacity_samples,
            verify_samples=verify_samples,
            seed=seed,
            scenarios=scenarios,
        )
    cells = {cell.key: cell for cell in _execute(tasks, jobs)}
    options = {
        "capacity_samples": capacity_samples,
        "verify_samples": verify_samples,
        "seed": seed,
        "jobs": jobs,
        "scenarios": list(scenarios),
        "max_gpus": None,
        "allow_pd": True,
        "allow_mixed_chunk": True,
    }
    result = SweepResult(cells=cells, options=options)
    if check_monotonicity and "deploy" in scenarios and tasks:
        result.monotonicity = _monotonicity(cells, tasks[0], jobs)
    return result


def fingerprint_diff(
    expected: dict[str, Any], actual: dict[str, Any], *, limit: int = DIFF_LINE_LIMIT
) -> str:
    """Human-readable field diff: combo, field, old value, new value, at most ``limit`` lines."""
    lines: list[str] = []
    truncated = False

    def add(combo: str, field_name: str, old: Any, new: Any) -> None:
        nonlocal truncated
        if len(lines) >= limit:
            truncated = True
            return
        lines.append(f"{combo}  {field_name or '(组合)'}  旧值 {old!r}  新值 {new!r}")

    def walk(combo: str, path: str, old: Any, new: Any) -> None:
        nonlocal truncated
        if truncated or len(lines) >= limit:
            truncated = True
            return
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                if truncated or len(lines) >= limit:
                    truncated = True
                    return
                child = f"{path}.{key}" if path else str(key)
                if key not in old:
                    add(combo, child, "<缺失>", new[key])
                elif key not in new:
                    add(combo, child, old[key], "<缺失>")
                else:
                    walk(combo, child, old[key], new[key])
            return
        if isinstance(old, list) and isinstance(new, list):
            for index, (left, right) in enumerate(zip(old, new, strict=False)):
                walk(combo, f"{path}[{index}]", left, right)
                if truncated:
                    return
            if len(old) != len(new):
                add(combo, f"{path}.length", len(old), len(new))
            return
        if old != new:
            add(combo, path, old, new)

    for combo in sorted(set(expected) | set(actual)):
        if truncated or len(lines) >= limit:
            truncated = True
            break
        if combo not in expected:
            add(combo, "", "<缺失>", actual[combo])
        elif combo not in actual:
            add(combo, "", expected[combo], "<缺失>")
        else:
            walk(combo, "", expected[combo], actual[combo])
    if not lines:
        return ""
    text = "基线不一致：\n" + "\n".join(lines[:limit])
    if truncated:
        text += "\n... diff truncated"
    return text


def scenario_fingerprints(result: SweepResult, scenario: str) -> dict[str, Any]:
    return {
        key: cell.fingerprints[scenario]
        for key, cell in sorted(result.cells.items())
        if scenario in cell.fingerprints
    }


def write_baselines(result: SweepResult, directory: Path) -> None:
    """Write one sorted JSON file per scenario. Keys are ``model/hardware/workload``."""
    if not _is_baseline_options(result.options):
        raise ValueError(
            "基线必须用 capacity_samples="
            f"{BASELINE_CAPACITY_SAMPLES}、verify_samples={BASELINE_VERIFY_SAMPLES}、seed={BASELINE_SEED} 生成"
        )
    directory.mkdir(parents=True, exist_ok=True)
    for scenario in SCENARIOS:
        payload = scenario_fingerprints(result, scenario)
        if not payload:
            continue
        path = directory / f"{scenario}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def render_sweep(result: SweepResult) -> str:
    options = result.options
    lines = [
        "# 回归 sweep",
        "",
        (
            f"参数：capacity_samples={options['capacity_samples']}，"
            f"verify_samples={options['verify_samples']}，seed={options['seed']}，"
            f"jobs={options['jobs']}。"
        ),
        "数值来自成本模型和调度仿真，不是实测。",
        "",
        "## 组合",
        "",
        table(
            [
                "组合",
                "方案",
                "卡数",
                "超预算",
                "满足 SLO",
                "单实例容量",
                "瓶颈",
                "P90 TTFT",
                "P90 TPOT",
                "耗时 s",
            ],
            [
                [
                    key,
                    cell.summary.get("chosen"),
                    cell.summary.get("num_gpus"),
                    cell.summary.get("over_budget"),
                    cell.summary.get("meets_slo"),
                    cell.summary.get("capacity") or None,
                    cell.summary.get("limit") or None,
                    cell.summary.get("ttft_p90"),
                    cell.summary.get("tpot_p90"),
                    cell.seconds,
                ]
                for key, cell in sorted(result.cells.items())
            ],
        ),
        "",
        "## 不变量",
        "",
        bullets(f"`{invariant_id}` {text}" for invariant_id, text in INVARIANTS.items()),
        "",
        "## 不变量违反",
        "",
    ]
    violations = result.violations()
    if not violations:
        lines.append("没有。")
    else:
        lines.append(
            table(
                ["组合", "id", "说明", "登记"],
                [
                    [
                        item["combo"],
                        item["id"],
                        item["detail"],
                        "已知" if _is_known(item) else "未登记",
                    ]
                    for item in violations
                ],
            )
        )
    if any(_is_known(item) for item in violations):
        lines.extend(["", "已知违反（这次不修）：", ""])
        lines.append(
            bullets(
                f"`{item.combo}` `{item.id}`：{item.reason}"
                for item in KNOWN_VIOLATIONS
                if any(v["combo"] == item.combo and v["id"] == item.id for v in violations)
            )
        )
    missing = result.missing_known()
    if missing:
        lines.extend(["", "已登记、但这次没有出现（需要更新登记）：", ""])
        lines.append(bullets(f"`{item.combo}` `{item.id}`" for item in missing))
    lines.extend(["", "## 没选出方案", ""])
    missing_plans = [
        (key, cell)
        for key, cell in sorted(result.cells.items())
        if "deploy" in cell.fingerprints and cell.summary.get("chosen") is None
    ]
    if not missing_plans:
        lines.append("没有。")
    else:
        for key, cell in missing_plans:
            lines.append(f"### {key}")
            lines.append("")
            lines.append("notes：")
            lines.append(bullets(cell.notes) or "- （空）")
            lines.append("")
            lines.append("bottlenecks：")
            lines.append(bullets(cell.bottlenecks) or "- （空）")
            lines.append("")
    slowest = sorted(result.cells.items(), key=lambda item: item[1].seconds, reverse=True)[:5]
    lines.extend(
        [
            "## 最慢的 5 个组合",
            "",
            table(
                ["组合", "耗时 s"],
                [[key, cell.seconds] for key, cell in slowest],
            ),
            "",
            "## 单调性",
            "",
            INVARIANTS["monotonicity"],
            "",
        ]
    )
    if not result.monotonicity:
        lines.append("这次没有检查。")
    else:
        lines.append(
            table(
                ["组合", "到达率", "翻倍后", "卡数", "翻倍后卡数", "变少"],
                [
                    [row.key, row.rate, row.doubled_rate, row.gpus, row.doubled_gpus, row.decreased]
                    for row in result.monotonicity
                ],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_sweep(result: SweepResult, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sweep.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "sweep.md").write_text(render_sweep(result), encoding="utf-8")


def split_csv(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def default_jobs() -> int:
    return max(1, os.cpu_count() or 1)


def iter_known() -> Iterable[KnownViolation]:
    return KNOWN_VIOLATIONS
