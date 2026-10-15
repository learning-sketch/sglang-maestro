"""What a command works on: a model on a device, optionally a workload, a fixed layout and a
calibration. The CLI and ``sglang-maestro run`` resolve their inputs through here."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sglang_maestro.adapt.platform import resolve_platform
from sglang_maestro.adapt.precision import PrecisionRoute, precision_plan
from sglang_maestro.features.base import DEFAULT_HOST_MEMORY_GB, FeatureContext, build_context
from sglang_maestro.hardware.spec import HardwareSpec, load_hardware
from sglang_maestro.knowledge import SGLangKnowledge, load_knowledge
from sglang_maestro.model.config import ModelArch, load_model
from sglang_maestro.model.ops import Calibration, ExecutionConfig
from sglang_maestro.model.parallel import ParallelConfig
from sglang_maestro.perf.calibrate import load_calibration
from sglang_maestro.workload import Workload, load_workload

LAYOUT_FIELDS = ("tp", "pp", "attn_dp", "attn_cp", "dcp", "ep", "dense_mlp_tp", "moe_a2a")


def layout_from(values: Mapping[str, Any]) -> ParallelConfig | None:
    """A layout from ``tp``/``attn_dp``/... values; None when no ``tp`` is given."""
    given = {name: values[name] for name in LAYOUT_FIELDS if values.get(name) is not None}
    if "tp" not in given:
        if given:
            raise ValueError(f"给了 {', '.join(sorted(given))} 但没有给 --tp")
        return None
    return ParallelConfig(**given)


@dataclass
class Target:
    arch: ModelArch
    hardware: HardwareSpec
    knowledge: SGLangKnowledge
    workload: Workload | None = None
    parallel: ParallelConfig | None = None
    """Layout fixed by the user; None lets each scenario pick its own."""
    model_path: str | None = None
    kv_dtype: str | None = None
    host_memory_gb: float = DEFAULT_HOST_MEMORY_GB
    calibration: Calibration | None = None
    hardware_ref: str | None = None
    """What the user passed as ``--hardware`` (a preset name or a YAML path)."""

    @classmethod
    def load(
        cls,
        model: str,
        hardware: str,
        workload: str | None = None,
        *,
        parallel: ParallelConfig | None = None,
        model_path: str | None = None,
        kv_dtype: str | None = None,
        host_memory_gb: float | None = None,
        calibration: str | Path | Calibration | None = None,
        request_rate: float | None = None,
        knowledge: str | Path | None = None,
    ) -> Target:
        arch = load_model(model)
        loaded = load_workload(workload) if workload else None
        if loaded is not None and request_rate is not None:
            loaded = loaded.with_(request_rate=request_rate)
        if parallel is not None:
            errors = parallel.validate(arch)
            if errors:
                raise ValueError(
                    f"并行配置 {parallel.describe()} 不适用于 {arch.name}：" + "；".join(errors)
                )
        if isinstance(calibration, (str, Path)):
            calibration = load_calibration(calibration)
        return cls(
            arch=arch,
            hardware=load_hardware(hardware),
            knowledge=load_knowledge(knowledge),
            workload=loaded,
            parallel=parallel,
            model_path=model_path,
            kv_dtype=kv_dtype,
            host_memory_gb=host_memory_gb or DEFAULT_HOST_MEMORY_GB,
            calibration=calibration,
            hardware_ref=hardware,
        )

    def describe(self) -> str:
        text = f"{self.arch.name} @ {self.hardware.name}"
        if self.workload is not None:
            text += f"，负载 {self.workload.name}"
        return text

    def require_workload(self) -> Workload:
        if self.workload is None:
            raise ValueError(
                "这一步需要负载：传 --workload（`sglang-maestro list` 列出内置负载）或 YAML 路径"
            )
        return self.workload

    def route(self) -> PrecisionRoute | None:
        """Scenario 1's recommended precision route."""
        profile = resolve_platform(self.hardware, self.knowledge)
        return precision_plan(self.arch, self.hardware, profile).recommended

    def layout(self) -> ParallelConfig:
        """The user's layout, else the recommended precision route's."""
        if self.parallel is not None:
            return self.parallel
        route = self.route()
        if route is None or route.sizing is None:
            raise ValueError(
                f"{self.arch.name} 在 {self.hardware.name} 上没有可用的精度路线（见 sglang-maestro adapt），"
                "请用 --tp 指定布局"
            )
        return route.sizing.parallel

    def execution(self, weight_dtype: str | None = None) -> ExecutionConfig:
        """Cost-model precision: ``weight_dtype``, else the recommended route's weights."""
        if weight_dtype is None:
            route = self.route()
            weight_dtype = route.weight_override if route else None
        return ExecutionConfig(weight_dtype=weight_dtype, kv_dtype=self.kv_dtype)

    def feature_context(self, workload: Workload | None = None) -> FeatureContext:
        return build_context(
            self.arch,
            self.hardware,
            self.knowledge,
            workload or self.require_workload(),
            model_path=self.model_path,
            parallel=self.parallel,
            kv_dtype=self.kv_dtype,
            host_memory_gb=self.host_memory_gb,
            calibration=self.calibration,
        )
