"""State shared by the trace detectors (anomalies, fusion, overlap)."""

from __future__ import annotations

from dataclasses import dataclass, field

from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.knowledge import SGLangKnowledge
from sgl_agent.model.config import ModelArch
from sgl_agent.model.ops import StepCostModel
from sgl_agent.model.parallel import ParallelConfig
from sgl_agent.perf.expected import GroupExpectation
from sgl_agent.perf.metrics import GroupSummary, StepMetrics
from sgl_agent.perf.trace import KernelEvent, ParsedTrace, StepSpan, TraceDialect


@dataclass
class AnalysisContext:
    trace: ParsedTrace
    groups: list[GroupSummary]
    kernels_by_step: dict[int, list[KernelEvent]]
    metrics_by_step: dict[int, StepMetrics]
    knowledge: SGLangKnowledge
    dialect: TraceDialect = field(default_factory=TraceDialect)
    hardware: HardwareSpec | None = None
    arch: ModelArch | None = None
    parallel: ParallelConfig | None = None
    cost_model: StepCostModel | None = None
    expectations: dict[int, GroupExpectation] = field(default_factory=dict)
    """Keyed by ``id(group)``."""

    def __post_init__(self) -> None:
        self.steps_by_id: dict[int, StepSpan] = {step.step_id: step for step in self.trace.steps}
        self._step_by_position: dict[tuple[str, int], StepSpan] = {
            (step.rank, step.ordinal): step for step in self.trace.steps
        }

    @property
    def primary_rank(self) -> str:
        return self.trace.primary_rank

    def step_at(self, rank: str, ordinal: int) -> StepSpan | None:
        """The ``ordinal``-th step recorded on ``rank`` (ranks run the same steps in lockstep)."""
        return self._step_by_position.get((rank, ordinal))

    def aligned_steps(self, step: StepSpan) -> list[StepSpan]:
        """The same step on every rank, when all ranks ran the same stage and batch size."""
        aligned = [self.step_at(rank, step.ordinal) for rank in self.trace.ranks]
        if any(
            other is None or other.stage != step.stage or other.batch_size != step.batch_size
            for other in aligned
        ):
            return []
        return [other for other in aligned if other is not None]

    def primary_groups(self) -> list[GroupSummary]:
        return [group for group in self.groups if group.rank == self.primary_rank]

    @staticmethod
    def group_weight_us(group: GroupSummary) -> float:
        """Total GPU time the group represents in the trace (steps × median span)."""
        return group.steps * group.step_us

    def expectation(self, group: GroupSummary) -> GroupExpectation | None:
        return self.expectations.get(id(group))

    def representative_step_id(self, group: GroupSummary) -> int:
        """Step id whose span is closest to the group median."""
        target = group.step_us
        return min(
            group.step_ids, key=lambda step_id: abs(self.metrics_by_step[step_id].span_us - target)
        )

    @property
    def sm_major(self) -> int | None:
        capability = self.hardware.software.compute_capability if self.hardware else None
        if not capability:
            return None
        try:
            return int(str(capability).split(".")[0])
        except ValueError:
            return None

    @property
    def architecture(self) -> str | None:
        return self.arch.primary_architecture if self.arch else None

    def uses_dp_attention(self) -> bool:
        return bool(self.parallel and self.parallel.attn_dp > 1)
