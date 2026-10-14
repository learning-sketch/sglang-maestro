"""Structured findings shared by every analysis (adaptation, trace, features, deployment)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SEVERITIES = ("high", "medium", "low", "info")
_SEVERITY_RANK = {name: index for index, name in enumerate(SEVERITIES)}


@dataclass
class Finding:
    id: str
    kind: str
    severity: str
    title: str
    evidence: list[str] = field(default_factory=list)
    recommendation: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    gain_ms_per_step: float | None = None
    gain_note: str | None = None
    scope: str | None = None
    references: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.severity not in _SEVERITY_RANK:
            raise ValueError(f"unknown severity {self.severity!r}; known: {SEVERITIES}")

    def sort_key(self) -> tuple[int, float]:
        return (_SEVERITY_RANK[self.severity], -(self.gain_ms_per_step or 0.0))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def severity_for_share(share: float, high: float = 0.10, medium: float = 0.03) -> str:
    """Map an estimated gain (fraction of step time) to a severity."""
    if share >= high:
        return "high"
    if share >= medium:
        return "medium"
    return "low"


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=Finding.sort_key)
