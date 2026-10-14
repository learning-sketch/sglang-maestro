"""Small Markdown builders shared by the report renderers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sgl_agent.commands import Command
from sgl_agent.findings import Finding

SEVERITY_LABELS = {"high": "高", "medium": "中", "low": "低", "info": "提示"}
STATUS_LABELS = {
    "recommend": "推荐",
    "optional": "可选",
    "not_recommended": "不推荐",
    "unsupported": "不支持",
}


def _cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        text = f"{value:,.3g}" if abs(value) < 100 else f"{value:,.0f}"
    elif isinstance(value, bool):
        text = "是" if value else "否"
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(_cell(value) for value in row) + " |" for row in rows]
    return "\n".join(lines)


def bullets(items: Iterable[str], indent: int = 0) -> str:
    prefix = " " * indent + "- "
    return "\n".join(prefix + item for item in items if item)


def code(text: str, language: str = "bash") -> str:
    return f"```{language}\n{text.rstrip()}\n```"


def command_block(command: Command) -> str:
    parts = [f"**{command.title}**"]
    if command.note:
        parts.append(f"> {command.note}")
    parts.append(code(command.shell()))
    if command.problems:
        parts.append("⚠ 参数问题：\n" + bullets(command.problems))
    return "\n\n".join(parts)


def flags_text(flags: dict[str, Any]) -> str:
    rendered = []
    for flag, value in flags.items():
        if value is None or value is False:
            continue
        rendered.append(flag if value is True else f"{flag} {value}")
    return " ".join(rendered)


def finding_block(finding: Finding) -> str:
    severity = SEVERITY_LABELS.get(finding.severity, finding.severity)
    lines = [f"#### [{severity}] {finding.title}"]
    if finding.evidence:
        lines.append("证据：\n" + bullets(finding.evidence))
    if finding.recommendation:
        lines.append("建议：\n" + bullets(finding.recommendation))
    if finding.flags:
        lines.append(f"参数：`{flags_text(finding.flags)}`")
    if finding.env:
        lines.append("环境变量：`" + " ".join(f"{k}={v}" for k, v in finding.env.items()) + "`")
    if finding.gain_ms_per_step is not None:
        note = f"（{finding.gain_note}）" if finding.gain_note else ""
        lines.append(f"预计收益：每步约 {finding.gain_ms_per_step:.2f} ms{note}")
    if finding.references:
        lines.append("依据：" + "；".join(finding.references))
    return "\n\n".join(lines)


def findings_section(
    findings: Sequence[Finding], *, title: str = "发现", limit: int | None = None
) -> str:
    if not findings:
        return f"### {title}\n\n未发现问题。"
    shown = list(findings[:limit]) if limit else list(findings)
    body = "\n\n".join(finding_block(item) for item in shown)
    more = (
        f"\n\n（另有 {len(findings) - len(shown)} 条，见 JSON 输出）"
        if len(shown) < len(findings)
        else ""
    )
    return f"### {title}\n\n{body}{more}"
