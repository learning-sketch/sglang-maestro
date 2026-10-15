"""Shell commands for SGLang servers, the router, benchmarks and debug tools.

Every command is validated when it is built: server flags against the bundled snapshot of the
deployed SGLang, tool flags against the argparse definitions recorded in ``knowledge.facts``.
A command that would not parse carries its problems instead of failing later on the cluster.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any

from sglang_maestro.knowledge import SGLangKnowledge
from sglang_maestro.knowledge.facts import LAUNCH_SERVER_MODULE, TOOL_CHOICES, TOOL_FLAGS
from sglang_maestro.model.parallel import ParallelConfig

PYTHON = "python3"
DIST_INIT_PORT = 5000


@dataclass(frozen=True)
class Repeat:
    """Pass a flag once per item: ``Repeat(("http://p0 8998", "http://p1"))`` for ``--prefill``."""

    items: tuple[Any, ...]


@dataclass(frozen=True)
class Command:
    title: str
    program: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...] = ()
    """``("--tp-size", "8")``-style argument groups, one per rendered line."""
    env: tuple[tuple[str, str], ...] = ()
    problems: tuple[str, ...] = ()
    note: str = ""

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def argv(self) -> list[str]:
        return [*self.program, *(token for group in self.groups for token in group)]

    def shell(self, *, multiline: bool = True) -> str:
        head = " ".join(
            [*(f"{key}={shlex.quote(value)}" for key, value in self.env), shlex.join(self.program)]
        )
        if not multiline or not self.groups:
            return " ".join([head, *(shlex.join(group) for group in self.groups)])
        return " \\\n".join([head, *(f"  {shlex.join(group)}" for group in self.groups)])

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "argv": self.argv,
            "env": dict(self.env),
            "shell": self.shell(),
            "problems": list(self.problems),
            "note": self.note,
        }


def _render_value(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [json.dumps(value, separators=(",", ":"), ensure_ascii=False)]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return [str(int(value))]
    return [str(value)]


def flag_groups(flags: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    """``True`` renders a bare switch; ``None``/``False`` drop the flag; lists become values."""
    groups: list[tuple[str, ...]] = []
    for flag, value in flags.items():
        if value is None or value is False:
            continue
        if value is True:
            groups.append((flag,))
        elif isinstance(value, Repeat):
            for item in value.items:
                parts = item.split() if isinstance(item, str) else _render_value(item)
                groups.append((flag, *parts))
        else:
            groups.append((flag, *_render_value(value)))
    return tuple(groups)


def parallel_server_flags(parallel: ParallelConfig) -> dict[str, Any]:
    """``--tp-size`` and friends for one server; ``attn_dp`` is DP attention inside the TP group."""
    flags: dict[str, Any] = {"--tp-size": parallel.tp}
    if parallel.pp > 1:
        flags["--pp-size"] = parallel.pp
    if parallel.attn_dp > 1:
        flags["--attn-dp-size"] = parallel.attn_dp
    if parallel.attn_cp > 1:
        flags["--attn-cp-size"] = parallel.attn_cp
    if parallel.dcp > 1:
        flags["--dcp-size"] = parallel.dcp
    if parallel.ep > 1:
        flags["--ep-size"] = parallel.ep
    if parallel.moe_a2a_backend != "none":
        flags["--moe-a2a-backend"] = parallel.moe_a2a_backend
    if parallel.dense_mlp_tp:
        flags["--moe-dense-tp-size"] = parallel.dense_mlp_tp
    return flags


def multi_node_flags(num_gpus: int, devices_per_node: int, *, node_rank: int = 0) -> dict[str, Any]:
    """``--nnodes/--node-rank/--dist-init-addr`` when one instance spans several nodes."""
    if num_gpus <= devices_per_node:
        return {}
    nodes = -(-num_gpus // devices_per_node)
    return {
        "--nnodes": nodes,
        "--node-rank": node_rank,
        "--dist-init-addr": f"<node0_ip>:{DIST_INIT_PORT}",
    }


def _env(env: dict[str, Any] | None) -> tuple[tuple[str, str], ...]:
    return tuple((key, str(value)) for key, value in (env or {}).items() if value is not None)


def _active(flags: dict[str, Any]) -> dict[str, Any]:
    return {
        flag: value for flag, value in flags.items() if value is not None and value is not False
    }


def launch_server_command(
    knowledge: SGLangKnowledge,
    flags: dict[str, Any],
    *,
    env: dict[str, Any] | None = None,
    title: str = "启动 SGLang 服务",
    note: str = "",
) -> Command:
    active = _active(flags)
    problems = knowledge.validate_flags(active)
    if "--model-path" not in active:
        problems.append("缺少 --model-path")
    return Command(
        title=title,
        program=(PYTHON, "-m", LAUNCH_SERVER_MODULE),
        groups=flag_groups(flags),
        env=_env(env),
        problems=tuple(problems),
        note=note,
    )


def tool_command(
    module: str,
    flags: dict[str, Any],
    *,
    env: dict[str, Any] | None = None,
    title: str,
    note: str = "",
) -> Command:
    """``python3 -m <module> ...`` for a tool whose flags ``knowledge.facts.TOOL_FLAGS`` records."""
    known = TOOL_FLAGS.get(module)
    if known is None:
        raise KeyError(f"no recorded flags for {module!r}; known: {sorted(TOOL_FLAGS)}")
    problems: list[str] = []
    for flag, value in _active(flags).items():
        if flag not in known:
            problems.append(f"{flag} 不是 {module} 的参数")
            continue
        choices = TOOL_CHOICES.get((module, flag))
        if not choices or value is True:
            continue
        values = value.items if isinstance(value, Repeat) else value
        for single in values if isinstance(values, (list, tuple)) else (values,):
            if str(single) not in choices:
                problems.append(f"{module} {flag}={single} 不是合法取值 {list(choices)}")
    return Command(
        title=title,
        program=(PYTHON, "-m", module),
        groups=flag_groups(flags),
        env=_env(env),
        problems=tuple(problems),
        note=note,
    )


def shell_command(
    title: str,
    argv: list[str] | tuple[str, ...],
    *,
    env: dict[str, Any] | None = None,
    note: str = "",
) -> Command:
    """A command sglang-maestro cannot validate (curl, python -c, ...); rendered on one line."""
    return Command(title=title, program=tuple(argv), env=_env(env), note=note)
