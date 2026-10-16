"""The bundled capability snapshot must be reproducible from the SGLang commit it names."""

from __future__ import annotations

import difflib
import os
from pathlib import Path

import pytest

from sglang_maestro.cli import main
from sglang_maestro.knowledge import BUNDLED_SNAPSHOT

_DIFF_LINE_LIMIT = 80


def _snapshot_diff(expected: bytes, generated: bytes) -> str:
    diff = difflib.unified_diff(
        expected.decode("utf-8").splitlines(),
        generated.decode("utf-8").splitlines(),
        fromfile="sglang_capabilities.json",
        tofile="sync-knowledge",
        lineterm="",
        n=1,
    )
    lines: list[str] = []
    for index, line in enumerate(diff):
        if index >= _DIFF_LINE_LIMIT:
            lines.append("... diff truncated")
            break
        lines.append(line)
    return (
        f"内置快照与 sync-knowledge 输出不一致"
        f"（{len(expected)} 字节 vs {len(generated)} 字节）：\n" + "\n".join(lines)
    )


@pytest.mark.slow
def test_sync_knowledge_reproduces_bundled_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``sglang-maestro sync-knowledge`` on ``SGLANG_SRC`` matches the bundled snapshot byte for byte.

    CI runs this on the ``h20`` slow shard only. A local ``pytest -m slow`` with
    ``SGLANG_SRC`` set and no shard variable still runs it.
    """
    shard = os.environ.get("SGLANG_MAESTRO_SLOW_HARDWARE")
    if shard and shard != "h20":
        pytest.skip("快照可复现只在 h20 分片跑")
    sglang_src = os.environ.get("SGLANG_SRC")
    if not sglang_src:
        pytest.skip("未设置 SGLANG_SRC")
    output = tmp_path / "sglang_capabilities.json"
    status = main(["sync-knowledge", "--sglang-src", sglang_src, "-o", str(output)])
    captured = capsys.readouterr()
    assert status == 0, captured.err or captured.out
    generated = output.read_bytes()
    expected = BUNDLED_SNAPSHOT.read_bytes()
    if generated != expected:
        raise AssertionError(_snapshot_diff(expected, generated))
