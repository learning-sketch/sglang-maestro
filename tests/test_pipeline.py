from __future__ import annotations

import json

import pytest

from conftest import SMALL_DEPLOY
from sgl_agent.pipeline import RunOptions, run_pipeline, section
from sgl_agent.target import Target


def test_section_extracts_one_heading():
    markdown = "## 标题\n\n### 结论\n\n- a\n- b\n\n### 其它\n\n- c\n"
    assert section(markdown, "结论") == "- a\n- b"
    assert section(markdown, "缺失") == ""


def test_demo_run_closes_the_calibration_loop(tmp_path, dense_chat):
    result = run_pipeline(dense_chat, tmp_path, RunOptions(deploy=SMALL_DEPLOY, demo=True))
    keys = [stage.key for stage in result.stages]
    assert keys == ["adapt", "roofline", "trace", "features", "deploy", "calibrate", "replan"]
    assert not any(stage.error for stage in result.stages)
    assert result.calibration is not None
    assert result.calibration.decode_scale == pytest.approx(1.3, rel=0.02)
    assert result.calibration.prefill_scale == pytest.approx(1.15, rel=0.02)
    for stage in result.stages:
        assert (tmp_path / stage.file).is_file()
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "校准后" in report and "合成" in report
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert set(payload["stages"]) == set(keys)
    assert json.loads((tmp_path / "calibration.json").read_text(encoding="utf-8"))["source"]


def test_run_without_workload_stops_after_scenario_two(tmp_path):
    target = Target.load("qwen3-30b-a3b", "h20")
    result = run_pipeline(target, tmp_path, RunOptions(deploy=SMALL_DEPLOY))
    assert [stage.key for stage in result.stages] == ["adapt", "roofline"]
    assert any("--workload" in note for note in result.notes)
