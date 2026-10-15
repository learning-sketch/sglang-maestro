from __future__ import annotations

import json

import pytest

from sglang_maestro.cli import main
from sglang_maestro.target import layout_from


def test_list_json(capsys):
    assert main(["list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert "qwen3-32b" in data["models"]
    assert "h20" in data["hardware"]
    assert "chat" in data["workloads"]


def test_adapt_writes_json_by_suffix(tmp_path):
    output = tmp_path / "adapt.json"
    assert main(["adapt", "--model", "qwen3-32b", "--hardware", "h20", "-o", str(output)]) == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["support"]["supported"] is True


def test_unknown_model_is_a_clean_error(capsys):
    assert main(["adapt", "--model", "no-such-model", "--hardware", "h20"]) == 2
    assert "错误" in capsys.readouterr().err


def test_layout_needs_tp():
    assert layout_from({"tp": None}) is None
    assert layout_from({"tp": 8, "attn_dp": 4}).describe() == "TP8·DPA4"
    with pytest.raises(ValueError):
        layout_from({"attn_dp": 2})


def test_synthetic_log_calibrate_then_features(tmp_path, capsys):
    target = ["--model", "qwen3-32b", "--hardware", "h20", "--tp", "1"]
    log = tmp_path / "server.log"
    calibration = tmp_path / "calibration.json"
    assert main(["synth-log", *target, "-o", str(log)]) == 0
    assert main(["calibrate", "--log", str(log), *target, "-o", str(calibration)]) == 0
    fitted = json.loads(calibration.read_text(encoding="utf-8"))["calibration"]
    assert fitted["decode_scale"] == pytest.approx(1.3, rel=0.02)
    capsys.readouterr()
    features = ["features", *target, "--workload", "chat", "--calibration", str(calibration)]
    assert main([*features, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["combined"]["command"]["problems"] == []


def test_synth_trace_then_trace(tmp_path, capsys):
    target = ["--model", "qwen3-32b", "--hardware", "h20", "--tp", "2"]
    assert main(["synth-trace", *target, "--output-dir", str(tmp_path)]) == 0
    capsys.readouterr()
    assert main(["trace", str(tmp_path), *target, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["synthetic"] is True and report["findings"]
