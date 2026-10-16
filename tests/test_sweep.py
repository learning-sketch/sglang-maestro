"""Decision baselines for the bundled model × hardware × workload matrix.

Scenarios 1–3 run on the full matrix. Scenario 5 runs a short subset here and the full
matrix under ``pytest -m slow``. A mismatch names the combo, the field, and the old and new
values (at most 80 lines).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from sglang_maestro.cli import main
from sglang_maestro.hardware.spec import list_presets
from sglang_maestro.model.config import list_bundled_models
from sglang_maestro.sweep import (
    BASELINE_CAPACITY_SAMPLES,
    BASELINE_SEED,
    BASELINE_VERIFY_SAMPLES,
    DIFF_LINE_LIMIT,
    INVARIANTS,
    KNOWN_VIOLATIONS,
    SCENARIOS,
    BaselineWriteError,
    SweepCell,
    SweepResult,
    SweepTask,
    _cell_from_dict,
    _run_task,
    bundled_names,
    default_jobs,
    fingerprint_diff,
    no_plan_detail,
    render_sweep,
    scenario_fingerprints,
    sweep,
    write_baselines,
)
from sglang_maestro.workload import list_workloads

_BASELINES = Path(__file__).parent / "baselines"
# Keep this list small: the default suite's added time has to stay within about 10 seconds.
DEPLOY_SUBSET = (
    "qwen3-235b-a22b/h20/chat",
    "deepseek-v3/template-h20-class/rag-long-context",
    "qwen3-235b-a22b/h20/shared-prefix-agent",
    "qwen2.5-72b-instruct/h100-sxm/rag-long-context",
    "qwen3-30b-a3b/h20/offline-batch",
    "deepseek-v3/template-h20-class/offline-batch",
    "qwen2.5-72b-instruct/h100-sxm/offline-batch",
)
_DETERMINISM = (
    "qwen3-30b-a3b/h20/offline-batch",
    "deepseek-v3/template-h20-class/offline-batch",
)
_OVER_BUDGET = "deepseek-v3/template-h20-class/rag-long-context"


def _load(scenario: str) -> dict:
    return json.loads((_BASELINES / f"{scenario}.json").read_text(encoding="utf-8"))


def _baseline_options(**overrides: object) -> dict:
    options = {
        "capacity_samples": BASELINE_CAPACITY_SAMPLES,
        "verify_samples": BASELINE_VERIFY_SAMPLES,
        "seed": BASELINE_SEED,
        "max_gpus": None,
        "allow_pd": True,
        "allow_mixed_chunk": True,
    }
    options.update(overrides)
    return options


def _cell(key: str, violations: list[dict[str, str]]) -> SweepCell:
    return SweepCell(
        key=key,
        seconds=0.0,
        fingerprints={},
        violations=violations,
        summary={},
    )


def _assert_matches(expected: dict, actual: dict) -> None:
    diff = fingerprint_diff(expected, actual)
    if diff:
        raise AssertionError(diff)


@pytest.fixture(scope="module")
def light_matrix() -> SweepResult:
    return sweep(scenarios=("adapt", "roofline", "features"), jobs=default_jobs())


@pytest.fixture(scope="module")
def deploy_subset() -> SweepResult:
    return sweep(
        combos=DEPLOY_SUBSET,
        scenarios=("deploy",),
        capacity_samples=BASELINE_CAPACITY_SAMPLES,
        verify_samples=BASELINE_VERIFY_SAMPLES,
        seed=BASELINE_SEED,
        jobs=default_jobs(),
    )


@pytest.mark.parametrize("scenario", ("adapt", "roofline", "features"))
def test_light_scenarios_match_baseline(light_matrix: SweepResult, scenario: str) -> None:
    assert light_matrix.ok
    assert all(cell.error is None for cell in light_matrix.cells.values())
    _assert_matches(_load(scenario), scenario_fingerprints(light_matrix, scenario))


def test_slow_ci_shards_match_bundled_hardware() -> None:
    document = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml").read_text(
            encoding="utf-8"
        )
    )
    shards = document["jobs"]["slow"]["strategy"]["matrix"]["hardware"]
    expected = sorted(list_presets())
    actual = sorted(shards)
    missing = [name for name in expected if name not in set(actual)]
    extra = [name for name in actual if name not in set(expected)]
    assert actual == expected, f"CI slow 分片和内置硬件不一致：缺了 {missing}，多了 {extra}"


def test_baseline_keys_are_the_full_bundled_matrix() -> None:
    models = sorted(list_bundled_models())
    hardware = sorted(list_presets())
    workloads = sorted(list_workloads())
    by_device = {f"{model}/{device}" for model in models for device in hardware}
    by_workload = {
        f"{model}/{device}/{workload}"
        for model in models
        for device in hardware
        for workload in workloads
    }
    assert set(_load("adapt")) == by_device
    assert set(_load("roofline")) == by_device
    assert set(_load("features")) == by_workload
    assert set(_load("deploy")) == by_workload


def test_write_baselines_refuses_errors_and_missing_registrations(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    known = KNOWN_VIOLATIONS[0]
    crashed = SweepResult(
        cells={
            "qwen3-32b/h20/chat": SweepCell(
                key="qwen3-32b/h20/chat",
                seconds=0.0,
                fingerprints={"deploy": {"chosen": None}},
                violations=[],
                summary={},
                error="RuntimeError: boom",
            )
        },
        options=_baseline_options(),
    )
    dest = tmp_path / "crashed"
    with pytest.raises(BaselineWriteError):
        write_baselines(crashed, dest)
    assert not dest.exists()
    assert "boom" in capsys.readouterr().err

    missing = SweepResult(
        cells={known.combo: _cell(known.combo, [])},
        options=_baseline_options(),
    )
    dest = tmp_path / "missing"
    with pytest.raises(BaselineWriteError):
        write_baselines(missing, dest)
    assert not dest.exists()
    assert "删掉登记" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ("--models", "--hardware", "--workloads"))
def test_write_baselines_rejects_a_subset_before_the_sweep(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    dest = tmp_path / "baselines"
    status = main(["sweep", flag, "qwen3-32b", "--write-baselines", str(dest)])
    captured = capsys.readouterr()
    assert status == 2
    assert "基线只能整矩阵生成" in captured.err
    assert not dest.exists()


def test_deploy_subset_matches_baseline(deploy_subset: SweepResult) -> None:
    expected = {key: _load("deploy")[key] for key in DEPLOY_SUBSET}
    _assert_matches(expected, scenario_fingerprints(deploy_subset, "deploy"))
    assert deploy_subset.ok
    cell = deploy_subset.cells[_OVER_BUDGET]
    assert cell.summary["over_budget"] is True
    assert cell.summary["meets_slo"] is True
    assert all(item["id"] != "budget" for item in cell.violations)


def test_repeat_and_jobs_do_not_change_decisions() -> None:
    kwargs = dict(
        combos=_DETERMINISM,
        capacity_samples=BASELINE_CAPACITY_SAMPLES,
        verify_samples=BASELINE_VERIFY_SAMPLES,
        seed=BASELINE_SEED,
    )
    first = sweep(**kwargs, jobs=1)
    second = sweep(**kwargs, jobs=1)
    pooled = sweep(**kwargs, jobs=2)
    fingerprints = [
        {scenario: scenario_fingerprints(result, scenario) for scenario in SCENARIOS}
        for result in (first, second, pooled)
    ]
    assert fingerprints[0] == fingerprints[1] == fingerprints[2]


def test_task_exception_is_not_a_command_problem() -> None:
    task = SweepTask("no-such-model", "h20", "chat", 40, 120, 1, ("adapt",))
    cell = _cell_from_dict(_run_task(task))
    assert cell.error is not None
    assert cell.violations
    assert {item["id"] for item in cell.violations} == {"error"}
    detail = cell.violations[0]["detail"]
    assert detail == cell.error
    assert detail.startswith("FileNotFoundError:")
    assert "组合崩溃" in INVARIANTS["error"]
    assert "启动命令问题" in INVARIANTS["commands"]
    assert "异常" not in INVARIANTS["commands"]
    report = render_sweep(
        SweepResult(
            cells={cell.key: cell},
            options=_baseline_options(
                capacity_samples=240, verify_samples=1200, jobs=1, scenarios=["adapt"]
            ),
        )
    )
    assert "`error` 只表示这个组合在运行时抛了异常" in report
    assert "`commands` 只表示启动命令的参数问题" in report


def test_no_plan_contradiction_is_invariant_5() -> None:
    impossible = ["单实例在任何到达率下都满足不了 SLO"]
    queue = ["单请求的 prefill 和 decode step 都在 SLO 以内：瓶颈是排队"]
    assert no_plan_detail([], [], []) == "没选出方案，但 notes 和 bottlenecks 都是空的"
    assert "瓶颈是排队" in (no_plan_detail(["PD 说明"], queue, [impossible]) or "")
    assert no_plan_detail(["有原因"], ["prompt 单独 prefill 就超过 TTFT"], [impossible]) is None
    assert no_plan_detail(["有原因"], [], []) is None


def test_registered_violation_does_not_fail_and_a_new_one_does() -> None:
    known = KNOWN_VIOLATIONS[0]
    registered = SweepResult(
        cells={
            known.combo: _cell(known.combo, [{"combo": known.combo, "id": known.id, "detail": "x"}])
        },
        options=_baseline_options(),
    )
    assert registered.ok
    assert registered.missing_known() == []

    extra = SweepResult(
        cells={
            known.combo: _cell(
                known.combo, [{"combo": known.combo, "id": known.id, "detail": "x"}]
            ),
            "qwen3-32b/h20/chat": _cell(
                "qwen3-32b/h20/chat",
                [
                    {
                        "combo": "qwen3-32b/h20/chat",
                        "id": "slo",
                        "detail": "选出了方案，但 meets_slo 为假",
                    }
                ],
            ),
        },
        options=_baseline_options(),
    )
    assert not extra.ok
    assert extra.unexpected_violations()[0]["combo"] == "qwen3-32b/h20/chat"

    gone = SweepResult(
        cells={known.combo: _cell(known.combo, [])},
        options=_baseline_options(),
    )
    assert not gone.ok
    assert gone.missing_known() == [known]

    # Default deploy samples are not the baseline, so a known violation that only
    # shows up at 40/120 must not be required there.
    default_samples = SweepResult(
        cells={known.combo: _cell(known.combo, [])},
        options=_baseline_options(capacity_samples=240, verify_samples=1200),
    )
    assert default_samples.ok
    assert default_samples.missing_known() == []


def test_fingerprint_diff_names_combo_field_and_truncates() -> None:
    expected = {"m/h/w": {"chosen": {"num_gpus": 8}}}
    actual = {"m/h/w": {"chosen": {"num_gpus": 16}}}
    text = fingerprint_diff(expected, actual)
    assert text.startswith("基线不一致：")
    assert "m/h/w" in text
    assert "chosen.num_gpus" in text
    assert "旧值 8" in text and "新值 16" in text
    wide_old = {f"c{i}": i for i in range(DIFF_LINE_LIMIT + 5)}
    wide_new = {f"c{i}": i + 1 for i in range(DIFF_LINE_LIMIT + 5)}
    truncated = fingerprint_diff(wide_old, wide_new)
    body = truncated.splitlines()
    assert body[0] == "基线不一致："
    assert body[-1] == "... diff truncated"
    assert len(body) == DIFF_LINE_LIMIT + 2


def test_sweep_cli_writes_the_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "sweep"
    status = main(
        [
            "sweep",
            "--models",
            "qwen3-30b-a3b",
            "--hardware",
            "h20",
            "--workloads",
            "offline-batch",
            "--capacity-samples",
            "40",
            "--verify-samples",
            "120",
            "--seed",
            "1",
            "--jobs",
            "1",
            "-o",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    assert status == 0, captured.err
    report = (output / "sweep.md").read_text(encoding="utf-8")
    assert "qwen3-30b-a3b/h20/offline-batch" in report
    assert "不是实测" in report
    assert "不变量违反" in report
    payload = json.loads((output / "sweep.json").read_text(encoding="utf-8"))
    assert payload["options"]["jobs"] == 1
    assert payload["unexpected_violations"] == []


def test_sweep_cli_requires_an_output_and_baseline_samples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["sweep"]) == 2
    assert "错误" in capsys.readouterr().err
    capsys.readouterr()
    assert (
        main(
            [
                "sweep",
                "--models",
                "no-such-model",
                "-o",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "sweep",
                "--capacity-samples",
                "20",
                "--write-baselines",
                str(tmp_path / "baselines"),
            ]
        )
        == 2
    )
    assert "40" in capsys.readouterr().err


@pytest.mark.slow
def test_deploy_baseline_full_matrix() -> None:
    """Deployment plans at the checked-in sample counts.

    Unset ``SGLANG_MAESTRO_SLOW_HARDWARE`` runs all 90 combos. CI sets it to one
    hardware stem so each job runs that stem's 30 combos and its own monotonicity
    check. The registered violation is only required on the shard that contains it.
    """
    shard = os.environ.get("SGLANG_MAESTRO_SLOW_HARDWARE")
    models, hardware, workloads = bundled_names()
    if shard and shard not in hardware:
        raise AssertionError(f"未知硬件分片 {shard}")
    result = sweep(
        hardware=[shard] if shard else None,
        scenarios=("deploy",),
        capacity_samples=BASELINE_CAPACITY_SAMPLES,
        verify_samples=BASELINE_VERIFY_SAMPLES,
        seed=BASELINE_SEED,
        jobs=default_jobs(),
        check_monotonicity=True,
    )
    expected = _load("deploy")
    if shard:
        expected = {key: value for key, value in expected.items() if key.split("/")[1] == shard}
    assert len(result.cells) == len(models) * len(workloads) * (1 if shard else len(hardware))
    _assert_matches(expected, scenario_fingerprints(result, "deploy"))
    assert result.unexpected_violations() == []
    assert all(cell.error is None for cell in result.cells.values())
    known = KNOWN_VIOLATIONS[0]
    if shard is None or known.combo.split("/")[1] == shard:
        assert result.missing_known() == []
        matched = [
            item
            for item in result.violations()
            if item["combo"] == known.combo and item["id"] == known.id
        ]
        assert matched, f"已登记的违反没有出现：{known.combo} {known.id}"
        assert "瓶颈是排队" in matched[0]["detail"]
    else:
        assert result.missing_known() == []
        assert known.combo not in result.cells
    assert len(result.monotonicity) == 2
    assert {row.key for row in result.monotonicity}.isdisjoint(
        # Throughput workloads have no arrival rate to double.
        {key for key in result.cells if key.endswith("/offline-batch")}
    )
