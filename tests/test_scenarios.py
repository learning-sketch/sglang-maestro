from __future__ import annotations

import pytest

from conftest import SMALL_DEPLOY
from sgl_agent.adapt import analyze_adaptation
from sgl_agent.deploy import plan_deployment
from sgl_agent.deploy.plan import simulate_plan
from sgl_agent.features import advise_features
from sgl_agent.model import Calibration, ParallelConfig
from sgl_agent.perf import analyze_roofline, analyze_trace, fit_calibration
from sgl_agent.perf.synthetic import (
    SyntheticIssues,
    generate_synthetic_traces,
    write_synthetic_traces,
)
from sgl_agent.perf.synthetic_log import synthetic_server_log, write_synthetic_log


def test_in_tree_model_needs_no_model_code(dense_chat):
    report = analyze_adaptation(dense_chat.arch, dense_chat.hardware, dense_chat.knowledge)
    assert report.support.supported
    assert not report.profile.out_of_tree
    assert all(item.status != "missing" for item in report.inventory.requirements)
    assert report.precision.recommended is not None


def test_out_of_tree_device_lists_plugin_hooks_and_operator_gaps(oot_moe):
    report = analyze_adaptation(oot_moe.arch, oot_moe.hardware, oot_moe.knowledge)
    assert report.profile.out_of_tree
    assert any(item.level == "required" for item in report.checklist.requirements)
    assert any(item.status == "missing" for item in report.inventory.requirements)
    markdown = report.to_markdown()
    assert "### 平台插件 hook 清单" in markdown and "### 上手步骤" in markdown


def test_roofline_decode_is_memory_bound_at_small_batch(dense_chat):
    report = analyze_roofline(
        dense_chat.arch, dense_chat.hardware, dense_chat.layout(), dense_chat.execution()
    )
    decode = next(item for item in report.scenarios if item.phase == "decode")
    assert decode.batch_size == 1
    assert decode.by_bound_ms.get("memory", 0) > decode.by_bound_ms.get("compute", 0)


def test_trace_analysis_finds_injected_problems(tmp_path, dense_chat):
    layout = ParallelConfig(tp=2)

    def titles(issues: SyntheticIssues) -> list[str]:
        payloads = generate_synthetic_traces(
            dense_chat.arch, dense_chat.hardware, layout, issues=issues, num_ranks=2
        )
        paths = write_synthetic_traces(payloads, tmp_path / str(id(issues)))
        report = analyze_trace(
            paths,
            knowledge=dense_chat.knowledge,
            hardware=dense_chat.hardware,
            arch=dense_chat.arch,
            parallel=layout,
        )
        assert report.synthetic
        return [finding.title for finding in report.findings]

    injected = titles(SyntheticIssues())
    assert any("RMSNorm" in title for title in injected)
    assert any("异常" in title for title in injected)
    clean = titles(SyntheticIssues.none())
    assert not any("残差 add 与 RMSNorm" in title for title in clean)


def test_feature_launch_command_passes_flag_validation(dense_chat):
    report = advise_features(dense_chat.feature_context())
    assert report.combined.command.ok, report.combined.command.problems
    assert {item.key for item in report.advices} >= {"dp_attention", "hicache", "speculative"}


@pytest.fixture(scope="module")
def chat_plan(dense_chat):
    return plan_deployment(
        dense_chat.arch,
        dense_chat.hardware,
        dense_chat.knowledge,
        dense_chat.workload,
        options=SMALL_DEPLOY,
    )


def test_deploy_plan_meets_slo_with_valid_commands(chat_plan):
    assert chat_plan.chosen is not None and chat_plan.chosen.meets_slo
    commands = [command for node in chat_plan.nodes for command in node.commands]
    assert commands and all(command.ok for command in commands)
    assert chat_plan.bench is not None and chat_plan.bench.ok
    if chat_plan.chosen.num_instances > 1:
        assert chat_plan.router is not None and chat_plan.router.ok


def test_simulate_plan_reproduces_and_slows_down(chat_plan):
    options = SMALL_DEPLOY
    same = simulate_plan(chat_plan, verify_samples=options.verify_samples, seed=options.seed)
    assert same.ttft_ms == chat_plan.chosen.result.ttft_ms
    slower = simulate_plan(
        chat_plan,
        calibration=Calibration(decode_scale=1.5),
        verify_samples=options.verify_samples,
        seed=options.seed,
    )
    assert slower.tpot_ms[90] > same.tpot_ms[90]


def test_calibration_recovers_synthetic_slowdown(tmp_path, dense_chat):
    layout = dense_chat.layout()
    execution = dense_chat.execution()
    lines = synthetic_server_log(
        dense_chat.arch,
        dense_chat.hardware,
        layout,
        decode_scale=1.4,
        prefill_scale=1.2,
        execution=execution,
    )
    log = write_synthetic_log(lines, tmp_path / "server.log")
    report = fit_calibration(log, dense_chat.arch, dense_chat.hardware, layout, execution=execution)
    assert report.decode.used and report.prefill.used
    assert report.calibration.decode_scale == pytest.approx(1.4, rel=0.02)
    assert report.calibration.prefill_scale == pytest.approx(1.2, rel=0.02)
    assert any("合成" in note for note in report.notes)
