"""``sgl-agent run``: scenarios 1, 2, 3 and 5 for one model, device and workload in one go.

Each stage writes its Markdown report into the output directory; ``report.md`` collects the
conclusions and ``report.json`` the full results. With server logs (``--log``, optionally a
``--bench`` record) or ``--demo`` the cost model is calibrated and the deployment planned again.
``--demo`` stands in synthetic measurements for a machine without the device: a profiler trace
with injected problems, and server logs and a benchmark from a device assumed slower than the
cost model (``DEMO_TRUTH``). Every report built from them says so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sgl_agent.adapt.analysis import analyze_adaptation
from sgl_agent.deploy.plan import DeployOptions, DeployPlan, plan_deployment, simulate_plan
from sgl_agent.features.report import advise_features
from sgl_agent.findings import Finding
from sgl_agent.markdown import SEVERITY_LABELS, bullets, table
from sgl_agent.model.ops import Calibration, ExecutionConfig
from sgl_agent.model.parallel import ParallelConfig
from sgl_agent.perf.analyze import analyze_trace
from sgl_agent.perf.calibrate import (
    CalibrationReport,
    calibration_to_dict,
    check_plan,
    fit_calibration,
)
from sgl_agent.perf.logs import analyze_logs
from sgl_agent.perf.roofline import analyze_roofline
from sgl_agent.perf.synthetic import generate_synthetic_traces, write_synthetic_traces
from sgl_agent.perf.synthetic_log import (
    synthetic_bench_record,
    synthetic_server_log,
    write_bench_record,
    write_synthetic_log,
)
from sgl_agent.target import Target

DEMO_TRUTH = Calibration(
    prefill_scale=1.15,
    decode_scale=1.3,
    source="假设设备的 prefill 比成本模型慢 15%、decode 慢 30%",
)
SUMMARY_FINDINGS = 5


@dataclass
class RunOptions:
    deploy: DeployOptions = field(default_factory=DeployOptions)
    traces: list[Path] = field(default_factory=list)
    logs: list[Path] = field(default_factory=list)
    bench: Path | None = None
    demo: bool = False
    findings_limit: int | None = 10


@dataclass
class Stage:
    key: str
    title: str
    file: str
    summary: str
    markdown: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class RunResult:
    target: str
    output_dir: Path
    stages: list[Stage]
    plan: DeployPlan | None = None
    calibration: Calibration | None = None
    notes: list[str] = field(default_factory=list)

    def stage(self, key: str) -> Stage | None:
        return next((stage for stage in self.stages if stage.key == key), None)


def section(markdown: str, heading: str) -> str:
    """Body of the ``### heading`` section of a rendered report ('' when absent)."""
    body: list[str] = []
    inside = False
    for line in markdown.splitlines():
        if line.startswith(("## ", "### ")):
            if inside:
                break
            inside = line == f"### {heading}"
            continue
        if inside:
            body.append(line)
    return "\n".join(body).strip()


def findings_summary(findings: list[Finding], limit: int = SUMMARY_FINDINGS) -> str:
    if not findings:
        return "未发现问题。"
    lines = [f"[{SEVERITY_LABELS[item.severity]}] {item.title}" for item in findings[:limit]]
    if len(findings) > limit:
        lines.append(f"另有 {len(findings) - limit} 条，见详细报告")
    return bullets(lines)


def _prediction(plan: DeployPlan) -> str:
    chosen = plan.chosen
    if chosen is None or chosen.result is None:
        return "无可行方案"
    result = chosen.result
    percentile = plan.workload.slo.percentile
    return (
        f"{chosen.describe()}，{chosen.num_gpus} 卡；P{percentile} TTFT "
        f"{result.ttft_ms[percentile]:.0f} ms、TPOT {result.tpot_ms[percentile]:.1f} ms，"
        f"输出 {result.output_throughput:,.0f} tok/s"
    )


class _Run:
    def __init__(self, target: Target, output_dir: Path, options: RunOptions):
        self.target = target
        self.out = output_dir
        self.options = options
        self.result = RunResult(target.describe(), output_dir, [])
        self.measurements = output_dir / "measurements"

    def stage(self, key: str, title: str, file: str, build) -> Any:
        """Run ``build() -> (report, markdown, summary)``; a failing stage is recorded, not raised."""
        try:
            report, markdown, summary = build()
        except (ValueError, FileNotFoundError) as error:
            self.result.stages.append(Stage(key, title, file, f"失败：{error}", error=str(error)))
            return None
        (self.out / file).write_text(markdown, encoding="utf-8")
        self.result.stages.append(Stage(key, title, file, summary, markdown, report.to_dict()))
        return report

    def run(self) -> RunResult:
        target, options = self.target, self.options
        self.out.mkdir(parents=True, exist_ok=True)
        limit = options.findings_limit

        def adapt():
            report = analyze_adaptation(
                target.arch,
                target.hardware,
                target.knowledge,
                parallel=target.parallel,
                model_path=target.model_path,
                hardware_ref=target.hardware_ref,
            )
            markdown = report.to_markdown(findings_limit=limit)
            return report, markdown, section(markdown, "结论")

        adaptation = self.stage("adapt", "场景 1：模型适配", "01-adapt.md", adapt)
        if adaptation is None:
            return self.finish()
        route = adaptation.precision.recommended
        layout = adaptation.parallel
        execution = ExecutionConfig(
            weight_dtype=route.weight_override if route else None, kv_dtype=target.kv_dtype
        )

        def roofline():
            report = analyze_roofline(target.arch, target.hardware, layout, execution)
            markdown = report.to_markdown()
            return report, markdown, section(markdown, "结论")

        self.stage("roofline", "场景 2：算子性能（roofline）", "02-roofline.md", roofline)
        traces = list(options.traces)
        if not traces and options.demo:
            traces = self.synthetic_traces(layout, execution)
        if traces:
            self.stage(
                "trace",
                "场景 2：profiler trace 分析",
                "02-trace.md",
                lambda: self.trace(traces, layout, execution),
            )
        if options.logs:
            self.stage(
                "logs",
                "场景 2：server 日志分析",
                "02-logs.md",
                lambda: self.server_logs(execution),
            )
        if target.workload is None:
            self.result.notes.append("没有给负载（--workload），跳过场景 3 和场景 5")
            return self.finish()

        def features():
            report = advise_features(target.feature_context())
            markdown = report.to_markdown()
            return report, markdown, section(markdown, "组合配置")

        self.stage("features", "场景 3：框架特性适配与调优", "03-features.md", features)
        plan = self.stage(
            "deploy", "场景 5：部署方案", "04-deploy.md", lambda: self.deploy(target.calibration)
        )
        self.result.plan = plan
        if plan is None or plan.chosen is None or not plan.chosen.feasible:
            return self.finish()
        if options.logs or options.demo:
            self.calibrate(plan, execution)
        return self.finish()

    # ------------------------------------------------------------------ scenario 2
    def synthetic_traces(self, layout: ParallelConfig, execution: ExecutionConfig) -> list[Path]:
        payloads = generate_synthetic_traces(
            self.target.arch,
            self.target.hardware,
            layout,
            num_ranks=min(layout.num_gpus, 2),
            execution=execution,
        )
        self.result.notes.append(
            "--demo：profiler trace 是成本模型生成的合成 trace（注入了未融合的 add+RMSNorm、逐元素链、"
            "prefill 里的主机同步、个别 decode step 的注意力变慢和一次 eager decode），不是实测"
        )
        return write_synthetic_traces(payloads, self.measurements / "trace")

    def trace(self, paths: list[Path], layout: ParallelConfig, execution: ExecutionConfig):
        target = self.target
        report = analyze_trace(
            paths,
            knowledge=target.knowledge,
            hardware=target.hardware,
            arch=target.arch,
            parallel=target.parallel or layout,
            execution=execution,
            calibration=target.calibration,
        )
        markdown = report.to_markdown(findings_limit=self.options.findings_limit)
        return report, markdown, findings_summary(report.findings)

    def server_logs(self, execution: ExecutionConfig):
        target = self.target
        report = analyze_logs(
            self.options.logs,
            arch=target.arch,
            hardware=target.hardware,
            parallel=target.parallel,
            execution=execution,
            calibration=target.calibration,
            knowledge=target.knowledge,
        )
        markdown = report.to_markdown(findings_limit=self.options.findings_limit)
        return report, markdown, findings_summary(report.findings)

    # ------------------------------------------------------------------ scenario 5
    def deploy(self, calibration: Calibration | None):
        target = self.target
        plan = plan_deployment(
            target.arch,
            target.hardware,
            target.knowledge,
            target.require_workload(),
            options=self.options.deploy,
            model_path=target.model_path,
            kv_dtype=target.kv_dtype,
            host_memory_gb=target.host_memory_gb,
            calibration=calibration,
        )
        markdown = plan.to_markdown()
        return plan, markdown, section(markdown, "结论")

    # ------------------------------------------------------------------ calibration loop
    def calibrate(self, plan: DeployPlan, execution: ExecutionConfig) -> None:
        try:
            reports = (
                self.demo_fits(plan) if not self.options.logs else self.log_fits(plan, execution)
            )
        except (ValueError, FileNotFoundError) as error:
            self.result.stages.append(
                Stage("calibrate", "校准", "05-calibrate.md", f"失败：{error}", error=str(error))
            )
            return
        calibration = merge_fits(reports)
        self.result.calibration = calibration
        (self.out / "calibration.json").write_text(
            json.dumps(calibration_to_dict(calibration), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        markdown = "\n".join(report.to_markdown() for report in reports.values())
        checks = [check for report in reports.values() for check in report.plan_checks]
        lines = [
            f"拟合系数：prefill ×{calibration.prefill_scale:.2f}，decode ×{calibration.decode_scale:.2f}"
            "（已写入 calibration.json）"
        ]
        if checks:
            worst = max(checks, key=lambda check: abs(check.deviation))
            lines.append(
                f"实测与部署方案预测相比，偏差最大的是 {worst.metric}：预测 {worst.predicted:.4g}、"
                f"实测 {worst.measured:.4g}（{worst.deviation:+.0%}）"
            )
        (self.out / "05-calibrate.md").write_text(markdown, encoding="utf-8")
        self.result.stages.append(
            Stage(
                "calibrate",
                "校准：实测与成本模型",
                "05-calibrate.md",
                bullets(lines),
                markdown,
                {role: report.to_dict() for role, report in reports.items()},
            )
        )
        replanned = self.stage(
            "replan",
            "场景 5：校准后的部署方案",
            "06-deploy-calibrated.md",
            lambda: self.deploy(calibration),
        )
        if replanned is None:
            return
        self.result.plan = replanned
        stage = self.result.stages[-1]
        lines = [f"校准前：{_prediction(plan)}", f"校准后：{_prediction(replanned)}"]
        if self.options.demo and not self.options.logs and replanned.chosen is not None:
            lines.append(self.demo_verification(replanned))
        stage.summary = bullets(lines) + "\n\n" + stage.summary

    def log_fits(
        self, plan: DeployPlan, execution: ExecutionConfig
    ) -> dict[str, CalibrationReport]:
        target, options = self.target, self.options
        layout = target.parallel
        try:
            report = fit_calibration(
                options.logs,
                target.arch,
                target.hardware,
                layout,
                execution=execution,
                bench=options.bench,
                plan=plan.to_dict(),
            )
        except ValueError:
            if layout is not None:
                raise
            pool = next(iter(plan.chosen.pools.values()))
            report = fit_calibration(
                options.logs,
                target.arch,
                target.hardware,
                pool.spec.layout,
                execution=execution,
                bench=options.bench,
                plan=plan.to_dict(),
            )
            report.notes.append(
                f"没有给布局、压测记录里也没有 server_info：假设日志来自本方案的实例（{pool.spec.layout.describe()}）"
            )
        return {"colocated": report}

    def demo_fits(self, plan: DeployPlan) -> dict[str, CalibrationReport]:
        """Synthetic logs and benchmark of ``plan`` on a device that follows ``DEMO_TRUTH``."""
        target = self.target
        chosen = plan.chosen
        assert chosen is not None
        self.measurements.mkdir(parents=True, exist_ok=True)
        bench = self.measurements / "bench_result.jsonl"
        bench.unlink(missing_ok=True)
        front = chosen.pools.get("decode") or chosen.pools["colocated"]
        self.write_demo_bench(plan, bench, front.spec.layout)
        reports = {}
        for role, pool in chosen.pools.items():
            execution = pool.model.cost.execution
            lines = synthetic_server_log(
                target.arch,
                target.hardware,
                pool.spec.layout,
                decode_scale=DEMO_TRUTH.decode_scale,
                prefill_scale=DEMO_TRUTH.prefill_scale,
                execution=execution,
            )
            log = write_synthetic_log(lines, self.measurements / f"server-{role}.log")
            with_bench = role != "prefill"
            reports[role] = fit_calibration(
                log,
                target.arch,
                target.hardware,
                pool.spec.layout,
                execution=execution,
                bench=bench if with_bench else None,
                plan=plan.to_dict() if with_bench else None,
            )
        self.result.notes.append(
            f"--demo：server 日志和压测结果是合成的（{DEMO_TRUTH.source}），用来演示“实测→校准→重新规划”闭环，"
            "不是实测；真实设备上请用 --log/--bench 传入实测文件"
        )
        return reports

    def write_demo_bench(self, plan: DeployPlan, path: Path, layout: ParallelConfig) -> dict:
        options = self.options.deploy
        measured = simulate_plan(
            plan, calibration=DEMO_TRUTH, verify_samples=options.verify_samples, seed=options.seed
        )
        record = synthetic_bench_record(
            measured.to_dict(), parallel=layout, request_rate=plan.workload.request_rate
        )
        write_bench_record(record, path)
        return record

    def demo_verification(self, plan: DeployPlan) -> str:
        chosen = plan.chosen
        assert chosen is not None
        front = chosen.pools.get("decode") or chosen.pools["colocated"]
        path = self.measurements / "bench_result.calibrated.jsonl"
        path.unlink(missing_ok=True)
        record = self.write_demo_bench(plan, path, front.spec.layout)
        checks = check_plan(plan.to_dict(), record)
        if not checks:
            return "校准后方案没有可对比的预测"
        worst = max(checks, key=lambda check: abs(check.deviation))
        verdict = "全部在 ±20% 内" if all(check.within for check in checks) else "仍有指标超出 ±20%"
        return (
            f"校准后的方案放到同一台（假设的）设备上再“压测”一次：{verdict}，偏差最大的 {worst.metric} "
            f"预测 {worst.predicted:.4g}、实测 {worst.measured:.4g}（{worst.deviation:+.0%}）"
        )

    # ------------------------------------------------------------------ output
    def finish(self) -> RunResult:
        result = self.result
        (self.out / "report.md").write_text(render_run(result), encoding="utf-8")
        payload = {
            "target": result.target,
            "notes": result.notes,
            "calibration": calibration_to_dict(result.calibration) if result.calibration else None,
            "stages": {
                stage.key: {
                    "title": stage.title,
                    "file": stage.file,
                    "error": stage.error,
                    "report": stage.data,
                }
                for stage in result.stages
            },
        }
        (self.out / "report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8"
        )
        return result


def merge_fits(reports: dict[str, CalibrationReport]) -> Calibration:
    """Prefill scale from the prefill (or colocated) server, decode scale from the decode one."""
    prefill = reports.get("prefill") or reports.get("colocated") or next(iter(reports.values()))
    decode = reports.get("decode") or reports.get("colocated") or next(iter(reports.values()))
    used = [
        f"{name} {fit.samples} 个样本"
        for name, fit in (("prefill", prefill.prefill), ("decode", decode.decode))
        if fit.used
    ]
    return Calibration(
        prefill_scale=prefill.calibration.prefill_scale,
        decode_scale=decode.calibration.decode_scale,
        source=f"sgl-agent run（{'、'.join(used) or '无可用样本'}）",
    )


def render_run(result: RunResult) -> str:
    parts = [f"# sgl-agent 报告：{result.target}"]
    rows = [
        [
            stage.title,
            f"[{stage.file}]({stage.file})" if not stage.error else "—",
            stage.error or "完成",
        ]
        for stage in result.stages
    ]
    parts += ["## 流程", table(["阶段", "详细报告", "状态"], rows)]
    if result.notes:
        parts += ["## 说明", bullets(result.notes)]
    for stage in result.stages:
        parts += [f"## {stage.title}", stage.summary or "（无）"]
        if not stage.error:
            parts.append(f"详见 [{stage.file}]({stage.file})。")
    plan = result.plan
    if plan is not None and plan.chosen is not None and plan.chosen.feasible:
        steps = [f"{index}. {step}" for index, step in enumerate(plan.steps, 1)]
        parts += [
            "## 下一步",
            "按最终部署方案（"
            + ("校准后" if result.stage("replan") else "未校准")
            + "）上线，部署命令见对应报告的“部署命令”一节：",
            "\n".join(steps),
        ]
    return "\n\n".join(parts) + "\n"


def run_pipeline(
    target: Target, output_dir: str | Path, options: RunOptions | None = None
) -> RunResult:
    return _Run(target, Path(output_dir), options or RunOptions()).run()
