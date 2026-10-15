"""``sgl-agent <command>``: every analysis prints Markdown; ``--json`` (or ``-o x.json``) prints JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sgl_agent import __version__
from sgl_agent.adapt.analysis import analyze_adaptation
from sgl_agent.adapt.precision import PREFERENCES
from sgl_agent.deploy.plan import DeployOptions, plan_deployment
from sgl_agent.features.report import FEATURE_KEYS, advise_features
from sgl_agent.hardware.spec import list_presets, load_hardware
from sgl_agent.knowledge import load_knowledge
from sgl_agent.knowledge.facts import PD_TRANSFER_BACKENDS
from sgl_agent.knowledge.sync import write_snapshot
from sgl_agent.markdown import table
from sgl_agent.model.config import list_bundled_models, load_model
from sgl_agent.model.ops import ExecutionConfig
from sgl_agent.perf.analyze import analyze_trace, load_dialect
from sgl_agent.perf.calibrate import fit_calibration, load_calibration
from sgl_agent.perf.logs import analyze_logs
from sgl_agent.perf.roofline import analyze_roofline
from sgl_agent.perf.synthetic import (
    SyntheticIssues,
    generate_synthetic_traces,
    write_synthetic_traces,
)
from sgl_agent.perf.synthetic_log import synthetic_server_log, write_synthetic_log
from sgl_agent.pipeline import RunOptions, run_pipeline
from sgl_agent.target import Target, layout_from
from sgl_agent.workload import list_workloads

DEFAULT_RUN_DIR = "sgl-agent-report"


# ---------------------------------------------------------------------- arguments
def _add_layout(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("并行布局（不给 --tp 时用场景 1 推荐的精度路线的布局）")
    group.add_argument("--tp", type=int, help="--tp-size")
    group.add_argument("--pp", type=int, help="--pp-size")
    group.add_argument("--attn-dp", type=int, dest="attn_dp", help="--attn-dp-size（DP attention）")
    group.add_argument("--attn-cp", type=int, dest="attn_cp", help="--attn-cp-size（prefill CP）")
    group.add_argument("--dcp", type=int, help="--dcp-size")
    group.add_argument("--ep", type=int, help="--ep-size")
    group.add_argument("--dense-mlp-tp", type=int, dest="dense_mlp_tp", help="--moe-dense-tp-size")
    group.add_argument(
        "--moe-a2a", dest="moe_a2a", help="--moe-a2a-backend；默认 auto（ep>1 时按 deepep 估算）"
    )


def _add_target(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
    workload: bool = False,
    layout: bool = True,
    calibration: bool = True,
) -> None:
    group = parser.add_argument_group("对象")
    group.add_argument(
        "--model", required=required, help="内置模型别名（sgl-agent list）、config.json 或模型目录"
    )
    group.add_argument(
        "--hardware", required=required, help="硬件预设（h20、template-h20-class…）或 YAML 路径"
    )
    if workload:
        group.add_argument(
            "--workload", required=True, help="内置负载名（sgl-agent list）或 YAML 路径"
        )
        group.add_argument(
            "--request-rate", type=float, help="覆盖负载的到达率（req/s；多轮负载是会话/s）"
        )
        group.add_argument(
            "--host-memory-gb", type=float, help="每台机器可给 HiCache 用的主机内存（默认 1024）"
        )
    group.add_argument("--model-path", help="launch_server 的 --model-path（默认 HF 仓库名）")
    group.add_argument("--kv-dtype", help="KV cache 精度（fp8_e4m3 等，默认跟随模型）")
    if calibration:
        group.add_argument("--calibration", help="`sgl-agent calibrate` 的 JSON 输出")
    group.add_argument(
        "--knowledge", help="SGLang 能力快照（默认内置；也可设 SGL_AGENT_KNOWLEDGE）"
    )
    if layout:
        _add_layout(parser)


def _add_output(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("输出")
    group.add_argument("--json", action="store_true", help="输出 JSON")
    group.add_argument("-o", "--output", help="写到文件（.json 后缀时写 JSON）")


def _add_deploy_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("部署搜索")
    group.add_argument(
        "--max-gpus", type=int, help="GPU 预算（默认：有到达率时 8 台机器，否则 1 台）"
    )
    group.add_argument("--sizes", help="只评估这些实例规模（每实例卡数），逗号分隔，例如 8,16")
    group.add_argument("--no-pd", action="store_true", help="不评估 PD 分离")
    group.add_argument("--no-mixed-chunk", action="store_true", help="不尝试 --enable-mixed-chunk")
    group.add_argument(
        "--transfer-backend",
        default="mooncake",
        choices=PD_TRANSFER_BACKENDS,
        help="PD 的 KV 传输后端",
    )
    group.add_argument("--ib-device", help="--disaggregation-ib-device")
    group.add_argument("--capacity-samples", type=int, default=240, help="单实例容量探测的请求数")
    group.add_argument("--verify-samples", type=int, default=1200, help="整集群仿真的请求数")
    group.add_argument("--seed", type=int, default=1)


def _deploy_options(args: argparse.Namespace) -> DeployOptions:
    sizes = (
        tuple(int(item) for item in args.sizes.split(",") if item.strip()) if args.sizes else None
    )
    return DeployOptions(
        max_gpus=args.max_gpus,
        sizes=sizes,
        allow_pd=not args.no_pd,
        allow_mixed_chunk=not args.no_mixed_chunk,
        transfer_backend=args.transfer_backend,
        ib_device=args.ib_device,
        capacity_samples=args.capacity_samples,
        verify_samples=args.verify_samples,
        seed=args.seed,
    )


def _target(args: argparse.Namespace, *, with_workload: bool = False) -> Target:
    values = vars(args)
    return Target.load(
        args.model,
        args.hardware,
        values.get("workload") if with_workload else None,
        parallel=layout_from(values),
        model_path=values.get("model_path"),
        kv_dtype=values.get("kv_dtype"),
        host_memory_gb=values.get("host_memory_gb"),
        calibration=values.get("calibration"),
        request_rate=values.get("request_rate"),
        knowledge=values.get("knowledge"),
    )


def _emit(args: argparse.Namespace, payload: dict[str, Any], markdown: Callable[[], str]) -> int:
    output = getattr(args, "output", None)
    as_json = getattr(args, "json", False) or (output is not None and output.endswith(".json"))
    text = (
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
        if as_json
        else markdown()
    )
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"已写入 {path}", file=sys.stderr)
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


# ---------------------------------------------------------------------- commands
def cmd_list(args: argparse.Namespace) -> int:
    data = {
        "models": list_bundled_models(),
        "hardware": list_presets(),
        "workloads": list_workloads(),
    }
    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    print("## 内置模型\n")
    print(table(["别名", "HF 仓库"], sorted(data["models"].items())))
    print("\n## 硬件预设\n")
    print(
        table(
            ["名称", "型号", "别名"],
            [
                [stem, names[0], "、".join(names[1:])]
                for stem, names in sorted(data["hardware"].items())
            ],
        )
    )
    print("\n## 内置负载\n")
    print(table(["名称", "说明"], sorted(data["workloads"].items())))
    return 0


def cmd_adapt(args: argparse.Namespace) -> int:
    target = _target(args)
    report = analyze_adaptation(
        target.arch,
        target.hardware,
        target.knowledge,
        parallel=target.parallel,
        model_path=target.model_path,
        hardware_ref=args.hardware,
        plugin_name=args.plugin_name,
        prefer=args.prefer,
        max_gpus=args.max_gpus,
    )
    return _emit(args, report.to_dict(), lambda: report.to_markdown(findings_limit=args.findings))


def cmd_roofline(args: argparse.Namespace) -> int:
    target = _target(args)
    report = analyze_roofline(
        target.arch,
        target.hardware,
        target.layout(),
        target.execution(args.weight_dtype),
        sweep_context=args.context,
        top_ops=args.top_ops,
    )
    return _emit(args, report.to_dict(), report.to_markdown)


def _optional_target(args: argparse.Namespace) -> dict[str, Any]:
    """Model, device and layout for the trace/log analyzers, all optional."""
    arch = load_model(args.model) if args.model else None
    hardware = load_hardware(args.hardware) if args.hardware else None
    parallel = layout_from(vars(args))
    execution = ExecutionConfig(weight_dtype=args.weight_dtype, kv_dtype=args.kv_dtype)
    if arch is not None and hardware is not None:
        target = Target(
            arch,
            hardware,
            load_knowledge(args.knowledge),
            parallel=parallel,
            kv_dtype=args.kv_dtype,
        )
        execution = target.execution(args.weight_dtype)
    return {
        "arch": arch,
        "hardware": hardware,
        "parallel": parallel,
        "execution": execution,
        "calibration": load_calibration(args.calibration) if args.calibration else None,
        "knowledge": load_knowledge(args.knowledge),
    }


def cmd_trace(args: argparse.Namespace) -> int:
    inputs = _optional_target(args)
    report = analyze_trace(
        args.paths,
        dialect=load_dialect(args.dialect) if args.dialect else None,
        context_len_hint=args.context_len,
        server_url=args.server_url,
        **inputs,
    )
    return _emit(args, report.to_dict(), lambda: report.to_markdown(findings_limit=args.findings))


def cmd_logs(args: argparse.Namespace) -> int:
    report = analyze_logs(args.paths, **_optional_target(args))
    return _emit(args, report.to_dict(), lambda: report.to_markdown(findings_limit=args.findings))


def cmd_synth_trace(args: argparse.Namespace) -> int:
    target = _target(args)
    layout = target.layout()
    payloads = generate_synthetic_traces(
        target.arch,
        target.hardware,
        layout,
        issues=SyntheticIssues.none() if args.clean else SyntheticIssues(),
        num_ranks=args.ranks or min(layout.num_gpus, 2),
        execution=target.execution(args.weight_dtype),
        seed=args.seed,
    )
    paths = write_synthetic_traces(payloads, args.output_dir)
    print(f"已写入 {len(paths)} 个合成 trace（{layout.describe()}，不是实测）：")
    for path in paths:
        print(f"  {path}")
    print(
        f"分析：sgl-agent trace {args.output_dir} --model {args.model} --hardware {args.hardware} "
        f"--tp {layout.tp}" + (f" --attn-dp {layout.attn_dp}" if layout.attn_dp > 1 else "")
    )
    return 0


def cmd_synth_log(args: argparse.Namespace) -> int:
    target = _target(args)
    layout = target.layout()
    lines = synthetic_server_log(
        target.arch,
        target.hardware,
        layout,
        decode_scale=args.decode_scale,
        prefill_scale=args.prefill_scale,
        execution=target.execution(args.weight_dtype),
        seed=args.seed,
    )
    path = write_synthetic_log(lines, args.output)
    print(
        f"已写入合成 server 日志 {path}（{layout.describe()}，decode ×{args.decode_scale:g}、"
        f"prefill ×{args.prefill_scale:g}，不是实测）"
    )
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    target = _target(args, with_workload=True)
    only = [item.strip() for item in args.only.split(",") if item.strip()] if args.only else None
    report = advise_features(target.feature_context(), only=only)
    return _emit(args, report.to_dict(), report.to_markdown)


def cmd_deploy(args: argparse.Namespace) -> int:
    target = _target(args, with_workload=True)
    plan = plan_deployment(
        target.arch,
        target.hardware,
        target.knowledge,
        target.require_workload(),
        options=_deploy_options(args),
        model_path=target.model_path,
        kv_dtype=target.kv_dtype,
        host_memory_gb=target.host_memory_gb,
        calibration=target.calibration,
    )
    return _emit(args, plan.to_dict(), plan.to_markdown)


def cmd_calibrate(args: argparse.Namespace) -> int:
    target = _target(args)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8")) if args.plan else None
    report = fit_calibration(
        args.log,
        target.arch,
        target.hardware,
        target.parallel,
        execution=target.execution(args.weight_dtype),
        bench=args.bench,
        plan=plan,
    )
    return _emit(args, report.to_dict(), report.to_markdown)


def cmd_run(args: argparse.Namespace) -> int:
    target = _target(args, with_workload=True)
    options = RunOptions(
        deploy=_deploy_options(args),
        traces=[Path(path) for path in args.trace or []],
        logs=[Path(path) for path in args.log or []],
        bench=Path(args.bench) if args.bench else None,
        demo=args.demo,
        findings_limit=args.findings,
    )
    result = run_pipeline(target, args.output_dir, options)
    print(f"sgl-agent run：{result.target}")
    for stage in result.stages:
        status = f"失败：{stage.error}" if stage.error else stage.file
        print(f"  {stage.title}：{status}")
    for note in result.notes:
        print(f"  注意：{note}")
    print(f"汇总报告：{Path(args.output_dir) / 'report.md'}")
    return 1 if any(stage.error for stage in result.stages) else 0


def cmd_sync_knowledge(args: argparse.Namespace) -> int:
    output = Path(args.output)
    snapshot = write_snapshot(Path(args.sglang_src), output)
    commit = (snapshot.get("sglang_commit") or "?")[:8]
    print(
        f"已从 {args.sglang_src}（SGLang {commit}）生成能力快照 {output}："
        f"{len(snapshot.get('server_flags', {}))} 个启动参数、"
        f"{len(snapshot.get('model_architectures', {}))} 个模型架构"
    )
    print(f"使用：SGL_AGENT_KNOWLEDGE={output} sgl-agent ...，或给命令加 --knowledge {output}")
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sgl-agent",
        description="在 H20 类设备上用 SGLang 做模型适配、性能分析、特性调优和部署规划",
    )
    parser.add_argument("--version", action="version", version=f"sgl-agent {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def command(name: str, handler, help_text: str) -> argparse.ArgumentParser:
        child = sub.add_parser(name, help=help_text, description=help_text)
        child.set_defaults(handler=handler)
        return child

    child = command("list", cmd_list, "列出内置模型、硬件预设和负载")
    child.add_argument("--json", action="store_true")

    child = command("adapt", cmd_adapt, "场景 1：模型在设备上怎么适配、框架要补什么、改动原则")
    _add_target(child, calibration=False)
    child.add_argument("--plugin-name", help="平台插件包名（默认用 device_type）")
    child.add_argument("--prefer", choices=PREFERENCES, default="accuracy", help="精度路线偏好")
    child.add_argument("--max-gpus", type=int, help="精度路线最多用几张卡")
    child.add_argument("--findings", type=int, default=20, help="Markdown 里最多列几条发现")
    _add_output(child)

    child = command("roofline", cmd_roofline, "场景 2：逐算子 roofline、瓶颈和 decode batch 扫描")
    _add_target(child, calibration=False)
    child.add_argument("--weight-dtype", help="按这个权重精度估算（默认推荐精度路线的）")
    child.add_argument("--context", type=int, default=4096, help="decode 扫描的上下文长度")
    child.add_argument("--top-ops", type=int, default=12, help="每个场景列出的算子数")
    _add_output(child)

    for name, handler, help_text in (
        (
            "trace",
            cmd_trace,
            "场景 2：分析 /start_profile 采的 trace（算子异常、融合、通算重叠、多流、预取）",
        ),
        ("logs", cmd_logs, "场景 2：分析 launch_server 日志和 bench_serving 输出"),
    ):
        child = command(name, handler, help_text)
        child.add_argument("paths", nargs="+", help="文件或目录")
        _add_target(child, required=False)
        child.add_argument("--weight-dtype", help="成本模型的权重精度（默认推荐精度路线的）")
        child.add_argument("--findings", type=int, default=20, help="Markdown 里最多列几条发现")
        if name == "trace":
            child.add_argument(
                "--dialect", help="非 CUDA profiler 的事件/kernel 命名映射（YAML/JSON）"
            )
            child.add_argument(
                "--context-len", type=int, help="trace 没有详细注解时 decode 的上下文长度"
            )
            child.add_argument(
                "--server-url", default="http://127.0.0.1:30000", help="采集建议里的服务地址"
            )
        _add_output(child)

    child = command(
        "synth-trace", cmd_synth_trace, "生成注入了问题的合成 trace（演示/回归用，不是实测）"
    )
    _add_target(child, calibration=False)
    child.add_argument("--output-dir", required=True)
    child.add_argument("--ranks", type=int, help="生成几个 rank（默认 min(卡数, 2)）")
    child.add_argument("--clean", action="store_true", help="不注入问题")
    child.add_argument("--weight-dtype")
    child.add_argument("--seed", type=int, default=0)

    child = command("synth-log", cmd_synth_log, "生成合成 server 日志（演示校准用，不是实测）")
    _add_target(child, calibration=False)
    child.add_argument("-o", "--output", required=True)
    child.add_argument(
        "--decode-scale", type=float, default=1.3, help="设备比成本模型慢多少（decode）"
    )
    child.add_argument("--prefill-scale", type=float, default=1.15, help="同上（prefill）")
    child.add_argument("--weight-dtype")
    child.add_argument("--seed", type=int, default=1)

    child = command(
        "features",
        cmd_features,
        "场景 3：DP attention、CP、DCP、SP、HiCache、DeepEP、TBO、EPLB、投机解码怎么开、怎么调",
    )
    _add_target(child, workload=True)
    child.add_argument("--only", help=f"只评估这些特性，逗号分隔：{','.join(FEATURE_KEYS)}")
    _add_output(child)

    child = command(
        "deploy", cmd_deploy, "场景 5：按负载和 SLO 规划实例规模、副本、PD、router 和命令"
    )
    _add_target(child, workload=True, layout=False)
    _add_deploy_options(child)
    _add_output(child)

    child = command(
        "calibrate", cmd_calibrate, "用实测日志/压测结果校准成本模型，并对比部署方案的预测"
    )
    _add_target(child, calibration=False)
    child.add_argument("--log", nargs="+", required=True, help="launch_server 日志")
    child.add_argument("--bench", help="bench_serving --output-file 的 JSONL（或打印结果的文本）")
    child.add_argument("--plan", help="`sgl-agent deploy --json` 的输出，用来对比预测")
    child.add_argument("--weight-dtype")
    _add_output(child)

    child = command(
        "run",
        cmd_run,
        "一键流水线：场景 1→2→3→5，给了日志或 --demo 时再校准并重新规划",
    )
    _add_target(child, workload=True)
    _add_deploy_options(child)
    child.add_argument("--trace", nargs="+", help="profiler trace 文件或目录")
    child.add_argument("--log", nargs="+", help="launch_server 日志（用于场景 2 和校准）")
    child.add_argument("--bench", help="bench_serving --output-file 的 JSONL")
    child.add_argument(
        "--demo", action="store_true", help="没有设备时用合成 trace/日志/压测演示完整闭环"
    )
    child.add_argument("--findings", type=int, default=10)
    child.add_argument("-o", "--output-dir", default=DEFAULT_RUN_DIR)

    child = command("sync-knowledge", cmd_sync_knowledge, "从 SGLang 源码重新生成能力快照")
    child.add_argument("--sglang-src", required=True, help="SGLang 源码目录")
    child.add_argument("-o", "--output", default="sglang_capabilities.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return handler(args)
    except (ValueError, FileNotFoundError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2
