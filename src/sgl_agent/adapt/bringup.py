"""Change principles and an ordered bring-up plan for one model on one device.

The principles are the rules every change follows: where it plugs into SGLang, what it must
preserve and how it is verified. The plan orders the work into steps with the commands to run
and the exit criteria to meet before moving on. Both are derived from the hook checklist, the
operator inventory and the precision plan, so they only list work this model needs.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from sgl_agent.adapt.operators import GAP_STATUSES, OperatorInventory, operator_inventory
from sgl_agent.adapt.platform import HookChecklist, PlatformProfile
from sgl_agent.adapt.precision import PrecisionPlan, PrecisionRoute
from sgl_agent.commands import (
    Command,
    launch_server_command,
    multi_node_flags,
    parallel_server_flags,
    shell_command,
    tool_command,
)
from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.knowledge import SGLangKnowledge
from sgl_agent.knowledge.facts import (
    BENCH_SERVING_MODULE,
    DUMP_COMPARATOR_MODULE,
    GENERAL_PLUGIN_GROUP,
    MODEL_TRUNCATOR_MODULE,
    PLATFORM_PLUGIN_GROUP,
    RUN_EVAL_MODULE,
    source,
)
from sgl_agent.model.config import ModelArch
from sgl_agent.model.moe_tuning import triton_moe_config_key, tuning_command
from sgl_agent.model.parallel import ParallelConfig

SERVER_PORT = 30000
GSM8K_QUICK_EXAMPLES = 200
GSM8K_TEST_SIZE = 1319
TYPICAL_GSM8K_ACCURACY = 0.9
DUMP_ROOT = "/tmp/sgl-dump"
PROFILE_DIR = "/tmp/sgl-profile"
_GRAPH_OFF = {"--cuda-graph-backend-decode": "disabled", "--cuda-graph-backend-prefill": "disabled"}


@dataclass(frozen=True)
class Principle:
    title: str
    detail: str
    references: tuple[str, ...] = ()


@dataclass(frozen=True)
class BringupStep:
    key: str
    title: str
    goal: str
    actions: tuple[str, ...]
    commands: tuple[Command, ...] = ()
    exit_criteria: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    """Operators or hooks that have to exist before this step can pass."""
    references: tuple[str, ...] = ()


@dataclass
class BringupPlan:
    principles: list[Principle]
    steps: list[BringupStep]

    @property
    def command_problems(self) -> list[str]:
        return [
            f"{step.key} / {command.title}: {problem}"
            for step in self.steps
            for command in step.commands
            for problem in command.problems
        ]


def accuracy_tolerance(examples: int, accuracy: float = TYPICAL_GSM8K_ACCURACY) -> float:
    """Two standard errors of the difference between two runs scored on ``examples`` questions."""
    return 2 * math.sqrt(2 * accuracy * (1 - accuracy) / examples)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "model"


def _needs_remote_code(arch: ModelArch) -> bool:
    return bool(arch.raw.get("auto_map"))


def _generate_request(port: int = SERVER_PORT) -> Command:
    body = {
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
    }
    return shell_command(
        "发一个贪心解码请求",
        [
            "curl",
            "-s",
            f"http://127.0.0.1:{port}/generate",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(body, separators=(",", ":")),
        ],
    )


def _gap_items(inventory: OperatorInventory, *, skip: tuple[str, ...] = ()) -> list:
    return [
        item
        for item in inventory.requirements
        if item.status in GAP_STATUSES and item.key not in skip
    ]


def _share_text(item) -> str:
    parts = []
    if item.share_decode is not None:
        parts.append(f"decode {item.share_decode:.0%}")
    if item.share_prefill is not None:
        parts.append(f"prefill {item.share_prefill:.0%}")
    return "，".join(parts)


def change_principles(
    arch: ModelArch,
    hardware: HardwareSpec,
    profile: PlatformProfile,
    knowledge: SGLangKnowledge,
    inventory: OperatorInventory,
) -> list[Principle]:
    """The rules every change for this model/device follows, most fundamental first."""
    model_file = knowledge.model_file_for(arch.primary_architecture)
    if model_file:
        model_rule = (
            f"SGLang 已有 {arch.primary_architecture} 的实现（models/{model_file}.py），模型代码不需要改。"
            "确实要改模型时，不在 SGLang 里打补丁：把新类放进 SGLANG_EXTERNAL_MODEL_PACKAGE 指向的包"
            "（模块里定义 EntryClass，类名等于 config.json 的 architectures[0]）。"
        )
    else:
        model_rule = (
            f"SGLang 没有 {arch.primary_architecture} 的实现：先用 --model-impl transformers 跑通并对齐精度，"
            "再把正式实现放进 SGLANG_EXTERNAL_MODEL_PACKAGE 指向的包（模块里定义 EntryClass，类名等于 "
            "architectures[0]），通用的层与算子复用 SGLang 的 BaseFusedOp，平台适配才能自动生效。"
        )
    principles: list[Principle] = []
    if profile.in_tree:
        principles.append(
            Principle(
                "不需要平台插件，改动集中在参数和 kernel 配置",
                f"{hardware.name} 是 SGLang 内置的 {profile.platform} 平台，kernel 选择、图捕获、通信库都由 "
                "SGLang 决定。适配工作是：选精度路线、给 MoE/GEMM 补调优配置、用启动参数打开适合该卡的特性，"
                "并用场景 2 的 trace 分析确认实际走到的 kernel。",
                (source("platform_plugin_discovery"),),
            )
        )
    else:
        principles.append(
            Principle(
                "不改 SGLang 源码，只在扩展点上接入",
                f"厂商代码放在独立的包里：平台插件（entry point {PLATFORM_PLUGIN_GROUP}）负责设备身份、显存、"
                "通信、图捕获、量化和注意力默认值；通用插件（"
                f"{GENERAL_PLUGIN_GROUP}）负责在 ServerArgs 构造前追加参数选项，并用 HookRegistry 替换没有"
                "平台接缝的函数。升级 SGLang 时重新生成能力快照（sgl-agent sync-knowledge）复核清单，"
                "而不是 rebase 补丁。",
                (
                    source("platform_plugin_discovery"),
                    source("general_plugin_hooks"),
                    source("general_plugin_load_order"),
                ),
            )
        )
        principles.append(
            Principle(
                "在最窄的接缝上注册",
                "按算子选择接缝：BaseFusedOp 子类（norm、RoPE、激活、TopK、未量化 MoE）用 "
                "BaseFusedOp.register_oot_forward(类, fn, key)——按精确类匹配，子类要逐个注册——或在基类上"
                f"定义 forward_{profile.dispatch_key}（子类经 MRO 继承）；注意力用 register_attention_backend；"
                "量化 Linear/MoE 用 get_quantization_config 返回厂商 QuantizationConfig；采样用 "
                "register_sampler_backend；KV 布局用 get_kv_pool_cls；只有没有接缝的函数才用 HookRegistry "
                "REPLACE，并记成待上游的接缝。所有注册放在 init_backend() 里：BaseFusedOp 在第一次 forward "
                "时解析并缓存路径，之后的注册对已建好的模块无效。",
                (
                    source("oot_dispatch_order"),
                    source("fused_op_resolution_cache"),
                    source("attention_registry"),
                    source("oot_quantization_hook"),
                    source("sampler_backend"),
                    source("kv_pool_hook"),
                ),
            )
        )
        principles.append(
            Principle(
                "dispatch key 唯一，能力标志如实声明",
                "get_dispatch_key_name() 返回厂商唯一的 key：与 cuda/npu 等同名会选中那个平台的 kernel。"
                "PlatformCapabilities 只声明真正可用的能力：supports_triton=True 会让 SGLang 直接调用 Triton "
                "kernel（位置编号、MLA KV 写入、分页分配器），graph_capture=True 要求 get_graph_runner_cls()。",
                (
                    source("fused_op_dispatch"),
                    source("graph_runner_required"),
                    source("paged_allocator_triton"),
                ),
            )
        )
    principles.append(
        Principle(
            "模型代码走外部包，不改主干",
            model_rule,
            (source("external_model_package"), source("transformers_fallback")),
        )
    )
    if profile.out_of_tree:
        principles.append(
            Principle(
                "先跑通 torch 参考路径，再逐个替换成厂商 kernel",
                "SGLANG_FORCE_FUSED_OP_BACKEND=torch 让所有 BaseFusedOp 走 forward_native（优先级高于 OOT "
                "注册），先在设备上得到一份正确的基线；然后按算子清单的顺序（会报错 → 缺实现 → 有同步 → "
                "回落 torch，同级按耗时占比）一次只替换一个算子，每次替换都与基线逐层对齐。",
                (source("fused_op_force_backend"), source("oot_dispatch_order")),
            )
        )
    contracts = [
        item.title for item in inventory.requirements if item.contract and item.status != "in_tree"
    ]
    principles.append(
        Principle(
            "守住算子契约与数值",
            "厂商实现与 forward_native 的签名、返回值和可选参数为 None 时的行为保持一致（算子清单的 contract "
            "字段）；数值上 RMSNorm 在 fp32 中累加、TopK 输出 fp32 权重与 int32 专家 id、MLA 的 softmax scale "
            "使用传入的 scaling。"
            + (f"本模型需要按契约实现：{'、'.join(contracts)}。" if contracts else ""),
            (
                source("rmsnorm_contract"),
                source("rope_contract"),
                source("topk_output_dtypes"),
                source("rmsnorm_fp32"),
            ),
        )
    )
    syncs = [item.title for item in inventory.requirements if item.status == "slow"]
    principles.append(
        Principle(
            "前向路径不做 host 同步，保证可以图捕获",
            "图捕获要求每步 kernel 序列固定：前向里不能有 .item()/.cpu()、不能按设备张量做 Python 循环、"
            "不能动态分配。"
            + (
                f"当前路径上的同步点：{'、'.join(syncs)}。"
                if syncs
                else "当前算子清单里没有已知的同步点。"
            )
            + (
                ""
                if hardware.software.graph_capture or profile.in_tree
                else "该设备声明不支持图捕获，decode 只能 eager，每个 kernel 都付一次 launch 开销，算子融合更重要。"
            ),
            (
                source("graph_capture_capability"),
                source("compute_position"),
                source("moe_native_fallback"),
            ),
        )
    )
    principles.append(
        Principle(
            "每次改动都过精度与性能两道门",
            f"精度：gsm8k 与参考平台（或同设备 torch 基线）对比，快速迭代用 {GSM8K_QUICK_EXAMPLES} 题（差距门限约 "
            f"±{accuracy_tolerance(GSM8K_QUICK_EXAMPLES):.0%}，只能抓大错），合入前跑全量 {GSM8K_TEST_SIZE} 题"
            f"（约 ±{accuracy_tolerance(GSM8K_TEST_SIZE):.1%}）；性能：bench_serving 固定数据集和并发，与上一版"
            "对比吞吐、TTFT、TPOT。任何一项退化就回退该改动。",
            (source("run_eval"), source("benchmark_serving")),
        )
    )
    return principles


def _smoke_route(precision: PrecisionPlan) -> PrecisionRoute | None:
    """The route that needs the fewest vendor kernels: the plan's pick if complete, else BF16."""
    routes = {route.key: route for route in precision.routes}
    recommended = precision.recommended
    if recommended is not None and recommended.available and not recommended.kernels_missing:
        return recommended
    if "bf16" in routes and routes["bf16"].available:
        return routes["bf16"]
    return recommended


def _route_flags(route: PrecisionRoute | None) -> dict[str, Any]:
    return dict(route.flags) if route else {}


def _model_step(arch: ModelArch, knowledge: SGLangKnowledge) -> BringupStep | None:
    if knowledge.is_architecture_supported(arch.primary_architecture):
        return None
    return BringupStep(
        key="model",
        title="补模型实现",
        goal=f"SGLang 没有 {arch.primary_architecture}，先让模型能被加载",
        actions=(
            "用 --model-impl transformers 启动（SGLang 的 Transformers 后端），先得到能跑的精度基线",
            "在外部包里实现 SGLang 模型类：模块里定义 EntryClass，类名等于 config.json 的 architectures[0]，"
            "启动前设置 SGLANG_EXTERNAL_MODEL_PACKAGE=<包名>",
            "层用 SGLang 的 RMSNorm/RotaryEmbedding/FusedMoE/RadixAttention 等组件，平台注册才能覆盖到它们",
        ),
        exit_criteria=("两种实现在同一请求上的贪心输出一致",),
        references=(source("transformers_fallback"), source("external_model_package")),
    )


def _plugin_step(profile: PlatformProfile, checklist: HookChecklist, plugin: str) -> BringupStep:
    required = [item.name for item in checklist.by_level("required")]
    conditional = [item.name for item in checklist.by_level("conditional")]
    actions = [
        f'新建厂商包，在 pyproject.toml 声明 [project.entry-points."{PLATFORM_PLUGIN_GROUP}"] {plugin} = '
        '"<pkg>:activate"；activate() 在设备可用时返回 SRTPlatform 子类的全限定类名，否则返回 None',
        f"SRTPlatform 子类：_enum = PlatformEnum.OOT，device_type = {profile.device_type!r}，"
        f"get_dispatch_key_name() 返回 {profile.dispatch_key!r}",
        f"必须实现（{len(required)} 项）：" + "；".join(required),
    ]
    if conditional:
        actions.append(f"按本配置需要（{len(conditional)} 项）：" + "；".join(conditional))
    actions.append(
        f'早于 ServerArgs 的注册（add_attention_backend_choices 等）放进通用插件 [project.entry-points."'
        f'{GENERAL_PLUGIN_GROUP}"]'
    )
    actions.extend(f"先修正：{problem}" for problem in profile.problems)
    probe = (
        "from sglang.srt.platforms import current_platform as p; "
        "print(type(p).__name__, p.is_out_of_tree(), p.device_type, p.get_dispatch_key_name())"
    )
    return BringupStep(
        key="plugin",
        title="平台插件骨架",
        goal="SGLang 启动时识别并选中厂商平台",
        actions=tuple(actions),
        commands=(
            shell_command(
                "确认 SGLang 选中了厂商平台",
                ["python3", "-c", probe],
                env={"SGLANG_PLATFORM": plugin},
            ),
        ),
        exit_criteria=(
            f"打印厂商的 SRTPlatform 子类名、True、{profile.device_type!r}、{profile.dispatch_key!r}",
            "bring-up 期间始终设置 SGLANG_PLATFORM：不设置时插件里的异常只写日志，SGLang 静默回退到基础平台",
        ),
        references=(
            source("platform_plugin_discovery"),
            source("platform_resolution"),
            source("platform_interface"),
        ),
    )


def _smoke_step(
    arch: ModelArch,
    hardware: HardwareSpec,
    profile: PlatformProfile,
    knowledge: SGLangKnowledge,
    precision: PrecisionPlan,
    *,
    model_path: str,
    plugin: str,
) -> tuple[BringupStep, dict[str, Any]]:
    """Truncated model on one device through the torch reference path; returns its launch flags."""
    keep = max(2, arch.num_dense_layers + 1) if arch.moe else 2
    truncated = f"/tmp/{_slug(arch.name)}-{keep}layers"
    route = _smoke_route(precision)
    commands = [
        tool_command(
            MODEL_TRUNCATOR_MODULE,
            {"--input": model_path, "--output": truncated, "--keep-num-layers": keep},
            title=f"截出前 {keep} 层的小模型",
            note=(
                f"前 {arch.num_dense_layers} 层是稠密层，保留 {keep} 层才能覆盖至少一个 MoE 层"
                if arch.moe and arch.num_dense_layers
                else ""
            ),
        )
    ]
    served = truncated
    served_arch = arch
    actions = [
        "设置 SGLANG_FORCE_FUSED_OP_BACKEND=torch：所有 BaseFusedOp 走 forward_native，只依赖设备的 ATen 算子",
        "关闭图捕获（decode 与 prefill 都设为 disabled），先在 eager 下跑通",
    ]
    if route is not None and route.key == "bf16" and arch.quant is not None:
        served = f"{truncated}-bf16"
        served_arch = dataclasses.replace(arch, quant=None)
        if arch.quant.weight_dtype == "fp8":
            commands.append(
                shell_command(
                    "把截断后的 FP8 权重转成 BF16",
                    [
                        "python",
                        "fp8_cast_bf16.py",
                        "--input-fp8-hf-path",
                        truncated,
                        "--output-bf16-hf-path",
                        served,
                    ],
                    note="脚本在 DeepSeek-V3 仓库 inference/ 目录；转换后删除 config.json 的 quantization_config",
                )
            )
        actions.append(
            f"checkpoint 是 {arch.quant.method}，设备还没有对应的量化 kernel：冒烟阶段先用 BF16 副本，"
            "量化路线留到全量阶段（见精度部分）"
        )
    flags: dict[str, Any] = {
        "--model-path": served,
        "--device": profile.device_type,
        "--tp-size": 1,
        **_GRAPH_OFF,
        "--trust-remote-code": _needs_remote_code(arch),
        "--port": SERVER_PORT,
        **(_route_flags(route) if route and route.key != "checkpoint" else {}),
    }
    env = {"SGLANG_PLATFORM": plugin, "SGLANG_FORCE_FUSED_OP_BACKEND": "torch"}
    commands.append(
        launch_server_command(knowledge, flags, env=env, title="torch 参考路径启动截断模型")
    )
    commands.append(_generate_request())
    smoke_inventory = operator_inventory(
        served_arch, hardware, profile, knowledge, ParallelConfig(tp=1)
    )
    blocked = tuple(
        f"{item.key}（{item.title}：{item.status_label}）"
        for item in smoke_inventory.requirements
        if item.status in ("broken", "missing")
    )
    step = BringupStep(
        key="smoke",
        title="截断模型 + torch 参考路径冒烟",
        goal="单卡 eager 跑通，得到设备上的正确性基线",
        actions=tuple(actions),
        commands=tuple(commands),
        exit_criteria=(
            "日志出现 “The server is fired up and ready to roll!”，请求返回 16 个 token"
            "（截断模型的文本没有意义，这一步只验证链路）",
            "日志里的 max_total_num_tokens 与 sgl-agent 的显存规划同量级：明显偏大说明 "
            "get_current_memory_usage 少报了已用显存",
        ),
        blocked_by=blocked,
        references=(
            source("fused_op_force_backend"),
            source("model_truncator_layers"),
            source("server_ready_log"),
            source("kv_pool_budget"),
        ),
    )
    return step, flags


def _route_kernel_line(precision: PrecisionPlan) -> str | None:
    route = precision.recommended
    if route is None or not route.kernels:
        return None
    missing = set(route.kernels_missing)
    listed = "、".join(
        f"{kernel}（缺）" if kernel in missing else kernel for kernel in route.kernels
    )
    return (
        f"全量阶段的精度路线「{route.title}」需要这些量化 kernel：{listed}。清单里的 linear/moe_experts 描述的是 "
        "checkpoint 原样加载，按这条路线实现时以这里的 kernel 为准，经 get_quantization_config 返回的厂商 "
        "QuantizationConfig 接入"
    )


def _kernel_step(
    knowledge: SGLangKnowledge,
    inventory: OperatorInventory,
    precision: PrecisionPlan,
    *,
    smoke_flags: dict[str, Any],
    plugin: str,
) -> BringupStep:
    gaps = _gap_items(inventory, skip=("collectives", "moe_a2a"))
    ordered = []
    for index, item in enumerate(gaps, 1):
        share = _share_text(item)
        classes = f"；注册类 {'/'.join(item.op_classes)}" if item.op_classes else ""
        ordered.append(
            f"{index}. {item.title}【{item.status_label}{'，' + share if share else ''}】{classes}"
        )
    route_line = _route_kernel_line(precision)
    if route_line:
        ordered.append(route_line)
    reference_env = {
        "SGLANG_PLATFORM": plugin,
        "SGLANG_FORCE_FUSED_OP_BACKEND": "torch",
        "DUMPER_ENABLE": 1,
        "DUMPER_DIR": DUMP_ROOT,
        "DUMPER_EXP_NAME": "reference",
        "DUMPER_CLEANUP_PREVIOUS": 1,
    }
    target_env = {**reference_env, "DUMPER_EXP_NAME": "target"}
    target_env.pop("SGLANG_FORCE_FUSED_OP_BACKEND")
    commands = (
        launch_server_command(
            knowledge, smoke_flags, env=reference_env, title="基线：torch 路径 + dump"
        ),
        _generate_request(),
        launch_server_command(
            knowledge, smoke_flags, env=target_env, title="目标：厂商 kernel + dump"
        ),
        _generate_request(),
        tool_command(
            DUMP_COMPARATOR_MODULE,
            {"--baseline-path": f"{DUMP_ROOT}/reference", "--target-path": f"{DUMP_ROOT}/target"},
            title="逐层对比两次 dump",
        ),
    )
    actions = (
        "按下面的顺序一次替换一个算子（同一状态按耗时占比排序）：",
        *ordered,
        "每替换一个算子：用同一请求分别跑基线与目标并 dump（dump 时保持图捕获关闭，图重放不执行 Python hook），"
        "dump_comparator 报出的第一个超阈值张量就是引入误差的位置",
        "在插件里调用 sglang.kernels.enable_fused_op_trace()，跑完请求后用 get_fused_op_trace() 确认已替换的算子 "
        "backend 不再是 'torch'（forward_<key> 显示为 key，注册函数显示为函数名）",
    )
    return BringupStep(
        key="kernels",
        title="按清单替换厂商算子",
        goal="把回落路径换成厂商 kernel，每一步都与基线逐层对齐",
        actions=actions if ordered else ("算子清单里没有需要替换的算子",),
        commands=commands,
        exit_criteria=(
            "dump_comparator 没有超过阈值（默认 1e-3）的张量，或超阈值处已确认是可接受的舍入差异",
            "fused-op trace 里清单上的算子都不是 'torch'",
        ),
        references=(
            source("dumper"),
            source("dumper_non_intrusive"),
            source("dump_comparator"),
            source("fused_op_trace"),
        ),
    )


def _full_flags(
    arch: ModelArch,
    hardware: HardwareSpec,
    profile: PlatformProfile,
    route: PrecisionRoute | None,
    parallel: ParallelConfig,
    model_path: str,
) -> dict[str, Any]:
    flags: dict[str, Any] = {"--model-path": model_path}
    if profile.out_of_tree:
        flags["--device"] = profile.device_type
    flags.update(parallel_server_flags(parallel))
    if route is not None and route.key != "checkpoint":
        flags.update(route.flags)
    sizing = route.sizing if route is not None else None
    if sizing is not None and sizing.parallel == parallel and sizing.chunked_prefill_size:
        flags["--chunked-prefill-size"] = sizing.chunked_prefill_size
    flags["--trust-remote-code"] = _needs_remote_code(arch)
    flags.update(multi_node_flags(parallel.num_gpus, hardware.devices_per_node))
    flags["--host"] = "0.0.0.0"
    flags["--port"] = SERVER_PORT
    return flags


def _eval_commands() -> tuple[Command, ...]:
    return (
        tool_command(
            RUN_EVAL_MODULE,
            {"--port": SERVER_PORT, "--eval-name": "gsm8k", "--num-examples": GSM8K_QUICK_EXAMPLES},
            title=f"gsm8k 快速验收（{GSM8K_QUICK_EXAMPLES} 题）",
        ),
        tool_command(
            RUN_EVAL_MODULE,
            {"--port": SERVER_PORT, "--eval-name": "gsm8k", "--num-examples": GSM8K_TEST_SIZE},
            title=f"gsm8k 全量验收（{GSM8K_TEST_SIZE} 题）",
        ),
    )


def _full_model_step(
    arch: ModelArch,
    hardware: HardwareSpec,
    profile: PlatformProfile,
    knowledge: SGLangKnowledge,
    precision: PrecisionPlan,
    inventory: OperatorInventory,
    *,
    parallel: ParallelConfig,
    model_path: str,
    plugin: str,
) -> tuple[BringupStep, dict[str, Any], dict[str, Any]]:
    route = precision.recommended
    flags = _full_flags(arch, hardware, profile, route, parallel, model_path)
    env: dict[str, Any] = {"SGLANG_PLATFORM": plugin} if profile.out_of_tree else {}
    eager = {**flags, **_GRAPH_OFF} if profile.out_of_tree else flags
    actions = []
    if route is not None:
        actions.append(f"精度路线：{route.title}（{route.accuracy}）")
        actions.extend(route.steps)
        if route.sizing is not None:
            actions.append(
                f"{route.sizing.parallel.describe()}，每卡权重约 {route.sizing.weights_gb_per_gpu:.1f} GB，"
                f"KV 池约 {route.sizing.kv_tokens:,} token"
            )
    else:
        actions.append(
            "没有能放下该模型的精度路线（见精度部分的 blockers），先解决显存或 kernel 缺口"
        )
    if parallel.num_gpus > 1 and profile.out_of_tree:
        actions.append(
            f"先用截断模型加 --tp-size 2 验证集合通信（torch.distributed 后端 {profile.distributed_backend!r}），"
            "再上全量"
        )
    if parallel.num_gpus > hardware.devices_per_node:
        actions.append(
            "跨节点：把 <node0_ip> 换成 0 号节点地址，其余节点只改 --node-rank；请求只发给 0 号节点"
        )
    blocked = tuple(
        f"{item.key}（{item.title}：{item.status_label}）"
        for item in inventory.requirements
        if item.status in ("broken", "missing")
    )
    step = BringupStep(
        key="full_model",
        title="全量模型 + 精度验收",
        goal=f"按推荐精度路线在 {parallel.num_gpus} 卡上跑全量权重，gsm8k 与参考平台一致",
        actions=tuple(actions),
        commands=(
            launch_server_command(
                knowledge,
                eager,
                env=env,
                title="启动全量模型" + ("（eager）" if profile.out_of_tree else ""),
            ),
            *_eval_commands(),
        ),
        exit_criteria=(
            f"gsm8k 与参考平台的差距在门限内：{GSM8K_QUICK_EXAMPLES} 题约 ±{accuracy_tolerance(GSM8K_QUICK_EXAMPLES):.0%}，"
            f"全量 {GSM8K_TEST_SIZE} 题约 ±{accuracy_tolerance(GSM8K_TEST_SIZE):.1%}",
            "长上下文（接近 context_length 的输入）不出现乱码：RoPE cache 精度与 KV dtype 都会在这里暴露",
        ),
        blocked_by=blocked,
        references=(source("run_eval"), source("launch_server")),
    )
    return step, flags, env


def _graph_step(
    hardware: HardwareSpec,
    knowledge: SGLangKnowledge,
    *,
    flags: dict[str, Any],
    env: dict[str, Any],
    compile_required: bool,
) -> BringupStep:
    software = hardware.software
    actions = []
    if software.graph_capture:
        actions.append(
            "去掉 --cuda-graph-backend-decode disabled，让 SGLang 捕获 decode 图（get_graph_runner_cls 返回厂商 runner）"
        )
        actions.append("对比同一请求在 eager 与图模式下的贪心输出，必须逐 token 一致")
    else:
        actions.append(
            "设备声明不支持图捕获：decode 保持 eager，优先做减少 kernel 数的融合（场景 2）"
        )
    if compile_required:
        actions.append(
            "get_compile_backend() 默认返回 'inductor'，而设备没有 torch.compile 后端：先返回 'eager' 保证正确，"
            "再接入厂商编译后端"
        )
    elif software.torch_compile:
        actions.append(
            "设备有 torch.compile 后端时可再 A/B --enable-torch-compile（只覆盖 batch ≤ --torch-compile-max-bs 的 decode 图）"
        )
    command = launch_server_command(knowledge, flags, env=env, title="打开 decode 图后启动")
    return BringupStep(
        key="graph",
        title="图捕获与编译",
        goal="消掉 decode 的逐 kernel launch 开销",
        actions=tuple(actions),
        commands=(command,) if software.graph_capture else (),
        exit_criteria=(
            "日志里 decode 图按各个 batch size 捕获成功，无 host 同步报错",
            "图模式与 eager 的贪心输出一致；小 batch decode 的 TPOT 明显下降",
        ),
        references=(
            source("graph_capture_capability"),
            source("graph_runner_required"),
            source("cuda_graph_defaults"),
            source("torch_compile_bs"),
        ),
    )


def _perf_step(
    arch: ModelArch,
    hardware: HardwareSpec,
    knowledge: SGLangKnowledge,
    *,
    flags: dict[str, Any],
    env: dict[str, Any],
    parallel: ParallelConfig,
    model_path: str,
    hardware_ref: str,
    in_tree: bool,
) -> BringupStep:
    bench = tool_command(
        BENCH_SERVING_MODULE,
        {
            "--backend": "sglang",
            "--port": SERVER_PORT,
            "--dataset-name": "random",
            "--random-input-len": 1024,
            "--random-output-len": 512,
            "--random-range-ratio": 1.0,
            "--num-prompts": 256,
            "--max-concurrency": 32,
        },
        title="固定负载的性能基线",
    )
    profiled = tool_command(
        BENCH_SERVING_MODULE,
        {
            "--backend": "sglang",
            "--port": SERVER_PORT,
            "--dataset-name": "random",
            "--random-input-len": 1024,
            "--random-output-len": 64,
            "--random-range-ratio": 1.0,
            "--num-prompts": 32,
            "--max-concurrency": 32,
            "--profile": True,
        },
        title="采集 profiler trace",
        note=f"服务端需带 SGLANG_TORCH_PROFILER_DIR={PROFILE_DIR} 启动",
    )
    server = launch_server_command(
        knowledge,
        flags,
        env={**env, "SGLANG_TORCH_PROFILER_DIR": PROFILE_DIR},
        title="带 profiler 目录启动",
    )
    analyze = shell_command(
        "用场景 2 分析 trace",
        [
            "sgl-agent",
            "trace",
            f"{PROFILE_DIR}/<trace>.trace.json.gz",
            "--model",
            model_path,
            "--hardware",
            hardware_ref,
            "--tp",
            str(parallel.tp),
        ],
    )
    actions = [
        "固定数据集、输入输出长度和并发，记录吞吐、TTFT、TPOT 作为后续每次改动的对照",
        "采一次 profiler trace，用 sgl-agent trace 找算子异常、融合与通算重叠机会（场景 2）",
    ]
    if in_tree and arch.moe is not None:
        key = triton_moe_config_key(arch, parallel, hardware)
        if key is not None:
            tuned = knowledge.moe_tuned_configs(
                device=key.device,
                experts=key.experts,
                intermediate=key.intermediate,
                dtype=key.dtype,
                block_shape=list(key.block_shape) if key.block_shape else None,
            )
            if tuned:
                actions.append(f"SGLang 自带该形状的 Triton MoE 调优配置 {key.file_name()}")
            else:
                actions.append(
                    f"SGLang 没有 {key.file_name()}（走 Triton MoE 时会用默认配置）：在目标卡上运行 "
                    f"`{tuning_command(arch, parallel, key, model_path)}`"
                )
    return BringupStep(
        key="perf",
        title="性能基线与 trace",
        goal="得到可复现的性能基线，交给场景 2/3/5 继续调优",
        actions=tuple(actions),
        commands=(bench, server, profiled, analyze),
        exit_criteria=("基线数字与 trace 归档；sgl-agent trace 的高优先级发现进入下一轮迭代",),
        references=(
            source("benchmark_serving"),
            source("profiler_output"),
            source("profile_request"),
            source("moe_config_dir"),
        ),
    )


def _in_tree_support_step(
    arch: ModelArch, knowledge: SGLangKnowledge, precision: PrecisionPlan
) -> BringupStep:
    model_file = knowledge.model_file_for(arch.primary_architecture)
    route = precision.recommended
    actions = [
        f"模型实现：models/{model_file}.py"
        if model_file
        else "SGLang 没有该架构的实现（见 model 步骤）",
        f"精度路线：{route.title}" if route else "没有可行的精度路线（见精度部分）",
        "用 python3 -m sglang.check_env 记录驱动、CUDA、torch、sglang 与 sgl-kernel 版本",
    ]
    return BringupStep(
        key="support",
        title="确认内置支持与环境",
        goal="确认模型、精度路线和软件版本都在 SGLang 的支持范围内",
        actions=tuple(actions),
        commands=(shell_command("记录环境版本", ["python3", "-m", "sglang.check_env"]),),
        exit_criteria=("check_env 输出归档，版本与生成能力快照的 SGLang 一致",),
    )


def bringup_plan(
    arch: ModelArch,
    hardware: HardwareSpec,
    profile: PlatformProfile,
    knowledge: SGLangKnowledge,
    *,
    checklist: HookChecklist,
    inventory: OperatorInventory,
    precision: PrecisionPlan,
    parallel: ParallelConfig,
    model_path: str | None = None,
    hardware_ref: str | None = None,
    plugin_name: str | None = None,
    compile_required: bool = False,
) -> BringupPlan:
    """Principles plus ordered steps; ``compile_required`` comes from the get_compile_backend hook.

    ``hardware_ref`` is what the user passed as ``--hardware`` (preset name or YAML path).
    """
    model_path = model_path or arch.hf_repo or arch.name
    hardware_ref = hardware_ref or hardware.name
    plugin = plugin_name or profile.device_type
    principles = change_principles(arch, hardware, profile, knowledge, inventory)
    steps: list[BringupStep] = []
    model_step = _model_step(arch, knowledge)
    if model_step is not None:
        steps.append(model_step)
    if profile.in_tree:
        steps.append(_in_tree_support_step(arch, knowledge, precision))
        full, flags, env = _full_model_step(
            arch,
            hardware,
            profile,
            knowledge,
            precision,
            inventory,
            parallel=parallel,
            model_path=model_path,
            plugin=plugin,
        )
        steps.append(full)
    else:
        steps.append(_plugin_step(profile, checklist, plugin))
        smoke, smoke_flags = _smoke_step(
            arch, hardware, profile, knowledge, precision, model_path=model_path, plugin=plugin
        )
        steps.append(smoke)
        steps.append(
            _kernel_step(knowledge, inventory, precision, smoke_flags=smoke_flags, plugin=plugin)
        )
        full, flags, env = _full_model_step(
            arch,
            hardware,
            profile,
            knowledge,
            precision,
            inventory,
            parallel=parallel,
            model_path=model_path,
            plugin=plugin,
        )
        steps.append(full)
        steps.append(
            _graph_step(
                hardware, knowledge, flags=flags, env=env, compile_required=compile_required
            )
        )
    graphs_work = profile.in_tree or hardware.software.graph_capture
    steps.append(
        _perf_step(
            arch,
            hardware,
            knowledge,
            flags=flags if graphs_work else {**flags, **_GRAPH_OFF},
            env=env,
            parallel=parallel,
            model_path=model_path,
            hardware_ref=hardware_ref,
            in_tree=profile.in_tree,
        )
    )
    return BringupPlan(principles=principles, steps=steps)
