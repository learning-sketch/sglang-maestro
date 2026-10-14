"""Which SGLang platform-plugin hooks a device needs for one model, derived from the snapshot.

An out-of-tree device plugs into SGLang through a ``sglang.srt.platforms`` entry point that
returns an ``SRTPlatform`` subclass. The snapshot records every ``DeviceMixin``/``SRTPlatform``
hook with its default behaviour and call sites; the rules below decide which of them this
model, parallel size and software stack actually reach.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sgl_agent.adapt.rope import rope_class_for
from sgl_agent.hardware.spec import HardwareSpec
from sgl_agent.knowledge import SGLangKnowledge
from sgl_agent.knowledge.facts import (
    APPLY_ROTARY_POS_EMB_MODEL_FILES,
    DEFAULT_OOT_COMPILE_BACKEND,
    DEFAULT_OOT_DISPATCH_KEY,
    DEVICE_TO_DISTRIBUTED_BACKEND,
    GENERAL_PLUGIN_GROUP,
    IN_TREE_COMPILE_BACKENDS,
    IN_TREE_DEVICE_TYPES,
    IN_TREE_DISPATCH_KEYS,
    PLATFORM_PLUGIN_GROUP,
    source,
)
from sgl_agent.model.config import ModelArch

OOT_PLATFORM_NAMES = frozenset({"oot", "out_of_tree", "out-of-tree", "plugin"})
HOOK_LEVELS = ("required", "conditional", "recommended", "optional")
_LEVEL_RANK = {name: index for index, name in enumerate(HOOK_LEVELS)}
COMPILE_SITE_APPLIES = ("yes", "conditional", "no", "manual")
INDUCTOR = "inductor"


@dataclass(frozen=True)
class PlatformProfile:
    """How SGLang will see the device: identity, fused-op dispatch key, compile and comm backends."""

    platform: str
    in_tree: bool
    device_type: str
    dispatch_key: str | None
    """``forward_<key>`` BaseFusedOp selects; None when the platform has no platform forward."""
    compile_backend: str
    distributed_backend: str
    default_distributed_backend: str
    """What ``get_torch_distributed_backend_str()`` returns without an override."""
    dispatch_key_collides: bool = False
    """An out-of-tree key equal to an in-tree one selects that platform's ``forward_*`` kernels."""
    problems: tuple[str, ...] = ()

    @property
    def out_of_tree(self) -> bool:
        return not self.in_tree


@dataclass(frozen=True)
class HookRequirement:
    name: str
    owner: str
    level: str
    reason: str
    how: str
    default: str | None = None
    call_sites: tuple[str, ...] = ()
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.level not in _LEVEL_RANK:
            raise ValueError(f"unknown hook level {self.level!r}; known: {HOOK_LEVELS}")


@dataclass(frozen=True)
class CompileSiteImpact:
    function: str
    location: str
    applies: str
    reason: str

    def __post_init__(self) -> None:
        if self.applies not in COMPILE_SITE_APPLIES:
            raise ValueError(f"unknown applicability {self.applies!r}")


@dataclass
class HookChecklist:
    requirements: list[HookRequirement] = field(default_factory=list)
    missing_from_snapshot: list[str] = field(default_factory=list)
    """Curated hooks the snapshot does not list (renamed or removed upstream)."""

    def by_level(self, level: str) -> list[HookRequirement]:
        return [item for item in self.requirements if item.level == level]


def resolve_platform(hardware: HardwareSpec, knowledge: SGLangKnowledge) -> PlatformProfile:
    software = hardware.software
    platform = (software.platform or "").strip().lower()
    in_tree_names = set(knowledge.raw.get("in_tree_platforms") or IN_TREE_DEVICE_TYPES)
    problems: list[str] = []
    if platform in in_tree_names and platform not in OOT_PLATFORM_NAMES:
        device_type = IN_TREE_DEVICE_TYPES.get(platform)
        if device_type is None:
            device_type = platform
            problems.append(
                f"SGLang 新增了内置平台 {platform!r}，facts.py 没有它的 device_type，按平台名处理"
            )
        if software.device_type and software.device_type != device_type:
            problems.append(
                f"software.device_type={software.device_type!r} 与内置平台 {platform} 的 "
                f"device_type={device_type!r} 不一致，以 SGLang 内置值为准"
            )
        default_backend = DEVICE_TO_DISTRIBUTED_BACKEND.get(device_type, "gloo")
        return PlatformProfile(
            platform=platform,
            in_tree=True,
            device_type=device_type,
            dispatch_key=IN_TREE_DISPATCH_KEYS.get(platform),
            compile_backend=IN_TREE_COMPILE_BACKENDS.get(platform, INDUCTOR),
            distributed_backend=default_backend,
            default_distributed_backend=default_backend,
            problems=tuple(problems),
        )

    if platform not in OOT_PLATFORM_NAMES:
        problems.append(
            f"platform={software.platform!r} 不是 SGLang 内置平台 {sorted(in_tree_names)}，"
            "按 out-of-tree 插件处理"
        )
    device_type = (software.device_type or "").strip()
    if not device_type:
        device_type = "cpu"
        readers = knowledge.platform_attributes.get("device_type", {}).get("readers") or []
        example = f"（如 {readers[0]}）" if readers else ""
        problems.append(
            "software.device_type 未填写：DeviceMixin.device_type 默认是 'cpu'，SGLang 有 "
            f"{len(readers)} 处按它判断设备{example}，插件必须把它设成厂商的 torch.device 类型"
        )
    dispatch_hook = knowledge.platform_hook("get_dispatch_key_name") or {}
    dispatch_key = (software.dispatch_key or "").strip() or str(
        dispatch_hook.get("default_return") or DEFAULT_OOT_DISPATCH_KEY
    )
    in_tree_methods = knowledge.fused_op_dispatch.get("platform_methods") or {
        key: [f"forward_{key}"] for key in IN_TREE_DISPATCH_KEYS.values()
    }
    collides = dispatch_key in in_tree_methods
    if collides:
        methods = "/".join(in_tree_methods[dispatch_key])
        problems.append(
            f"dispatch_key={dispatch_key!r} 与内置平台同名：BaseFusedOp 会选中 {methods}，"
            f"那是 {dispatch_key} 平台的 kernel；请换成厂商唯一的 key"
        )
    compile_hook = knowledge.platform_hook("get_compile_backend") or {}
    compile_backend = (software.compile_backend or "").strip() or str(
        compile_hook.get("default_return") or DEFAULT_OOT_COMPILE_BACKEND
    )
    default_backend = DEVICE_TO_DISTRIBUTED_BACKEND.get(device_type, "gloo")
    distributed_backend = (software.collective_library or "").strip() or default_backend
    return PlatformProfile(
        platform=platform or "oot",
        in_tree=False,
        device_type=device_type,
        dispatch_key=dispatch_key,
        compile_backend=compile_backend,
        distributed_backend=distributed_backend,
        default_distributed_backend=default_backend,
        dispatch_key_collides=collides,
        problems=tuple(problems),
    )


def compile_site_impact(
    arch: ModelArch, profile: PlatformProfile, knowledge: SGLangKnowledge, *, tp: int
) -> list[CompileSiteImpact]:
    """Which import-time ``torch.compile(backend=get_compiler_backend())`` functions this model runs."""
    model_file = knowledge.model_file_for(arch.primary_architecture)
    moe = arch.moe
    grouped = bool(moe and moe.routing.endswith("grouped_topk"))
    impacts: list[CompileSiteImpact] = []
    for site in knowledge.platform_compile_sites:
        function = str(site.get("function") or "")
        location = str(site.get("location") or "")
        short_name = function.rsplit(".", 1)[-1]
        parts = function.split(".")
        if site.get("referenced") is False:
            applies, reason = "no", "定义了但没有任何调用（死代码）"
        elif site.get("disabled_when") == "_is_npu" and profile.platform == "npu":
            applies, reason = "no", "NPU 上 disable=True"
        elif short_name == "get_masked_input_and_mask":
            applies = "yes" if tp > 1 else "no"
            reason = f"词表并行 embedding 在 TP>1 时屏蔽本 rank 之外的 token（当前 TP={tp}）"
        elif short_name == "apply_scaling_penalties":
            applies, reason = "conditional", "请求设置 repetition_penalty 时每步调用"
        elif short_name == "biased_grouped_topk_impl":
            applies = "yes" if grouped and moe and moe.has_router_bias else "no"
            reason = "带 correction bias 的分组 top-k 路由（DeepSeek-V3 类）在非 CUDA/HIP 上走它"
        elif short_name == "grouped_topk_gpu":
            applies = "yes" if grouped and moe and not moe.has_router_bias else "no"
            reason = "不带 bias 的分组 top-k 路由"
        elif short_name == "_biased_grouped_topk_postprocess":
            applies, reason = "no", "只在 _is_cuda 分支里调用（EPLB/共享专家 id 重映射）"
        elif short_name == "apply_rotary_pos_emb_native":
            uses = rope_class_for(arch).class_name == "FourierRotaryEmbedding" or (
                model_file in APPLY_ROTARY_POS_EMB_MODEL_FILES
            )
            vision = isinstance(arch.raw.get("vision_config"), dict)
            applies = "yes" if uses else ("conditional" if vision else "no")
            reason = (
                "FourierRotaryEmbedding、直接调用 apply_rotary_pos_emb 的模型文件和视觉塔 RoPE 使用"
            )
        elif short_name == "draft_decode_set_expand_metadata":
            applies, reason = "no", "属于 FA3 注意力后端（CUDA）"
        elif short_name == "mask_cpu_expert_ids":
            applies, reason = "no", "只用于 KTransformers CPU 专家卸载（kt_ep_wrapper）"
        elif len(parts) > 3 and parts[:3] == ["sglang", "srt", "models"]:
            module = parts[3]
            if module == "dflash":
                applies, reason = "conditional", "只在 DFLASH 投机解码的 draft 模型里使用"
            elif model_file is None:
                applies, reason = "manual", "该架构没有 SGLang 实现，无法按模型文件判断"
            elif module == model_file:
                applies, reason = "yes", f"本模型文件 models/{module}.py 内的函数"
            else:
                applies, reason = "no", f"属于其他模型文件 models/{module}.py"
        else:
            applies, reason = "manual", "新出现的编译点，需要人工确认是否在本模型的路径上"
        impacts.append(CompileSiteImpact(function, location, applies, reason))
    return impacts


def _default_text(entry: dict | None) -> str | None:
    if not entry:
        return None
    default = entry.get("default")
    if default == "raises":
        return "默认抛 NotImplementedError"
    if default == "returns":
        return f"默认返回 {entry.get('default_return')!r}"
    if default == "noop":
        return "默认空操作"
    if default == "implemented":
        return "已有默认实现"
    return str(default) if default is not None else None


def hook_checklist(
    arch: ModelArch,
    hardware: HardwareSpec,
    profile: PlatformProfile,
    knowledge: SGLangKnowledge,
    *,
    num_gpus: int,
    compile_sites: list[CompileSiteImpact] | None = None,
) -> HookChecklist:
    """Hooks, attributes and registrations an out-of-tree plugin needs for this model."""
    checklist = HookChecklist()
    if profile.in_tree:
        return checklist
    software = hardware.software
    hooks = knowledge.platform_hooks
    capabilities = knowledge.platform_capabilities
    attributes = knowledge.platform_attributes
    covered: set[str] = set()

    def snapshot_hooks(*names: str) -> tuple[str | None, tuple[str, ...]]:
        defaults: list[str] = []
        sites: list[str] = []
        for name in names:
            covered.add(name)
            entry = hooks.get(name)
            if entry is None:
                checklist.missing_from_snapshot.append(name)
                continue
            text = _default_text(entry)
            if text:
                defaults.append(f"{name}: {text}" if len(names) > 1 else text)
            sites.extend(entry.get("call_sites") or [])
        return ("；".join(defaults) or None), tuple(dict.fromkeys(sites))

    def add(
        name: str,
        owner: str,
        level: str,
        reason: str,
        how: str,
        *,
        hook_names: tuple[str, ...] = (),
        default: str | None = None,
        call_sites: tuple[str, ...] = (),
        references: tuple[str, ...] = (),
    ) -> None:
        if hook_names:
            hook_default, hook_sites = snapshot_hooks(*hook_names)
            default = default or hook_default
            call_sites = tuple(dict.fromkeys(call_sites + hook_sites))
        checklist.requirements.append(
            HookRequirement(name, owner, level, reason, how, default, call_sites, references)
        )

    device_type = profile.device_type
    attention = arch.attention

    add(
        f"entry point {PLATFORM_PLUGIN_GROUP} → activate()",
        "打包（pyproject.toml）",
        "required",
        "SGLang 只通过该 entry point 组发现 out-of-tree 平台",
        f'[project.entry-points."{PLATFORM_PLUGIN_GROUP}"] <name> = "<pkg>:activate"；activate() '
        "在硬件可用时返回 SRTPlatform 子类的全限定类名字符串，否则返回 None。部署时设置 "
        "SGLANG_PLATFORM=<name>：不设置时插件抛出的异常只会写日志，SGLang 静默回退到基础平台；"
        "设置后找不到插件或 activate() 返回 None 都会直接报错。",
        references=(source("platform_plugin_discovery"),),
    )

    device_attr = attributes.get("device_type", {})
    name_attr = attributes.get("device_name", {})
    readers = tuple(
        dict.fromkeys((device_attr.get("readers") or []) + (name_attr.get("readers") or []))
    )
    add(
        "_enum / device_type / device_name",
        "SRTPlatform 类属性",
        "required",
        f"is_out_of_tree() 只看 _enum；device_type 默认 'cpu'，有 "
        f"{len(device_attr.get('readers') or [])} 处读取",
        f'_enum = PlatformEnum.OOT；device_type = "{device_type}"（与 torch.device 的类型一致，'
        "PrivateUse1 设备先 torch.utils.rename_privateuse1_backend）；device_name 用于日志和"
        " apply_server_args_defaults 的来源标记",
        default="_enum=UNSPECIFIED, device_type='cpu', device_name='unknown'",
        call_sites=readers,
        references=(source("platform_interface"),),
    )

    add(
        "get_device",
        "DeviceMixin",
        "required",
        "未传 --device 时 ServerArgs 取 get_device() 的返回值再 .split(':')；out-of-tree 平台回落到 "
        "current_platform.get_device(None)",
        f'get_device(device_id=None) 返回字符串：None → "{device_type}"，整数 → '
        f'"{device_type}:<id>"（同一字符串也能直接传给 torch.device）；或者部署时总是显式传 '
        f"--device {device_type}",
        hook_names=("get_device",),
        references=(source("oot_device_fallback"), source("server_device_default")),
    )

    add(
        "get_device_total_memory / get_current_memory_usage",
        "DeviceMixin",
        "required",
        "out-of-tree 平台的可用显存 = get_device_total_memory() - get_current_memory_usage()；"
        "KV 池预算 = 可用显存 - 加载前显存×(1-mem_fraction_static)；SGLang 也按总显存挑 "
        "chunked_prefill_size 与 decode graph max_bs 的默认值",
        "get_device_total_memory 返回设备总字节数；get_current_memory_usage 应返回设备上当前已占用"
        "的字节数。内置 NPU/XPU 返回 torch 分配器峰值（max_memory_allocated），它不含通信库缓冲、"
        "其他进程和分配器缓存，会让 KV 池偏大而 OOM——优先使用驱动层的已用显存。上线前把日志里的 "
        "max_total_num_tokens 与 sgl-agent 的显存规划对比。",
        hook_names=("get_device_total_memory", "get_current_memory_usage"),
        references=(source("oot_memory_hooks"), source("kv_pool_budget")),
    )

    backend_default, backend_sites = snapshot_hooks("get_torch_distributed_backend_str")
    distributed = profile.distributed_backend
    default_backend = profile.default_distributed_backend
    if num_gpus > 1 and distributed != default_backend:
        level, reason = (
            "required",
            f"默认按 device_type={device_type!r} 查表得到 {default_backend!r}，而设备的集合通信库"
            f"是 {distributed!r}；{num_gpus} 卡的 TP/EP 通信都走它",
        )
    elif num_gpus > 1 and default_backend == "gloo":
        level, reason = (
            "required",
            f"device_type={device_type!r} 不在内置映射表里，默认回落到 CPU 后端 'gloo'，"
            "设备张量的集合通信需要厂商通信库",
        )
    else:
        level, reason = "optional", f"默认值 {default_backend!r} 已满足当前配置"
    add(
        "get_torch_distributed_backend_str",
        "DeviceMixin",
        level,
        reason,
        f"返回 {distributed!r}；该名字必须先通过 torch.distributed.Backend.register_backend 注册",
        default=backend_default,
        call_sites=backend_sites,
        references=(source("distributed_backend_fallback"), source("platform_distributed_backend")),
    )

    choices = knowledge.choices("--attention-backend") or []
    declared = software.default_attention_backend
    if declared and declared not in choices:
        attention_reason = (
            f"YAML 声明默认注意力后端 {declared!r}，它不在 SGLang 的 --attention-backend 选项里"
        )
    elif declared:
        attention_reason = (
            f"YAML 声明默认注意力后端 {declared!r}（复用了内置名字，注册时会覆盖内置工厂）"
        )
    else:
        attention_reason = (
            "out-of-tree 平台的默认注意力后端只来自该 hook（默认抛异常），YAML 也没有声明可用的后端"
        )
    attention_how = (
        '在 init_backend() 或通用插件里用 @register_attention_backend("<name>") 注册工厂函数'
        "（同名会静默覆盖）；get_default_attention_backend() 返回该名字；要允许 "
        "--attention-backend <name>，在通用插件里调用 sglang.srt.server_args."
        'add_attention_backend_choices(["<name>"])（必须早于 ServerArgs 构造）。'
    )
    references = [source("oot_default_attention_backend"), source("attention_registry")]
    if attention.is_mla:
        attention_how += (
            ' MLA 模型还要 AttentionBackendRegistry.register("<name>", handler) 决定每个 batch 的'
            "前向方式；未注册的名字使用 triton handler：无前缀的 prefill 走 MHA（q/k head_dim="
            f"{attention.qk_head_dim}，v head_dim={attention.v_head_dim}），其余走吸收后的 MLA"
            f"（latent {attention.kv_lora_rank}+RoPE {attention.qk_rope_head_dim}，单个 KV 头）。"
        )
        references.append(source("mla_backend_handler"))
    references.append(source("attention_backend_choices"))
    add(
        "get_default_attention_backend + register_attention_backend",
        "SRTPlatform / 注意力注册表",
        "required",
        attention_reason,
        attention_how,
        hook_names=("get_default_attention_backend",),
        references=tuple(references),
    )

    graph_capability = capabilities.get("graph_capture", {})
    if software.graph_capture:
        add(
            "capabilities.graph_capture + get_graph_runner_cls (+ get_full_graph_backend_cls)",
            "SRTPlatform",
            "required",
            "YAML 声明设备支持图捕获；graph_capture=True 时 SGLang 捕获 decode 图并要求 "
            "get_graph_runner_cls()，返回 None 会直接报错",
            "capabilities = PlatformCapabilities(graph_capture=True, ...)；get_graph_runner_cls() "
            "返回厂商 graph runner；如果复用 SGLang 的 runner backend，再实现 "
            "get_full_graph_backend_cls()（否则用 torch.cuda.CUDAGraph 的 FullCudaGraphBackend）。"
            "被捕获的 kernel 不能做 host 同步（.cpu()/.item()）或动态分配。",
            hook_names=("get_graph_runner_cls", "get_full_graph_backend_cls"),
            call_sites=tuple(graph_capability.get("readers") or ()),
            references=(
                source("graph_capture_capability"),
                source("graph_runner_required"),
                source("full_graph_backend"),
            ),
        )
    else:
        add(
            "capabilities.graph_capture",
            "PlatformCapabilities",
            "recommended",
            "graph_capture 默认 False：decode 不会报错，而是静默走 eager，每个 kernel 都付一次 "
            "launch 开销（小 batch decode 影响最大，见 adapt.graph 发现）",
            "设备支持图捕获后再打开，并实现 get_graph_runner_cls()",
            default=f"默认 {graph_capability.get('default', False)!r}",
            call_sites=tuple(graph_capability.get("readers") or ()),
            references=(source("graph_capture_capability"),),
        )

    add(
        "get_piecewise_backend_cls",
        "SRTPlatform",
        "conditional" if software.piecewise_graph else "optional",
        "非 CUDA 平台 prefill 图默认关闭；只有启用 SGLang 的 piecewise 编译时才调用，返回 None 会报错",
        "返回 piecewise backend 类（XPU/NPU 的内置实现可作参考）",
        hook_names=("get_piecewise_backend_cls",),
        references=(source("prefill_graph_default"), source("oot_guards")),
    )

    applicable = [site for site in compile_sites or [] if site.applies in ("yes", "conditional")]
    if applicable:
        names = "、".join(site.function.rsplit(".", 1)[-1] for site in applicable[:4])
        if len(applicable) > 4:
            names += " 等"
        if profile.compile_backend == INDUCTOR and not software.torch_compile:
            level = "required"
            reason = (
                f"本模型会调用 {len(applicable)} 个 import 时就用 torch.compile(backend="
                f"get_compiler_backend()) 装饰的函数（{names}），插件默认返回 'inductor'，"
                "而 YAML 声明 torch_compile=false"
            )
        elif profile.compile_backend == INDUCTOR:
            level = "recommended"
            reason = f"本模型会调用 {len(applicable)} 个编译点（{names}），确认 inductor 能为该设备生成代码"
        else:
            level = "optional"
            reason = f"YAML 声明 compile_backend={profile.compile_backend!r}，覆盖 {len(applicable)} 个编译点"
    else:
        level, reason = "optional", "本模型路径上没有 import 时编译的函数"
    add(
        "get_compile_backend",
        "SRTPlatform",
        level,
        reason,
        "返回 'eager'（只追踪不编译，正确但没有融合）或厂商注册的 torch.compile backend 名字；"
        "mode 参数会透传",
        hook_names=("get_compile_backend",),
        references=(source("oot_compile_backend"), source("torch_compile_sites")),
    )

    add(
        "get_torch_profiler_activity_str / get_torch_profiler_activity",
        "DeviceMixin",
        "recommended",
        "两者默认抛 NotImplementedError：/start_profile 与 SGLANG_TORCH_PROFILER_DIR 的 trace 采集会失败，"
        "场景 2 的 trace 分析依赖它",
        "返回厂商设备对应的 torch.profiler.ProfilerActivity 及其字符串名；trace 里的 kernel 名用 "
        "sgl-agent trace --dialect 映射",
        hook_names=("get_torch_profiler_activity_str", "get_torch_profiler_activity"),
        references=(source("oot_profiler_activity"),),
    )

    add(
        "is_pin_memory_available",
        "DeviceMixin",
        "recommended",
        "默认返回 False：每步构造 batch 的 H2D 拷贝（schedule_batch / forward_batch_info）用 pageable "
        "内存，拷贝同步、不能与计算重叠",
        "设备支持 pinned host 内存时返回 True",
        hook_names=("is_pin_memory_available",),
        references=(source("pinned_batch_copies"), source("pinned_h2d")),
    )

    key = profile.dispatch_key or DEFAULT_OOT_DISPATCH_KEY
    add(
        "get_dispatch_key_name",
        "SRTPlatform",
        "required" if profile.dispatch_key_collides else "recommended",
        "BaseFusedOp 在 out-of-tree 平台依次找 register_oot_forward 注册表（精确类匹配）→ "
        f"forward_<key> → forward_native；当前 key={key!r}",
        f"返回厂商唯一的 key（例如 {device_type!r}），所有 BaseFusedOp.register_oot_forward("
        "op_cls, fn, key) 都用这个 key；不要返回 cuda/hip/npu 等内置 key",
        hook_names=("get_dispatch_key_name",),
        references=(source("fused_op_dispatch"),),
    )

    add(
        "init_backend",
        "SRTPlatform",
        "recommended",
        "model_runner 模块导入时在每个 worker 里调用一次，适合集中注册设备实现",
        "在这里调用 register_attention_backend / BaseFusedOp.register_oot_forward / "
        "register_moe_runner_core / register_sampler_backend；需要早于 ServerArgs 的 add_*_choices "
        f"放到通用插件（{GENERAL_PLUGIN_GROUP}，launch_server 在解析参数前加载，scheduler 等子进程"
        "也会重新加载）",
        hook_names=("init_backend",),
        references=(source("oot_init_backend"), source("general_plugin_load_order")),
    )

    add(
        "post_load_model",
        "SRTPlatform",
        "optional",
        "权重加载完成后在 worker 里调用一次",
        "把权重转换成厂商 GEMM 需要的布局（INT8/FP8 重排、预打包），避免每步转换",
        hook_names=("post_load_model",),
    )

    quant = arch.quant
    if quant is not None:
        add(
            f"get_quantization_config({quant.method!r})",
            "SRTPlatform",
            "required",
            f"checkpoint 是 {quant.method} 量化；SGLang 内置的量化方法依赖 CUDA/Triton kernel"
            "（量化 MoE 走 Triton runner，W8A8 INT8 的 int8_scaled_mm 只在 CUDA 导入）",
            f"get_quantization_config({quant.method!r}) 返回厂商 QuantizationConfig 子类，由它给 "
            "Linear/FusedMoE 返回厂商的 quant method。该 hook 只对 SGLang 已有的量化名生效："
            "新名字会在调用 hook 之前就被拒绝。精度路线见 precision 部分。",
            hook_names=("get_quantization_config",),
            references=(
                source("oot_quantization_hook"),
                source("moe_triton_fused_func"),
                source("w8a8_int8_cuda_only"),
            ),
        )
    else:
        add(
            "get_quantization_config",
            "SRTPlatform",
            "optional",
            "checkpoint 未量化；只有在线量化（--quantization fp8 / w8a8_int8 等）时才会调用",
            "返回对应量化名的厂商 QuantizationConfig 子类",
            hook_names=("get_quantization_config",),
            references=(source("oot_quantization_hook"),),
        )

    kind = "dsa" if attention.sparse_indexer else ("mla" if attention.is_mla else "mha")
    add(
        f"get_kv_pool_cls(kind={kind!r}) / get_paged_allocator_cls",
        "SRTPlatform",
        "optional",
        "内置 KV 池在 device 上用 torch 分配，对任何 torch 设备都正确；分页分配器在 "
        "supports_triton=False 时自动走 torch 实现",
        "厂商注意力 kernel 需要特殊的 KV 布局/对齐时，返回 MHATokenToKVPool / MLATokenToKVPool / "
        "DSATokenToKVPool 的子类并重写 _create_buffers",
        hook_names=("get_kv_pool_cls", "get_paged_allocator_cls"),
        references=(source("kv_pool_hook"), source("paged_allocator_triton")),
    )

    sync_entry = hooks.get("synchronize") or {}
    empty_entry = hooks.get("empty_cache") or {}
    add(
        "synchronize / empty_cache",
        "DeviceMixin",
        "recommended",
        f"默认空操作；SGLang 有 {len(sync_entry.get('call_sites') or [])} 处 synchronize、"
        f"{len(empty_entry.get('call_sites') or [])} 处 empty_cache 经平台接口调用（计时、显存测量、"
        "图捕获前后）",
        f"映射到 torch.{device_type}.synchronize() / empty_cache()（或厂商等价 API）",
        hook_names=("synchronize", "empty_cache"),
    )

    add(
        "apply_server_args_defaults",
        "SRTPlatform",
        "recommended",
        "参数解析阶段按平台改默认值，来源记为 platform:<device_name>",
        "把注意力后端、page_size（对齐厂商 attention kernel）、sampling backend（无 FlashInfer 时默认 "
        "pytorch，可用 register_sampler_backend 注册厂商实现）等默认值放在这里，用户仍可显式覆盖",
        hook_names=("apply_server_args_defaults",),
        references=(source("sampler_backend"),),
    )

    if arch.num_mtp_layers > 0:
        add(
            "get_speculative_cache_locs_fn",
            "SRTPlatform",
            "conditional",
            f"模型带 {arch.num_mtp_layers} 层 MTP，可用 EAGLE/NEXTN 投机解码；不实现时用 Triton kernel "
            f"计算 KV 位置（YAML supports_triton={software.supports_triton}）",
            "返回 assign_extend_cache_locs 的设备实现",
            hook_names=("get_speculative_cache_locs_fn",),
            references=(source("speculative_cache_locs"), source("speculative_nextn_alias")),
        )
    add(
        "supports_speculative_algorithm / *_speculative_draft_attention_backend",
        "SRTPlatform",
        "optional",
        "只有 DFLASH 投机解码会查询这些 hook",
        "支持 DFLASH 时返回 True 并给出 draft 注意力后端",
        hook_names=(
            "supports_speculative_algorithm",
            "supports_speculative_draft_attention_backend",
            "get_default_speculative_draft_attention_backend",
        ),
        references=(source("dflash_platform_gate"),),
    )

    uuid_entry = hooks.get("get_device_uuid") or {}
    add(
        "get_device_uuid / get_device_capability / set_device",
        "DeviceMixin",
        "conditional",
        f"get_device_uuid 有 {len(uuid_entry.get('call_sites') or [])} 处调用（weight cache、"
        "checkpoint engine、Engine 的 weight-cache 守护进程文件），capability/set_device 用于 weight cache",
        "使用 --weight-cache-mode 或 RL 在线权重更新时实现",
        hook_names=("get_device_uuid", "get_device_capability", "set_device"),
    )

    triton_capability = capabilities.get("supports_triton", {})
    add(
        "capabilities.supports_triton",
        "PlatformCapabilities",
        "optional",
        "控制 MLA KV buffer 写入、分页分配器和 support_triton() 是否走 Triton；为 False 时自动用 torch "
        "实现，但量化 MoE、W8A8 INT8 激活量化等仍直接调用 Triton（见算子清单）",
        "设备有可用的 Triton 后端时设为 True",
        default=f"默认 {triton_capability.get('default', False)!r}",
        call_sites=tuple(triton_capability.get("readers") or ()),
        references=(source("paged_allocator_triton"),),
    )

    hicache_capability = capabilities.get("hicache_device_kernels", {})
    add(
        "capabilities.hicache_device_kernels",
        "PlatformCapabilities",
        "conditional",
        "开启 HiCache 时：传输 kernel 只在 CUDA/HIP(/XPU) 与 NPU 上导入，out-of-tree 平台没有 seam；"
        "该能力只控制写回 staging",
        "用通用插件的 HookRegistry 替换 host pool 的传输函数，或推动上游增加平台 seam（见特性建议）",
        default=f"默认 {hicache_capability.get('default', False)!r}",
        call_sites=tuple(hicache_capability.get("readers") or ()),
        references=(source("hicache_transfer_kernels"), source("general_plugin_hooks")),
    )

    for name, entry in sorted(hooks.items()):
        if name in covered or not entry.get("call_sites"):
            continue
        add(
            name,
            str(entry.get("defined_in") or "SRTPlatform"),
            "optional",
            f"未在规则里覆盖的 hook（{entry.get('summary') or '无说明'}）",
            "按调用点确认是否需要实现",
            default=_default_text(entry),
            call_sites=tuple(entry.get("call_sites") or ()),
        )
    checklist.requirements.sort(key=lambda item: _LEVEL_RANK[item.level])
    return checklist
