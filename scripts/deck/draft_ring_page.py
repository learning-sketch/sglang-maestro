"""Draft of the repository's runtime view: one iteration drawn as a ring of components.

Each node is a group of modules from src/m100_model_agent/ with what it does and what it did this
run; each arc is the hand-off to the next node, in the order of the iteration sequence. The
controller sits in the middle because it schedules and judges every node; the planner is the one
external node on the ring. The corners hold what sits outside the loop: the frozen contract, the
planner's limits, the one-off baseline entry, and the system under test. Every one of the 24
modules appears exactly once somewhere on the page.

    python draft_ring_page.py [out.pptx]
"""

import math
import sys

from build_agent_deck import (
    DASH,
    FONT,
    LOOPS,
    MIDDLE,
    MONO,
    MX,
    RIGHT,
    SH,
    SW,
    WHITE,
    P,
    add_arrow,
    add_arrowheads,
    add_box,
    add_text,
    drop_theme_style,
    ellipse_point,
    rgb,
    runs,
)
from build_model_layer_page import THEMES
from check_text_fit import SAFETY, char_em
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

T = THEMES["review"]
TITLE_BAR, TITLE_INK = "306FFF", "000000"
OUTPUT = "../../slides/ring_page_draft.pptx"
TITLE = "M100 Model Agent：一轮迭代的运行视图"

EVIDENCE, PLANNER, CONTROLLER = LOOPS["B"]["main"], T.accent, LOOPS["A"]["main"]
LEGEND = ((EVIDENCE, "证据层"), (PLANNER, "规划器层 · 外部托管"), (CONTROLLER, "控制器层"))

CX, CY, RX, RY = 6.667, 4.15, 4.8, 2.4
NODE_W, NODE_H = 2.5, 1.04

# (angle, layer, title, modules, does, this run, external) -- clockwise from 9 o'clock in run order
NODES = (
    (180, EVIDENCE, "采 trace", ("profile_session", "roofline_instrumentation"), (
        "另起一次性服务，关 CG 与 overlap",
        "start / stop_profile 夹固定诊断请求",
    ), "profile 0 → 8，每轮 D 后重采", False),
    (221, EVIDENCE, "归因", ("roofline_trace · trace_diagnosis",), (
        "截 decode 窗口，按源码 scope 归因",
        "kernel 名只在缺 scope 时回退",
    ), "总和 23.921 ≈ 并集 23.920 ms", False),
    (262, EVIDENCE, "账本", ("roofline · hardware-ceilings/",), (
        "config → FLOPs/字节 → 下界 → ratio",
        "专家字节用路由实测 distinct experts",
    ), "两类 GEMM 自检失败，按授权隔离", False),
    (298, PLANNER, "规划请求", ("optimization_planner", "planner_memory · planner_mailbox"), (
        "UUID + 请求 SHA + 动作 schema",
        "上下文 ≤ 18 KB，整条消息 ≤ 28 KB",
    ), "18 轮往来，全部落盘", False),
    (343, PLANNER, "Codex subagent", ("/root/optimization_planner",), (
        "读源码与冻结证据，提假设、写补丁",
        "read/search/list/probe/patch/stop",
        "不能碰设备、起服务、宣布成功",
    ), "6 轮读了 roofline-fusion-advisor", True),
    (17, CONTROLLER, "落地", ("git_workspace · server", "development_workspace",
                            "optimization_adapters"), (
        "原位建版本、打隔离补丁、只编登记组件",
    ), "5 / 16 卡在构建身份 → retry", False),
    (62, CONTROLLER, "C 筛选", ("device 侧 · 隔离算子链",), (
        "对照与候选对称预热，µs 级微测",
        "≥ 2% 才记 device_screened",
    ), "6 个通过，+1.61% / +0.77% 被拒", False),
    (98, CONTROLLER, "数值 + 质量门", ("evaluation",), (
        "3 个文本探针 |Δlogprob| ≤ 0.03",
        "MMStar 冻结 60 题，解析、截断都计入",
    ), "交付版 Δ 0.0 · MMStar 45/60", False),
    (139, CONTROLLER, "D 三方 A/B", ("optimization_staging · benchmark", "selfiterate_gates"), (
        "原始 / 上一版 / 候选共用加载库 SHA",
        "各 3 次取中位，关 profiler，过噪声门",
    ), "3 轮都过；034 c32 +45.61%", False),
)

# hand-off from node i to node i + 1: (x, y, w, h, align, lines, both ways); a line is a tuple of
# (text, mono) parts
HANDOFFS = (
    (2.55, 3.14, 1.85, 0.44, None, ((("Chrome trace", False),), (("+ 路由探针", False),)), False),
    (4.36, 2.34, 1.9, 0.36, None,
     ((("每类 kernel 时长、调用数", False),), (("区间并集与总和", False),)), False),
    (6.3, 2.34, 1.32, 0.36, RIGHT, ((("ranked_targets", True),), (("+ 隔离项与原因", False),)), False),
    (8.2, 2.6, 1.75, 0.36, RIGHT, ((("请求 ⇄ 回复", False),), (("ID / SHA 不匹配即拒收", False),)), True),
    (8.95, 3.99, 2.3, 0.32, RIGHT,
     ((("提案：假设 + 补丁 + 证据引用", False),), (("证据齐备才落地", False),)), False),
    (8.0, 5.35, 1.95, 0.36, RIGHT, ((("构建身份一致才进入比较", False),),
                                     (("否则 ", False), ("optimization_retry", True), (" 重测", False))),
     False),
    (6.3, 5.6, 1.32, 0.36, RIGHT, ((("device_screened", True),),), False),
    (4.36, 5.6, 1.9, 0.36, None, ((("进 bundle", False),), (("三方同库构建，本轮 8 次", False),)), False),
    (2.55, 4.72, 1.85, 0.44, None,
     ((("verified_gain", True),), (("→ best 晋升 → 再采 trace", False),)), False),
)
PROVENANCE = "24 个模块全部在图上 · codex/m100-selfiterate-30b-20261019 @ 04f3965 · 快照 2026-10-20 01:11 UTC"

HUB = (
    "Python，确定性：设备锁、状态机、恢复、验证、报告",
    "规划器不能自行宣布实验成功，数值、图文质量、性能收益全部由控制器独立检查",
    ("35 条 trial：candidate 16（含 6 次 retry）· 构建准备 8",
     "baseline 1 · baseline-recheck 7 · verification 3",
     "候选尝试累计 25 次：新主线 16 + 旧主线 9"),
    "拒绝原因经 planner_memory 带回下一轮请求；基础设施中断不记成优化无效",
)

CONTRACT = (
    ("optimization-task.json", "模型 / 精度 / 负载 / 门限 / 预算"),
    ("generation-task.json", "数值探针 + 质量合同"),
    ("backup/", "manifest 组件 SHA + 冻结基准 + planner-skills"),
)
PLANNER_LIMITS = (
    ("planner-skills/", ("INDEX.md + 10 个 SKILL.md", "默认只附 INDEX，一轮最多读 2 个")),
    ("probe · planner_diagnostics", ("证据不足时的有界诊断，≤ 64 KiB", "本轮 5 个 probe，都改变了决策")),
)
HOSTING = ("依赖当前 Codex 会话托管，", "还不能无人值守长跑")
ENTRY = (
    ("memory_preflight", "显存预检；控制器另查卡空闲、备份、SHA"),
    ("server · evaluation · benchmark", "冻结配置起服务、测基线"),
    ("diagnostic_profiles", "任何候选执行前生成诊断映射"),
    ("optimization_source", "导入已完整验证的结果"),
)
# (name, detail, detail is an identifier)
SUT = (
    ("SGLang 原仓库", "原 Git 版本，未提交修改保留", False),
    ("xInfer / xtorch_ops", "只编登记的 standalone 组件", False),
    ("M100 单卡", "Qwen3-VL-30B-A3B · channel-fp8", True),
)
SUT_NOTE = ("native kernel 由 subagent 写，", "不是 xpugen 生成")


def text_width(text, size):
    return sum(char_em(ch, False) for ch in text) * size / 72


def header(slide):
    add_box(slide, MX, 0.44, 0.07, 0.38, fill=TITLE_BAR)
    add_text(slide, MX + 0.22, 0.33, 7.6, 0.6, [P(TITLE, 24, TITLE_INK, True)], anchor=MIDDLE)
    x = SW - MX
    for color, label in reversed(LEGEND):
        label_w = text_width(label, 9) / SAFETY + 0.04
        x -= 0.18 + label_w
        add_box(slide, x, 0.58, 0.12, 0.12, fill=color)
        add_text(slide, x + 0.18, 0.5, label_w, 0.28, [P(label, 9, T.g500)], anchor=MIDDLE)
        x -= 0.24
    add_text(slide, SW - MX - 5.5, 0.78, 5.5, 0.2, [P(PROVENANCE, 7.5, T.g400, align=RIGHT)],
             anchor=MIDDLE)


def node_corner(angle):
    x, y = ellipse_point(CX, CY, RX, RY, angle)
    return x - NODE_W / 2, y - NODE_H / 2


def track(slide):
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(CX - RX), Inches(CY - RY),
                                   Inches(2 * RX), Inches(2 * RY))
    drop_theme_style(shape)
    shape.fill.background()
    shape.line.color.rgb = rgb(T.g100)
    shape.line.width = Pt(10)


def arc(slide, start, end, both):
    """Clockwise arc between two parametric angles, arrowed at the far end (and the near one)."""

    def visual(degrees):
        angle = math.radians(degrees)
        return math.degrees(math.atan2(RY * math.sin(angle), RX * math.cos(angle))) % 360

    shape = slide.shapes.add_shape(MSO_SHAPE.ARC, Inches(CX - RX), Inches(CY - RY),
                                   Inches(2 * RX), Inches(2 * RY))
    drop_theme_style(shape)
    # Preset arc angles are measured from the centre, not parametric, so convert them.
    shape.adjustments[0] = visual(start) * 0.6
    shape.adjustments[1] = visual(end) * 0.6
    shape.fill.background()
    shape.line.color.rgb = rgb(T.g500)
    shape.line.width = Pt(1.5)
    add_arrowheads(shape.line, head=both, tail=True)


def ring_arcs(slide):
    corners = [node_corner(spec[0]) for spec in NODES]

    def covered(angle, corner):
        px, py = ellipse_point(CX, CY, RX, RY, angle)
        x, y = corner
        return x - 0.05 <= px <= x + NODE_W + 0.05 and y - 0.05 <= py <= y + NODE_H + 0.05

    for index, spec in enumerate(NODES):
        start, end = spec[0], NODES[(index + 1) % len(NODES)][0]
        span = (end - start) % 360
        pair = (corners[index], corners[(index + 1) % len(NODES)])
        free = [start + span * step / 400 for step in range(401)
                if not any(covered(start + span * step / 400, corner) for corner in pair)]
        if free:
            arc(slide, free[0], free[-1], HANDOFFS[index][6])


def node(slide, number, spec):
    angle, color, title, modules, does, this_run, external = spec
    x, y = node_corner(angle)
    add_box(slide, x, y, NODE_W, NODE_H, fill=WHITE, line=color if external else T.g300,
            line_w=1.0 if external else 0.75, dash=DASH if external else None)
    add_box(slide, x, y, 0.06, NODE_H, fill=color)
    paragraphs = [runs([(f"{number} ", 10.5, color, True, "Arial"), (title, 10.5, T.g900, True)],
                       after=1)]
    paragraphs += [P(module, 7.5, T.g500, False, MONO) for module in modules]
    paragraphs[-1]["after"] = 2
    paragraphs += [P(line, 8.5, T.drop if line.startswith("不能") else T.g700) for line in does]
    paragraphs.append(runs([("本轮 ", 8.5, color, True), (this_run, 8.5, T.g900)]))
    add_text(slide, x + 0.16, y + 0.04, NODE_W - 0.26, NODE_H - 0.08, paragraphs, anchor=MIDDLE)


def handoff(slide, spec):
    x, y, w, h, align, lines, _ = spec
    options = {"align": align} if align else {}
    add_text(slide, x, y, w, h, [
        runs([(text, 8, T.g500, False, MONO if mono else FONT) for text, mono in parts], **options)
        for parts in lines
    ], anchor=MIDDLE)


def hub(slide):
    x, y, w, h = 4.45, 3.05, 4.45, 2.23
    add_box(slide, x, y, w, h, fill=LOOPS["A"]["tint"])
    add_box(slide, x, y, w, 0.05, fill=CONTROLLER)
    role, constraint, this_run, feedback = HUB
    add_text(slide, x + 0.2, y + 0.12, w - 0.4, h - 0.2, [
        P("控制器", 12, CONTROLLER, True),
        P("optimization · selfiterate · optimization_export", 7.5, T.g500, False, MONO, after=2),
        P(role, 9, T.g700, after=4),
        runs([("硬约束　", 9, T.drop, True), (constraint, 9, T.g900)], after=4),
        runs([("本轮 ", 8.5, CONTROLLER, True), (this_run[0], 8.5, T.g700)]),
        *[P(line, 8.5, T.g700) for line in this_run[1:]],
        P(feedback, 9, T.g700, before=4),
    ])


def corner_box(slide, x, y, w, h, title, color, dashed=False):
    add_box(slide, x, y, w, h, fill=WHITE if dashed else T.panel,
            line=T.g300 if dashed else None, dash=DASH if dashed else None)
    add_text(slide, x + 0.14, y + 0.06, w - 0.28, 0.22, [P(title, 9.5, color, True)],
             anchor=MIDDLE)


def name_rows(rows):
    return [runs([(name, 7.5, T.g700, False, MONO), (f"　{text}", 8, T.g500)]) for name, text in rows]


def corners(slide):
    x, y, w, h = MX, 1.0, 3.6, 0.9
    corner_box(slide, x, y, w, h, "合同 · 冻结并哈希校验", T.g700, dashed=True)
    add_text(slide, x + 0.14, y + 0.3, w - 0.28, h - 0.34, name_rows(CONTRACT))

    x, y, w, h = 10.28, 1.0, SW - MX - 10.28, 1.42
    corner_box(slide, x, y, w, h, "规划器的边界", PLANNER)
    add_text(slide, x + 0.14, y + 0.3, w - 0.28, h - 0.34, [
        *[paragraph for name, lines in PLANNER_LIMITS for paragraph in (
            P(name, 7.5, T.g700, False, MONO), *[P(line, 8, T.g500) for line in lines])],
        *[P(line, 8, T.drop) for line in HOSTING],
    ])

    x, y, w, h = MX, 6.4, 3.75, 0.92
    corner_box(slide, x, y, w, h, "0  入口 · A 阶段，只跑一次", CONTROLLER)
    add_text(slide, x + 0.14, y + 0.3, w - 0.28, h - 0.34, name_rows(ENTRY))
    entry_x = MX + 0.32
    add_arrow(slide, entry_x, y, entry_x, ellipse_point(CX, CY, RX, RY, 180)[1] + NODE_H / 2,
              CONTROLLER, 1.25)

    x, y, w, h = 10.28, 5.88, SW - MX - 10.28, 1.44
    corner_box(slide, x, y, w, h, "被测系统", T.g700)
    add_text(slide, x + 0.14, y + 0.3, w - 0.28, h - 0.34, [
        *[paragraph for name, text, ident in SUT for paragraph in (
            P(name, 8.5, T.g900, True),
            P(text, 7.5, T.g500, False, MONO) if ident else P(text, 8, T.g500))],
        *[P(line, 7.5, T.g400, before=3 if not index else 0) for index, line in enumerate(SUT_NOTE)],
    ])


def ring_page(slide):
    header(slide)
    corners(slide)
    track(slide)
    ring_arcs(slide)
    for number, spec in enumerate(NODES, start=1):
        node(slide, number, spec)
    for spec in HANDOFFS:
        handoff(slide, spec)
    hub(slide)


def build(path):
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    ring_page(prs.slides.add_slide(prs.slide_layouts[6]))
    prs.save(path)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else OUTPUT
    build(target)
    print(target)
