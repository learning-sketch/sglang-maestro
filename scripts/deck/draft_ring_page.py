"""Draft of the repository's runtime view: one iteration drawn as a ring of components.

This page is the mechanism, not one run: it says what each component does, what it hands to the
next one and where a candidate gets stopped, without naming a model, a dataset or a measurement
from any particular session. Those numbers live on the record and gate pages instead.

Each node is a group of modules from src/m100_model_agent/; each arrow is the hand-off to the
next node, in the order of the iteration sequence. The ring is a racetrack rather than a circle
because a 16:9 page gives the straight runs the width the hand-off labels need. The controller
sits inside the loop because it schedules and judges every node. The one node where code
actually changes is drawn tall, with three lanes, because three parties meet there: the
controller frames the request, the planner is an external session, and a person steps in when a
case is too hard to hand off. The strips above and below hold what sits outside the loop: the
frozen contract, the one-off entry and the system under test. Every one of the 24 modules
appears exactly once somewhere on the page.

    python draft_ring_page.py [out.pptx]
"""

import sys

from build_agent_deck import (
    DASH,
    FONT,
    LOOPS,
    MIDDLE,
    MONO,
    MX,
    RIGHT,
    SW,
    SH,
    WHITE,
    P,
    add_arrow,
    add_box,
    add_text,
    runs,
)
from build_model_layer_page import THEMES
from check_text_fit import SAFETY, char_em
from pptx import Presentation
from pptx.util import Inches

T = THEMES["review"]
TITLE_BAR, TITLE_INK = "306FFF", "000000"
OUTPUT = "../../slides/ring_page_draft.pptx"
TITLE = "M100 Model Agent：一轮迭代的运行视图"
SUBTITLE = "不绑定具体模型：每个组件做什么、把什么交给下一个、在哪一步会被拦下"
PROVENANCE = "全链路 24 个核心模块 · 本页只讲机制，单次运行的数据见记录页"

EVIDENCE, PLANNER, CONTROLLER = LOOPS["B"]["main"], T.accent, LOOPS["A"]["main"]
HUMAN = T.g700
LEGEND = ((EVIDENCE, "证据层"), (PLANNER, "规划器层 · 外部会话"), (CONTROLLER, "控制器层"),
          (HUMAN, "人 · 按需介入"))

# The racetrack: two straight runs of nodes joined down the right and back up the left.
CONTRACT_Y, CONTRACT_H = 0.95, 0.52
TOP_Y, ROW_H = 1.58, 0.98
BOT_Y = 5.56
FOOT_Y, FOOT_H = 6.70, 0.70
NODE_W = 2.6
RIGHT_X = 9.6
RIGHT_W = SW - MX - RIGHT_X
TALL_H = 2.88
TOP_X = (MX, 3.65, 6.7)
BOT_X = (MX, 3.5, 6.55)
HUB = (3.6, 3.22, 5.45, 1.78)

# (rect, layer, title, modules, what it does) -- clockwise, in run order; the tall node is drawn
# separately because it holds three lanes instead of one body.
NODES = (
    ((TOP_X[0], TOP_Y, NODE_W, ROW_H), EVIDENCE, "采 trace",
     ("profile_session", "roofline_instrumentation"), (
         "另起一次性服务，关 CG 与 overlap",
         "start / stop_profile 圈定固定诊断请求",
         "诊断数据不计入收益口径",
     )),
    ((TOP_X[1], TOP_Y, NODE_W, ROW_H), EVIDENCE, "归因",
     ("roofline_trace · trace_diagnosis",), (
         "截 decode 窗口，按源码 scope 归因",
         "kernel 名只在缺 scope 时回退",
         "区间并集与总和互校，防重叠漏记",
     )),
    ((TOP_X[2], TOP_Y, NODE_W, ROW_H), EVIDENCE, "账本",
     ("roofline · hardware-ceilings/",), (
         "config → FLOPs / 字节 → 下界与 ratio",
         "专家层字节取路由实测 distinct experts",
         "快于下界者判自检证伪，按授权隔离",
     )),
    ((RIGHT_X, TOP_Y, RIGHT_W, TALL_H), PLANNER, "方案与补丁", (), ()),
    ((RIGHT_X, BOT_Y, RIGHT_W, ROW_H), CONTROLLER, "落地 · 构建",
     ("git_workspace · server · development_workspace", "optimization_adapters"), (
         "在源码工作区原位构建，打隔离补丁",
         "只增量编译已登记的 standalone 组件",
     )),
    ((BOT_X[2], BOT_Y, NODE_W, ROW_H), CONTROLLER, "算子级微测初筛",
     ("device 侧 · 隔离算子链",), (
         "对照与候选对称预热，µs 级微测",
         "提速达到筛选阈值才记 device_screened",
     )),
    ((BOT_X[1], BOT_Y, NODE_W, ROW_H), CONTROLLER, "数值 + 质量门", ("evaluation",), (
        "固定文本探针比对数概率，偏差须在容差内",
        "评测得分不降，解析失败与截断都计入",
    )),
    ((BOT_X[0], BOT_Y, NODE_W, ROW_H), CONTROLLER, "端到端三方复验",
     ("optimization_staging · benchmark", "selfiterate_gates"), (
         "原始 / 上一版 / 候选共用加载库 SHA",
         "重复取中位，关 profiler，须过噪声门",
     )),
)
TALL_INDEX = 3

# The three parties inside the tall node: (colour, role, modules, lines)
LANES = (
    (CONTROLLER, "控制器侧 · 封装请求",
     ("optimization_planner · planner_memory", "planner_mailbox"), (
         "UUID + 请求 SHA + 动作 schema",
         "上下文 ≤ 18 KB，单条消息 ≤ 28 KB",
         "ID / SHA 不匹配即拒收，往返全部落盘",
     )),
    (PLANNER, "规划器 · 外部会话",
     ("/root/optimization_planner", "planner-skills/ · planner_diagnostics"), (
         "读源码与冻结证据，提假设、写补丁",
         "默认只附 INDEX，一轮最多再取 2 个 SKILL",
         "证据不足可发起有界诊断，输出 ≤ 64 KiB",
         "不能碰设备、起服务、宣布成功",
     )),
    (HUMAN, "人 · 按需介入", (), (
        "复杂场景由人直接写补丁，或与规划器协同实现",
        "介入不改变判决路径，同样走下面三道门",
    )),
)

# (x, y, w, h, align, lines) for each hand-off; a line is a tuple of (text, mono) parts
HANDOFFS = (
    (2.45, 2.64, 1.85, 0.52, None, ((("Chrome trace", False),), (("+ 路由探针", False),))),
    (5.45, 2.64, 1.85, 0.52, None,
     ((("每类 kernel 时长、调用数", False),), (("区间并集与总和", False),))),
    (7.35, 2.64, 1.85, 0.52, RIGHT,
     ((("ranked_targets", True),), (("+ 隔离项与原因", False),))),
    (8.75, 4.54, 2.3, 0.48, RIGHT,
     ((("提案：假设 + 补丁 + 证据引用", False),), (("证据齐备才落地", False),))),
    (8.05, 5.08, 1.95, 0.44, RIGHT, ((("构建身份一致才比较，否则", False),),
                                     (("optimization_retry", True), (" 重测", False)))),
    (5.0, 5.08, 1.95, 0.44, RIGHT, ((("device_screened", True),),)),
    (1.95, 5.08, 1.95, 0.44, RIGHT, ((("进 bundle，三方同库构建", False),),)),
    (2.05, 3.52, 1.5, 0.62, None,
     ((("verified_gain", True),), (("→ best 晋升", False),), (("→ 再采 trace", False),))),
)

CONTROLLER_BODY = (
    "Python，确定性：设备锁、状态机、失败恢复、验证、报告",
    "规划器不能自行宣布实验成功，数值、图文质量、性能收益全部由控制器独立检查",
    "每次尝试按 trial 落盘：候选、构建准备、基线、基线复核、最终复验",
    "拒绝原因经 planner_memory 带回下一轮请求；基础设施中断不计入方案失败",
)
CONTRACT = (
    ("optimization-task.json", "模型 / 精度 / 负载 / 门限 / 预算"),
    ("generation-task.json", "数值探针 + 质量契约"),
    ("backup/", "manifest 组件 SHA + 冻结基准 + planner-skills"),
)
ENTRY = (
    ("memory_preflight", "显存预检"),
    ("server · evaluation · benchmark", "起服务、测基线"),
    ("diagnostic_profiles", "生成诊断映射"),
    ("optimization_source", "导入已验证结果"),
)
SUT = (
    ("SGLang 原仓库", "原 Git 版本，未提交修改保留"),
    ("xInfer / xtorch_ops", "只编登记的 standalone 组件"),
    ("M100 单卡", "目标模型与精度由任务契约锁定"),
)
SUT_NOTE = "算子层的跨层调用尚未接入：融合算子由规划器直接写成内联补丁，不经 xpugen"


def text_width(text, size, mono=False):
    em = 0.6 if mono else 1.0
    return sum(char_em(ch, False) * (em if ch.isascii() else 1) for ch in text) * size / 72


def header(slide):
    add_box(slide, MX, 0.26, 0.07, 0.38, fill=TITLE_BAR)
    add_text(slide, MX + 0.22, 0.15, 7.6, 0.6, [P(TITLE, 24, TITLE_INK, True)], anchor=MIDDLE)
    add_text(slide, MX + 0.22, 0.62, 8.4, 0.26, [P(SUBTITLE, 10, T.g500)], anchor=MIDDLE)
    x = SW - MX
    for color, label in reversed(LEGEND):
        label_w = text_width(label, 9) * 1.15
        x -= 0.18 + label_w
        add_box(slide, x, 0.30, 0.12, 0.12, fill=color)
        add_text(slide, x + 0.18, 0.22, label_w, 0.28, [P(label, 9, T.g500)], anchor=MIDDLE)
        x -= 0.22
    add_text(slide, SW - MX - 5.5, 0.52, 5.5, 0.2, [P(PROVENANCE, 7.5, T.g400, align=RIGHT)],
             anchor=MIDDLE)


def arrows(slide):
    top_y, bot_y = TOP_Y + ROW_H / 2, BOT_Y + ROW_H / 2
    left_x, right_x = TOP_X[0] + NODE_W / 2, RIGHT_X + RIGHT_W / 2
    for index in range(len(TOP_X) - 1):
        add_arrow(slide, TOP_X[index] + NODE_W, top_y, TOP_X[index + 1], top_y, T.g500, 1.5)
    add_arrow(slide, TOP_X[-1] + NODE_W, top_y, RIGHT_X, top_y, T.g500, 1.5)
    add_arrow(slide, right_x, TOP_Y + TALL_H, right_x, BOT_Y, T.g500, 1.5)
    add_arrow(slide, RIGHT_X, bot_y, BOT_X[-1] + NODE_W, bot_y, T.g500, 1.5)
    for index in range(len(BOT_X) - 1, 0, -1):
        add_arrow(slide, BOT_X[index], bot_y, BOT_X[index - 1] + NODE_W, bot_y, T.g500, 1.5)
    add_arrow(slide, left_x, BOT_Y, left_x, TOP_Y + ROW_H, T.g500, 1.5)


def node(slide, number, index):
    (x, y, w, h), color, title, modules, does = NODES[index]
    add_box(slide, x, y, w, h, fill=WHITE, line=T.g300)
    add_box(slide, x, y, 0.06, h, fill=color)
    paragraphs = [runs([(f"{number} ", 10.5, color, True, "Arial"), (title, 10.5, T.g900, True)],
                       after=1)]
    paragraphs += [P(module, 7.5, T.g500, False, MONO) for module in modules]
    paragraphs[-1]["after"] = 2
    paragraphs += [P(line, 8.5, T.g700) for line in does]
    add_text(slide, x + 0.16, y + 0.04, w - 0.26, h - 0.08, paragraphs, anchor=MIDDLE)


def tall_node(slide, number):
    """The step where the patch comes from: one box, three parties, stacked as lanes."""
    (x, y, w, h), color, title, _, _ = NODES[TALL_INDEX]
    add_box(slide, x, y, w, h, fill=WHITE, line=color, line_w=1.0)
    add_box(slide, x, y, 0.06, h, fill=color)
    head = 0.3
    add_text(slide, x + 0.16, y + 0.03, w - 0.26, head, [runs([
        (f"{number} ", 10.5, color, True, "Arial"), (title, 10.5, T.g900, True),
        ("　全环唯一产生改动的一步", 8.5, T.g500),
    ])], anchor=MIDDLE)

    weights = [1.0 + len(modules) * 0.85 + len(lines) * 0.95 for _, _, modules, lines in LANES]
    lane_y = y + head
    for (tone, role, modules, lines), weight in zip(LANES, weights):
        lane_h = (h - head - 0.06 - 0.06 * len(LANES)) * weight / sum(weights)
        add_box(slide, x + 0.14, lane_y, w - 0.28, lane_h, fill=T.panel)
        add_box(slide, x + 0.14, lane_y, 0.04, lane_h, fill=tone)
        paragraphs = [P(role, 9, tone, True, after=1)]
        paragraphs += [P(module, 7.5, T.g500, False, MONO) for module in modules]
        if modules:
            paragraphs[-1]["after"] = 2
        paragraphs += [P(line, 8.5, T.drop if line.startswith("不能") else T.g700)
                       for line in lines]
        add_text(slide, x + 0.30, lane_y + 0.02, w - 0.48, lane_h - 0.04, paragraphs,
                 anchor=MIDDLE)
        lane_y += lane_h + 0.06


def handoff(slide, spec):
    x, y, w, h, align, lines = spec
    options = {"align": align} if align else {}
    add_text(slide, x, y, w, h, [
        runs([(text, 8, T.g500, False, MONO if mono else FONT) for text, mono in parts], **options)
        for parts in lines
    ], anchor=MIDDLE)


def hub(slide):
    x, y, w, h = HUB
    add_box(slide, x, y, w, h, fill=LOOPS["A"]["tint"])
    add_box(slide, x, y, w, 0.05, fill=CONTROLLER)
    role, constraint, trials, feedback = CONTROLLER_BODY
    add_text(slide, x + 0.2, y + 0.1, w - 0.4, h - 0.18, [
        P("控制器", 12, CONTROLLER, True),
        P("optimization · selfiterate · optimization_export", 7.5, T.g500, False, MONO, after=2),
        P(role, 9, T.g700, after=4),
        runs([("硬约束　", 9, T.drop, True), (constraint, 9, T.g900)], after=4),
        P(trials, 8.5, T.g700),
        P(feedback, 8.5, T.g700),
    ])


def inline(rows, size=8):
    parts = []
    for index, (name, text) in enumerate(rows):
        if index:
            parts.append(("　｜　", size, T.g300))
        parts.append((name, size - 0.5, T.g700, False, MONO))
        parts.append((f"　{text}", size, T.g500))
    return runs(parts)


def strips(slide):
    """What the loop takes as given: the contract above it, the entry and the target below."""
    add_box(slide, MX, CONTRACT_Y, SW - 2 * MX, CONTRACT_H, fill=WHITE, line=T.g300, dash=DASH)
    add_text(slide, MX + 0.16, CONTRACT_Y + 0.02, SW - 2 * MX - 0.32, CONTRACT_H - 0.04, [
        runs([("任务契约 · 冻结并哈希校验", 9.5, T.g700, True)], after=1),
        inline(CONTRACT),
    ], anchor=MIDDLE)

    entry_w = 5.6
    add_box(slide, MX, FOOT_Y, entry_w, FOOT_H, fill=T.panel)
    add_text(slide, MX + 0.16, FOOT_Y + 0.04, entry_w - 0.32, FOOT_H - 0.08, [
        runs([("0  入口", 9.5, CONTROLLER, True), ("　进环之前只跑一次", 8.5, T.g500)], after=1),
        inline(ENTRY[:2]),
        inline(ENTRY[2:]),
    ], anchor=MIDDLE)

    sut_x = MX + entry_w + 0.2
    sut_w = SW - MX - sut_x
    add_box(slide, sut_x, FOOT_Y, sut_w, FOOT_H, fill=T.panel)
    add_text(slide, sut_x + 0.16, FOOT_Y + 0.04, sut_w - 0.32, FOOT_H - 0.08, [
        runs([("被测系统", 9.5, T.g700, True)], after=1),
        inline(SUT),
        P(SUT_NOTE, 7.5, T.g500),
    ], anchor=MIDDLE)


def ring_page(slide):
    header(slide)
    strips(slide)
    arrows(slide)
    for number, index in enumerate(range(len(NODES)), start=1):
        if index == TALL_INDEX:
            tall_node(slide, number)
        else:
            node(slide, number, index)
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
