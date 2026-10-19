"""Build the M100 agent deck on the 2026 KUNLUNXIN 1024 程序员节 template.

Same nine pages of content as build_review_deck, re-themed onto the conference background:
a blue-to-mint vertical gradient crossed by a faint 1.12in grid, pixel confetti in the
top-right and bottom-left, and the 1024 程序员节 lockup in the top-right corner.

Three things about that background drive the layout here.

The gradient is light everywhere — luminance runs from 0.38 at the top to 0.93 at the
bottom — so white text never reaches 3:1 against it and every word on these pages is dark
instead. The palette is accordingly a set of deep blues over white cards, with the lockup's
own magenta and orange reserved for the few marks that have to stand out.

The lockup occupies the top right from 11.29in across and down to 1.16in, so the header
block is capped at HEAD_W rather than running the full column width.

Dense content sits on opaque cards rather than straight on the gradient: the grid and
confetti read as texture behind a card but as noise behind a paragraph.
"""

import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

from build_agent_deck import (
    CENTER,
    CW,
    DASH,
    FONT,
    MIDDLE,
    MONO,
    MX,
    RIGHT,
    SH,
    SW,
    TOP,
    WHITE,
    Y_FOOT,
    P,
    add_arrow,
    add_box,
    add_line,
    add_path_arrow,
    add_text,
    boxed_text,
    number_dot,
    runs,
    style_run,
)
from speaker_notes import clock, rounded, seconds

HERE = Path(__file__).resolve().parent
BACKGROUND = next(
    (str(p) for p in (HERE / "assets" / "tpl_bg.png", HERE / "tpl_bg.png") if p.exists()),
    str(HERE / "assets" / "tpl_bg.png"),
)

# Sampled from the template: the lockup's own accents are #FF9201 and #FF0893, its blue is
# #60A4FF. Text tones are deepened from that blue so they stay on-palette while clearing
# 4.5:1 against the lightest and darkest parts of the gradient alike.
G900 = "0A2B45"
G700 = "1B4D69"
G500 = "40708B"
G400 = "7FA4BA"
G300 = "AFCEE1"
G200 = "CFE3F0"
G100 = "DEEDFA"
# Neutral cards go pure white: against cyan that is what makes a card read as a card, and it
# leaves the tints free to carry meaning rather than just lifting text off the gradient.
G050 = "FFFFFF"

AGENT = "0F62CE"
TINT = "C9E1FC"
MODEL_LEVEL = "D4147A"
MODEL_TINT = "FBD9E9"
BRAND_ORANGE = "E98300"
SHARE_GEMM, SHARE_ATTN, SHARE_OTHER = "2E72C8", BRAND_ORANGE, "A9C8DC"

# The lockup's keep-out box, measured off the template.
LOGO_X, LOGO_Y = 11.29, 1.16
HEAD_W = LOGO_X - MX - 0.2
TOTAL = 10

# The three model-level gates. One vocabulary for the whole deck: the label states the
# passing condition, the rule states the threshold, and the code name stays in the footer.
GATES = (
    ("①", "输出未变", "贪心 token 完全一致", "固定文本对照，数值容差按任务设定"),
    ("②", "质量达标", "正确率、解析率均不降", "同一批评测题，指标按任务选择"),
    ("③", "确实更快", "收益 > 3%，回归 ≤ 5%", "重复压测取中位，并须大于噪声极差"),
)
GATE_CHIPS = tuple(name for _, name, _, _ in GATES)


# ----------------------------------------------------------------------- shared furniture
def add_background(slide):
    slide.shapes.add_picture(BACKGROUND, 0, 0, Inches(SW), Inches(SH))


def card(slide, x, y, w, h, fill=WHITE, line=None):
    """An opaque panel. Dense content needs one so the grid does not read through it."""
    return add_box(slide, x, y, w, h, fill=fill, line=line)


def chip_row(slide, x, y, w, h, labels, color, gap=0.3, size=11):
    chip_w = (w - gap * (len(labels) - 1)) / len(labels)
    for index, label in enumerate(labels):
        left = x + index * (chip_w + gap)
        chip = add_box(slide, left, y, chip_w, h, fill=WHITE, line=color, line_w=1.0,
                       kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        boxed_text(chip, [P(label, size, G900, True, align=CENTER)], inset=(0.03, 0, 0.03, 0))
        if index:
            add_arrow(slide, left - gap + 0.06, y + h / 2, left - 0.06, y + h / 2, color, 1.25)


def share_bar(slide, x, y, w, h, label, shares):
    add_text(slide, x - 0.52, y - 0.04, 0.46, h + 0.08,
             [P(label, 10, G500, True, MONO, align=RIGHT)], anchor=MIDDLE)
    left = x
    for fraction, color, text in shares:
        seg = add_box(slide, left, y, w * fraction, h, fill=color)
        if text:
            boxed_text(seg, [P(text, 9.5, WHITE, True, align=CENTER)], inset=(0, 0, 0, 0))
        left += w * fraction


def page_header(slide, eyebrow, title, subtitle):
    add_box(slide, MX, 0.46, 0.055, 0.26, fill=MODEL_LEVEL)
    add_text(slide, MX + 0.18, 0.42, HEAD_W - 0.18, 0.32,
             [P(eyebrow, 11, MODEL_LEVEL, True, MONO)], anchor=MIDDLE)
    add_text(slide, MX, 0.8, HEAD_W, 0.58, [P(title, 25, G900, True)], anchor=MIDDLE)
    add_text(slide, MX, 1.38, CW, 0.36, [P(subtitle, 13, G700)], anchor=MIDDLE)


def page_footer(slide, page, note):
    add_text(slide, MX, Y_FOOT, CW - 1.5, 0.28, [P(note, 10, G500)], anchor=MIDDLE)
    add_text(slide, SW - MX - 1.4, Y_FOOT, 1.4, 0.28,
             [P(f"{page} / {TOTAL}", 10, AGENT, True, MONO, align=RIGHT)], anchor=MIDDLE)


def band(slide, y, h, lead, statement, note=None):
    add_box(slide, MX, y, CW, h, fill=TINT)
    add_box(slide, MX, y, 0.07, h, fill=AGENT)
    add_text(slide, MX + 0.34, y + (0.12 if note else 0.0), CW - 0.7, 0.4 if note else h,
             [runs([(f"{lead}　", 12, AGENT, True, MONO), (statement, 16, G900, True)])],
             anchor=MIDDLE)
    if note:
        add_text(slide, MX + 0.34, y + h - 0.36, CW - 0.7, 0.3, [P(note, 11.5, G700)],
                 anchor=MIDDLE)


def bullets(slide, x, y, w, items, size=10.5, pitch=0.42, color=G700):
    for index, item in enumerate(items):
        top = y + index * pitch
        add_box(slide, x, top + (pitch - 0.1) / 2 - 0.02, 0.1, 0.1, fill=AGENT,
                kind=MSO_SHAPE.OVAL)
        add_text(slide, x + 0.22, top, w - 0.22, pitch, [P(item, size, color)], anchor=MIDDLE)


# ------------------------------------------------------------------------------------ cover
COVER_TITLE = "把模型适配里的重复判断，交给 Agent"
COVER_SUB = "M100 Model-Agent 与 XPUGen 算子 Agent：两层自动迭代，一套固定验收"
COVER_META = (("SGLang", MONO), ("　后端　·　昆仑芯 ", FONT), ("M100", MONO),
              ("　·　技术评审", FONT))
AGENDA = (
    ("1 - 2", "痛点与位置", "成本花在哪，Agent 在哪一层"),
    ("3 - 5", "模型层 Agent", "怎么跑完，收益凭什么可信"),
    ("6 - 7", "算子层 Agent", "写与判分开，一次实测迭代记录"),
    ("8 - 10", "收益与现状", "算子快了值多少，哪些已跑通"),
)


def cover_slide(slide):
    add_box(slide, MX, 2.26, 0.055, 0.28, fill=MODEL_LEVEL)
    add_text(slide, MX + 0.18, 2.22, 6.0, 0.32, [runs([
        ("2026 KUNLUNXIN · 1024", 11.5, MODEL_LEVEL, True, MONO),
        (" 程序员节", 11.5, MODEL_LEVEL, True),
    ])], anchor=MIDDLE)
    add_text(slide, MX, 2.66, HEAD_W, 0.84, [P(COVER_TITLE, 40, G900, True)], anchor=MIDDLE)
    add_text(slide, MX, 3.60, HEAD_W, 0.36, [P(COVER_SUB, 15, G700)], anchor=MIDDLE)
    add_line(slide, MX, 4.22, MX + 2.4, 4.22, AGENT, 2.5)

    y, gap = 4.56, 0.22
    w = (CW - 3 * gap) / 4
    for index, (pages, name, gist) in enumerate(AGENDA):
        x = MX + index * (w + gap)
        card(slide, x, y, w, 1.42)
        add_box(slide, x, y, w, 0.05, fill=AGENT)
        add_text(slide, x + 0.22, y + 0.2, w - 0.44, 0.26,
                 [P(f"P{pages}", 10, AGENT, True, MONO)], anchor=MIDDLE)
        add_text(slide, x + 0.22, y + 0.5, w - 0.44, 0.3,
                 [P(name, 14, G900, True)], anchor=MIDDLE)
        add_text(slide, x + 0.22, y + 0.84, w - 0.44, 0.46,
                 [P(gist, 10.5, G500)], anchor=MIDDLE)

    add_text(slide, MX, 6.56, CW, 0.3,
             [runs([(text, 11, G500, False, font) for text, font in COVER_META])],
             anchor=MIDDLE)


# ------------------------------------------------------------------------ 1 adaptation cost
MODEL_SIDE = (
    ("多模型", "主流模型持续更新", "Qwen · DeepSeek · Kimi"),
    ("新架构", "结构与精度不断变化", "MoE、多模态、低比特量化"),
    ("快迭代", "社区实现快速演进", "框架与算子需要同步跟进"),
)
ADAPT_SIDE = (
    ("接入", "从启动到精度、性能达标", "跨层排查，依赖工程经验"),
    ("补齐", "框架与算子缺口逐项补齐", "缺少实现或路径不匹配"),
    ("复验", "每次改动都要重新验证", "判断重复，经验难以沉淀"),
)
PANEL_W, PANEL_Y, PANEL_H = 5.3, 1.96, 2.96
PANEL_RIGHT_X = SW - MX - PANEL_W


def cost_panel(slide, x, title, gist, rows, tinted):
    add_box(slide, x, PANEL_Y, PANEL_W, PANEL_H, fill=TINT if tinted else G050)
    add_text(slide, x + 0.26, PANEL_Y + 0.14, PANEL_W - 0.52, 0.34, [runs([
        (title, 13.5, G900, True), (f"：{gist}", 13.5, G900, True),
    ])], anchor=MIDDLE)
    for index, (word, headline, note) in enumerate(rows):
        top = PANEL_Y + 0.58 + index * 0.74
        add_text(slide, x + 0.26, top, 1.3, 0.68, [P(word, 17, G900, True)], anchor=MIDDLE)
        add_text(slide, x + 1.66, top, PANEL_W - 1.92, 0.34, [P(headline, 12, G900, True)],
                 anchor=MIDDLE)
        add_text(slide, x + 1.66, top + 0.32, PANEL_W - 1.92, 0.32, [P(note, 10.5, G500)],
                 anchor=MIDDLE)


def cost_slide(slide):
    page_header(
        slide,
        "痛点 · 适配成本",
        "模型适配调优，是生态建设的关键成本",
        "模型按天迭代，适配按月交付；这一页先说清楚这笔成本花在哪里",
    )
    cost_panel(slide, MX, "模型侧", "持续上新", MODEL_SIDE, False)
    cost_panel(slide, PANEL_RIGHT_X, "适配侧", "反复验证", ADAPT_SIDE, True)
    pill_x = (MX + PANEL_W + PANEL_RIGHT_X) / 2
    pill = add_box(slide, pill_x - 0.7, PANEL_Y + PANEL_H / 2 - 0.19, 1.4, 0.38, fill=AGENT,
                   kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
    boxed_text(pill, [P("迭代压力", 11.5, WHITE, True, align=CENTER)])

    card(slide, MX, 5.02, CW, 0.66, fill=G050)
    add_text(slide, MX + 0.22, 5.04, CW - 0.44, 0.3,
             [P("芯片竞争力取决于硬件能力、软件成熟度与模型覆盖度", 13, AGENT, True)],
             anchor=MIDDLE)
    add_text(slide, MX + 0.22, 5.34, CW - 0.44, 0.3,
             [P("硬件能力在流片那天就定型，模型覆盖度每天被社区改写——适配效率决定硬件能力能否及时发挥",
                11.5, G700)], anchor=MIDDLE)
    band(slide, 5.76, 0.86, "命题", "降低单个模型的适配成本，才能持续扩大模型覆盖",
         "把重复的排查、评测与调优组织成可复用的 Agent 流程")
    page_footer(slide, 1, "本页为定性描述；人周、算子缺口数与复验轮次三个数待实测后填入")


# --------------------------------------------------------------------------- 2 software stack
FRAMEWORK = (
    ("SGLang", None, True),
    ("vLLM", None, False),
    ("FlashInfer", "推理内核库", False),
    ("XInfer", None, True),
)
OPERATORS = (
    ("XPU Math", None, False),
    ("XCN_ATEN_OPS", None, True),
    ("XHPC", "XBLAS / XPUDNN / XFA", False),
    ("XCCL", None, False),
)
BASE = "XCUDA（XTDK）　·　xTriton　·　XPU Runtime & Driver　·　昆仑芯 M100"
LABEL_W, FRAME_X = 1.4, 2.1
FRAME_W = SW - MX - FRAME_X
PAD, GAP = 0.16, 0.16
BOX_X, BOX_W = FRAME_X + PAD, FRAME_W - 2 * PAD


def layer_label(slide, y, h, lines):
    add_box(slide, MX, y, LABEL_W, h, fill=G100)
    add_text(slide, MX, y, LABEL_W, h,
             [P(line, 11, G700, align=CENTER) for line in lines], anchor=MIDDLE)


def module_row(slide, y, h, items, size):
    width = (BOX_W - GAP * (len(items) - 1)) / len(items)
    for index, (name, sub, agent) in enumerate(items):
        box = add_box(slide, BOX_X + index * (width + GAP), y, width, h,
                      fill=AGENT if agent else WHITE, line=None if agent else G300)
        lines = [P(name, size, WHITE if agent else G900, agent, align=CENTER)]
        if sub:
            lines.append(P(sub, 9.5, G200 if agent else G500, align=CENTER, before=1))
        boxed_text(box, lines)


def work_frame(slide, y, h, tag, gist, pointer):
    add_box(slide, FRAME_X, y, FRAME_W, h, line=AGENT, line_w=1.5,
            kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.04)
    pill = add_box(slide, FRAME_X + 0.22, y - 0.14, 1.46, 0.28, fill=AGENT,
                   kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
    boxed_text(pill, [P(tag, 10.5, WHITE, True, align=CENTER)])
    add_text(slide, BOX_X, y + h - 0.42, BOX_W, 0.28,
             [runs([(gist, 10.5, G700), (f"　→　{pointer}", 10.5, G500)])], anchor=MIDDLE)


def stack_slide(slide):
    page_header(
        slide,
        "软件栈 · AGENT 介入的位置",
        "昆仑芯 M100 软件栈：Agent 只动上面两层",
        "两类工作——框架层做模型接入调优，算子层做算子补齐提速；底层按原样使用，不改一行",
    )
    add_text(slide, 6.0, 1.8, SW - MX - 6.0, 0.28, [runs([
        ("■", 10, AGENT), ("　Agent 直接参与　　", 10, G700),
        ("□", 10, G400), ("　昆仑芯 / 社区提供", 10, G500),
    ], align=RIGHT)], anchor=MIDDLE)
    work_frame(slide, 2.12, 1.52, "模型层的工作",
               "模型接入 · 服务验证 · 按实测调优", "第 4、5 页展开")
    layer_label(slide, 2.46, 0.5, ["框架与", "推理库"])
    module_row(slide, 2.28, 0.86, FRAMEWORK, 14)
    layer_label(slide, 3.76, 0.5, ["深度学习框架"])
    module_row(slide, 3.76, 0.5, (("PyTorch", None, False),), 11.5)
    work_frame(slide, 4.5, 1.28, "算子层的工作",
               "算子缺口补齐 · 生成与调优 · 真机验证收益", "第 6、7 页展开")
    layer_label(slide, 4.72, 0.5, ["算子库"])
    module_row(slide, 4.66, 0.62, OPERATORS, 11.5)
    layer_label(slide, 5.94, 0.62, ["编译器 · 运行时", "硬件与驱动"])
    base = add_box(slide, FRAME_X, 5.94, FRAME_W, 0.62, fill=G100)
    boxed_text(base, [P(BASE, 11, G700, align=CENTER)])
    page_footer(slide, 2, "依据昆仑芯 M100 软件栈示意图整理；模块名称与分层沿用原图")


# ------------------------------------------------------------------------------- 3 overview
MODEL_COLUMN = {
    "name": "model-agent",
    "layer": "模型层",
    "gist": "模型接入、服务验证与性能调优",
    "lead": "验收链路　先确认精度正确，再确认收益",
    "chain": GATE_CHIPS,
    "chain_note": "独立采 Trace；正式压测关闭 profiler",
    "fields": (
        ("测量范围", "整个模型、真实服务与固定评测集", True),
        ("迭代约束", "轮次、预算与验收条件按任务冻结", False),
    ),
}
OPERATOR_COLUMN = {
    "name": "xpugen",
    "layer": "算子层",
    "gist": "面向 M100 生成算子实现并调优",
    "lead": "验收链路　先确认精度正确，再确认收益",
    "chain": ("能编译运行", "数值对齐参考", "真机更快"),
    "chain_note": "真机采样与模拟分析用于定位瓶颈",
    "fields": (
        ("测量范围", "单算子，多形状与独立参考对齐", True),
        ("迭代约束", "数值先过门，性能按预算持续迭代", False),
    ),
}
EXCHANGES = (
    ("输入规格 → 算子请求", "由 Trace 定位到的瓶颈算子", True),
    ("新内核 → 模型复验", "回到候选迭代，重新判一次", False),
)
COL_W, COL_Y, COL_H = 4.6, 1.86, 3.86
COL_RIGHT_X = SW - MX - COL_W
MID_X, MID_W = MX + COL_W, SW - 2 * (MX + COL_W)


def agent_column(slide, x, column):
    add_box(slide, x, COL_Y, COL_W, COL_H, fill=G050)
    add_box(slide, x, COL_Y, COL_W, 0.05, fill=AGENT)
    inner = COL_W - 0.44
    add_text(slide, x + 0.22, COL_Y + 0.18, inner, 0.34, [runs([
        (column["name"], 14, G900, True, MONO), (f"　|　{column['layer']}", 13, AGENT, True),
    ])], anchor=MIDDLE)
    add_text(slide, x + 0.22, COL_Y + 0.54, inner, 0.28,
             [P(column["gist"], 10.5, G500)], anchor=MIDDLE)
    add_line(slide, x + 0.22, COL_Y + 0.96, x + COL_W - 0.22, COL_Y + 0.96, G200, 1.0)
    add_text(slide, x + 0.22, COL_Y + 1.02, inner, 0.24,
             [P(column["lead"], 10, G500)], anchor=MIDDLE)
    chip_row(slide, x + 0.22, COL_Y + 1.32, inner, 0.42, column["chain"], AGENT,
             gap=0.26, size=9.5)
    add_text(slide, x + 0.22, COL_Y + 1.84, inner, 0.26,
             [P(column["chain_note"], 10, G400)], anchor=MIDDLE)
    for index, (label, value, highlight) in enumerate(column["fields"]):
        top = COL_Y + 2.22 + index * 0.72
        add_box(slide, x + 0.22, top, inner, 0.64, fill=TINT if highlight else WHITE,
                line=None if highlight else G200)
        add_text(slide, x + 0.4, top + 0.08, inner - 0.36, 0.26,
                 [P(label, 10, AGENT if highlight else G500, False, MONO)], anchor=MIDDLE)
        add_text(slide, x + 0.4, top + 0.32, inner - 0.36, 0.28,
                 [P(value, 11, G900)], anchor=MIDDLE)


def exchange(slide, y, title, note, rightward):
    add_text(slide, MID_X + 0.05, y, MID_W - 0.1, 0.28,
             [P(title, 11, G900, True, align=CENTER)], anchor=MIDDLE)
    left, right = MID_X + 0.16, MID_X + MID_W - 0.16
    tip, tail = (right, left) if rightward else (left, right)
    add_line(slide, tail, y + 0.38, tip, y + 0.38, AGENT, 1.5, tail=True)
    add_text(slide, MID_X + 0.05, y + 0.44, MID_W - 0.1, 0.26,
             [P(note, 10, G500, align=CENTER)], anchor=MIDDLE)


def overview_slide(slide):
    page_header(
        slide,
        "纵览 · 模型层与算子层",
        "两个 Agent：model-agent 调模型，xpugen 写算子",
        "两层结构相同——模型只负责提出，判定交给固定验收与真机测量；区别在于测量范围",
    )
    agent_column(slide, MX, MODEL_COLUMN)
    agent_column(slide, COL_RIGHT_X, OPERATOR_COLUMN)
    for index, (title, note, rightward) in enumerate(EXCHANGES):
        exchange(slide, 2.67 + index * 1.54, title, note, rightward)
    band(slide, 5.88, 0.82, "要点", "同一个算子内核，既要通过算子验收，也要通过整模型复验",
         "算子 bench 衡量的是局部收益；能否交付，由真实模型上的复测决定")
    page_footer(slide, 3, "两个 Agent 分属不同仓库，本页只说明职责与交接关系，各自机制见后续页")


# -------------------------------------------------------------------------- 4 human and agent
# The three command lines are context, not content, so they are one line of text rather than
# a banner of boxes: that buys the vertical room for each phase to state what it produces,
# which is what makes the A/B/C/D chain legible as a chain.
COMMAND_LINES = (
    ("三条命令线共用同一套契约　", 10, G500),
    ("架构接入", 10.5, G900, True),
    (" onboard-*", 9.5, G500, False, MONO),
    ("　→　", 10, G400),
    ("性能调优", 10.5, AGENT, True),
    (" run / resume", 9.5, AGENT, False, MONO),
    ("　→　", 10, G400),
    ("持续优化", 10.5, G900, True),
    (" optimize-*", 9.5, G500, False, MONO),
    ("　　本页讲中间这条", 10, G500),
)
HUMAN_IN = (
    "模型 / 负载 / 并发",
    "精度阈值与预算",
    "已知缺陷与修复方案",
    "确定输入与任务约定",
)
HUMAN_IN_NOTE = "契约改动后，已有结果不再沿用"
HUMAN_OUT = (
    "核对精度、性能与失败记录",
    "验证通过：导出启动方案",
    "失败或无收益：转人工诊断",
    "逐轮追溯，必要时回退",
)
HUMAN_OUT_NOTE = "交付引用已验证的隔离副本"
PHASES = (
    ("A", "建立基线", "先完成预检，再跑全量评测", "产出：基线指标与可复现的启动方式", False),
    ("B", "采集 Trace", "独立进程采样，不参与计时", "产出：热点清单，只用于选方向", False),
    ("C", "候选迭代", "小步改动，逐个按三道条件验收", "产出：通过筛选的候选与各自收益　→ 下一页", True),
    ("D", "最终复验", "重测基线，独立复验每个候选", "产出：可交付的最优候选", False),
)
FEEDBACK = "将验证过的规则、补丁与失败记录沉淀下来，供下一次任务复用"


def human_column(slide, x, w, y, h, role, items, note, inbound):
    add_box(slide, x, y, w, h, fill=G050)
    add_box(slide, x, y, w, 0.05, fill=AGENT)
    add_text(slide, x + 0.22, y + 0.14, w - 0.44, 0.34, [runs([
        ("人", 13, G900, True), (f"：{role}", 13, G900, True),
    ])], anchor=MIDDLE)
    bullets(slide, x + 0.22, y + 0.6, w - 0.44, items, size=10.5, pitch=0.46)
    add_text(slide, x + 0.22, y + h - 0.52, w - 0.44, 0.3, [P(note, 10, G500)], anchor=MIDDLE)
    add_text(slide, x, y + h - 0.24, w, 0.22,
             [P("交给 Agent" if inbound else "交回给人", 9.5, AGENT, False, MONO, align=CENTER)],
             anchor=MIDDLE)


def iteration_slide(slide):
    page_header(
        slide,
        "模型层 · 一次任务怎么跑完",
        "M100 Agent 执行迭代，结果交回人验收",
        "人只在两头定目标与判交付；中间四个阶段无人值守，每个阶段都有明确产出",
    )
    card(slide, MX, 1.78, CW, 0.36, fill=G050)
    add_text(slide, MX + 0.18, 1.78, CW - 0.36, 0.36, [runs(COMMAND_LINES)], anchor=MIDDLE)

    y, h = 2.2, 3.56
    left_w, mid_w, right_w = 2.5, 6.3, 2.933
    mid_x = MX + left_w + 0.2
    right_x = mid_x + mid_w + 0.2

    human_column(slide, MX, left_w, y, h, "定义目标", HUMAN_IN, HUMAN_IN_NOTE, True)
    add_box(slide, mid_x, y, mid_w, h, fill=WHITE, line=AGENT)
    add_text(slide, mid_x + 0.22, y + 0.14, mid_w - 0.44, 0.34, [runs([
        ("Agent", 13, G900, True, MONO), ("：在固定约束内自动迭代", 13, G900, True),
        ("　全程无人值守", 11, G500),
    ])], anchor=MIDDLE)
    row_x, detail_x, detail_w = mid_x + 0.22, mid_x + 2.05, 3.45
    for index, (letter, title, detail, yields, looping) in enumerate(PHASES):
        row_y = y + 0.56 + index * 0.72
        add_box(slide, row_x, row_y, mid_w - 0.44, 0.68, fill=TINT if looping else G050)
        number_dot(slide, row_x + 0.12, row_y + 0.19, 0.3, AGENT, letter, 11)
        add_text(slide, row_x + 0.52, row_y, 1.5, 0.68, [P(title, 12, G900, True)], anchor=MIDDLE)
        add_text(slide, detail_x, row_y + 0.06, detail_w, 0.3, [P(detail, 10, G700)],
                 anchor=MIDDLE)
        add_text(slide, detail_x, row_y + 0.36, detail_w, 0.28,
                 [P(yields, 9.5, AGENT if looping else G500)], anchor=MIDDLE)
        if looping:
            loop_x = mid_x + mid_w - 0.7
            add_path_arrow(slide, [
                (loop_x, row_y + 0.54), (loop_x + 0.26, row_y + 0.54),
                (loop_x + 0.26, row_y + 0.14), (loop_x, row_y + 0.14),
            ], AGENT, 1.25)

    human_column(slide, right_x, right_w, y, h, "判断交付", HUMAN_OUT, HUMAN_OUT_NOTE, False)
    back_y = y + h + 0.26
    add_path_arrow(slide, [
        (right_x + right_w / 2, y + h), (right_x + right_w / 2, back_y),
        (MX + left_w / 2, back_y), (MX + left_w / 2, y + h),
    ], AGENT, 1.25)
    add_text(slide, MX, back_y + 0.08, CW, 0.3, [P(FEEDBACK, 11, G700, align=CENTER)],
             anchor=MIDDLE)
    page_footer(slide, 4, "已有流程可自动执行；新架构、未知缺陷与关键决策仍需要人工参与")


# ----------------------------------------------------------------------------- 5 candidate gate
PREPARE = "隔离副本　→　应用改动　→　构建并记录产物信息"
SERVER = "　核对端口、设备与模型路径，确认被测对象就是该候选"
# The three gates above show how a candidate is judged; these three say why the resulting
# number can be trusted. Together they are the whole answer to "凭什么信这个收益".
TRUST = (
    "口径先冻结：模型、负载、阈值与预算随任务写定，改了之后已有结果不再沿用",
    "测的是同一个候选：隔离副本、记录产物身份，启动时核对端口与模型路径",
    "不能靠少干活变快：输入、精度、上下文与缓存全部冻结，诊断与正式压测分开",
)
# What the three gates actually did on the 30B-A3B run. One gate stopped a single mode inside
# a candidate rather than the whole candidate, so its tally is counted in modes: five groups in,
# three stopped by the performance gate, two delivered.
GATE_RESULT = (
    ("①", "输出未变", "挡掉 1 个模式", True,
     ("分组关掉后 cos 0.000，算出来直接是错的",)),
    ("②", "质量达标", "本轮没拦下", False,
     ("过了 ① 的候选输出逐位一致，不必再看质量",)),
    ("③", "确实更快", "挡掉 3 组", True,
     ("强制不分组 −20%；换现成核 0.61× 更慢；",
      "路由融合最多只值 0.9%，够不着 3% 的门槛")),
)
GATE_SHIPPED = ("激活与量化合成一个核", "换成耗时不随规模涨的核")
GATE_PEAK = (1370.0, 1883.0)


def verdict_panel(slide, x, y, w, h):
    """The left half says how a candidate is judged; this says what that judging produced."""
    card(slide, x, y, w, h, fill=G050)
    add_box(slide, x, y, 0.05, h, fill=MODEL_LEVEL)
    add_text(slide, x + 0.22, y + 0.06, w - 0.44, 0.28, [runs([
        ("本轮实际判定", 12.5, G900, True), ("　30B-A3B 单卡", 10, G500),
    ])], anchor=MIDDLE)

    add_box(slide, x + 0.22, y + 0.38, w - 0.44, 0.32, fill=G100)
    add_text(slide, x + 0.40, y + 0.38, w - 0.80, 0.32, [runs([
        ("送进验收", 11, G900, True), ("　五组候选，由便宜到贵逐道验", 10, G500),
    ])], anchor=MIDDLE)

    for index, (mark, name, tally, stopped, reasons) in enumerate(GATE_RESULT):
        top = y + 0.78 + index * 0.72
        number_dot(slide, x + 0.26, top + 0.03, 0.3, AGENT, mark, 11)
        add_text(slide, x + 0.66, top, 1.3, 0.28, [P(name, 11.5, G900, True)], anchor=MIDDLE)
        add_text(slide, x + w - 1.72, top, 1.5, 0.28,
                 [P(tally, 11, DROP_C if stopped else G500, stopped, align=RIGHT)],
                 anchor=MIDDLE)
        add_text(slide, x + 0.66, top + 0.28, w - 0.88, 0.40,
                 [P(line, 10, G700 if stopped else G500) for line in reasons], anchor=MIDDLE)

    ship_y = y + 2.96
    add_box(slide, x + 0.22, ship_y, w - 0.44, 0.78, fill=TINT)
    add_text(slide, x + 0.40, ship_y + 0.04, w - 0.80, 0.26, [runs([
        ("三道全过", 11, G900, True), (f"　{len(GATE_SHIPPED)} 组 · 交付", 11, SHIP_C, True),
    ])], anchor=MIDDLE)
    for index, item in enumerate(GATE_SHIPPED):
        add_text(slide, x + 0.40, ship_y + 0.30 + index * 0.22, w - 0.80, 0.22,
                 [P(f"· {item}", 10, G700)], anchor=MIDDLE)
    base, shipped = GATE_PEAK
    add_text(slide, x + 0.22, y + 3.82, w - 0.44, 0.26, [runs([
        ("峰值吞吐 ", 10, G500), (f"{base:.0f} → {shipped:.0f} tok/s", 10, G900, True, MONO),
        (f"　+{shipped / base - 1:.0%}", 11, SHIP_C, True, MONO),
    ])], anchor=MIDDLE)


def gate_slide(slide):
    page_header(
        slide,
        "模型层 · 凭什么相信这些数字",
        "每个候选按三道固定条件验收，未通过即停止后续测试",
        "这一页回答一个问题：Agent 报出来的收益，凭什么可信",
    )
    fig_w = 5.0
    fig_x = SW - MX - fig_w
    w = fig_x - MX - 0.4

    add_text(slide, MX, 1.88, w, 0.28, [runs([
        ("准备　", 11, G900, True), (PREPARE, 10.5, G700),
    ])], anchor=MIDDLE)
    add_box(slide, MX, 2.22, w, 0.42, fill=G050)
    add_text(slide, MX + 0.24, 2.22, w - 0.48, 0.42, [runs([
        ("启动服务", 11.5, G900, True), (SERVER, 10, G700),
    ])], anchor=MIDDLE)
    for index, (mark, name, rule, detail) in enumerate(GATES):
        y = 2.76 + index * 0.56
        add_box(slide, MX, y, w, 0.5, fill=WHITE, line=G200)
        add_text(slide, MX + 0.18, y, 0.36, 0.5, [P(mark, 13, AGENT, True)], anchor=MIDDLE)
        add_text(slide, MX + 0.58, y, 1.0, 0.5, [P(name, 12, G900, True)], anchor=MIDDLE)
        add_text(slide, MX + 1.66, y, 2.2, 0.5, [P(rule, 11, G900)], anchor=MIDDLE)
        add_text(slide, MX + 3.94, y, w - 4.12, 0.5, [P(detail, 10, G500)], anchor=MIDDLE)
    add_text(slide, MX, 4.40, w, 0.22,
             [P("由便宜到贵排列，任一道不通过即停止后续测试", 9.5, G500)], anchor=MIDDLE)

    card(slide, MX, 4.72, w, 1.3, fill=G050)
    add_box(slide, MX, 4.72, 0.05, 1.3, fill=MODEL_LEVEL)
    add_text(slide, MX + 0.22, 4.80, w - 0.44, 0.28, [runs([
        ("凭什么可信", 12, G900, True), ("　三条防止自欺的约束", 10.5, G500),
    ])], anchor=MIDDLE)
    bullets(slide, MX + 0.22, 5.10, w - 0.44, TRUST, size=10.5, pitch=0.32)

    verdict_panel(slide, fig_x, 1.88, fig_w, 4.18)
    band(slide, 6.16, 0.52, "口径", "只有独立复验通过的收益，才进入最终交付")
    page_footer(slide, 5, "代码中三道条件分别为数值门、质量门与性能门；阈值随任务冻结，候选无权修改")


# ------------------------------------------------------------------------ 6 operator generation
WRITE_STEPS = (
    ("澄清需求与约束", "比较方案，固定输入语义"),
    ("依据硬件说明卡生成实现", "明确数据搬运、计算与同步"),
)
JUDGE_STEPS = (
    ("L0", "编译与运行", "构建并启动，失败即返工"),
    ("L1", "数值验证", "多形状多精度对照独立参考"),
    ("Bench", "真机计时", "固定负载，重复测量"),
    ("Profile", "瓶颈分析", "模拟器定位，不作判据"),
)
BATCHNORM = (
    ("固定工作负载", "256 × 4096", "fp32 · 总流量 8.39 MB", False),
    ("相对基线加速", "1.37×", "13.24 → 9.66 µs", True),
    ("最优版本重复测量", "9.66 µs · iters=200", "空闲卡上复测", False),
)


def operator_slide(slide):
    page_header(
        slide,
        "算子层 · 生成与性能迭代",
        "M100 Agent：算子生成与性能迭代",
        "从需求或已有实现出发，先确认精度正确，再迭代性能；写与判分开，判的一侧没有模型",
    )
    panel_x = 9.1
    w = panel_x - MX - 0.4
    block_y, block_h = 2.56, 2.5
    block_w = (w - 0.5) / 2
    left, right = MX, MX + block_w + 0.5

    add_box(slide, left, block_y, block_w, block_h, fill=TINT)
    add_text(slide, left + 0.24, block_y + 0.14, block_w - 0.48, 0.32, [runs([
        ("写算子", 13, G900, True), ("　|　Agent", 12, AGENT, True, MONO),
        ("　全流程唯一的模型", 10, G500),
    ])], anchor=MIDDLE)
    for index, (name, detail) in enumerate(WRITE_STEPS):
        top = block_y + 0.66 + index * 0.92
        add_text(slide, left + 0.24, top, block_w - 0.48, 0.3,
                 [P(name, 12.5, G900, True)], anchor=MIDDLE)
        add_text(slide, left + 0.24, top + 0.3, block_w - 0.48, 0.28,
                 [P(detail, 10.5, G700)], anchor=MIDDLE)

    add_box(slide, right, block_y, block_w, block_h, fill=G050)
    add_text(slide, right + 0.24, block_y + 0.14, block_w - 0.48, 0.32, [runs([
        ("判算子", 13, G900, True), ("　|　Harness", 12, G500, True, MONO),
        ("　固定规则，无模型", 10, G500),
    ])], anchor=MIDDLE)
    for index, (tag, question, detail) in enumerate(JUDGE_STEPS):
        top = block_y + 0.62 + index * 0.44
        add_text(slide, right + 0.24, top, block_w - 0.48, 0.4, [runs([
            (tag, 9.5, AGENT, True, MONO), (f"　{question}", 11.5, G900, True),
            (f"　{detail}", 10, G500),
        ])], anchor=MIDDLE)

    out, back = left + block_w / 2, right + block_w / 2
    add_path_arrow(slide, [(out, block_y), (out, block_y - 0.3), (back, block_y - 0.3),
                           (back, block_y - 0.04)], AGENT, 1.5)
    add_text(slide, out, block_y - 0.62, back - out, 0.3,
             [P("交付候选实现", 10.5, G700, align=CENTER)], anchor=MIDDLE)
    floor = block_y + block_h
    add_path_arrow(slide, [(back, floor), (back, floor + 0.3), (out, floor + 0.3),
                           (out, floor + 0.04)], AGENT, 1.5)
    add_text(slide, out, floor + 0.34, back - out, 0.3,
             [P("评测结果与瓶颈证据回传，进入下一轮", 10.5, G700, align=CENTER)], anchor=MIDDLE)

    add_box(slide, MX, 5.92, w, 0.66, fill=WHITE, line=G200)
    add_text(slide, MX + 0.24, 5.98, w - 0.48, 0.54, [
        runs([("与硬件耦合的只有后端这一层：编译 · 运行 · 性能采集", 10.5, G900, True),
              ("　backend/", 10, G500, False, MONO)], after=2),
        P("换新芯片只需补齐这一层，写与判的流程无需改动", 10.5, G500),
    ])

    add_box(slide, panel_x, 1.9, SW - MX - panel_x, 4.68, fill=TINT)
    pw = SW - MX - panel_x
    add_text(slide, panel_x + 0.26, 2.04, pw - 0.52, 0.34,
             [runs([("batch_norm 实测样例", 14, G900, True), ("　单算子", 10.5, G400)])],
             anchor=MIDDLE)
    row_y = 2.56
    for label, value, note, hero in BATCHNORM:
        add_box(slide, panel_x + 0.26, row_y, pw - 0.52, 1.0,
                fill=MODEL_TINT if hero else WHITE, line=None if hero else G200)
        add_text(slide, panel_x + 0.44, row_y, pw - 0.88, 1.0, [
            P(label, 10.5, MODEL_LEVEL if hero else G500, after=2),
            P(value, 19, MODEL_LEVEL if hero else G900, True, "Arial", after=2),
            P(note, 9.5, G400, after=0),
        ], anchor=MIDDLE)
        row_y += 1.12
    add_text(slide, panel_x + 0.26, row_y - 0.04, pw - 0.52, 0.5, [
        P("单算子样例，尚无规模化统计；", 10, G500, after=1),
        P("基线 batch_norm_v4，详见下一页", 10, G500),
    ])
    page_footer(slide, 6, "与算子团队协作，沉淀 Skills 与开发经验；真机实测为准，模拟器只用于定位")


# ------------------------------------------------------------ 7 batch_norm iteration record
# Measured on M100 / houyi. The roofline here is a pure bandwidth bound -- no issue-slot
# constraint and no fixed launch cost -- so it is an upper bound, not a reachable target.
# Round 1 is what makes the page: it changed no code, it only measured how long a pure copy
# of the same bytes takes. That answer (9.83us, 0.717) is the real ceiling, and everything
# after it is read against that rather than against the roofline.
WORKLOAD = (
    ("工作负载　", 10, G900, True),
    ("N=1 · C=256 · HW=4096 · fp32", 9.5, AGENT, True, MONO),
    ("　1,048,576 元素，读写各 4 MB，总流量 8.39 MB　|　roofline ", 10, G700),
    ("7.049 µs", 9.5, G900, True, MONO),
    ("　按带宽上限计算，不含发射槽约束与固定开销", 10, G500),
)
ROUNDS = (
    ("0", "batch_norm_v4 基线", "13.24", "基线", "base"),
    ("1", "标定纯拷贝地板（探针）", "9.83", "探针", "probe"),
    ("2", "换原生 rsqrt 指令", "11.72", "保留", "keep"),
    ("3", "shared 广播系数", "15.06", "丢弃", "drop"),
    ("4", "host 折系数 + 每次 H2D", "17.60", "丢弃", "drop"),
    ("5", "修 copy 的寄存器 spill", "9.81", "保留", "keep"),
    ("6", "fold 成 scale*x+shift", "9.82", "保留", "keep"),
    ("7 · 8", "模拟器脚手架 · 修 ABI", "—", "基建", "infra"),
    ("交付", "空闲卡复测 iters=200", "9.66", "交付", "ship"),
)
BEST = (0.532, 0.532, 0.601, 0.601, 0.601, 0.719, 0.719, 0.719, 0.719, 0.730)
MARKS = (
    (0, 0.532, "base", "0.532", False),
    (2, 0.601, "keep", "0.601", True),
    (3, 0.468, "drop", "0.468", True),
    (4, 0.401, "drop", "0.401", True),
    (5, 0.719, "keep", "0.719", True),
    (6, 0.718, "keep", None, True),
    (9, 0.730, "ship", "0.730", True),
)
X_TICKS = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "交付")
FLOOR, TARGET = 0.717, 0.700
SHAPES = (
    ("2 × 1600", "0.143", "0.344 → 2.407 µs", False),
    ("32 × 6400", "0.364", "1.377 → 3.788 µs", False),
    ("64 × 4096", "0.449", "1.762 → 3.925 µs", False),
    ("256 × 4096", "0.724", "7.049 → 9.736 µs", True),
)
PLOT_X0, PLOT_X1 = 1.30, 7.15
PLOT_Y0, PLOT_Y1 = 2.56, 4.70
# Data is inset from the frame so the first marker's label clears the y-axis numbers.
DATA_X0, DATA_X1 = 1.85, 6.85
# Headroom above 0.73 leaves room for the delivery label without hitting the legend row.
Y_LO, Y_HI = 0.38, 0.80
Y_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
# Named by meaning so each deck can bind the chart to its own accents. The delivery marker and
# its legend dot must stay the same colour, which is why SHIP_C is used for both.
LINE_C = SHIP_C = AGENT
KEEP_C, DROP_C = "0E9A72", "D94A3D"
TARGET_C, FLOOR_C = AGENT, MODEL_LEVEL
VERDICT = {"keep": KEEP_C, "drop": DROP_C, "ship": SHIP_C,
           "base": G500, "probe": G400, "infra": G400}


def _px(index):
    return DATA_X0 + index * (DATA_X1 - DATA_X0) / (len(X_TICKS) - 1)


def _py(value):
    return PLOT_Y1 - (value - Y_LO) / (Y_HI - Y_LO) * (PLOT_Y1 - PLOT_Y0)


def swatch(slide, x, y, color, dash, text, width):
    add_line(slide, x, y + 0.15, x + 0.26, y + 0.15, color, 1.25, dash=dash)
    add_text(slide, x + 0.32, y, width - 0.32, 0.30, [P(text, 9, G500)], anchor=MIDDLE)


def iteration_chart(slide, x, y, w, h):
    add_box(slide, x, y, w, h, fill=G050)
    add_text(slide, x + 0.22, y + 0.06, 2.85, 0.26, [runs([
        ("achieved", 10, G900, True, MONO), (" = roofline 耗时 ÷ 实测耗时", 10, G700),
    ])], anchor=MIDDLE)
    swatch(slide, x + 3.17, y + 0.04, TARGET_C, DASH, "目标 0.70 · 10.07 µs", 1.80)
    swatch(slide, x + 5.05, y + 0.04, FLOOR_C, DASH, "拷贝地板 0.717 · 9.83 µs", 2.10)

    for value in Y_GRID:
        gy = _py(value)
        if value != TARGET:
            add_line(slide, PLOT_X0, gy, PLOT_X1, gy, G200, 0.75)
        add_text(slide, PLOT_X0 - 0.62, gy - 0.11, 0.5, 0.22,
                 [P(f"{value:.2f}", 9, G400, align=RIGHT)], anchor=MIDDLE)
    for value, color in ((TARGET, TARGET_C), (FLOOR, FLOOR_C)):
        add_line(slide, PLOT_X0, _py(value), PLOT_X1, _py(value), color, 1.0, dash=DASH)
    add_line(slide, PLOT_X0, PLOT_Y1, PLOT_X1, PLOT_Y1, G300, 1.0)

    for index in range(len(BEST) - 1):
        add_line(slide, _px(index), _py(BEST[index]), _px(index + 1), _py(BEST[index + 1]),
                 LINE_C, 2.0)
    for index, tick in enumerate(X_TICKS):
        add_text(slide, _px(index) - 0.33, PLOT_Y1 + 0.04, 0.66, 0.22,
                 [P(tick, 9.5, G500, align=CENTER)], anchor=MIDDLE)

    for index, value, kind, label, above in MARKS:
        color = VERDICT[kind]
        size = 0.19 if kind == "ship" else 0.14
        cx, cy = _px(index), _py(value)
        add_box(slide, cx - size / 2, cy - size / 2, size, size, fill=color,
                line=WHITE, line_w=1.25, kind=MSO_SHAPE.OVAL)
        if label:
            top = cy - 0.3 if above else cy + 0.1
            add_text(slide, cx - 0.4, top, 0.8, 0.22,
                     [P(label, 10.5 if kind == "ship" else 9.5, color, True, "Arial",
                        align=CENTER)], anchor=MIDDLE)
    add_text(slide, x + 0.22, y + h - 0.3, w - 0.44, 0.24, [runs([
        ("●", 9, KEEP_C), (" 保留　", 9, G500),
        ("●", 9, DROP_C), (" 真机测出更慢，当场丢弃　", 9, G500),
        ("●", 9, SHIP_C), (" 交付复测", 9, G500),
    ])], anchor=MIDDLE)


def round_list(slide, x, y, w, h):
    add_box(slide, x, y, w, h, fill=G050)
    add_text(slide, x + 0.22, y + 0.08, w - 0.44, 0.28, [runs([
        ("八轮迭代", 12.5, G900, True), ("　三轮保留，两轮丢弃", 10, G500),
    ])], anchor=MIDDLE)
    for index, (tag, what, micros, verdict, kind) in enumerate(ROUNDS):
        top = y + 0.5 + index * 0.31
        color = VERDICT[kind]
        shipped = kind == "ship"
        if shipped or kind == "drop":
            add_box(slide, x + 0.14, top, w - 0.28, 0.29,
                    fill=TINT if shipped else G100)
        add_text(slide, x + 0.22, top, 0.52, 0.29,
                 [P(tag, 9.5, color, True, MONO)], anchor=MIDDLE)
        add_text(slide, x + 0.8, top, 2.3, 0.29, [P(what, 9.5, G900 if shipped else G700)],
                 anchor=MIDDLE)
        add_text(slide, x + 3.1, top, 0.74, 0.29,
                 [P(micros, 9.5, G900, shipped, MONO, align=RIGHT)], anchor=MIDDLE)
        add_text(slide, x + 3.98, top, 0.7, 0.29,
                 [P(verdict, 9.5, color, True)], anchor=MIDDLE)


def shape_strip(slide, y, h):
    lead_w = 2.9
    add_text(slide, MX, y, lead_w, h, [
        P("同一实现，换个 shape", 11.5, G900, True, after=2),
        P("固定 launch 开销约 1.95 µs，", 9.5, G500, after=0),
        P("小 shape 整个被它吃掉", 9.5, G500),
    ], anchor=MIDDLE)
    gap = 0.16
    x0 = MX + lead_w + 0.1
    cell_w = (SW - MX - x0 - 3 * gap) / 4
    for index, (shape, achieved, detail, hero) in enumerate(SHAPES):
        x = x0 + index * (cell_w + gap)
        add_box(slide, x, y, cell_w, h, fill=TINT if hero else WHITE,
                line=None if hero else G200)
        add_text(slide, x + 0.16, y + 0.05, cell_w - 0.32, 0.22,
                 [P(shape, 9.5, AGENT if hero else G500, hero, MONO)], anchor=MIDDLE)
        add_text(slide, x + 0.16, y + 0.25, cell_w - 0.32, 0.26,
                 [P(achieved, 15, G900, True, "Arial")], anchor=MIDDLE)
        add_text(slide, x + 0.16, y + 0.49, cell_w - 0.32, 0.18,
                 [P(detail, 8.5, G400, False, MONO)], anchor=MIDDLE)


def record_slide(slide):
    page_header(
        slide,
        "算子层 · 一次完整的迭代记录",
        "batch_norm：八轮迭代，achieved 从 0.532 到 0.730",
        "判定全部来自真机测量——更慢的两轮当场丢弃，没有一轮是靠模型自己说好",
    )
    add_box(slide, MX, 1.78, CW, 0.32, fill=G050)
    add_text(slide, MX + 0.2, 1.78, CW - 0.4, 0.32, [runs(WORKLOAD)], anchor=MIDDLE)

    chart_w = 7.15
    iteration_chart(slide, MX, 2.16, chart_w, 3.30)
    list_x = MX + chart_w + 0.2
    round_list(slide, list_x, 2.16, SW - MX - list_x, 3.30)
    shape_strip(slide, 5.56, 0.68)

    band(slide, 6.34, 0.52, "读数",
         "roofline 只是上界，纯拷贝地板 9.83 µs 才是这张卡的天花板——交付已经打在上面")
    page_footer(slide, 7, "第 4 轮的 17.60 µs 含每次调用的 H2D 上传，不只是 kernel 本身；"
                          "交付复测、地板探针、下排 shape 表各为独立批次，"
                          "0.730 与 0.724 不应互相换算")


# ---------------------------------------------------------------- 8 local speedup to step gain
SCENARIOS = (
    ("场景 A", "短输入，GEMM 为主", (("GEMM", 0.80), ("Attention", 0.10), ("其他", 0.10))),
    ("场景 B", "长上下文，Attention 变重", (("GEMM", 0.50), ("Attention", 0.40), ("其他", 0.10))),
)
CUT = 0.30
REASONS = (
    ("占比决定直接收益", ("占比小的算子再快，也带不动整步",)),
    ("整步耗时还不是服务容量", ("排队、批次、缓存与时延目标仍会改变结果，", "下一页给出完整的换算办法")),
    ("每轮优化后重新定位", ("局部提速后瓶颈可能转移，", "下一轮应按新的 Trace 重排优先级")),
)


def gain_slide(slide):
    page_header(
        slide,
        "难点 · 回到模型层看收益",
        "同样把一类算子耗时降低 30%，整步收益取决于它原来的占比",
        "假设算例：固定总耗时 100 ms，其他部分不变；非模型实测，也不是 SLO 容量预测",
    )
    label_w, cell_w, gap = 2.0, 1.5, 0.08
    xs = [MX + label_w + gap + index * (cell_w + gap) for index in range(3)]
    head_y = 1.96
    add_text(slide, MX + 0.16, head_y, label_w - 0.16, 0.56,
             [P("单步耗时示意", 11, G700, True), P("其他部分不变", 9.5, G500)], anchor=MIDDLE)
    for x, head in zip(xs, ("基线", "Attention\n耗时 −30%", "GEMM\n耗时 −30%")):
        add_text(slide, x, head_y, cell_w, 0.56,
                 [P(line, 11, G700, True, align=CENTER) for line in head.split("\n")],
                 anchor=MIDDLE)

    y, row_h = 2.62, 1.5
    for name, gist, shares in SCENARIOS:
        label = add_box(slide, MX, y, label_w, row_h, fill=G100)
        lines = [P(name, 15, G900, True, after=3), P(gist, 9.5, G500, after=4)]
        lines += [P(f"{op} {int(part * 100)}%", 10, G500, after=1) for op, part in shares]
        boxed_text(label, lines, inset=(0.16, 0.04, 0.08, 0.04))
        parts = dict(shares)
        cells = [("100 ms", ("固定初始耗时",), False)]
        for op in ("Attention", "GEMM"):
            step = 100 - parts[op] * 100 * CUT
            cells.append((f"−{round(parts[op] * 100 * CUT)}%",
                          (f"{step:g} ms", f"100 − {int(parts[op] * 100)}×30%"),
                          parts[op] == max(parts.values())))
        for x, (value, note, hero) in zip(xs, cells):
            baseline = value.endswith("ms")
            add_box(slide, x, y, cell_w, row_h,
                    fill=MODEL_TINT if hero else (G050 if baseline else WHITE),
                    line=None if hero or baseline else G200)
            color = G900 if baseline else (MODEL_LEVEL if hero else AGENT)
            add_text(slide, x, y + 0.24, cell_w, 0.6,
                     [P(value, 23 if baseline else 25, color, True, "Arial", align=CENTER)],
                     anchor=MIDDLE)
            add_text(slide, x + 0.06, y + 0.9, cell_w - 0.12, 0.46,
                     [P(line, 9.5, G500, align=CENTER, after=1) for line in note], anchor=TOP)
        y += row_h + 0.12

    # The table's last cell ends at 7.34, so the panel starts clear of it rather than
    # clipping the GEMM column.
    panel_x = 7.40
    rx = panel_x + 0.18
    rw = SW - MX - rx - 0.18
    card(slide, panel_x, 1.9, SW - MX - panel_x, 3.88, fill=G050)
    add_text(slide, rx, 1.96, rw, 0.4, [P("为什么要回到模型看收益", 14, G900, True)],
             anchor=MIDDLE)
    tops = (2.40, 4.14, 5.10)
    for index, ((title, body), top) in enumerate(zip(REASONS, tops), start=1):
        number_dot(slide, rx, top, 0.32, AGENT, str(index), 11)
        add_text(slide, rx + 0.45, top - 0.03, rw - 0.45, 0.36,
                 [P(title, 12.5, G900, True)], anchor=MIDDLE)
        add_text(slide, rx + 0.45, top + 0.34, rw - 0.45, 0.5,
                 [P(line, 10.5, G700, after=1) for line in body])
    bar_x, bar_w = rx + 0.97, rw - 0.97
    for offset, (name, _, shares) in zip((0.76, 1.06), SCENARIOS):
        parts = dict(shares)
        share_bar(slide, bar_x, tops[0] + offset, bar_w, 0.24, name.replace("场景 ", ""), (
            (parts["GEMM"], SHARE_GEMM, f"GEMM {int(parts['GEMM'] * 100)}%"),
            (parts["Attention"], SHARE_ATTN,
             f"Attention {int(parts['Attention'] * 100)}%" if parts["Attention"] > 0.2 else ""),
            (parts["其他"], SHARE_OTHER, ""),
        ))
    legend_x, legend_y = bar_x, tops[0] + 1.32
    for color, text, width in ((SHARE_GEMM, "GEMM", 0.82), (SHARE_ATTN, "Attention", 1.1),
                               (SHARE_OTHER, "通信等其他", 1.12)):
        add_box(slide, legend_x, legend_y + 0.055, 0.13, 0.13, fill=color)
        add_text(slide, legend_x + 0.18, legend_y, width - 0.18, 0.24,
                 [P(text, 9.5, G500)], anchor=MIDDLE)
        legend_x += width
    add_text(slide, legend_x, legend_y, bar_x + bar_w - legend_x, 0.24,
             [P("假设占比", 9.5, G400, align=RIGHT)], anchor=MIDDLE)

    band(slide, 5.98, 0.58, "结论", "先按负载判断优化优先级，再用整模型压测确认最终收益")
    page_footer(slide, 8, "表中四个数为按占比直接换算；真实服务还要叠加排队、批次与缓存的影响")


# ------------------------------------------------------------------- 8 kernel time to capacity
CHAIN = (
    ("KERNEL", "算子耗时", ("max(计算时间, 访存时间)", "+ 启动 / 通信开销"), False),
    ("STEP", "单步耗时", ("按真实 batch 形状", "汇总各类算子耗时", "区分 prefill / decode"), False),
    ("SCHEDULE", "请求时延", ("模拟请求与批次", "考虑缓存准入", "得到时延与吞吐"), True),
    ("SLO", "服务目标", ("按业务目标判断", "TTFT / TPOT 达标率", "观察先超标的环节"), True),
    ("CAPACITY", "单机容量", ("搜索满足 SLO 的", "最大请求到达率", "比较候选部署方案"), True),
)
USES = (
    ("定位热点", "已接入", "按真实 Trace 定位热点", "在固定负载上验证候选"),
    ("What-if 推演", "待 M100 校准", "算子提速 → 预测变化", "校准后用于筛选方向"),
    ("向算子层提需求", "待补齐", "形成 shape 与实现需求", "新内核回接后做模型复验"),
)


def capacity_slide(slide):
    page_header(
        slide,
        "方法 · 模型层如何给算子排优先级",
        "M100 Agent：先估算，再实测校准，最后用于优化优先级判断",
        "五层共用同一套参数：算子、特性、布局任何一处变化，都能一路传到 SLO 与单机容量",
    )
    add_text(slide, MX, 1.8, CW, 0.26, [runs([
        ("算子级", 10, AGENT, True, MONO),
        ("　　越往右越接近业务指标，也越依赖实测校准　　", 10, G500),
        ("模型级", 10, MODEL_LEVEL, True, MONO),
    ])], anchor=MIDDLE)
    add_line(slide, MX, 2.14, SW - MX, 2.14, G300, 1.0, tail=True)

    gap = 0.18
    box_w = (CW - gap * (len(CHAIN) - 1)) / len(CHAIN)
    top, box_h = 2.3, 1.82
    for index, (tag, title, lines, model_level) in enumerate(CHAIN):
        x = MX + index * (box_w + gap)
        color = MODEL_LEVEL if model_level else AGENT
        add_box(slide, x, top, box_w, box_h, fill=MODEL_TINT if model_level else TINT)
        add_text(slide, x + 0.2, top + 0.14, box_w - 0.4, 0.26,
                 [P(tag, 9.5, color, True, MONO)], anchor=MIDDLE)
        add_text(slide, x + 0.2, top + 0.42, box_w - 0.4, 0.3,
                 [P(title, 13.5, G900, True)], anchor=MIDDLE)
        add_text(slide, x + 0.2, top + 0.78, box_w - 0.4, box_h - 0.92,
                 [P(line, 10, G700, after=2) for line in lines])
        if index:
            add_line(slide, x - gap + 0.02, top + box_h / 2, x - 0.02, top + box_h / 2,
                     G400, 1.25, tail=True)

    hub_x = MX + box_w / 2
    floor = top + box_h
    add_path_arrow(slide, [
        (MX + 4 * (box_w + gap) + box_w / 2, floor), (MX + 4 * (box_w + gap) + box_w / 2, floor + 0.3),
        (hub_x, floor + 0.3), (hub_x, floor + 0.04),
    ], MODEL_LEVEL, 1.25)
    add_text(slide, MX, floor + 0.34, CW, 0.28,
             [P("实测校准：用服务日志与压测校准预测偏差，再重新选择优化方向", 10.5, G700,
                align=CENTER)], anchor=MIDDLE)

    add_text(slide, MX, 4.96, CW, 0.32, [runs([
        ("Agent 如何使用这条链", 14, G900, True),
        ("　三件事进度不同，页面上如实标注", 10.5, G500),
    ])], anchor=MIDDLE)
    use_w = (CW - 2 * 0.24) / 3
    for index, (name, status, first, second) in enumerate(USES):
        x = MX + index * (use_w + 0.24)
        done = status == "已接入"
        add_box(slide, x, 5.36, use_w, 1.16, fill=G050 if done else WHITE,
                line=None if done else G300, dash=None if done else DASH)
        add_text(slide, x + 0.22, 5.46, use_w - 1.6, 0.3, [P(name, 12.5, G900, True)],
                 anchor=MIDDLE)
        pill = add_box(slide, x + use_w - 1.42, 5.48, 1.2, 0.26,
                       fill=AGENT if done else None, line=None if done else G400,
                       kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.4)
        boxed_text(pill, [P(status, 9, WHITE if done else G500, True, align=CENTER)],
                   inset=(0.02, 0, 0.02, 0))
        add_text(slide, x + 0.22, 5.82, use_w - 0.44, 0.56,
                 [P(first, 10.5, G700, after=2), P(second, 10.5, G500)])
    page_footer(slide, 9, "换算链只用于排优先级；任何进入交付的收益，都以整模型实测为准")


# ------------------------------------------------------------------------------- 9 where we are
# The deck opened with a proposition and then spent seven pages on mechanism; without this
# page it ends on page eight's list of unfinished items, which reads as a weak close rather
# than as a plan. Three columns, honest about which is which.
LEDGER = (
    ("已经自动跑", "第 4 - 7 页", "done", (
        "基线建立与全量评测",
        "候选迭代与三道条件验收",
        "Trace 采集与热点定位",
        "失败证据与逐轮回放",
    ), "人不在回路里，跑完才看结果"),
    ("仍然靠人", "第 4 页两端", "human", (
        "定目标、冻结验收口径",
        "新架构的接入判断",
        "未知缺陷的根因定位",
        "交付与否的最终决定",
    ), "这几项短期内不打算交出去"),
    ("下一步补齐", "第 8 - 9 页", "todo", (
        "换算链在 M100 上校准",
        "模型层与算子层自动交接",
        "规则与补丁沉淀为可复用知识",
        "适配成本的量化对账",
    ), "按这个顺序做，先校准再谈预测"),
)


def closing_slide(slide):
    page_header(
        slide,
        "收尾 · 现状与下一步",
        "回到第 1 页的命题：适配成本降下来了多少",
        "如实对账——哪些已经自动跑，哪些仍然靠人，哪些还没做",
    )
    gap = 0.3
    col_w = (CW - 2 * gap) / 3
    y, h = 2.0, 3.5
    for index, (name, ref, kind, items, note) in enumerate(LEDGER):
        x = MX + index * (col_w + gap)
        done, todo = kind == "done", kind == "todo"
        add_box(slide, x, y, col_w, h, fill=WHITE if todo else (TINT if done else G050),
                line=G300 if todo else None, dash=DASH if todo else None)
        if not todo:
            add_box(slide, x, y, col_w, 0.05, fill=AGENT if done else G400)
        add_text(slide, x + 0.24, y + 0.2, col_w - 0.48, 0.36,
                 [runs([(name, 15, G900, True)])], anchor=MIDDLE)
        add_text(slide, x + 0.24, y + 0.58, col_w - 0.48, 0.26,
                 [P(ref, 9.5, AGENT if done else G500, False, MONO)], anchor=MIDDLE)
        bullets(slide, x + 0.24, y + 0.98, col_w - 0.48, items, size=11, pitch=0.5,
                color=G900 if done else G700)
        add_text(slide, x + 0.24, y + h - 0.5, col_w - 0.48, 0.34,
                 [P(note, 10, G500)], anchor=MIDDLE)

    band(slide, 5.72, 0.62, "命题", "把重复判断交给 Agent，一个模型一个模型往下压")
    page_footer(slide, 10, "三列按当前进度如实分栏；左列已在 M100 上跑通，右列尚未开始")


# ------------------------------------------------------------------------------------ speech
SPEECH = {
    1: ((
        "先把成本摆清楚。左边是模型侧的节奏：主流模型持续更新，结构与精度不断变化，"
        "社区从发布到跑通是按天算的。"
    ), (
        "右边是我们这一侧的工作，三件事都要反复做：接入，把模型从启动带到精度、性能达标；"
        "补齐，框架与算子的缺口逐项补上；复验，每改一次就要重新验证一遍。"
    ), (
        "这三件里最贵的不是算力，是第三件——重复判断。同样的问题换个模型再判一遍，经验沉不下来。"
    ), (
        "中间这句是要害：芯片竞争力取决于硬件能力、软件成熟度与模型覆盖度。"
        "硬件能力在流片那天就定型了，模型覆盖度每天都在被社区改写，"
        "所以适配效率直接决定硬件能力能不能及时发挥出来。"
    ), (
        "命题因此只有一个：要扩大模型覆盖，只能把单个模型的适配成本降下来，"
        "把重复的排查、评测与调优组织成可复用的 Agent 流程。后面七页回答这件事怎么做。"
    )),
    2: ((
        "这一页说清楚 Agent 站在软件栈的哪一层。"
    ), (
        "从下往上，芯片、运行时与驱动、编译器，这几层昆仑芯已经做好了，我们按原样使用、一行不改，"
        "所以合并成一条。"
    ), (
        "往上两层才是动手的地方，页面上框出来了。上面一框是框架层：SGLang 是主线，"
        "XInfer 是昆仑芯自己的推理框架，这一层做模型接入、服务验证和按实测调优，第 4、5 页展开。"
    ), (
        "下面一框是算子层：XCN_ATEN_OPS。这一层补齐算子缺口、生成与调优实现、再用真机验证收益，"
        "第 6、7 页展开。"
    ), (
        "实心的三个模块是 Agent 直接参与的，其余由昆仑芯或社区提供，我们只做对接。"
    )),
    3: ((
        "两个 Agent 先给一个全貌。左边 model-agent 工作在模型层，做模型接入、服务验证与性能调优；"
        "右边 xpugen 工作在算子层，面向 M100 生成算子实现并调优。"
    ), (
        "两边的结构是一样的：都是先确认精度正确，再确认收益，中间那三个方块就是各自的验收链路，"
        "写的都是通过条件。模型只负责提出方案，是否合格由固定验收与真机测量决定，不由模型自己解释。"
        "两边也都有一个不作判据的诊断步骤——model-agent 独立采 Trace，xpugen 用模拟器定位瓶颈。"
    ), (
        "真正的区别是加了底色的那一行，测量范围。model-agent 测的是整个模型、真实服务与固定评测集；"
        "xpugen 测的是单算子，多形状、对照独立参考。"
    ), (
        "中间两条箭头是交接。向右：由 Trace 定位到的瓶颈算子，转成算子层的输入规格与请求。"
        "向左：新内核回来，进入模型层的候选迭代，重新判一次。"
    ), (
        "最下面这句是整页的要点：同一个算子内核，既要通过算子验收，也要通过整模型复验。"
        "算子 bench 衡量的是局部收益，能不能交付由真实模型上的复测决定。"
        "第 8 页会用一个算例说明这两者能差多少。"
    )),
    4: ((
        "这一页讲模型层的一次任务是怎么跑完的：人定目标，Agent 跑完四个阶段，结果交回人验收。"
    ), (
        "顶上这一行先交代范围。三条命令线共用同一套契约——架构接入、性能调优、持续优化；"
        "今天讲中间这条。"
    ), (
        "左边是人的第一段工作：确定模型、负载与并发，精度阈值与预算，已知缺陷对应的修复方案，"
        "以及输入与任务约定。这一步定完就冻结——契约改动之后，已有结果不再沿用。"
    ), (
        "中间整段无人值守，四个阶段，每个阶段都有明确产出，这是看懂这条链的关键。"
        "A 先完成预检再跑全量评测，产出基线指标和一套可复现的启动方式；"
        "B 用独立进程采 Trace，不参与计时，产出热点清单，只用来选方向，不计入任何收益；"
        "C 是候选迭代，小步改动、逐个按三道条件验收，产出通过筛选的候选和各自收益，下一页整页展开；"
        "D 最终复验，重测基线再独立复验每个候选，产出可交付的那一个。"
    ), (
        "D 为什么不能省：C 阶段的收益是在筛选过程中测出来的，基线和候选分处不同时间段。"
        "重新测一遍基线、再独立复验一次，数才算数。"
    ), (
        "右边是人的第二段工作：核对精度、性能与失败记录，通过就导出启动方案，"
        "失败或没有收益转人工诊断。每一轮都能追溯，必要时回退。"
    ), (
        "最下面这条回环是外循环：把验证过的规则、补丁与失败记录沉淀下来，供下一次任务复用。"
        "这正是第 1 页那个命题——让重复判断逐次减少。"
    )),
    5: ((
        "上一页 C 阶段里的每一个候选，都要走完这一页。"
        "但这一页真正回答的是另一个问题：Agent 报出来的收益，凭什么可信。"
    ), (
        "先准备：在隔离副本上应用改动，构建并记录产物信息。"
        "服务启动时核对端口、设备与模型路径，确认被测的确实是这个候选。"
    ), (
        "然后三道条件，顺序按成本由低到高。第一道看输出有没有变，判据是贪心 token 完全一致；"
        "第二道看质量，正确率与解析率都不许降；第三道看是不是确实更快，"
        "收益要大于 3%、回归不超过 5%，重复压测取中位，还要大于噪声极差。"
        "任一道不过就停止后续测试，不再占用设备时间，但失败证据保留。"
        "为什么按这个顺序——越靠前的越便宜，先用几秒钟的对比挡掉大部分不合格的候选。"
    ), (
        "下面三条是这一页的落点，三条防止自欺的约束。"
        "第一，口径先冻结：模型、负载、阈值与预算随任务写定，改了之后已有结果不再沿用。"
        "第二，测的是同一个候选：隔离副本、记录产物身份、启动时核对，不会出现测 A 报 B。"
        "第三，不能靠少干活变快：输入、精度、上下文与缓存全部冻结，"
        "prompt 的 token 数必须等于冻结值，诊断和正式压测分开跑。"
    ), (
        "右边是这条流水线在 30B-A3B 上判出来的实际结果。五组候选送进去："
        "第一道挡掉一个模式——分组关掉之后 cos 直接是 0，算出来就是错的；"
        "第二道这一轮没拦下任何一个，因为过了第一道的候选输出逐位一致，不必再看质量；"
        "第三道挡掉三组——强制不分组慢 20%，换现成核只有 0.61 倍更慢，"
        "路由融合算到头也只值 0.9%，够不着 3% 的门槛。"
        "三道全过的两组交付，峰值吞吐从 1370 涨到 1883。"
        "最后一条口径：只有独立复验通过的收益才进入最终交付，"
        "筛选阶段看起来有收益、复验没过的，一个都不算。"
    )),
    6: ((
        "这一页是算子层：xpugen 怎么写算子、怎么判算子。"
    ), (
        "左边写算子，是全流程唯一的模型。生成之前先澄清需求与约束、比较方案、固定输入语义，"
        "然后依据硬件说明卡生成实现，明确数据搬运、计算与同步。"
    ), (
        "右边判算子，是固定规则，这一侧没有模型。L0 编译与运行，L1 多形状多精度对照独立参考，"
        "Bench 在固定负载上真机计时，Profile 用模拟器定位瓶颈——注意 Profile 不作判据。"
        "评测结果与瓶颈证据原样回传，进入下一轮。"
    ), (
        "下面一行是架构要点：写与判都不碰硬件，与硬件耦合的只有后端这一层，"
        "编译、运行、性能采集。换新芯片只需补齐这一层。"
    ), (
        "右边是目前的实测样例，batch_norm：固定 256 乘 4096 的负载，fp32，"
        "相对 batch_norm_v4 基线加速 1.37 倍，13.24 微秒降到 9.66。"
        "这是单算子样例，还没有规模化统计，所以这里不报聚合指标。"
    ), (
        "下一页用 batch_norm 给一次完整的迭代记录，包括被丢弃的那两轮。"
    )),
    7: ((
        "上一页讲 xpugen 怎么写、怎么判，这一页给一次完整的记录：batch_norm，从基线到交付。"
    ), (
        "工作负载先说清楚：N=1、C=256、HW=4096，fp32，一百零四万八千个元素，"
        "读写各 4 MB，总流量 8.39 MB。按带宽上限算，roofline 是 7.05 微秒。"
    ), (
        "先看第 1 轮，它是这一页的关键。这一轮没有改任何实现，只做了一件事："
        "测一个纯拷贝 kernel 搬同样多的字节要多久。答案是 9.83 微秒，achieved 0.717。"
        "也就是说，roofline 算出来 7.05，但这张卡上光把数据搬一遍就要 9.83——"
        "真实的天花板是 0.717，不是 1.0。先把天花板标出来，后面每一轮才知道还剩多少空间。"
    ), (
        "右边是五轮改动，表上都有，我只点两处。"
        "第 3 轮想用 shared 广播系数，加了同步之后反而更慢，15.06 微秒、0.468，当场丢弃；"
        "第 4 轮想把系数挪到 host 侧折好，但每次调用都要把系数 H2D 上传一遍，"
        "连上传一起算 17.60 微秒、0.401，是八轮里最差的一个，也丢弃。"
    ), (
        "真正有效的是第 2、5、6 轮：换原生 rsqrt 指令到 0.601，修 copy 路径的寄存器 spill 到 0.719，"
        "再把折叠做成 scale 乘 x 加 shift 持平交付。第 7、8 轮是补脚手架和修 ABI，没有性能变化。"
        "交付前在空闲卡上复测两百次，9.66 微秒，achieved 0.730。"
    ), (
        "两件事值得强调。第一，八轮里只有三轮带来收益，两轮被真机测量当场否掉——"
        "这和第 5 页模型层是同一套做法：判定不听模型解释，只看测量。"
        "第二，现在已经打在拷贝地板上，再怎么优化计算都不会有收益，要再快只能换访存方式。"
    ), (
        "最下面一条是同一个实现换 shape 的结果。256 乘 4096 能到 0.724，"
        "但 2 乘 1600 只有 0.143——固定 launch 开销大约 1.95 微秒，小 shape 整个被它吃掉。"
        "所以算子的 achieved 必须连着 shape 一起报，单拎一个数没有意义。"
    )),
    8: ((
        "刚才那条 batch_norm 曲线最后停在 0.730。但那是算子自己的口径——"
        "放到整个模型里值多少，这一页用一个算例说明。"
    ), (
        "先说清楚这是假设算例：固定总耗时 100 毫秒，其他部分不变，"
        "既不是模型实测，也不是 SLO 容量预测。"
    ), (
        "两个场景，区别只在算子占比，而占比是由输入形状决定的。"
        "场景 A 是短输入，GEMM 占 80%、Attention 占 10%；"
        "场景 B 是长上下文，Attention 涨到 40%，GEMM 降到 50%。"
    ), (
        "现在同样把一类算子的耗时降低 30%。场景 A 里优化 Attention，整步只降 3%；"
        "优化 GEMM 降 24%——同样的算子提速幅度，对整步的贡献差了八倍。"
        "换到场景 B，两者变成 12% 和 15%，差距几乎没有了。"
    ), (
        "右边三条是为什么要回到模型看收益。第一，占比决定直接收益，占比小的算子再快也带不动整步。"
        "第二，整步耗时还不是服务容量，排队、批次、缓存和时延目标都会改变最终结果——"
        "下一页给出完整的换算办法。"
        "第三，每轮优化之后瓶颈可能转移，下一轮要按新的 Trace 重排优先级。"
    ), (
        "结论就是最下面这句：先按负载判断优化优先级，再用整模型压测确认最终收益。"
    )),
    9: ((
        "上一页留了个问题：怎么系统地判断该先优化哪个算子。"
        "这一页就是那个办法——从算子耗时一路算到单机容量。"
    ), (
        "五层从左到右，越往右越接近业务指标，也越依赖实测校准。"
        "算子耗时取计算时间和访存时间的较大者，再加启动与通信开销；"
        "单步耗时按真实 batch 形状汇总，区分 prefill 和 decode；"
        "请求时延要模拟请求与批次、考虑缓存准入；"
        "再按业务目标判断 TTFT、TPOT 的达标率，看哪个环节先超标；"
        "最后搜索满足 SLO 的最大请求到达率，也就是单机容量。"
    ), (
        "下面这条回环是关键：用服务日志与压测校准预测偏差，再重新选择优化方向。"
        "没有校准，这条链只能排序，不能报数。"
    ), (
        "Agent 目前怎么用它，三件事进度不同，页面上如实标注了。"
        "定位热点已经接入，按真实 Trace 找热点、在固定负载上验证候选。"
        "What-if 推演等 M100 校准，校准之后用来筛选优化方向。"
        "向算子层提需求这条还待补齐，要形成 shape 与实现需求，新内核回接后再做模型复验。"
    ), (
        "一句话过渡：这条链只用于排优先级；任何进入交付的收益，都以整模型实测为准。"
    )),
    10: ((
        "最后回到第 1 页的命题，如实对一次账。"
    ), (
        "左边这列已经能自动跑：基线建立与全量评测、候选迭代与三道条件验收、"
        "Trace 采集与热点定位、失败证据与逐轮回放。这几件人不在回路里，跑完才看结果。"
    ), (
        "中间这列仍然靠人：定目标和冻结验收口径、新架构的接入判断、未知缺陷的根因定位，"
        "以及交付与否的最终决定。这几项短期内不打算交出去。"
    ), (
        "右边这列还没做：换算链在 M100 上校准、模型层与算子层的自动交接、"
        "规则与补丁沉淀为可复用知识，以及适配成本本身的量化对账。"
        "顺序是定的——先校准再谈预测，没有校准的预测只能排序，不能报数。"
    ), (
        "命题没有变：把重复判断交给 Agent，一个模型一个模型往下压。"
    )),
}
BACKUP = {
    1: ((
        "三个待填的数，口径是：人周按 bring-up 到精度与性能双达标统计；"
        "算子缺口按该模型新增或需要重写的算子实现计；复验轮次按一次完整的改—测—判算一轮。"
    ), (
        "未调优内核与最优实现相差一个数量级，这是适配不能跳过的原因，也是算子层那条线要解决的问题。"
    )),
    2: ((
        "为什么不动下面几层：编译器、运行时、驱动是昆仑芯的交付物，改了既不好升级也不好交接；"
        "Agent 只通过它们暴露的接口拿编译、运行、性能和理论上限。"
    ), (
        "vLLM 与 FlashInfer 画在图上只是标清楚位置，当前投入都在 SGLang 这条线上。"
    )),
    3: ((
        "两个 Agent 分属不同仓库，本页画的是职责与交接关系；"
        "交接目前走到哪一步——人工传递需求与内核，还是已有接口——按实际情况回答。"
    ), (
        "机制上算子层的产出可以直接作为模型层的候选：候选补丁允许的后缀里包含 .xpu，"
        "不需要额外的集成改造。"
    ), (
        "两层提速之间的落差取决于该算子在端到端耗时中的占比，第 8 页有算例。"
        "没有占比数据时，单算子的倍数不能外推为模型加速比。"
    )),
    4: ((
        "预算与预留：总预算和最终复验预留都写在任务契约里。"
        "最终复验要对每个候选重跑一遍完整评测，预留偏小会导致 D 阶段做不完，"
        "建议按候选数量与实测单轮耗时自适应，而不是写死一个常数。"
    ), (
        "恢复运行不重置截止时间，所以临近结束被中断再恢复，D 阶段可能拿不到足额预留，"
        "必要时应显式延长并记入状态。"
    ), (
        "已验收的候选不会成为新基线，后续候选始终从基线分叉，因此多个独立收益不会叠加——"
        "一次运行交付的是其中最好的一个，而不是它们的组合。"
    )),
    5: ((
        "质量这一道如果取零容差，在小筛选集上一道题就是几个百分点，"
        "而且与容许一定的数值漂移自相矛盾：漂移唯一的表现形式就是临界题翻转。"
        "更稳妥的做法是同一批题目上的配对检验。"
    ), (
        "重复三次取极差只能作为粗略估计，而且基线与候选分处不同时间段，设备漂移会直接进入对比。"
        "要让数字经得起追问，需要在同一窗口内交错测量 A / B / A。"
    ), (
        "第一道比较的是候选与 M100 基线，而不是参考实现，因此基线自身的数值错误发现不了。"
        "架构接入阶段已有冻结的独立参考，性能调优阶段应该在 A 阶段继承一次。"
    )),
    6: ((
        "1.37 倍的基线是 batch_norm_v4 的 13.24 微秒，不是现有库实现，也不是未调优的草稿，"
        "被问到时按这个口径回答。"
    ), (
        "9.66 微秒是在空闲卡上复测 200 次的结果。这能说明没有明显抖动，"
        "但不同卡况之间仍有差异，被追问时按实际测量条件回答，不要说成已验证稳定。"
    ), (
        "轮数口径：精度收敛默认八轮；性能调优不设轮数上限，"
        "铁律是连续若干轮没有进展才允许下停滞结论。"
    )),
    7: ((
        "roofline 的口径：按带宽上限算，不含发射槽约束，也不含约 1.95 微秒的固定开销。"
        "所以它是上界，不是可达目标——这正是第 1 轮要去标定纯拷贝地板的原因。"
    ), (
        "交付复测的 9.66 微秒，略快于第 1 轮地板探针测到的 9.79 到 9.88。"
        "两者不是同一次测量，卡的状态不同，应当理解为交付已经打在拷贝地板上，"
        "不要解读成超过了地板。"
    ), (
        "第 4 轮的 0.401 是按 17.60 微秒的整体耗时算的，里面含每次调用的 H2D 上传。"
        "kernel 本身没有变慢，是这次上传把整体拖垮了，所以丢弃的理由是整体变慢。"
    ), (
        "这是单算子的记录。它能说明迭代机制有效，不能说明模型级收益——"
        "0.532 到 0.730 换到整模型值多少，取决于 batch_norm 在端到端耗时里的占比，下一页展开。"
    )),
    8: ((
        "四个数都是按占比直接换算：整步降幅等于该算子占比乘以提速幅度。"
        "场景 A 的 GEMM 是 80% × 30% = 24%，Attention 是 10% × 30% = 3%。"
    ), (
        "这里只算了单步耗时。真实服务里 TTFT 包含排队，提速会被放大；"
        "而 decode 阶段受显存带宽约束，提速可能被吃掉。所以最终仍要压测。"
    ), (
        "占比本身随输入形状变化：长上下文下 attention 的占比会显著上升，"
        "所以优先级要按目标负载重新算，不能套用别的场景的结论。"
    )),
    9: ((
        "这条链目前的预测偏差还没有在 M100 上标定，所以 What-if 那一格标的是待校准。"
        "在标定之前，它只用来比较候选的相对优劣，不对外报绝对数字。"
    ), (
        "容量定义是满足 SLO 的最大请求到达率，依赖具体的 TTFT / TPOT 目标；"
        "换一组业务目标，排序可能变化。"
    ), (
        "这条链按单实例算容量。多机与 PD 分离下还要叠加跨机通信与路由，不在本页范围内。"
    )),
    10: ((
        "左列那四项已经在 M100 上跑通，但样本是单模型；"
        "规模化之后哪些规则需要调整，现在没有数据，不要说成已经验证。"
    ), (
        "适配成本的量化对账还没做。第 1 页那三个待填的数就是对账口径——"
        "人周、算子缺口数、复验轮次。被问到省了多少成本时，如实说这个数还没有。"
    )),
}
PAGES = (
    (cost_slide, "适配成本"),
    (stack_slide, "软件栈"),
    (overview_slide, "两个 Agent 纵览"),
    (iteration_slide, "模型层 · 一次任务怎么跑完"),
    (gate_slide, "模型层 · 候选验收与可信度"),
    (operator_slide, "算子层 · 生成与迭代"),
    (record_slide, "算子层 · batch_norm 迭代记录"),
    (gain_slide, "回到模型层看收益"),
    (capacity_slide, "算子耗时到服务容量"),
    (closing_slide, "现状与下一步"),
)


def notes_text(page, label):
    speech, backup = SPEECH[page], BACKUP[page]
    spoken = rounded(seconds("".join(speech)))
    head = f"【{page} / {TOTAL} · {label} · 约 {spoken} 秒（{clock(spoken)}）】"
    return "\n".join((head, *speech, "被问到再说 ——", *backup)), spoken


def add_notes(slide, text, speech_count):
    frame = slide.notes_slide.notes_text_frame
    frame.text = text
    backup_from = speech_count + 1
    for index, paragraph in enumerate(frame.paragraphs):
        paragraph.space_after = Pt(6)
        for run in paragraph.runs:
            muted = index == 0 or index >= backup_from
            style_run(run, 12, G500 if muted else G900, bold=index in (0, backup_from))


COVER_NOTE = (
    "封面停留十秒左右。一句话开场：今天讲的是怎么把模型适配里那些重复做的判断，"
    "组织成两个能自动跑、而且结果可验收的 Agent。右边四格是接下来的顺序。"
)


def build(path):
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    timings = []
    cover = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(cover)
    cover_slide(cover)
    add_notes(cover, f"【封面】\n{COVER_NOTE}", 1)
    for page, (make, label) in enumerate(PAGES, start=1):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_background(slide)
        make(slide)
        text, spoken = notes_text(page, label)
        add_notes(slide, text, len(SPEECH[page]))
        timings.append((page, label, spoken))
    prs.save(path)
    return timings


def markdown(timings):
    total = sum(spoken for _, _, spoken in timings)
    lines = [
        f"# M100 Agent 讲稿（封面 + {TOTAL} 页 · 1024 程序员节模板）",
        "",
        (f"全程约 {total // 60} 分 {total % 60} 秒，另加封面约 10 秒。"
         "每页的「被问到再说」是备用问答，不在正常讲述中使用。"),
        "",
        "| 页 | 内容 | 时长 |",
        "| --- | --- | --- |",
        "| 封面 | 开场 | 0:10 |",
    ]
    lines += [f"| {page} | {label} | {clock(spoken)} |" for page, label, spoken in timings]
    lines += [f"| — | **合计** | **{clock(total)}** |", ""]
    lines += ["## 封面", "", "*约 10 秒*", "", COVER_NOTE + "\n"]
    for page, label, spoken in timings:
        lines += [f"## {page} / {TOTAL}　{label}", "", f"*约 {spoken} 秒（{clock(spoken)}）*", ""]
        lines += [paragraph + "\n" for paragraph in SPEECH[page]]
        lines += ["**被问到再说**", ""]
        lines += ["- " + paragraph for paragraph in BACKUP[page]]
        lines.append("")
    lines += [
        "## 超时时可以压缩的地方",
        "",
        "- 第 2 页软件栈：只讲两个框和实心模块的含义，底层一句带过，可省约 25 秒。",
        "- 第 6 页算子层：写与判的细分步骤不必逐条念，指图即可，可省约 20 秒。",
        "- 第 9 页换算链：五层只讲两端和校准回环，中间三层略过，可省约 30 秒。",
        "- 第 10 页对账：三列各念一句标题加最后那句命题即可，可省约 40 秒。",
        "",
        "## 大概率会被问到的问题",
        "",
        "- 两个 Agent 之间的交接自动化了吗？——见第 3 页备用问答，按实际情况回答。",
        "- 第 8 页的数是实测吗？——不是，是按占比换算的假设算例，页面上已标注。\n  第 7 页的 batch_norm 曲线才是实测。",
        "- 适配成本到底是多少人周？——第 1 页三个数待实测填入，目前只能定性回答。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "m100_agent_deck.pptx"
    result = build(target)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8") as handle:
            handle.write(markdown(result))
    grand = sum(spoken for _, _, spoken in result)
    for page, label, spoken in result:
        print(f"{page} / {TOTAL}  {label:<18} {clock(spoken)}")
    print(f"合计 {clock(grand)}")
