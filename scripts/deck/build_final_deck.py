"""Render the 13-page M100 Agent deck on the 1024 程序员节 template.

Every string on these pages is the author's own and is reproduced verbatim; only the furniture
changes. What that means in practice: the template puts a blue-to-mint gradient behind the slide
and parks its lockup in the top-right corner, so text tones are deepened off the template's own
blue until they clear 4.5:1 against both ends of the gradient, dense panels go opaque white so the
pixel grid does not read through them, and every header stays left of the lockup's keep-out box.

Colour keeps the author's own division of labour: orange wherever a model is in the loop, blue
wherever the verdict is fixed rules, grey for everything that is only scaffolding.

    python build_final_deck.py [out.pptx]
"""

import math
import sys
from pathlib import Path

from build_agent_deck import (
    CENTER,
    CW,
    DASH,
    FONT,
    LEFT,
    MIDDLE,
    MONO,
    MX,
    RIGHT,
    SH,
    SW,
    TOP,
    WHITE,
    P,
    add_arrow,
    add_box,
    add_line,
    add_path_arrow,
    add_text,
    boxed_text,
    drop_theme_style,
    rgb,
    runs,
)
from check_text_fit import char_em
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
BACKGROUND = str(HERE / "assets" / "tpl_bg.png")
OUTPUT = "../../slides/m100_agent_final_deck.pptx"

# Deepened from the template's own blue (#60A4FF) so body text clears 4.5:1 against the lightest
# and darkest parts of the gradient alike.
G900 = "0A2B45"
G800 = "123C5B"
G700 = "1B4D69"
G500 = "40708B"
G400 = "7FA4BA"
G300 = "AFCEE1"
G200 = "CFE3F0"
G100 = "DEEDFA"
G050 = "F2F8FD"

AGENT = "0F62CE"
TINT = "C9E1FC"
PALE = "E6F0FC"
ORANGE = "C2570A"
ORANGE_FILL = "D2690C"
ORANGE_TINT = "FBE6D4"
OK = "17795F"
BAD = "B4382F"

# The lockup's keep-out box, measured off the template.
LOGO_X, LOGO_Y = 11.29, 1.16
HEAD_W = LOGO_X - MX - 0.2
TOTAL = 15


# ----------------------------------------------------------------------- shared furniture
def add_background(slide):
    slide.shapes.add_picture(BACKGROUND, 0, 0, Inches(SW), Inches(SH))


def card(slide, x, y, w, h, fill=WHITE, line=None, line_w=0.75):
    """An opaque panel. Dense content needs one so the template grid does not read through it."""
    return add_box(slide, x, y, w, h, fill=fill, line=line, line_w=line_w)


def page_header(slide, title, subtitle=None, sub_size=12.5):
    add_box(slide, MX, 0.42, 0.075, 0.42, fill=AGENT)
    add_text(slide, MX + 0.24, 0.34, HEAD_W - 0.24, 0.58, [P(title, 25, G900, True)],
             anchor=MIDDLE)
    if subtitle is not None:
        add_text(slide, MX + 0.02, 0.94, HEAD_W, 0.34, [P(subtitle, sub_size, G700)],
                 anchor=MIDDLE)


def page_footer(slide, page):
    add_text(slide, SW - MX - 1.4, 7.08, 1.4, 0.28,
             [P(f"{page} / {TOTAL}", 10, G500, True, MONO, align=RIGHT)], anchor=MIDDLE)


def text_height(lines, size, width, lead=0.0, gap=0.0):
    """Estimated height of `lines` set at `size` pt inside `width` inches, plus a lead line."""
    per_line = width * 72 * 0.95
    total = lead
    for line in lines:
        ems = sum(char_em(ch, False) for ch in line) * size
        total += max(1, math.ceil(ems / per_line)) * size * 1.22 + gap
    return total / 72


def band(slide, y, h, lead, statement, note=None, tone=AGENT, fill=TINT, size=15):
    add_box(slide, MX, y, CW, h, fill=fill)
    add_box(slide, MX, y, 0.08, h, fill=tone)
    add_text(slide, MX + 0.36, y + (0.1 if note else 0.0), CW - 0.72, 0.42 if note else h,
             [runs([(f"{lead}　", 12, tone, True), (statement, size, G900, True)])],
             anchor=MIDDLE)
    if note:
        add_text(slide, MX + 0.36, y + h - 0.38, CW - 0.72, 0.32, [P(note, 11.5, G700)],
                 anchor=MIDDLE)


def dot_list(slide, x, y, w, items, size=11, pitch=0.42, color=G700, tone=ORANGE_FILL,
             bold=False):
    for index, item in enumerate(items):
        top = y + index * pitch
        add_box(slide, x, top + pitch / 2 - 0.055, 0.11, 0.11, fill=tone, kind=MSO_SHAPE.OVAL)
        add_text(slide, x + 0.26, top, w - 0.26, pitch, [P(item, size, color, bold)],
                 anchor=MIDDLE)


def polyline(slide, points, color, width=1.5, dash=None):
    """An open polyline through points (inches), no arrowhead."""
    (x0, y0), rest = points[0], points[1:]
    builder = slide.shapes.build_freeform(Inches(x0), Inches(y0))
    builder.add_line_segments([(Inches(x), Inches(y)) for x, y in rest], close=False)
    shape = builder.convert_to_shape()
    drop_theme_style(shape)
    shape.fill.background()
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width)
    if dash is not None:
        shape.line.dash_style = dash
    return shape


def pill(slide, x, y, w, h, text, fill, size=11, color=WHITE):
    shape = add_box(slide, x, y, w, h, fill=fill, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    boxed_text(shape, [P(text, size, color, True, align=CENTER)], inset=(0.04, 0, 0.04, 0))
    return shape


def metric(slide, x, y, w, h, label, value, detail, tone=G900, fill=WHITE, line=G200,
           label_tone=G500):
    card(slide, x, y, w, h, fill=fill, line=line)
    add_text(slide, x + 0.14, y + 0.03, w - 0.26, h - 0.06, [
        P(label, 9, label_tone, font=MONO),
        P(value, 14, tone, True),
        P(detail, 8, G500, font=MONO),
    ], anchor=MIDDLE)


def divider_slide(slide, title):
    add_background(slide)
    add_box(slide, SW / 2 - 0.5, 3.04, 1.0, 0.06, fill=AGENT)
    add_text(slide, MX, 3.36, CW, 0.9, [P(title, 34, G900, True, align=CENTER)], anchor=MIDDLE)


# --------------------------------------------------------------- 1  适配成本是生态关键成本
P1_TITLE = "模型适配调优，是生态建设的关键成本"
P1_SUB = "开源模型按天迭代，优化适配按月交付——生态跟进存在效率差"
P1_CARDS = (
    (0.60, ORANGE, "模型侧：高频上新", (
        ("多模型", "主流模型持续更新", "Qwen · DeepSeek · Kimi"),
        ("新架构", "模型结构与计算精度持续演进", "MoE、多模态、低比特量化"),
        ("快迭代", "开源社区实现快速演进", "推理框架与底层算子需同步跟进"),
    )),
    (7.21, AGENT, "适配侧：链路冗长", (
        ("适配", "从启动到精度、性能达标", "跨层排查，依赖工程经验"),
        ("补齐", "框架特性与算子缺口逐项补齐", "算子缺少实现或路径不匹配"),
        ("复验", "每次改动均需全流程重新验证", "人工回归重复度高，调优经验难以沉淀"),
    )),
)
P1_CLAIM = "芯片竞争力取决于硬件能力、软件成熟度与模型覆盖度"
P1_NOTE = ("硬件能力在发布那天就定型，模型覆盖度每天被社区改写——",
           "适配效率直接决定了硬件算力能否及时转化为业务价值。")
P1_BAND = ("唯", "有大幅降低单模型适配成本，才能持续扩大模型生态覆盖面",
           "把重复的排查、评测与调优工作，沉淀为可复用的自动化 Agent 流水线。")


def page_one(slide):
    add_background(slide)
    page_header(slide, P1_TITLE, P1_SUB)
    card_w, card_y, card_h = 4.92, 1.46, 3.28
    for x, tone, head, rows in P1_CARDS:
        card(slide, x, card_y, card_w, card_h, fill=WHITE, line=G200)
        add_box(slide, x, card_y, card_w, 0.055, fill=tone)
        add_text(slide, x + 0.3, card_y + 0.16, card_w - 0.5, 0.36,
                 [P(head, 13, G900, True)], anchor=MIDDLE)
        for index, (label, line, detail) in enumerate(rows):
            top = card_y + 0.66 + index * 0.84
            add_text(slide, x + 0.3, top, 1.1, 0.74, [P(label, 16, G900, True)], anchor=MIDDLE)
            add_text(slide, x + 1.46, top, card_w - 1.72, 0.74, [
                P(line, 11.5, G800, True, after=2),
                P(detail, 10, G500),
            ], anchor=MIDDLE)

    pill(slide, SW / 2 - 0.58, card_y + card_h / 2 - 0.21, 1.16, 0.42, "迭代压力", ORANGE_FILL,
         size=11.5)

    add_text(slide, MX + 0.02, 4.96, CW, 0.34, [P(P1_CLAIM, 13.5, ORANGE, True)], anchor=MIDDLE)
    add_text(slide, MX + 0.02, 5.32, CW, 0.32, [runs([
        (P1_NOTE[0], 11.5, G700), (P1_NOTE[1], 11.5, G900, True),
    ])], anchor=MIDDLE)
    band(slide, 5.86, 0.86, *P1_BAND, tone=ORANGE_FILL, fill=ORANGE_TINT)


# --------------------------------------------------------------------------- 3  软件栈
P3_TITLE = "昆仑芯 M100 软件栈"
P3_LEGEND = "Agent 直接参与"
P3_SUB = (("Agent 聚焦两大核心任务：", G900), ("模型", AGENT), ("适配调优、", G900),
          ("算子", AGENT), ("补齐提速", G900))
# (tag, label, cells, note) -- cells are (text, detail, filled)
P3_GROUPS = (
    ("模型层的工作", ("框架与", "推理库"), (
        ("SGLang", None, True), ("vLLM", None, False),
        ("FlashInfer", "推理内核库", False), ("XInfer", None, True),
    ), "模型接入 · 服务验证 · 按实测调优"),
    ("算子层的工作", ("算子库",), (
        ("XPU Math", None, False), ("XCN_ATEN_OPS", None, True),
        ("XHPC", "XBLAS / XPUDNN / XFA", False), ("XCCL", None, False),
    ), "算子缺口补齐 · 生成与调优 · 真机验证收益"),
)
P3_PLAIN = (
    (("深度学习框架",), "PyTorch", 3.14, 0.62),
    (("编译器 · 运行时", "硬件与驱动"), "XCUDA（XTDK）　·　xTriton　·　XPU Runtime & Driver　·　昆仑芯 M100",
     5.50, 0.80),
)
LABEL_X, LABEL_W = MX, 1.42
CELLS_X = LABEL_X + LABEL_W + 0.30
CELLS_W = SW - MX - CELLS_X


def stack_row_label(slide, y, h, lines):
    card(slide, LABEL_X, y, LABEL_W, h, fill=G050, line=G200)
    add_text(slide, LABEL_X, y, LABEL_W, h,
             [P(line, 10.5, G700, align=CENTER) for line in lines], anchor=MIDDLE)


def page_three(slide):
    add_background(slide)
    page_header(slide, P3_TITLE, None)
    add_text(slide, MX + 0.02, 0.94, HEAD_W, 0.34,
             [runs([(text, 12.5, tone, True) for text, tone in P3_SUB])], anchor=MIDDLE)
    add_box(slide, SW - MX - 1.62, 1.33, 0.14, 0.14, fill=ORANGE_FILL)
    add_text(slide, SW - MX - 1.40, 1.26, 1.40, 0.28, [P(P3_LEGEND, 10.5, G700)], anchor=MIDDLE)

    for index, (tag, label, cells, note) in enumerate(P3_GROUPS):
        frame_y = 1.60 + index * 2.36
        add_box(slide, CELLS_X - 0.16, frame_y, CELLS_W + 0.16, 1.46, line=ORANGE_FILL,
                line_w=1.25, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
        pill(slide, CELLS_X + 0.04, frame_y - 0.14, 1.34, 0.28, tag, ORANGE_FILL, size=9.5)
        cell_y, cell_h = frame_y + 0.20, 0.76
        stack_row_label(slide, cell_y, cell_h, label)
        gap, count = 0.18, len(cells)
        cell_w = (CELLS_W - gap * (count - 1)) / count
        for position, (text, detail, filled) in enumerate(cells):
            left = CELLS_X + position * (cell_w + gap)
            card(slide, left, cell_y, cell_w, cell_h,
                 fill=ORANGE_FILL if filled else WHITE, line=None if filled else G300)
            add_text(slide, left, cell_y, cell_w, cell_h, [
                P(text, 12.5, WHITE if filled else G900, True, align=CENTER),
                *([P(detail, 9, G500, align=CENTER)] if detail else []),
            ], anchor=MIDDLE)
        add_text(slide, CELLS_X + 0.04, cell_y + cell_h + 0.06, CELLS_W, 0.3,
                 [P(note, 10, ORANGE)], anchor=MIDDLE)

    for label, text, y, h in P3_PLAIN:
        stack_row_label(slide, y, h, label)
        card(slide, CELLS_X, y, CELLS_W, h, fill=G050, line=G200)
        add_text(slide, CELLS_X, y, CELLS_W, h, [P(text, 12, G700, align=CENTER)], anchor=MIDDLE)


# ------------------------------------------------------------------------- 4  总体方案
P4_TITLE = "M100 Agent： 总体方案（写判分离 · 双层同构 · 跨层闭环）"
P4_SUB = "两层架构同构——Agent 仅负责生成优化方案，准入判定完全交由固定规则与真机实测；核心差异在于评测粒度。"
# (x, width) for 层级定位 and the four stages
P4_GRID = ((MX, 1.70), (2.42, 1.96), (4.64, 2.92), (7.82, 3.12), (11.20, 1.52))
# (number, name, note, tone, fill)
P4_HEAD = (
    (None, "层级定位", None, G700, G050),
    ("①", "任务输入", None, G700, G050),
    ("②", "AI 生成侧（写）", "唯一引入大模型的位置", ORANGE, ORANGE_TINT),
    ("③", "确定性验证侧（判）", "纯规则驱动 · 真机实测", AGENT, PALE),
    ("④", "准入出口", None, G700, G050),
)
# (eyebrow, layer, role, cells, return label)
P4_ROWS = (
    ("model-agent", "【模型层】", "模型接入与性能调优", (
        ("人：定目标与边界", "负载 / 精度 / 预算"),
        ("Agent 生成接入、优化方案", "服务配置与图融合补丁"),
        ("整模型真机端到端验收", "输出对齐 · 质量不降 · 吞吐提升"),
        ("人", "判断交付"),
    ), "✕　不过即回退，带着原因退回"),
    ("xpugen", "【算子层】", "面向 M100 自动生成算子实现并调优，提升aten算子的覆盖泛化", (
        ("算子规格与约束", "目标 Shape 与硬件规格"),
        ("Agent 基于已有skills优化算子", "生成并调优 Kernel 代码"),
        ("Harness 单算子评测", "编译通过 · Golden 对齐 · 真机实测更快"),
        ("候选库", "交付实现"),
    ), "✕　不过即淘汰，带着证据退回"),
)
# (column the arrow sits under, upward, number, direction, detail)
P4_EXCHANGE = (
    (2, True, "②", "算子层 → 模型层", "交付新算子内核，合入候选分支重跑整模型复验"),
    (3, False, "①", "模型层 → 算子层", "Trace 定位瓶颈算子，下发 Shape 规格"),
)
P4_KEY = ("核心原则：", "算子局部加速 ≠ 整模型收益；任何算子优化必须回流通过整模型真机复验方可交付",
          "算子 bench 衡量的是局部收益；能否交付，由真实模型上的复测决定")
P4_HEAD_Y, P4_HEAD_H = 1.34, 0.52
P4_ROW_Y, P4_ROW_H = (1.92, 4.50), 1.62
P4_CELL_DY, P4_CELL_H = 0.38, 1.16
P4_BAND_Y, P4_BAND_H = 3.62, 0.80
P4_STAGE_TONES = (None, ORANGE_FILL, AGENT, None)


def p4_mid(column):
    x, w = P4_GRID[column]
    return x + w / 2


def p4_head(slide):
    for (x, w), (number, name, note, tone, fill) in zip(P4_GRID, P4_HEAD):
        card(slide, x, P4_HEAD_Y, w, P4_HEAD_H, fill=fill, line=None)
        if fill is not G050:
            add_box(slide, x, P4_HEAD_Y, w, 0.05, fill=tone)
        add_text(slide, x + 0.16, P4_HEAD_Y + 0.04, w - 0.3, P4_HEAD_H - 0.08, [
            runs([*([(f"{number} ", 11, tone, True)] if number else []),
                  (name, 11, G900, True)], align=CENTER, after=2),
            *([P(note, 9, tone, align=CENTER)] if note else []),
        ], anchor=MIDDLE)


def p4_row(slide, index):
    eyebrow, layer, role, cells, back = P4_ROWS[index]
    y = P4_ROW_Y[index]
    add_box(slide, MX, y, CW, P4_ROW_H, fill=G050)

    label_x, label_w = P4_GRID[0]
    add_box(slide, label_x, y, 0.07, P4_ROW_H, fill=ORANGE_FILL)
    add_text(slide, label_x + 0.22, y + 0.3, label_w - 0.36, P4_ROW_H - 0.6, [
        P(layer, 13, G900, True, after=3),
        P(eyebrow, 10, ORANGE, True, MONO, after=4),
        P(role, 8, G500),
    ], anchor=TOP)

    cell_y = y + P4_CELL_DY
    for column, ((title, detail), tone) in enumerate(zip(cells, P4_STAGE_TONES), start=1):
        x, w = P4_GRID[column]
        card(slide, x, cell_y, w, P4_CELL_H, fill=WHITE, line=G200)
        if tone:
            add_box(slide, x, cell_y, 0.07, P4_CELL_H, fill=tone)
        add_text(slide, x + (0.24 if tone else 0.16), cell_y + 0.06, w - 0.4, P4_CELL_H - 0.12, [
            P(title, 11, G900, True, after=4),
            P(detail, 9, G700),
        ], anchor=MIDDLE)
        if column < len(cells):
            right = P4_GRID[column + 1][0]
            add_arrow(slide, x + w + 0.05, cell_y + P4_CELL_H / 2, right - 0.05,
                      cell_y + P4_CELL_H / 2, G500, 1.5)

    arc_y = y + 0.28
    generate, judge = p4_mid(2), p4_mid(3)
    polyline(slide, [(judge, cell_y), (judge, arc_y), (generate, arc_y)], ORANGE_FILL, 1.25)
    add_arrow(slide, generate, arc_y, generate, cell_y - 0.02, ORANGE_FILL, 1.25)
    add_text(slide, (generate + judge) / 2 - 1.6, y + 0.03, 3.2, 0.24,
             [P(back, 9.5, ORANGE, align=CENTER)], anchor=MIDDLE)


def page_four(slide):
    add_background(slide)
    page_header(slide, P4_TITLE, P4_SUB, sub_size=11.5)
    p4_head(slide)
    for index in range(len(P4_ROWS)):
        p4_row(slide, index)

    add_box(slide, MX, P4_BAND_Y, CW, P4_BAND_H, fill=ORANGE_TINT)
    top, bottom = P4_BAND_Y + 0.1, P4_BAND_Y + P4_BAND_H - 0.1
    for column, upward, number, direction, detail in P4_EXCHANGE:
        arrow_x = p4_mid(column)
        add_arrow(slide, arrow_x, bottom if upward else top, arrow_x,
                  top if upward else bottom, ORANGE_FILL, 1.75)
        if upward:
            text_x, text_w, align = P4_GRID[1][0], arrow_x - 0.2 - P4_GRID[1][0], RIGHT
        else:
            text_x = arrow_x + 0.2
            text_w, align = SW - MX - text_x, LEFT
        add_text(slide, text_x, P4_BAND_Y, text_w, P4_BAND_H, [
            runs([(f"{number} ", 10.5, ORANGE, True), (direction, 10.5, G900, True)],
                 align=align, after=3),
            P(detail, 9.5, G700, align=align),
        ], anchor=MIDDLE)

    band(slide, 6.24, 0.68, *P4_KEY, tone=ORANGE_FILL, fill=ORANGE_TINT, size=13.5)


# ------------------------------------------------------------------- 5  模型层适配与优化
P5_TITLE = "M100 Agent：模型层适配与优化"
P5_SUB = "已有流程可自动执行；新架构、未知缺陷与关键决策仍需要人工参与"
P5_PHASES = (("模型适配", "新架构逐层验收"), ("性能调优", "实测环境下迭代优化"),
             ("持续优化", "按 Trace 持续迭代"))
P5_LEFT = ("人：定义目标与边界",
           ("模型 / 负载 / 并发", "精度阈值与预算", "已知缺陷与修复方案", "确定输入与任务约定"),
           "目标改动后，历史结果不再沿用", "交给 ", "Agent")
P5_STAGES = (
    ("A", "建立基线", "先完成环境预检，再验收基线", False),
    ("B", "采集 Trace", "执行性能剖析采样，仅用于瓶颈诊断", False),
    ("C", "候选迭代", "小步改动，逐轮验证收益（大型修改，人工介入）", True),
    ("D", "最终复验", "重测基线与候选，独立复核收益", False),
)
P5_RIGHT = ("人：判断交付",
            ("复核最终精度、性能收益与失败日志", "验证通过：导出标准化部署与启动方案",
             "验收失败或无收益：转交人工深度诊断", "支持逐轮溯源，必要时一键回退"),
            "交付引用已验证的隔离副本", "交回给人")
P5_BOTTOM = "将验证过的规则、补丁与失败记录沉淀下来，供下一次任务复用"


def page_five(slide):
    add_background(slide)
    page_header(slide, P5_TITLE, P5_SUB)

    strip_y, strip_h = 1.42, 0.52
    gap = 0.46
    strip_w = (CW - gap * 2) / 3
    for index, (name, detail) in enumerate(P5_PHASES):
        left = MX + index * (strip_w + gap)
        card(slide, left, strip_y, strip_w, strip_h,
             fill=ORANGE_TINT if index == 1 else G050, line=None)
        add_text(slide, left + 0.3, strip_y, strip_w - 0.5, strip_h, [runs([
            (name, 12.5, G900, True), (f"　{detail}", 10.5, G500),
        ])], anchor=MIDDLE)
        if index:
            add_arrow(slide, left - gap + 0.1, strip_y + strip_h / 2, left - 0.1,
                      strip_y + strip_h / 2, ORANGE_FILL, 1.5)

    col_y, col_h = 2.18, 3.96
    left_w, mid_w, right_w = 2.45, 5.70, 3.62
    left_x = MX
    mid_x = left_x + left_w + 0.18
    right_x = mid_x + mid_w + 0.18

    card(slide, left_x, col_y, left_w, col_h, fill=G050, line=None)
    add_box(slide, left_x, col_y, left_w, 0.07, fill=ORANGE_FILL)
    add_text(slide, left_x + 0.26, col_y + 0.2, left_w - 0.44, 0.42,
             [P(P5_LEFT[0], 14, G900, True)], anchor=MIDDLE)
    dot_list(slide, left_x + 0.3, col_y + 0.84, left_w - 0.5, P5_LEFT[1], size=11, pitch=0.5)
    add_text(slide, left_x + 0.22, col_h + col_y - 0.9, left_w - 0.44, 0.3,
             [P(P5_LEFT[2], 10, G500, align=CENTER)], anchor=MIDDLE)
    add_text(slide, left_x + 0.22, col_h + col_y - 0.56, left_w - 0.44, 0.3, [runs([
        (P5_LEFT[3], 11, ORANGE), (P5_LEFT[4], 11, ORANGE, False, MONO),
    ], align=CENTER)], anchor=MIDDLE)

    card(slide, mid_x, col_y, mid_w, col_h, fill=WHITE, line=ORANGE_FILL, line_w=1.25)
    add_text(slide, mid_x + 0.26, col_y + 0.16, mid_w - 0.5, 0.42,
             [P("Agent：在固定约束内自动迭代", 14, G900, True)], anchor=MIDDLE)
    stage_y, stage_h = col_y + 0.72, 0.62
    for index, (letter, name, detail, active) in enumerate(P5_STAGES):
        top = stage_y + index * (stage_h + 0.12)
        card(slide, mid_x + 0.22, top, mid_w - 0.44, stage_h,
             fill=ORANGE_TINT if active else G050, line=None)
        dot = add_box(slide, mid_x + 0.32, top + stage_h / 2 - 0.19, 0.38, 0.38,
                      fill=ORANGE_FILL, kind=MSO_SHAPE.OVAL)
        boxed_text(dot, [P(letter, 12, WHITE, True, align=CENTER)], inset=(0, 0, 0, 0))
        add_text(slide, mid_x + 0.78, top, 1.0, stage_h, [P(name, 11.5, G900, True)],
                 anchor=MIDDLE)
        add_text(slide, mid_x + 1.82, top, mid_w - 2.42, stage_h, [P(detail, 10, G700)],
                 anchor=MIDDLE)
    loop_x = mid_x + mid_w - 0.1
    loop_top = stage_y + 2 * (stage_h + 0.12) + 0.12
    polyline(slide, [(loop_x - 0.34, loop_top), (loop_x, loop_top), (loop_x, loop_top + 0.38),
                     (loop_x - 0.34, loop_top + 0.38)], ORANGE_FILL, 1.5)
    add_text(slide, mid_x + 0.26, col_y + col_h - 0.36, mid_w - 0.5, 0.3,
             [P("每个候选如何验收 → 下一页", 10.5, ORANGE)], anchor=MIDDLE)

    card(slide, right_x, col_y, right_w, col_h, fill=G050, line=None)
    add_box(slide, right_x, col_y, right_w, 0.07, fill=ORANGE_FILL)
    add_text(slide, right_x + 0.26, col_y + 0.2, right_w - 0.44, 0.42,
             [P(P5_RIGHT[0], 14, G900, True)], anchor=MIDDLE)
    dot_list(slide, right_x + 0.3, col_y + 0.84, right_w - 0.5, P5_RIGHT[1], size=11,
             pitch=0.5)
    add_text(slide, right_x + 0.22, col_h + col_y - 0.9, right_w - 0.44, 0.3,
             [P(P5_RIGHT[2], 10, G500, align=CENTER)], anchor=MIDDLE)
    add_text(slide, right_x + 0.22, col_h + col_y - 0.56, right_w - 0.44, 0.3,
             [P(P5_RIGHT[3], 11, ORANGE, align=CENTER)], anchor=MIDDLE)

    hand_y = col_y + col_h + 0.24
    polyline(slide, [(right_x + right_w / 2, col_y + col_h), (right_x + right_w / 2, hand_y),
                     (left_x + left_w / 2, hand_y)], ORANGE_FILL, 1.5)
    add_arrow(slide, left_x + left_w / 2, hand_y, left_x + left_w / 2, col_y + col_h + 0.02,
              ORANGE_FILL, 1.5)
    add_text(slide, MX, 6.58, CW, 0.34, [P(P5_BOTTOM, 12, G700, align=CENTER)], anchor=MIDDLE)


# ------------------------------------------------------------- 6  模型层Agent运行机制视图
P6_TITLE = "M100 Agent：模型层Agent运行机制视图"
P6_CONTRACT = ("任务基准 · 锁定评测条件与版本", (
    ("optimization-task.json", "确定模型、精度、负载、准入门槛"),
    ("generation-task.json", "设定固定文本对齐测试与评测集达标要求"),
    ("backup/", "留存初始基线备份 + 加载调优知识库"),
))
P6_TOP = (
    ("1", "性能采样（Trace）", ("拉起一次性独立服务，通过profile接口获取性能数据",)),
    ("2", "耗时归因（对齐到代码）",
     ("获取目标阶段（Prefill/Decode），将kernel耗时直接映射到模型代码层，加入累加等数值校验",)),
    ("3", "理论上限核算",
     ("解析模型配置 → 折算计算量与访存字节 → 推导理论耗时下界与差距倍数",)),
)
P6_TOP_EDGES = (("Trace", "文件"), ("各类算子", "耗时、", "调用频次"),
                ("瓶颈优先", "级列表、", "异常算子", "耗时"))
P6_PLAN = ("4", "方案与补丁", (
    ("规划器", ORANGE, (
        "看证据写代码：结合模型源码与锁定的诊断证据，提出优化假设并判断",
        "按需skills：默认只读目录索引，每轮最多按需调取 2 份专项调优指南（Skill），防上下文膨胀",
    )),
    ("人 · 按需介入", G800, (
        "复杂场景由人直接修改，或与规划器协同实现",
        "人工介入不改变准入判定路径，同样须通过下方三道验收门禁",
    )),
))
P6_HAND = ("提交规划方案：", "优化假设 + 代码补丁 + 证据引用")
P6_CONTROL = ("控制器", (
    "确定性调度机制（Python）：统一接管硬件设备锁、状态机流转、故障恢复、准入验证与报告生成",
    "硬性约束：规划器无权自行宣告实验成功；数值精度、多模态生成质量及性能收益，全部由控制器独立核验",
    "全量实验追溯：每次实验尝试（Trial）均独立落盘记录，涵盖：候选评估、构建准备、基线测试、基线复核与最终复验",
    "闭环容错机制：方案被拒原因经历史记忆库（Memory）带回下一轮请求；底层基础设施中断不计入方案优化失败",
))
P6_BOTTOM = (
    (MX, 2.70, "8", "端到端A/B对比实验",
     ("多次压测取中位数，全程关闭性能剖析器（Profiler），增幅须跨越噪声门槛",)),
    (3.58, 2.70, "7", "端到端质量评估", ("数据集得分不降，格式解析失败与输出截断均严判计入",)),
    (6.56, 2.70, "6", "单算子性能初筛",
     ("隔离被改算子，直接测速", "新旧版本同条件预热，确保对比公平",
      "局部提速达标才进入整模型评测，快速淘汰无效补丁")),
)
P6_RETURN = "确认真实收益 → 更新为最新基线 → 触发新一轮采样"
P6_MERGE = ("5", "补丁合入与增量构建", ("在源码工作区执行原位构建，打入隔离补丁",))


P6_TOP_X = (MX, 3.58, 6.56)
P6_TOP_W = 2.42
P6_TOP_Y, P6_TOP_H = 1.98, 1.30
P6_PLAN_X, P6_PLAN_W = 9.54, 3.19
P6_BOT_Y, P6_BOT_H = 5.50, 1.22


def p6_node(slide, x, y, w, h, number, title, lines, tone, size=11):
    card(slide, x, y, w, h, fill=WHITE, line=G200)
    add_box(slide, x, y, 0.07, h, fill=tone)
    add_text(slide, x + 0.2, y + 0.05, w - 0.34, h - 0.1, [
        runs([(f"{number} ", size, tone, True), (title, size, G900, True)], after=2),
        *[P(line, 10, G700) for line in lines],
    ], anchor=MIDDLE)


def page_six(slide):
    add_background(slide)
    page_header(slide, P6_TITLE, None)

    con_y, con_h = 0.98, 0.66
    add_box(slide, MX, con_y, CW, con_h, fill=WHITE, line=G300, dash=DASH)
    head, items = P6_CONTRACT
    add_text(slide, MX + 0.26, con_y + 0.05, CW - 0.5, 0.26, [P(head, 10.5, G900, True)],
             anchor=MIDDLE)
    parts = []
    for index, (code, text) in enumerate(items):
        if index:
            parts.append(("　｜　", 9, G300))
        parts += [(code, 9, G700, True, MONO), (f"　{text}", 9, G700)]
    add_text(slide, MX + 0.26, con_y + 0.32, CW - 0.5, 0.3, [runs(parts)], anchor=MIDDLE)

    top_y, top_h, top_w = P6_TOP_Y, P6_TOP_H, P6_TOP_W
    plan_x, plan_w = P6_PLAN_X, P6_PLAN_W
    for index, (number, title, lines) in enumerate(P6_TOP):
        p6_node(slide, P6_TOP_X[index], top_y, top_w, top_h, number, title, lines, OK)
    for index, label in enumerate(P6_TOP_EDGES):
        start = P6_TOP_X[index] + top_w
        end = P6_TOP_X[index + 1] if index + 1 < len(P6_TOP_X) else plan_x
        add_arrow(slide, start + 0.04, top_y + top_h / 2, end - 0.04, top_y + top_h / 2,
                  G500, 1.25)
        add_text(slide, start + 0.04, top_y + top_h / 2 + 0.05, end - start - 0.08, 0.58,
                 [P(line, 8, G500) for line in label], anchor=TOP)

    plan_blocks = []
    for name, tone, lines in P6_PLAN[2]:
        height = text_height(lines, 9.5, plan_w - 0.76, lead=11 * 1.22 + 2) + 0.2
        plan_blocks.append((name, tone, lines, height))
    plan_h = 0.40 + sum(height for *_, height in plan_blocks) + 0.1 * len(plan_blocks)

    card(slide, plan_x, top_y, plan_w, plan_h, fill=WHITE, line=ORANGE_FILL, line_w=1.5)
    add_box(slide, plan_x, top_y, 0.07, plan_h, fill=ORANGE_FILL)
    add_text(slide, plan_x + 0.24, top_y + 0.05, plan_w - 0.42, 0.3, [
        runs([(f"{P6_PLAN[0]} ", 11.5, ORANGE, True), (P6_PLAN[1], 11.5, G900, True)]),
    ], anchor=MIDDLE)
    sub_y = top_y + 0.40
    for name, tone, lines, height in plan_blocks:
        card(slide, plan_x + 0.2, sub_y, plan_w - 0.4, height, fill=G050, line=None)
        add_box(slide, plan_x + 0.2, sub_y, 0.055, height, fill=tone)
        add_text(slide, plan_x + 0.38, sub_y + 0.04, plan_w - 0.76, height - 0.08, [
            P(name, 11, tone, True, after=2),
            *[P(line, 9.5, G700) for line in lines],
        ], anchor=MIDDLE)
        sub_y += height + 0.1

    ctrl_x, ctrl_y, ctrl_w, ctrl_h = 2.78, 3.48, 6.46, 1.88
    card(slide, ctrl_x, ctrl_y, ctrl_w, ctrl_h, fill=PALE, line=None)
    add_box(slide, ctrl_x, ctrl_y, ctrl_w, 0.06, fill=AGENT)
    name, lines = P6_CONTROL
    add_text(slide, ctrl_x + 0.24, ctrl_y + 0.1, ctrl_w - 0.46, ctrl_h - 0.18, [
        P(name, 12, AGENT, True, after=2),
        *[P(line, 9.5, G700, after=2) for line in lines],
    ], anchor=TOP)

    bot_y, bot_h = P6_BOT_Y, P6_BOT_H
    for x, w, number, title, lines in P6_BOTTOM:
        p6_node(slide, x, bot_y, w, bot_h, number, title, lines, AGENT)
    p6_node(slide, plan_x, bot_y, plan_w, bot_h, *P6_MERGE, AGENT)

    hand_x = plan_x + plan_w - 0.52
    add_arrow(slide, hand_x, top_y + plan_h + 0.04, hand_x, bot_y - 0.04, G500, 1.25)
    add_text(slide, plan_x + 0.1, top_y + plan_h + 0.1, plan_w - 0.78, 0.58, [
        P(P6_HAND[0], 8, G500), P(P6_HAND[1], 8, G500),
    ], anchor=TOP)

    for target, source in ((plan_x, 6.56 + 2.70), (6.56, 3.58 + 2.70), (3.58, MX + 2.70)):
        add_arrow(slide, source + 0.04, bot_y + bot_h / 2, target - 0.04, bot_y + bot_h / 2,
                  G500, 1.25)

    ret_x = 1.74
    add_arrow(slide, ret_x, bot_y - 0.04, ret_x, top_y + top_h + 0.04, G500, 1.25)
    add_text(slide, MX + 0.02, 3.80, 1.02, 1.0, [P(P6_RETURN, 8, G500, align=RIGHT)],
             anchor=MIDDLE)


# --------------------------------------------------------------- 7  理论上限核算
P7B_TITLE = "M100 Agent： 单算子理论下界与逐层核算"
P7B_SUB = "一个算子的四条路径并行推进，耗时由最慢的一条决定；逐算子求和得到单层，再乘层数得到整模型下界"
P7B_FORM = ("单算子理论下界", "t_算子 ＝ max（ t_读, t_写, t_算, t_其它 ）",
            "四条路径在硬件上并行推进，算子耗时由最慢的一条决定，不是四项求和", (
    ("t_读", "读字节 ÷ 读带宽"),
    ("t_写", "写字节 ÷ 写带宽"),
    ("t_算", "FLOP ÷ 路径算力（cluster / sdnn × bf16 / fp8）"),
    ("t_其它", "特殊单元耗时，如 SFU"),
))
P7B_CASE = ("代入一个算子：expert gate+up GEMM",
            "max（ 读 98.28 ， 写 4.27 ， 算 7.19 ）＝ 98.28 µs",
            "读 127.76 MB 定死了下界，算力只跑到 7.3%，写和算全被读掩盖")
P7B_ROOF = ("Roofline：所有算子都落在访存区", "算术强度（FLOP / Byte）")
P7B_LEGEND = (("带宽屋顶 ＝ 算术强度 × 带宽", G500), ("fp8 算力屋顶（拐点 171）", ORANGE),
              ("bf16 算力屋顶（拐点 85）", AGENT))
P7B_RIDGE = ((85.0, 1.0, AGENT), (171.0, 2.0, ORANGE_FILL))
# (算术强度, 可达性能, 标签, 标签相对散点的纵向偏移)
P7B_DOTS = ((8.0, 0.093, "decode attention", 0.10), (12.5, 0.145, "expert GEMM ×2", 0.01),
            (31.3, 0.365, "qkv_proj / o_proj", 0.01))
P7B_ROOF_NOTE = "6 个带计算的算子强度 8–31 FLOP/Byte，另外 10 个 FLOP ＝ 0"
P7B_COLS = (("模块", 1.40, LEFT), ("路径", 0.50, CENTER), ("读 MB", 0.54, RIGHT),
            ("写 MB", 0.54, RIGHT), ("GFLOP", 0.54, RIGHT), ("t_read", 0.56, RIGHT),
            ("t_write", 0.56, RIGHT), ("t_comp", 0.54, RIGHT), ("t_其它", 0.64, RIGHT),
            ("取 max", 0.56, RIGHT), ("×48 ms", 0.58, RIGHT))
# 末位 = 决定 max 的那一列（5 = t_read，8 = t_其它）
P7B_ROWS = (
    ("input_rmsnorm", "cluster", "0.14", "0.131", "0", "0.18", "0.18", "0", "", "0.18", "0.009", 5),
    ("qkv_proj GEMM", "sdnn", "21.10", "0.328", "0.67", "16.23", "1.78", "5.99", "", "16.23",
     "0.779", 5),
    ("q_norm/k_norm", "cluster", "0.33", "0.328", "0", "0.44", "0.44", "0", "", "0.44", "0.021", 5),
    ("3D-RoPE(mrope)", "cluster", "0.33", "0.328", "0", "0.44", "0.44", "0", "", "0.44",
     "0.021", 5),
    ("store_kv(page)", "cluster", "0.07", "0.066", "0", "0.09", "0.09", "0", "", "0.09", "0.004", 5),
    ("decode attention", "sdnn", "100.93", "0.262", "0.81", "54.85", "1.42", "7.19", "32.4",
     "54.85", "2.633", 5),
    ("o_proj GEMM", "sdnn", "17.04", "0.131", "0.54", "13.11", "0.71", "4.79", "", "13.11",
     "0.629", 5),
    ("post_rmsnorm", "cluster", "0.14", "0.131", "0", "0.18", "0.18", "0", "", "0.18", "0.009", 5),
    ("router gate GEMM", "cluster", "0.66", "0.008", "0.02", "0.50", "0.04", "0.15", "", "0.50",
     "0.024", 5),
    ("topk+softmax", "cluster", "0.01", "0.001", "0", "0.01", "0.00", "0", "", "0.01", "0.001", 5),
    ("act_quant(fp8)", "cluster", "0.13", "0.066", "0", "0.18", "0.09", "0", "", "0.18", "0.009", 5),
    ("permute/shuffle", "cluster", "0.52", "0.524", "0", "0.71", "0.71", "0", "", "0.71",
     "0.034", 5),
    ("expert gate+up GEMM", "sdnn", "127.76", "0.786", "1.61", "98.28", "4.27", "7.19", "",
     "98.28", "4.717", 5),
    ("silu_and_mul(+quant)", "cluster", "0.79", "0.197", "0", "1.06", "0.27", "0", "6.6(SFU)",
     "6.55", "0.315", 8),
    ("expert down GEMM", "sdnn", "63.82", "1.049", "0.81", "49.09", "5.70", "3.60", "", "49.09",
     "2.356", 5),
    ("unpermute+combine", "cluster", "1.05", "0.131", "0", "1.42", "0.18", "0", "", "1.42",
     "0.068", 5),
)
P7B_TOTAL = ("单层合计", "242.3", "11.629")
P7B_TABLE_HEAD = "单步 Decode 逐算子理论下界（µs / 层）"
P7B_TABLE_NOTE = "橙色为决定该算子下界的那一列：16 个算子里 15 个卡在读，唯一例外是 silu_and_mul 卡在 SFU"
P7B_BAND = ("单层 242.3 µs × 48 层 ＝ 11.63 ms/步", "其中 97.7% 是把权重与 KV 读一遍的时间")

P7B_TABLE_X, P7B_TABLE_W = MX, 7.40
P7B_SIDE_X, P7B_SIDE_W = 8.14, SW - MX - 8.14


def p7b_table(slide):
    x, w = P7B_TABLE_X, P7B_TABLE_W
    y, h = 1.34, 5.04
    card(slide, x, y, w, h, fill=WHITE, line=G200)
    add_text(slide, x + 0.14, y + 0.06, w - 0.3, 0.24, [P(P7B_TABLE_HEAD, 10.5, G900, True)],
             anchor=MIDDLE)

    head_y, row_h = y + 0.32, 0.25
    left = x + 0.14
    spans = []
    for label, width, align in P7B_COLS:
        spans.append((left, width, align))
        add_text(slide, left, head_y, width - (0.06 if align == RIGHT else 0), 0.22,
                 [P(label, 8, G700, True, align=align)], anchor=MIDDLE)
        left += width
    add_line(slide, x + 0.14, head_y + 0.23, left, head_y + 0.23, G300, 0.75)

    body_y = head_y + 0.28
    for index, row in enumerate(P7B_ROWS):
        top = body_y + index * row_h
        if index % 2:
            add_box(slide, x + 0.14, top, left - x - 0.14, row_h, fill=G050)
        bound = row[-1]
        for column, ((cell_x, width, align), value) in enumerate(zip(spans, row[:-1])):
            if not value:
                continue
            winner = column == bound
            result = column >= 9
            if align == RIGHT:
                width -= 0.06
            add_text(slide, cell_x, top, width, row_h, [P(
                value, 8, ORANGE if winner else (G900 if result else G700), winner or result,
                FONT if column < 2 else MONO, align=align,
            )], anchor=MIDDLE)
    add_line(slide, spans[9][0] - 0.02, head_y, spans[9][0] - 0.02,
             body_y + len(P7B_ROWS) * row_h, G300, 0.75)

    total_y = body_y + len(P7B_ROWS) * row_h + 0.04
    add_box(slide, x + 0.14, total_y, left - x - 0.14, 0.26, fill=PALE)
    name, total_max, total_ms = P7B_TOTAL
    add_text(slide, spans[0][0] + 0.06, total_y, 2.0, 0.26, [P(name, 9, AGENT, True)],
             anchor=MIDDLE)
    for (cell_x, width, align), value in ((spans[9], total_max), (spans[10], total_ms)):
        add_text(slide, cell_x, total_y, width - 0.06, 0.26,
                 [P(value, 9, AGENT, True, MONO, align=align)], anchor=MIDDLE)
    add_text(slide, x + 0.14, y + h - 0.28, w - 0.3, 0.22, [P(P7B_TABLE_NOTE, 8, G500)],
             anchor=MIDDLE)


def p7b_formula(slide):
    x, w = P7B_SIDE_X, P7B_SIDE_W
    y, h = 1.34, 2.46
    card(slide, x, y, w, h, fill=WHITE, line=G200)
    head, formula, note, terms = P7B_FORM
    add_text(slide, x + 0.2, y + 0.06, w - 0.36, 0.22, [P(head, 10.5, G900, True)], anchor=MIDDLE)
    add_box(slide, x + 0.2, y + 0.30, w - 0.4, 0.32, fill=PALE)
    add_text(slide, x + 0.2, y + 0.30, w - 0.4, 0.32,
             [P(formula, 11, AGENT, True, MONO, align=CENTER)], anchor=MIDDLE)
    add_text(slide, x + 0.2, y + 0.64, w - 0.4, 0.20,
             [P(note, 8.5, G500, align=CENTER)], anchor=MIDDLE)
    for index, (term, meaning) in enumerate(terms):
        top = y + 0.86 + index * 0.20
        add_text(slide, x + 0.24, top, 0.52, 0.20, [P(term, 9, AGENT, True, MONO)], anchor=MIDDLE)
        add_text(slide, x + 0.80, top, w - 1.02, 0.20, [P(meaning, 8.5, G700)], anchor=MIDDLE)

    case_head, case_formula, case_note = P7B_CASE
    case_y = y + 1.70
    add_box(slide, x + 0.2, case_y, w - 0.4, 0.66, fill=ORANGE_TINT)
    add_box(slide, x + 0.2, case_y, 0.06, 0.66, fill=ORANGE_FILL)
    add_text(slide, x + 0.36, case_y + 0.02, w - 0.62, 0.20,
             [P(case_head, 8.5, ORANGE, True)], anchor=MIDDLE)
    add_text(slide, x + 0.36, case_y + 0.22, w - 0.62, 0.22,
             [P(case_formula, 9.5, G900, True, MONO)], anchor=MIDDLE)
    add_text(slide, x + 0.36, case_y + 0.44, w - 0.62, 0.20,
             [P(case_note, 8, G700)], anchor=MIDDLE)


def p7b_roofline(slide):
    x, w = P7B_SIDE_X, P7B_SIDE_W
    y, h = 3.92, 2.46
    card(slide, x, y, w, h, fill=WHITE, line=G200)
    head, x_label = P7B_ROOF
    add_text(slide, x + 0.2, y + 0.06, w - 0.36, 0.22, [P(head, 10.5, G900, True)], anchor=MIDDLE)

    px0, pw = x + 0.62, w - 0.94
    py0, ph = y + 0.42, 1.40

    def px(intensity):
        return px0 + pw * (math.log2(intensity) - 2) / 6

    def py(performance):
        return py0 + ph * (1 - (math.log2(performance) + 5) / 6)

    add_box(slide, px0, py0, px(85.0) - px0, ph, fill=G050)
    add_line(slide, px(4.0), py(4.0 / 85.0), px(171.0), py(2.0), G500, 1.5)
    for ridge, ceiling, tone in P7B_RIDGE:
        add_line(slide, px(ridge), py(ceiling), px(256.0), py(ceiling), tone, 1.25)
        add_box(slide, px(ridge) - 0.04, py(ceiling) - 0.04, 0.08, 0.08, fill=tone,
                kind=MSO_SHAPE.OVAL)
    for index, (label, tone) in enumerate(P7B_LEGEND):
        add_text(slide, px0 + 0.08, py0 + 0.02 + index * 0.18, 1.86, 0.18,
                 [P(label, 8, tone)], anchor=MIDDLE)

    add_line(slide, px0, py0 + ph, px0 + pw, py0 + ph, G400, 0.75)
    add_line(slide, px0, py0, px0, py0 + ph, G400, 0.75)
    for tick in (4, 16, 64, 256):
        add_text(slide, px(float(tick)) - 0.2, py0 + ph + 0.02, 0.4, 0.18,
                 [P(str(tick), 8, G500, font=MONO, align=CENTER)], anchor=MIDDLE)
    for tick, text in ((0.0625, "1/16"), (0.25, "1/4"), (1.0, "1"), (2.0, "2")):
        add_text(slide, px0 - 0.44, py(tick) - 0.09, 0.38, 0.18,
                 [P(text, 8, G500, font=MONO, align=RIGHT)], anchor=MIDDLE)

    for intensity, performance, label, offset in P7B_DOTS:
        dot_x, dot_y = px(intensity), py(performance)
        add_box(slide, dot_x - 0.055, dot_y - 0.055, 0.11, 0.11, fill=ORANGE_FILL,
                kind=MSO_SHAPE.OVAL)
        add_text(slide, dot_x + 0.09, dot_y + offset, 1.2, 0.18,
                 [P(label, 8, ORANGE)], anchor=MIDDLE)

    add_text(slide, x + 0.2, py0 + ph + 0.21, w - 0.36, 0.18,
             [P(x_label, 8, G500, align=CENTER)], anchor=MIDDLE)
    add_text(slide, x + 0.24, y + h - 0.3, w - 0.44, 0.24,
             [P(P7B_ROOF_NOTE, 8, G500)], anchor=MIDDLE)


def page_seven_bound(slide):
    add_background(slide)
    page_header(slide, P7B_TITLE, P7B_SUB, sub_size=11.5)
    p7b_table(slide)
    p7b_formula(slide)
    p7b_roofline(slide)
    add_box(slide, MX, 6.50, CW, 0.54, fill=ORANGE_TINT)
    add_box(slide, MX, 6.50, 0.08, 0.54, fill=ORANGE_FILL)
    add_text(slide, MX + 0.36, 6.50, CW - 0.72, 0.54, [runs([
        (P7B_BAND[0], 13.5, G900, True), ("　——　", 12, G400), (P7B_BAND[1], 12.5, G700),
    ])], anchor=MIDDLE)


# ------------------------------------------------------------------------- 7  优化记录
P7_TITLE = "M100 Agent： Qwen3-VL- 30B-A3B 优化记录"
P7_SPEC = (("使用模型：", G900, True, FONT), ("Qwen3-VL-30B-A3B-Instruct-channel-fp8", ORANGE,
                                           True, MONO),
           ("　·　M100 单卡 TP1　·　1K/1K　·　场景优化 Decode 阶段输出吞吐", G700, False, FONT))
P7_TABLE_HEAD = "各类算子耗时与理论上限（Roofline）差距：32 并发纯 Decode · 单位：ms/步 · 理论上限 = 访存字节数 ÷ 硬件带宽"
P7_COLS = ("类别", "耗时：基线 → 交付", "距理论上限", "关联优化项")
P7_LEGEND = (("上限内", "7D8894"), ("超出", ORANGE_FILL), ("优化部分", None))
# (name, within, over, saved, cost, gap, links)
P7_ROWS = (
    ("专家 GEMM（fp8）", 8.33, 0.0, 0.13, "8.46 → 8.33", "≈1×", (("3", BAD),)),
    ("Attention", 2.82, 0.0, 0.0, "2.82", "≈1×", (("—", G400),)),
    ("稠密 GEMM（QKV/O/头）", 2.15, 0.0, 0.0, "2.15", "≈1×", (("—", G400),)),
    ("专家重排与合并（含路由）", 0.12, 2.98, 0.0, "3.1", "27×",
     (("2", AGENT), (" · ", G400), ("4", BAD), (" · ", G400), ("5", AGENT))),
    ("逐元素操作 + 量化", 0.04, 1.71, 1.76, "3.51 → 1.75", "82× → 41×", (("1", AGENT),)),
    ("3D-RoPE（多模态）", 0.03, 1.84, 0.0, "1.87", "73×", (("—", G400),)),
    ("Norm 与其他小算子", 0.02, 0.98, 0.0, "~1.0", "44×", (("—", G400),)),
)
# (number, title, detail, verdict, verdict tone, highlight)
P7_LOG = (
    ("1", "融合 swiglu + 动态量化", "单算子优化后 1.75 ms/步（原 3.51）· D device 24.7 → 22.2 ms",
     "交付", AGENT, True),
    ("2", "量化 + shuffle 融合", "单算子测试快5.43倍，端到端吞吐增加3%，TPOT 减少3.4%",
     "交付", AGENT, True),
    ("3", "尝试grouped-GEMM group-mode 不同分组策略，没有明显受益", None, "放弃", BAD, False),
    ("4", "FFN 输出合并换 shuffle_mul_sum_sdnn", "劣化0.61×，更慢", "放弃", BAD, False),
    ("5", "路由 softmax + topk + renorm 融合", "tpot受益只有0.9%小于波动线，放弃",
     "收益不明显，放弃", BAD, False),
)
P7_SCALE = ("同一优化方案在不同并发下的表现", "输出 TPS：并发越高，吞吐收益越显著")
P7_POINTS = (("c1", "+9.9%", "94.9 → 104.3", False), ("c32 · Trace 工作点", "+22.5%",
                                                     "959.6 → 1175.7", False),
             ("c64", "+30.7%", "1199.9 → 1567.7", False),
             ("c96", "+35.3%", "1307.3 → 1768.8", False),
             ("c128 · 峰值", "+37.4%", "1370.0 → 1883.0", True))
P7_BAND = ("经", "Agent 多轮自动化迭代，128并发吞吐提升 37.4%")


def page_seven(slide):
    add_background(slide)
    page_header(slide, P7_TITLE, None)
    spec_y = 0.98
    add_box(slide, MX, spec_y, HEAD_W, 0.40, fill=G050)
    add_text(slide, MX + 0.24, spec_y, HEAD_W - 0.4, 0.40,
             [runs([(t, 10.5, c, b, f) for t, c, b, f in P7_SPEC])], anchor=MIDDLE)

    table_x, table_w = MX, 7.18
    table_y, table_h = 1.52, 4.14
    card(slide, table_x, table_y, table_w, table_h, fill=WHITE, line=G200)
    add_text(slide, table_x + 0.24, table_y + 0.08, table_w - 0.48, 0.6,
             [P(P7_TABLE_HEAD, 10.5, G900, True)], anchor=TOP)

    name_x, name_w = table_x + 0.24, 2.02
    bar_x, bar_w = table_x + 2.36, 1.62
    cost_x, cost_w = table_x + 4.10, 1.30
    gap_x, gap_w = table_x + 5.44, 0.82
    link_x, link_w = table_x + 6.30, 0.88
    head_y = table_y + 0.74
    add_text(slide, name_x, head_y, name_w, 0.26, [P(P7_COLS[0], 9, G500)], anchor=MIDDLE)
    legend_parts = []
    for label, tone in P7_LEGEND:
        legend_parts.append((f"{'■' if tone else '□'}{label}　", 8, tone or G400))
    add_text(slide, bar_x, head_y, bar_w + 0.2, 0.26, [runs(legend_parts)], anchor=MIDDLE)
    add_text(slide, cost_x, head_y, cost_w, 0.26, [P(P7_COLS[1], 9, G500, align=RIGHT)],
             anchor=MIDDLE)
    add_text(slide, gap_x, head_y, gap_w, 0.26, [P(P7_COLS[2], 9, G500, align=RIGHT)],
             anchor=MIDDLE)
    add_text(slide, link_x, head_y, link_w, 0.26, [P(P7_COLS[3], 9, G500, align=CENTER)],
             anchor=MIDDLE)
    add_line(slide, name_x, head_y + 0.28, table_x + table_w - 0.24, head_y + 0.28, G200, 0.75)

    scale = bar_w / 8.46
    row_y, row_h = head_y + 0.34, 0.44
    for index, (name, within, over, saved, cost, gap, links) in enumerate(P7_ROWS):
        top = row_y + index * row_h
        add_text(slide, name_x, top, name_w, row_h, [P(name, 10, G900)], anchor=MIDDLE)
        left, mid = bar_x, top + row_h / 2 - 0.09
        if within > 0:
            add_box(slide, left, mid, max(within * scale, 0.03), 0.18, fill="7D8894")
            left += max(within * scale, 0.03)
        if over > 0:
            add_box(slide, left, mid, over * scale, 0.18, fill=ORANGE_FILL)
            left += over * scale
        if saved > 0:
            add_box(slide, left, mid, saved * scale, 0.18, line=G400, dash=DASH)
        add_text(slide, cost_x, top, cost_w, row_h,
                 [P(cost, 10, G900, font=MONO, align=RIGHT)], anchor=MIDDLE)
        add_text(slide, gap_x, top, gap_w, row_h,
                 [P(gap, 10, G700, font=MONO, align=RIGHT)], anchor=MIDDLE)
        add_text(slide, link_x, top, link_w, row_h,
                 [runs([(t, 10, c, True, MONO) for t, c in links], align=CENTER)], anchor=MIDDLE)

    log_x, log_w = table_x + table_w + 0.24, SW - MX - (table_x + table_w + 0.24)
    card(slide, log_x, table_y, log_w, table_h, fill=G050, line=None)
    add_text(slide, log_x + 0.26, table_y + 0.08, log_w - 0.5, 0.34,
             [P("优化记录", 13, G900, True)], anchor=MIDDLE)
    entry_y = table_y + 0.52
    for number, title, detail, verdict, tone, highlight in P7_LOG:
        entry_h = 0.72 if detail else 0.60
        if highlight:
            add_box(slide, log_x + 0.1, entry_y, log_w - 0.2, entry_h, fill=ORANGE_TINT)
        add_text(slide, log_x + 0.24, entry_y, 0.3, entry_h,
                 [P(number, 11, AGENT, True, MONO)], anchor=MIDDLE)
        add_text(slide, log_x + 0.52, entry_y + 0.04, log_w - 2.1, entry_h - 0.08, [
            P(title, 11, G900, True, after=2),
            *([P(detail, 9.5, G500)] if detail else []),
        ], anchor=MIDDLE)
        add_text(slide, log_x + log_w - 1.5, entry_y, 1.3, entry_h,
                 [P(verdict, 10.5, tone, True, align=RIGHT)], anchor=MIDDLE)
        entry_y += entry_h + 0.05

    scale_y = 5.78
    add_text(slide, MX, scale_y, 2.44, 0.66, [
        P(P7_SCALE[0], 10.5, G900, True, after=2),
        P(P7_SCALE[1], 9.5, G500),
    ], anchor=MIDDLE)
    point_x, point_gap = 3.14, 0.14
    point_w = (SW - MX - point_x - point_gap * 4) / 5
    for index, (label, value, detail, peak) in enumerate(P7_POINTS):
        metric(slide, point_x + index * (point_w + point_gap), scale_y, point_w, 0.66,
               label, value, detail,
               tone=ORANGE if peak else G900,
               fill=ORANGE_TINT if peak else WHITE,
               line=None if peak else G200,
               label_tone=ORANGE if peak else G500)
    band(slide, 6.50, 0.52, *P7_BAND, tone=ORANGE_FILL, fill=ORANGE_TINT, size=14)


# ------------------------------------------------------------------- 8  算子层对抗生成
P8_TITLE = "M100 Agent：算子层对抗生成"
P8_SUB = "算子需求或既有实现出发，先保精度对齐，再做性能寻优；生成与评测严格解耦，评测侧完全基于确定性规则"
P8_WRITE = ("写算子", "Agent", "全流程唯一的模型", (
    ("澄清需求与约束", "比较方案，固定输入语义"),
    ("依据硬件规格生成实现", "显式规划数据搬运（DMA）、计算流水与同步机制"),
))
P8_JUDGE = ("判算子", "Harness", "规则驱动，无模型", (
    ("L0", "编译与运行", "编译构建并拉起测试，失败即打回重改"),
    ("L1", "数值验证", "多 Shape、多精度下与golden比较"),
    ("Bench", "真机计时", "固定负载，重复测量"),
    ("Profile", "瓶颈分析", "模拟器定位，不作判据"),
))
P8_SAMPLE = ("batch_norm实测样例", (
    ("固定工作负载", "256 × 4096", "FP32 总数据量8.39MB"),
    ("相对基线加速", "1.37 ×", "13.24 → 9.66 µs"),
    ("最优版本测量200轮平均", "9.66 µs", None),
), "详细迭代过程见下一页")
P8_TOP_EDGE = "交付候选实现"
P8_BOT_EDGE = "评测结果与瓶颈证据回传，进入下一轮"
P8_BAND = ("与硬件耦合的只有后端这一层：编译 · 运行 · 性能采集　", "backend/",
           "迁移至新芯片时仅需适配该后端接口，“生成与评测”主流程无需改动。")
P8_FOOT = "使用算子团队相关skills 和 算子开发优化经验;"


def page_eight(slide):
    add_background(slide)
    page_header(slide, P8_TITLE, P8_SUB, sub_size=11.5)

    col_y, col_h = 1.88, 3.66
    write_x, write_w = MX, 4.14
    judge_x, judge_w = 4.96, 5.05
    sample_x, sample_w = 10.23, SW - MX - 10.23

    card(slide, write_x, col_y, write_w, col_h, fill=ORANGE_TINT, line=None)
    name, role, note, blocks = P8_WRITE
    add_text(slide, write_x + 0.3, col_y + 0.16, write_w - 0.5, 0.4, [runs([
        (name, 15, G900, True), ("　｜　", 15, "E8C9AC"), (role, 13, ORANGE, True, MONO),
        (f"　{note}", 10.5, G500),
    ])], anchor=MIDDLE)
    for index, (head, detail) in enumerate(blocks):
        top = col_y + 0.78 + index * 1.24
        add_text(slide, write_x + 0.3, top, write_w - 0.6, 1.0, [
            P(head, 12.5, G900, True, after=3),
            P(detail, 10.5, G700),
        ], anchor=TOP)

    card(slide, judge_x, col_y, judge_w, col_h, fill=G050, line=None)
    name, role, note, levels = P8_JUDGE
    add_text(slide, judge_x + 0.3, col_y + 0.16, judge_w - 0.5, 0.4, [runs([
        (name, 15, G900, True), ("　｜　", 15, G300), (role, 13, G700, True, MONO),
        (f"　{note}", 10.5, G500),
    ])], anchor=MIDDLE)
    for index, (tag, head, detail) in enumerate(levels):
        top = col_y + 0.76 + index * 0.70
        add_text(slide, judge_x + 0.26, top, 0.72, 0.6, [P(tag, 10.5, ORANGE, True, MONO)],
                 anchor=MIDDLE)
        add_text(slide, judge_x + 0.98, top, judge_w - 1.24, 0.6, [runs([
            (head, 11.5, G900, True), (f"　{detail}", 10, G500),
        ])], anchor=MIDDLE)

    card(slide, sample_x, col_y, sample_w, col_h, fill=ORANGE_TINT, line=None)
    head, stats, foot = P8_SAMPLE
    add_text(slide, sample_x + 0.26, col_y + 0.14, sample_w - 0.5, 0.36,
             [P(head, 13, G900, True)], anchor=MIDDLE)
    stat_y = col_y + 0.62
    for label, value, detail in stats:
        stat_h = 0.84
        card(slide, sample_x + 0.2, stat_y, sample_w - 0.4, stat_h, fill=WHITE, line=G200)
        add_text(slide, sample_x + 0.4, stat_y + 0.06, sample_w - 0.8, stat_h - 0.12, [
            P(label, 10, G500, after=2),
            P(value, 17, G900, True),
            *([P(detail, 9.5, G500)] if detail else []),
        ], anchor=MIDDLE)
        stat_y += stat_h + 0.12
    add_text(slide, sample_x + 0.26, col_y + col_h - 0.36, sample_w - 0.5, 0.3,
             [P(foot, 10, ORANGE)], anchor=MIDDLE)

    write_mid, judge_mid = write_x + write_w / 2, judge_x + judge_w / 2
    top_edge = col_y - 0.26
    polyline(slide, [(write_mid, col_y), (write_mid, top_edge), (judge_mid, top_edge)],
             ORANGE_FILL, 1.5)
    add_arrow(slide, judge_mid, top_edge, judge_mid, col_y - 0.02, ORANGE_FILL, 1.5)
    add_text(slide, write_mid + 0.18, top_edge - 0.3, 4.0, 0.28,
             [P(P8_TOP_EDGE, 10, G700)], anchor=MIDDLE)
    bot_edge = col_y + col_h + 0.28
    polyline(slide, [(judge_mid, col_y + col_h), (judge_mid, bot_edge), (write_mid, bot_edge)],
             ORANGE_FILL, 1.5)
    add_arrow(slide, write_mid, bot_edge, write_mid, col_y + col_h + 0.02, ORANGE_FILL, 1.5)
    add_text(slide, write_mid + 0.18, bot_edge + 0.02, 4.6, 0.28,
             [P(P8_BOT_EDGE, 10, G700)], anchor=MIDDLE)

    band_y, band_h = 6.16, 0.70
    add_box(slide, MX, band_y, CW, band_h, fill=WHITE, line=G200)
    add_box(slide, MX, band_y, 0.08, band_h, fill=ORANGE_FILL)
    add_text(slide, MX + 0.34, band_y + 0.08, CW - 0.7, 0.32, [runs([
        (P8_BAND[0], 13, G900, True), (P8_BAND[1], 12, G500, False, MONO),
    ])], anchor=MIDDLE)
    add_text(slide, MX + 0.34, band_y + band_h - 0.36, CW - 0.7, 0.3,
             [P(P8_BAND[2], 11, G700)], anchor=MIDDLE)
    add_text(slide, MX, 6.98, 6.0, 0.3, [P(P8_FOOT, 10, G500)], anchor=MIDDLE)


# ----------------------------------------------------------------- 9  batch_norm 优化过程
P9_TITLE = "M100 Agent： batch_norm算子优化过程"
P9_SPEC = (("目标shape", G900, True, FONT),
           ("　N=1 · C=256 · HW=4096 · fp32", ORANGE, True, MONO),
           ("　1,048,576 元素，读写各 4 MB，总流量 8.39 MB　｜　", G700, False, FONT),
           ("roofline ", G700, False, FONT), ("7.049 µs", G900, True, MONO),
           ("（按纯带宽上限推算，不含指令发射约束与 Kernel Launch 固定开销）", G700, False, FONT))
P9_AXIS = "带宽达成率（Achieved）= 理论极限耗时 ÷ 实测耗时"
P9_GUIDES = (("目标 0.70 · 10.07 µs", 0.70, G400), ("拷贝地板 0.717 · 9.83 µs", 0.717, ORANGE_FILL))
P9_TICKS = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "交付")
# (slot, value, tone, where the reading goes: above, below, or nowhere)
P9_DOTS = ((0, 0.532, G500, "below"), (2, 0.601, OK, "above"), (3, 0.468, BAD, "above"),
           (4, 0.401, BAD, "above"), (5, 0.719, OK, "above"), (6, 0.719, OK, None),
           (9, 0.730, AGENT, "above"))
P9_TRACK = ((0, 0.532), (1, 0.532), (2, 0.601), (4, 0.601), (5, 0.719), (8, 0.719), (9, 0.730))
P9_KEY = (("保留", OK), ("真机实测更慢，废弃", BAD), ("交付复测", AGENT))
# (round, what, cost, verdict, tone, highlight)
P9_STEPS = (
    ("0", "batch_norm 基线", "13.24", "基线", G700, False),
    ("1", "标定纯内存拷贝耗时下限（探针）", "9.83", "探针", G500, False),
    ("2", "替换为硬件原生 rsqrt 指令", "11.72", "保留", OK, False),
    ("3", "shared 广播系数", "15.06", "丢弃", BAD, True),
    ("4", "host 折系数 + 每次 H2D（非 kernel耗时）", "17.60", "丢弃", BAD, True),
    ("5", "修 copy 的寄存器 spill", "9.81", "保留", OK, False),
    ("6", "计算折叠为 scale * x + shift（乘加融合）", "9.82", "保留", OK, False),
    ("7 · 8", "完善模拟器脚手架与修复 ABI 接口", "–", "基建", G500, False),
    ("交付", "空闲卡复测 iters=200", "9.66", "交付", AGENT, True),
)
P9_SHAPES = ("同一算子实现，泛化至不同 Shape 的实测表现",
             "固定 launch 开销约 1.95 µs，小 Shape 下 Launch 固定开销占比极高，稀释了计算优化收益。")
P9_SHAPE_POINTS = (("2 × 1600", "0.143", "0.344 → 2.407 µs", False),
                   ("32 × 6400", "0.364", "1.377 → 3.788 µs", False),
                   ("64 × 4096", "0.449", "1.762 → 3.925 µs", False),
                   ("256 × 4096", "0.724", "7.049 → 9.736 µs", True))
P9_BAND = "对比纯内存拷贝极限（9.83 µs），当前算子优化耗时（9.66 µs）已逼近硬件物理极限"

PLOT_X, PLOT_W = 1.26, 5.32
PLOT_Y, PLOT_H = 2.44, 2.30
Y_LO, Y_HI = 0.385, 0.775


def p9_px(slot):
    return PLOT_X + PLOT_W * (slot + 0.5) / len(P9_TICKS)


def p9_py(value):
    return PLOT_Y + PLOT_H * (Y_HI - value) / (Y_HI - Y_LO)


def page_nine(slide):
    add_background(slide)
    page_header(slide, P9_TITLE, None)
    spec_y = 0.98
    add_box(slide, MX, spec_y, HEAD_W, 0.46, fill=G050)
    add_text(slide, MX + 0.24, spec_y, HEAD_W - 0.4, 0.46,
             [runs([(t, 10, c, b, f) for t, c, b, f in P9_SPEC])], anchor=MIDDLE)

    chart_x, chart_w = MX, 6.50
    chart_y, chart_h = 1.56, 4.04
    card(slide, chart_x, chart_y, chart_w, chart_h, fill=WHITE, line=G200)
    add_text(slide, chart_x + 0.24, chart_y + 0.08, 3.2, 0.3, [P(P9_AXIS, 9, G700)],
             anchor=MIDDLE)
    legend_x = chart_x + 3.42
    for index, (label, value, tone) in enumerate(P9_GUIDES):
        add_text(slide, legend_x + index * 1.52, chart_y + 0.08, 1.5, 0.3,
                 [runs([("- - ", 8.5, tone), (label, 8, G500)])], anchor=MIDDLE)

    for ratio in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        y = p9_py(ratio)
        add_line(slide, PLOT_X, y, PLOT_X + PLOT_W, y, G200, 0.5)
        add_text(slide, PLOT_X - 0.54, y - 0.13, 0.46, 0.26,
                 [P(f"{ratio:.2f}", 8.5, G500, font=MONO, align=RIGHT)], anchor=MIDDLE)
    for label, value, tone in P9_GUIDES:
        y = p9_py(value)
        add_line(slide, PLOT_X, y, PLOT_X + PLOT_W, y, tone, 1.0, dash=DASH)
    add_line(slide, PLOT_X, PLOT_Y + PLOT_H, PLOT_X + PLOT_W, PLOT_Y + PLOT_H, G400, 0.75)
    for index, tick in enumerate(P9_TICKS):
        add_text(slide, p9_px(index) - 0.3, PLOT_Y + PLOT_H + 0.04, 0.6, 0.26,
                 [P(tick, 9.5, G700 if tick == "交付" else G500, tick == "交付", align=CENTER)],
                 anchor=MIDDLE)

    polyline(slide, [(p9_px(slot), p9_py(value)) for slot, value in P9_TRACK], AGENT, 2.0)
    for slot, value, tone, where in P9_DOTS:
        x, y = p9_px(slot), p9_py(value)
        add_box(slide, x - 0.085, y - 0.085, 0.17, 0.17, fill=tone, kind=MSO_SHAPE.OVAL)
        if where:
            add_text(slide, x - 0.44, y + (-0.34 if where == "above" else 0.12), 0.88, 0.24,
                     [P(f"{value:.3f}", 9.5, tone, True, font=MONO, align=CENTER)],
                     anchor=MIDDLE)
    add_text(slide, chart_x + 0.24, chart_y + chart_h - 0.36, chart_w - 0.5, 0.3, [runs([
        part for label, tone in P9_KEY
        for part in (("● ", 10, tone), (f"{label}　", 9, G700))
    ])], anchor=MIDDLE)

    steps_x = chart_x + chart_w + 0.24
    steps_w = SW - MX - steps_x
    card(slide, steps_x, chart_y, steps_w, chart_h, fill=G050, line=None)
    add_text(slide, steps_x, chart_y + 0.08, steps_w, 0.34,
             [P("八轮迭代优化", 13, G900, True, align=CENTER)], anchor=MIDDLE)
    step_y, step_h = chart_y + 0.52, 0.38
    for index, (number, what, cost, verdict, tone, highlight) in enumerate(P9_STEPS):
        top = step_y + index * step_h
        if highlight:
            add_box(slide, steps_x + 0.1, top, steps_w - 0.2, step_h,
                    fill=ORANGE_TINT if verdict == "交付" else WHITE)
        add_text(slide, steps_x + 0.2, top, 0.56, step_h,
                 [P(number, 9.5, tone, True, MONO)], anchor=MIDDLE)
        add_text(slide, steps_x + 0.8, top, steps_w - 2.1, step_h, [P(what, 9.5, G900)],
                 anchor=MIDDLE)
        add_text(slide, steps_x + steps_w - 1.26, top, 0.66, step_h,
                 [P(cost, 9.5, G900, font=MONO, align=RIGHT)], anchor=MIDDLE)
        add_text(slide, steps_x + steps_w - 0.56, top, 0.4, step_h,
                 [P(verdict, 9.5, tone, True, align=RIGHT)], anchor=MIDDLE)

    shape_y = 5.72
    add_text(slide, MX, shape_y, 3.0, 0.66, [
        P(P9_SHAPES[0], 9.5, G900, True, after=2),
        P(P9_SHAPES[1], 8.5, G500),
    ], anchor=MIDDLE)
    point_x, point_gap = 3.64, 0.16
    point_w = (SW - MX - point_x - point_gap * 3) / 4
    for index, (label, value, detail, peak) in enumerate(P9_SHAPE_POINTS):
        metric(slide, point_x + index * (point_w + point_gap), shape_y, point_w, 0.66,
               label, value, detail,
               tone=ORANGE if peak else G900,
               fill=ORANGE_TINT if peak else WHITE,
               line=None if peak else G200,
               label_tone=ORANGE if peak else G500)
    add_box(slide, MX, 6.50, CW, 0.54, fill=ORANGE_TINT)
    add_box(slide, MX, 6.50, 0.08, 0.54, fill=ORANGE_FILL)
    add_text(slide, MX + 0.36, 6.50, CW - 0.72, 0.54, [P(P9_BAND, 13.5, G900, True)],
             anchor=MIDDLE)


# ------------------------------------------------------- 11  算子层 Agent 内部组件
P11_TITLE = "M100 Agent：算子层（xpugen）内部组件与交互机制"
P11_SUB = "基于「环境感知 — 认知推理 — 工具执行」闭环，把专家调优经验（Skills）沉淀为确定性的算子实现"
# (标题, 英文, 色条)
P11_HEADS = (("任务环境与知识", "Environment & Context", G500),
             ("xpugen Agent 内部", "唯一引入大模型的位置", ORANGE_FILL),
             ("工具与评测后端", "确定性执行 · 无模型", AGENT))
# (标题, 英文, 条目, 顶, 高)
# 大括号尖端落在前两张卡片的正中，所以第二张卡片的底边由状态感知的中线决定；
# 评测反馈黑板与 Harness 同高，两者中线都对准动作执行
P11_LEFT = (
    ("任务上下文池", "Task Context", ("算子规格与 Shape", "输入输出语义约束"), 1.72, 0.80),
    ("领域知识挂载", "Domain Knowledge", ("M100 硬件规格卡", "算子调优 Skills"), 2.76, 0.80),
    ("评测反馈黑板", "Feedback Blackboard",
     ("L0 编译报错日志", "L1 精度偏差详情", "Profile 瓶颈证据"), 4.24, 1.42),
)
P11_RIGHT = (
    ("源码与检索工具", "Source & Retrieval",
     ("算子模板读取", "AST 语法树检查", "代码补丁生成"), 1.72, 1.18),
    ("Harness & 后端", "backend/ 耦合",
     ("L0 交叉编译运行", "L1 Golden 精度对齐", "Bench 真机微秒计时", "Profile 模拟器分析"),
     4.24, 1.42),
)
# (标题, 英文, 说明行, 顶, 高)
P11_STAGES = (
    ("迭代记忆库", "Memory", ("历史补丁与失败避坑",), 1.72, 0.50),
    ("状态感知", "Observation", ("汇总需求 + 报错 + 瓶颈",), 2.38, 0.52),
    ("策略推演", "Thought", ("规划 DMA 搬运 / 计算流水", "寄存器分配与指令替换"), 3.06, 1.44),
    ("动作执行", "Action", ("生成补丁，触发编译与评测",), 4.66, 0.58),
)
P11_LLM = "大模型唯一介入点"
P11_OUT = ("达标输出 · 候选库", "交付最优 Kernel 代码", 5.44, 0.48)
P11_EDGES = ("读取 / 感知", "观察", "按需调用", "触发评测", "发布达标实现")
P11_RETURN = "回写：编译报错 · 精度偏差 · 瓶颈证据"
P11_BAND = ("核心设计", "Agent 只负责消费证据做推演；编译、对精度、测速全部交由右侧确定性工具完成")

P11_LX, P11_LW = MX, 2.90
P11_CX, P11_CW_ = 4.50, 3.72
P11_RX = 9.22
P11_RW = SW - MX - P11_RX
# 中栏圆角框占容器 88% 宽并左右居中
P11_IW = P11_CW_ * 0.88
P11_IX = P11_CX + (P11_CW_ - P11_IW) / 2
P11_SPINE = P11_IX + P11_IW / 2
P11_RADIUS = 0.08
P11_HEAD_Y, P11_HEAD_H = 1.32, 0.32
P11_PANEL_TOP, P11_PANEL_BOT = 1.66, 6.02
P11_BRACE_X, P11_BRACE_W = P11_LX + P11_LW + 0.04, 0.16
# 「观察」与「按需调用」两条折线关于策略推演的中心点对称
P11_WATCH_X = P11_LX + P11_LW - 0.36
P11_CALL_X = 2 * P11_SPINE - P11_WATCH_X
P11_RETURN_X, P11_RETURN_Y = P11_LX + P11_LW / 2, 6.24


def p11_card(slide, x, y, w, h, title, english, items, tone):
    card(slide, x, y, w, h, fill=WHITE, line=G200)
    add_box(slide, x, y, 0.07, h, fill=tone)
    add_text(slide, x + 0.2, y + 0.05, w - 0.36, h - 0.1, [
        runs([(title, 10.5, G900, True), (f"　{english}", 8, tone, True, MONO)], after=3),
        *[P(f"·　{item}", 9, G700) for item in items],
    ], anchor=MIDDLE)


def p11_columns(slide):
    for x, w, (title, english, tone) in zip((P11_LX, P11_CX, P11_RX),
                                            (P11_LW, P11_CW_, P11_RW), P11_HEADS):
        add_box(slide, x, P11_HEAD_Y + 0.03, 0.06, P11_HEAD_H - 0.06, fill=tone)
        add_text(slide, x + 0.16, P11_HEAD_Y, w - 0.16, P11_HEAD_H, [
            runs([(title, 11, G900, True), (f"　{english}", 8.5, tone, True)]),
        ], anchor=MIDDLE)


def p11_tag(slide, cx, cy, text, tone, w, size=7.5):
    shape = add_box(slide, cx - w / 2, cy - 0.11, w, 0.22, fill=WHITE, line=tone,
                    kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    boxed_text(shape, [P(text, size, tone, True, align=CENTER)], inset=(0.02, 0, 0.02, 0))


def p11_stage(slide, y, h, fill, tone):
    return add_box(slide, P11_IX, y, P11_IW, h, fill=fill, line=tone,
                   kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=P11_RADIUS / min(P11_IW, h))


def p11_centre(slide):
    card(slide, P11_CX, P11_PANEL_TOP, P11_CW_, P11_PANEL_BOT - P11_PANEL_TOP,
         fill=WHITE, line=G300)
    for index, (title, english, lines, y, h) in enumerate(P11_STAGES):
        model = index == 2
        tone = ORANGE_FILL if model else AGENT
        shape = p11_stage(slide, y, h, ORANGE_TINT if model else PALE, tone)
        head = runs([(title, 11.5 if model else 11, G900, True),
                     (f"　{english}", 8.5, ORANGE if model else AGENT, True, MONO)],
                    align=CENTER, after=2)
        body = [P(line, 9 if model else 8.5, G700, align=CENTER) for line in lines]
        if model:
            # 徽章压在框的中线上，左右两条折线正好指向它
            mid = y + h / 2
            add_text(slide, P11_IX, mid - 0.50, P11_IW, 0.28, [head], anchor=MIDDLE)
            pill(slide, P11_SPINE - 0.72, mid - 0.13, 1.44, 0.26, P11_LLM, ORANGE_FILL, size=8)
            add_text(slide, P11_IX, mid + 0.20, P11_IW, 0.38, body, anchor=MIDDLE)
        else:
            boxed_text(shape, [head, *body])
        if index:
            previous = P11_STAGES[index - 1]
            add_arrow(slide, P11_SPINE, previous[3] + previous[4] + 0.01, P11_SPINE, y - 0.01,
                      AGENT, 1.5)

    title, detail, y, h = P11_OUT
    shape = p11_stage(slide, y, h, PALE, AGENT)
    boxed_text(shape, [runs([
        (title, 10.5, AGENT, True), (f"　{detail}", 8.5, G700),
    ], align=CENTER)])
    action = P11_STAGES[3]
    add_arrow(slide, P11_SPINE, action[3] + action[4] + 0.01, P11_SPINE, y - 0.01, AGENT, 1.5)
    add_text(slide, P11_SPINE + 0.1, action[3] + action[4], 1.1, y - action[3] - action[4],
             [P(P11_EDGES[4], 7.5, AGENT, True)], anchor=MIDDLE)


def page_eleven(slide):
    add_background(slide)
    page_header(slide, P11_TITLE, P11_SUB, sub_size=11.5)
    p11_columns(slide)
    for title, english, items, y, h in P11_LEFT:
        p11_card(slide, P11_LX, y, P11_LW, h, title, english, items, G500)
    for title, english, items, y, h in P11_RIGHT:
        p11_card(slide, P11_RX, y, P11_RW, h, title, english, items, AGENT)
    p11_centre(slide)

    read, observe, call, fire, _ = P11_EDGES
    centre_left, centre_right = P11_IX - 0.04, P11_IX + P11_IW + 0.04
    left_lane = (P11_BRACE_X + P11_BRACE_W + P11_CX) / 2
    right_lane = (P11_CX + P11_CW_ + P11_RX) / 2

    top, bottom = P11_LEFT[0][3], P11_LEFT[1][3] + P11_LEFT[1][4]
    add_box(slide, P11_BRACE_X, top + 0.04, P11_BRACE_W, bottom - top - 0.08, line=G500,
            line_w=1.25, kind=MSO_SHAPE.RIGHT_BRACE, radius=0.8)
    watch_mid = P11_STAGES[1][3] + P11_STAGES[1][4] / 2
    add_arrow(slide, P11_BRACE_X + P11_BRACE_W + 0.03, watch_mid, centre_left, watch_mid,
              G500, 1.25)
    p11_tag(slide, left_lane, watch_mid - 0.17, read, G500, 0.76)

    think_mid = P11_STAGES[2][3] + P11_STAGES[2][4] / 2
    board_top = P11_LEFT[2][3]
    add_path_arrow(slide, [(P11_WATCH_X, board_top - 0.03), (P11_WATCH_X, think_mid),
                           (centre_left, think_mid)], G500, 1.25)
    p11_tag(slide, left_lane, think_mid - 0.17, observe, G500, 0.46)
    tool_bottom = P11_RIGHT[0][3] + P11_RIGHT[0][4]
    add_path_arrow(slide, [(centre_right, think_mid), (P11_CALL_X, think_mid),
                           (P11_CALL_X, tool_bottom + 0.04)], ORANGE_FILL, 1.5)
    p11_tag(slide, right_lane, think_mid - 0.17, call, ORANGE, 0.66)

    action_mid = P11_STAGES[3][3] + P11_STAGES[3][4] / 2
    add_arrow(slide, centre_right, action_mid, P11_RX - 0.04, action_mid, ORANGE_FILL, 1.5)
    p11_tag(slide, right_lane, action_mid - 0.17, fire, ORANGE, 0.66)

    harness_x = P11_RX + P11_RW / 2
    harness_bottom = P11_RIGHT[1][3] + P11_RIGHT[1][4]
    board_bottom = P11_LEFT[2][3] + P11_LEFT[2][4]
    add_path_arrow(slide, [(harness_x, harness_bottom + 0.04), (harness_x, P11_RETURN_Y),
                           (P11_RETURN_X, P11_RETURN_Y), (P11_RETURN_X, board_bottom + 0.04)],
                   G500, 1.25)
    p11_tag(slide, P11_SPINE, P11_RETURN_Y, P11_RETURN, G500, 2.56, size=8)

    add_box(slide, MX, 6.46, CW, 0.58, fill=TINT)
    add_box(slide, MX, 6.46, 0.08, 0.58, fill=AGENT)
    add_text(slide, MX + 0.36, 6.46, CW - 0.72, 0.58, [runs([
        (P11_BAND[0], 12, AGENT, True), ("　｜　", 12, G400), (P11_BAND[1], 13, G900, True),
    ])], anchor=MIDDLE)


# ----------------------------------------------------------------------- 10  模型Roofline
P10_TITLE = "M100 Agent：全栈模型 Roofline 协同优化体系"
P10_SUB = ("演进目标（Target State）：算子 · 计算图 · 引擎三层协同寻优；优化目标由约束条件正向推导，"
           "最终准入交由真机实测一锤定音")
P10_AIM = ("目标层", "锚定上限 · 拆解差距")
P10_AIM_HUMAN = (2.30, "人（输入约束）", ("业务流量画像", "服务时延约束（SLO）"))
P10_AIM_CEIL = (4.40, "上限核算器 · 纯理论推导（无模型）",
                ("仅依据模型配置与硬件物理规格直接推算",
                 "理论上限不含实测校准项，不靠实测数据反向拼凑"))
P10_AIM_GAP = (4.23, "总差距拆解为三个维度", "（三个倍率相乘，不是耗时相加）",
               ("工作点", "步间调度", "单步执行"), "责任到层，每一项差距均有明确的优化归属")
P10_DISPATCH = "↓　任务派发：按理论差距大小排序，明确标注归属层级"
P10_PLAN = ("规划层", "外环寻优工作点 · 内环逼近硬件上限")
P10_LEGEND = "方括号标签表示当前能力成熟度"
P10_OUTER = ("外环 · 推理引擎", "探索中", (
    ("工作点", "显存切分策略 · 单步批大小（Batch Size） · 投机解码（Speculative Decoding）"),
    ("步间调度", "调度与执行重叠（Overlap Scheduler） · 分块预填充（Chunked Prefill）与混跑 · "
              "前缀缓存 · PD 分离部署"),
))
P10_CONTRACT = ("↓　下发：张量形状（Shape）分布 + 新算子需求",
                "↑　反馈：单步耗时代价曲线 + 时延余量（Budget）")
P10_INNER = ("内环 · 计算图 + 算子：在锁定工作点上逼近硬件极限", "单步执行", (
    ("计算图", "部分就绪", "算子融合 · 多流并发（Multi-Stream） · 通信窗口预取"),
    ("算子", "已就绪", "聚焦单个算子内核（Kernel）的极致代码实现"),
))
P10_INNER_EDGE = ("↓　下发：算子融合规范（契约）", "↑　反馈：新算子内核 + 实测耗时")
P10_SUBMIT = ("↓　提交候选方案（附带层级标签）", "↑　淘汰打回：附带失败原因退回重改")
P10_VERDICT = ("判决层", "规则驱动（无大模型） · 真机实测终审")
P10_SIM = ("仿真器 · 仅做初筛淘汰，不作准入放行",
           ("快速筛除明显劣化的候选方案", "（主要用于引擎层策略初筛）"))
P10_SIM_EDGE = ("→　初筛送测", "←　实测校准")
P10_REAL = ("真机实测 · 唯一准入终审（一票否决）", (
    ("算子级", ("L0（编译）/ L1（精度）/ Bench（测速），严格按引擎下发的真实 Shape 评测",)),
    ("计算图级", ("整模型端到端 A/B 性能压测 + 生成质量门禁评测",)),
    ("引擎级", ("真实流量回放 · 约束下的吞吐与 P99 · 实测接受长度 · OOM 与抢占 · 长稳 · 多画像不退化",
             "重叠开关前后一致性校验：固定批次逐位一致，混合负载按质量门禁容差判定；",
             "高负载长稳压测排查并发竞态（Race Condition）")),
))
P10_CLOSE = (("闭环演进", "验证通过即晋升为最新基线，回流目标层重新核算差距"),
             ("收敛条件", "候选收益无法跨越系统噪声门限时，迭代自动收敛终止"))
P10_NOTE = "第 7–8 页是单步口径（ms/步），本页是服务口径（约束下的吞吐），两种口径的数不能直接对齐"

P10_SPINE_X = 0.66
P10_CX = 0.88
P10_IX = P10_CX + 0.20
P10_IW = SW - MX - P10_IX


def p10_width(parts):
    """Inches one line of styled runs needs, with the margin the fit checker keeps in reserve."""
    points = sum(char_em(ch, bold) * size for text, size, bold in parts for ch in text)
    return points / 72 / 0.95


def p10_layer(slide, y, h, name, role, tone):
    """A layer panel whose name rides a pill straddling its top edge, so content keeps full width."""
    card(slide, P10_CX, y, SW - MX - P10_CX, h, fill=G050, line=None)
    add_box(slide, P10_CX, y, 0.06, h, fill=tone)
    parts = ((name, 11, True), (f"　{role}", 9.5, False))
    width = p10_width(parts) + 0.34
    shape = add_box(slide, P10_IX, y - 0.13, width, 0.26, fill=tone,
                    kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    boxed_text(shape, [runs([
        (name, 11, WHITE, True), (f"　{role}", 9.5, G050),
    ])], inset=(0.16, 0, 0.12, 0))
    return width


def p10_rule(slide, y, labels, fill=G050, size=9.5):
    """A hand-off between layers: a hairline captioned with what travels in each direction."""
    add_line(slide, P10_IX, y + 0.11, SW - MX, y + 0.11, G300, 0.75, dash=DASH)
    for index, label in enumerate(labels):
        width = p10_width(((label, size, False),)) + 0.40
        add_box(slide, P10_IX + index * 4.40, y, width, 0.22, fill=fill)
        add_text(slide, P10_IX + index * 4.40, y, width, 0.22,
                 [P(label, size, G500, align=CENTER)], anchor=MIDDLE)


def p10_capsules(slide, y, h, labels, size=9):
    """The two directions of one ring-to-ring contract, each in its own capsule."""
    tones = ((ORANGE_TINT, ORANGE), (G100, G500))
    left = P10_IX + 0.22
    for label, (fill, tone) in zip(labels, tones):
        width = p10_width(((label, size, False),)) + 0.46
        shape = add_box(slide, left, y, width, h, fill=fill,
                        kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        boxed_text(shape, [P(label, size, tone, align=CENTER)], inset=(0.04, 0, 0.04, 0))
        left += width + 0.26


def p10_tag(slide, x, y, w, h, label, tone=AGENT, fill=PALE, size=9, bold=True):
    shape = add_box(slide, x, y, w, h, fill=fill, kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    boxed_text(shape, [P(label, size, tone, bold, align=CENTER)], inset=(0.02, 0, 0.02, 0))
    return shape


def page_ten(slide):
    add_background(slide)
    page_header(slide, P10_TITLE, P10_SUB, sub_size=11.5)

    # ---- 目标层：约束进来，上限算出来，差距按维度拆开——拆法本身就是派单规则
    aim_y, aim_h = 1.38, 0.86
    p10_layer(slide, aim_y, aim_h, *P10_AIM, AGENT)
    box_y, box_h = aim_y + 0.14, aim_h - 0.16
    box_x = P10_IX
    for width, head, lines in (P10_AIM_HUMAN, P10_AIM_CEIL):
        card(slide, box_x, box_y, width, box_h, fill=WHITE, line=G200)
        add_text(slide, box_x + 0.22, box_y + 0.04, width - 0.4, box_h - 0.08, [
            P(head, 11.5, G900, True, after=4),
            *[P(line, 9, G700, after=2) for line in lines],
        ], anchor=MIDDLE)
        add_arrow(slide, box_x + width + 0.05, aim_y + aim_h / 2, box_x + width + 0.31,
                  aim_y + aim_h / 2, AGENT, 1.25)
        box_x += width + 0.36

    width, head, qualifier, terms, note = P10_AIM_GAP
    card(slide, box_x, box_y, width, box_h, fill=WHITE, line=G200)
    add_text(slide, box_x + 0.22, box_y + 0.04, width - 0.4, 0.22,
             [runs([(head, 11.5, G900, True), (qualifier, 8.5, G500)])], anchor=MIDDLE)
    term_x = box_x + 0.22
    for index, term in enumerate(terms):
        if index:
            add_text(slide, term_x, box_y + 0.26, 0.22, 0.22,
                     [P("×", 10, G500, align=CENTER)], anchor=MIDDLE)
            term_x += 0.22
        term_w = p10_width(((term, 9, True),)) + 0.22
        p10_tag(slide, term_x, box_y + 0.27, term_w, 0.20, term)
        term_x += term_w
    add_text(slide, box_x + 0.22, box_y + 0.50, width - 0.4, 0.20,
             [P(note, 9, G700)], anchor=MIDDLE)

    p10_rule(slide, 2.28, (P10_DISPATCH,))

    # ---- 规划层：外环挑工作点，内环在锁定的工作点上逼近物理上限，两环互相放大
    plan_y, plan_h = 2.68, 2.02
    p10_layer(slide, plan_y, plan_h, *P10_PLAN, ORANGE_FILL)
    legend_w = p10_width(((P10_LEGEND, 8, False),)) + 0.30
    add_box(slide, SW - MX - legend_w, plan_y - 0.13, legend_w, 0.26, fill=G050)
    add_text(slide, SW - MX - legend_w, plan_y - 0.13, legend_w, 0.26,
             [P(P10_LEGEND, 8, G500, align=CENTER)], anchor=MIDDLE)

    outer_y, outer_h = plan_y + 0.14, 0.64
    head, status, rows = P10_OUTER
    card(slide, P10_IX, outer_y, P10_IW, outer_h, fill=WHITE, line=G200)
    head_w = p10_width(((head, 11, True),))
    add_text(slide, P10_IX + 0.22, outer_y + 0.03, head_w + 0.1, 0.22,
             [P(head, 11, G900, True)], anchor=MIDDLE)
    p10_tag(slide, P10_IX + head_w + 0.40, outer_y + 0.04, 0.80, 0.20, f"[ {status} ]",
            tone=ORANGE, fill=ORANGE_TINT, size=8.5)
    for index, (term, text) in enumerate(rows):
        top = outer_y + 0.25 + index * 0.20
        p10_tag(slide, P10_IX + 0.22, top, 0.84, 0.18, term, size=8.5)
        add_text(slide, P10_IX + 1.16, top, P10_IW - 1.38, 0.18,
                 [P(text, 9, G700)], anchor=MIDDLE)

    p10_capsules(slide, outer_y + outer_h + 0.02, 0.22, P10_CONTRACT)

    inner_y, inner_h = outer_y + outer_h + 0.28, 0.92
    head, term, rows = P10_INNER
    card(slide, P10_IX, inner_y, P10_IW, inner_h, fill=WHITE, line=ORANGE_FILL, line_w=1.25)
    add_text(slide, P10_IX + 0.22, inner_y + 0.03, 5.6, 0.22,
             [P(head, 11, G900, True)], anchor=MIDDLE)
    term_w = p10_width(((term, 9, True),)) + 0.22
    p10_tag(slide, SW - MX - 0.22 - term_w, inner_y + 0.04, term_w, 0.20, term)
    for index, (lead, status, text) in enumerate(rows):
        top = inner_y + 0.26 + index * 0.42
        add_text(slide, P10_IX + 0.22, top, 0.84, 0.20,
                 [P(lead, 9.5, ORANGE, True)], anchor=MIDDLE)
        p10_tag(slide, P10_IX + 1.08, top + 0.01, 0.88, 0.18, f"[ {status} ]",
                tone=G500, fill=G100, size=8.5, bold=False)
        add_text(slide, P10_IX + 2.06, top, P10_IW - 2.28, 0.20,
                 [P(text, 9, G700)], anchor=MIDDLE)
    p10_capsules(slide, inner_y + 0.48, 0.18, P10_INNER_EDGE, size=8.5)

    p10_rule(slide, 4.74, P10_SUBMIT)

    # ---- 判决层：仿真只负责淘汰，放行只认真机；量具按候选所属的尺度挑
    ver_y, ver_h = 5.14, 1.46
    p10_layer(slide, ver_y, ver_h, *P10_VERDICT, AGENT)
    box_y, box_h = ver_y + 0.14, 1.28
    sim_w = (P10_IW - 0.86) * 0.35
    real_x = P10_IX + sim_w + 0.86
    real_w = SW - MX - real_x

    head, lines = P10_SIM
    card(slide, P10_IX, box_y, sim_w, box_h, fill=WHITE, line=G200)
    add_text(slide, P10_IX + 0.22, box_y + 0.06, sim_w - 0.4, box_h - 0.12, [
        P(head, 10.5, G900, True, after=5),
        *[P(line, 9, G700, after=2) for line in lines],
    ], anchor=MIDDLE)
    for index, label in enumerate(P10_SIM_EDGE):
        add_text(slide, P10_IX + sim_w, box_y + 0.40 + index * 0.26, 0.86, 0.24,
                 [P(label, 8.5, G500, align=CENTER)], anchor=MIDDLE)

    head, rows = P10_REAL
    card(slide, real_x, box_y, real_w, box_h, fill=WHITE, line=AGENT, line_w=1.25)
    add_text(slide, real_x + 0.22, box_y + 0.04, real_w - 0.4, 0.22,
             [P(head, 10.5, G900, True)], anchor=MIDDLE)
    top = box_y + 0.26
    for lead, lines in rows:
        height = 0.20 * len(lines)
        add_text(slide, real_x + 0.22, top, 0.94, 0.20,
                 [P(lead, 9, AGENT, True)], anchor=MIDDLE)
        add_text(slide, real_x + 1.20, top, real_w - 1.42, height,
                 [P(line, 9, G700) for line in lines], anchor=MIDDLE)
        top += height

    # ---- 回边：判决结果回到目标层，差距重算；队列见底即收敛
    band_y, band_h = 6.64, 0.34
    band_w = SW - MX - P10_CX
    add_box(slide, P10_CX, band_y, band_w, band_h, fill=PALE)
    add_box(slide, P10_CX, band_y, 0.06, band_h, fill=AGENT)
    half = band_w / 2
    add_line(slide, P10_CX + half, band_y + 0.06, P10_CX + half, band_y + band_h - 0.06,
             G300, 0.75)
    for index, (lead, text) in enumerate(P10_CLOSE):
        add_text(slide, P10_CX + 0.24 + index * half, band_y, half - 0.3, band_h, [runs([
            (f"{lead}　", 11, AGENT, True), (text, 10.5, G900),
        ])], anchor=MIDDLE)

    spine_top, spine_bottom = aim_y + 0.44, band_y + band_h / 2
    polyline(slide, [(P10_CX, spine_bottom), (P10_SPINE_X, spine_bottom),
                     (P10_SPINE_X, spine_top)], AGENT, 1.5)
    add_arrow(slide, P10_SPINE_X, spine_top, P10_CX - 0.02, spine_top, AGENT, 1.5)

    add_text(slide, MX, 7.02, 9.4, 0.28, [P(P10_NOTE, 9, G500)], anchor=MIDDLE)


# ------------------------------------------------------------------------- 12  产品化
P12_TITLE = "把M100 + Agent 能力沉淀为可交付的智能体产品"
P12_SUB = "把内部沉淀的适配智能体能力产品化，做到可复用、可交付、可复制"
P12_GOAL = ("核心目标", "面向昆仑芯M100智能体产品：可复用 Skills + 统一知识库 + 多智能体协同")
P12_CHIPS = ("模型适配与评估优化", "算子性能分析与生成", "故障诊断与智能问答", "性能与精度自动化评估")
P12_CARDS = (
    ("Skill", "执行", ("标准化 SOP：能力快照 / 启动配置", "框架与插件缺口清单，换版即同步")),
    ("知识库", "沉淀", ("源码级事实：缺口、热点与拐点清单", "不靠人工维护，结论跟着版本走")),
    ("测评体系", "把关", ("收益评估 · 压测校准", "SLO 与单机容量验证，结果回流知识库")),
)
P12_BAND = ("统一编排 · 自我进化", "把三套体系沉淀为一个多Agent产品")
P12_FLOW = (
    ("Teams协作流程编排", ("分阶段执行工作流，每个阶段明确 输入 / 输出 / 条件",
                     "定义模型适配、调优、算子绑定工作流程",
                     "并行调度：多Agent并行执行，减少等待")),
    ("Agent角色执行", ("角色：可行性评估Agent、适配与性能分析Agent、算子绑定与开发Agent、测试验证Agent",
                   "每个Agent独立执行，协同推进")),
    ("Skill 知识技能", ("包含硬件产品、源码、文档、工具、等先验知识",
                    "调试工具和结果可挂载到Agent上下文全面分析",
                    "优化分析过程和结果可作为Skill自进化输入")),
)


def page_twelve(slide):
    add_background(slide)
    page_header(slide, P12_TITLE, P12_SUB)

    goal_y, goal_h = 1.42, 0.94
    card(slide, MX, goal_y, CW, goal_h, fill=PALE, line=AGENT, line_w=1.25)
    add_text(slide, MX + 0.3, goal_y + 0.08, 2.4, 0.42, [P(P12_GOAL[0], 17, AGENT, True)],
             anchor=MIDDLE)
    add_text(slide, MX + 0.42, goal_y + goal_h - 0.4, 6.2, 0.3, [P(P12_GOAL[1], 10.5, G700)],
             anchor=MIDDLE)
    chip_x, chip_gap, chip_w = 6.90, 0.16, 2.72
    for index, label in enumerate(P12_CHIPS):
        left = chip_x + (index % 2) * (chip_w + chip_gap)
        top = goal_y + 0.14 + (index // 2) * 0.38
        shape = add_box(slide, left, top, chip_w, 0.32, fill=WHITE, line=TINT, line_w=1.0,
                        kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
        boxed_text(shape, [runs([("●　", 9, "4FC3F7"), (label, 9.5, G900)])],
                   inset=(0.12, 0, 0.06, 0))

    cards_y, cards_h = 2.54, 1.30
    gap = 0.26
    card_w = (CW - gap * 2) / 3
    for index, (name, role, lines) in enumerate(P12_CARDS):
        left = MX + index * (card_w + gap)
        card(slide, left, cards_y, card_w, cards_h, fill=WHITE, line=G200)
        add_box(slide, left, cards_y, card_w, 0.05, fill=AGENT)
        add_text(slide, left + 0.26, cards_y + 0.14, card_w - 0.5, 0.36, [runs([
            (name, 13.5, AGENT, True), ("　｜　", 12, G300), (role, 12, G700, True),
        ])], anchor=MIDDLE)
        add_text(slide, left + 0.26, cards_y + 0.58, card_w - 0.5, 0.62,
                 [P(line, 10.5, G700, after=3) for line in lines], anchor=TOP)

    band_y, band_h = 4.00, 0.44
    add_box(slide, MX, band_y, CW, band_h, fill=PALE)
    add_text(slide, MX, band_y, CW, band_h, [runs([
        (P12_BAND[0], 12, AGENT, True), ("　→　", 12, G400), (P12_BAND[1], 12, G900, True),
    ], align=CENTER)], anchor=MIDDLE)

    flow_y, flow_h = 4.64, 2.26
    gap = 0.48
    flow_w = (CW - gap * 2) / 3
    for index, (name, lines) in enumerate(P12_FLOW):
        left = MX + index * (flow_w + gap)
        add_box(slide, left, flow_y, flow_w, 0.56, fill=AGENT)
        add_text(slide, left + 0.24, flow_y, flow_w - 0.44, 0.56,
                 [P(name, 14, WHITE, True)], anchor=MIDDLE)
        card(slide, left + 0.22, flow_y + 0.68, flow_w - 0.22, flow_h - 0.68, fill=WHITE,
             line=G200)
        add_text(slide, left + 0.4, flow_y + 0.78, flow_w - 0.6, flow_h - 0.88,
                 [runs([("•", 10, AGENT), (line, 10, G700)], after=4) for line in lines],
                 anchor=TOP)
        if index:
            chevron = add_box(slide, left - gap + 0.02, flow_y + 0.04, gap - 0.06, 0.48,
                              fill=TINT, kind=MSO_SHAPE.PENTAGON)
            drop_theme_style(chevron)


# ------------------------------------------------------------------------- 13  生态飞轮
P13_TITLE = "生态飞轮：面向伙伴、最终客户，放大生态收益"
P13_SUB = (("把 M100智能体产品 开放出去 —— ", G500), ("降低适配门槛、加速模型覆盖，让昆仑芯生态进入正循环", G700))
P13_NODES = ("降低模型适配成本", "扩大模型覆盖", "反哺软件栈能力", "提升芯片竞争力", "吸引更多伙伴与客")
P13_BAND = ("生态收益最大化", "让每一次适配都可沉淀、可复用，把个体投入转化为昆仑芯的生态资产")


def page_thirteen(slide):
    add_background(slide)
    page_header(slide, P13_TITLE, None)
    add_text(slide, MX + 0.02, 0.94, HEAD_W, 0.34,
             [runs([(text, 12, tone) for text, tone in P13_SUB])], anchor=MIDDLE)

    cx, cy = SW / 2, 3.98
    rx, ry = 3.15, 1.92
    node_w, node_h = 1.80, 0.82

    centres = []
    for index in range(len(P13_NODES)):
        angle = -math.pi / 2 + index * 2 * math.pi / len(P13_NODES)
        centres.append((cx + rx * math.cos(angle), cy + ry * math.sin(angle)))

    def leave(centre, ux, uy):
        """Where a ray out of a node's centre crosses that node's own edge, plus a gap."""
        span = min(abs((node_w / 2 + 0.12) / ux) if ux else 1e9,
                   abs((node_h / 2 + 0.12) / uy) if uy else 1e9)
        return centre[0] + ux * span, centre[1] + uy * span

    for index, centre in enumerate(centres):
        nxt = centres[(index + 1) % len(centres)]
        dx, dy = nxt[0] - centre[0], nxt[1] - centre[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        start = leave(centre, ux, uy)
        end = leave(nxt, -ux, -uy)
        add_arrow(slide, start[0], start[1], end[0], end[1], AGENT, 1.75)

    for (ncx, ncy), label in zip(centres, P13_NODES):
        shape = add_box(slide, ncx - node_w / 2, ncy - node_h / 2, node_w, node_h, fill=AGENT,
                        kind=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12)
        boxed_text(shape, [P(label, 12, WHITE, True, align=CENTER)], inset=(0.08, 0, 0.08, 0))
    add_text(slide, cx - 1.4, cy - 0.3, 2.8, 0.6,
             [P("Looping", 22, AGENT, True, align=CENTER)], anchor=MIDDLE)

    band_y, band_h = 6.42, 0.62
    add_box(slide, MX, band_y, CW, band_h, fill=PALE)
    add_box(slide, MX, band_y, 0.08, band_h, fill=AGENT)
    add_text(slide, MX + 0.36, band_y, CW - 0.72, band_h, [runs([
        (P13_BAND[0], 14, AGENT, True), ("　——　", 13, G400), (P13_BAND[1], 13, G900, True),
    ])], anchor=MIDDLE)


# --------------------------------------------------------------------------------- build
PAGES = (
    (1, page_one),
    (2, lambda slide: divider_slide(slide, "如何把 M100 上模型适配的成本打下来")),
    (3, page_three),
    (4, page_four),
    (5, page_five),
    (6, page_six),
    (7, page_seven_bound),
    (8, page_seven),
    (9, page_eight),
    (10, page_nine),
    (11, page_eleven),
    (12, page_ten),
    (13, lambda slide: divider_slide(slide, "从『我们自己做生态』到『合作伙伴共建生态飞轮』")),
    (14, page_twelve),
    (15, page_thirteen),
)
DIVIDERS = {2, 13}


def build(path):
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    for number, draw in PAGES:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        draw(slide)
        if number not in DIVIDERS:
            page_footer(slide, number)
    prs.save(path)
    return len(PAGES)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else OUTPUT
    print(target, build(target), "pages")
