"""Two candidate layouts for a model-layer page, drawn with every figure left blank.

Both reuse the 1024 程序员节 theme from build_kunlunxin_deck so they can be judged next to
the existing pages. Version A charts phases C and D of a model-agent run: every candidate
forks from the baseline, so it plots each candidate against that baseline rather than a
best-so-far line. Version B charts phase B: the Trace placed on a roofline, which answers
"who is worth optimizing next" rather than "how far did we get".

Every measured figure is written as __ and every schematic mark is drawn in grey and badged
示意, so nothing here can be mistaken for data.
"""

import math
import sys

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches

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
    WHITE,
    Y_FOOT,
    P,
    add_arrow,
    add_box,
    add_line,
    add_text,
    runs,
)
from build_kunlunxin_deck import (
    AGENT,
    G050,
    G100,
    G200,
    G300,
    G400,
    G500,
    G700,
    G900,
    MODEL_LEVEL,
    MODEL_TINT,
    TINT,
    add_background,
    band,
    page_header,
    swatch,
)

BLANK = "__"
MUTED = G300


def badge(slide, x, y, text="示意　非数据"):
    add_box(slide, x, y, 1.18, 0.24, fill=MODEL_TINT)
    add_text(slide, x, y, 1.18, 0.24,
             [P(text, 8.5, MODEL_LEVEL, True, align=CENTER)], anchor=MIDDLE)


def mock_footer(slide, tag, note):
    add_text(slide, MX, Y_FOOT, CW, 0.28,
             [runs([(f"{tag}　", 10, MODEL_LEVEL, True), (note, 10, G500)])], anchor=MIDDLE)


def strip(slide, y, parts):
    add_box(slide, MX, y, CW, 0.32, fill=G050)
    add_text(slide, MX + 0.2, y, CW - 0.4, 0.32, [runs(parts)], anchor=MIDDLE)


def dot(slide, cx, cy, size, hollow=False, color=MUTED):
    add_box(slide, cx - size / 2, cy - size / 2, size, size,
            fill=WHITE if hollow else color, line=color if hollow else WHITE,
            line_w=1.75 if hollow else 1.25, kind=MSO_SHAPE.OVAL)


# ------------------------------------------------------------- A  candidates against baseline
# (tick, gain found while screening in phase C, gain on the phase-D re-test, verdict, shipped).
# A gain of None means the candidate stopped at gate ① or ② and was never timed.
A_CANDIDATES = (
    ("C1", 0.8, None, "③ 未过", False),
    ("C2", 6.5, 5.0, "通过", False),
    ("C3", None, None, "① 未过", False),
    ("C4", 10.5, 8.6, "交付", True),
    ("C5", -2.5, None, "③ 未过", False),
)
A_CHANGES = (
    "例：调大 batch 上限",
    "例：换 attention 后端",
    "例：接入算子层新内核",
    "例：KV 缓存与复用策略",
    "例：通信与计算重叠",
)
A_FUNNEL = ("候选", "过 ① 输出未变", "过 ② 质量达标", "过 ③ 确实更快", "过 D 独立复验")
A_X0, A_X1 = 1.30, 7.15
A_Y0, A_Y1 = 2.62, 4.70
A_DX0, A_DX1 = 1.85, 6.75
A_GAIN_LO, A_GAIN_HI = -5.0, 14.0
A_GRID = (-4.0, 0.0, 4.0, 8.0, 12.0)
A_GATE, A_NOISE = 3.0, 1.8
A_PAIR = 0.12


def _ax(index):
    return A_DX0 + index * (A_DX1 - A_DX0) / (len(A_CANDIDATES) - 1)


def _ay(gain):
    return A_Y1 - (gain - A_GAIN_LO) / (A_GAIN_HI - A_GAIN_LO) * (A_Y1 - A_Y0)


def a_chart(slide, x, y, w, h):
    add_box(slide, x, y, w, h, fill=G050)
    add_text(slide, x + 0.22, y + 0.06, 2.85, 0.26, [runs([
        ("收益", 10, G900, True), (" = 候选 ÷ 基线 − 1　重复压测取中位", 10, G700),
    ])], anchor=MIDDLE)
    swatch(slide, x + 3.17, y + 0.04, AGENT, DASH, "性能门 +3%", 1.50)
    swatch(slide, x + 4.80, y + 0.04, MODEL_LEVEL, DASH, f"噪声极差 {BLANK}%", 1.80)

    for gain in A_GRID:
        gy = _ay(gain)
        baseline = gain == 0.0
        add_line(slide, A_X0, gy, A_X1, gy, G400 if baseline else G200, 1.0 if baseline else 0.75)
        add_text(slide, A_X0 - 0.62, gy - 0.11, 0.5, 0.22,
                 [P("基线" if baseline else BLANK, 9, G500 if baseline else G400,
                    font=FONT if baseline else MONO, align=RIGHT)], anchor=MIDDLE)
    for gain, color in ((A_GATE, AGENT), (A_NOISE, MODEL_LEVEL)):
        add_line(slide, A_X0, _ay(gain), A_X1, _ay(gain), color, 1.0, dash=DASH)
    add_line(slide, A_X0, A_Y1, A_X1, A_Y1, G300, 1.0)

    for index, (tick, screened, retested, _verdict, shipped) in enumerate(A_CANDIDATES):
        cx = _ax(index)
        add_text(slide, cx - 0.33, A_Y1 + 0.04, 0.66, 0.22,
                 [P(tick, 9.5, G500, False, MONO, align=CENTER)], anchor=MIDDLE)
        if screened is None:
            arm = 0.06
            add_line(slide, cx - arm, A_Y1 - arm, cx + arm, A_Y1 + arm, G400, 1.5)
            add_line(slide, cx - arm, A_Y1 + arm, cx + arm, A_Y1 - arm, G400, 1.5)
            add_text(slide, cx - 0.45, A_Y1 - 0.34, 0.9, 0.22,
                     [P(f"停在 {BLANK}", 9, G400, align=CENTER)], anchor=MIDDLE)
            continue
        if retested is None:
            dot(slide, cx, _ay(screened), 0.14)
            add_text(slide, cx + 0.12, _ay(screened) - 0.11, 0.5, 0.22,
                     [P(BLANK, 9.5, G400, True, MONO)], anchor=MIDDLE)
            continue
        left, right = cx - A_PAIR, cx + A_PAIR
        add_line(slide, left, _ay(screened), right, _ay(retested), MUTED, 1.25)
        dot(slide, left, _ay(screened), 0.14)
        dot(slide, right, _ay(retested), 0.20 if shipped else 0.16, hollow=True,
            color=G500 if shipped else MUTED)
        add_text(slide, cx - 0.55, _ay(max(screened, retested)) - 0.33, 1.1, 0.22,
                 [P(f"{BLANK} → {BLANK}", 9.5, G400, True, MONO, align=CENTER)], anchor=MIDDLE)
        if shipped:
            add_text(slide, right - 0.3, _ay(retested) + 0.13, 0.6, 0.22,
                     [P("交付", 9.5, G700, True, align=CENTER)], anchor=MIDDLE)

    badge(slide, x + w - 1.40, y + h - 0.34)
    add_text(slide, x + 0.22, y + h - 0.34, w - 1.70, 0.24, [runs([
        ("●", 9, G400), (" C 阶段筛选　", 9, G500),
        ("○", 9, G400), (" D 阶段独立复验　", 9, G500),
        ("×", 9, G400), (" 停在 ① 或 ②，未测性能", 9, G500),
    ])], anchor=MIDDLE)


def a_candidates(slide, x, y, w, h):
    add_box(slide, x, y, w, h, fill=G050)
    add_text(slide, x + 0.22, y + 0.08, w - 0.44, 0.28, [runs([
        (f"{BLANK} 个候选", 12.5, G900, True),
        (f"　{BLANK} 个过三道门，复验后交付 1 个", 10, G500),
    ])], anchor=MIDDLE)
    columns = ((x + 0.66, 2.0, "改动", LEFT), (x + 2.70, 0.62, "C 筛选", RIGHT),
               (x + 3.38, 0.62, "D 复验", RIGHT), (x + 4.10, 0.56, "结论", LEFT))
    for cell_x, cell_w, head, align in columns:
        add_text(slide, cell_x, y + 0.44, cell_w, 0.24,
                 [P(head, 9, G400, align=align)], anchor=MIDDLE)
    add_line(slide, x + 0.22, y + 0.70, x + w - 0.22, y + 0.70, G200, 0.75)

    rows = [("基线", "A 阶段建立，D 阶段重测", BLANK, BLANK, "—", False)]
    for (tick, screened, retested, verdict, shipped), change in zip(A_CANDIDATES, A_CHANGES):
        rows.append((tick, change, "—" if screened is None else BLANK,
                     "—" if retested is None else BLANK, verdict, shipped))
    for index, (tick, change, screened, retested, verdict, shipped) in enumerate(rows):
        top = y + 0.76 + index * 0.40
        if shipped or index == 0:
            add_box(slide, x + 0.14, top, w - 0.28, 0.36, fill=TINT if shipped else G100)
        add_text(slide, x + 0.22, top, 0.44, 0.36,
                 [P(tick, 9.5, G500, True, MONO)], anchor=MIDDLE)
        add_text(slide, columns[0][0], top, columns[0][1], 0.36,
                 [P(change, 9.5, G900 if shipped else G700)], anchor=MIDDLE)
        for (cell_x, cell_w, _head, align), value in zip(columns[1:3], (screened, retested)):
            add_text(slide, cell_x, top, cell_w, 0.36,
                     [P(value, 9.5, G400, False, MONO, align=align)], anchor=MIDDLE)
        add_text(slide, columns[3][0], top, columns[3][1], 0.36,
                 [P(verdict, 9.5, G700 if shipped else G400, shipped)], anchor=MIDDLE)


def a_funnel(slide, y, h):
    lead_w = 2.9
    add_text(slide, MX, y, lead_w, h, [
        P("三道门的漏斗", 11.5, G900, True, after=2),
        P("由便宜到贵，任一道不过即停，", 9.5, G500, after=0),
        P("越靠前的门越省设备时间", 9.5, G500),
    ], anchor=MIDDLE)
    gap = 0.16
    x0 = MX + lead_w + 0.1
    cell_w = (SW - MX - x0 - (len(A_FUNNEL) - 1) * gap) / len(A_FUNNEL)
    for index, name in enumerate(A_FUNNEL):
        x = x0 + index * (cell_w + gap)
        add_box(slide, x, y, cell_w, h, fill=WHITE, line=G200)
        add_text(slide, x + 0.16, y + 0.04, cell_w - 0.32, 0.20,
                 [P(name, 9.5, G500)], anchor=MIDDLE)
        add_text(slide, x + 0.16, y + 0.26, cell_w - 0.32, 0.22,
                 [P(BLANK, 15, G400, True, MONO)], anchor=MIDDLE)
        add_text(slide, x + 0.16, y + 0.50, cell_w - 0.32, 0.18,
                 [P(f"设备时间 {BLANK}", 8.5, G400)], anchor=MIDDLE)


def version_a(slide):
    page_header(
        slide,
        "模型层 · 一次完整的候选迭代",
        f"{BLANK} 模型：{BLANK} 个候选，{BLANK} 个过三道门，交付收益 +{BLANK}%",
        "候选都从基线分叉、收益不叠加——交付的是 D 阶段独立复验后最好的那一个",
    )
    strip(slide, 1.78, (
        ("冻结口径　", 10, G900, True),
        (f"{BLANK} · {BLANK} 精度 · TP{BLANK} · 并发 {BLANK} · 输入 {BLANK} / 输出 {BLANK}",
         9.5, AGENT, True, MONO),
        ("　主指标 ", 10, G700),
        (BLANK, 9.5, G900, True, MONO),
        ("　基线 ", 10, G700),
        (f"A 阶段 {BLANK} → D 阶段重测 {BLANK}", 9.5, G900, True, MONO),
        ("　两者之差即设备漂移", 10, G500),
    ))
    chart_w = 7.15
    a_chart(slide, MX, 2.16, chart_w, 3.30)
    list_x = MX + chart_w + 0.2
    a_candidates(slide, list_x, 2.16, SW - MX - list_x, 3.30)
    a_funnel(slide, 5.56, 0.68)
    band(slide, 6.34, 0.52, "读数", "待填：交付了哪个候选、复验收益多少、哪道门挡掉的最多")
    mock_footer(slide, "方案 A", "与第 7 页同骨架，但候选都从基线分叉、收益不叠加，所以画逐个候选而不画爬升曲线；"
                                "要一次跑完 A–D 的任务才能填满")


# -------------------------------------------------------------------- B  roofline placement
# Both axes are log2. Performance is measured relative to the compute peak, so the plateau is
# 0 and the bandwidth slope is log2(intensity) - log2(ridge).
B_LX0, B_LX1 = -2.0, 12.0
B_RIDGE = 6.5
B_PLO = -9.5
# (category, stage, log2 intensity, fraction of its own roof attained, label placement).
# Decode GEMM intensity tracks the batch size, prefill sits far right of any ridge, decode
# attention tracks the GQA group size, and Norm-like kernels stay below 1 FLOP/Byte.
B_KINDS = (
    ("Norm / 逐元素", "decode", -0.3, 0.40, "right"),
    ("Attention", "decode", 3.0, 0.50, "below"),
    ("GEMM", "decode", 4.0, 0.70, "below"),
    ("Attention", "prefill", 9.6, 0.35, "below"),
    ("GEMM", "prefill", 11.2, 0.60, "right"),
)
B_ROWS = (
    ("prefill · GEMM", False, True),
    ("prefill · Attention", False, True),
    ("decode · GEMM", True, True),
    ("decode · Attention", False, True),
    ("decode · Norm/逐元素", False, True),
    ("通信", False, False),
)
B_X0, B_X1 = 1.52, 7.95
B_Y0, B_Y1 = 2.90, 5.30


def _bx(log2_intensity):
    return B_X0 + (log2_intensity - B_LX0) / (B_LX1 - B_LX0) * (B_X1 - B_X0)


def _by(log2_perf):
    return B_Y1 - (log2_perf - B_PLO) / (0.0 - B_PLO) * (B_Y1 - B_Y0)


def _roof(log2_intensity):
    """log2 of the reachable fraction of peak: the bandwidth slope below the ridge, 0 above."""
    return min(0.0, log2_intensity - B_RIDGE)


def b_plot(slide, x, y, w, h):
    add_box(slide, x, y, w, h, fill=G050)
    add_text(slide, x + 0.22, y + 0.06, 3.9, 0.26, [runs([
        ("可达性能", 10, G900, True), (" = min(峰值算力, 实测带宽 × 算术强度)", 10, G700),
    ])], anchor=MIDDLE)
    swatch(slide, x + 4.22, y + 0.04, MODEL_LEVEL, None, "访存受限段（斜线）", 1.70)
    swatch(slide, x + 5.98, y + 0.04, AGENT, None, "算力受限段（平台）", 1.70)

    for log2_perf in (-8.0, -6.0, -4.0, -2.0, 0.0):
        gy = _by(log2_perf)
        add_line(slide, B_X0, gy, B_X1, gy, G200, 0.75)
        add_text(slide, B_X0 - 0.66, gy - 0.11, 0.54, 0.22,
                 [P(BLANK, 9, G400, font=MONO, align=RIGHT)], anchor=MIDDLE)
    add_line(slide, B_X0, B_Y1, B_X1, B_Y1, G300, 1.0)
    add_line(slide, B_X0, B_Y0, B_X0, B_Y1, G300, 1.0)
    for log2_intensity in range(int(B_LX0), int(B_LX1) + 1, 2):
        add_text(slide, _bx(log2_intensity) - 0.3, B_Y1 + 0.04, 0.6, 0.22,
                 [P(BLANK, 9.5, G400, font=MONO, align=CENTER)], anchor=MIDDLE)

    ridge_x, ceiling = _bx(B_RIDGE), _by(0.0)
    add_line(slide, B_X0, _by(_roof(B_LX0)), ridge_x, ceiling, MODEL_LEVEL, 2.0)
    add_line(slide, ridge_x, ceiling, B_X1, ceiling, AGENT, 2.0)
    add_box(slide, ridge_x - 0.07, ceiling - 0.07, 0.14, 0.14,
            fill=G900, line=WHITE, line_w=1.25, kind=MSO_SHAPE.OVAL)
    add_text(slide, ridge_x - 0.90, ceiling - 0.34, 1.80, 0.22,
             [P(f"拐点 {BLANK} FLOP/Byte", 9, G900, True, align=CENTER)], anchor=MIDDLE)

    for name, stage, log2_intensity, attained, place in B_KINDS:
        cx = _bx(log2_intensity)
        cy = _by(_roof(log2_intensity) + math.log2(attained))
        dot(slide, cx, cy, 0.18, hollow=stage == "prefill")
        label_w = 0.42 + 0.075 * len(name)
        if place == "right":
            box, align = (cx + 0.14, cy - 0.11), LEFT
        else:
            box, align = (cx - label_w / 2, cy + 0.12), CENTER
        add_text(slide, box[0], box[1], label_w, 0.22,
                 [P(name, 9, G700, True, align=align)], anchor=MIDDLE)
        if (name, stage) == ("GEMM", "decode"):
            # Parallel to the slope: decode GEMM intensity grows with the batch size.
            ux, uy = _bx(1.0) - _bx(0.0), _by(0.0) - _by(-1.0)
            norm = math.hypot(ux, uy)
            sx, sy = cx + 0.16, cy - 0.09
            add_arrow(slide, sx, sy, sx + 0.62 * ux / norm, sy + 0.62 * uy / norm, G400, 1.25)
            add_text(slide, sx + 0.36, sy - 0.02, 1.48, 0.22,
                     [P(f"并发 ↑ 至 {BLANK} 越过拐点", 9, G500)], anchor=MIDDLE)

    add_text(slide, x + 0.22, y + h - 0.32, w - 1.80, 0.24, [runs([
        ("○", 9, G400), (" prefill　", 9, G500), ("●", 9, G400), (" decode　　", 9, G500),
        ("横轴 算术强度 FLOP/Byte　纵轴 可达性能 TFLOPS　两轴均为对数", 9, G500),
    ])], anchor=MIDDLE)
    badge(slide, x + w - 1.40, y + h - 0.34)


def b_table(slide, x, y, w, h):
    add_box(slide, x, y, w, h, fill=G050)
    add_text(slide, x + 0.22, y + 0.08, w - 0.44, 0.28, [runs([
        ("每类算子的读数", 12.5, G900, True), ("　时间与占比来自 Trace", 10, G500),
    ])], anchor=MIDDLE)
    heads = ("类别", "强度", "实测", "距上界", "占比")
    widths = (1.56, 0.42, 0.42, 0.52, 0.42)
    xs, run_x = [], x + 0.18
    for width in widths:
        xs.append((run_x, width))
        run_x += width + 0.05
    for (cell_x, width), head in zip(xs, heads):
        add_text(slide, cell_x, y + 0.46, width, 0.26, [P(head, 9, G400)], anchor=MIDDLE)
    add_line(slide, x + 0.22, y + 0.74, x + w - 0.22, y + 0.74, G200, 0.75)
    for index, (name, hero, plotted) in enumerate(B_ROWS):
        top = y + 0.80 + index * 0.37
        if hero:
            add_box(slide, x + 0.14, top, w - 0.28, 0.34, fill=TINT)
        add_text(slide, xs[0][0], top, xs[0][1], 0.34,
                 [P(name, 9.5, G900 if hero else G700)], anchor=MIDDLE)
        for column, (cell_x, width) in enumerate(xs[1:], start=1):
            value = BLANK if plotted or column == len(xs) - 1 else "—"
            add_text(slide, cell_x, top, width, 0.34,
                     [P(value, 9.5, G400, False, MONO)], anchor=MIDDLE)
    add_text(slide, x + 0.22, y + h - 0.84, w - 0.44, 0.78, [
        P("距上界 = 实测 ÷ 该强度的上界；占比按所在阶段算", 9.5, G500, after=1),
        P("贴斜线的只能少搬字节，贴平台的只能少算", 9.5, G500, after=1),
        P("通信受互联带宽限制，只列占比，不上图", 9.5, G500),
    ])


def version_b(slide):
    page_header(
        slide,
        "模型层 · roofline 定位",
        "整步耗时里，谁贴着上界，谁还有空间",
        "用 B 阶段的 Trace 把各类算子放上同一张 roofline——只用来选下一轮的方向",
    )
    strip(slide, 1.78, (
        ("卡常数　", 10, G900, True),
        (f"峰值算力 {BLANK} TFLOPS（{BLANK} 精度）", 9.5, AGENT, True, MONO),
        ("　|　", 10, G300),
        (f"实测带宽 {BLANK} GB/s", 9.5, AGENT, True, MONO),
        ("　|　", 10, G300),
        (f"拐点 {BLANK} FLOP/Byte", 9.5, G900, True, MONO),
        ("　|　", 10, G300),
        (f"工作点 并发 {BLANK} · 输入 {BLANK} / 输出 {BLANK}", 9.5, G900, True, MONO),
        ("　常数用探针实测值", 10, G500),
    ))
    plot_w = 8.0
    b_plot(slide, MX, 2.16, plot_w, 3.94)
    table_x = MX + plot_w + 0.2
    b_table(slide, table_x, 2.16, SW - MX - table_x, 3.94)
    band(slide, 6.34, 0.52, "读数", "待填：按点落在哪一段，写一句下一轮该优化谁")
    mock_footer(slide, "方案 B", "只需一次 Trace 加 FLOP/Byte 计数，今天就能填；"
                                "它回答「下一轮该优化谁」，不回答「优化到哪了」")


PAGES = ((version_a, "方案 A · 模型层候选迭代"), (version_b, "方案 B · roofline 定位"))


def build(path):
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    for make, _label in PAGES:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_background(slide)
        make(slide)
    prs.save(path)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "model_layer_mock.pptx")
    print("ok")
