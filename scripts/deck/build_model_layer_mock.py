"""Two model-layer pages: a real candidate record, and a roofline layout still waiting for data.

One script, two looks: the 1024 程序员节 template (build_kunlunxin_deck) and the original white
deck (build_review_deck). Every page is drawn against a Theme taken from that deck's own
module, so the two versions cannot drift apart and each line takes the accent its deck already
gives the same meaning on page 7.

Version A records the first model-layer run, Qwen3-VL-30B-A3B on one M100: what each candidate
changed, what screening and re-testing measured, and what was shipped. Version B charts phase B
-- the Trace placed on a roofline, which answers "who is worth optimizing next" rather than "how
far did we get". It has no data yet, so every figure is __ and every schematic mark is grey and
badged 示意.

    python build_model_layer_mock.py                    # both styles, default file names
    python build_model_layer_mock.py review out.pptx    # one style
"""

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Inches

import build_kunlunxin_deck as kunlunxin
import build_review_deck as review
from build_agent_deck import (
    CENTER,
    CW,
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

BLANK = "__"


@dataclass(frozen=True)
class Theme:
    g900: str
    g700: str
    g500: str
    g400: str
    g300: str
    g200: str
    g100: str
    panel: str
    accent: str
    tint: str
    slope: str
    roof: str
    ship: str
    drop: str
    flag: str
    flag_tint: str
    page_header: Callable
    band: Callable
    swatch: Callable
    background: Callable | None


def _theme(deck, **own):
    # The bandwidth slope is bound like page 7's copy floor, and shipped and dropped verdicts
    # like its record table, so a reader moving between pages keeps one colour per meaning.
    return Theme(
        g900=deck.G900, g700=deck.G700, g500=deck.G500, g400=deck.G400, g300=deck.G300,
        g200=deck.G200, g100=deck.G100, panel=deck.G050, accent=deck.AGENT, tint=deck.TINT,
        slope=deck.FLOOR_C, ship=deck.SHIP_C, drop=deck.DROP_C,
        page_header=deck.page_header, band=deck.band, swatch=deck.swatch, **own,
    )


THEMES = {
    "1024": _theme(kunlunxin, roof=kunlunxin.AGENT, flag=kunlunxin.MODEL_LEVEL,
                   flag_tint=kunlunxin.MODEL_TINT, background=kunlunxin.add_background),
    "review": _theme(review, roof=review.LINE_C, flag=review.AGENT, flag_tint=review.TINT,
                     background=None),
}
OUTPUTS = {"1024": "model_layer_mock_1024.pptx", "review": "model_layer_mock.pptx"}


def badge(slide, t, x, y, text="示意　非数据"):
    add_box(slide, x, y, 1.18, 0.24, fill=t.flag_tint)
    add_text(slide, x, y, 1.18, 0.24,
             [P(text, 8.5, t.flag, True, align=CENTER)], anchor=MIDDLE)


def mock_footer(slide, t, tag, note):
    add_text(slide, MX, Y_FOOT, CW, 0.28,
             [runs([(f"{tag}　", 10, t.flag, True), (note, 10, t.g500)])], anchor=MIDDLE)


def strip(slide, t, y, parts):
    add_box(slide, MX, y, CW, 0.32, fill=t.panel)
    add_text(slide, MX + 0.2, y, CW - 0.4, 0.32, [runs(parts)], anchor=MIDDLE)


def dot(slide, cx, cy, size, color, hollow=False):
    add_box(slide, cx - size / 2, cy - size / 2, size, size,
            fill=WHITE if hollow else color, line=color if hollow else WHITE,
            line_w=1.75 if hollow else 1.25, kind=MSO_SHAPE.OVAL)


# ------------------------------------------------------------------ A  a real candidate record
# Qwen3-VL-30B-A3B on one M100, as the run reported it. Screening (C) was timed per kernel or
# micro-benchmark and re-testing (D) per step or end to end, so each column keeps its own units:
# converting one into the other would take a step breakdown the run did not record.
A_BASELINE, A_SHIPPED = 1370, 1883
A_ROWS = (
    # (change, C screening, D re-test, verdict, kind); "\n" breaks a cell where it reads best,
    # since PowerPoint sets CJK wider than LibreOffice and would otherwise wrap mid-phrase.
    ("融合 swiglu + 动态量化\n替掉散点 elementwise 3.08 ms\n+ per-token quant 0.43 ms",
     "单核 1.75 ms/步", "device 单步时间\n24.7 → 22.2 ms", "交付，全矩阵 +22~38%", "ship"),
    ("量化 + shuffle 换 SDNN 恒定核\n现用核随 rows×hidden 从 20 涨到 141 µs",
     "微基准 c128 5.43×，cos = 1.0", "E2E 吞吐 +3%\nTPOT −3.4%", "交付", "ship"),
    ("grouped-GEMM group-mode 四个旋钮",
     "四模式完全同速\nforce_nogroup −20%\nno_group cos 0.000", "—",
     "① 挡掉一个，③ 挡掉其余\n判定为闭源核内在开销", "drop"),
    ("FFN 输出合并换 shuffle_mul_sum_sdnn", "0.61×，更慢", "—", "③ 未过", "drop"),
    ("路由 softmax + topk + renorm 融合", "内核 2×，数值对（Δ 3e-8）", "只占 ~0.9% TPOT",
     "③ 过但不值当，未接线", "idle"),
)
A_FUNNEL = (("候选", "10"), ("过三道门", "5"), ("交付", "2"))


def a_peak(slide, t, x, y, w, h):
    add_box(slide, x, y, w, h, fill=t.panel)
    add_text(slide, x + 0.22, y + 0.08, w - 0.44, 0.28, [runs([
        ("峰值吞吐", 12.5, t.g900, True), ("　持续输出 tok/s", 10, t.g500),
    ])], anchor=MIDDLE)
    add_text(slide, x + 0.22, y + 0.40, w - 0.44, 0.52,
             [P(f"+{A_SHIPPED / A_BASELINE - 1:.0%}", 28, t.ship, True, "Arial")],
             anchor=MIDDLE)
    floor, tallest = y + 2.58, 1.38
    for index, (label, value, color) in enumerate(
        (("基线", A_BASELINE, t.g300), ("交付", A_SHIPPED, t.ship))
    ):
        bx, bar_h = x + 0.50 + index * 1.30, tallest * value / A_SHIPPED
        add_box(slide, bx, floor - bar_h, 0.70, bar_h, fill=color)
        add_text(slide, bx - 0.25, floor - bar_h - 0.28, 1.20, 0.24,
                 [P(str(value), 11.5, t.g900, True, "Arial", align=CENTER)], anchor=MIDDLE)
        add_text(slide, bx - 0.25, floor + 0.07, 1.20, 0.22,
                 [P(label, 9.5, t.g700, align=CENTER)], anchor=MIDDLE)
    add_line(slide, x + 0.30, floor, x + w - 0.30, floor, t.g300, 1.0)
    add_text(slide, x + 0.22, y + h - 0.34, w - 0.44, 0.24,
             [P("基线峰值在 c128，1K/1K", 9, t.g500)], anchor=MIDDLE)


def a_table(slide, t, x, y, w, h):
    add_box(slide, x, y, w, h, fill=t.panel)
    add_text(slide, x + 0.22, y + 0.08, w - 0.44, 0.28, [runs([
        ("候选明细", 12.5, t.g900, True), ("　五组，全部为单卡 TP1 口径", 10, t.g500),
    ])], anchor=MIDDLE)
    columns = ((x + 0.22, 2.90), (x + 3.22, 2.19), (x + 5.51, 1.40), (x + 7.01, 1.70))
    for (cell_x, cell_w), head in zip(columns, ("改动", "C 筛选", "D 复验", "结论")):
        add_text(slide, cell_x, y + 0.40, cell_w, 0.24, [P(head, 9, t.g400)], anchor=MIDDLE)
    add_line(slide, x + 0.22, y + 0.66, x + w - 0.22, y + 0.66, t.g200, 0.75)

    verdict_color = {"ship": t.ship, "drop": t.drop, "idle": t.g500}
    top = y + 0.72
    for change, screened, retested, verdict, kind in A_ROWS:
        cells = (change, screened, retested, verdict)
        row_h = 0.14 + 0.16 * max(cell.count("\n") + 1 for cell in cells)
        shipped = kind == "ship"
        if shipped:
            add_box(slide, x + 0.14, top, w - 0.28, row_h, fill=t.tint)
        colors = (t.g900 if shipped else t.g700, t.g700, t.g700, verdict_color[kind])
        for (cell_x, cell_w), cell, color, bold in zip(
            columns, cells, colors, (False, False, False, shipped)
        ):
            ink = t.g400 if cell == "—" else color
            add_text(slide, cell_x, top, cell_w, row_h,
                     [P(line, 9.5, ink, bold) for line in cell.split("\n")], anchor=MIDDLE)
        top += row_h + 0.04


def a_funnel(slide, t, y, h):
    lead_w = 2.9
    add_text(slide, MX, y, lead_w, h, [
        P("三道门的漏斗", 11.5, t.g900, True, after=2),
        P("由便宜到贵，任一道不过即停，", 9.5, t.g500, after=0),
        P("数值不对、更慢的当场挡掉", 9.5, t.g500),
    ], anchor=MIDDLE)
    gap = 0.46
    x0 = MX + lead_w + 0.1
    cell_w = (SW - MX - x0 - (len(A_FUNNEL) - 1) * gap) / len(A_FUNNEL)
    for index, (name, count) in enumerate(A_FUNNEL):
        cx = x0 + index * (cell_w + gap)
        last = index == len(A_FUNNEL) - 1
        add_box(slide, cx, y, cell_w, h, fill=t.tint if last else WHITE,
                line=None if last else t.g200)
        add_text(slide, cx + 0.2, y + 0.08, cell_w - 0.4, 0.22,
                 [P(name, 9.5, t.accent if last else t.g500, last)], anchor=MIDDLE)
        add_text(slide, cx + 0.2, y + 0.32, cell_w - 0.4, 0.30,
                 [P(count, 16, t.g900, True, "Arial")], anchor=MIDDLE)
        if index:
            add_arrow(slide, cx - gap + 0.1, y + h / 2, cx - 0.1, y + h / 2, t.g400, 1.25)


def version_a(slide, t):
    t.page_header(
        slide,
        "模型层 · 一次完整的候选迭代",
        "30B-A3B 单卡：10 个候选，5 个过三道门，峰值吞吐 +37%",
        "判定全部来自 M100 真机——数值不对、更慢的当场挡掉，占比太小的不接线",
    )
    strip(slide, t, 1.78, (
        ("冻结口径　", 10, t.g900, True),
        ("Qwen3-VL-30B-A3B-Instruct-channel-fp8", 9.5, t.accent, True, MONO),
        ("　128 专家 / top-8 · 48 层 · M100 单卡 TP1 · qsx8_lever2 env · 1K/1K · "
         "CG-on + overlap-on", 9.5, t.g700),
        ("　主指标 持续输出 tok/s", 10, t.g500),
    ))
    peak_w = 3.0
    a_peak(slide, t, MX, 2.16, peak_w, 3.30)
    table_x = MX + peak_w + 0.2
    a_table(slide, t, table_x, 2.16, SW - MX - table_x, 3.30)
    a_funnel(slide, t, 5.56, 0.68)
    t.band(slide, 6.34, 0.52, "读数",
           "两个交付都在替换小核；闭源 grouped-GEMM 的旋钮调不动，占比不到 1% 的核再快也不接线")
    add_text(slide, MX, Y_FOOT, CW, 0.28, [P(
        "C 列是内核或微基准读数，D 列是整步或端到端读数，两列口径不同、不能互相换算；"
        "表中全部为 M100 单卡 TP1", 10, t.g500)], anchor=MIDDLE)


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


def b_plot(slide, t, x, y, w, h):
    add_box(slide, x, y, w, h, fill=t.panel)
    add_text(slide, x + 0.22, y + 0.06, 3.9, 0.26, [runs([
        ("可达性能", 10, t.g900, True), (" = min(峰值算力, 实测带宽 × 算术强度)", 10, t.g700),
    ])], anchor=MIDDLE)
    t.swatch(slide, x + 4.22, y + 0.04, t.slope, None, "访存受限段（斜线）", 1.70)
    t.swatch(slide, x + 5.98, y + 0.04, t.roof, None, "算力受限段（平台）", 1.70)

    for log2_perf in (-8.0, -6.0, -4.0, -2.0, 0.0):
        gy = _by(log2_perf)
        add_line(slide, B_X0, gy, B_X1, gy, t.g200, 0.75)
        add_text(slide, B_X0 - 0.66, gy - 0.11, 0.54, 0.22,
                 [P(BLANK, 9, t.g400, font=MONO, align=RIGHT)], anchor=MIDDLE)
    add_line(slide, B_X0, B_Y1, B_X1, B_Y1, t.g300, 1.0)
    add_line(slide, B_X0, B_Y0, B_X0, B_Y1, t.g300, 1.0)
    for log2_intensity in range(int(B_LX0), int(B_LX1) + 1, 2):
        add_text(slide, _bx(log2_intensity) - 0.3, B_Y1 + 0.04, 0.6, 0.22,
                 [P(BLANK, 9.5, t.g400, font=MONO, align=CENTER)], anchor=MIDDLE)

    ridge_x, ceiling = _bx(B_RIDGE), _by(0.0)
    add_line(slide, B_X0, _by(_roof(B_LX0)), ridge_x, ceiling, t.slope, 2.0)
    add_line(slide, ridge_x, ceiling, B_X1, ceiling, t.roof, 2.0)
    add_box(slide, ridge_x - 0.07, ceiling - 0.07, 0.14, 0.14,
            fill=t.g900, line=WHITE, line_w=1.25, kind=MSO_SHAPE.OVAL)
    add_text(slide, ridge_x - 0.90, ceiling - 0.34, 1.80, 0.22,
             [P(f"拐点 {BLANK} FLOP/Byte", 9, t.g900, True, align=CENTER)], anchor=MIDDLE)

    for name, stage, log2_intensity, attained, place in B_KINDS:
        cx = _bx(log2_intensity)
        cy = _by(_roof(log2_intensity) + math.log2(attained))
        dot(slide, cx, cy, 0.18, t.g300, hollow=stage == "prefill")
        label_w = 0.42 + 0.075 * len(name)
        if place == "right":
            box, align = (cx + 0.14, cy - 0.11), LEFT
        else:
            box, align = (cx - label_w / 2, cy + 0.12), CENTER
        add_text(slide, box[0], box[1], label_w, 0.22,
                 [P(name, 9, t.g700, True, align=align)], anchor=MIDDLE)
        if (name, stage) == ("GEMM", "decode"):
            # Parallel to the slope: decode GEMM intensity grows with the batch size.
            ux, uy = _bx(1.0) - _bx(0.0), _by(0.0) - _by(-1.0)
            norm = math.hypot(ux, uy)
            sx, sy = cx + 0.16, cy - 0.09
            add_arrow(slide, sx, sy, sx + 0.62 * ux / norm, sy + 0.62 * uy / norm, t.g400, 1.25)
            add_text(slide, sx + 0.36, sy - 0.02, 1.48, 0.22,
                     [P(f"并发 ↑ 至 {BLANK} 越过拐点", 9, t.g500)], anchor=MIDDLE)

    add_text(slide, x + 0.22, y + h - 0.32, w - 1.80, 0.24, [runs([
        ("○", 9, t.g400), (" prefill　", 9, t.g500), ("●", 9, t.g400), (" decode　　", 9, t.g500),
        ("横轴 算术强度 FLOP/Byte　纵轴 可达性能 TFLOPS　两轴均为对数", 9, t.g500),
    ])], anchor=MIDDLE)
    badge(slide, t, x + w - 1.40, y + h - 0.34)


def b_table(slide, t, x, y, w, h):
    add_box(slide, x, y, w, h, fill=t.panel)
    add_text(slide, x + 0.22, y + 0.08, w - 0.44, 0.28, [runs([
        ("每类算子的读数", 12.5, t.g900, True), ("　时间与占比来自 Trace", 10, t.g500),
    ])], anchor=MIDDLE)
    heads = ("类别", "强度", "实测", "距上界", "占比")
    widths = (1.56, 0.42, 0.42, 0.52, 0.42)
    xs, run_x = [], x + 0.18
    for width in widths:
        xs.append((run_x, width))
        run_x += width + 0.05
    for (cell_x, width), head in zip(xs, heads):
        add_text(slide, cell_x, y + 0.46, width, 0.26, [P(head, 9, t.g400)], anchor=MIDDLE)
    add_line(slide, x + 0.22, y + 0.74, x + w - 0.22, y + 0.74, t.g200, 0.75)
    for index, (name, hero, plotted) in enumerate(B_ROWS):
        top = y + 0.80 + index * 0.37
        if hero:
            add_box(slide, x + 0.14, top, w - 0.28, 0.34, fill=t.tint)
        add_text(slide, xs[0][0], top, xs[0][1], 0.34,
                 [P(name, 9.5, t.g900 if hero else t.g700)], anchor=MIDDLE)
        for column, (cell_x, width) in enumerate(xs[1:], start=1):
            value = BLANK if plotted or column == len(xs) - 1 else "—"
            add_text(slide, cell_x, top, width, 0.34,
                     [P(value, 9.5, t.g400, False, MONO)], anchor=MIDDLE)
    add_text(slide, x + 0.22, y + h - 0.84, w - 0.44, 0.78, [
        P("距上界 = 实测 ÷ 该强度的上界；占比按所在阶段算", 9.5, t.g500, after=1),
        P("贴斜线的只能少搬字节，贴平台的只能少算", 9.5, t.g500, after=1),
        P("通信受互联带宽限制，只列占比，不上图", 9.5, t.g500),
    ])


def version_b(slide, t):
    t.page_header(
        slide,
        "模型层 · roofline 定位",
        "整步耗时里，谁贴着上界，谁还有空间",
        "用 B 阶段的 Trace 把各类算子放上同一张 roofline——只用来选下一轮的方向",
    )
    strip(slide, t, 1.78, (
        ("卡常数　", 10, t.g900, True),
        (f"峰值算力 {BLANK} TFLOPS（{BLANK} 精度）", 9.5, t.accent, True, MONO),
        ("　|　", 10, t.g300),
        (f"实测带宽 {BLANK} GB/s", 9.5, t.accent, True, MONO),
        ("　|　", 10, t.g300),
        (f"拐点 {BLANK} FLOP/Byte", 9.5, t.g900, True, MONO),
        ("　|　", 10, t.g300),
        (f"工作点 并发 {BLANK} · 输入 {BLANK} / 输出 {BLANK}", 9.5, t.g900, True, MONO),
        ("　常数用探针实测值", 10, t.g500),
    ))
    plot_w = 8.0
    b_plot(slide, t, MX, 2.16, plot_w, 3.94)
    table_x = MX + plot_w + 0.2
    b_table(slide, t, table_x, 2.16, SW - MX - table_x, 3.94)
    t.band(slide, 6.34, 0.52, "读数", "待填：按点落在哪一段，写一句下一轮该优化谁")
    mock_footer(slide, t, "方案 B", "只需一次 Trace 加 FLOP/Byte 计数，今天就能填；"
                                   "它回答「下一轮该优化谁」，不回答「优化到哪了」")


def build(path, t):
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    for make in (version_a, version_b):
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        if t.background:
            t.background(slide)
        make(slide, t)
    prs.save(path)


if __name__ == "__main__":
    for style in sys.argv[1:2] or list(THEMES):
        target = sys.argv[2] if len(sys.argv) > 2 else OUTPUTS[style]
        build(target, THEMES[style])
        print(target)
