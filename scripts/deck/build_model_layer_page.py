"""The model-layer page: one decode step's Trace against its bounds, beside the candidates.

One script, two looks: the 1024 程序员节 template (build_kunlunxin_deck) and the original white
deck (build_review_deck). Every page is drawn against a Theme taken from that deck's own
module, so the two versions cannot drift apart and each mark takes the accent its deck already
gives the same meaning on page 7.

The first page is page 7's shape at the model layer, for Qwen3-VL-30B-A3B on one M100: where a
decode step's time goes and how far each category sits from its own bound (left), what the
candidates did about it (right), and how the delivery holds up across concurrency (bottom).
The second page is a backup that keeps every candidate's full wording in a wide table.

    python build_model_layer_page.py                    # both styles, default file names
    python build_model_layer_page.py review out.pptx    # one style
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass

import build_kunlunxin_deck as kunlunxin
import build_review_deck as review
from build_agent_deck import (
    CENTER,
    CW,
    DASH,
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
from pptx import Presentation
from pptx.util import Inches


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
    ship: str
    drop: str
    page_header: Callable
    band: Callable
    background: Callable | None


def _theme(deck, background):
    # Shipped and dropped verdicts take the colours page 7's record table gives them, so a
    # reader moving between the two pages keeps one colour per meaning.
    return Theme(
        g900=deck.G900, g700=deck.G700, g500=deck.G500, g400=deck.G400, g300=deck.G300,
        g200=deck.G200, g100=deck.G100, panel=deck.G050, accent=deck.AGENT, tint=deck.TINT,
        ship=deck.SHIP_C, drop=deck.DROP_C,
        page_header=deck.page_header, band=deck.band, background=background,
    )


THEMES = {"1024": _theme(kunlunxin, kunlunxin.add_background), "review": _theme(review, None)}
OUTPUTS = {"1024": "model_layer_page_1024.pptx", "review": "model_layer_page.pptx"}


def scope_strip(slide, t):
    add_box(slide, MX, 1.78, CW, 0.32, fill=t.panel)
    add_text(slide, MX + 0.2, 1.78, CW - 0.4, 0.32, [runs([
        ("冻结口径　", 10, t.g900, True),
        ("Qwen3-VL-30B-A3B-Instruct-channel-fp8", 9.5, t.accent, True, MONO),
        ("　128 专家 / top-8 · 48 层 · M100 单卡 TP1 · qsx8_lever2 env · 1K/1K · "
         "CG-on + overlap-on", 9.5, t.g700),
        ("　主指标 持续输出 tok/s", 10, t.g500),
    ])], anchor=MIDDLE)


def note(slide, t, text, lead=None):
    parts = [(f"{lead}　", 10, t.accent, True)] if lead else []
    add_text(slide, MX, Y_FOOT, CW, 0.28, [runs([*parts, (text, 10, t.g500)])], anchor=MIDDLE)


# ------------------------------------------------------------------ the page: Trace + candidates
# One decode step at c32, 1K+1K, with CUDA Graph off so each kernel is timed on its own; times
# are torch-profiler device time per step. Bytes come from config.json, not from the profiler,
# so a category whose time matches bytes ÷ bandwidth cross-checks both. Expert GEMM bytes were
# not counted -- its bound is back-derived from its own time -- so it is drawn as such and gets
# no ratio.
BANDWIDTH = 1100  # GB/s, measured on the card
DEVICE = ("24.7", "22.2")
WALL = ("27.25", "23.90")  # c32 TPOT, measured with CUDA Graph on and overlap
EXPERT_ALL_GB = 28.99  # every one of the 128 experts read once per step, fp8, all 48 layers
TRACE = (
    # (group, category, baseline ms, shipped ms, GB moved per step, candidates aimed at it)
    ("bound", "专家 GEMM（fp8）", "8.46", "8.33", None, "3"),
    ("bound", "Attention", "2.83", "2.82", 3.221, "—"),
    ("bound", "稠密 GEMM（QKV/O/头）", "2.17", "2.15", 2.434, "—"),
    # Routing runs inside this category's measured time, so its bytes join the bound.
    ("glue", "专家重排与合并（含路由）", "3.1", "3.1", 0.101 + 0.025, "2 · 4 · 5"),
    ("glue", "逐元素 + 量化", "3.51", "1.75", 0.047, "1"),
    ("glue", "3D-RoPE（多模态）", "1.86", "1.87", 0.028, "—"),
    ("glue", "Norm 与其他小核", "~1.0", "~1.0", 0.025, "—"),
)
UNTOUCHED = "3D-RoPE（多模态）"
CANDIDATES = (
    # (number, change, screening and re-test on one line, verdict, kind)
    ("1", "融合 swiglu + 动态量化",
     "C 单核 1.75 ms/步（原 3.51）· D device 24.7 → 22.2 ms", "交付", "ship"),
    ("2", "量化 + shuffle 换 SDNN 恒定核",
     "C 微基准 c128 5.43×，cos = 1.0 · D 吞吐 +3%，TPOT −3.4%", "交付", "ship"),
    ("3", "grouped-GEMM group-mode 四个旋钮",
     "C 四模式同速；force_nogroup −20%；no_group cos 0.000", "① 挡一个，③ 挡其余", "drop"),
    ("4", "FFN 输出合并换 shuffle_mul_sum_sdnn", "C 0.61×，更慢", "③ 未过", "drop"),
    ("5", "路由 softmax + topk + renorm 融合",
     "C 内核 2×，Δ 3e-8 · D 只占 ~0.9% TPOT", "③ 过但不值当，未接线", "idle"),
)
# The same 1K+1K peak sweep for baseline and delivery, in sustained output tok/s.
CONCURRENCY = (
    ("c1", 94.9, 104.3), ("c32", 959.6, 1175.7), ("c64", 1199.9, 1567.7),
    ("c96", 1307.3, 1768.8), ("c128", 1370.0, 1883.0),
)
TRACED, PEAK = "c32", "c128"
ROW, HEAD = 0.23, 0.21


def _verdict(t, kind):
    return {"ship": t.ship, "drop": t.drop, "idle": t.g500}[kind]


def _aimed(t, aimed):
    """The candidate numbers of one Trace row, each in its verdict's colour."""
    if aimed == "—":
        return [(aimed, 9.5, t.g400, False, "Arial")]
    kinds = {number: kind for number, *_rest, kind in CANDIDATES}
    parts = []
    for number in aimed.split(" · "):
        if parts:
            parts.append((" · ", 9.5, t.g400, False, "Arial"))
        parts.append((number, 9.5, _verdict(t, kinds[number]), True, "Arial"))
    return parts


def _ms(text):
    return float(text.lstrip("~"))


def _bound(gb):
    return gb / BANDWIDTH * 1e3


def _span(base, shipped):
    return shipped if abs(_ms(base) - _ms(shipped)) < 0.05 else f"{base} → {shipped}"


def _distance(base, shipped, gb):
    if gb is None:
        return "反推"
    before, after = _ms(base) / _bound(gb), _ms(shipped) / _bound(gb)
    if after < 1.05:
        return "≈1×"
    return f"{before:.0f}× → {after:.0f}×" if before > after * 1.1 else f"{after:.0f}×"


def _derived():
    """Device time no category claims, and wall time the device sits idle; both derived."""
    named = [sum(_ms(row[index]) for row in TRACE) for index in (2, 3)]
    rest = [f"{float(total) - part:.2f}" for total, part in zip(DEVICE, named, strict=True)]
    idle = [f"{float(wall) - float(device):.2f}"
            for wall, device in zip(WALL, DEVICE, strict=True)]
    return (("未归类（推导）", *rest), ("device 空闲（推导）", *idle))


def _group(name):
    rows = [row for row in TRACE if row[0] == name]
    base, shipped = (sum(_ms(row[index]) for row in rows) for index in (2, 3))
    bounds = [row[4] for row in rows]
    ratio = None if None in bounds else shipped / sum(_bound(gb) for gb in bounds)
    return rows, base, shipped, shipped / float(DEVICE[1]), ratio


def _trace_row(slide, t, x, w, top, scale, name, base, shipped, gb, aimed, kind):
    bar_x, bar_y, bar_h = x + 2.00, top + (ROW - 0.13) / 2, 0.13
    length, before = _ms(shipped) * scale, _ms(base) * scale
    add_text(slide, x + 0.22, top, 1.72, ROW, [P(name, 9.5, t.g700)], anchor=MIDDLE)
    if kind == "derived":
        add_box(slide, bar_x, bar_y, length, bar_h, fill=t.g200)
    elif gb is None:
        add_box(slide, bar_x, bar_y, length, bar_h, fill=t.g200, line=t.g400, dash=DASH)
        touched = _ms(shipped) / _bound(EXPERT_ALL_GB) * 128
        add_text(slide, bar_x + 0.08, top, length - 0.14, ROW,
                 [P(f"反推：每层至多约 {touched:.0f}/128 个专家", 8.5, t.g700)], anchor=MIDDLE)
    else:
        add_box(slide, bar_x, bar_y, length, bar_h, fill=t.accent)
        add_box(slide, bar_x, bar_y, max(0.02, min(_bound(gb) * scale, length)), bar_h,
                fill=t.g400)
    # A saving under ~0.2 ms is within run-to-run noise and would only draw a stray tick.
    if before - length > 0.05:
        add_box(slide, bar_x + length, bar_y, before - length, bar_h,
                line=t.g400, line_w=0.75, dash=DASH)
    if name == UNTOUCHED:
        add_text(slide, bar_x + length + 0.08, top, 1.52, ROW,
                 [P("← 离上界最远，还没动过", 9, t.accent, True)], anchor=MIDDLE)
    distance = "—" if kind == "derived" else _distance(base, shipped, gb)
    for cell_x, cell_w, text in ((x + 4.12, 0.92, _span(base, shipped)),
                                 (x + 5.10, 0.82, distance)):
        quiet = text in ("—", "反推")
        add_text(slide, cell_x, top, cell_w, ROW,
                 [P(text, 9.5, t.g400 if quiet else t.g900, False, MONO, align=RIGHT)],
                 anchor=MIDDLE)
    add_text(slide, x + 5.98, top, 0.68, ROW, [runs(_aimed(t, aimed), align=CENTER)],
             anchor=MIDDLE)


def trace_panel(slide, t, x, y, w, h):
    add_box(slide, x, y, w, h, fill=t.panel)
    add_text(slide, x + 0.22, y + 0.05, w - 0.44, 0.26, [runs([
        ("每类算子离上界多远", 12.5, t.g900, True),
        (f"　{TRACED} 纯 decode · CG-off · ms/步 · 上界 = 访存字节 ÷ {BANDWIDTH} GB/s", 9.5,
         t.g500),
    ])], anchor=MIDDLE)
    add_text(slide, x + 0.22, y + 0.31, 1.72, 0.22, [P("类别", 9, t.g400)], anchor=MIDDLE)
    add_text(slide, x + 2.00, y + 0.31, 2.05, 0.22, [runs([
        ("■", 9, t.g400), (" 上界内　", 9, t.g400), ("■", 9, t.accent), (" 超出　", 9, t.g400),
        ("▭", 9, t.g400), (" 交付省掉", 9, t.g400),
    ])], anchor=MIDDLE)
    for cell_x, cell_w, align, head in ((x + 4.12, 0.92, RIGHT, "基线 → 交付"),
                                        (x + 5.10, 0.82, RIGHT, "离上界"),
                                        (x + 5.98, 0.68, CENTER, "候选")):
        add_text(slide, cell_x, y + 0.31, cell_w, 0.22,
                 [P(head, 9, t.g400, align=align)], anchor=MIDDLE)
    add_line(slide, x + 0.22, y + 0.55, x + w - 0.22, y + 0.55, t.g200, 0.75)

    derived = _derived()
    longest = max([_ms(v) for row in TRACE for v in row[2:4]]
                  + [float(v) for row in derived for v in row[1:]])
    scale = 2.05 / longest
    top = y + 0.59
    for group in ("bound", "glue"):
        rows, base, shipped, share, ratio = _group(group)
        figures = f"{base:.2f} → {shipped:.2f} ms · 占 device {share:.0%}"
        if group == "bound":
            head = [("贴住上界", 9.5, t.g900, True), (f"　读权重 / 读 KV · {figures}", 9, t.g500)]
        else:
            head = [("glue", 9.5, t.accent, True),
                    (f"　离上界 {ratio:.0f}× · {figures} · 发射与 occupancy 开销", 9, t.g500)]
        add_text(slide, x + 0.22, top, w - 0.44, HEAD, [runs(head)], anchor=MIDDLE)
        top += HEAD
        for _group_name, name, base_ms, shipped_ms, gb, aimed in rows:
            _trace_row(slide, t, x, w, top, scale, name, base_ms, shipped_ms, gb, aimed, group)
            top += ROW
    add_line(slide, x + 0.22, top + 0.02, x + w - 0.22, top + 0.02, t.g200, 0.75)
    top += 0.04
    device_drop = 1 - float(DEVICE[1]) / float(DEVICE[0])
    add_text(slide, x + 0.22, top, w - 0.44, HEAD, [runs([
        ("整步", 9.5, t.g900, True),
        (f"　device {' → '.join(DEVICE)} ms（−{device_drop:.0%}）· 墙钟 = {TRACED} TPOT "
         f"{' → '.join(WALL)} ms", 9, t.g500),
    ])], anchor=MIDDLE)
    top += HEAD
    for name, base_ms, shipped_ms in derived:
        _trace_row(slide, t, x, w, top, scale, name, base_ms, shipped_ms, None, "—", "derived")
        top += ROW


def candidate_panel(slide, t, x, y, w, h):
    add_box(slide, x, y, w, h, fill=t.panel)
    add_text(slide, x + 0.22, y + 0.08, w - 0.44, 0.28, [runs([
        ("候选明细", 12.5, t.g900, True), ("　五组 · 编号对应左表「候选」列", 10, t.g500),
    ])], anchor=MIDDLE)
    verdict_w = 1.50
    for index, (number, change, evidence, verdict, kind) in enumerate(CANDIDATES):
        top = y + 0.46 + index * 0.58
        shipped = kind == "ship"
        if shipped:
            add_box(slide, x + 0.14, top, w - 0.28, 0.54, fill=t.tint)
        add_text(slide, x + 0.22, top + 0.04, 0.30, 0.24,
                 [P(number, 10, _verdict(t, kind), True, "Arial")], anchor=MIDDLE)
        add_text(slide, x + 0.56, top + 0.04, w - 0.78 - verdict_w, 0.24,
                 [P(change, 9.5, t.g900 if shipped else t.g700, shipped)], anchor=MIDDLE)
        add_text(slide, x + w - 0.22 - verdict_w, top + 0.04, verdict_w, 0.24,
                 [P(verdict, 9.5, _verdict(t, kind), shipped, align=RIGHT)], anchor=MIDDLE)
        add_text(slide, x + 0.56, top + 0.29, w - 0.78, 0.22,
                 [P(evidence, 9, t.g500)], anchor=MIDDLE)


def concurrency_strip(slide, t, y, h):
    lead_w = 2.9
    add_text(slide, MX, y, lead_w, h, [
        P("同一交付，换个并发", 11.5, t.g900, True, after=2),
        P("持续输出 tok/s，并发越高收益越大；", 9.5, t.g500, after=0),
        P(f"{TRACED} 是左表 Trace 的工作点", 9.5, t.g500),
    ], anchor=MIDDLE)
    gap = 0.14
    x0 = MX + lead_w + 0.1
    cell_w = (SW - MX - x0 - (len(CONCURRENCY) - 1) * gap) / len(CONCURRENCY)
    for index, (label, base, shipped) in enumerate(CONCURRENCY):
        cx = x0 + index * (cell_w + gap)
        hero = label == PEAK
        tag = {TRACED: f"{label} · Trace 工作点", PEAK: f"{label} · 峰值"}.get(label, label)
        add_box(slide, cx, y, cell_w, h, fill=t.tint if hero else WHITE,
                line=None if hero else t.g200)
        add_text(slide, cx + 0.16, y + 0.03, cell_w - 0.32, 0.19,
                 [P(tag, 9.5, t.accent if hero else t.g500, hero, MONO)], anchor=MIDDLE)
        add_text(slide, cx + 0.16, y + 0.22, cell_w - 0.32, 0.22,
                 [P(f"+{shipped / base - 1:.1%}", 15, t.g900, True, "Arial")], anchor=MIDDLE)
        add_text(slide, cx + 0.16, y + 0.44, cell_w - 0.32, 0.16,
                 [P(f"{base:.1f} → {shipped:.1f}", 9, t.g500, False, MONO)], anchor=MIDDLE)


def model_page(slide, t):
    t.page_header(
        slide,
        "模型层 · 一次完整的迭代记录",
        "30B-A3B 单卡：10 个候选，5 个过三道门，峰值吞吐 +37%",
        f"Trace 与按 config 算的字节账独立对上——Attention、稠密 GEMM 都贴在 {BANDWIDTH} GB/s 上",
    )
    scope_strip(slide, t)
    trace_w = 6.88
    trace_panel(slide, t, MX, 2.16, trace_w, 3.40)
    list_x = MX + trace_w + 0.2
    candidate_panel(slide, t, list_x, 2.16, SW - MX - list_x, 3.40)
    concurrency_strip(slide, t, 5.64, 0.62)
    t.band(slide, 6.34, 0.52, "读数",
           "GEMM 与 Attention 已贴住上界，glue 的时间几乎全是开销——交付只动了 glue，没碰一个 GEMM")
    note(slide, t, "Trace 关 CUDA Graph 逐核计时，吞吐与 TPOT 是 CG-on + overlap：占比与开不开图"
                   "无关，绝对耗时受影响；空闲 = TPOT − device 跨了两种口径", lead="口径差")


# -------------------------------------------------------------------- backup: the full table
# The same run with every candidate's full wording. Screening (C) was timed per kernel or
# micro-benchmark and re-testing (D) per step or end to end, so each column keeps its own units.
BACKUP_ROWS = (
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
BACKUP_FUNNEL = (("候选", "10"), ("过三道门", "5"), ("交付", "2"))


def backup_peak(slide, t, x, y, w, h):
    base, shipped = CONCURRENCY[-1][1:]
    add_box(slide, x, y, w, h, fill=t.panel)
    add_text(slide, x + 0.22, y + 0.08, w - 0.44, 0.28, [runs([
        ("峰值吞吐", 12.5, t.g900, True), ("　持续输出 tok/s", 10, t.g500),
    ])], anchor=MIDDLE)
    add_text(slide, x + 0.22, y + 0.40, w - 0.44, 0.52,
             [P(f"+{shipped / base - 1:.0%}", 28, t.ship, True, "Arial")], anchor=MIDDLE)
    floor, tallest = y + 2.58, 1.38
    for index, (label, value, color) in enumerate(
        (("基线", base, t.g300), ("交付", shipped, t.ship))
    ):
        bx, bar_h = x + 0.50 + index * 1.30, tallest * value / shipped
        add_box(slide, bx, floor - bar_h, 0.70, bar_h, fill=color)
        add_text(slide, bx - 0.25, floor - bar_h - 0.28, 1.20, 0.24,
                 [P(f"{value:.0f}", 11.5, t.g900, True, "Arial", align=CENTER)], anchor=MIDDLE)
        add_text(slide, bx - 0.25, floor + 0.07, 1.20, 0.22,
                 [P(label, 9.5, t.g700, align=CENTER)], anchor=MIDDLE)
    add_line(slide, x + 0.30, floor, x + w - 0.30, floor, t.g300, 1.0)
    add_text(slide, x + 0.22, y + h - 0.34, w - 0.44, 0.24,
             [P(f"基线峰值在 {PEAK}，1K/1K", 9, t.g500)], anchor=MIDDLE)


def backup_table(slide, t, x, y, w, h):
    add_box(slide, x, y, w, h, fill=t.panel)
    add_text(slide, x + 0.22, y + 0.08, w - 0.44, 0.28, [runs([
        ("候选明细", 12.5, t.g900, True), ("　五组，全部为单卡 TP1 口径", 10, t.g500),
    ])], anchor=MIDDLE)
    columns = ((x + 0.22, 2.90), (x + 3.22, 2.19), (x + 5.51, 1.40), (x + 7.01, 1.70))
    for (cell_x, cell_w), head in zip(columns, ("改动", "C 筛选", "D 复验", "结论"), strict=True):
        add_text(slide, cell_x, y + 0.40, cell_w, 0.24, [P(head, 9, t.g400)], anchor=MIDDLE)
    add_line(slide, x + 0.22, y + 0.66, x + w - 0.22, y + 0.66, t.g200, 0.75)

    top = y + 0.72
    for change, screened, retested, verdict, kind in BACKUP_ROWS:
        cells = (change, screened, retested, verdict)
        row_h = 0.14 + 0.16 * max(cell.count("\n") + 1 for cell in cells)
        shipped = kind == "ship"
        if shipped:
            add_box(slide, x + 0.14, top, w - 0.28, row_h, fill=t.tint)
        colors = (t.g900 if shipped else t.g700, t.g700, t.g700, _verdict(t, kind))
        for (cell_x, cell_w), cell, color, bold in zip(
            columns, cells, colors, (False, False, False, shipped), strict=True
        ):
            ink = t.g400 if cell == "—" else color
            add_text(slide, cell_x, top, cell_w, row_h,
                     [P(line, 9.5, ink, bold) for line in cell.split("\n")], anchor=MIDDLE)
        top += row_h + 0.04


def backup_funnel(slide, t, y, h):
    lead_w = 2.9
    add_text(slide, MX, y, lead_w, h, [
        P("三道门的漏斗", 11.5, t.g900, True, after=2),
        P("由便宜到贵，任一道不过即停，", 9.5, t.g500, after=0),
        P("数值不对、更慢的当场挡掉", 9.5, t.g500),
    ], anchor=MIDDLE)
    gap = 0.46
    x0 = MX + lead_w + 0.1
    cell_w = (SW - MX - x0 - (len(BACKUP_FUNNEL) - 1) * gap) / len(BACKUP_FUNNEL)
    for index, (name, count) in enumerate(BACKUP_FUNNEL):
        cx = x0 + index * (cell_w + gap)
        last = index == len(BACKUP_FUNNEL) - 1
        add_box(slide, cx, y, cell_w, h, fill=t.tint if last else WHITE,
                line=None if last else t.g200)
        add_text(slide, cx + 0.2, y + 0.08, cell_w - 0.4, 0.22,
                 [P(name, 9.5, t.accent if last else t.g500, last)], anchor=MIDDLE)
        add_text(slide, cx + 0.2, y + 0.32, cell_w - 0.4, 0.30,
                 [P(count, 16, t.g900, True, "Arial")], anchor=MIDDLE)
        if index:
            add_arrow(slide, cx - gap + 0.1, y + h / 2, cx - 0.1, y + h / 2, t.g400, 1.25)


def backup_page(slide, t):
    t.page_header(
        slide,
        "模型层 · 候选明细（备份）",
        "30B-A3B 单卡：10 个候选，5 个过三道门，峰值吞吐 +37%",
        "判定全部来自 M100 真机——数值不对、更慢的当场挡掉，占比太小的不接线",
    )
    scope_strip(slide, t)
    peak_w = 3.0
    backup_peak(slide, t, MX, 2.16, peak_w, 3.30)
    table_x = MX + peak_w + 0.2
    backup_table(slide, t, table_x, 2.16, SW - MX - table_x, 3.30)
    backup_funnel(slide, t, 5.56, 0.68)
    t.band(slide, 6.34, 0.52, "读数",
           "两个交付都在替换小核；闭源 grouped-GEMM 的旋钮调不动，占比不到 1% 的核再快也不接线")
    note(slide, t, "C 列是内核或微基准读数，D 列是整步或端到端读数，两列口径不同、不能互相换算；"
                   "表中全部为 M100 单卡 TP1")


def build(path, t):
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    for make in (model_page, backup_page):
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
