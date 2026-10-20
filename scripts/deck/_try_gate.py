"""Three ways to draw the gate page's right-hand panel, for picking one before editing the deck.

    python try_gate.py review out.pptx
"""

import sys

import build_kunlunxin_deck as kunlunxin
import build_review_deck as review
from build_agent_deck import (
    CENTER,
    DASH,
    MIDDLE,
    MONO,
    RIGHT,
    SH,
    SW,
    P,
    add_box,
    add_line,
    add_text,
    runs,
)
from pptx import Presentation
from pptx.util import Inches

# (number, short name, gate it stopped at or None, verdict, reason)
CANDS = (
    ("1", "激活与量化合成一个核", None, "交付", "单核 3.51 → 1.75 ms"),
    ("2", "耗时不随规模涨的核", None, "交付", "吞吐 +3%，每字延迟 −3.4%"),
    ("3", "四种分组开关", 1, "挡掉", "关掉分组后 cos 0.000，是错的"),
    ("4", "加权求和换现成核", 3, "挡掉", "0.61×，更慢"),
    ("5", "路由三个小算子融合", 3, "挡掉", "最多只值 0.9%"),
)
MARKS = ("①", "②", "③")
GATE_NAMES = ("输出未变", "质量达标", "确实更快")
PEAK = (1370.0, 1883.0)


def _deck(name):
    return {"review": review, "1024": kunlunxin}[name]


def _head(d, slide, x, y, w, sub):
    add_text(slide, x + 0.22, y + 0.06, w - 0.44, 0.28, [runs([
        ("本轮实际判定", 12.5, d.G900, True), (f"　{sub}", 10, d.G500),
    ])], anchor=MIDDLE)


def _peak(d, slide, x, y, w):
    base, shipped = PEAK
    add_text(slide, x + 0.22, y, w - 0.44, 0.26, [runs([
        ("峰值吞吐 ", 10, d.G500), (f"{base:.0f} → {shipped:.0f} tok/s", 10, d.G900, True, MONO),
        (f"　+{shipped / base - 1:.0%}", 11, d.SHIP_C, True, MONO),
    ])], anchor=MIDDLE)


def _frame(d, slide, x, y, w, h):
    if d is kunlunxin:
        d.card(slide, x, y, w, h, fill=d.G050)
        add_box(slide, x, y, 0.05, h, fill=d.MODEL_LEVEL)
    else:
        add_box(slide, x, y, w, h, fill=d.G050)


# --------------------------------------------------------------- A  停在哪一道（轨道）
def panel_track(d, slide, x, y, w, h):
    """Each candidate is a track through the three gates that stops where it was stopped."""
    _frame(d, slide, x, y, w, h)
    _head(d, slide, x, y, w, "每个候选停在哪一道")
    seg, gap = 0.74, 0.10
    track_x = x + 1.90
    for index, mark in enumerate(MARKS):
        add_text(slide, track_x + index * (seg + gap), y + 0.36, seg, 0.2,
                 [P(mark, 10, d.G400, align=CENTER)], anchor=MIDDLE)
    add_text(slide, x + w - 1.08, y + 0.36, 0.86, 0.2,
             [P("结论", 9, d.G400, align=RIGHT)], anchor=MIDDLE)

    for row, (number, name, stop, verdict, reason) in enumerate(CANDS):
        top = y + 0.60 + row * 0.62
        shipped = stop is None
        color = d.SHIP_C if shipped else d.DROP_C
        if shipped:
            add_box(slide, x + 0.14, top, w - 0.28, 0.56, fill=d.TINT)
        add_text(slide, x + 0.22, top + 0.02, 0.26, 0.26,
                 [P(number, 10, color, True, "Arial")], anchor=MIDDLE)
        add_text(slide, x + 0.50, top + 0.02, 1.40, 0.26,
                 [P(name, 9.5, d.G900 if shipped else d.G700, shipped)], anchor=MIDDLE)
        add_text(slide, x + 0.50, top + 0.28, w - 1.60, 0.24,
                 [P(reason, 8.5, d.G500)], anchor=MIDDLE)
        for gate in range(3):
            gx = track_x + gate * (seg + gap)
            passed = shipped or gate + 1 < stop
            stopped = not shipped and gate + 1 == stop
            if stopped:
                add_box(slide, gx, top + 0.08, seg, 0.14, fill=d.DROP_C)
            elif passed:
                add_box(slide, gx, top + 0.08, seg, 0.14, fill=d.SHIP_C)
            else:
                add_box(slide, gx, top + 0.08, seg, 0.14, line=d.G300, dash=DASH)
        add_text(slide, x + w - 1.08, top + 0.02, 0.86, 0.26,
                 [P(verdict, 9.5, color, shipped, align=RIGHT)], anchor=MIDDLE)
    add_text(slide, x + 0.22, y + h - 0.60, w - 0.44, 0.22, [runs([
        ("■", 8.5, d.SHIP_C), (" 过　", 8.5, d.G500), ("■", 8.5, d.DROP_C), (" 挡在这一道　", 8.5,
         d.G500), ("▭", 8.5, d.G400), (" 没往下测", 8.5, d.G500),
    ])], anchor=MIDDLE)
    _peak(d, slide, x, y + h - 0.34, w)


# --------------------------------------------------------------- B  判定矩阵
def panel_matrix(d, slide, x, y, w, h):
    """Five rows against three gates: the eye finds the one cross on each dropped row."""
    _frame(d, slide, x, y, w, h)
    _head(d, slide, x, y, w, "五组候选 × 三道门")
    col_w = 0.52
    col_x = x + w - 0.94 - 3 * col_w
    for index, mark in enumerate(MARKS):
        add_text(slide, col_x + index * col_w, y + 0.34, col_w, 0.42,
                 [P(mark, 11, d.G500, align=CENTER),
                  P(GATE_NAMES[index], 7.5, d.G400, align=CENTER)], anchor=MIDDLE)
    add_text(slide, x + w - 0.94, y + 0.34, 0.72, 0.42,
             [P("结论", 9, d.G400, align=RIGHT)], anchor=MIDDLE)
    add_line(slide, x + 0.22, y + 0.80, x + w - 0.22, y + 0.80, d.G200, 0.75)

    for row, (number, name, stop, verdict, reason) in enumerate(CANDS):
        top = y + 0.86 + row * 0.56
        shipped = stop is None
        color = d.SHIP_C if shipped else d.DROP_C
        if shipped:
            add_box(slide, x + 0.14, top, w - 0.28, 0.52, fill=d.TINT)
        add_text(slide, x + 0.22, top + 0.02, 0.26, 0.24,
                 [P(number, 10, color, True, "Arial")], anchor=MIDDLE)
        add_text(slide, x + 0.50, top + 0.02, col_x - x - 0.56, 0.24,
                 [P(name, 9.5, d.G900 if shipped else d.G700, shipped)], anchor=MIDDLE)
        add_text(slide, x + 0.50, top + 0.26, col_x - x - 0.56, 0.22,
                 [P(reason, 8.5, d.G500)], anchor=MIDDLE)
        for gate in range(3):
            if shipped or gate + 1 < stop:
                glyph, ink = "✓", d.SHIP_C
            elif gate + 1 == stop:
                glyph, ink = "✗", d.DROP_C
            else:
                glyph, ink = "·", d.G300
            add_text(slide, col_x + gate * col_w, top, col_w, 0.52,
                     [P(glyph, 13 if glyph != "·" else 15, ink, True, align=CENTER)],
                     anchor=MIDDLE)
        add_text(slide, x + w - 0.94, top + 0.02, 0.72, 0.24,
                 [P(verdict, 9.5, color, shipped, align=RIGHT)], anchor=MIDDLE)
    add_text(slide, x + 0.22, y + h - 0.58, w - 0.44, 0.22,
             [P("· 表示前一道已挡下，这一道不再测", 8.5, d.G500)], anchor=MIDDLE)
    _peak(d, slide, x, y + h - 0.34, w)


# --------------------------------------------------------------- C  漏斗
def panel_funnel(d, slide, x, y, w, h):
    """The five shrink to two, with each gate's casualties hanging off the side."""
    _frame(d, slide, x, y, w, h)
    _head(d, slide, x, y, w, "五组进，两组出")
    stages = (
        ("送进验收", 5, None, "由便宜到贵逐道验"),
        ("① 输出未变", 4, "挡掉 1 个模式", "关掉分组后 cos 0.000"),
        ("② 质量达标", 4, None, "过了 ① 的输出逐位一致"),
        ("③ 确实更快", 2, "挡掉 2 组", "0.61× 更慢；最多只值 0.9%"),
    )
    full = w - 1.90
    for index, (name, left, dropped, note) in enumerate(stages):
        top = y + 0.44 + index * 0.72
        bar_w = full * left / 5
        add_text(slide, x + 0.22, top, 1.60, 0.26, [P(name, 10.5, d.G900, True)], anchor=MIDDLE)
        add_text(slide, x + 0.22, top + 0.26, 1.60, 0.22, [P(note, 8.5, d.G500)], anchor=MIDDLE)
        add_box(slide, x + 1.90, top + 0.06, bar_w, 0.30,
                fill=d.SHIP_C if index == len(stages) - 1 else d.G300)
        add_text(slide, x + 1.98, top + 0.06, bar_w - 0.16, 0.30,
                 [P(f"{left} 组", 10, "FFFFFF", True)], anchor=MIDDLE)
        if dropped:
            add_text(slide, x + 1.90 + bar_w + 0.08, top + 0.06, w - 2.10 - bar_w, 0.30,
                     [P(dropped, 9, d.DROP_C, True)], anchor=MIDDLE)

    ship_y = y + h - 1.16
    add_box(slide, x + 0.22, ship_y, w - 0.44, 0.74, fill=d.TINT)
    add_text(slide, x + 0.40, ship_y + 0.04, w - 0.80, 0.26, [runs([
        ("交付", 11, d.G900, True), ("　两组都在替换小核", 10, d.G500),
    ])], anchor=MIDDLE)
    for index, item in enumerate(("激活与量化合成一个核", "换成耗时不随规模涨的核")):
        add_text(slide, x + 0.40, ship_y + 0.28 + index * 0.22, w - 0.80, 0.22,
                 [P(f"· {item}", 9.5, d.G700)], anchor=MIDDLE)
    _peak(d, slide, x, y + h - 0.34, w)


# --------------------------------------------------------------- D  实测样例卡片
CARDS = (
    ("这一轮试过的改动", "5 组", "同一张 M100 卡，输入输出各 1000 字"),
    ("三道都过、真正交付的", "2 组", "另外 3 组当场挡掉：1 组算错，2 组不够快"),
    ("每秒出字数", "+37%", "从每秒 1370 字涨到 1883 字"),
)


def panel_cards(d, slide, x, y, w, h):
    """Page 6's sample panel, applied to the gating result: three readings and nothing else."""
    add_box(slide, x, y, w, h, fill=d.TINT)
    add_text(slide, x + 0.26, y + 0.14, w - 0.52, 0.34,
             [runs([("这一轮的实测结果", 14, d.G900, True),
                    ("　整个模型，不是单个算子", 10.5, d.G400)])],
             anchor=MIDDLE)
    row_y = y + 0.62
    for label, value, note in CARDS:
        add_box(slide, x + 0.26, row_y, w - 0.52, 0.94, fill=d.G050, line=d.G200)
        add_text(slide, x + 0.44, row_y, w - 0.88, 0.94, [
            P(label, 10.5, d.G500, after=2),
            P(value, 19, d.G900, True, "Arial", after=2),
            P(note, 9.5, d.G400),
        ], anchor=MIDDLE)
        row_y += 1.06
    add_text(slide, x + 0.26, row_y, w - 0.52, 0.30,
             [P("每一组为什么过、为什么没过，见模型层那页", 10, d.G500)], anchor=MIDDLE)


# --------------------------------------------------------------- E/F  两张卡片
USER_CARDS = (
    ("M100 单卡 TP1", "1k输入 + 1k输出场景", None),
    ("128并发下相对基线加速", "1.37×", "1370.0 → 1883.0"),
)
FIXED_CARDS = (
    ("这一轮试过的改动", "5 组", "3 组当场挡掉，2 组交付"),
    ("128 并发下的吞吐", "+37%", "每秒 1370 → 1883 token"),
)


def _two_cards(d, slide, x, y, w, h, title, sub, cards, foot):
    add_box(slide, x, y, w, h, fill=d.TINT)
    add_text(slide, x + 0.26, y + 0.14, w - 0.52, 0.34,
             [runs([(title, 14, d.G900, True), (f"　{sub}", 10.5, d.G400)])], anchor=MIDDLE)
    row_y = y + 0.70
    for label, value, note in cards:
        lines = [P(label, 10.5, d.G500, after=2), P(value, 19, d.G900, True, "Arial", after=2)]
        if note:
            lines.append(P(note, 9.5, d.G400, False, MONO))
        add_box(slide, x + 0.26, row_y, w - 0.52, 1.10, fill=d.G050, line=d.G200)
        add_text(slide, x + 0.44, row_y, w - 0.88, 1.10, lines, anchor=MIDDLE)
        row_y += 1.22
    add_text(slide, x + 0.26, row_y, w - 0.52, 0.30, [P(foot, 10, d.G500)], anchor=MIDDLE)


def panel_user(d, slide, x, y, w, h):
    _two_cards(d, slide, x, y, w, h, "Qwen3-VL- 30B-A3B", "", USER_CARDS, "详细迭代过程见下一页")


def panel_user_fixed(d, slide, x, y, w, h):
    _two_cards(d, slide, x, y, w, h, "Qwen3-VL-30B-A3B", "单卡 · 输入输出各 1K",
               FIXED_CARDS, "每一组为什么过、为什么没过，见模型层那页")


PANELS = (("A 停在哪一道", panel_track), ("B 判定矩阵", panel_matrix),
          ("C 漏斗", panel_funnel), ("D 实测样例卡片", panel_cards),
          ("E 你这一版", panel_user), ("F 调整后", panel_user_fixed))


if __name__ == "__main__":
    style = sys.argv[1]
    d = _deck(style)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)
    original = d.verdict_panel
    for label, panel in PANELS:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        if style == "1024":
            d.add_background(slide)
        d.verdict_panel = lambda s, x, y, w, h, _p=panel: _p(d, s, x, y, w, h)
        d.gate_slide(slide)
        add_text(slide, 7.733, 1.62, 5.0, 0.22,
                 [P(label, 10, d.MODEL_LEVEL, True, MONO)], anchor=MIDDLE)
    d.verdict_panel = original
    prs.save(sys.argv[2])
    print(sys.argv[2])
