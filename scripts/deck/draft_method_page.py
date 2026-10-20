"""Draft of the method page: how one decode step's numbers are produced, and how they self-check.

The flow runs along the top as four stages so the whole method reads in three seconds; each
stage's detail and its own self-check hang directly under its header, so the page carries the
process and the detail at once. The bottom row is the payoff: one ledger, three different
outcomes -- it confirmed the method on attention, it found the target on RoPE, and it failed
its own check on expert GEMM. Every figure comes from the frozen profile-000 ledger.

    python draft_method_page.py [out.pptx]
"""

import sys

from build_agent_deck import (
    CW,
    MIDDLE,
    MONO,
    MX,
    RIGHT,
    SH,
    SW,
    P,
    add_arrow,
    add_box,
    add_line,
    add_text,
    runs,
)
from build_model_layer_page import THEMES
from pptx import Presentation
from pptx.util import Inches

T = THEMES["review"]
TITLE_BAR, TITLE_INK = "306FFF", "000000"
OUTPUT = "../../slides/method_page_draft.pptx"

TITLE = "M100 Agent：热点是怎么定位出来的"
SCOPE = (
    ("诊断负载　", "batch 32 · 输入 1024 · 生成 64 · 取 20 个 decode step · 实测平均 KV 1034.5"),
    ("　口径　", "采样进程关闭 CUDA Graph 与 overlap，只用于定位，不参与任何收益计算"),
)

# (mark, name, ((lead, (line, ...)), ...), check head, ((label, value), ...))
# A value-less check row spans the full column width and takes the accent colour.
STAGES = (
    ("①", "采 Trace", (
        ("独立进程，用完即弃", ("发 start / stop，跑固定诊断请求",)),
        ("关 CUDA Graph 与 overlap", ("为的是每个 kernel 能单独计时",)),
        ("正式压测另起一套", ("保留冻结配置、关掉 profiler",)),
    ), "一个 step 三个时间 · ms", (
        ("采样进程 · 主机", "180.89"),
        ("采样进程 · 设备 kernel", "23.92"),
        ("真实服务 · 出字间隔 p95", "29.39"),
    )),
    ("②", "归因到源码", (
        ("四步定位", ("step 窗口 → CPU 最内层源码 scope", "→ launch 关联 → 设备 kernel")),
        ("源码 scope 优先", ("kernel 名只在没有 scope 时兜底",)),
        ("不重复计时", ("CPU 标注与 GPU 事件不叠加",)),
    ), "各类时间能不能相加 · ms", (
        ("kernel 时间总和", "23.920670"),
        ("各 kernel 区间并集", "23.920008"),
        ("重叠 0.0028% · 未归因 0.0106%", ""),
    )),
    ("③", "算 Roofline 账", (
        ("字节从 config 解析", ("与设备计时是独立的两条路",)),
        ("按精度分域计费", ("专家 FP8 记 1 字节，其余 2 字节",)),
        ("专家字节用路由实测", ("实测均值 40.0，均匀假设 110.8",)),
    ), "下界取三者中最大的", (
        ("算力下界 = FLOPs ÷ 精度上界", ""),
        ("访存下界 = 最小字节 ÷ 带宽", ""),
        ("离上限 = 实测 ÷ 下界", ""),
    )),
    ("④", "排序并证伪", (
        ("按离上限倍数排序", ("RoPE 280× · 重排 45×", "Norm 33× · 逐元素 22×")),
        ("账本只排序，不定因", ("「发射太多」是假设，不是结论",)),
        ("先找反例再动手", ("probe 先证伪，再提补丁",)),
    ), "改之前先拿反例验一次 · µs", (
        ("连续合成输入", "65.86"),
        ("真实 stride · 回退", "120.56"),
        ("重排后 native 恢复", "68.60"),
    )),
)

OUTCOMES = (
    ("验证了方法", "ship", "decode attention", (
        "按 config 算 KV 读取量，折合下界 2.958 ms；",
        "profiler 实测 3.086 ms，两条独立的路只差 4.3%",
    )),
    ("找到了目标", "accent", "多模态 3D-RoPE", (
        "每步 912 次调用，每次只搬 30.3 KB、耗 7.9 µs，",
        "等效 3.9 GB/s——这张卡的带宽是 1100 GB/s",
    )),
    ("自检判失败", "drop", "专家 GEMM", (
        "实测 5.843 ms 比下界 8.242 ms 还快 29%；",
        "不是跑得超神，是账本高估了访存量，隔离待校准",
    )),
)
BAND = "账本能排出顺序，不能证明瓶颈；每个方向都先用反例实验证伪，再动手改"
FOOT = (
    "HBM 1100 GB/s、BF16 122.5 TFLOPS 为历史测量值，FP8 294.912 为标称值，SFU 上界至今未测得",
    "逻辑最小字节量不是物理 HBM 流量；理论可省时间不是已兑现收益",
)
LEAD_H, LINE_H, ROW_H = 0.19, 0.17, 0.21


def tone(kind):
    return {"ship": T.ship, "drop": T.drop, "accent": T.accent}[kind]


def header(slide):
    add_box(slide, MX, 0.44, 0.07, 0.38, fill=TITLE_BAR)
    add_text(slide, MX + 0.22, 0.33, 10.4, 0.6, [P(TITLE, 26, TITLE_INK, True)], anchor=MIDDLE)


def scope(slide, y):
    add_box(slide, MX, y, CW, 0.34, fill=T.panel)
    parts = []
    for lead, text in SCOPE:
        parts.append((lead, 10.5, T.g900, True))
        parts.append((text, 10, T.g700))
    add_text(slide, MX + 0.2, y, CW - 0.4, 0.34, [runs(parts)], anchor=MIDDLE)


def stage_column(slide, x, w, y, head_h, body_h, stage):
    mark, name, details, check_head, check_rows = stage
    add_box(slide, x, y, w, head_h, fill=T.tint)
    add_text(slide, x + 0.16, y, 0.30, head_h, [P(mark, 13, T.accent, True)], anchor=MIDDLE)
    add_text(slide, x + 0.48, y, w - 0.62, head_h, [P(name, 12.5, T.g900, True)], anchor=MIDDLE)

    top = y + head_h
    add_box(slide, x, top, w, body_h, fill=T.panel)
    cursor = top + 0.10
    for lead, lines in details:
        add_text(slide, x + 0.17, cursor, w - 0.34, LEAD_H, [P(lead, 9.5, T.g900, True)],
                 anchor=MIDDLE)
        for index, line in enumerate(lines):
            add_text(slide, x + 0.17, cursor + LEAD_H + index * LINE_H, w - 0.34, LINE_H,
                     [P(line, 9, T.g500)], anchor=MIDDLE)
        cursor += LEAD_H + LINE_H * len(lines) + 0.05

    check_y = top + body_h - (0.26 + ROW_H * len(check_rows))
    add_line(slide, x + 0.17, check_y - 0.07, x + w - 0.17, check_y - 0.07, T.g200, 0.75)
    add_text(slide, x + 0.17, check_y, w - 0.34, 0.22, [runs([
        ("自检　", 9, T.accent, True), (check_head, 9, T.g700),
    ])], anchor=MIDDLE)
    for index, (label, value) in enumerate(check_rows):
        row_y = check_y + 0.24 + index * ROW_H
        if not value:
            add_text(slide, x + 0.17, row_y, w - 0.34, ROW_H,
                     [P(label, 8.5, T.accent)], anchor=MIDDLE)
            continue
        add_text(slide, x + 0.17, row_y, w - 0.34 - 0.82, ROW_H,
                 [P(label, 8.5, T.g500)], anchor=MIDDLE)
        add_text(slide, x + w - 0.99, row_y, 0.82, ROW_H,
                 [P(value, 9, T.g900, True, MONO, align=RIGHT)], anchor=MIDDLE)


def outcome_card(slide, x, w, y, h, outcome):
    label, kind, subject, lines = outcome
    color = tone(kind)
    add_box(slide, x, y, w, h, fill=T.panel)
    add_box(slide, x, y, 0.06, h, fill=color)
    add_text(slide, x + 0.26, y + 0.07, w - 0.48, 0.25, [runs([
        (label, 12, color, True), (f"　{subject}", 10.5, T.g900, True),
    ])], anchor=MIDDLE)
    for index, line in enumerate(lines):
        add_text(slide, x + 0.26, y + 0.34 + index * 0.22, w - 0.48, 0.22,
                 [P(line, 9.5, T.g700)], anchor=MIDDLE)


def method_page(slide):
    header(slide)
    scope(slide, 1.06)

    spine_y, head_h, body_h, gap = 1.52, 0.40, 2.58, 0.22
    col_w = (CW - gap * (len(STAGES) - 1)) / len(STAGES)
    for index, stage in enumerate(STAGES):
        x = MX + index * (col_w + gap)
        stage_column(slide, x, col_w, spine_y, head_h, body_h, stage)
        if index:
            add_arrow(slide, x - gap + 0.03, spine_y + head_h / 2, x - 0.03,
                      spine_y + head_h / 2, T.accent, 1.25)

    cards_y = spine_y + head_h + body_h + 0.44
    add_text(slide, MX, cards_y - 0.30, CW, 0.26, [runs([
        ("同一套账本，三种结局", 12.5, T.g900, True),
        ("　方法可信，靠的是它敢判自己失败", 10, T.g500),
    ])], anchor=MIDDLE)
    card_gap = 0.20
    card_w = (CW - card_gap * (len(OUTCOMES) - 1)) / len(OUTCOMES)
    for index, outcome in enumerate(OUTCOMES):
        outcome_card(slide, MX + index * (card_w + card_gap), card_w, cards_y, 0.92, outcome)

    T.band(slide, cards_y + 1.06, 0.50, "口径", BAND)
    add_text(slide, MX, cards_y + 1.64, CW, 0.44, [
        runs([("天花板　", 10, T.accent, True), (FOOT[0], 9.5, T.g500)]),
        P(FOOT[1], 9.5, T.g500),
    ])


def build(path):
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    method_page(prs.slides.add_slide(prs.slide_layouts[6]))
    prs.save(path)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else OUTPUT
    build(target)
    print(target)
