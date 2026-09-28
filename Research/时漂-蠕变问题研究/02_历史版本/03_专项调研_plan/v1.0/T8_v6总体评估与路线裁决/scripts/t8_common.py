# -*- coding: utf-8 -*-
"""t8_common.py -- T8 收口席共享工具（只读；不重跑任何参数扫描）。

职责：
  1) 统一路径锚点（00-项目组织文档 §4.1 口径）；
  2) 提供 Cell/MATRIX 的写入器，强制每格带 `value + unit + source_task + source_file + column`；
  3) 提供图中文字体自检（缺字体则退回英文，禁止出方框）。

用法：被 t8_matrix.py / t8_figs.py 等 import。
"""
import os
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))
TEMP = os.path.join(ROOT, "temp")
PROG = os.path.join(ROOT, "temp", "v4.1flash", "progress")
RESULTS = os.path.join(TASK, "results")
FIGURES = os.path.join(TASK, "figures")

# 各任务 results 目录（只读引用）
T1R = os.path.join(PLAN, "T1_稳定时间定义与鲁棒性口径", "results")
T2R = os.path.join(PLAN, "T2_三阶段时间特征实测", "results")
T3R = os.path.join(PLAN, "T3_快相爬升可重复性与处理必要性", "results")
T4R = os.path.join(PLAN, "T4_两种快相形态与分支判据", "results")
T5R = os.path.join(PLAN, "T5_v6失效模式_过充与基线识别", "results")
T6R = os.path.join(PLAN, "T6_v6可重复性下降归因", "results")
T7R = os.path.join(PLAN, "T7_卸载与部分卸载时漂规律", "results")

DIM_ORDER = [
    "稳定时间",
    "稳态时漂",
    "台阶保真",
    "过充",
    "下冲",
    "重复性",
    "强扰动鲁棒性",
    "卸载行为",
    "计算量",
    "可标定性",
]

COL_ORDER = ["v3", "v5.1", "v6", "v6.1", "T3-B第三条路"]

MATRIX_FIELDS = [
    "dimension", "route", "value", "unit",
    "source_task", "source_file", "column", "caliber", "n", "note",
]


def cell(dim, route, value, unit, task, src, col, caliber="", n="", note=""):
    return {
        "dimension": dim, "route": route, "value": value, "unit": unit,
        "source_task": task, "source_file": src, "column": col,
        "caliber": caliber, "n": n, "note": note,
    }


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", path, len(rows), "rows")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    print("wrote", path)


def setup_cjk_font():
    """中文字体自检：可用则用，不可用则全图改英文。返回 True 表示可用中文。"""
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimSun"]:
        if cand in names:
            plt.rcParams["font.family"] = cand
            plt.rcParams["axes.unicode_minus"] = False
            print("CJK font OK:", cand)
            # 该字体缺 U+21D2/⇒ 等符号，统一替换为可显示的等价写法（禁止出方框）
            _install_glyph_fallback()
            return True
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    print("CJK font MISSING -> fallback to English labels")
    return False


_GLYPH_MAP = {"\u21d2": "=>", "\u21d0": "<=", "\u2192": "->", "\u2190": "<-",
              "\u2265": ">=", "\u2264": "<=", "\u2260": "!=", "\u2248": "~"}


def _install_glyph_fallback():
    """在 Text 绘制前把字体缺失的符号替换成 ASCII 等价写法（避免输出方框）。"""
    from matplotlib.text import Text

    old = Text.set_text

    def patched(self, s):
        if isinstance(s, str):
            for k, v in _GLYPH_MAP.items():
                s = s.replace(k, v)
        return old(self, s)

    Text.set_text = patched


def save_fig(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    p = os.path.join(FIGURES, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
