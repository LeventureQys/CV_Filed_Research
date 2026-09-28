# -*- coding: utf-8 -*-
"""论文正式配图共用的绘图层设置（Agg 后端 + 中文字体 + 统一保存）。"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

from pd import FIG                                                     # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.autolayout"] = False
plt.rcParams["savefig.facecolor"] = "white"

# 全文统一配色
C_RAW = "#8c8c8c"          # 原始（无补偿）
C_ALG = "#C0392B"          # 本算法
C_ALT = "#2980B9"          # 对比档 / 辅助量
C_FAST = "#E67E22"         # 快相 / 免责期
C_SLOW = "#16A085"         # 慢相 / 蠕变扣除
C_GRAY = "#5D6D7E"
C_FILL = "#F5B041"


def save_figure(figure, name):
    """统一保存：dpi=170、tight、保存后关闭。返回绝对路径。"""
    out = os.path.join(FIG, name)
    figure.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(f"  -> {out}")
    return out
