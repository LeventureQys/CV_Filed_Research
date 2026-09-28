# -*- coding: utf-8 -*-
"""生成《在线双态蠕变观测器补偿算法》§2.3 用的快态推导图。

与 v34_make_concept_figs.py 的两点分工：
  1) 概念图 12 张（figure/concept/）是无量纲示意图，图中不含任何数值；
     本脚本的两张推导图（figure/derive/）按设计含参数数值（r1、tau_c1、
     tau_r1、tau_eff 与归一化载荷），用于把 §2.2 的公式形态讲清楚，
     不进入概念配图口径。
  2) 曲线用与产品实现同构的显式欧拉递推（dt=0.01 s，dt 截断 [0,0.1]）
     积分；积分时把慢态 x2 与零点 z0 置零，以隔离快态本身。

输出：../figure/derive/fig06_fast_state_model.png
      ../figure/derive/fig07_fast_state_display.png
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 11

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure", "derive")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 参数（§3）
R1, TC1, TR1 = 0.12, 8.0, 6.0          # 快态幅度比 / 收敛 tau / 恢复 tau
TAU_EFF = TC1 / (1.0 + R1)             # 闭环有效时间常数 = 7.14 s
X1_INF = R1 / (1.0 + R1)               # 闭环饱和值（相对载荷）= 0.1071
DT = 0.01

C_FAST = "#1f6feb"
C_DISP = "#c0392b"
C_IN = "#9aa0a6"
C_OK = "#27ae60"
C_NOTE = "#c0392b"


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", os.path.normpath(path))


def simulate(y_fun, t_end, x1_0=0.0, r1=R1, tc1=TC1, tr1=TR1, dt=DT):
    """与产品实现同构的显式欧拉递推（x2=z0=0，隔离快态）。"""
    n = int(round(t_end / dt))
    ts = np.arange(n) * dt
    x1 = float(x1_0)
    x1s = np.empty(n)
    es = np.empty(n)
    for k, t in enumerate(ts):
        y = y_fun(t)
        e = y - x1
        if e > 0.0:
            x1 += dt * (r1 * e - x1) / tc1
        else:
            e = 0.0
            x1 -= dt * x1 / tr1
        x1s[k] = x1
        es[k] = e
    return ts, x1s, es


def box(ax, text, loc="upper left"):
    xy = {"upper left": (0.03, 0.97, "left", "top"),
          "upper right": (0.97, 0.97, "right", "top"),
          "lower left": (0.03, 0.04, "left", "bottom"),
          "lower right": (0.97, 0.04, "right", "bottom")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=xy[2], va=xy[3],
            fontsize=10, color="#333333",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fbfbfb", ec="#bbbbbb", lw=0.8))


# ================================================================ 图 13
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
for ax in axes:
    ax.grid(alpha=0.25)

# ---- (a) 有载分支：把 e 当外部输入，就是一阶惯性环节 --------------------
ax = axes[0]
t = np.linspace(0, 30, 600)
x1_open = R1 * (1.0 - np.exp(-t / TC1))
ax.plot(t, x1_open, lw=2.4, color=C_FAST,
        label=r"$x_1(t)=r_1 e\,(1-e^{-t/\tau_{c1}})$")
ax.axhline(R1, ls="--", lw=1.3, color="#888888")
ax.text(0.6, R1 + 0.004, r"饱和值 $r_1 e=0.12\,e$（幅度与载荷成正比）",
        fontsize=10, color="#555555")
for tt, lab in [(TC1, r"$\tau_{c1}=8$ s → 63%"), (3 * TC1, r"$3\tau\to95\%$")]:
    ax.axvline(tt, ls=":", lw=1.1, color=C_NOTE)
    ax.annotate(lab, xy=(tt, R1 * (1 - np.exp(-tt / TC1))),
                xytext=(tt + 0.6, R1 * (1 - np.exp(-tt / TC1)) + 0.018),
                fontsize=10, color=C_NOTE,
                arrowprops=dict(arrowstyle="->", color=C_NOTE, lw=1.1))
box(ax, "把 $e$ 当固定输入看到的形状：\n一阶惯性环节，无延迟、无振荡、无超调。", "lower right")
ax.set_xlabel("时间 (s)")
ax.set_ylabel(r"$x_1$（相对载荷）")
ax.set_title("(a) 有载分支：开环一阶滞后", fontsize=12)
ax.set_ylim(-0.01, 0.16)
ax.legend(loc="center right", fontsize=10)

# ---- (b) 闭环真相：e 自己含 x1 -----------------------------------------
ax = axes[1]
ts, x1s, es = simulate(lambda t: 1.0, 30.0)
ax.plot(ts, x1s, lw=2.4, color=C_FAST, label=r"$x_1$（快态＝扣掉的量）")
ax.plot(ts, es, lw=2.4, color="#e67e22", label=r"$e=y-x_1$（弹性估计）")
ax.plot(t, x1_open, lw=1.6, ls="--", color=C_FAST, alpha=0.55,
        label=r"若 $e$ 不下降（(a) 的开环曲线）")
ax.axhline(X1_INF, ls=":", lw=1.2, color=C_FAST)
ax.axhline(1 - X1_INF, ls=":", lw=1.2, color="#e67e22")
ax.annotate(r"$x_1\to\dfrac{r_1}{1+r_1}=10.7\%$", xy=(18, X1_INF),
            xytext=(9.5, 0.055), fontsize=11, color=C_FAST,
            arrowprops=dict(arrowstyle="->", color=C_FAST, lw=1.1))
ax.annotate(r"$e\to\dfrac{1}{1+r_1}=89.3\%$", xy=(22, 1 - X1_INF),
            xytext=(5.5, 0.72), fontsize=11, color="#e67e22",
            arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1.1))
ax.annotate(r"耦合后更快：$\tau_{\rm eff}=\dfrac{\tau_{c1}}{1+r_1}=7.14$ s",
            xy=(TAU_EFF, 0.5), xytext=(10.5, 0.30), fontsize=10.5, color=C_NOTE,
            arrowprops=dict(arrowstyle="->", color=C_NOTE, lw=1.1))
box(ax, "公式里 $e=y-x_1$ 自己含 $x_1$：\n快态长得越快，$e$ 掉得越快，目标随之变小。", "lower right")
ax.set_xlabel("时间 (s)")
ax.set_ylabel("相对载荷")
ax.set_title("(b) 闭环真相：$e$ 不是外部输入", fontsize=12)
ax.set_ylim(0, 1.08)
ax.legend(loc="center right", fontsize=9.5)

# ---- (c) 空载分支：指数恢复，且不清零 ----------------------------------
ax = axes[2]
t = np.linspace(0, 20, 600)
ax.plot(t, X1_INF * np.exp(-t / TR1), lw=2.4, color=C_OK,
        label=r"本算法：$x_1(t)=x_1(0)e^{-t/\tau_{r1}}$")
ax.axhline(0, lw=2.4, color=C_DISP, label="上一代：卸载即清零")
for tt in (1.0, 3.0, TR1):
    frac = np.exp(-tt / TR1)
    ax.plot([tt], [X1_INF * frac], "o", color=C_OK, ms=5)
    ax.annotate(f"{tt:g} s → 残留 {frac*100:.0f}%", xy=(tt, X1_INF * frac),
                xytext=(tt + 0.5, X1_INF * frac + 0.008), fontsize=10, color="#1e7a45")
ax.axvline(TR1, ls=":", lw=1.1, color=C_NOTE)
box(ax, "恢复比生长更快（6 s < 7.14 s），\n但不清零：短暂卸载后残留还在，\n重载时正是这笔残留把显示顶住。",
    "upper right")
ax.set_xlabel("卸载后时间 (s)")
ax.set_ylabel(r"$x_1$（相对载荷）")
ax.set_title("(c) 空载分支：衰减但不清零", fontsize=12)
ax.set_ylim(-0.008, 0.135)
ax.legend(loc="lower left", fontsize=10)

fig.tight_layout()
save(fig, "fig06_fast_state_model.png")

# ================================================================ 图 14
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
for ax in axes:
    ax.grid(alpha=0.25)

T_LOAD1, T_HOLD1, T_UNLOAD, T_END = 2.0, 20.0, 1.5, 45.0

# ---- (d) 加载—保持—短卸载—重载 的完整时序 ------------------------------
ax = axes[0]


def y_seq(t):
    if t < T_LOAD1:
        return 0.0
    if t < T_LOAD1 + T_HOLD1:
        return 1.0
    if t < T_LOAD1 + T_HOLD1 + T_UNLOAD:
        return 0.0
    return 1.0


ts, x1s, es = simulate(y_seq, T_END)
ys = np.array([y_seq(t) for t in ts])
disp = ys - x1s
t_unload_end = T_LOAD1 + T_HOLD1 + T_UNLOAD
i_re = int(round(t_unload_end / DT))
disp_inf = 1.0 - X1_INF
band = 0.05 * disp_inf

ax.fill_between(ts, disp_inf - band, disp_inf + band, color="#f1c40f", alpha=0.22,
                label=r"终值 $\pm5\%$ 带")
ax.plot(ts, ys, lw=1.6, color=C_IN, label=r"读数 $y$（纯弹性阶跃）")
ax.plot(ts, x1s, lw=2.2, color=C_FAST, label=r"快态 $x_1$（扣掉的量）")
ax.plot(ts, disp, lw=2.4, color=C_DISP, label=r"显示 $=y-x_1$")
ax.axhline(disp_inf, ls=":", lw=1.1, color=C_DISP)
ax.annotate("首次加载：显示从 1.00 沉到 0.893（−10.7%），\n约 6.2 s 才落进带内",
            xy=(T_LOAD1 + 4.0, 0.955), xytext=(5.0, 1.035), fontsize=10,
            color=C_DISP, arrowprops=dict(arrowstyle="->", color=C_DISP, lw=1.1))
ax.annotate("重载：起点 %.3f 距终值只差 %.1f%%\n→ 一开始就在 $\\pm5\\%%$ 带内"
            % (disp[i_re], (disp[i_re] - disp_inf) / disp_inf * 100),
            xy=(t_unload_end + 0.4, disp[i_re]), xytext=(25.0, 1.02), fontsize=10,
            color="#1e7a45", arrowprops=dict(arrowstyle="->", color="#1e7a45", lw=1.1))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("相对载荷")
ax.set_title("(d) 加载—保持—卸载 1.5 s—重载", fontsize=12)
ax.set_ylim(-0.05, 1.16)
ax.legend(loc="lower left", fontsize=9.5, ncol=2)

# ---- (e) 材料确有蠕变（幅度比与模型一致）→ 显示被钉平 -------------------
ax = axes[1]


def y_creep(t):
    if t < T_LOAD1:
        return 0.0
    return 1.0 + R1 * (1.0 - np.exp(-(t - T_LOAD1) / 8.0))


ts, x1s, es = simulate(y_creep, 60.0)
ys = np.array([y_creep(t) for t in ts])
disp = ys - x1s
ax.plot(ts, ys, lw=2.0, color=C_IN, label=r"读数 $y$（阶跃 + 快相蠕变，自身在爬）")
ax.plot(ts, x1s, lw=2.0, color=C_FAST, label=r"快态 $x_1$（跟着爬，把它扣掉）")
ax.plot(ts, disp, lw=2.6, color=C_DISP, label=r"显示 $=y-x_1$（钉在弹性电平上）")
ax.axhline(1.0, ls=":", lw=1.1, color=C_DISP)
ax.axhline(1.0 + R1, ls=":", lw=1.1, color=C_IN)
ax.annotate(r"终值恰为 $\dfrac{y_\infty}{1+r_1}=1.00$：材料蠕变幅度" "\n"
            r"等于 $r_1$ 时，扣除量正好等于蠕变量",
            xy=(44, 1.0), xytext=(19.0, 0.982), fontsize=10, color=C_DISP)
ax.annotate("读数爬 12%", xy=(30, 1.09), xytext=(23.5, 1.108), fontsize=10,
            color="#555555", arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("相对载荷")
ax.set_title("(e) 材料有蠕变 → 显示被钉平", fontsize=12)
ax.set_ylim(0.96, 1.135)
ax.legend(loc="lower right", fontsize=9.5)

# ---- (f) 材料纯弹性（没有蠕变）→ 显示被误伤 -----------------------------
ax = axes[2]
ts, x1s, es = simulate(lambda t: 0.0 if t < T_LOAD1 else 1.0, 30.0)
ys = np.array([0.0 if t < T_LOAD1 else 1.0 for t in ts])
disp = ys - x1s
ax.plot(ts, ys, lw=1.8, color=C_IN, label=r"读数 $y$（纯弹性阶跃，自己不爬）")
ax.plot(ts, x1s, lw=2.2, color=C_FAST, label=r"快态 $x_1$（仍然长出来）")
ax.plot(ts, disp, lw=2.6, color=C_DISP, label=r"显示 $=y-x_1$（被扣掉 10.7%）")
ax.axhline(1.0, ls=":", lw=1.1, color=C_IN)
ax.annotate("显示下沉 −10.7%：材料其实没有蠕变，\n这是模型按时间尺度强行分账的结果",
            xy=(12.0, 0.905), xytext=(11.5, 0.952), fontsize=10, color=C_DISP,
            arrowprops=dict(arrowstyle="->", color=C_DISP, lw=1.1))
ax.set_xlabel("时间 (s)")
ax.set_ylabel("相对载荷")
ax.set_title("(f) 材料无蠕变 → 显示被误伤", fontsize=12)
ax.set_ylim(0.86, 1.06)
ax.legend(loc="lower left", fontsize=9.5)

fig.tight_layout()
save(fig, "fig07_fast_state_display.png")

# ================================================================ 数值核对
print("[参数] r1=%.2f  tau_c1=%.1f s  tau_r1=%.1f s" % (R1, TC1, TR1))
print("[解析] tau_eff = %.4f s" % TAU_EFF)
print("[解析] x1 饱和 r1/(1+r1) = %.4f；e 稳态 1/(1+r1) = %.4f"
      % (X1_INF, 1 - X1_INF))
ts, x1s, es = simulate(lambda t: 1.0, 30.0)
print("[数值] t = tau_eff = %.2f s 时 x1 = %.4f（应为 %.4f）"
      % (TAU_EFF, x1s[int(TAU_EFF / DT)], X1_INF * (1 - np.exp(-1.0))))
print("[数值] 首次加载落进 ±5%% 带：x1 需 >= %.4f，实现于 t = %.2f s"
      % (X1_INF - 0.05 * (1 - X1_INF),
         ts[int(np.argmax(x1s >= X1_INF - 0.05 * (1 - X1_INF)))]))
print("[数值] 卸载 1.5 s 后 x1 残留 %.1f%%" % (np.exp(-1.5 / TR1) * 100))
