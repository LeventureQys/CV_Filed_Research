# -*- coding: utf-8 -*-
"""放大局部：13ffca @18.68 s 负载内变载 —— 把「pending 冻结」看清楚。

图分上下两栏（共用时间轴，x = 距真实加载沿）：
  上：总量（原始 / 当前实现 / **不冻结对照**）
  下：已生效扣除量（当前实现 平 vs 不冻结对照 一路涨）
四段色带：① 检测窗 ② 确认窗（负载态下=pending 冻结）③ 免责期（含 A 窗）④ 重建期

"不冻结对照"由源码补丁构造（把 pending 分支的 `self.hold = True` 改成 False），
除不冻结外它还会失去 D2 的前提（hold_comp_ 不再被记录），如实标注。

产出：figures/H5_zoom_1868_pending_freeze.png
"""
import os
import sys
import types
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
FIG = os.path.join(OUT, "figures")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FAST, AWIN, ARM, TAU_G = 3.0, 1.0, 3.0, 3.0
CSVP = os.path.join(TEMP, "变化负载", "零负载-中途切换负载-零负载-切换负载",
                    "最终测试目标", "device_001_seg000.csv")
C_DET, C_CONF, C_EX, C_AWIN, C_REB = "#f2c14e", "#8ea9db", "#f6bdbd", "#d64545", "#86c586"

src = open(os.path.join(HERE, "glm53_v51.py"), encoding="utf-8").read()
old = "        elif self.pending:\n            self.hold = True"
assert src.count(old) == 1
mod = types.ModuleType("nofreeze")
exec(compile(src.replace(old, old.replace("True", "False")), "nofreeze", "exec"), mod.__dict__)
NoFreeze = mod.GLM53v51


def run(cls, tu, Xu):
    c = cls(Xu.shape[1])
    c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = FAST, AWIN, ARM
    Y = np.empty_like(Xu)
    st, ep = [], []
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
        st.append((float(tu[i]), int(c.in_load), int(c.pending), int(c.hold),
                   int(c.fast_done), float(c.g)))
    return Y, np.array(st)


d = L.prep(CSVP)
tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
tot_s = L.med_smooth(tot, 0.5 / dtm)
peak = float(tot.max())
YA, stA = run(GLM53v51, tu, Xu)
YB, stB = run(NoFreeze, tu, Xu)
ya = L.med_smooth(YA.sum(axis=1), 0.5 / dtm)
yb = L.med_smooth(YB.sum(axis=1), 0.5 / dtm)
dedA = L.med_smooth((Xu - YA).sum(axis=1), 0.5 / dtm)
dedB = L.med_smooth((Xu - YB).sum(axis=1), 0.5 / dtm)

E = 18.68
i0 = int(np.searchsorted(tu, E))
amp = float(abs(tot_s[i0 + int(8 / dtm)] - tot_s[max(0, i0 - int(2 / dtm))]))
pen = np.where(stA[:, 2] > 0)[0]
pen = pen[pen >= i0 - 2]
p0 = int(pen[0])
p1 = p0
while p1 + 1 < len(stA) and stA[p1 + 1, 2] > 0:
    p1 += 1
t_det = tu[p0] - E
t_ep = tu[p1 + 1] - E                     # 确认窗结束 = epoch 起点（pending 消失的下一帧）
ek = p1 + 1
ex = np.where(stA[ek:, 4] == 1)[0]
t_ex = tu[ek + ex[0]] - E
t_vis = np.nan
seg = (Xu - YA).sum(axis=1)[i0:i0 + int(14 / dtm)]
t_first = t_ep + FAST
jump = float(np.median(tot_s[i0 + int(4 / dtm):i0 + int(6 / dtm)])
             - np.median(tot_s[max(0, i0 - int(2 / dtm)):i0]))
print(f"@18.68 s 台阶 {jump:+.0f}（占前级电平 {abs(jump)/abs(tot_s[i0]):.2f}）")
print(f"  ① 检测窗 0 → {t_det:.2f}s   ② 确认窗(冻结) {t_det:.2f} → {t_ep:.2f}s"
      f"（{t_ep - t_det:.2f}s）  ③ 免责期 {t_ep:.2f} → {t_ex:.2f}s  ④ 重建 {t_ex:.2f} → +{t_ex + 3*TAU_G:.1f}s")
kA0 = (Xu - YA).sum(axis=1)
kB0 = (Xu - YB).sum(axis=1)
print(f"  扣除量：命中时 {kA0[p0]:.0f} → 确认时 {kA0[p1]:.0f}（冻结期 Δ{kA0[p1]-kA0[p0]:+.0f}）| "
      f"不冻结对照同期 {kB0[p0]:.0f} → {kB0[p1]:.0f}（Δ{kB0[p1]-kB0[p0]:+.0f}）")
print(f"  显示差（不冻结 − 冻结）在确认结束时刻 = {yb[p1] - ya[p1]:+.0f} ADC "
      f"（台阶 {abs(jump):.0f}）")

# ═══════════════════════ 图：上下两栏，放大 0~16 s ═══════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15.5, 10), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.12))
x0, x1 = -2.0, 16.0
fig.suptitle("放大看「pending 冻结」：13ffca @18.68 s 的负载内加重（台阶 +6405，占电平 0.40）· 无责 3 s 档",
             fontsize=15)

BANDS = [("① 检测窗", 0.0, t_det, C_DET), ("② 确认窗 = pending 冻结", t_det, t_ep, C_CONF),
         ("③ 免责期", t_ep, t_ex, C_EX), ("④ 重建期", t_ex, t_ex + 3 * TAU_G, C_REB)]
for ax in (ax1, ax2):
    for nm, a, b, col in BANDS:
        if b > a:
            ax.axvspan(E + a, E + b, color=col, alpha=0.20, lw=0)
    ax.axvspan(E + t_ex - AWIN, E + t_ex, color=C_AWIN, alpha=0.35, lw=0)   # A 采集窗
    for t, col in ((t_det, C_DET), (t_ep, C_CONF), (t_ex, C_EX)):
        ax.axvline(E + t, color=col, ls="--", lw=1.4)
    for t in (t_ex + TAU_G, t_ex + 3 * TAU_G):
        ax.axvline(E + t, color=C_REB, ls=":", lw=1.2)
    ax.axvline(E, color="k", lw=1.6)
    ax.set_xlim(E + x0, E + x1)
    ax.grid(alpha=0.25)

w = slice(max(0, i0 + int(x0 / dtm)), min(len(tu), i0 + int(x1 / dtm)))
xx = tu[w] - E
ax1.plot(xx, tot_s[w], color="0.55", lw=1.4, label="原始读数")
ax1.plot(xx, ya[w], color="#1f77b4", lw=2.4, label="当前实现（pending 期冻结）")
ax1.plot(xx, yb[w], color="#d95f02", lw=2.0, ls="--", label="对照：不冻结（源码补丁关掉该分支）")
ax1.set_ylabel("总量 (ADC)", fontsize=11)
ax1.legend(fontsize=10.5, loc="upper left")
ax1.set_title("上：显示总量 —— 冻结与否，在这 2.5 s 内就分出了胜负", fontsize=12, loc="left", pad=8)
d_gap = float(yb[p1] - ya[p1])
JUMP = abs(jump)
ax1.annotate(f"不冻结会把 +6405 的台阶\n吃掉 {abs(d_gap):.0f} ADC（台阶 {JUMP:.0f} 的 {100*abs(d_gap)/max(JUMP,1):.0f}%）",
             xy=(E + t_ep, (yb[p1] + ya[p1]) / 2), xytext=(E + t_ep + 2.6, yb[p1] - 3000),
             fontsize=10, color="#d95f02",
             arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.2))

ax2.plot(xx, dedA[w], color="#1f77b4", lw=2.4, label="当前实现：扣除量被按住")
ax2.plot(xx, dedB[w], color="#d95f02", lw=2.0, ls="--", label="对照：不冻结，扣除量一路涨")
ax2.set_ylabel("已生效扣除量 (ADC)", fontsize=11)
ax2.set_xlabel("距真实加载沿的时间 (s)", fontsize=11)
ax2.legend(fontsize=10.5, loc="upper left")
ax2.set_title("下：已生效扣除量 —— 「冻结」指的就是这条线在 ② 段被按住不动", fontsize=12, loc="left", pad=8)
ax2.annotate(f"② 段 Δ = {kA0[p1]-kA0[p0]:+.0f} ADC\n（g 不更新、γ 不动）",
             xy=(E + (t_det + t_ep) / 2, kA0[p1]), xytext=(E + (t_det + t_ep) / 2 - 1.0, kA0[p1] + 3300),
             fontsize=10, color="#1f77b4", ha="center",
             arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))
ax2.annotate(f"同期不冻结：Δ = {kB0[p1]-kB0[p0]:+,.0f} ADC\n→ 显示被拖低 {abs(d_gap):,.0f} ADC",
             xy=(E + t_ep, kB0[p1]), xytext=(E + t_ep + 0.6, kB0[p1] + 1200),
             fontsize=10, color="#d95f02",
             arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.2))

for nm, a, b, col in BANDS:
    if b > a:
        ax2.text(E + (a + b) / 2, -3400, nm, fontsize=10.5, ha="center", va="top",
                 color="0.2", weight="bold")
ax2.text(E + t_ex - AWIN / 2, -3400, "A 窗", fontsize=9, ha="center", va="top", color=C_AWIN,
         weight="bold")
ax2.set_ylim(-4200, max(kB0[i0:i0 + int(10 / dtm)].max(), 100) * 1.35)

fig.text(0.012, 0.012,
         "「pending 冻结」= 在 ② 这段（2.5 s）里，算法**不更新蠕变估计**：既不算新的 g、也不动 γ，"
         "扣除量停在「检测命中那一刻」的值（本例 ≈850 ADC）。\n"
         "为什么要按住：此刻算法手里还是**上一段的旧模型**（旧 A、旧 g）。若不按住，+6405 的新台阶会被当成蠕变一路积分（橙色虚线），"
         "显示反过来把真实台阶吃掉。\n"
         "它**不额外增加时延**——② 与「确认窗」是同一段时间；决定总时延的是 ②(2.5 s) + ③(无责 3 s) + ④(τ=3 s 的建立)。",
         fontsize=10, va="bottom", ha="left",
         bbox=dict(fc="#fff8e1", ec="0.7", alpha=0.95))
fig.subplots_adjust(left=0.075, right=0.985, top=0.915, bottom=0.175)
fp = os.path.join(FIG, "H5_zoom_1868_pending_freeze.png")
fig.savefig(fp, dpi=120)
plt.close(fig)
print(f"\n图已保存：{fp}")
