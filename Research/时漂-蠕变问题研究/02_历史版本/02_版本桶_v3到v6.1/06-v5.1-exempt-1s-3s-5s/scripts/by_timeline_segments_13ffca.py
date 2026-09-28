# -*- coding: utf-8 -*-
"""13ffca（最终测试目标）· **无责 3 s 档** 的全过程时间分段图。

目的（用户需求）：把「从真实加载沿开始」的每一段时间都划分出来，并标清每段由哪个参数决定，
以便判断这些时间可以怎么分配。

四段（全部相对**真实加载沿**）：
    ① 检测窗        真实加载沿 → 判据命中（`pending=1`）
    ② 确认窗        判据命中 → 阶跃确认（epoch 起点）；**负载态下这段就是 "pending 冻结"**（`hold_=true`）
    ③ 免责期        epoch 起点 → 首扣；A 采集窗是它的最后 1/3（深色），内不扣新段、只保留 carry
    ④ 重建期        首扣 → g 一阶建立（τ=3 s：63% 在 +3 s、95% 在 +9 s）

三个真实加载沿各画一份（空载→负载 @8.42 s、负载内加重 @18.68 s / @27.66 s），
另加全长总览、时间分配甘特图与"可分配性"表。

产出：figures/H4_13ffca_time_segments.png、results/timeline_segments_13ffca.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
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


class Probe(GLM53v51):
    def __init__(self, n):
        super().__init__(n)
        self.log = []
        self.epoch_t = []

    def process(self, ts, v):
        y = super().process(ts, v)
        self.log.append((float(ts), int(self.in_load), int(self.pending), int(self.hold),
                         int(self.a_captured), int(self.fast_done), float(self.g)))
        return y

    def _begin(self, ts):
        super()._begin(ts)
        self.epoch_t.append(float(ts))

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self.epoch_t.append(float(ts))


d = L.prep(CSVP)
tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
tot_s = L.med_smooth(tot, 0.5 / dtm)
peak = float(tot.max())
c = Probe(Xu.shape[1])
c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = FAST, AWIN, ARM
Y = np.empty_like(Xu)
for i in range(len(tu)):
    Y[i] = c.process(tu[i], Xu[i])
st = np.array(c.log)
y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
ded = (Xu - Y).sum(axis=1)

ev = L.event_table(d, [e for e, _ in L.detect_events(tot, dtm)], {"raw": Xu}, [], algos=[])
edges = ev[(ev.jump > 2000)].copy()
print(f"[最终测试目标] {d['span']:.1f}s / {len(tu)} 帧 / {Xu.shape[1]}ch / 峰值 {peak:.0f} ADC")
print(f"真实加载沿 {len(edges)} 个：" + "  ".join(f"{r.t:.2f}s({r.jump:+.0f})" for _, r in edges.iterrows()))


def segments(e_t):
    """返回该加载沿的四段边界（相对加载沿，秒）与实测标注。"""
    i0 = int(np.searchsorted(tu, e_t))
    base = float(np.median(tot_s[max(0, i0 - int(2 / dtm)):i0]))
    amp = float(abs(tot_s[min(len(tu) - 1, i0 + int(8 / dtm))] - base))
    m = st[:, 1] == 1
    pen = np.where(st[:, 2] > 0)[0]              # pending 在两种状态下都会置位
    pen = pen[pen >= i0 - 2]
    fresh = float(np.median(tot_s[max(0, i0 - int(2 / dtm)):i0])) < 0.05 * peak
    out = dict(t_edge=float(e_t), jump=float(ev[ev.t == e_t].jump.iloc[0]), amp=amp,
               fresh=fresh, i0=i0)
    if len(pen):
        p0 = int(pen[0])
        p1 = p0
        while p1 + 1 < len(st) and st[p1 + 1, 2] > 0:
            p1 += 1
        out["t_detect"] = float(tu[p0] - e_t)
        out["frozen"] = bool(st[p0:p1 + 1, 3].mean() > 0.5)     # 该确认窗内是否真的冻结
        out["conf_dur"] = None   # 见下：确认窗按 [检测命中, epoch 起点] 计
    else:
        out["t_detect"], out["frozen"], out["conf_dur"] = np.nan, False, np.nan
    ep = [x for x in c.epoch_t if x >= e_t - 0.25]
    out["t_epoch"] = (ep[0] - e_t) if ep else np.nan
    out["conf_dur"] = out["t_epoch"] - out["t_detect"] if (ep and np.isfinite(out["t_detect"])) else np.nan
    if ep:
        k = int(np.searchsorted(tu, ep[0]))
        ex = np.where(st[k:, 5] == 1)[0]        # fast_done 变 True = 免责期结束
        out["t_exempt_end"] = float(tu[k + ex[0]] - e_t) if len(ex) else np.nan
    else:
        out["t_exempt_end"] = np.nan
    # 首扣（按新幅度扣）——对负载内变载，扣除量本来就非零，故用 epoch+FAST 作为口径
    if np.isfinite(out["t_epoch"]):
        out["t_first_ded"] = out["t_epoch"] + FAST
    else:
        out["t_first_ded"] = np.nan
    # 首个"可见"扣除（仅空载→负载可测）
    if fresh:
        seg = ded[i0:i0 + int(12 / dtm)]
        hit = np.where(seg > 0.005 * amp)[0]
        out["t_visible"] = float(hit[0] * dtm) if len(hit) else np.nan
    else:
        out["t_visible"] = np.nan
    out["t_63"] = out["t_first_ded"] + TAU_G
    out["t_95"] = out["t_first_ded"] + 3 * TAU_G
    return out


SEGS = [segments(float(r.t)) for _, r in edges.iterrows()]
for s in SEGS:
    print(f"  沿 @{s['t_edge']:6.2f}s 台阶{s['jump']:+7.0f} "
          f"{'空载→负载' if s['fresh'] else '负载内变载'} | "
          f"检测命中 +{s['t_detect']:.2f}s | epoch +{s['t_epoch']:.2f}s | "
          f"免责结束 +{s['t_exempt_end']:.2f}s | 首扣(理论) +{s['t_first_ded']:.2f}s | "
          f"可见扣除 +{s['t_visible']:.2f}s | 63%/95% +{s['t_63']:.1f}/+{s['t_95']:.1f}s")
pd.DataFrame(SEGS).to_csv(os.path.join(RES, "timeline_segments_13ffca.csv"),
                          index=False, encoding="utf-8-sig")

# ═══════════════════════════ 出图 ═══════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(20, 10.5))
fig.suptitle("13ffca（最终测试目标）· 无责 3 s 档 —— 从「真实加载沿」起的全过程时间分段"
             f"（{len(tu)} 帧 / {d['span']:.1f}s / {Xu.shape[1]}ch / 峰值 {peak:.0f} ADC）", fontsize=14)


def draw_bands(ax, s, ymin=0.02, h=0.085):
    """在轴底部画一条时间分色带 + 边界标注。"""
    segs = [("①", 0.0, s["t_detect"], C_DET),
            ("②", s["t_detect"], s["t_epoch"], C_CONF),
            ("③", s["t_epoch"], s["t_exempt_end"], C_EX),
            ("④", s["t_exempt_end"], s["t_95"], C_REB)]
    for nm, a, b, col in segs:
        if np.isfinite(a) and np.isfinite(b) and b > a:
            al = 0.75 if (nm != "②" or s.get("frozen")) else 0.30
            ax.axvspan(s["t_edge"] + a, s["t_edge"] + b, ymin=ymin, ymax=ymin + h,
                       color=col, alpha=al, lw=0,
                       hatch=None if (nm != "②" or s.get("frozen")) else "///")
            ax.text(s["t_edge"] + (a + b) / 2, ymin + h / 2, nm, fontsize=10, ha="center",
                    va="center", weight="bold")
            if nm == "②" and not s.get("frozen"):
                ax.text(s["t_edge"] + (a + b) / 2, ymin - 0.012, "空载态：无冻结",
                        fontsize=7.5, ha="center", va="top", color="0.3")
    # A 采集窗（免责期最后 1/3）
    if np.isfinite(s["t_exempt_end"]):
        ax.axvspan(s["t_edge"] + s["t_exempt_end"] - AWIN, s["t_edge"] + s["t_exempt_end"],
                   ymin=ymin, ymax=ymin + h, color=C_AWIN, alpha=0.95, lw=0)
    for t, lab, col in ((s["t_detect"], "检测命中", C_DET), (s["t_epoch"], "确认=epoch起点", C_CONF),
                        (s["t_exempt_end"], "免责结束=首扣", C_EX)):
        if np.isfinite(t):
            ax.axvline(s["t_edge"] + t, color=col, ls="--", lw=1.2)
    if np.isfinite(s["t_first_ded"]):
        for t, lab in ((s["t_first_ded"] + TAU_G, "63%"), (s["t_first_ded"] + 3 * TAU_G, "95%")):
            ax.axvline(s["t_edge"] + t, color=C_REB, ls=":", lw=1.2)
            ax.text(s["t_edge"] + t, ymin + h + 0.012, lab, fontsize=8, color="#2f7d32",
                    ha="center", va="bottom")
    if np.isfinite(s["t_visible"]):
        ax.axvline(s["t_edge"] + s["t_visible"], color="k", ls="-.", lw=1.0)
        ax.text(s["t_edge"] + s["t_visible"], 0.995, f"可见扣除 +{s['t_visible']:.1f}s",
                fontsize=8.5, rotation=90, va="top", ha="right", transform=ax.get_xaxis_transform())


# (1) 全长总览
ax = axes[0, 0]
ax.plot(tu, tot_s, color="0.55", lw=1.0, label="原始")
ax.plot(tu, y, color="#1f77b4", lw=1.3, label="无责 3 s")
for s in SEGS:
    draw_bands(ax, s, ymin=0.02, h=0.05)
ax.set_title("(1) 全长（下方彩带 = 每个真实加载沿的四段划分）")
ax.set_xlabel("时间 (s)"); ax.set_ylabel("总量 (ADC)")
ax.legend(fontsize=8, loc="upper left")

# (2)(4)(5) 三个加载沿放大
for ax, s, tag in ((axes[0, 1], SEGS[0], "①空载→负载"),
                   (axes[1, 0], SEGS[1], "②负载内加重"),
                   (axes[1, 1], SEGS[2], "③负载内加重")):
    i0 = s["i0"]
    xw = 18.0
    w = slice(max(0, i0 - int(2 / dtm)), min(len(tu), i0 + int(xw / dtm)))
    xx = tu[w] - s["t_edge"]
    ax.plot(xx, tot_s[w], color="0.55", lw=1.1, label="原始")
    ax.plot(xx, y[w], color="#1f77b4", lw=1.5, label="无责 3 s")
    draw_bands(ax, s)
    ax.axvline(0, color="k", lw=1.2)
    ax.set_xlim(-2, xw)
    ax.set_title(f"({['①', '②', '③'][SEGS.index(s)]}) 真实加载沿 {tag} @{s['t_edge']:.2f}s"
                 f"（台阶 {s['jump']:+.0f}，占电平 {abs(s['jump']) / max(s['amp'], 1e-9):.2f}）")
    ax.set_xlabel("距真实加载沿的时间 (s)"); ax.set_ylabel("总量 (ADC)")
    ax.legend(fontsize=8, loc="lower right")
    txt = (f"① 检测 {s['t_detect']:.2f}s → ② 确认（pending 冻结）{s['t_epoch'] - s['t_detect']:.2f}s → "
           f"③ 免责 {s['t_exempt_end'] - s['t_epoch']:.2f}s → ④ 重建 {s['t_95'] - s['t_exempt_end']:.1f}s")
    ax.text(0.5, 0.965, txt, transform=ax.transAxes, fontsize=8.5, ha="center", va="top",
            bbox=dict(fc="white", ec="0.8", alpha=0.9))

# (2) 时间分配甘特图（两种工况各一行）
ax = axes[0, 2]
s0 = SEGS[0]
SEGDEF = [("① 检测", C_DET), ("② 确认", C_CONF), ("③ 免责", C_EX), ("④ 重建", C_REB)]
for ri, (nm, s) in enumerate((("空载→负载\n@8.42 s", SEGS[0]), ("负载内变载\n@18.68 s", SEGS[1]))):
    yv = 1.15 - ri * 1.25
    bnd = [0.0, s["t_detect"], s["t_epoch"], s["t_exempt_end"], s["t_95"]]
    for i in range(4):
        a, b = bnd[i], bnd[i + 1]
        ax.add_patch(Rectangle((a, yv - 0.26), b - a, 0.52, fc=SEGDEF[i][1], alpha=0.8, ec="0.4"))
        ax.text((a + b) / 2, yv, f"{b - a:.2f}s", ha="center", va="center",
                fontsize=9.5, weight="bold")
        ax.text((a + b) / 2, yv - 0.33, SEGDEF[i][0], ha="center", va="top",
                fontsize=8.5, color="0.25")
    ax.text(-0.5, yv, nm, fontsize=9, ha="right", va="center")
    if not s.get("frozen"):
        ax.text(s["t_epoch"] + 0.15, yv + 0.36, "（本行确认窗处于空载态：无冻结）",
                fontsize=8, color="0.35")
for xv, lab, col in ((s0["t_exempt_end"], "首扣 ≈5.5 s", "k"),
                     (s0["t_95"], "扣到位 95% ≈14.5 s", "#2f7d32")):
    ax.axvline(xv, color=col, lw=0.9, ls=":")
    ax.text(xv, 2.02, lab, fontsize=8.5, color=col, ha="center")
ax.text(0.02, 0.03, "参数归属：① 判据窗（0.3s 近窗+0.5s 滞后窗）　② kStepPersistS=2.5s（可调）　"
                    "③ FAST_S=3s（可切 5s，A 窗=后 1/3 强制绑定）　④ τ_g=3s",
        transform=ax.transAxes, fontsize=8.2, color="0.25")
ax.set_xlim(-5.6, 19); ax.set_ylim(-0.9, 2.3); ax.set_yticks([])
ax.set_xlabel("距真实加载沿的时间 (s)")
ax.set_title("(2) 时间分配图（3 s 档）：两种工况逐段时长")
ax.grid(axis="x", alpha=0.3)

# (6) 可分配性表
ax = axes[1, 2]
ax.axis("off")
cell = [["段", "时长", "由什么决定", "能不能「分配」（调整后果）"],
        ["① 检测窗", f"{s0['t_detect']:.2f} s",
         "判据窗口 + 台阶占电平比例",
         "可压缩：缩窗 → 更快，但小台阶漏检、误触发增多"],
        ["② 确认窗", f"{s0['t_epoch'] - s0['t_detect']:.2f} s",
         "kStepPersistS = 2.5 s",
         "可压缩：响应更快；但手指调整/磕碰会被当真加载。\n负载态下这段就是「pending 冻结」（扣除量停住）；\n空载态下不存在冻结（本数据两者相差 0.01~0.02 s）"],
        ["③ 免责期", f"{s0['t_exempt_end'] - s0['t_epoch']:.2f} s",
         "FAST_S（3 s，可切 5 s）\nA 窗 = FAST/3 强制绑定",
         "可切换：3 s 长保压更好、5 s 变载跟踪更好；\n缩短 → 更多快相被当蠕变扣掉（欠报）"],
        ["④ 重建期", f"{s0['t_95'] - s0['t_exempt_end']:.1f} s",
         "τ_g = 3 s 的一阶建立",
         "可压缩：τ 越小越快；但噪声会直接进显示"],
        ["合计到首扣", f"{s0['t_first_ded']:.1f} s", "= ② + ③ + 检测",
         "物理下限：检测≈0.1~0.5 s，其余由两个参数决定"],
        ["合计到扣到位", f"{s0['t_95']:.1f} s", "= 首扣 + 3τ_g", "再压就要动 τ_g"]]
tb = ax.table(cellText=cell, cellLoc="left", bbox=[0.0, 0.02, 1.0, 0.93])
tb.auto_set_font_size(False); tb.set_fontsize(8.6)
for j in range(4):
    tb[0, j].set_facecolor("#e8e8e8"); tb[0, j].set_text_props(weight="bold")
for i2, col in enumerate([C_DET, C_CONF, C_EX, C_REB, "0.9", "0.85"], start=1):
    tb[i2, 0].set_facecolor(col)
ax.set_title("(3) 各段能不能分配：谁决定它、动了会怎样", pad=14)

fig.subplots_adjust(left=0.045, right=0.985, top=0.90, bottom=0.07, hspace=0.44, wspace=0.20)
fp = os.path.join(FIG, "H4_13ffca_time_segments.png")
fig.savefig(fp, dpi=115)
plt.close(fig)
print(f"\n图已保存：{fp}")
