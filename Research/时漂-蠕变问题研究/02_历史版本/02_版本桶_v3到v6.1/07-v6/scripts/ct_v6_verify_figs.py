# -*- coding: utf-8 -*-
"""验收图：用户报告的两处，放大 + 与"修复前"对照。

四条曲线：
  原始（灰） / v5-3s（橙） / **v6 现默认**（绿实线，G1 重锚平滑 + G2 死区慢修正） /
  ≈修复前（绿虚线：重锚窗≈0 + trim 关，用来近似复现用户看到的那两处）
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
FIG = os.path.join(OUT, "figures")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402
from glm53_v6 import GLM53v6                           # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CASES = [
    ("右拇指指尖/数据2", os.path.join(TEMP, "右拇指指尖", "数据2", "device_001_seg000.csv"),
     (138.0, 162.0), "现象①：卸载前的长保压（原为 A 被单帧掉点带偏后下滑 ~4%）"),
    ("切换负载-快相无责", os.path.join(TEMP, "变化负载", "切换负载-快相无责的测试",
                                       "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"),
     (128.0, 192.0), "现象②：133.84 s 加载后的平台（原为钉住值偏高 +4.9% -> +875 过充）"),
]


def run_v5(tu, Xu):
    class _T(GLM53v51):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    c = _T(Xu.shape[1])
    c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = 3.0, 1.0, 3.0
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


def run_v6(tu, Xu, old=False):
    c = GLM53v6(Xu.shape[1])
    if old:                       # 近似"修复前"：重锚窗≈0（等于取瞬时帧）+ trim 关
        c.REANCHOR_SMOOTH_S = 0.001
        c.TRIM_RATE = 0.0
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y


fig, axs = plt.subplots(2, 2, figsize=(17.5, 10.5))
fig.suptitle("v6 修复验收：用户报告的两处（四路对照）—— 灰=原始　橙=免责3s　"
             "绿实=v6 现默认　绿虚=≈修复前", fontsize=14)

for r, (tag, path, zoom, desc) in enumerate(CASES):
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot = Xu.sum(axis=1)
    Y5 = run_v5(tu, Xu)
    Y6 = run_v6(tu, Xu)
    Y6o = run_v6(tu, Xu, old=True)
    t5, t6, t6o = Y5.sum(axis=1), Y6.sum(axis=1), Y6o.sum(axis=1)

    ax = axs[r, 0]
    ax.plot(tu, tot, color="0.62", lw=0.8, label="原始")
    ax.plot(tu, t5, color="#ff7f0e", lw=1.0, label="免责 3s")
    ax.plot(tu, t6o, color="#2ca02c", lw=1.0, ls="--", alpha=0.85, label="≈修复前")
    ax.plot(tu, t6, color="#0b7a0b", lw=1.3, label="v6 现默认")
    ax.axvspan(zoom[0], zoom[1], color="#d62728", alpha=0.10)
    ax.set_title(f"{tag}｜全程（红带 = 右图放大区）\n{desc}", fontsize=10)
    ax.set_xlabel("t (s)"); ax.set_ylabel("显示总量")
    ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="best")

    ax = axs[r, 1]
    ax.plot(tu, tot, color="0.62", lw=1.0, label="原始")
    ax.plot(tu, t5, color="#ff7f0e", lw=1.2, label="免责 3s")
    ax.plot(tu, t6o, color="#2ca02c", lw=1.2, ls="--", alpha=0.9, label="≈修复前")
    ax.plot(tu, t6, color="#0b7a0b", lw=1.8, label="v6 现默认")
    i0 = int(np.searchsorted(tu, zoom[0])); i1 = int(np.searchsorted(tu, zoom[1]))
    lo = min(tot[i0:i1].min(), t5[i0:i1].min(), t6[i0:i1].min(), t6o[i0:i1].min())
    hi = max(tot[i0:i1].max(), t5[i0:i1].max(), t6[i0:i1].max(), t6o[i0:i1].max())
    pad = 0.08 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"放大：{zoom[0]:.0f} ~ {zoom[1]:.0f} s", fontsize=10)
    ax.set_xlabel("t (s)"); ax.set_ylabel("显示总量")
    ax.grid(alpha=0.25); ax.legend(fontsize=8.5, loc="best")
    if r == 0:
        ax.annotate("修复前：这里悄悄下滑 ~4%", xy=(152, t6o[int(np.searchsorted(tu, 152))]),
                    xytext=(140.5, hi - 0.12 * (hi - lo)), fontsize=9, color="#2ca02c",
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.1))
    else:
        ax.annotate("修复前：偏高 +875", xy=(178, t6o[int(np.searchsorted(tu, 178))]),
                    xytext=(140, hi - 0.10 * (hi - lo)), fontsize=9, color="#2ca02c",
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.1))

fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=0.065, hspace=0.32, wspace=0.16)


def _inter(b1, b2):
    return max(0.0, min(b1.x1, b2.x1) - max(b1.x0, b2.x0)) * \
        max(0.0, min(b1.y1, b2.y1) - max(b1.y0, b2.y0))


f = os.path.join(FIG, "K1_v6_fix_verify.png")
fig.savefig(f, dpi=118)
# 非视觉自检
fig.canvas.draw()
rr = fig.canvas.get_renderer()
fb = fig.bbox
print(f"[figcheck] {os.path.basename(f)} canvas={fb.width:.0f}x{fb.height:.0f} axes={len(fig.axes)}")
tot_ov = 0
for i, a in enumerate(fig.axes):
    ov = []
    items = [(t.get_text().strip()[:22], t.get_window_extent(rr)) for t in a.texts if t.get_text().strip()]
    if a.title.get_text().strip():
        items.append(("<标题>", a.title.get_window_extent(rr)))
    for j in range(len(items)):
        for k in range(j + 1, len(items)):
            if _inter(items[j][1], items[k][1]) > 0.12 * min(
                    items[j][1].width * items[j][1].height,
                    items[k][1].width * items[k][1].height):
                ov.append((items[j][0], items[k][0]))
    tot_ov += len(ov)
    print(f"  ax{i} lines={len(a.lines)} texts={len(a.texts)} 文本重叠={len(ov)}"
          + (f"  {ov}" if ov else ""))
print(f"  -> 文本重叠合计 {tot_ov}")
plt.close(fig)
print(f"图已保存：{f}")
