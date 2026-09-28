# -*- coding: utf-8 -*-
"""定位「v5 对零点的抑制非常厉害 / 零点快速塌陷进 0」的来源。

对含零负载段的实采录制逐帧对比 raw / v3 / v5：
  1. 零负载段（含起始零负载、卸载后零负载）里显示的塌陷速度与是否被压到 0（含负值→显示层钳 0）；
  2. 卸载沿之后显示跌到 0 的耗时（v3 vs v5）；
  3. 负载段内是否出现显示被扣到 ≤0（过扣除）；
  4. 顶部零点相关状态（b、carry、g、A）的轨迹摘要。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402

B = r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
RECS = [("零负载-切换负载-零负载-再切换负载",
         os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("零负载-中途切换-零负载-切换负载",
         os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("切换负载-快相无责",
         os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc",
                      "device_001_seg000.csv"))]


def run(cls, tu, Xu, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


for tag, path in RECS:
    if not os.path.exists(path):
        print(f"[skip] {tag}: 缺文件")
        continue
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    peak = float(tot_s.max())
    Yr = Xu.copy()
    Y3, c3 = run(GLM53v3, tu, Xu)
    Y5, c5 = run(GLM53v5, tu, Xu)

    print("=" * 108)
    print(f"[{tag}] {d['span']:.1f}s / {Xu.shape[1]}ch / dtm={dtm*1000:.2f}ms / 峰值(0.5s 中值)={peak:.0f}")
    for name, Y, c in (("v3", Y3, c3), ("v5", Y5, c5)):
        s = Y.sum(axis=1)
        neg = int((Y < 0).sum())
        neg_frames = int((s < 0).sum())
        print(f"  {name}: 逐通道负值帧数={neg}（占 {100*neg/Y.size:.1f}%） "
              f"整阵为负的帧数={neg_frames}  整阵最小={s.min():.0f}  整阵末值={s[-1]:.0f}")

    # ── 零负载段识别（用原始总量）──
    zero = tot_s < 0.05 * peak
    segs, i = [], 0
    while i < len(zero):
        if zero[i]:
            j = i
            while j + 1 < len(zero) and zero[j + 1]:
                j += 1
            if (j - i) * dtm >= 2.0:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    print(f"  零负载段（原始总量 < 5% 峰值，≥2s）：{[(round(float(tu[a]),1), round(float(tu[b]),1)) for a,b in segs]}")

    for a, b in segs:
        r = tot[a:b + 1]
        y3 = Y3[a:b + 1].sum(axis=1)
        y5 = Y5[a:b + 1].sum(axis=1)
        p3 = (Y3[a:b + 1] > 0).sum() / Y3[a:b + 1].size
        p5 = (Y5[a:b + 1] > 0).sum() / Y5[a:b + 1].size
        print(f"    [{tu[a]:6.1f}~{tu[b]:6.1f}s] 原始均值={r.mean():8.0f} | "
              f"v3 均值={y3.mean():8.0f} 末值={y3[-1]:8.0f} | "
              f"v5 均值={y5.mean():8.0f} 末值={y5[-1]:8.0f} | "
              f"整阵显示>0 比例 v3={100*p3:.1f}% v5={100*p5:.1f}%")

    # ── 卸载沿之后：显示跌到 ≤5% 负载电平的耗时 ──
    load = ~zero
    trans = np.where((load[:-1]) & (zero[1:]))[0] + 1
    print(f"  卸载沿（负载→零负载）共 {len(trans)} 处：")
    for e in trans:
        lvl = float(np.median(Y3[max(0, e - int(2 / dtm)):e].sum(axis=1)))
        if lvl < 0.05 * peak:
            lvl = float(np.median(tot_s[max(0, e - int(2 / dtm)):e]))
        tgt = 0.05 * lvl
        cnt = {}
        for name, Y in (("v3", Y3), ("v5", Y5)):
            s = Y[e:].sum(axis=1)
            k = np.where(s <= tgt)[0]
            cnt[name] = float(k[0] * dtm) if len(k) else float("nan")
        print(f"    t={tu[e]:6.1f}s（卸载前电平 {lvl:.0f}）→ 显示跌到 ≤5% 用时："
              f"v3={cnt['v3']:.2f}s  v5={cnt['v5']:.2f}s")

    # ── 负载段内的过扣除（显示 ≤0）──
    for name, Y, c in (("v3", Y3, c3), ("v5", Y5, c5)):
        s = Y.sum(axis=1)
        bad = np.where((s <= 0.02 * peak) & load)[0]
        if len(bad):
            runs, st = [], bad[0]
            for k in range(1, len(bad)):
                if bad[k] - bad[k - 1] > 1:
                    runs.append((st, bad[k - 1]))
                    st = bad[k]
            runs.append((st, bad[-1]))
            runs = [(a, b) for a, b in runs if (b - a) * dtm >= 0.2]
            if runs:
                print(f"  {name} 负载段内显示被扣到 ≤2% 峰值的区段："
                      + ", ".join(f"{tu[a]:.1f}~{tu[b]:.1f}s" for a, b in runs[:8]))
    # ── 状态摘要 ──
    print(f"  v5 末态: b(max)={np.abs(c5.b).max():.0f} carry(max)={np.abs(c5.carry).max():.0f} "
          f"g={c5.g:.3f} A(max)={c5.A.max():.0f} in_load={c5.in_load} armed={c5.armed} | "
          f"v3 末态: b(max)={np.abs(c3.b).max():.0f} g={c3.g:.3f} A(max)={c3.A.max():.0f} in_load={c3.in_load}")
