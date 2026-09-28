# -*- coding: utf-8 -*-
"""尖峰的定量口径：对每个加载沿，量"显示在阶跃后冲到真值之上多少 + 之后是否回落"。

口径（逐事件）：
  t0     = v6 事件原点（kind_log 里的 t0，已回溯到真实加载沿）
  pre    = t0 前 0.3 s 的原始总量中位（空载/前级电平）
  P      = 真值 = t0+4.6~5.4 s 的原始总量中位（v6 口径的"5 s 电平"）
  step   = P − pre
  peak   = max(显示总量, τ∈[0,5 s])          # 阶跃后 5 s 内的显示最高点
  超调   = (peak − P)/step                    # 冲到真值之上多少（占阶跃）
  平台   = (显示(t0+5 s) − P)/step            # 5 s 时的残留偏差
  瞬态   = 超调 − 平台                        # 先冲高再回落的那部分（= 尖峰）
  回撤   = (peak − min(显示, τ∈[peak_t, peak_t+6 s]))/step   # 峰后 6 s 内掉回多少
  回撤时长 = 从 peak 掉到 peak−0.5·(peak−谷) 所需时间

产物：results/v61_overshoot.csv / results/_v61_overshoot.log
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                          # noqa: E402
from glm53_v6 import GLM53v6                                # noqa: E402

B = os.path.join(TEMP, "变化负载")
VARY = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]
HOLD = [(f"{loc}/数据{i}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"))
        for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]


def run(tag, path, cls=GLM53v6, **kw):
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    ys = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
    rs = L.med_smooth(Xu.sum(axis=1), 0.5 / dtm)
    return d, Y, c, rs, ys


def event_rows(tag, kind, d, c, rs, ys):
    tu, dtm = d["tu"], d["dtm"]
    n = len(tu)
    out = []
    for t0, kd in c.kind_log:
        i0 = int(np.searchsorted(tu, t0))
        i_pre = max(0, i0 - int(0.3 / dtm))
        pre = float(np.median(rs[i_pre:max(i_pre + 1, i0)]))
        a = min(n - 1, i0 + int(4.6 / dtm))
        b = min(n, i0 + int(5.4 / dtm))
        if b - a < 3:
            continue
        P = float(np.median(rs[a:b]))
        step = P - pre
        if abs(step) < 1e-9 or step < 0:
            continue
        j5 = min(n - 1, i0 + int(5.0 / dtm))
        seg = ys[i0:j5 + 1]
        if len(seg) < 5:
            continue
        k = int(np.argmax(seg))
        peak, tp = float(seg[k]), float(tu[i0 + k] - t0)
        e = min(n, i0 + k + int(6.0 / dtm))
        trough = float(np.min(ys[i0 + k:e])) if e > i0 + k else peak
        back = peak - trough
        # 回撤半程所需时间
        tt = np.nan
        for j in range(i0 + k, e):
            if ys[j] <= peak - 0.5 * back:
                tt = float(tu[j] - tu[i0 + k])
                break
        out.append(dict(dataset=tag, kind=kind, t0=t0, ev=kd, t_peak=tp,
                        step=step, P=P,
                        over_peak=100 * (peak - P) / step,
                        over_5s=100 * (float(ys[j5]) - P) / step,
                        transient=100 * back / step,
                        back_half_s=tt,
                        peak_minus_raw=float(peak - np.max(rs[i0:i0 + k + 1]))))
    return out


rows = []
for tag, path in VARY + HOLD:
    if not os.path.exists(path):
        print(f"[skip] 缺文件 {tag}")
        continue
    kind = "恒载" if tag in [t for t, _ in HOLD] else "实采"
    d, Y, c, rs, ys = run(tag, path)
    rr = event_rows(tag, kind, d, c, rs, ys)
    rows.append(pd.DataFrame(rr))
    print(f"\n{tag:>18} [{kind}] 事件 {len(rr)}")
    for r in rr:
        print(f"   t0={r['t0']:7.2f} {r['ev']:<14s} 阶跃={r['step']:9,.0f}  "
              f"超调={r['over_peak']:+6.1f}%  5s残留={r['over_5s']:+6.1f}%  "
              f"瞬态回落={r['transient']:+6.1f}%  回落半程={r['back_half_s'] if r['back_half_s']==r['back_half_s'] else float('nan'):5.2f}s  "
              f"峰高于原始={r['peak_minus_raw']:9,.0f}")

df = pd.concat(rows, ignore_index=True)
df.to_csv(os.path.join(RES, "v61_overshoot.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 118)
print("按族汇总（超调 / 5s 残留 / 瞬态回落 / 回落半程，%）")
print("=" * 118)
g = df.groupby("kind").agg(事件数=("t0", "size"),
                           超调_中位=("over_peak", "median"), 超调_p90=("over_peak", lambda s: s.quantile(0.9)),
                           超调_max=("over_peak", "max"),
                           残留5s_中位=("over_5s", "median"), 残留5s_max=("over_5s", "max"),
                           瞬态_中位=("transient", "median"), 瞬态_max=("transient", "max"),
                           回落半程_中位=("back_half_s", "median"))
print(g.round(2).to_string())
print("\n按族 × 事件类型：")
g2 = df.groupby(["kind", "ev"]).agg(事件数=("t0", "size"), 超调_中位=("over_peak", "median"),
                                    超调_max=("over_peak", "max"),
                                    瞬态_中位=("transient", "median"), 瞬态_max=("transient", "max"))
print(g2.round(2).to_string())
