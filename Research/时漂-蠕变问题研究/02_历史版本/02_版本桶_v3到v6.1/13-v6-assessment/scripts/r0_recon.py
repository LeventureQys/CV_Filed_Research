# -*- coding: utf-8 -*-
"""r0：13 份录制的阶段地图（阶跃 / 快相 / 慢相 / 卸载），产出 results/phase_map.csv。

数据集在 temp/ 下（未搬动）：右拇指指尖、左拇指指尖、四指指尖（恒载 9 组，processed_display）
与 变化负载（4 份实录，ADC 域）。
"""
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)                      # 13-v6-assessment
FLASH = os.path.dirname(os.path.dirname(OUT))    # temp/v4.1flash
TEMP = os.path.dirname(FLASH)                    # temp/
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, os.path.join(FLASH, "progress", "04-v5", "scripts"))
import ad_lib as L  # noqa: E402

B = os.path.join(TEMP, "变化负载")
HOLD = [(f"{loc}/数据{i}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"))
        for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]
VARY = [
    ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                       "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
    ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
    ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
    ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "最终测试目标", "device_001_seg000.csv")),
]
ALL = HOLD + VARY
MAIN_CH = {"右拇指指尖": 17, "左拇指指尖": 18, "四指指尖": 11}


def segments(tot, dt, frac=0.15, min_s=3.0):
    """按总量电平切出受载段（比总体峰值高 frac 以上）。"""
    thr = frac * np.percentile(tot, 99.5)
    ld = tot > thr
    d = np.diff(ld.astype(int))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    seg = [(a, min(b, len(ld) - 1)) for a, b in zip(s, e) if (b - a) * dt >= min_s]
    return sorted(seg)


def rise_times(t, y, i0, i1, lo=0.1, hi=0.9):
    """段内上升：相对段内基线/平台，给出 t10/t50/t90（秒，自 i0 起）。"""
    n = i1 - i0
    if n < 10:
        return {}
    pre = float(np.median(y[max(0, i0 - int(2.0 / max(t[1] - t[0], 1e-6))):i0 + 1])) if i0 > 3 else y[i0]
    plat = float(np.median(y[i0 + int(0.8 * n):i1]))
    base0 = float(y[i0])
    amp = plat - min(pre, base0)
    if amp <= 0:
        return {}
    out = {}
    for tag, f in (("t10", lo), ("t50", 0.5), ("t90", hi)):
        target = min(pre, base0) + f * amp
        k = np.where(y[i0:i1] >= target)[0]
        out[tag] = float(t[i0 + int(k[0])] - t[i0]) if len(k) else np.nan
    out["plat"] = plat
    out["pre"] = pre
    return out


def main():
    rows = []
    for name, path in ALL:
        if not os.path.isfile(path):
            print("!! 缺文件:", path)
            continue
        d = L.prep(path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
        tot = Xu.sum(axis=1)
        seg = segments(tot, dt)
        loc = name.split("/")[0]
        ch = MAIN_CH.get(loc)
        if ch is None or ch >= Xu.shape[1]:
            ch = int(np.argmax(Xu[int(len(tu) * 0.3):int(len(tu) * 0.7)].std(axis=0)))
        y = L.med_smooth(Xu[:, ch], 0.3 / dt)
        for k, (a, b) in enumerate(seg):
            b = min(b, len(tu) - 1)
            r = rise_times(tu, y, a, b)
            rows.append(dict(rec=name, seg=k, t0=round(float(tu[a]), 2), t1=round(float(tu[b]), 2),
                             dur=round(float(tu[b] - tu[a]), 2), ch=ch,
                             pre=round(r.get("pre", np.nan), 1), plat=round(r.get("plat", np.nan), 1),
                             amp=round(r.get("plat", np.nan) - r.get("pre", np.nan), 1),
                             t10=round(r.get("t10", np.nan), 3), t50=round(r.get("t50", np.nan), 3),
                             t90=round(r.get("t90", np.nan), 3),
                             slowgain=round(100 * (float(np.median(y[b - max(1, int(2 / dt)):b]))
                                                   - r.get("plat", np.nan)) / max(r.get("plat", np.nan) - r.get("pre", np.nan), 1e-9), 2)
                             if r else np.nan))
        print("%-18s span=%6.1f s  dt=%6.2f ms  dup=%.3f%%  受载段 %d  %s"
              % (name, d["span"], dt * 1000, 100 * d["dup"] / max(len(d["t"]), 1), len(seg),
                 [(round(float(tu[a]), 1), round(float(tu[b]), 1)) for a, b in seg][:6]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "phase_map.csv"), index=False, encoding="utf-8-sig")
    print("\n-> results/phase_map.csv  %d 行" % len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
