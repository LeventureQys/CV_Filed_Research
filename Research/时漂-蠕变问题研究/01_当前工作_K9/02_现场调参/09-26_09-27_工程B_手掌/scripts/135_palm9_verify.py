# -*- coding: utf-8 -*-
"""135_palm9_verify：右手掌 20260927_164332（算法开、最小档参数）验证。
① parity：pre 流 + 录制参数复算 vs seg 流；② 指标：台阶/E/落点/下坠/过扣/尾段带宽；
③ 台账：输入蠕变形态，验证最小档在新按压上是否达标。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

s90 = import_module("90_palm4_prep")
obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
OUT = TEMP / "palm9" / "out"
OUT.mkdir(parents=True, exist_ok=True)
SESS = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\手掌数据\右手掌\20260927_164332_single_device_4b9744")
P = {"r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
     "slope_cap_frac": 0.005, "r_slow_max": 0.15, "tau_r_fast_s": 6.0,
     "tau_r_slow_idle_s": 0.5}


def steps(tin, t):
    k = 30
    d = np.abs(tin[k:] - tin[:-k])
    hot = np.flatnonzero(d > 400)
    if not hot.size:
        return []
    groups = [[int(hot[0])]]
    for i in hot[1:]:
        if i - groups[-1][-1] <= 60:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [g[int(np.argmax(d[g]))] for g in groups]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t, pre = s90.load_csv(SESS / "device_001_pre_seg0.csv")
    _, seg = s90.load_csv(SESS / "device_001_seg000.csv")
    t = t - t[0]
    fps = (len(t) - 1) / (t[-1] - t[0])
    print(f"帧数={len(t)} 时长={t[-1]:.1f}s fps={fps:.1f}")
    np.savez(OUT / "streams.npz", t=t, pre=pre, seg=seg,
             tot_pre=pre.sum(axis=1), tot_seg=seg.sum(axis=1))
    r = obs.run(t, pre, obs.default_with(P))
    dev = r["out_tot"] - seg.sum(axis=1)
    print(f"parity：均差={np.abs(dev).mean():.2f} 最大={np.abs(dev).max():.2f} ADC")
    tin = pre.sum(axis=1)
    st = steps(tin, t)
    base = tin[: st[0]].mean() if st else tin[0]
    print(f"台阶数={len(st)}")
    segs = []
    for j, i in enumerate(st):
        m = (t >= t[i] + 0.1) & (t <= t[i] + 0.6)
        E = float(tin[m].min())
        nxt = t[st[j + 1]] if j + 1 < len(st) else t[-1]
        segs.append({"t0": float(t[i]), "E": E, "t1": float(nxt)})
        print(f"  台阶{j+1} @{t[i]:7.2f}s E≈{E:8.0f} 段长={nxt-t[i]:6.1f}s")
    (OUT / "segs.json").write_text(json.dumps(segs), encoding="utf-8")
    # 指标：录制显示（设备实际输出）
    yseg = s93.medfilt1s(seg.sum(axis=1), fps)
    print(f"\n{'段':>3s} {'E':>8s} {'落点(N)':>9s} {'10s→末(N)':>10s} {'尾段带宽(N)':>11s}"
          f"{'下坠(N)':>9s} {'过扣(N)':>9s} {'末斜率':>8s} {'卸载':>4s}")
    for j, sg in enumerate(segs):
        i0 = int(np.searchsorted(t, sg["t0"] + 0.6))
        i1 = int(np.searchsorted(t, min(sg["t1"] - 0.6, t[-1])))
        if i1 - i0 < 100:
            continue
        yy, tt = yseg[i0:i1 + 1], t[i0:i1 + 1]
        E = sg["E"]
        unloaded = tin[i1] < base + 0.25 * E
        end = float(yy[-1])
        i10 = min(i0 + int(10 * fps), i1)
        mm = tt >= tt[-1] - 5.0
        i50 = min(i0 + int(50 * fps), i1)
        tail = yy[i50:]
        print(f"{j+1:3d} {E:8.0f} {end-E:+9.3f} {end-yy[i10]:+10.3f} "
              f"{tail.max()-tail.min():11.3f} {yy.max()-end:9.3f} {max(0.0, E-yy.min()):9.3f} "
              f"{np.polyfit(tt[mm], yy[mm], 1)[0]:+8.2f} {'是' if unloaded else '否':>4s}")
    # 输入蠕变形态（以最长段为例）
    if segs:
        sg = max(segs, key=lambda s: s["t1"] - s["t0"])
        print(f"\n最长段输入蠕变（E={sg['E']:.0f}）：")
        for frac in [0.05, 0.15, 0.3, 0.5, 0.8, 1.0]:
            tt = sg["t0"] + frac * (sg["t1"] - sg["t0"])
            i = min(int(np.searchsorted(t, tt)), len(t) - 1)
            print(f"  @{t[i]-sg['t0']:6.1f}s 输入−E={tin[i]-sg['E']:+7.0f} "
                  f"({(tin[i]-sg['E'])/sg['E']*100:+5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
