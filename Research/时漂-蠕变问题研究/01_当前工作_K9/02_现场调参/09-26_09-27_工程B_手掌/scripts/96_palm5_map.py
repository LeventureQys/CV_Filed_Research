# -*- coding: utf-8 -*-
"""96_palm5_map：用力值会话（算法关）标定 ADC→N 映射，并复核候选档的 N 尺度指标。

会话 20260927_102924：display_mode=force、algorithm.enabled=false。
- raw = 原始 ADC；seg = 力值显示（N）→ 拟合逐通道/总值线性映射；
- 用该映射把 palm4（ADC 显示）各候选档的下坠/落点换算成 N；
- 红线：下坠（向下回调）≤ 0.2 N；上漂不限。
"""
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
OUT4 = TEMP / "palm4" / "out"
OUT5 = TEMP / "palm5" / "out"
OUT5.mkdir(parents=True, exist_ok=True)
SESS5 = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\手掌数据\20260927_102924_single_device_ebfaca")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t, seg = s90.load_csv(SESS5 / "device_001_seg000.csv")
    t2, raw = s90.load_csv(SESS5 / "device_001_raw_seg000.csv")
    assert len(t) == len(t2)
    t = t - t[0]
    np.savez(OUT5 / "streams.npz", t=t, seg=seg, raw=raw,
             tot_seg=seg.sum(axis=1), tot_raw=raw.sum(axis=1))
    adc, n = raw.sum(axis=1), seg.sum(axis=1)
    # 总值线性拟合 N = a·ADC + b（去掉加载过渡前后 1 s）
    k = np.polyfit(adc, n, 1)
    res = n - np.polyval(k, adc)
    print(f"总值映射拟合：N = {k[0]:.6e}·ADC + {k[1]:.3f}（残差 RMS={res.std():.4f} N "
          f"max={np.abs(res).max():.4f} N）")
    adc_per_n = 1.0 / k[0]
    print(f"换算：1 N ≈ {adc_per_n:.1f} ADC；0.2 N ≈ {0.2*adc_per_n:.0f} ADC")
    # 台阶
    kk = 30
    d = np.abs(adc[kk:] - adc[:-kk])
    hot = np.flatnonzero(d > 400)
    groups = [[int(hot[0])]]
    for i in hot[1:]:
        if i - groups[-1][-1] <= 60:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    t0 = 0.0
    for g in groups:
        i = g[int(np.argmax(d[g]))]
        t0 = float(t[i])
        print(f"台阶 @{t0:.2f}s ΔADC={adc[min(i+kk,len(adc)-1)]-adc[i]:+.0f}")
    me = (t >= t0 + 0.1) & (t <= t0 + 0.6)
    E5 = float(adc[me].min())
    print(f"E={E5:.0f} ADC ≈ {E5/1000.0:.2f} N；末值 {adc[-1]:.0f} ADC "
          f"({adc[-1]/1000.0:.2f} N)；输入蠕变 +{adc[-1]-E5:.0f} ADC "
          f"(+{(adc[-1]-E5)/max(E5-adc[0],1)*100:.1f}% 首帧电平)")
    print(f"显示(算法关)末值 {n[-1]:.2f} N，段内变化 {n[me].min():.2f}→{n.max():.2f} N（纯蠕变上漂）")

    # 候选档换算成 N
    z4 = np.load(OUT4 / "streams.npz")
    t4, pre4, tin4 = z4["t"], z4["pre"], z4["tot_pre"]
    fps4 = (len(t4) - 1) / (t4[-1] - t4[0])
    REC = s93.REC
    cands = [
        ("rec", REC, "录制参数 rf.3"),
        ("c6", {**REC, "r_fast": 0.10, "tau_c_fast_s": 15, "slope_cap_frac": 0.006,
                "r_slow_max": 0.10}, "稳显档 cap.006"),
        ("c8", {**REC, "r_fast": 0.10, "tau_c_fast_s": 15, "slope_cap_frac": 0.008,
                "r_slow_max": 0.10}, "上轮推荐 cap.008"),
        ("c12", {**REC, "r_fast": 0.10, "tau_c_fast_s": 15, "slope_cap_frac": 0.012,
                 "r_slow_max": 0.10}, "控漂档 cap.012"),
        ("c5", {**REC, "r_fast": 0.10, "tau_c_fast_s": 15, "slope_cap_frac": 0.005,
                "r_slow_max": 0.10}, "更保守 cap.005"),
    ]
    print(f"\n{'参数集':<22s} {'下坠(N)':>8s} {'落点(N)':>9s} {'过减(N)':>8s} {'末斜率(N/s)':>10s} 稳定(s)")
    for name, pp, label in cands:
        r = obs.run(t4, pre4, obs.default_with(pp))
        m = s93.evaluate(r, t4, tin4, fps4)
        print(f"{label:<22s} {m['drop']*k[0]:8.3f} {m['dev']*k[0]:+9.3f} "
              f"{m['over']*k[0]:8.3f} {m['slope_end']*k[0]:+10.4f} {m['tsettle']:6.2f}")
    (OUT5 / "map.json").write_text(json.dumps(
        {"a": k[0], "b": k[1], "adc_per_n": adc_per_n, "E_adc": E5},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
