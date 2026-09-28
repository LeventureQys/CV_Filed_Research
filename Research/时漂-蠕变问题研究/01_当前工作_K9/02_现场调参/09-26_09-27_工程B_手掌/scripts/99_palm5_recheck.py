# -*- coding: utf-8 -*-
"""99_palm5_recheck：正确口径——raw=算法输入(ADC)、seg=嵌入式算法结果(N)。

① 输入（raw/1000）自身行为：是否峰后松弛（天然回调）还是持续蠕变；
② 显示（seg）下坠分解；applied = 输入 − 显示；
③ parity：Python 观测器以 raw(ADC) 为输入 + 已写入参数，结果/1000 vs seg。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
OUT = TEMP / "palm5" / "out"
WRITTEN = {"r_fast": 0.10, "tau_c_fast_s": 15, "slow_confirm_s": 1,
           "soft_unfreeze_s": 2, "slope_cap_frac": 0.008, "r_slow_max": 0.10,
           "tau_r_fast_s": 6, "tau_r_slow_idle_s": 0.5}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "streams.npz")
    t, raw, seg = z["t"], z["raw"], z["seg"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    tin = raw.sum(axis=1) / 1000.0        # 输入 N
    disp = seg.sum(axis=1)                # 显示 N（嵌入式算法结果）
    appl = tin - disp                     # 嵌入式实际扣除

    # 输入台阶
    kk = 30
    d = np.abs(tin[kk:] - tin[:-kk])
    i0 = int(np.argmax(d))
    T0 = float(t[i0])
    # 压上过程结束 ≈ 输入峰时刻
    ip = int(np.argmax(tin))
    Tp = float(t[ip])
    m = t >= Tp - 0.05
    print(f"台阶@{T0:.2f}s（Δ≈{d[i0]:.2f} N/0.3s），输入峰 {tin[ip]:.3f} N @{Tp:.2f}s")
    print(f"输入：峰→末 = {tin[ip]:.3f} → {tin[-1]:.3f} N（Δ={tin[-1]-tin[ip]:+.3f}）")
    E = float(tin[(t >= Tp) & (t <= Tp + 0.5)].min())
    print(f"E(峰后最小)={E:.3f} N")
    sm_i = s93.medfilt1s(tin, fps)
    sm_d = s93.medfilt1s(disp, fps)
    sm_a = s93.medfilt1s(appl, fps)
    m2 = t >= Tp - 0.05
    print(f"输入(1s中值): 峰={sm_i[m2].max():.3f} 末={sm_i[m2][-1]:.3f} "
          f"回调={sm_i[m2].max()-sm_i[m2][-1]:+.3f} N")
    print(f"显示(1s中值): 峰={sm_d[m2].max():.3f} 末={sm_d[m2][-1]:.3f} "
          f"下坠={sm_d[m2].max()-sm_d[m2][-1]:+.3f} N  落点(末−E)={sm_d[m2][-1]-E:+.3f} N")
    print(f"显示过减(E−段内最小)={max(0.0, E - sm_d[m2].min()):.3f} N")
    print(f"嵌入式扣除 ap   : 峰={sm_a[m2].max():.3f}@{t[m2][np.argmax(sm_a[m2])]:.2f}s "
          f"末={sm_a[m2][-1]:.3f} N")
    # parity：嵌入式算法 vs Python 复算（输入=raw ADC，同参数）
    r = obs.run(t, raw, obs.default_with(WRITTEN))
    mine = r["out_tot"] / 1000.0
    dev = mine - disp
    print(f"\nparity（Python 复算 vs 嵌入式 seg）：均差={np.abs(dev).mean():.4f} N "
          f"最大={np.abs(dev).max():.4f} N")
    print(f"复算显示：下坠={s93.medfilt1s(mine,fps)[m2].max()-s93.medfilt1s(mine,fps)[m2][-1]:.3f} N")
    print(f"\n{'t':>6s} {'输入N':>8s} {'显示N':>8s} {'扣除N':>8s} {'复算N':>8s} {'差':>7s}")
    sm_m = s93.medfilt1s(mine, fps)
    for tt in (1.0, 1.8, 2.5, 3.5, 5.0, 7.0, 9.0, 11.0, 13.0, 15.4):
        i = int(np.searchsorted(t, tt))
        if i >= len(t):
            break
        print(f"{t[i]:6.2f} {tin[i]:8.3f} {disp[i]:8.3f} {appl[i]:8.3f} {mine[i]:8.3f} {dev[i]:+7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
