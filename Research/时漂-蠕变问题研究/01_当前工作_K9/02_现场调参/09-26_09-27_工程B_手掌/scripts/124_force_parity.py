# -*- coding: utf-8 -*-
"""124_force_parity：力值会话复算一致性复核。

三路对比：
  ① 离线复算器（链接产品源码，v34_sweep_k11.exe）--time raw / uniform
  ② temp/scripts/91_observer.py 的 Python 移植
  ③ 录制 seg 流（设备当时显示）
同时核对「E 口径」：pre 流（N）vs raw/1000 的差别。
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
fl = import_module("122_force_lib")
OUT = TEMP / "palm7" / "out"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "force_1d925c.npz")
    t, pre, seg, raw = z["t"], z["pre"], z["seg"], z["raw"]
    tin, tout_mine, tout_rec = pre.sum(1), seg.sum(1), seg.sum(1)
    rec = fl.REC8
    # ① 离线复算器
    rows_raw, ddir = fl.run_exe(OUT / "force_1d925c.bin",
                                [("rec", rec)], esum=12.224, t0=1.17,
                                tag="parity_raw", time_mode="raw")
    rows_uni, _ = fl.run_exe(OUT / "force_1d925c.bin",
                             [("rec", rec)], esum=12.224, t0=1.17,
                             tag="parity_uni", time_mode="uniform")
    _, tin_e, tout_e = fl.read_trace(ddir / "rec.bin")
    # ② Python 移植（raw 时间轴）
    r = obs.run(t, pre, obs.default_with(rec))
    # Python 移植（uniform 时间轴）
    fps = (len(t) - 1) / (t[-1] - t[0])
    tu = np.arange(len(t)) / fps
    ru = obs.run(tu, pre, obs.default_with(rec))
    print(f"帧数={len(t)}  fps={fps:.2f}  时长={t[-1]:.3f}s")
    print("\n== 复算一致性（总值 N）==")
    print(f"  离线复算器 vs Python 移植（raw 轴）："
          f"均差={np.abs(tout_e - r['out_tot']).mean():.5f} "
          f"最大={np.abs(tout_e - r['out_tot']).max():.5f} N")
    print(f"  离线复算器 vs 录制 seg（raw 轴）：   "
          f"均差={np.abs(tout_e - tout_rec).mean():.5f} "
          f"最大={np.abs(tout_e - tout_rec).max():.5f} N")
    print(f"  Python(raw)  vs 录制 seg：           "
          f"均差={np.abs(r['out_tot'] - tout_rec).mean():.5f} "
          f"最大={np.abs(r['out_tot'] - tout_rec).max():.5f} N")
    print(f"  Python(uni)  vs 录制 seg：           "
          f"均差={np.abs(ru['out_tot'] - tout_rec).mean():.5f} "
          f"最大={np.abs(ru['out_tot'] - tout_rec).max():.5f} N")
    print(f"  离线复算器 输入 vs npz 输入：        "
          f"最大={np.abs(tin_e - tin).max():.6f} N")
    # E 口径对照
    print("\n== E 口径对照（N）==")
    sg = fl.segments(t, tin)
    i0 = int(np.searchsorted(t, 1.17))
    for j, s in enumerate(sg):
        a = int(np.searchsorted(t, s["t0"]))
        w = (t >= s["t0"] + 0.1) & (t <= s["t0"] + 0.6)
        print(f"  段{j+1}: E(台阶后0.1~0.6s最小)={tin[w].min():7.3f}  "
              f"E(台阶后1.0~1.5s)={tin[(t >= s['t0']+1.0) & (t <= s['t0']+1.5)].min():7.3f}  "
              f"E(台阶后2~2.5s)={tin[(t >= s['t0']+2.0) & (t <= s['t0']+2.5)].min():7.3f}  "
              f"台阶前电平={tin[max(0, a-30):a].mean():7.3f}")
    # raw/1000 与 pre 的关系
    print("\n== 输入口径：pre(N) 与 raw/1000 对比 ==")
    print(f"  raw/1000 首帧逐通道均值={raw[0].mean()/1000:.5f} N；pre 首帧逐通道均值={pre[0].mean():.5f} N")
    d = (raw.sum(1) - pre.sum(1) * 1000.0) / 1000.0
    print(f"  raw/1000 − pre 总值：均={d.mean():+.4f} 最大={d.max():+.4f} 最小={d.min():+.4f} N")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
