# -*- coding: utf-8 -*-
"""101_joint：双数据集联合选参。

判据（N 尺度，红线 0.2 N）：
- 恒压合成（102924 按压形状、无手部变化）：显示下坠 ≤0.2 N（算法自身回调上限）；
- 101043 真实输入（parity 已验）：过减≈0、下坠 ≤0.2 N、稳定 ≤2 s；上漂不限。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402

obs = import_module("91_observer")
s93 = import_module("93_palm4_sweep")
s100 = import_module("100_palm5_synth")
OUT4 = TEMP / "palm4" / "out"


def ev_flat(p) -> dict:
    t = np.arange(0, 15.5, 1.0 / 100.7)
    ytot = s100.synth(t)
    V = np.repeat(ytot[:, None] * 1000.0 / 71.0, 71, axis=1)
    r = obs.run(t, V, obs.default_with(p))
    y = s93.medfilt1s(r["out_tot"] / 1000.0, 100.7)
    m = t >= 2.2
    return {"drop": float(y[m].max() - y[m][-1]),
            "dev": float(y[m][-1] - 14.97),
            "x1": float(r["x1"][-1] / 1000.0), "x2": float(r["x2"][-1] / 1000.0)}


def ev_real(p) -> dict:
    z = np.load(OUT4 / "streams.npz")
    t, pre, tin = z["t"], z["pre"], z["tot_pre"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    r = obs.run(t, pre, obs.default_with(p))
    m = s93.evaluate(r, t, tin, fps)
    return {k: v * 0.001 for k, v in m.items() if k in ("over", "dev", "drop",
                                                        "slope_end", "tsettle")}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    BASE = {"r_fast": 0.10, "tau_c_fast_s": 15, "slow_confirm_s": 1,
            "soft_unfreeze_s": 2, "slope_cap_frac": 0.008, "r_slow_max": 0.10,
            "tau_r_fast_s": 6, "tau_r_slow_idle_s": 0.5}
    cands = [("written", BASE, "已写入 rf.10 conf1 cap.008")]
    for rf, conf, cap in itertools.product([0.01, 0.03], [1.0, 3.0, 5.0],
                                            [0.003, 0.006]):
        cands.append((f"c_{rf}_{conf:g}_{cap}",
                      {**BASE, "r_fast": rf, "slow_confirm_s": conf,
                       "slope_cap_frac": cap, "r_slow_max": 0.05,
                       "tau_c_fast_s": 40.0},
                      f"rf={rf:g} conf={conf:g} cap={cap:g} τc1=40 rsm.05"))
    rows = []
    print(f"{'参数集':<30s} | 恒压: 下坠   落点   x1末 | 真实101043: 下坠   落点   过减  稳定s")
    for name, p, label in cands:
        f, rr = ev_flat(p), ev_real(p)
        rows.append({"name": name, "label": label, "params": p, "flat": f, "real": rr})
        print(f"{label:<30s} | {f['drop']:6.3f} {f['dev']:+7.3f} {f['x1']:5.2f} "
              f"| {rr['drop']:6.3f} {rr['dev']:+7.3f} {rr['over']:5.3f} {rr['tsettle']:5.2f}")
    (OUT4 / "101_joint.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    ok = [r for r in rows if r["flat"]["drop"] <= 0.2 and r["real"]["drop"] <= 0.2
          and r["real"]["over"] <= 0.06]
    print("\n== 双红线内（两数据集下坠均 ≤0.2N、真实过减 ≤0.06N）==")
    for r in ok:
        print(f"  {r['label']:<30s} 恒压下坠={r['flat']['drop']:.3f} 真实下坠={r['real']['drop']:.3f} "
              f"真实落点={r['real']['dev']:+.3f} 稳定={r['real']['tsettle']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
