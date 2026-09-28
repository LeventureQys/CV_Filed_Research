# -*- coding: utf-8 -*-
"""140_claims：把本轮的结构性结论逐条用数字验证（便于写进报告）。

① 前 3.5 s 空窗的结构来源：显示在 2 s/3 s 与 cap 无关（斜率门未开）
② soft_unfreeze_s 在力值模式是死参数（沿/缓坡前馈都不触发）
③ slow_confirm_s ≤2 s 近乎无效（受制于斜率门）
④ r_slow_max ≥0.1 在力值模式无作用（= 绝对上限 0.1 N/通道 × 41 通道 ≫ 所需）
⑤ 恒压合成（纯阶跃、零蠕变）的扣除量 ≈ 台阶尾积分伪影
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
c128 = import_module("128_force_cases")
OUT = TEMP / "palm7" / "out"


def show(tag: str, p: dict, t, V, marks=(2.0, 3.0, 3.5)) -> None:
    r = obs.run(t, V, obs.default_with({**fl.REC8, **p}))
    vals = " ".join(f"{m:g}s={r['out_tot'][int(np.searchsorted(t, m))]:.3f}"
                    for m in marks)
    print(f"  {tag:<42s} {vals}")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "force_1d925c.npz")
    t, pre = z["t"], z["pre"]
    cases = {c["name"]: c for c in c128.build()}
    tf, Vf = cases["synth_flat17"]["t"], cases["synth_flat17"]["V"]

    print("① 加载后 2/3/3.5 s 的显示值 vs slope_cap_frac（会话 1d925c 段1，t0=1.17 s）")
    print("   （空窗未结束时 cap 不应有任何影响）")
    for cap in (0.002, 0.011, 0.05, 0.3):
        show(f"cap={cap:g}", {"slope_cap_frac": cap}, t, pre, marks=(3.17, 4.17, 4.67))

    print("\n② soft_unfreeze_s 扫（其余取现役）")
    for soft in (0.1, 2.0, 8.0, 60.0):
        r = obs.run(t, pre, obs.default_with({**fl.REC8, "soft_unfreeze_s": soft}))
        rows = fl.evaluate(r["out_tot"], t, fl.segments(t, pre.sum(1)),
                           (len(t) - 1) / (t[-1] - t[0]))
        print(f"  soft={soft:<5g} 落点={rows[0]['dev']:+.4f}/{rows[1]['dev']:+.4f} "
              f"下坠={rows[0]['drop']:.4f}/{rows[1]['drop']:.4f}")

    print("\n③ slow_confirm_s 扫（其余取现役）")
    for conf in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0):
        r = obs.run(t, pre, obs.default_with({**fl.REC8, "slow_confirm_s": conf}))
        rows = fl.evaluate(r["out_tot"], t, fl.segments(t, pre.sum(1)),
                           (len(t) - 1) / (t[-1] - t[0]))
        print(f"  conf={conf:<5g} 落点={rows[0]['dev']:+.4f}/{rows[1]['dev']:+.4f} "
              f"下坠={rows[0]['drop']:.4f}/{rows[1]['drop']:.4f}")

    print("\n④ r_slow_max 扫（其余取现役）")
    for rsm in (0.02, 0.03, 0.05, 0.08, 0.10, 0.20, 0.35, 0.60):
        r = obs.run(t, pre, obs.default_with({**fl.REC8, "r_slow_max": rsm}))
        rows = fl.evaluate(r["out_tot"], t, fl.segments(t, pre.sum(1)),
                           (len(t) - 1) / (t[-1] - t[0]))
        print(f"  rsm={rsm:<5g} 落点={rows[0]['dev']:+.4f}/{rows[1]['dev']:+.4f} "
              f"下坠={rows[0]['drop']:.4f}/{rows[1]['drop']:.4f} "
              f"末斜率={rows[0]['slope_end']:+.5f}")

    print("\n⑤ 恒压合成 17 N（纯阶跃、零蠕变）的扣除量 = 纯伪影（无蠕变可扣）")
    for tag, p in (("现役", {}),
                   ("推荐B rf.05 τc1 2 cap.009", {"r_fast": 0.05, "tau_c_fast_s": 2.0,
                                                  "slope_cap_frac": 0.009}),
                   ("cap=0.002", {"slope_cap_frac": 0.002})):
        r = obs.run(tf, Vf, obs.default_with({**fl.REC8, **p}))
        m = int(np.searchsorted(tf, 10.0))
        print(f"  {tag:<28s} t=10s 显示={r['out_tot'][m]:.3f} N "
              f"（扣除 {17.0 - r['out_tot'][m]:.3f} N = {(17.0-r['out_tot'][m])/17*100:.1f}%）"
              f"  x1={r['x1'][m]:.3f} x2={r['x2'][m]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
