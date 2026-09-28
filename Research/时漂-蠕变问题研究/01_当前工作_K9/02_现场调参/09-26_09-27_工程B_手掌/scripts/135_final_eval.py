# -*- coding: utf-8 -*-
"""135_final_eval：候选档全测试台评估（真实力值 3 段 + 合成 3 台 + ADC 会话 + 卸载段）。

输出可直接抄进报告的指标表。
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

obs = import_module("91_observer")
fl = import_module("122_force_lib")
c128 = import_module("128_force_cases")
OUT7 = TEMP / "palm7" / "out"

CANDS: dict[str, dict] = {
    "现役(录制值)": {},
    "K1 稳显 cap.008 rf.03 τc1 1 conf0": {"r_fast": 0.03, "tau_c_fast_s": 1.0,
                                          "slow_confirm_s": 0.0, "slope_cap_frac": 0.008},
    "K2 准读 cap.012 rf.03 τc1 1 conf0": {"r_fast": 0.03, "tau_c_fast_s": 1.0,
                                          "slow_confirm_s": 0.0, "slope_cap_frac": 0.012},
    "K3 cap.010 rf.03 τc1 1 conf0": {"r_fast": 0.03, "tau_c_fast_s": 1.0,
                                     "slow_confirm_s": 0.0, "slope_cap_frac": 0.010},
    "K4 极致稳 cap.006 rf.03 τc1 1 conf0": {"r_fast": 0.03, "tau_c_fast_s": 1.0,
                                            "slow_confirm_s": 0.0, "slope_cap_frac": 0.006},
    "K5 cap.010 rf.05 τc1 4 conf0": {"r_fast": 0.05, "tau_c_fast_s": 4.0,
                                     "slow_confirm_s": 0.0, "slope_cap_frac": 0.010},
    "K6 rsm.05 封顶 cap.012 rf.03 τc1 1": {"r_fast": 0.03, "tau_c_fast_s": 1.0,
                                          "slow_confirm_s": 0.0, "slope_cap_frac": 0.012,
                                          "r_slow_max": 0.05},
    "K7 大 rf 准读 cap.008 rf.12 τc1.5": {"r_fast": 0.12, "tau_c_fast_s": 0.5,
                                          "slow_confirm_s": 0.0, "slope_cap_frac": 0.008},
}


def full(p: dict, name: str) -> None:
    print(f"\n===== {name} =====")
    pp = {**fl.REC8, **p}
    for k in ("r_fast", "tau_c_fast_s", "slow_confirm_s", "soft_unfreeze_s",
              "slope_cap_frac", "r_slow_max", "tau_r_fast_s", "tau_r_slow_idle_s"):
        print(f"   {k}={pp[k]:g}", end="")
    print()
    for case in c128.build():
        t, V = case["t"], case["V"]
        if not case["segs"]:
            continue
        fps = (len(t) - 1) / (t[-1] - t[0])
        r = obs.run(t, V, obs.default_with(pp))
        rows = fl.evaluate(r["out_tot"], t, case["segs"], fps)
        for j, x in enumerate(rows):
            print(f"   {case['name']:<14s} 段{j+1} E={x['E']:8.3f} 落点={x['dev']:+7.3f} "
                  f"下坠={x['drop']:6.3f} 过扣={x['over']:6.3f} 漂移={x['drift']:+7.3f} "
                  f"稳定={x['tsettle']:5.2f}s 末斜率={x['slope_end']:+7.4f}")


def adc_block() -> None:
    from importlib import import_module as im
    s132 = im("132_adc_cross")
    z = np.load(TEMP / "palm6" / "out" / "streams.npz")
    t, pre = z["t"], z["pre"]
    tin = pre.sum(1)
    fps = (len(t) - 1) / (t[-1] - t[0])
    sg = s132.segs_adc(t, tin)
    print(f"\n== ADC 会话 3efae9 跨模式对照（{len(sg)} 段，E={[round(s['E']) for s in sg]}）==")
    for name, p in CANDS.items():
        r = obs.run(t, pre, obs.default_with({**fl.REC8, **p}))
        rows = s132.eval_adc(r["out_tot"], t, sg, fps)
        cell = " | ".join(f"过减{x['over']:6.0f} 落点{x['dev']:+7.0f} 下坠{x['drop']:5.0f} "
                          f"稳{x['tsettle']:4.2f}s" for x in rows)
        print(f"  {name:<38s} {cell}")


def relax_block() -> None:
    """in_102924（输入天然下沉）——算法是否帮倒忙。"""
    z = np.load(OUT7 / "in_102924.npz")
    t, V = z["t"], z["V"]
    tin = V.sum(1)
    fps = (len(t) - 1) / (t[-1] - t[0])
    sg = fl.segments(t, tin)
    print(f"\n== in_102924（松弛型输入，E={[round(s['E'],2) for s in sg]}）==")
    for name, p in CANDS.items():
        r = obs.run(t, V, obs.default_with({**fl.REC8, **p}))
        rows = fl.evaluate(r["out_tot"], t, sg, fps)
        for x in rows:
            print(f"  {name:<38s} 落点={x['dev']:+7.3f} 下坠={x['drop']:6.3f} "
                  f"过扣={x['over']:6.3f} 段末显示={x['end']:7.3f}")


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    for name, p in CANDS.items():
        full(p, name)
    (OUT7 / "135_cands.json").write_text(json.dumps(CANDS, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    relax_block()
    adc_block()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
