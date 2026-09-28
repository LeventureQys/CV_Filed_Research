# -*- coding: utf-8 -*-
"""155_final_check2：b2d896「清尾漂」终选档 + 跨会话/跨模式代价。

候选（都在界面步长上）：
  M0 出厂默认（rf.01 τc1 40 conf5 soft8 cap.05 rsm.2）
  M1 只补幅度上限：rsm .15（其余默认）
  M2 幅度+速率：cap .005、rsm .15（其余默认）
  M3 M2 + 快态：rf .06、τc1 2、conf 2、soft 2
  M4 rsm .15 + cap .009 + rf .05 τc1 2（＝力值推荐B 的 cap/rsm 换成 ADC 可行的组合）
  M5 上一轮观测到的"现参数"：rf.04 τc1 1 conf1.5 soft1.5 cap.025 rsm.03
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
s93 = import_module("93_palm4_sweep")
fl = import_module("122_force_lib")
s132 = import_module("132_adc_cross")
O10 = TEMP / "palm10" / "out"

E_B, WIN_B = 16502.0, 1.30
CANDS = {
    "M0 出厂默认 rf.01 τc1 40 conf5 soft8 cap.05 rsm.2": {},
    "M1 只改幅度上限 rsm.15": {"r_slow_max": 0.15},
    "M2 幅度+速率 cap.005 rsm.15": {"slope_cap_frac": 0.005, "r_slow_max": 0.15},
    "M3 cap.005+rsm.15+rf.06 τc1 2 conf2 soft2": {
        "r_fast": 0.06, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0, "soft_unfreeze_s": 2.0,
        "slope_cap_frac": 0.005, "r_slow_max": 0.15},
    "M4 cap.009+rsm.15+rf.05 τc1 2 conf2 soft2": {
        "r_fast": 0.05, "tau_c_fast_s": 2.0, "slow_confirm_s": 2.0, "soft_unfreeze_s": 2.0,
        "slope_cap_frac": 0.009, "r_slow_max": 0.15},
    "M5 上一轮现参数 rf.04 τc1 1 conf1.5 soft1.5 cap.025 rsm.03": {
        "r_fast": 0.04, "tau_c_fast_s": 1.0, "slow_confirm_s": 1.5, "soft_unfreeze_s": 1.5,
        "slope_cap_frac": 0.025, "r_slow_max": 0.03},
}


def eval_b(t, out_tot, fps):
    y = s93.medfilt1s(out_tot, fps)
    m = t >= WIN_B
    yy, tt = y[m], t[m]
    end = float(yy[-1])
    tail = tt >= tt[-1] - 60.0
    i10 = int(np.searchsorted(tt, 10.0))
    bad = np.abs(yy - end) > 100.0
    return {"dev": end - E_B, "over": max(0.0, E_B - float(yy.min())),
            "drop": float(yy.max()) - end,
            "slope_tail": float(np.polyfit(tt[tail], yy[tail], 1)[0]),
            "d10": float(yy[-1] - yy[i10]),
            "freeze": float(tt[np.flatnonzero(bad)[-1]]) if bad.any() else 0.0}


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    zb = np.load(O10 / "b2d896.npz")
    tb, Vb = zb["t"], zb["V"]
    fb = (len(tb) - 1) / (tb[-1] - tb[0])
    z6 = np.load(TEMP / "palm6" / "out" / "streams.npz")
    t6, p6 = z6["t"], z6["pre"]
    f6 = (len(t6) - 1) / (t6[-1] - t6[0])
    sg6 = s132.segs_adc(t6, p6.sum(1))
    zf = np.load(TEMP / "palm7" / "out" / "force_1d925c.npz")
    tf, pf = zf["t"], zf["pre"]
    ff = (len(tf) - 1) / (tf[-1] - tf[0])
    sgf = fl.segments(tf, pf.sum(1))

    print("A. b2d896（ADC、429 s、单台阶）")
    print(f"{'档':<48s} {'落点':>7s} {'过扣':>6s} {'下坠':>6s} {'尾端斜率':>9s} "
          f"{'10s→末':>8s} {'冻结@':>7s}")
    out = {}
    for name, over in CANDS.items():
        p = obs.default_with(over)
        e = eval_b(tb, obs.run(tb, Vb, p)["out_tot"], fb)
        print(f"{name:<48s} {e['dev']:+7.0f} {e['over']:6.0f} {e['drop']:6.0f} "
              f"{e['slope_tail']:+9.3f} {e['d10']:+8.0f} {e['freeze']:7.1f}")
        out[name] = {"b2d896": e}

    print("\nB. ADC 会话 152928（41.8 s，E≈16481）")
    print(f"{'档':<48s} " + " | ".join(f"段{i+1} 过减/落点/下坠/稳定" for i in range(len(sg6))))
    for name, over in CANDS.items():
        p = obs.default_with(over)
        rows = s132.eval_adc(obs.run(t6, p6, p)["out_tot"], t6, sg6, f6)
        print(f"{name:<48s} " + " | ".join(
            f"{x['over']:5.0f}/{x['dev']:+7.0f}/{x['drop']:5.0f}/{x['tsettle']:4.2f}s"
            for x in rows))
        out[name]["152928"] = rows

    print("\nC. 力值会话 1d925c（32.9 s，N 尺度：cap/rsm 变绝对量）")
    print(f"{'档':<48s} 段1 落点/下坠/过扣 | 段2 落点/下坠/过扣")
    for name, over in CANDS.items():
        p = obs.default_with(over)
        rows = fl.evaluate(obs.run(tf, pf, p)["out_tot"], tf, sgf, ff)
        print(f"{name:<48s} " + " | ".join(
            f"{x['dev']:+6.2f}/{x['drop']:5.2f}/{x['over']:5.2f}" for x in rows))
        out[name]["force"] = rows
    (O10 / "155_final.json").write_text(json.dumps(out, ensure_ascii=False, default=float),
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
