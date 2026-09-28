# -*- coding: utf-8 -*-
"""a8 拍击后效与 C++/原型检测器一致性核对。"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from a_common import REC, load_uniform, add_tap, add_step, run_traced, TRACED_V6, ROOT  # noqa: E402
from a6_fix2 import V6fix, make_traced, _plain                                        # noqa: E402
from glm53_v6 import GLM53v6                                                          # noqa: E402

RES = os.path.join(ROOT, "temp", "v4.1flash", "progress", "13-v6-assessment", "results")
RES_SRC = os.path.join(ROOT, "src", "domain", "drift_v6", "drift_v6_compensator.h")


def tap_aftereffect():
    """拍击后的基线偏移与恢复时间（相对"无拍击基线"）。"""
    rows = []
    for name, t_inj in [("零负载-切换负载-零负载-再切换负载", 21.5),
                        ("中途切换-最终测试目标", 53.1)]:
        d = load_uniform(REC[name])
        tu, X = d["tu"], d["Xu"]
        i0 = int(round(t_inj / 0.01))
        base = run_traced(TRACED_V6, tu, X)
        Y0 = base["Y"].sum(axis=1)
        lvl = float(np.median(Y0[i0 - 500:i0]))
        tol = 0.02 * lvl
        for amp in [1000, 2000, 4000, 6000]:
            Xp = add_tap(X, i0, amp, rise_ms=50, hold_ms=100)[0]
            r = run_traced(TRACED_V6, tu, Xp)
            disp = r["Y"].sum(axis=1)
            dev = disp - Y0
            w = slice(i0, min(len(tu), i0 + 3000))
            over = np.abs(dev[w]) > tol
            rec_s, settle = np.nan, np.nan
            if over.any():
                last = np.where(over)[0][-1]
                rest = np.where(~over[last:])[0]
                if len(rest):
                    rec_s = float(tu[i0 + last + rest[0]] - tu[i0])
                settle = float(np.median(dev[i0 + last + 200:i0 + 1500])) if last < 1300 else np.nan
            rows.append(dict(rec=name, amp=amp, lvl=round(lvl, 1),
                             peak_over=round(float(dev[w].max()), 1),
                             peak_under=round(float(dev[w].min()), 1),
                             rec_s=(None if rec_s != rec_s else round(rec_s, 2)),
                             settle_dev=(None if settle != settle else round(settle, 1)),
                             end_dev=round(float(np.median(dev[-500:])), 1),
                             dA=round(float(np.sum(r["comp"].A) - np.sum(base["comp"].A)), 1),
                             revoke=len(r["revoke"]), ho=len(r["handoff"]) - len(base["handoff"])))
            print(f"  [{name}] 拍击{amp}: 峰值超调={rows[-1]['peak_over']:.0f} "
                  f"回到±2%用时={rows[-1]['rec_s']}s 稳态残差={rows[-1]['settle_dev']} "
                  f"全程末偏差={rows[-1]['end_dev']:.0f} ΔΣA={rows[-1]['dA']:.0f} "
                  f"撤销={rows[-1]['revoke']} 交接Δ={rows[-1]['ho']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "a8_tap_aftereffect.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string())


def cpp_proto_check():
    """核对 C++ 检测器常量与原型、以及 C++ Push 的 1 帧额外滞后（读源码，不编译）。"""
    src = open(RES_SRC, encoding="utf-8").read()
    keys = {"kDetFastS": 0.20, "kDetGapS": 0.15, "kDetLagS": 0.30, "kDetK": 5.0,
            "kDetRel": 0.05, "kDetAbsFrac": 0.01, "kDetIdleFrac": 0.10,
            "kDetPersist": 3, "kIdleSettleS": 0.50, "kRevokeS": 0.40,
            "kTailGateS": 3.0, "kTailGateFrac": 0.10, "kBackdateS": 0.60}
    import re
    rows = []
    for k, v in keys.items():
        m = re.search(rf"{k}\s*=\s*([0-9.]+)", src)
        got = float(m.group(1)) if m else np.nan
        rows.append(dict(const=k, cpp=got, proto=v, same=abs(got - v) < 1e-9))
    # 原型常量
    P = GLM53v6
    pk = {"DET_FAST": P.DET_FAST, "DET_GAP": P.DET_GAP, "DET_LAG": P.DET_LAG,
          "DET_K": P.DET_K, "DET_REL": P.DET_REL, "DET_ABS_FRAC": P.DET_ABS_FRAC,
          "DET_IDLE_FRAC": P.DET_IDLE_FRAC, "DET_PERSIST": P.DET_PERSIST}
    print("原型检测器常量: " + ", ".join(f"{k}={v}" for k, v in pk.items()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "a8_cpp_proto_consts.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string())
    # 1 帧滞后核对：C++ Push 写的是"上一帧的 3 帧中值"
    cpp = open(os.path.join(ROOT, "src", "domain", "drift_v6",
                            "drift_v6_compensator.cpp"), encoding="utf-8").read()
    push_cpp = "buf_v_[i] = prev_med_total_;" in cpp
    print(f"C++ Push 用上一帧 3 帧中值（相对原型滞后 1 帧 ≈10 ms）: {push_cpp}")


if __name__ == "__main__":
    cpp_proto_check()
    tap_aftereffect()
