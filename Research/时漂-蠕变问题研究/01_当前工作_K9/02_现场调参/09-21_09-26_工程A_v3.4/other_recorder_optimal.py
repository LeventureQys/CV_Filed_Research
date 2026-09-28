# -*- coding: utf-8 -*-
"""other_recorder 单会话最优参数搜索（第三版：目标 = 后续段再收敛快 + 保压不漂 + 不过补偿）。

为什么要第三版：前两版的评分都能被"留住常驻扣除"或"只看稳态窗"误导——
本会话现役默认的**稳态窗残漂其实已接近 0**（段1 −0.055 N/11 s、段2 +0.003 N/15.6 s），
真正的现象是**卸载后（后续段开头）要 ~15 s 才把扣除补起来**：显示在 26~35 s 之间仍涨 0.29 N。

评分（越小越好）：
  A = |显示(26 s) − 显示(35 s)|        后续段再收敛残幅（用户报的"后续段还在爬"）
  B = |显示(5 s)  − 显示(14 s)|        段1 早期过补偿/欠补偿残幅（加快收敛的代价端）
  C = 0.5×(|段1稳态漂移| + |段2稳态漂移|)   保压稳态不该继续漂
  D = 0.5×保压内过补偿下冲
  E = 0.5×|卸载释放比 − 1|             卸载后扣除被撤销的比例（<1 即补偿被提前撤销）
  cost = A + B + C + D + E

用法：python other_recorder_optimal.py [--n 120] [--refine 3] [--plot]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402
from other_recorder_tune import LIVE, read_any_session_csv, CSV_PATH  # noqa: E402

DPTOOL_ROOT = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")

W1 = (10.0, 21.0)          # 段1 稳态窗
W2 = (35.0, 50.6)          # 段2 稳态窗
T_A1, T_A2 = 26.0, 35.0    # 后续段再收敛窗（卸载之后）
T_B1, T_B2 = 5.0, 14.0     # 段1 早期窗
PRE_UNLOAD, POST_UNLOAD = (19.0, 21.4), (26.0, 29.0)

SPACE = {
    "r_fast": [0.04, 0.05, 0.06, 0.08, 0.10, 0.12],
    "tau_c_fast_s": [2.0, 3.0, 4.0, 6.0, 8.0, 12.0],
    "slow_confirm_s": [0.0, 0.5, 1.0, 2.0, 3.0],
    "soft_unfreeze_s": [1.0, 2.0, 4.0],
    "slope_cap_frac": [0.010, 0.012, 0.015, 0.018, 0.020, 0.025, 0.030],
    "tau_r_slow_idle_s": [1.0, 2.0, 4.0, 8.0],
    "tau_r_fast_s": [0.25, 0.5, 1.0],
}


def run(p: Params, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    out = np.empty(len(t))
    for i in range(len(t)):
        out[i] = c.process(float(t[i]), V[i]).sum()
    return out


def mean_at(t, y, x):
    i = int(np.searchsorted(t, x))
    lo, hi = max(i - 30, 0), min(i + 30, len(y))
    return float(np.mean(y[lo:hi]))


def win_slope_drift(t, y, a, b):
    m = (t >= a) & (t <= b)
    k = float(np.polyfit(t[m], y[m], 1)[0])
    return k * (b - a)


def evaluate(p, t, V, tot_in):
    out = run(p, t, V)
    A = abs(mean_at(t, out, T_A1) - mean_at(t, out, T_A2))
    B = abs(mean_at(t, out, T_B1) - mean_at(t, out, T_B2))
    d1 = win_slope_drift(t, out, *W1)
    d2 = win_slope_drift(t, out, *W2)
    drop_out = mean_at(t, out, PRE_UNLOAD[0] + 2.0) - mean_at(t, out, POST_UNLOAD[0] + 1.5)
    drop_in = mean_at(t, tot_in, PRE_UNLOAD[0] + 2.0) - mean_at(t, tot_in, POST_UNLOAD[0] + 1.5)
    rel = drop_out / drop_in if abs(drop_in) > 1e-9 else 1.0
    under = 0.0
    for a, b in (W1, W2):
        m = (t >= a) & (t <= b)
        ys = out[m]
        if ys.size:
            under = max(under, float(np.mean(ys[:50]) - np.min(ys)))
    cost = A + B + 0.5 * (abs(d1) + abs(d2)) + 0.5 * under + 0.5 * abs(rel - 1.0)
    return {"cost": cost, "A": A, "B": B, "d1": d1, "d2": d2, "rel": rel, "under": under,
            "ded_end": float((tot_in - out)[-1]), "out": out}


KEYS = ("cost", "A", "B", "d1", "d2", "rel", "under", "ded_end")


def line(tag, r):
    return (f"{tag:22s} cost={r['cost']:.3f} A={r['A']:+.3f} B={r['B']:+.3f} "
            f"d1={r['d1']:+.3f} d2={r['d2']:+.3f} rel={r['rel']:.3f} "
            f"under={r['under']:.3f} 末扣={r['ded_end']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--refine", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    rng = random.Random(args.seed)
    d = read_any_session_csv(CSV_PATH)
    t, V = d["t"], d["V"]
    tot_in = V.sum(axis=1)

    seen: dict[tuple, dict] = {}

    def ev(p):
        key = tuple(getattr(p, k) for k in SPACE)
        if key not in seen:
            seen[key] = evaluate(p, t, V, tot_in)
        return seen[key]

    base = ev(LIVE)
    print(line("现役默认", base), flush=True)
    # 参照臂：单参数剂量点（便于对照搜索结论）
    for tag, p in (("τc1=6", replace(LIVE, tau_c_fast_s=6.0)),
                   ("τc1=4", replace(LIVE, tau_c_fast_s=4.0)),
                   ("cap=0.015", replace(LIVE, slope_cap_frac=0.015)),
                   ("cap=0.02", replace(LIVE, slope_cap_frac=0.02)),
                   ("τc1=6+cap=0.015", replace(LIVE, tau_c_fast_s=6.0, slope_cap_frac=0.015)),
                   ("τc1=6+confirm=0.5", replace(LIVE, tau_c_fast_s=6.0, slow_confirm_s=0.5))):
        print(line("  " + tag, ev(p)), flush=True)

    best_p, best = LIVE, base
    for i in range(args.n):
        p = replace(LIVE, **{k: rng.choice(v) for k, v in SPACE.items()})
        r = ev(p)
        if r["cost"] < best["cost"]:
            best_p, best = p, r
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{args.n}] 当前最优 cost={best['cost']:.3f}（已评估 {len(seen)} 组）",
                  flush=True)
    for it in range(args.refine):
        improved = False
        for k, vals in SPACE.items():
            for v in vals:
                if v == getattr(best_p, k):
                    continue
                r = ev(replace(best_p, **{k: v}))
                if r["cost"] < best["cost"] - 1e-6:
                    best_p, best = replace(best_p, **{k: v}), r
                    improved = True
        print(f"  精修第 {it+1} 轮：cost={best['cost']:.3f} 继续={improved}", flush=True)
        if not improved:
            break

    print("\n=== 本数据最优 ===")
    print(line("  最优", best))
    for k in SPACE:
        mark = "  ← 改动" if getattr(best_p, k) != getattr(LIVE, k) else ""
        print(f"  {k:20s} = {getattr(best_p, k):<9g}（默认 {getattr(LIVE, k):g}）{mark}")

    ranked = sorted(seen.items(), key=lambda kv: kv[1]["cost"])
    print("\n  前 6 名：")
    for key, r in ranked[:6]:
        vals = dict(zip(SPACE.keys(), key))
        changed = {k: v for k, v in vals.items() if v != getattr(LIVE, k)}
        print(f"    cost={r['cost']:.3f} A={r['A']:+.3f} B={r['B']:+.3f} d1={r['d1']:+.3f} "
              f"d2={r['d2']:+.3f} rel={r['rel']:.3f} | {changed}")

    out_dir = HERE / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "other_recorder_optimal.json").write_text(json.dumps({
        "csv": str(CSV_PATH),
        "objective": "cost = A + B + 0.5*(|d1|+|d2|) + 0.5*under + 0.5*|rel-1|；"
                     "A=后续段再收敛残幅(26→35s)；B=段1早期残幅(5→14s)；d=保压稳态漂移；"
                     "rel=卸载释放比；under=保压内过补偿下冲",
        "windows": {"W1": W1, "W2": W2, "A": [T_A1, T_A2], "B": [T_B1, T_B2]},
        "live": {k: getattr(LIVE, k) for k in Params.names()},
        "best": {k: getattr(best_p, k) for k in Params.names()},
        "best_diff": {k: getattr(best_p, k) for k in SPACE
                      if getattr(best_p, k) != getattr(LIVE, k)},
        "best_score": {k: best[k] for k in KEYS},
        "live_score": {k: base[k] for k in KEYS},
        "top6": [{"params": dict(zip(SPACE.keys(), k)), **{m: r[m] for m in KEYS}}
                 for k, r in ranked[:6]],
        "single_param_arms": {tag: {m: ev(p)[m] for m in KEYS}
                              for tag, p in (("tau_c_fast=6", replace(LIVE, tau_c_fast_s=6.0)),
                                             ("tau_c_fast=4", replace(LIVE, tau_c_fast_s=4.0)),
                                             ("slope_cap=0.015", replace(LIVE, slope_cap_frac=0.015)),
                                             ("slope_cap=0.02", replace(LIVE, slope_cap_frac=0.02)))},
        "evaluated": len(seen),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(out_dir / "other_recorder_optimal_curves.npz", t=t, tot_in=tot_in,
             live=base["out"], best=best["out"])
    print(f"\n写出 {out_dir / 'other_recorder_optimal.json'}")

    if args.plot:
        sys.path.insert(0, str(DPTOOL_ROOT))
        from dptool import api  # noqa: E402
        curves = [("00_输入(算法关闭)", tot_in), ("01_现役默认", base["out"]),
                  ("02_本数据最优", best["out"])]
        if len(ranked) > 1:
            k2 = ranked[1][0]
            curves.insert(2, ("03_次优方案",
                              ev(replace(LIVE, **dict(zip(SPACE.keys(), k2))))["out"]))
        tmp = out_dir / "_dptool_best"
        if tmp.exists():
            import shutil
            shutil.rmtree(tmp)
        n_ch = V.shape[1]
        zeros = ",".join(["0.000000"] * (n_ch - 1))
        ts = t + 171642.0
        for label, y in curves:
            dd = tmp / label
            dd.mkdir(parents=True, exist_ok=True)
            lines = ["##Session", "数据阶段,processed_display", "值阶段,processed_display",
                     f"行数,{len(t)}", f"列数,{n_ch}", "##Data",
                     "timestamp,elapsed,frame_index," + ",".join(f"ch{j}" for j in range(n_ch))]
            # 总值写进单元0、其余补 0：dptool 的 signal="sum" 求和即该变体的总值
            for i in range(len(t)):
                lines.append(f"{ts[i]:.6f},{t[i]:.6f},{i},{y[i]:.6f},{zeros}")
            (dd / "device_001_seg000.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        png = out_dir / "other_recorder_最优参数对比.png"
        res = api.plot_to_file([str(tmp / c[0]) for c in curves], str(png),
                               mode="overlay", series_by="dir", series_stream="out",
                               signal="sum", dpi=130,
                               suptitle="other_recorder：本数据最优参数 vs 现役（52 通道总值 / N）")
        print("dptool 出图：", res.get("path") or res.get("error"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
