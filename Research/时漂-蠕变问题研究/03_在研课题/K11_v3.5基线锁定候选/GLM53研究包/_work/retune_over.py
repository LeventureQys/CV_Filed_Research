# -*- coding: utf-8 -*-
"""复调：拇指/手掌四类在「过减深度 ≤ 150 ADC（绝对值）」硬约束下重搜参数。

用户口径
--------
* x1（快态）阶段出现大幅过减不可接受；可接受 50~150 ADC 的过减（绝对值）。
* 目标函数（越小越好，逐会话平均）：
    score = |末值残漂%| + |末30s%| + 0.5·max(0, 最深过减ADC − 150)
  其中「最深过减ADC」= max_t( ref(t) − out(t) )，ref(t) 为逐通道双指数拟合的
  「真实弹性+蠕变」轨迹（out 压到真实曲线下方才算过减，ΣE 恒定参考线测不出瞬态下冲）。
* 其余百分比仍以「本该扣掉的蠕变」为分母。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))

from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import EXE, PREP, SESSION_BIN, Set, write_setfile  # noqa: E402

TARGETS = ["右拇指指腹", "左拇指指腹", "右手掌", "左手掌"]
OVER_TOL_ADC = 150.0
OVER_W = 3.0
WORK = HERE / "_retune"
DUMP = WORK / "_dump"

GRID: dict[str, list[float]] = {
    "r_fast": [0.0, 0.01, 0.02, 0.04, 0.06, 0.10],
    "tau_c_fast_s": [2.0, 4.0, 8.0, 12.0, 20.0, 40.0],
    "tau_c_fast_boost_s": [0.5, 1.0, 2.0, 5.0, 12.0],
    "edge_boost_s": [0.0, 1.0, 2.0, 5.0],
    "soft_unfreeze_s": [0.5, 1.0, 2.0, 4.0, 8.0],
    "slow_confirm_s": [0.0, 0.5, 2.0, 5.0, 10.0],
    "slope_cap_frac": [0.005, 0.01, 0.02, 0.04, 0.08, 0.15],
    "slope_gate_frac": [0.2, 0.5, 1.0, 2.0, 5.0, 20.0],
    "tau_slope_s": [0.5, 1.0, 2.0, 3.0],
    "tau_r_fast_s": [0.2, 0.5, 1.0, 2.0],
    "r_slow_max": [0.35, 0.6, 1.0, 2.0, 5.0, 10.0],
    "hold_eps": [0.0, 0.5, 2.0, 8.0],
    "edge_slope_thres": [10.0, 20.0, 60.0, 150.0, 400.0],
    "tau_r_slow_idle_s": [0.5, 2.0, 8.0, 30.0],
    "idle_frac": [0.02, 0.05, 0.10],
    "ramp_slope_min": [0.0, 0.5, 2.0],
}
ORDER = list(GRID)

_refs: dict[str, np.ndarray] = {}


def ref_of(key: str) -> np.ndarray:
    """总量级双指数拟合出的「真实」总轨迹：E + [c1(1−e^−rel/τ1)+c2(1−e^−rel/τ2)]·ramp。
    （逐通道拟合的分量和与总量不自洽，不能用；此处与 02_prep.json 的 fit_total 同源。）"""
    if key in _refs:
        return _refs[key]
    p = PREP[key]
    z = np.load(OUT_DIR / "_prep" / (key.replace("/", "_") + ".npz"))
    t, t0 = z["t"], float(p["t0"])
    rel = np.maximum(t - t0 - 0.6, 0.0)
    ramp = np.minimum(np.maximum(t - t0, 0.0) / 0.6, 1.0)
    r = (p["E_total_fit"]
         + (p["c1"] * (1.0 - np.exp(-rel / p["tau1"]))
            + p["c2"] * (1.0 - np.exp(-rel / p["tau2"]))) * ramp)
    _refs[key] = r
    return r


def read_bin(f: Path) -> np.ndarray:
    raw = np.fromfile(f, dtype=np.int32, count=2)
    return np.fromfile(f, dtype=np.float64, offset=8).reshape(int(raw[0]), int(raw[1]))


def eval_batch(sensor: str, cands: list[dict]) -> list[dict]:
    """跑 C++ 复算器（--dump）→ 逐会话指标 + score。

    过减口径（调零语义）：这些会话是「装好负载 → 调零 → 开算法」，零点取在受载后首帧，
    理想显示应钉住 0；**过减深度 = max(0, −min_t out)**（显示跌破零基准多少 ADC），
    与用户「过减 50~150 ADC 可接受」的绝对值口径一致。
    """
    WORK.mkdir(parents=True, exist_ok=True)
    if DUMP.exists():
        shutil.rmtree(DUMP)
    DUMP.mkdir(parents=True)
    sets = [Set(f"c{j}", c) for j, c in enumerate(cands)]
    sf = write_setfile(sets, WORK / "_sets.txt")
    per: list[dict] = [dict(resid_pct=0.0, tail_pct=0.0, over_adc=0.0, ded_pct=0.0, score=0.0)
                       for _ in cands]
    for tag in SENSORS[sensor]["sessions"]:
        key = f"{sensor}/{tag}"
        p = PREP[key]
        src = SESSION_BIN.get(key, p["path"])
        cmd = [str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}", f"{p['t0']:.6f}",
               "--time", "uniform", "--zero", "--dump", str(DUMP)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")  # 重试一次
        if r.returncode != 0:
            raise RuntimeError(f"{key}: rc={r.returncode} {r.stderr[:500]}")
        ref = ref_of(key)
        creep = max(p["creep_total"], 1.0)
        for j in range(len(cands)):
            a = read_bin(DUMP / f"c{j}.bin")
            t, tin, out = a[:, 0], a[:, 1], a[:, 2]
            i30 = max(int(np.searchsorted(t, t[-1] - 30.0)), 0)
            tail = float(np.polyfit(t[i30:], out[i30:], 1)[0]) * 30.0 if t.size - i30 > 5 else 0.0
            over = float(max(0.0, -np.min(out)))       # 显示跌破零基准的深度（过减）
            resid = float(out[-1] - p["Esum_channels"]) / creep * 100.0
            ded = float((tin[-1] - out[-1]) / creep * 100.0)
            s = abs(resid) + abs(tail / creep * 100.0) + OVER_W * max(0.0, over - OVER_TOL_ADC)
            m = per[j]
            for kk, vv in (("resid_pct", resid), ("tail_pct", tail / creep * 100.0),
                           ("over_adc", over), ("ded_pct", ded), ("score", s)):
                m[kk] += vv / len(SENSORS[sensor]["sessions"])
    return per


CACHE: dict[str, tuple] = {}
NB = 0


def ck(sensor: str, c: dict) -> str:
    return f"{sensor}|{json.dumps(c, sort_keys=True)}"


def evaluate(sensor: str, cands: list[dict]) -> list[tuple]:
    global NB
    keys = [ck(sensor, c) for c in cands]
    if any(k not in CACHE for k in keys):
        for i0 in range(0, len(cands), 150):
            chunk = [c for c, k in zip(cands[i0:i0 + 150], keys[i0:i0 + 150]) if k not in CACHE]
            if not chunk:
                continue
            NB += 1
            for c, m in zip(chunk, eval_batch(sensor, chunk)):
                CACHE[ck(sensor, c)] = m
    return [CACHE[k] for k in keys]


def coord_descent(sensor: str, start: dict, passes: int = 4):
    cur = dict(start)
    best = evaluate(sensor, [cur])[0]["score"]
    trace = [{"params": dict(cur), "score": best}]
    for p in range(passes):
        improved = False
        for kk in ORDER:
            cands = []
            for v in GRID[kk]:
                if kk in cur and abs(v - cur[kk]) < 1e-12:
                    continue
                c = dict(cur)
                c[kk] = v
                cands.append(c)
            if not cands:
                continue
            per = evaluate(sensor, cands)
            bi = int(np.argmin([m["score"] for m in per]))
            if per[bi]["score"] < best - 1e-9:
                cur, best = cands[bi], per[bi]["score"]
                improved = True
                trace.append({"params": dict(cur), "score": best, "changed": kk})
        print(f"    pass{p + 1}: score={best:.2f}", flush=True)
        if not improved:
            break
    return cur, best, trace


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    only = sys.argv[1:] or TARGETS
    prev = json.loads((OUT_DIR / "04_search.json").read_text(encoding="utf-8"))
    prev_ext = json.loads((OUT_DIR / "04_search_ext.json").read_text(encoding="utf-8"))
    result = {}
    for sensor in only:
        print(f"\n########## {sensor} ##########", flush=True)
        starts = [prev[sensor]["runs"][0]["params"]]
        if sensor in prev_ext:
            starts.append(prev_ext[sensor]["runs"][0]["params"])
        base_m = evaluate(sensor, [{}])[0]
        print(f"  现役默认：score={base_m['score']:.1f} 过减={base_m['over_adc']:.0f} ADC "
              f"扣除={base_m['ded_pct']:.0f}%", flush=True)
        runs = []
        for i, st in enumerate(starts):
            m0 = evaluate(sensor, [st])[0]
            print(f"  起点{i}: score={m0['score']:.1f} 过减={m0['over_adc']:.0f} ADC", flush=True)
            p, val, tr = coord_descent(sensor, st)
            runs.append({"start_score": m0["score"], "score": val, "params": p, "trace": tr})
        runs.sort(key=lambda r: r["score"])
        best = runs[0]
        mb = evaluate(sensor, [best["params"]])[0]
        print(f"  ★ 最优 score={best['score']:.2f} 过减={mb['over_adc']:.0f} ADC "
              f"扣除={mb['ded_pct']:.0f}% 残漂={mb['resid_pct']:.0f}%", flush=True)
        print(f"    {json.dumps(best['params'], ensure_ascii=False)}", flush=True)
        result[sensor] = {"live": base_m, "best": {**mb, "params": best["params"]},
                          "runs": runs}
        (HERE / f"retune_{sensor}.json").write_text(
            json.dumps(result[sensor], ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "retune_all.json").write_text(
        json.dumps({k: {"live": v["live"], "best": v["best"]} for k, v in result.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nbatches={NB}  写出 {HERE / 'retune_all.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
