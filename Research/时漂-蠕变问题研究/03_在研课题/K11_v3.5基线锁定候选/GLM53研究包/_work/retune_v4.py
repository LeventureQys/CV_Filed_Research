# -*- coding: utf-8 -*-
"""v4 复调（K10 结构 track_base_frac 上机后的参数搜索）：
    目标 = 2s 内建基线、之后 1s 均值带宽 |m1| ≤150、过减 ≤150、末值上漂 ≤150。

score = 2·max(0,过减−150) + 2·max(0,2s后带宽−150) + 1·max(0,末上漂−150)
        + 0.15·settle_s + 0.1·|末残漂%|
指标由 --dump 的逐帧显示值计算（m1 = 1s 滑动均值；settle = 最后一次 |m1|>150 的时刻）。
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
WORK = HERE / "_retune4"
DUMP = WORK / "_dump"

GRID: dict[str, list[float]] = {
    "track_base_frac": [0.3, 0.6, 1.0],
    "slope_cap_frac": [0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8],
    "slope_gate_frac": [5.0, 20.0],
    "r_fast": [0.0, 0.01, 0.02, 0.04],
    "tau_c_fast_s": [4.0, 12.0, 20.0],
    "tau_c_fast_boost_s": [0.5, 1.0, 2.0],
    "edge_boost_s": [0.0, 2.0],
    "soft_unfreeze_s": [0.25, 0.5, 1.0],
    "slow_confirm_s": [0.0, 0.5],
    "tau_slope_s": [0.3, 0.5, 1.0, 2.0],
    "tau_r_fast_s": [0.5, 2.0],
    "r_slow_max": [5.0, 30.0],
    "hold_eps": [0.0, 2.0],
    "edge_slope_thres": [10.0, 60.0],
    "tau_r_slow_idle_s": [2.0, 30.0],
    "tau_r_slow_s": [150.0, 600.0],
    "idle_frac": [0.05, 0.10],
    "ramp_slope_min": [0.0, 0.5],
}
ORDER = list(GRID)


def read_bin(f: Path) -> np.ndarray:
    raw = np.fromfile(f, dtype=np.int32, count=2)
    return np.fromfile(f, dtype=np.float64, offset=8).reshape(int(raw[0]), int(raw[1]))


def m1(x: np.ndarray, t: np.ndarray, win: float = 1.0) -> np.ndarray:
    n = t.size
    y = np.empty(n)
    j, s, cnt = 0, 0.0, 0
    for i in range(n):
        s += x[i]
        cnt += 1
        while t[i] - t[j] > win:
            s -= x[j]
            cnt -= 1
            j += 1
        y[i] = s / cnt
    return y


def metrics(a: np.ndarray, esum: float, creep: float) -> dict:
    t, tin, out = a[:, 0], a[:, 1], a[:, 2]
    sm = m1(out, t)
    m2s = t >= 2.0
    band = float(np.max(np.abs(sm[m2s]))) if m2s.any() else 0.0
    bad = np.where(np.abs(sm) > 150.0)[0]
    settle = float(t[bad[-1]]) if bad.size else 0.0
    # 过减只计「持续型」：1s 均值低于 −150（瞬态下陷=真实输入波动的忠实透传，不算过减）
    over = float(max(0.0, -sm.min()))
    up_end = float(out[-1] - esum)
    ded = float((tin[-1] - out[-1]) / creep * 100.0)
    s = (2.0 * max(0.0, over - 150.0) + 2.0 * max(0.0, band - 150.0)
         + 1.0 * max(0.0, up_end - 150.0) + 0.15 * settle
         + 0.1 * abs(up_end / creep * 100.0))
    return {"over_adc": over, "band_adc": band, "settle_s": settle, "up_adc": up_end,
            "ded_pct": ded, "score": s}


def eval_batch(sensor: str, cands: list[dict]) -> list[dict]:
    WORK.mkdir(parents=True, exist_ok=True)
    if DUMP.exists():
        shutil.rmtree(DUMP)
    DUMP.mkdir(parents=True)
    sets = [Set(f"c{j}", c) for j, c in enumerate(cands)]
    sf = write_setfile(sets, WORK / "_sets.txt")
    n_sess = len(SENSORS[sensor]["sessions"])
    per = [dict(over_adc=0.0, band_adc=0.0, settle_s=0.0, up_adc=0.0, ded_pct=0.0, score=0.0)
           for _ in cands]
    for tag in SENSORS[sensor]["sessions"]:
        key = f"{sensor}/{tag}"
        p = PREP[key]
        src = SESSION_BIN.get(key, p["path"])
        cmd = [str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}", f"{p['t0']:.6f}",
               "--time", "raw", "--zero", "--dump", str(DUMP)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError(f"{key}: rc={r.returncode} {r.stderr[:500]}")
        creep = max(p["creep_total"], 1.0)
        for j in range(len(cands)):
            m = metrics(read_bin(DUMP / f"c{j}.bin"), p["Esum_channels"], creep)
            for kk in per[j]:
                per[j][kk] += m[kk] / n_sess
    return per


CACHE: dict[str, dict] = {}
NB = 0


def ck(sensor: str, c: dict) -> str:
    return f"{sensor}|{json.dumps(c, sort_keys=True)}"


def evaluate(sensor: str, cands: list[dict]) -> list[dict]:
    global NB
    keys = [ck(sensor, c) for c in cands]
    if any(k not in CACHE for k in keys):
        for i0 in range(0, len(cands), 100):
            chunk = [c for c, k in zip(cands[i0:i0 + 100], keys[i0:i0 + 100]) if k not in CACHE]
            if not chunk:
                continue
            NB += 1
            for c, m in zip(chunk, eval_batch(sensor, chunk)):
                CACHE[ck(sensor, c)] = m
    return [CACHE[k] for k in keys]


def coord_descent(sensor: str, start: dict, passes: int = 5):
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
    v3 = json.loads((HERE / "retune3_all.json").read_text(encoding="utf-8"))
    seed0 = {"track_base_frac": 1.0, "slope_cap_frac": 0.3, "slope_gate_frac": 5.0,
             "r_fast": 0.02, "tau_c_fast_s": 12.0, "tau_c_fast_boost_s": 1.0,
             "soft_unfreeze_s": 0.25, "slow_confirm_s": 0.0, "tau_slope_s": 1.0,
             "tau_r_fast_s": 1.0, "r_slow_max": 30.0, "hold_eps": 2.0,
             "edge_slope_thres": 10.0, "tau_r_slow_idle_s": 30.0, "tau_r_slow_s": 600.0,
             "idle_frac": 0.1, "ramp_slope_min": 0.0}
    result = {}
    for sensor in only:
        print(f"\n########## {sensor} ##########", flush=True)
        base_m = evaluate(sensor, [{}])[0]
        print(f"  现役默认：score={base_m['score']:.1f} 过减={base_m['over_adc']:.0f} "
              f"带宽={base_m['band_adc']:.0f} 入带={base_m['settle_s']:.0f}s "
              f"上漂={base_m['up_adc']:.0f}", flush=True)
        starts = [seed0, {**v3[sensor]["best"]["params"], "track_base_frac": 1.0}]
        runs = []
        for i, st in enumerate(starts):
            m0 = evaluate(sensor, [st])[0]
            print(f"  起点{i}: score={m0['score']:.1f} 过减={m0['over_adc']:.0f} "
                  f"带宽={m0['band_adc']:.0f} 入带={m0['settle_s']:.1f}s", flush=True)
            p, val, tr = coord_descent(sensor, st)
            runs.append({"start_score": m0["score"], "score": val, "params": p, "trace": tr})
        runs.sort(key=lambda r: r["score"])
        best = runs[0]
        mb = evaluate(sensor, [best["params"]])[0]
        print(f"  ★ 最优 score={best['score']:.2f} 过减={mb['over_adc']:.0f} 带宽={mb['band_adc']:.0f} "
              f"入带={mb['settle_s']:.1f}s 上漂={mb['up_adc']:.0f} 扣除={mb['ded_pct']:.0f}%", flush=True)
        print(f"    {json.dumps(best['params'], ensure_ascii=False)}", flush=True)
        result[sensor] = {"live": base_m, "best": {**mb, "params": best["params"]}, "runs": runs}
        (HERE / f"retune4_{sensor}.json").write_text(
            json.dumps(result[sensor], ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "retune4_all.json").write_text(
        json.dumps({k: {"live": v["live"], "best": v["best"]} for k, v in result.items()},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nbatches={NB}  写出 {HERE / 'retune4_all.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
