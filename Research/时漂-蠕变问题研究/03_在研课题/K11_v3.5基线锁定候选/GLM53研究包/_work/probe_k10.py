# -*- coding: utf-8 -*-
"""K10 探针：① track=0 与历史结果等价性；② track>0 在拇指/手掌上的即时行为。"""
import subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
from sensor_common import OUT_DIR  # noqa: E402
from sweep_lib import EXE, PREP, SESSION_BIN, Set, write_setfile  # noqa: E402

WORK = HERE / "_k10"
WORK.mkdir(exist_ok=True)


def run(key, named, time_mode="raw"):
    p = PREP[key]
    sf = write_setfile([Set(n, o) for n, o in named], WORK / "_s.txt")
    d = WORK / "_d"
    if d.exists():
        import shutil
        shutil.rmtree(d)
    d.mkdir()
    src = SESSION_BIN.get(key, p["path"])
    r = subprocess.run([str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}",
                        f"{p['t0']:.6f}", "--time", time_mode, "--zero", "--dump", str(d)],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:400])
    out = {}
    for n, _ in named:
        raw = np.fromfile(d / f"{n}.bin", dtype=np.int32, count=2)
        out[n] = np.fromfile(d / f"{n}.bin", dtype=np.float64, offset=8).reshape(int(raw[0]), int(raw[1]))
    return out


def m1(x, t, win=1.0):
    n = len(t)
    y = np.empty(n)
    j = 0
    s = 0.0
    cnt = 0
    for i in range(n):
        s += x[i]
        cnt += 1
        while t[i] - t[j] > win:
            s -= x[j]
            cnt -= 1
            j += 1
        y[i] = s / cnt
    return y


def report(tag, a, esum, t0):
    t, tin, out = a[:, 0], a[:, 1], a[:, 2]
    sm = m1(out, t)
    band = float(np.max(np.abs(sm[t >= 2.0]))) if (t >= 2.0).any() else 0.0
    settle = 0.0
    bad = np.where(np.abs(sm) > 150.0)[0]
    if bad.size:
        settle = float(t[bad[-1]])
    fl_out = float(np.std(out[t >= t[-1] - 60] - sm[t >= t[-1] - 60]))
    fl_in = float(np.std(tin[t >= t[-1] - 60] - m1(tin, t)[t >= t[-1] - 60]))
    print(f"    {tag:<28s} 过减={max(0.0, -out.min()):6.0f} 末值={out[-1] - esum:8.1f} "
          f"2s后带宽={band:6.0f} 入带时刻={settle:5.1f}s 波动保持={fl_out / max(fl_in, 1e-9):4.2f}")


# ① 等价性：live（track 默认 0）应与历史 05_report 一致（右拇指 d1 err_end≈1902）
print("== 等价性（track=0）==")
res = run("右拇指指腹/d1", [("live", {})])
print(f"  右拇指/d1 live err_end={res['live'][:, 2][-1] - PREP['右拇指指腹/d1']['Esum_channels']:.1f}"
      "（历史 1902.4）")

# ② 探针：track=1.0 + 宽门，拇指/手掌各一个会话
PROBE = {"r_fast": 0.02, "track_base_frac": 1.0, "slope_cap_frac": 0.15,
         "slope_gate_frac": 5.0, "slow_confirm_s": 0.0, "soft_unfreeze_s": 0.25,
         "edge_slope_thres": 10.0, "tau_slope_s": 1.0, "tau_r_slow_idle_s": 30.0,
         "tau_r_fast_s": 1.0, "hold_eps": 2.0, "idle_frac": 0.1, "ramp_slope_min": 0.0,
         "r_slow_max": 30.0, "tau_r_slow_s": 600.0}
for key in ("右拇指指腹/d1", "左拇指指腹/d1", "右手掌/d1/A", "左手掌/d1/A"):
    print(f"== {key} ==")
    named = [("live", {}), ("k10", PROBE), ("k10_cap3", {**PROBE, "slope_cap_frac": 0.3}),
             ("k10_cap6", {**PROBE, "slope_cap_frac": 0.6}),
             ("k10_cap10", {**PROBE, "slope_cap_frac": 1.0, "slope_gate_frac": 20.0})]
    r = run(key, named)
    esum = PREP[key]["Esum_channels"]
    for n, _ in named:
        report(n, r[n], esum, PREP[key]["t0"])
