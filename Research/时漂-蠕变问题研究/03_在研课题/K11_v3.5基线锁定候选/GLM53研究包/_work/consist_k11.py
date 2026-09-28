# -*- coding: utf-8 -*-
"""反复荷载一致性：逐加载周期提取 K11/live 的显示响应，量化跨周期一致性。

每个受载周期提取：
  in_step   该周期输入平台高度（相对前空隙基线）
  out_peak  加载后 2s 内显示峰值（应≈in_step：台阶透传）
  out_2s    加载 2s 时刻显示（freeze 窗刚过，开始被吸收）
  out_end   周期末显示（保持期残差）
  gap_base  周期前空隙的显示基线（棘轮检测：应恒≈0）
一致性 = 各量跨周期的 mean±std 与极差；棘轮 = gap_base 随周期数的斜率。
"""
import subprocess, shutil, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
import sweep_lib  # noqa: E402
sweep_lib.EXE = ST / "build" / "v34_sweep_k11.exe"
from sweep_lib import Set, write_setfile  # noqa: E402
from sensor_common import read_session_csv  # noqa: E402
from retune_v4 import read_bin  # noqa: E402

ROOT = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\真实复杂工况\算法数据&原始数据"
            r"\算法数据&原始数据\working")
CASES = [
    ("零基线-反复增减同一负载/20260919_193935_single_device_9c3ca5", 0.10, 0.3),
    ("零基线-反复增减同一负载/20260919_191748_single_device_795e5e/20260919_192141_single_device_73032d", 0.10, 0.3),
    ("零基线-同一荷载测试/回撤测试/20260920_171953_single_device_2e35b8", 0.10, 0.3),
    ("零基线-剧烈变化负载情况/20260919_192655_single_device_d675cf", 0.15, 0.3),
]
LOCK = {"r_fast": 0.0, "slope_gate_frac": 1e-6, "r_slow_max": 0.0, "hold_eps": 0.0,
        "hold_lock_freeze_s": 2.0}
NAMED = [("live", {}), ("k11_015", {**LOCK, "hold_lock_tau_s": 0.15, "edge_slope_thres": 150.0}),
         ("k11_030", {**LOCK, "hold_lock_tau_s": 0.3, "edge_slope_thres": 150.0})]
WORK = HERE / "_consist"
WORK.mkdir(exist_ok=True)


def cycles(t, tin, rel, min_s):
    base = float(np.percentile(tin, 5))
    peak = float(np.max(tin))
    thr = base + rel * (peak - base)
    load = tin > thr
    segs, i, n = [], 0, len(t)
    while i < n:
        if load[i]:
            j = i
            while j + 1 < n and load[j + 1]:
                j += 1
            if (t[j] - t[i]) >= min_s:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs, base, peak


for rel_dir, relth, min_s in CASES:
    p = next((ROOT / rel_dir).rglob("device_001_pre_seg0.csv"))
    d = read_session_csv(p)
    t = d["t"] - d["t"][0]
    V = d["V"] - d["V"][0]
    n, m = V.shape
    tin = V.sum(axis=1)
    f = WORK / "_s.bin"
    with f.open("wb") as fh:
        fh.write(np.int32(n).tobytes()); fh.write(np.int32(m).tobytes())
        rows = np.empty((n, m + 1)); rows[:, 0] = t; rows[:, 1:] = V
        fh.write(rows.astype(np.float64).tobytes())
    sf = write_setfile([Set(nm, o) for nm, o in NAMED], WORK / "_p.txt")
    dd = WORK / "_d"
    if dd.exists(): shutil.rmtree(dd)
    dd.mkdir()
    r = subprocess.run([str(sweep_lib.EXE), str(f), str(sf), "0.0", "0.0",
                        "--time", "raw", "--zero", "--dump", str(dd)],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(rel_dir, "FAIL", r.stderr[:200]); continue
    segs, base, peak = cycles(t, tin, relth, min_s)
    name = rel_dir.split("/")[-1][:24]
    print(f"\n===== {name}  周期数={len(segs)}")
    for nm, _ in NAMED:
        a = read_bin(dd / f"{nm}.bin")
        out = a[:, 2]
        rows_ = []
        for c, (i, j) in enumerate(segs):
            gap0 = i - 100
            gb = float(np.mean(out[max(gap0, 0):i])) if i > 20 else 0.0
            i2 = int(np.searchsorted(t, t[i] + 2.0))
            plat = float(np.mean(tin[i:min(i2, j)])) if j > i else 0.0
            in_step = plat - (float(np.mean(tin[max(gap0, 0):i])) if i > 20 else 0.0)
            i_end = min(j, n - 1)
            rows_.append((c + 1, in_step, float(np.max(out[i:min(i2, n)])), float(out[min(i2, n-1)]),
                          float(out[i_end]), gb))
        arr = np.array(rows_, dtype=object)
        ins = np.array([r_[1] for r_ in rows_]); pk = np.array([r_[2] for r_ in rows_])
        o2 = np.array([r_[3] for r_ in rows_]); oe = np.array([r_[4] for r_ in rows_])
        gb = np.array([r_[5] for r_ in rows_])
        ratio = pk / np.maximum(ins, 1.0)
        cyc = np.arange(len(gb))
        ratchet = float(np.polyfit(cyc, gb, 1)[0]) if len(gb) > 2 else 0.0
        print(f"  {nm:<8s} 峰/台阶={np.mean(ratio):.2f}±{np.std(ratio):.2f}  "
              f"2s末={np.mean(o2):7.0f}±{np.std(o2):5.0f}  周期末={np.mean(oe):7.0f}±{np.std(oe):5.0f}  "
              f"空隙基线={np.mean(gb):6.0f}±{np.std(gb):4.0f} (逐周期斜率={ratchet:+.1f}ADC/周期)")
        for r_ in rows_:
            print(f"      周期{r_[0]:>2d}: 台阶={r_[1]:7.0f} 峰={r_[2]:7.0f} 2s末={r_[3]:7.0f} "
                  f"周期末={r_[4]:7.0f} 空隙基线={r_[5]:6.0f}")
