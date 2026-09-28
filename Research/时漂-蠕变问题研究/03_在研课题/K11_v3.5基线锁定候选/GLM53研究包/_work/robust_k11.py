# -*- coding: utf-8 -*-
"""K11 面对真实复杂工况的鲁棒性/一致性验证。

工况：剧烈变化负载 / 反复增减同一负载 / 同一荷载 / 回撤测试×6 / 缓坡。
输入 = device_001_pre_seg0.csv（当时算法开启录制的「算法前」流 = 离线复算的正确输入）。
对比参数集：现役默认 / K11 τ=0.15（拇指预设）/ K11 τ=0.8（手掌预设）/ K11 τ=0.3。

指标（全部对显示值的 1s 滑动均值 m1 计算，raw 时间轴）：
  * 段数 / 峰值：输入受载段（> 基线+30%·峰）计数与输入总峰值
  * 带宽：max|m1(显示)|（全程）
  * 最深持续下冲：min m1(显示)（过减口径，持续型）
  * 台阶可见性：每段起始 1s 内 m1(显示) 的均值 / 该段输入台阶高度（K11 语义下应≈1：
    freeze 窗内台阶直通；之后衰减回 0 属预期行为，不算失效）
  * 卸载回零：段结束后 |m1| 回到 <150 的最长时间
产物：robust_k11.json + robust_k11.log（GLM53/_work）
"""
import json, shutil, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ST = HERE.parent.parent / "v3.4" / "sensor_tune"
sys.path.insert(0, str(ST))
import sweep_lib  # noqa: E402
sweep_lib.EXE = ST / "build" / "v34_sweep_k11.exe"
from sweep_lib import Set, write_setfile  # noqa: E402
from sensor_common import read_session_csv  # noqa: E402
from retune_v4 import m1, read_bin  # noqa: E402

ROOT = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\真实复杂工况\算法数据&原始数据"
            r"\算法数据&原始数据\working")
WORK = HERE / "_robust"
WORK.mkdir(exist_ok=True)

LOCK = {"r_fast": 0.0, "slope_gate_frac": 1e-6, "r_slow_max": 0.0, "hold_eps": 0.0,
        "hold_lock_freeze_s": 2.0, "edge_slope_thres": 60.0}
NAMED = [("live", {}),
         ("k11_015", {**LOCK, "hold_lock_tau_s": 0.15, "edge_slope_thres": 150.0}),
         ("k11_030", {**LOCK, "hold_lock_tau_s": 0.3}),
         ("k11_080", {**LOCK, "hold_lock_tau_s": 0.8})]


def to_bin(d):
    t = d["t"] - d["t"][0]
    V = d["V"]
    n, m = V.shape
    f = WORK / "_sess.bin"
    with f.open("wb") as fh:
        fh.write(np.int32(n).tobytes())
        fh.write(np.int32(m).tobytes())
        rows = np.empty((n, m + 1))
        rows[:, 0] = t
        rows[:, 1:] = V
        fh.write(rows.astype(np.float64).tobytes())
    return f, t, V


def segments(tin):
    base = float(np.percentile(tin, 5))
    peak = float(np.max(tin))
    thr = base + 0.30 * (peak - base)
    load = tin > thr
    segs, i, n = [], 0, load.size
    while i < n:
        if load[i]:
            j = i
            while j + 1 < n and load[j + 1]:
                j += 1
            if j - i >= 20:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs, base, peak


def main():
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8")
    sessions = sorted(ROOT.rglob("device_001_pre_seg0.csv"))
    print(f"共 {len(sessions)} 份会话\n")
    out = {}
    for path in sessions:
        cond = path.relative_to(ROOT).parts[0]
        tag = path.parent.name.replace("single_device_", "")
        key = f"{cond}/{tag}"
        d = read_session_csv(path)
        binf, t, V = to_bin(d)
        tin = V.sum(axis=1) - V[0].sum()
        segs, base, peak = segments(tin)
        sf = write_setfile([Set(n, o) for n, o in NAMED], WORK / "_s.txt")
        dd = WORK / "_d"
        if dd.exists():
            shutil.rmtree(dd)
        dd.mkdir()
        r = subprocess.run([str(sweep_lib.EXE), str(binf), str(sf), "0.0", "0.0",
                            "--time", "raw", "--zero", "--dump", str(dd)],
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode != 0:
            print(f"!! {key}: {r.stderr[:200]}")
            continue
        print(f"===== {key}  段数={len(segs)} 峰值={peak:.0f}ADC 时长={t[-1]:.0f}s")
        res = {"cond": cond, "n_seg": len(segs), "peak": peak, "params": {}}
        for nm, _ in NAMED:
            a = read_bin(dd / f"{nm}.bin")
            out_disp = a[:, 2]
            sm = m1(out_disp, t)
            band = float(np.max(np.abs(sm)))
            deep = float(sm.min())
            step_vis, recov_max = [], 0.0
            for (i, j) in segs:
                pre_ref = float(sm[max(i - 30, 0)]) if i > 30 else 0.0
                h0 = pre_ref
                i1 = min(i + 100, len(t) - 1)
                step_in = float(tin[min(i + 100, len(t) - 1)] - tin[max(i - 30, 0)])
                step_out = float(np.mean(sm[i:i1]) - h0)
                if step_in > 50:
                    step_vis.append(step_out / step_in)
                # 段末之后回带时间
                k = j + 1
                t0 = t[min(j, len(t) - 1)]
                rec = 99.0
                while k < len(t):
                    if abs(sm[k]) < 150.0:
                        rec = float(t[k] - t0)
                        break
                    k += 1
                recov_max = max(recov_max, rec)
            res["params"][nm] = {
                "band_m1": band, "deep_m1": deep,
                "step_vis_mean": float(np.mean(step_vis)) if step_vis else None,
                "step_vis_min": float(np.min(step_vis)) if step_vis else None,
                "recov_max_s": recov_max,
                "disp_end": float(out_disp[-1]),
            }
            p = res["params"][nm]
            print(f"  {nm:<8s} 带宽={p['band_m1']:7.0f} 最深持续下冲={p['deep_m1']:8.0f} "
                  f"台阶可见={'' if p['step_vis_mean'] is None else f'{p['step_vis_mean']:.2f}'} "
                  f"回带最慢={p['recov_max_s']:5.1f}s 末值={p['disp_end']:8.1f}")
        out[key] = res
    (HERE / "robust_k11.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n写出 {HERE / 'robust_k11.json'}")


main()
