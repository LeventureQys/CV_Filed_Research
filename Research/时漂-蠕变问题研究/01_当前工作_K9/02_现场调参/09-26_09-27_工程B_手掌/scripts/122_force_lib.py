# -*- coding: utf-8 -*-
"""122_force_lib：力值模式（N 尺度）指标口径与复算器调用。

指标（全部在 1 s 中值滤波后的显示总值上算，段末留 0.6 s 余量）：
  弹性电平 E   = 台阶后 0.1~0.6 s 输入最小值（逐级累加，全卸载后重新起算）
  落点偏差     = 段末显示 − E
  回调/下坠    = 段内峰值 − 段末
  过扣         = E − 段内最小值
  稳定时间     = 显示 1 s 中值末次偏离落点 > 0.3 N（=300 ADC）的时刻（自台阶起算）
"""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
OUT = TEMP / "palm7" / "out"
EXE = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\01_当前工作_K9\02_现场调参\09-21_09-26_工程A_v3.4\sensor_tune\build\v34_sweep_k11.exe")
TOL_SETTLE = 0.3  # N（=300 ADC）

REC8 = {"r_fast": 0.03, "tau_c_fast_s": 40.0, "slow_confirm_s": 2.0, "soft_unfreeze_s": 2.0,
        "slope_cap_frac": 0.011, "r_slow_max": 0.35, "tau_r_fast_s": 6.0,
        "tau_r_slow_idle_s": 0.5}


def medfilt1s(y: np.ndarray, fps: float) -> np.ndarray:
    k = max(int(round(fps)) | 1, 3)
    pad = np.pad(y, k // 2, mode="edge")
    idx = np.arange(y.size)[:, None] + np.arange(k)[None, :]
    return np.median(pad[idx], axis=1)


def steps(t: np.ndarray, tot: np.ndarray, thr: float = 1.0, k: int = 30) -> list[tuple[float, int]]:
    d = np.abs(tot[k:] - tot[:-k])
    hot = np.flatnonzero(d > thr)
    if hot.size == 0:
        return []
    groups = [[int(hot[0])]]
    for i in hot[1:]:
        if i - groups[-1][-1] <= 60:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return [(float(t[g[int(np.argmax(d[g]))]]), g[int(np.argmax(d[g]))]) for g in groups]


def segments(t: np.ndarray, tin: np.ndarray, e_win: tuple[float, float] = (0.35, 1.0),
             thr: float = 1.0, min_dur: float = 2.0) -> list[dict]:
    """把录制切成受载段（含全卸载后的重新起算）。

    本批力值录制的加载是 ~0.4~0.6 s 的斜坡而非理想阶跃，所以「台阶后最小值」若从
    t0+0.1 s 起算会落在斜坡中间（低估弹性电平）。主口径取 **加载斜坡结束后的平台**：
    E = 输入在 [t0+e_win[0], t0+e_win[1]] 的最小值（= 斜坡刚做完时的读数）。
    同时记录旧口径 E_fast（[t0+0.1, t0+0.6]）便于与历史结论对齐。
    """
    st = steps(t, tin, thr=thr)
    segs = []
    for j, (tt, i) in enumerate(st):
        m = (t >= tt + e_win[0]) & (t <= tt + e_win[1])
        E = float(tin[m].min()) if m.any() else float(tin[i])
        mf = (t >= tt + 0.1) & (t <= tt + 0.6)
        E_fast = float(tin[mf].min()) if mf.any() else E
        nxt = st[j + 1][0] if j + 1 < len(st) else float(t[-1])
        if E <= 0.2:  # 卸载台阶
            continue
        if nxt - tt < min_dur:
            continue
        segs.append({"t0": tt, "t1": nxt, "E": E, "E_fast": E_fast,
                     "i0": int(np.searchsorted(t, tt + e_win[0])),
                     "i1": int(np.searchsorted(t, nxt - 0.6))})
    return segs


def evaluate(out_tot: np.ndarray, t: np.ndarray, segs: list[dict], fps: float,
             raw: bool = False) -> list[dict]:
    """raw=True 时不作 1 s 中值（用于核对「未滤波」口径下的下坠）。"""
    y = out_tot if raw else medfilt1s(out_tot, fps)
    rows = []
    for s in segs:
        a, b = s["i0"], max(s["i1"], s["i0"] + 1)
        yy, ttv = y[a:b], t[a:b]
        E = s["E"]
        end = float(yy[-1])
        peak = float(yy.max())
        lo = float(yy.min())
        bad = np.abs(yy - end) > TOL_SETTLE
        tsettle = float(ttv[np.flatnonzero(bad)[-1]] - s["t0"]) if bad.any() else 0.0
        mm = ttv >= ttv[-1] - 3.0
        slope = float(np.polyfit(ttv[mm], yy[mm], 1)[0]) if mm.sum() > 10 else 0.0
        i04 = int(np.searchsorted(t, s["t0"] + 0.35))
        d04 = float(y[min(i04, len(y) - 1)])
        rows.append({"t0": s["t0"], "t1": s["t1"], "E": E, "E_fast": s["E_fast"],
                     "end": end, "dev": end - E, "dev_fast": end - s["E_fast"],
                     "drift": end - d04, "drop": peak - end,
                     "over": max(0.0, E - lo), "peak": peak, "min": lo,
                     "tsettle": tsettle, "slope_end": slope,
                     "peak_t": float(ttv[int(np.argmax(yy))])})
    return rows


def fmt(rows: list[dict]) -> str:
    out = []
    for j, r in enumerate(rows):
        out.append(f"段{j+1}[{r['t0']:.2f}→{r['t1']:.2f}s] E={r['E']:7.3f}N "
                   f"落点={r['dev']:+7.3f} 下坠={r['drop']:5.3f} 过扣={r['over']:5.3f} "
                   f"漂移={r['drift']:+6.3f} 稳定={r['tsettle']:5.2f}s "
                   f"末斜率={r['slope_end']:+7.4f}N/s 峰@{r['peak_t']:5.2f}s")
    return "\n".join(out)


def run_exe(bin_path: Path, sets: list[tuple[str, dict]], esum: float, t0: float,
            tag: str = "run", time_mode: str = "raw", dump: bool = True) -> tuple[list[dict], Path]:
    """调用离线复算器（链接产品源码）。sets: [(ASCII名, 覆盖字典)]。返回解析后的行与 dump 目录。"""
    pfile = OUT / f"_params_{tag}.txt"
    lines = []
    for name, over in sets:
        body = ",".join(f"{k}={v!r}" for k, v in over.items())
        lines.append(f"{name}|{body}" if body else name)
    pfile.write_text("\n".join(lines) + "\n", encoding="ascii")
    ddir = OUT / f"_dump_{tag}"
    ddir.mkdir(parents=True, exist_ok=True)
    cmd = [str(EXE), str(bin_path), str(pfile), f"{esum!r}", f"{t0!r}", "--time", time_mode]
    if dump:
        cmd += ["--dump", str(ddir)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"exe failed rc={p.returncode}: {p.stderr}")
    hdr = p.stdout.splitlines()[0].split(",")
    rows = []
    for ln in p.stdout.splitlines()[1:]:
        v = ln.split(",")
        if len(v) != len(hdr):
            continue
        rows.append({k: (val if k == "name" else float(val)) for k, val in zip(hdr, v)})
    return rows, ddir


def read_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as f:
        n, cols = struct.unpack("<ii", f.read(8))
        a = np.frombuffer(f.read(n * cols * 8), dtype="<f8").reshape(n, cols)
    return a[:, 0], a[:, 1], a[:, 2]


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    z = np.load(OUT / "force_1d925c.npz")
    t, tin, tout = z["t"], z["tot_pre"], z["tot_seg"]
    fps = (len(t) - 1) / (t[-1] - t[0])
    sg = segments(t, tin)
    print("测得的受载段：")
    for s in sg:
        print(f"  t0={s['t0']:.2f} t1={s['t1']:.2f} E={s['E']:.3f}N "
              f"段末输入={float(tin[s['i1']]):.3f}N")
    print("\n录制显示（算法开）指标：")
    print(fmt(evaluate(tout, t, sg, fps)))
