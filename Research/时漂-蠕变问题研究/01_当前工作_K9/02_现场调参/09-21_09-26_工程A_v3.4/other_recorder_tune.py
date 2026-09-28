# -*- coding: utf-8 -*-
"""other_recorder 会话调参：v3.4 观测器（K9 口径）逐帧复算 + 单图总值对比。

数据：other_recorder/device_001_seg000.csv（session 20260924_130514_single_device_d96ffa，
      algorithm.enabled = false ⇒ 该 CSV 就是「算法输入」，显示模式 force / 单位 N）。

口径：
  * 算法实现 = creep_observer_k9.py（与 src/domain/drift_v6/creep_observer.cpp 逐帧对齐）。
  * 「现役默认」= creep_observer.h::Params 的当前默认值（2026-09-23 现场调参后的 5 项）。
  * 出图用 toolbox\\数据解析工具 的 dptool（overlay 单面板，signal=sum）。

用法：python other_recorder_tune.py [--out out]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402

DPTOOL_ROOT = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
CSV_PATH = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\other_recorder\device_001_seg000.csv")

# 现役默认（src/domain/drift_v6/creep_observer.h::Params，2026-09-23 现场调参后）
LIVE = replace(
    Params(),
    r_fast=0.06,              # 调参 0.10→0.06
    tau_r_fast_s=0.5,         # 调参 2→0.5
    soft_unfreeze_s=2.0,      # 调参 4→2
    slow_confirm_s=2.0,       # 调参 10→2
    tau_r_slow_idle_s=2.0,    # 调参 8→2
)


def read_any_session_csv(path: Path) -> dict:
    """读会话 CSV：按位置取 [时间戳, 经过时间, 帧序号] + 其余数值列作通道。

    本文件的列名是中文（单元0..51），run_v34_on_csv.read_session_csv 只认 ch/raw_ch 前缀，
    故这里用位置口径读。
    """
    txt = path.read_text(encoding="utf-8-sig").splitlines()
    i = txt.index("##Data")
    names = [c.strip() for c in txt[i + 1].split(",")]
    rows = [[float(x) if x.strip() != "" else np.nan for x in ln.split(",")]
            for ln in txt[i + 2:] if ln.strip()]
    arr = np.asarray(rows, dtype=np.float64)
    t = arr[:, 1]                       # 经过时间（秒）
    V = arr[:, 3:]                      # 通道值
    header = {}
    for ln in txt[:i]:
        if ln.startswith("##") or not ln.strip():
            continue
        k, _, v = ln.partition(",")
        header[k.strip()] = v.strip()
    return {"t": t, "V": V, "names": names[3:], "header": header}


def run(p: Params, t: np.ndarray, V: np.ndarray) -> dict:
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None      # 关掉 trace，省内存
    n = len(t)
    out = np.empty(n)
    x1 = np.empty(n)
    x2 = np.empty(n)
    ap = np.empty(n)
    tp = np.empty(n)
    for i in range(n):
        out[i] = c.process(float(t[i]), V[i]).sum()
        x1[i] = c.x_fast.sum()
        x2[i] = c.x_slow.sum()
        ap[i] = c.applied.sum()
        tp[i] = float(np.max(np.abs((V[i] - c.v_lp) / p.tau_slope_s)))
    return {"out": out, "x1": x1, "x2": x2, "applied": ap, "max_slope": tp}


def load_segments(tot: np.ndarray, t: np.ndarray, rel: float = 0.30):
    base = float(np.percentile(tot, 3))
    peak = float(np.max(tot))
    thr = base + rel * (peak - base)
    load = tot > thr
    segs = []
    i, n = 0, load.size
    while i < n:
        if load[i]:
            j = i
            while j + 1 < n and load[j + 1]:
                j += 1
            if j - i >= 5:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs, base, peak, thr


def seg_metrics(tot_in, res, t, segs):
    """每段：末值、达到末值 90% 的耗时（相对段首）、段内残余漂移。"""
    out = res["out"]
    rows = []
    for (i, j) in segs:
        ti = t[i:j + 1]
        a = tot_in[i:j + 1]
        b = out[i:j + 1]
        span = float(a[-1] - a[0])
        ded = a - b
        d_end = float(ded[-1])
        tgt = 0.9 * d_end
        idx = np.nonzero(ded >= tgt)[0] if d_end > 0 else np.array([], dtype=int)
        t90 = float(ti[idx[0]] - ti[0]) if idx.size else float("nan")
        rows.append({
            "t0": float(ti[0]), "t1": float(ti[-1]), "hold_s": float(ti[-1] - ti[0]),
            "in_start": float(a[0]), "in_end": float(a[-1]), "in_span": span,
            "out_end": float(b[-1]), "ded_end": d_end,
            "x1_end": float(res["x1"][j]), "x2_end": float(res["x2"][j]),
            "residual_drift": float(span - (b[-1] - b[0])),
            "t90_s": t90,
            "max_drop": float(np.max(ded)),
        })
    return rows


def write_session_like(path: Path, t, V, header, stage):
    n, m = V.shape
    ts = t + 171642.0
    lines = ["##Session"]
    h = dict(header)
    h["数据阶段"] = stage
    h["值阶段"] = stage
    h["行数"], h["列数"] = str(n), str(m)
    for k, v in h.items():
        lines.append(f"{k},{v}")
    lines.append("##Data")
    lines.append("timestamp,elapsed,frame_index," + ",".join(f"ch{j}" for j in range(m)))
    for i in range(n):
        lines.append(f"{ts[i]:.6f},{t[i]:.6f},{i}," +
                     ",".join(f"{V[i, j]:.6f}" for j in range(m)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


UI5 = dict(r_fast=0.10, slow_confirm_s=0.5, soft_unfreeze_s=0.5,
           tau_r_slow_idle_s=2.0, tau_r_fast_s=0.5)

VARIANTS = [
    ("00_输入(算法关闭)", None),
    ("01_现役默认", LIVE),
    ("02_仅UI五参拉满", replace(LIVE, **UI5)),
    ("03_tau_c_fast_12to6", replace(LIVE, tau_c_fast_s=6.0)),
    ("04_推荐_tauc6_confirm0.5", replace(LIVE, tau_c_fast_s=6.0, slow_confirm_s=0.5)),
    ("05_x2速率上限0.02", replace(LIVE, slope_cap_frac=0.02)),
    ("06_积极_tauc6_cap0.02_UI5", replace(LIVE, tau_c_fast_s=6.0, slope_cap_frac=0.02,
                                     **UI5)),
]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    out_dir = HERE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    d = read_any_session_csv(CSV_PATH)
    t, V = d["t"], d["V"]
    tot_in = V.sum(axis=1)
    print(f"数据 {CSV_PATH.name}：{V.shape[0]} 帧 × {V.shape[1]} 通道，"
          f"时长 {t[-1] - t[0]:.2f} s，{V.shape[0] / (t[-1] - t[0]):.1f} Hz")
    print(f"头部：session={d['header'].get('会话ID', '')} "
          f"阶段={d['header'].get('数据阶段', '')} "
          f"显示模式={d['header'].get('显示模式', '')} "
          f"单位={d['header'].get('显示力值单位', '')}")

    segs, base, peak, thr = load_segments(tot_in, t)
    print(f"基线 {base:.3f} 峰值 {peak:.3f} 受载判据 {thr:.3f} 受载段 {len(segs)} 个")
    for k, (i, j) in enumerate(segs):
        print(f"  段{k+1}: {t[i]:7.2f}~{t[j]:7.2f}s ({t[j]-t[i]:5.1f}s) "
              f"首 {tot_in[i]:7.3f} 末 {tot_in[j]:7.3f} 峰 {tot_in[i:j+1].max():7.3f}")

    # 逐帧通道斜率尺度（判断绝对阈值是否落在有效量程内）
    dch = np.abs(np.diff(V, axis=0)) / np.maximum(np.diff(t)[:, None], 1e-9)
    print(f"逐通道 |dv/dt| 分位：p50={np.percentile(dch, 50):.4f} "
          f"p99={np.percentile(dch, 99):.4f} max={dch.max():.4f} "
          f"（沿阈 edge_slope_thres=60、缓坡阈 ramp_slope_min=0.5）")

    summary = {"csv": str(CSV_PATH), "frames": int(V.shape[0]), "channels": int(V.shape[1]),
               "duration_s": float(t[-1] - t[0]),
               "baseline_total": base, "peak_total": peak, "segments_raw": segs,
               "variants": {}}
    curves = {}

    for label, p in VARIANTS:
        if p is None:
            curves[label] = tot_in
            summary["variants"][label] = {"note": "算法关闭（录制原值）"}
            continue
        res = run(p, t, V)
        curves[label] = res["out"]
        m = seg_metrics(tot_in, res, t, segs)
        summary["variants"][label] = {
            "params_diff": {k: getattr(p, k) for k in Params.names()
                            if getattr(p, k) != getattr(LIVE, k)},
            "max_slope_ch": float(res["max_slope"].max()),
            "ded_end": float((tot_in - res["out"])[-1]),
            "ded_max": float((tot_in - res["out"]).max()),
            "segments": m,
        }
        tail = "  ".join(f"[段{i+1} 扣{s['ded_end']:.3f}(t90={s['t90_s']:.1f}s)]"
                         for i, s in enumerate(m))
        print(f"{label:16s} 末帧扣除 {float((tot_in - res['out'])[-1]):7.3f}  {tail}")

    # 落盘逐帧总量
    npz = out_dir / "other_recorder_totals.npz"
    np.savez(npz, t=t, **{k: v for k, v in curves.items()})
    (out_dir / "other_recorder_tune.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {npz.name} / other_recorder_tune.json")

    # ── dptool 单面板 overlay 出图：一个变体一个临时目录（series_by=dir 用目录名当图例）──
    sys.path.insert(0, str(DPTOOL_ROOT))
    from dptool import api  # noqa: E402

    tmp = out_dir / "_dptool"
    if tmp.exists():                     # 清掉上一次跑留下的旧变体目录，避免混进图里
        import shutil
        shutil.rmtree(tmp)
    for label, y in curves.items():
        dd = tmp / label
        dd.mkdir(parents=True, exist_ok=True)
        write_session_like(dd / "device_001_seg000.csv", t, y[:, None], d["header"],
                           "processed_display")
    dirs = [str(tmp / k) for k in curves]
    png = out_dir / "other_recorder_总值对比.png"
    res = api.plot_to_file(dirs, str(png), mode="overlay", series_by="dir",
                           series_stream="out", signal="sum", smooth_s=0.0,
                           suptitle="v3.4 观测器调参对比（other_recorder 20260924_130514，52 通道总值 / N）",
                           stats=False, dpi=130)
    print("dptool 出图：", res.get("path") or res.get("error"))

    png2 = out_dir / "other_recorder_总值对比_放大.png"
    from dptool import merge as M  # noqa: E402
    tbl, _ = M.merge(dirs, start=int(np.searchsorted(t, 24.5)))
    M.plot_table(tbl, str(png2), mode="overlay", series_by="dir",
                 series_stream="out", signal="sum", ylim=(7.6, 12.6), dpi=130,
                 suptitle="同图放大（24.5 s 之后 = 第二段/后续段，y 轴 7.6~12.6 N）")
    print("dptool 放大图：", png2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
