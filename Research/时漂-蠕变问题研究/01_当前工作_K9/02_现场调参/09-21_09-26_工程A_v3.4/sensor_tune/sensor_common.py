# -*- coding: utf-8 -*-
"""5 类传感器实机数据（D:\\workshop\\文档\\v2.7 - 抗蠕变补偿算法\\data）的
   通用读取 / v3.4(K9) 离线复算 / 指标评估 公共层。

口径
----
* 算法实现 = v3.4\\creep_observer_k9.py（与 src/domain/drift_v6/creep_observer.cpp 逐帧对齐）。
* 「现役默认」= src/domain/drift_v6/creep_observer.h::Params 当前默认值（2026-09-23 现场调参后）。
* 数据 = 算法关闭（session.json algorithm.enabled=false）的 processed_display 录制，
  即该 CSV 就是「算法输入」；显示模式 adc / 力值视会话而定。
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V34 = HERE.parent                                   # ...\v3.4
sys.path.insert(0, str(V34))

from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402

DATA_ROOT = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data")
DPTOOL_ROOT = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")
OUT_DIR = HERE / "out"

# 现役默认（creep_observer.h::Params，2026-09-23 现场调参后的 5 项覆盖）
LIVE = replace(
    Params(),
    r_fast=0.06,              # 调参 0.10→0.06
    tau_r_fast_s=0.5,         # 调参 2→0.5
    soft_unfreeze_s=2.0,      # 调参 4→2
    slow_confirm_s=2.0,       # 调参 10→2
    tau_r_slow_idle_s=2.0,    # 调参 8→2
)

# UI 可调的 8 项（creep_observer_params_dialog.cpp::kSpecs，顺序一致）
UI_KEYS = ["r_fast", "tau_c_fast_s", "slow_confirm_s", "soft_unfreeze_s",
           "slope_cap_frac", "r_slow_max", "tau_r_slow_idle_s", "tau_r_fast_s"]

# 5 类传感器 → 会话清单（相对 DATA_ROOT）
SENSORS: dict[str, dict] = {
    "四指指腹": {
        "sessions": {
            "d1/6a679f": "四指指腹/device1/20260926_091946_single_device_6a679f/device_001_seg000.csv",
            "d1/5ce2c4": "四指指腹/device1/20260926_092426_single_device_5ce2c4/device_001_seg000.csv",
            "d2/857759": "四指指腹/device2/20260926_092950_single_device_857759/device_001_seg000.csv",
            "d2/8e127e": "四指指腹/device2/20260926_093310_single_device_8e127e/device_001_seg000.csv",
        },
    },
    "右拇指指腹": {
        "sessions": {
            "d1": "右拇指指腹/device1/device_001_seg000.csv",
            "d2": "右拇指指腹/device2/device_001_seg000.csv",
        },
    },
    "左拇指指腹": {
        "sessions": {
            "d1": "左拇指指腹/device1/device_001_seg000.csv",
            "d2": "左拇指指腹/device2/device_001_seg000.csv",
        },
    },
    "右手掌": {
        "sessions": {
            "d1/A": "右手掌/device1/右手掌A/device_001_seg000.csv",
            "d1/B": "右手掌/device1/右手掌B/device_001_seg000.csv",
            "d2/A": "右手掌/device2/右手掌A/device_001_seg000.csv",
            "d2/B": "右手掌/device2/右手掌B/device_001_seg000.csv",
        },
    },
    "左手掌": {
        "sessions": {
            "d1/A": "左手掌/device1/左手掌A/device_001_seg000.csv",
            "d1/B": "左手掌/device1/左手掌B/device_001_seg000.csv",
            "d2/A": "左手掌/device2/左手掌A/device_001_seg000.csv",
            "d2/B": "左手掌/device2/左手掌B/device_001_seg000.csv",
        },
    },
}


def read_session_csv(path: Path) -> dict:
    """按位置读会话 CSV：[时间戳, 经过时间, 帧序号] + 其余数值列作通道。"""
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
    return {"t": t, "V": V, "names": names[3:], "header": header, "path": path}


_cache: dict[str, dict] = {}


def load(key: str) -> dict:
    if key not in _cache:
        _cache[key] = read_session_csv(DATA_ROOT / key)
    return _cache[key]


def sessions_of(sensor: str) -> dict[str, str]:
    return SENSORS[sensor]["sessions"]


def load_sensor(sensor: str) -> dict[str, dict]:
    return {tag: load(rel) for tag, rel in sessions_of(sensor).items()}


def run(p: Params, t: np.ndarray, V: np.ndarray) -> dict:
    """逐帧复算，返回总值/状态轨迹（trace 已关闭，省内存）。"""
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    n = len(t)
    tot_in = V.sum(axis=1)
    out = np.empty(n)
    x1 = np.empty(n)
    x2 = np.empty(n)
    ap = np.empty(n)
    byp = np.empty(n, dtype=bool)
    for i in range(n):
        out[i] = c.process(float(t[i]), V[i]).sum()
        x1[i] = c.x_fast.sum()
        x2[i] = c.x_slow.sum()
        ap[i] = c.applied.sum()
        byp[i] = bool(np.sum(V[i]) < max(p.bypass_release_frac * c.total_baseline,
                                         c.total_baseline + p.bypass_noise_sigma * c.total_noise))
    return {"t": t, "in": tot_in, "out": out, "x1": x1, "x2": x2,
            "applied": ap, "ded": tot_in - out, "bypass": byp, "n_ch": V.shape[1]}


# ────────────────────────── 工况自动分段 ──────────────────────────
def load_segments(tot: np.ndarray, rel: float = 0.30, min_frames: int = 20):
    """按总读数的相对高度切出「受载段」（> 基线 + rel·(峰−基线)）。"""
    base = float(np.percentile(tot, 3))
    peak = float(np.max(tot))
    if peak - base <= 0:
        return [], base, peak, base
    thr = base + rel * (peak - base)
    load = tot > thr
    segs = []
    i, n = 0, load.size
    while i < n:
        if load[i]:
            j = i
            while j + 1 < n and load[j + 1]:
                j += 1
            if j - i >= min_frames:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs, base, peak, thr


def summarize_run(res: dict, segs, base: float, peak: float) -> dict:
    """把一次复算压成标量指标（越大越好的语义统一写在 score）。"""
    t, tin, out = res["t"], res["in"], res["out"]
    n = len(t)
    ded = tin - out
    m60 = slice(int(n * 0.4), n)
    m_last30 = t >= t[-1] - 30.0
    tail30 = float(np.polyfit(t[m_last30], out[m_last30], 1)[0]) if m_last30.sum() > 5 else 0.0
    return {
        "frames": int(n),
        "duration_s": float(t[-1] - t[0]),
        "baseline": float(base),
        "peak": float(peak),
        "n_seg": len(segs),
        "ded_end": float(ded[-1]),
        "ded_max": float(ded.max()),
        "ded_ratio_end": float(ded[-1] / max(peak - base, 1e-9)),
        "out_tail30_slope": tail30,
        "out_tail30_slope_pct_per_s": tail30 / max(peak - base, 1e-9) * 100.0,
        "out_last": float(out[-1]),
        "in_last": float(tin[-1]),
        "idle_mean_err": float(np.mean((out - tin)[~res["bypass"]])) if (~res["bypass"]).any() else 0.0,
        "bypass_ratio": float(np.mean(res["bypass"])),
        "x2_end": float(res["x2"][-1]),
    }


def seg_end_table(res: dict, segs) -> list[dict]:
    """每受载段的「落点一致性」：段末扣除量 / 段内残余上漂（扣除后仍在下漂的速率）。"""
    t, tin, out = res["t"], res["in"], res["out"]
    rows = []
    for idx, (i, j) in enumerate(segs):
        ti = t[i:j + 1]
        a, b = tin[i:j + 1], out[i:j + 1]
        hold = float(ti[-1] - ti[0])
        ded = a - b
        m = ti >= ti[-1] - min(30.0, hold * 0.5)
        slope = float(np.polyfit(ti[m], b[m], 1)[0]) if m.sum() > 5 else 0.0
        rows.append({
            "seg": idx + 1, "t0": float(ti[0]), "t1": float(ti[-1]), "hold_s": hold,
            "in_span": float(a[-1] - a[0]), "out_span": float(b[-1] - b[0]),
            "ded_end": float(ded[-1]), "out_end": float(b[-1]),
            "tail_slope": slope,
            "residual_ratio": float((b[-1] - b[0]) / max(a[-1] - a[0], 1e-9)),
        })
    return rows


def fmt(v: float, w: int = 8, p: int = 2) -> str:
    return f"{v:{w}.{p}f}"
