# -*- coding: utf-8 -*-
"""T1-B 分析层：臂注册、事件表、单次运行与指标评估（供 t1b_04~t1b_07 共用）。

不修改任何原型：算法类来自本目录的 `t1b_glm53_v6/v61/v51`（复制件）。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))
RES = os.path.join(TASK, "results")
TEMP = os.path.join(ROOT, "temp")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from t1_common import (load_uniform, to_grid, med_smooth, make_traced, run_traced,   # noqa: E402
                       run_plain, t_stable, z_final_level, overshoot, settle_time, step_amp,
                       HOLD_REC, REC)

# T4-A 冻结事件表（只读引用；本任务的事件集以它为唯一真值来源，避免重做检测）
T4A_EVENTS = os.path.join(PLAN, "T4_两种快相形态与分支判据", "results", "t4a_morphology.csv")

import t1b_glm53_v6 as M6          # noqa: E402
import t1b_glm53_v61 as M61        # noqa: E402
import t1b_glm53_v51 as M51        # noqa: E402

V6 = M6.GLM53v6
V61 = M61.GLM53v61
V51 = M51.GLM53v51
TRACED = {"v6": make_traced(V6), "v61": make_traced(V61)}

ARMS = ("raw", "v6", "v61", "v51")

# 事件名 → 录制键（t4a_morphology.csv 的 rec 列 → 本库的短键）
REC_MAP = {
    "右拇指指尖/数据1": ("RT1", "显示域"), "右拇指指尖/数据2": ("RT2", "显示域"),
    "右拇指指尖/数据3": ("RT3", "显示域"), "左拇指指尖/数据1": ("LT1", "显示域"),
    "左拇指指尖/数据2": ("LT2", "显示域"), "左拇指指尖/数据3": ("LT3", "显示域"),
    "四指指尖/数据1": ("F41", "显示域"), "四指指尖/数据2": ("F42", "显示域"),
    "四指指尖/数据3": ("F43", "显示域"),
    "切换负载-快相无责": ("SW1", "ADC域"),
    "零负载-切换负载-零负载-再切换负载": ("SW2", "ADC域"),
    "零负载-中途切换负载-零负载-切换负载": ("SW3", "ADC域"),
    "中途切换-最终测试目标": ("SW4", "ADC域"),
    # T4-A 表里实录用的是短键（`t4a_morphology.csv:rec` 列的实际取值）
    "再切换负载": ("SW2", "ADC域"),
    "中途切换-1d9493": ("SW3", "ADC域"),
    "中途切换-13ffca": ("SW4", "ADC域"),
}

_CACHE = {}


def rec_path(key):
    if key in HOLD_REC:
        return HOLD_REC[key][0]
    if key in REC:
        return REC[key]
    from t1_common import REC_SHORT
    if key in REC_SHORT:
        return REC[REC_SHORT[key]]
    raise KeyError(key)


def get_grid(key):
    """载入并缓存 100 Hz 网格（每个 worker 进程一份，避免重复 I/O）。"""
    if key not in _CACHE:
        d = load_uniform(rec_path(key))
        d["key"] = key
        _CACHE[key] = d
    return _CACHE[key]


def event_table(path=T4A_EVENTS, min_hold_s=35.0):
    """从 T4-A 冻结事件表挑出 T1-B 可评估的事件：

    * 只取 `kind == "onset"`（空载→负载；T_stable/超调都以它为基准）；
    * `clean=True`（前后 3/8 s 无其它检出事件）；
    * 下一事件（或段末/记录末）距 `t_on` ≥ `min_hold_s`（否则 30 s 稳定窗被事件打断）。
    另外附：`L_hold`（保压电平，干净录制的中位）、`J_raw`（字典 §2.1 口径台阶幅度）、
    `gap_next`、`t_end`。
    """
    df = pd.read_csv(path)
    rows = []
    for rec, sub in df.groupby("rec"):
        sub = sub.sort_values("t_on").reset_index(drop=True)
        key, dom = REC_MAP[rec]
        d = get_grid(key)
        tu, Xu = d["tu"], d["Xu"]
        tot = med_smooth(Xu.sum(axis=1), 50)
        span = float(tu[-1])
        # 该录制全部事件时刻（用来算到下一事件的间隔）
        all_t = sub["t_on"].to_numpy(float)
        for _, r in sub.iterrows():
            if str(r["kind"]) != "onset" or not bool(r["clean"]):
                continue
            t_on = float(r["t_on"])
            later = all_t[all_t > t_on + 0.5]
            t_next = float(later.min()) if len(later) else span
            gap = t_next - t_on
            i0 = int(round(t_on / d["dtm"]))
            i1 = min(len(tu), i0 + int(60 / d["dtm"]))
            L = float(np.median(tot[i0 + int(10 / d["dtm"]):i1])) if i1 > i0 + 1000 else np.nan
            J = step_amp(tu, Xu.sum(axis=1), t_on, d["dtm"])
            rows.append(dict(ev=f"{key}@{t_on:.2f}", key=key, rec=rec, dom=dom, t_on=t_on,
                             t_end=span, t_next=t_next, gap_next=round(gap, 2),
                             main_ch=int(r["main_ch"]), J=J, L_hold=L,
                             t4a_jump=float(r["jump"]), t4a_pre=float(r["pre"]),
                             span_after=round(span - t_on, 2),
                             measurable=int(gap >= min_hold_s and (span - t_on) >= 70.0)))
    ev = pd.DataFrame(rows).sort_values(["dom", "key", "t_on"]).reset_index(drop=True)
    return ev


def crop(tu, Xu, t0, t1):
    a = int(round(t0 / 0.01))
    b = min(len(tu), int(round(t1 / 0.01)))
    a = max(0, a)
    return tu[a:b] - tu[a], Xu[a:b]


def run_arm(arm, tu, Xu, knobs=None):
    """跑一条臂：返回 dict(Y, tot_out, comp, epoch, revoke, handoff)（Z = Y.sum(1)）。"""
    knobs = dict(knobs or {})
    if arm == "raw":
        return dict(Y=Xu, Z=Xu.sum(axis=1), comp=None, epoch=[], revoke=[], handoff=[])
    if arm in TRACED:
        d = run_traced(TRACED[arm], tu, Xu, **knobs)
        return dict(Y=d["Y"], Z=d["Y"].sum(axis=1), comp=d["comp"], epoch=d["epoch"],
                    revoke=d["revoke"], handoff=d["handoff"])
    Cls = V51 if arm == "v51" else V6
    Y, c = run_plain(Cls, tu, Xu, **knobs)
    return dict(Y=Y, Z=Y.sum(axis=1), comp=c, epoch=getattr(c, "epoch_t", []),
                revoke=[], handoff=[])


def eval_event(tu, Zarm, t_on, J, t_next, dtm=0.01):
    """一个 (事件, 扰动, 臂) 的指标字典（口径 = 指标字典 §3）。"""
    span = float(tu[-1])
    horizon = min(30.0, max(0.0, t_next - t_on - 1.5))
    out = dict(z_at_1=None, z_at_2=None, t_stable_v1=np.nan, t_stable_v2=np.nan,
               t_stable_ok=0, os_pct=np.nan, us_pct=np.nan, z_final=np.nan, zfin_src="none",
               ts2=np.nan, ts5=np.nan, horizon=round(horizon, 2))
    if abs(J) < 1e-9 or not np.isfinite(J):
        return out
    Zb = med_smooth(Zarm, 50)
    i1 = int(round((t_on + 1.0) / dtm))
    i2 = int(round((t_on + 2.0) / dtm))
    i0 = int(round(t_on / dtm))
    if i2 < len(Zb):
        out["z_at_1"] = float((Zb[i1] - Zb[i0]) / J)
        out["z_at_2"] = float((Zb[i2] - Zb[i0]) / J)
    ts = t_stable(tu, Zarm, t_on, J, dtm=dtm, horizon=30.0)
    out["t_stable_v1"] = ts["tau_v1"]
    out["t_stable_v2"] = ts["tau_v2"]
    out["t_stable_ok"] = int(ts["ok"])
    zf, src = z_final_level(tu, Zarm, t_on, dtm=dtm)
    osd = overshoot(tu, Zarm, t_on, J, z_final=zf, dtm=dtm, win_s=30.0)
    out.update(os_pct=osd["os_pct"], us_pct=osd["us_pct"], z_final=osd["z_final"],
               zfin_src=src)
    out["ts2"] = settle_time(tu, Zarm, t_on, J, zf, eps=0.02, dtm=dtm)
    out["ts5"] = settle_time(tu, Zarm, t_on, J, zf, eps=0.05, dtm=dtm)
    return out


def maxdev(tu, Zref, Zarm, t_on, t1=None):
    """相对无扰动基线的最大偏差（ADC/显示单位）与慢相中位偏差。

    时序扰动会改变重采样后的网格长度（包抖动可加长/缩短记录末），这里按**公共长度**对齐。
    """
    i0 = int(round(t_on / 0.01))
    i1 = len(tu) if t1 is None else min(len(tu), int(round(t1 / 0.01)))
    a = med_smooth(Zarm, 50)[i0:i1]
    b = med_smooth(Zref, 50)[i0:i1]
    n = min(len(a), len(b))
    if n < 10:
        return np.nan, np.nan
    d = a[:n] - b[:n]
    half = n // 2
    return float(np.abs(d).max()), float(np.median(d[half:]))


def n_epoch_near(epoch_list, t_on, tol=0.8):
    """与真实事件配对（±tol s）的 epoch 数 / 该项之前/之后的额外 epoch 数。"""
    ts = np.array([e[0] for e in epoch_list], float) if len(epoch_list) else np.zeros(0)
    if ts.size == 0:
        return 0, 0, 0
    hit = int(np.sum(np.abs(ts - t_on) <= tol))
    pre = int(np.sum(ts < t_on - tol))
    post = int(np.sum(ts > t_on + tol))
    return hit, pre, post


def window_of(d, spec):
    """裁剪窗：`{"kind":"hold","t_on":x,"span":80}` = [0, t_on+span]（前缀，保证 warm-up 一致）；
    `{"kind":"full"}` = 全长。"""
    if spec.get("kind") == "full":
        return d["tu"], d["Xu"]
    return crop(d["tu"], d["Xu"], 0.0, float(spec["t_on"]) + float(spec["span"]))


def truth_events_all(path=T4A_EVENTS):
    """T4-A 事件表里**每份录制的全部事件时刻**（漏检/误触发配对的真值，含 restep/unload）。"""
    df = pd.read_csv(path)
    out = {}
    for rec, sub in df.groupby("rec"):
        key, _ = REC_MAP[rec]
        out.setdefault(key, []).extend(sub["t_on"].to_numpy(float).tolist())
    return {k: np.sort(np.array(v, float)) for k, v in out.items()}


def match_truth(truth, epoch_list, tol=0.8):
    """epoch ↔ 真值事件配对（±tol s）。返回 (匹配对数, 漏检时刻表, 误触发时刻表)。"""
    ts = np.array([e[0] for e in epoch_list], float) if len(epoch_list) else np.zeros(0)
    truth = np.asarray(truth, float)
    matched_t, matched_e = [], []
    for t in truth:
        if ts.size and np.min(np.abs(ts - t)) <= tol:
            matched_t.append(t)
    for e in ts:
        if truth.size and np.min(np.abs(truth - e)) <= tol:
            matched_e.append(e)
    miss = [t for t in truth if t not in matched_t]
    extra = [e for e in ts if e not in matched_e]
    return len(matched_t), miss, extra
