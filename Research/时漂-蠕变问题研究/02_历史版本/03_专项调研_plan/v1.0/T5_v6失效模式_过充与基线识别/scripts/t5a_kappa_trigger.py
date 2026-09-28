# -*- coding: utf-8 -*-
"""T5-A κ 上限触发诊断（**回应对 κ 语义的纠正：κ 是单侧上限 cap，不是增益**）。

κ 在原型里只在 `_inv_est` 结尾的一处生效：`min(A, kappa * inc)`。
**只有当未封顶反演值 `A_raw > κ·inc` 时 κ 才起作用。**

本脚本用探针补丁 `t5a_kappa_probe.py`（`t5a_patch_ab_zero.csv` 的 V6/V7/V8 已证零差）
把每帧的 `A_raw / inc` 落盘，然后给出：

  ① **触发率的直接判据**（最严格、无近似）：对每个 κ，检查该事件轨迹里
     **是否真的存在某一帧满足 `A_raw > κ·inc`**，以及有多少帧被截断
     （帧数 × 10 ms = 被截断的时长）。
  ② **触发阈值**：`r(τ) = A_raw(τ)/inc(τ)` 的逐事件分位数 —— κ 高于 `max r` 则**永远不触发**，
     κ 落在中位附近则约一半事件触发。
  ③ **κ × 触发率表**（onset / restep 两族分开），用来解释 T4-B 的零差。

产出：`results/t5a_kappa_trigger_raw.csv`（逐帧 r 轨迹）、
      `results/t5a_kappa_trigger_perevent.csv`（逐事件 r 分位与 r_max）、
      `results/t5a_kappa_trigger.csv`（κ × 触发率）、
      `_t5a_kappa_trigger.log`。

用法：`python scripts/t5a_kappa_trigger.py`
"""
import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t5a_common as C                                  # noqa: E402
from t5a_kappa_probe import KV6P                        # noqa: E402

LOG = []
T0 = time.time()


def rec(m):
    s = f"[{time.time() - T0:7.1f}s] {m}"
    print(s, flush=True)
    LOG.append(s)


GRID_ONSET = [1.30, 1.25, 1.20, 1.15, 1.10, 1.05, 1.00, 0.95]
GRID_RESTEP = [1.12, 1.08, 1.04, 1.00, 0.96]
TAU_GRID = [0.20, 0.30, 0.50, 0.80, 1.00]


def main():
    ev = C.ev_load_frozen()
    recs = C.recordings()
    rec(f"事件 {len(ev)}（onset {int((ev.kind=='onset').sum())} / "
        f"restep {int((ev.kind=='restep').sum())}）；录制 {len(recs)}")
    rec("κ 的语义：`Â = min(A_raw, κ·inc)`（单侧上限）；A_raw 含原型自带的 `max(A, inc)` 下限")

    rows = []          # 逐事件：r 的分位 + 触发帧数
    frames = []        # 抽样逐帧轨迹（每事件每 0.1 s 一点，控制体积）
    for k, d in recs.items():
        sub = ev[ev.key == k]
        if not len(sub):
            continue
        c = KV6P(d["Xu"].shape[1])
        c.kappa_onset, c.kappa_restep = 1.30, 1.12
        c.enable_probe = True
        out = c.run(d["tu"], d["Xu"])
        tu = d["tu"]
        A_raw, incv, kp = out["A_raw"], out["inc_probe"], out["kappa_probe"]
        for _, e in sub.iterrows():
            t0 = float(e["t_on"])
            i0 = int(np.searchsorted(tu, t0))
            i1 = int(np.searchsorted(tu, t0 + 30.0))
            a = A_raw[i0:i1]
            ic = incv[i0:i1]
            kk = kp[i0:i1]
            ok = np.isfinite(a) & np.isfinite(ic) & (ic > 1e-9)
            r = np.where(ok, a / np.maximum(ic, 1e-12), np.nan)
            tt = tu[i0:i1] - t0
            # 只在事件确实处于 event 态、且该帧的反演用的是本类事件的 κ 时统计
            is_ev = np.isfinite(kk)
            r_ev = np.where(is_ev, r, np.nan)
            row = dict(key=k, t_on=t0, kind=e["kind"], dom=d["dom"],
                       n_frames=int(ok.sum()), span_stats=float(tt[ok].max()) if ok.any() else np.nan)
            for tau_v in TAU_GRID:
                j = int(np.searchsorted(tt, tau_v))
                row[f"r_at_{tau_v}"] = (float(r_ev[j]) if (j < len(r_ev) and np.isfinite(r_ev[j]))
                                        else np.nan)
            rr = r_ev[np.isfinite(r_ev)]
            row["r_max"] = float(rr.max()) if len(rr) else np.nan
            row["r_p10"] = float(np.percentile(rr, 10)) if len(rr) else np.nan
            row["r_med"] = float(np.median(rr)) if len(rr) else np.nan
            row["inc_at_1s"] = (float(ic[int(np.searchsorted(tt, 1.0))])
                                if int(np.searchsorted(tt, 1.0)) < len(ic) else np.nan)
            rows.append(row)
            # 抽样逐帧（每 0.1 s）
            step = 10
            for j in range(0, len(tt), step):
                if np.isfinite(r_ev[j]):
                    frames.append(dict(key=k, t_on=t0, kind=e["kind"], tau=float(tt[j]),
                                       A_raw=float(a[j]), inc=float(ic[j]),
                                       ratio=float(r_ev[j])))
        rec(f"  probe {k} 完成")

    PE = pd.DataFrame(rows)
    PE.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_trigger_perevent.csv"),
              index=False, encoding="utf-8-sig", float_format="%.6g")
    FR = pd.DataFrame(frames)
    FR.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_trigger_raw.csv"),
              index=False, encoding="utf-8-sig", float_format="%.6g")
    rec(f"产出 t5a_kappa_trigger_perevent.csv（{len(PE)} 行）、"
        f"t5a_kappa_trigger_raw.csv（{len(FR)} 帧，每 0.1 s 抽样）")

    # ── r(τ) 分位 ──
    lines = ["\n===== r(τ) = A_raw(τ) / inc(τ)：未封顶反演值相对实测增量的比值 =====",
             "（r ≤ κ ⇒ 该帧 κ 不起作用；r > κ ⇒ 该帧被上限截断。r ≡ 1 表示原型自带的 max(A, inc) 下限在起作用）"]
    for kind in ["onset", "restep"]:
        s = PE[PE.kind == kind]
        if not len(s):
            continue
        lines.append(f"\n--- {kind}（n={len(s)}）---")
        for tau_v in TAU_GRID:
            v = s[f"r_at_{tau_v}"].dropna()
            if not len(v):
                continue
            lines.append(f"  τ={tau_v:.2f} s: n={len(v):2d}  中位 {v.median():7.4f}  "
                         f"p10 {v.quantile(.1):7.4f}  p90 {v.quantile(.9):7.4f}  "
                         f"max {v.max():8.4f}")
        lines.append(f"  全事件 r_max: 中位 {s.r_max.median():7.4f}  "
                     f"p10 {s.r_max.quantile(.1):7.4f}  p90 {s.r_max.quantile(.9):7.4f}  "
                     f"max {s.r_max.max():8.4f}")
    txt = "\n".join(lines)
    rec(txt)

    # ── 触发率表（直接判据：该事件轨迹里是否真的出现 r > κ）──
    rows2 = []
    for kind, grid in [("onset", GRID_ONSET), ("restep", GRID_RESTEP)]:
        s = PE[PE.kind == kind]
        g = s.set_index(["key", "t_on"])["r_max"].dropna()
        n = len(g)
        for kap in grid:
            trig = int((g > kap).sum())
            # 被截断帧数（用逐帧 r 轨迹里 r > κ 的帧数，抽样 ×10 还原）
            fr = FR[FR.kind == kind]
            gfr = fr.groupby(["key", "t_on"])["ratio"]
            n_frames_trunc = int(sum(int((v > kap).sum()) * 10 for _, v in gfr))
            n_frames_tot = int(len(fr) * 10)
            rows2.append(dict(kind=kind, kappa=kap, n_events=n,
                              n_trigger=trig,
                              trigger_rate=trig / n if n else np.nan,
                              trunc_frame_frac=(n_frames_trunc / n_frames_tot
                                                if n_frames_tot else np.nan),
                              r_max_med=g.median(), r_max_p10=g.quantile(.10)))
    TR = pd.DataFrame(rows2)
    TR.to_csv(os.path.join(C.TASK, "results", "t5a_kappa_trigger.csv"),
              index=False, encoding="utf-8-sig", float_format="%.6g")

    lines2 = ["\n===== κ × 触发率（直接判据：事件轨迹里出现过 r > κ 即算触发）=====",
              "（`trunc_frame_frac` = 被截断帧占全部事件帧的比例，由 0.1 s 抽样 ×10 还原）",
              f"{'类':7s} {'κ':>5s} {'n事件':>5s} {'触发数':>6s} {'触发率':>8s} "
              f"{'截断帧占比':>11s} {'r_max 中位':>11s} {'r_max p10':>10s}"]
    for _, r in TR.iterrows():
        lines2.append(f"{r['kind']:7s} {r.kappa:5.2f} {int(r.n_events):5d} "
                      f"{int(r.n_trigger):6d} {r.trigger_rate*100:7.1f}% "
                      f"{r.trunc_frame_frac*100:10.1f}% {r.r_max_med:11.4f} "
                      f"{r.r_max_p10:10.4f}")
    txt2 = "\n".join(lines2)
    rec(txt2)

    lines3 = ["\n===== 结论：触发阈值在哪里 ====="]
    for kind in ["onset", "restep"]:
        s = PE[PE.kind == kind]
        if not len(s):
            continue
        g = s["r_max"].dropna()
        lines3.append(
            f"{kind:7s} n={len(g):3d}  r_max: min {g.min():.4f} / p10 {g.quantile(.1):.4f} / "
            f"中位 {g.median():.4f} / p90 {g.quantile(.9):.4f} / max {g.max():.4f}"
            f"  ⇒ κ ≥ {g.max():.4f} 时 **0% 触发**（κ 完全失效）；"
            f"κ = {g.median():.4f} 时约 50% 触发")
    txt3 = "\n".join(lines3)
    rec(txt3)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_kappa_trigger.log"),
                "python scripts/t5a_kappa_trigger.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
