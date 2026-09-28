# -*- coding: utf-8 -*-
"""T5-A 指标实现自检：`T_stable` 的滑窗最大值实现 vs 逐点扫描（必须逐个相同）。

顺带自检：
  · `z_final_of` 的两种模式（t0+60 s / 后 1/3 段）在 13 份录制上各触发几次；
  · `OS%` 的符号约定（向上过充为正、下冲 US% 为正）；
  · `metrics_at` 在事件贴近段末时不越界（此前踩过的 IndexError 回归测试）。

产物：`results/t5a_metric_selfcheck.csv`、`_t5a_metric_selfcheck.log`。

用法：`python scripts/t5a_metric_selfcheck.py`
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    HERE_PARENT = None
    sys.path.insert(0, HERE)

import t5a_common as C                                  # noqa: E402

LOG = []


def rec(m):
    print(m, flush=True)
    LOG.append(m)


def t_stable_scan(seg, tu, t0, nwin, nmin, thr):
    """逐点扫描版（参考实现）：**穷举全部完整窗**（不设搜索上界），
    只接受 `τ + win ≤ 段末`，放不下则 NaN。用来验证 `metrics_at._tstab_scan` 的**有界搜索**
    与**穷举**给出同一答案（有界性由「更晚的起点要求其完整窗也安静」保证）。"""
    n = len(seg)
    i = min(int(np.searchsorted(tu, t0)), n - 1)
    while i + nwin <= n:
        if seg[i:i + nwin].max() <= thr:
            return float(tu[i] - t0)
        i += 1
    return np.nan


def main():
    ev = C.ev_load_frozen()
    recs = C.recordings()
    rows = []
    n_agree = n_tot = 0
    zf_modes = {}
    for k, d in recs.items():
        sub = ev[ev.key == k]
        if not len(sub):
            continue
        out = C.run_v6(d, kappa_onset=1.30, kappa_restep=1.12)
        Ys = out["Y"].sum(axis=1)
        Zs = d["Z"]
        for _, e in sub.iterrows():
            t0 = float(e["t_on"])
            m = C.metrics_at(Zs, Ys, d["tu"], t0, d["span"])
            pre, post, J = C.j_and_pre(Zs, d["tu"], t0, d["span"])
            Zf, mode = C.z_final_of(Zs, d["tu"], t0, d["span"])
            zf_modes[mode] = zf_modes.get(mode, 0) + 1
            seg = np.abs(Ys - Zf)
            dt = float(d["tu"][1] - d["tu"][0])
            nwin = int(round(30.0 / dt))
            nmin = 0
            ref = t_stable_scan(seg, d["tu"], t0, nwin, nmin, 0.05 * abs(J))
            ref10 = t_stable_scan(seg, d["tu"], t0, int(round(10.0 / dt)), 0, 0.05 * abs(J))
            got = m["T_stable"]
            got10 = m["T_stable10"]
            ok = ((np.isnan(ref) and np.isnan(got)) or (
                not np.isnan(ref) and not np.isnan(got) and abs(ref - got) < 1e-9)) and (
                (np.isnan(ref10) and np.isnan(got10)) or (
                    not np.isnan(ref10) and not np.isnan(got10)
                    and abs(ref10 - got10) < 1e-9))
            n_tot += 1
            n_agree += int(ok)
            rows.append(dict(key=k, t_on=t0, kind=e["kind"], T_stable_impl=got,
                             T_stable_scan=ref, T_stable10_impl=got10,
                             T_stable10_scan=ref10,
                             agree=bool(ok), zf_mode=mode,
                             Z_final=Zf, J=J, OS_pct=m["OS_pct"], US_pct=m["US_pct"],
                             MD=m["MD"], G20=m["G20"], err_1s_pct=m["err_1s_pct"]))
    df = pd.DataFrame(rows)
    p = os.path.join(C.TASK, "results", "t5a_metric_selfcheck.csv")
    df.to_csv(p, index=False, encoding="utf-8-sig", float_format="%.6g")
    rec(f"T_stable(30 s 窗) 有界搜索 vs 穷举扫描：一致 {n_agree}/{n_tot}")
    rec(f"Z_final 模式分布：{zf_modes}")
    rec(f"OS% 极值：max {df.OS_pct.max():.3f} / min {df.OS_pct.min():.3f}；"
        f"US% max {df.US_pct.max():.3f}")
    rec(f"G20 范围：{df.G20.min():.3f} ~ {df.G20.max():.3f}")
    rec(f"T_stable(30 s) 非空 {df.T_stable_impl.notna().sum()}/{len(df)}；"
        f"T_stable(10 s) 非空 {df.T_stable10_impl.notna().sum()}/{len(df)}")
    rec(f"产出 {p}")
    C.write_log(os.path.join(C.TASK, "results", "_t5a_metric_selfcheck.log"),
                "python scripts/t5a_metric_selfcheck.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
