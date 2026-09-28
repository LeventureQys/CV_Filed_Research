# -*- coding: utf-8 -*-
"""t1a_00_smoke：开工自检（按 00_共享/数据与脚本复用清单.md §5）。

检查项（任一失败立即停下排查，不得继续）：
  1. 依赖版本；
  2. 13 份录制可读 + 帧数与复用清单一致 + 通道数/域 + 包周期；
  3. 三条实现（v5.1/v6/v6.1）可实例化并跑通前 3000 帧；
  4. 本任务 100 Hz 网格与 `T4-A/t4a_morphology.csv` 的 `k_on/t_on` 一致（复用其 60 事件的**前提**）；
  5. 主通道口径的 `J_ch`（我算）与第一轮 `settle_arms.csv` 的 `step`（单通道口径）对拍；
  6. 稳定时间核 `t_stable` 与第一轮 `r4_filter_tradeoff.stable_time` 逐例等价（合成信号 200 例）。

产出：results/t1a_smoke_checks.csv、results/_t1a_00_smoke.log
"""
import os
import sys

import numpy as np
import pandas as pd

import t1a_common as C

T4A = os.path.join(C.PLAN, "T4_两种快相形态与分支判据", "results", "t4a_morphology.csv")
R1 = os.path.join(C.PROG, "13-v6-assessment", "results", "settle_arms.csv")


def r4_stable_time(tu, Y, i0, step, hold_s=30.0, tol_frac=0.05):
    """逐字复制 13-v6-assessment/r4_filter_tradeoff.py::stable_time（对拍用）。"""
    n = len(Y)
    H = int(hold_s / (tu[1] - tu[0]))
    tol = tol_frac * step
    for k in range(i0, n):
        e = min(n, k + H)
        if e - k < min(H, n - i0):
            break
        if np.max(np.abs(Y[k:e] - Y[k])) <= tol:
            return float(tu[k] - tu[i0])
    return np.nan


def main():
    C.start_log("00_smoke")
    rows = []
    print("\n== 1. 依赖 ==")
    print("python %s | numpy %s | pandas %s" % (sys.version.split()[0], np.__version__, pd.__version__))
    import scipy
    import matplotlib
    print("scipy %s | matplotlib %s" % (scipy.__version__, matplotlib.__version__))

    print("\n== 2. 13 份录制（100 Hz 网格，timestamp 列） ==")
    meta = {}
    for r in C.RECS:
        ok = os.path.isfile(r["path"])
        if not ok:
            print("!! 缺文件: %s" % r["path"])
            rows.append(dict(check="file_exists", key=r["rec"], ok=False, detail=r["path"]))
            continue
        tu, Xu, m = C.load_grid(r)
        exp = C.N_FRAMES[r["rec"]] - 1        # 复用清单按 (总行数−24) 计数，比真实帧数多 1（含列名行）
        good = (m["n_raw"] == exp)
        meta[r["rec"]] = m
        print("  %-22s n_raw=%6d (期望 %6d %s) n_ch=%2d span=%7.2f s pkt_dt=%5.3f s dup=%d 域=%s"
              % (r["rec"], m["n_raw"], exp, "OK" if good else "!! 不一致", m["n_ch"], m["span"],
                 m["pkt_p90"], m["dup"], m["dom"]))
        rows.append(dict(check="frames", key=r["rec"], ok=bool(good),
                         detail="n_raw=%d exp=%d n_ch=%d pkt_dt=%.4f" % (m["n_raw"], exp, m["n_ch"],
                                                                        m["pkt_p90"])))

    print("\n== 3. 三实现冒烟（前 3000 帧） ==")
    r0 = C.REC_BY_KEY["左拇指指尖/数据2"]
    tu, Xu, _ = C.load_grid(r0)
    n0 = min(3000, len(tu))
    for arm in ("v5.1", "v6", "v6.1"):
        try:
            out = C.run_arm("左拇指指尖/数据2", arm, tu[:n0], Xu[:n0])
            fin = bool(np.isfinite(out["Y"]).all())
            print("  %-5s OK  输出 shape=%s 有限=%s epoch=%d" % (arm, out["Y"].shape, fin, len(out["epoch"])))
            rows.append(dict(check="algo_smoke", key=arm, ok=bool(fin),
                             detail="epoch=%d" % len(out["epoch"])))
        except Exception as e:                                                   # noqa: BLE001
            print("  %-5s 失败: %r" % (arm, e))
            rows.append(dict(check="algo_smoke", key=arm, ok=False, detail=repr(e)))

    print("\n== 4. 与 T4-A 事件表的网格一致性（复用 60 事件的前提） ==")
    t4a = pd.read_csv(T4A, encoding="utf-8-sig")
    bad = 0
    for _, e in t4a.iterrows():
        k = int(e["k_on"])
        d_t = abs(float(e["t_on"]) - k * C.DT)
        if d_t > 0.005 or k >= meta[e["rec"]]["n_raw"] / 1.0:
            bad += 1
            print("  !! %s t_on=%.2f k_on=%d dt=%.4f" % (e["rec"], e["t_on"], k, d_t))
    print("  60 事件中网格不一致: %d 个（要求 0）" % bad)
    rows.append(dict(check="grid_vs_t4a", key="t4a_morphology", ok=bool(bad == 0),
                     detail="n=%d bad=%d" % (len(t4a), bad)))

    print("\n== 5. J_ch（我算）vs 第一轮 settle_arms.csv step（单通道口径） ==")
    r1 = pd.read_csv(R1, encoding="utf-8-sig")
    r1 = r1[r1.arm == "raw"]
    jr = []
    for _, e in r1.iterrows():
        rec = e["rec"].replace("中途切换-13ffca", "中途切换-13ffca")
        key = {"右拇指指尖/数据1": "右拇指指尖/数据1"}.get(rec, rec)
        if key not in C.REC_BY_KEY:
            continue
        tu, Xu, m = C.load_grid(key)
        y = Xu[:, m["ch"]]
        # 用 T4-A 的 onset 事件（恒载族取最早 onset、实录取最大 onset）定位主通道台阶
        sub = t4a[(t4a.rec == key) & (t4a.kind == "onset")]
        if not len(sub):
            continue
        if m["dom"] == "ADC域":
            sub = sub.loc[sub["jump"].idxmax():sub["jump"].idxmax()]
        else:
            sub = sub.loc[sub["t_on"].idxmin():sub["t_on"].idxmin()]
        k0 = int(sub["k_on"].iloc[0])
        J, pre, post = C.amp_5s(y, k0)
        d = J - float(e["step"])
        jr.append((key, J, float(e["step"]), d, k0, int(sub["t_on"].iloc[0] * 100)))
        flag = "OK" if abs(d) / max(abs(float(e["step"])), 1e-9) < 0.05 else "!! 差>5%"
        print("  %-22s J_ch=%9.4f  第一轮 step=%9.4f  Δ=%+8.4f  %s" % (key, J, e["step"], d, flag))
    if jr:
        dd = np.array([x[3] for x in jr]) / np.array([abs(x[2]) for x in jr])
        print("  相对差 中位 %.3f%%  最大 %.3f%%（%d 份）" % (100 * np.median(np.abs(dd)),
                                                             100 * np.max(np.abs(dd)), len(jr)))
        rows.append(dict(check="J_ch_vs_round1_step", key="settle_arms.step",
                         ok=bool(np.max(np.abs(dd)) < 0.05),
                         detail="n=%d max_rel=%.4f" % (len(jr), float(np.max(np.abs(dd))))))

    print("\n== 6. 稳定时间核 vs 第一轮 r4_filter_tradeoff.stable_time（合成信号 200 例） ==")
    rng = np.random.default_rng(20260918)
    same, diff = 0, 0
    worst = 0.0
    for i in range(200):
        n = 4000
        tu = np.arange(n) * C.DT
        y = np.cumsum(rng.normal(0, 0.02, n)) + 5.0
        k0 = 500
        y[k0:] += 10.0                                     # 阶跃
        y[k0 + 300:] += 0.3                                # 慢相
        if i % 3 == 0:
            y[k0 + 1200:k0 + 1400] += 1.5                  # 中途扰动
        j, _, _ = C.amp_5s(y, k0)
        a, _ = C.t_stable(tu, y, k0, j)
        b = r4_stable_time(tu, y, k0, j)
        if (np.isnan(a) and np.isnan(b)) or (np.isfinite(a) and np.isfinite(b) and abs(a - b) < 1e-9):
            same += 1
        else:
            diff += 1
            worst = max(worst, abs(a - b) if np.isfinite(a) and np.isfinite(b) else np.inf)
    print("  完全一致 %d/200，不一致 %d（最大差 %.6f s）" % (same, diff, worst))
    rows.append(dict(check="kernel_vs_r4_stable_time", key="t_stable",
                     ok=bool(diff == 0), detail="same=%d diff=%d worst=%.6f" % (same, diff, worst)))

    print("\n== 7. 主通道选择复核（该录制加载段内台阶最大的通道） ==")
    for r in C.RECS:
        key = r["rec"]
        sub = t4a[(t4a.rec == key) & (t4a.kind == "onset")]
        if not len(sub):
            continue
        sub = sub.loc[sub["jump"].idxmax() if r["dom"] == "ADC域" else sub["t_on"].idxmin()]
        tu, Xu, m = C.load_grid(key)
        k0 = int(sub["k_on"])
        pre = np.median(Xu[max(0, k0 - 200):k0], axis=0)
        post = np.median(Xu[k0 + 400:k0 + 600], axis=0)
        ch_peak = int(np.argmax(post - pre))
        print("  %-22s 指定主通道=%2d  该事件台阶最大通道=%2d  %s"
              % (key, m["ch"], ch_peak, "一致" if ch_peak == m["ch"] else "**不一致**"))
        rows.append(dict(check="main_channel", key=key, ok=bool(ch_peak == m["ch"]),
                         detail="fix=%d peak=%d" % (m["ch"], ch_peak)))

    df = pd.DataFrame(rows)
    out = os.path.join(C.RES, "t1a_smoke_checks.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    nbad = int((~df["ok"]).sum())
    print("\n== 自检汇总：%d 项，失败 %d 项 ==" % (len(df), nbad))
    print("-> %s" % out)
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
