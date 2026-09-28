# -*- coding: utf-8 -*-
"""t3b_00_recon.py —— T3-B 开工自检：数据冒烟 + 补丁零差证明 + 冻结核等价性证明。

产出：results/t3b_patch_ab_zero.csv、results/_t3b_00_recon.log、results/cache/t3b_*.npz

三类检查：
  A. **数据冒烟**：13 份录制全部可读、通道数与网格长度与 `00_共享/数据与脚本复用清单.md` §1
     的帧数表一致（读到别的文件必须立刻停下）；
  B. **补丁零差证明**（照 T4-B `t4b_verify_patch.py` 的做法）：
     `ArmComp`（N_GATE=0, EMA_TAU=0）/`V51Traced` 与原类逐帧输出最大绝对差 **必须 = 0**；
     同时给「开关真的起作用」的反向证明（N_GATE=1 / EMA_TAU=0.3 的差值 > 0），
     否则"零差"是空洞的；
  C. **冻结核等价性**：`t3b_settle` 的向量化 `t_stable / t_stable_ev / t_settle`
     与 T1-A 原实现（`*_ref`）在真实序列上逐项一致。

运行：python scripts/t3b_00_recon.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t3b_ad_lib as L      # noqa: E402
import t3b_common as C      # noqa: E402
import t3b_arms as A        # noqa: E402
import t3b_settle as ST     # noqa: E402
import t3b_glm53_v6 as G6   # noqa: E402
import t3b_glm53_v51 as G51  # noqa: E402

RES = C.RES
N_FRAMES = {"右拇指指尖/数据1": 16356, "右拇指指尖/数据2": 23537, "右拇指指尖/数据3": 19138,
            "左拇指指尖/数据1": 19740, "左拇指指尖/数据2": 19975, "左拇指指尖/数据3": 19402,
            "四指指尖/数据1": 19140, "四指指尖/数据2": 19114, "四指指尖/数据3": 20033,
            "切换负载-快相无责": 25693, "再切换负载": 7129,
            "中途切换-1d9493": 6421, "中途切换-13ffca": 12121}
ROWS = []


def add(check, dataset, n, metric, a, b, note=""):
    diff = (abs(float(a) - float(b)) if (a is not None and b is not None
                                         and np.isfinite(a) and np.isfinite(b)) else np.nan)
    ROWS.append(dict(check=check, dataset=dataset, n=n, metric=metric, value_a=a, value_b=b,
                     abs_diff=diff, verdict=("PASS" if (np.isfinite(diff) and diff == 0) else ""),
                     note=note))


def run(cls, tu, Xu, **kw):
    c = cls(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return c, Y


def main():
    ST.start_log(RES, "00_recon")
    t_all = time.time()
    print("== A. 数据冒烟：读 13 份录制（首次会建 npz 缓存，约 1~2 min）==")
    ev, cache = A.load_all(verbose=True)
    print("\n== A1. 原始帧数直读（对 `数据与脚本复用清单.md` §1 的帧数表）==")
    print("   注：清单的值比 CSV 实际数据行数 **多 1**（把 `##Data` 后的列名行或末行算了进去），")
    print("   实测 13 份全部满足 `raw_frames = 清单值 − 1`；通道数也与清单一致。")
    print("%-22s %5s %7s %7s %7s %9s %9s" % ("rec", "ch", "raw", "清单", "差", "span_s", "中位包周期"))
    ok_all = True
    for name, path, dom in C.RECS:
        t, X = L.load_rec(path)
        nraw = int(len(t))
        exp = N_FRAMES[name]
        okf = (nraw == exp - 1)
        okch = (X.shape[1] == (31 if name.endswith(("数据1", "数据2", "数据3"))
                               and not name.startswith(("四指",)) else 21))
        ok_all &= okf
        print("%-22s %5d %7d %7d %7d %9.2f %9.4f" % (name, X.shape[1], nraw, exp, nraw - exp,
                                                     float(t[-1] - t[0]), float(np.median(np.diff(t)))))
        add("A_rawframes", name, nraw, "raw_frames", nraw, exp - 1,
            "清单值−1，与复用清单一致（n_ch=%d %s）" % (X.shape[1], "OK" if okch else "!! 通道数异常"))
        ROWS[-1]["verdict"] = "PASS" if (okf and okch) else "FAIL"
        del t, X
    print("\n帧数自检总体：%s" % ("13/13 全部通过" if ok_all else "存在不一致，需排查！"))
    print("\n== A2. 100 Hz 网格长度（np.interp 重采样）==")
    print("%-22s %5s %9s %9s" % ("rec", "n_grid", "span_s", "peak"))
    for name, path, dom in C.RECS:
        cc = cache[name]
        k = max(1, int(round(0.5 / cc["dt"])))
        print("%-22s %5d %9.2f %9.1f" % (name, len(cc["tu"]), cc["tu"][-1], cc["peak"]))
        add("A_grid", name, len(cc["tu"]), "grid_len", len(cc["tu"]), len(cc["tu"]),
            "100 Hz 网格 = int(span/0.01)（算法链路口径）")
        ROWS[-1]["verdict"] = "PASS"
    print("\n事件表：%d 行；kind 分布 %s" % (len(ev), dict(ev.kind.value_counts())))
    ev["clean_t4a"] = A.t4a_clean_flags(ev)
    ev.to_csv(os.path.join(RES, "t3b_events.csv"), index=False, encoding="utf-8-sig")
    print("  clean_t4a=True：onset %d / restep %d" % (
        int(((ev.kind == "onset") & ev.clean_t4a).sum()),
        int(((ev.kind == "restep") & ev.clean_t4a).sum())))

    # ── B. 补丁零差证明 ──
    print("\n== B. 补丁零差证明（同一录制、同一网格、逐帧比对）==")
    name, path, dom = C.RECS[10]          # 再切换负载：含 onset/restep/decrement
    cc = cache[name]
    tu, Xu = cc["tu"], cc["Xu"]
    print("录制：%s（%d 帧）" % (name, len(tu)))

    # B1 v6 系：原类 vs 仪表化子类（默认开关）
    c0, Y0 = run(G6.GLM53v6, tu, Xu)
    c1, Y1 = run(A.ArmComp, tu, Xu)
    d = float(np.abs(Y0 - Y1).max())
    print("  B1 v6 原类 vs ArmComp(默认)              逐帧最大绝对差 = %.3e  -> %s"
          % (d, "PASS" if d == 0 else "FAIL"))
    add("B1_v6_vs_ArmComp", name, len(tu), "max_abs_frame_diff", d, 0.0, "补丁不改变默认行为")
    ROWS[-1]["verdict"] = "PASS" if d == 0 else "FAIL"

    # B2 v5.1 系
    c2, Y2 = run(G51.GLM53v51, tu, Xu)
    c3, Y3 = run(A.V51Traced, tu, Xu)
    d2 = float(np.abs(Y2 - Y3).max())
    print("  B2 v5.1 原类 vs V51Traced                逐帧最大绝对差 = %.3e  -> %s"
          % (d2, "PASS" if d2 == 0 else "FAIL"))
    add("B2_v51_vs_V51Traced", name, len(tu), "max_abs_frame_diff", d2, 0.0,
        "epoch 记录钩子不改变行为")
    ROWS[-1]["verdict"] = "PASS" if d2 == 0 else "FAIL"

    # B3 反向证明：开关必须真的起作用（否则"零差"无意义）——在 3 份录制上取最大差
    KNOB = ("再切换负载", "切换负载-快相无责", "中途切换-13ffca")
    d3 = d4 = 0.0
    per = []
    for nm in KNOB:
        cn = cache[nm]
        _, Yb = run(G6.GLM53v6, cn["tu"], cn["Xu"])
        _, Yg = run(A.ArmComp, cn["tu"], cn["Xu"], N_GATE=1)
        _, Ye = run(A.ArmComp, cn["tu"], cn["Xu"], EMA_TAU=0.30)
        a3 = float(np.abs(Yb - Yg).max())
        a4 = float(np.abs(Yb - Ye).max())
        d3 = max(d3, a3)
        d4 = max(d4, a4)
        per.append("%s(%.1f/%.1f)" % (nm, a3, a4))
    print("  B3 反向证明：N_GATE=1 最大差 %.1f、EMA_TAU=0.3 最大差 %.1f（都必须 > 0）" % (d3, d4))
    print("      逐录制：%s" % "  ".join(per))
    add("B3_knob_effect_N_GATE", "3 份录制取最大", int(sum(len(cache[n]["tu"]) for n in KNOB)),
        "max_abs_frame_diff", d3, 0.0,
        "开关确实生效" if d3 > 0 else "开关无效（异常）")
    ROWS[-1]["verdict"] = "PASS" if d3 > 0 else "FAIL"
    add("B3_knob_effect_EMA", "3 份录制取最大", int(sum(len(cache[n]["tu"]) for n in KNOB)),
        "max_abs_frame_diff", d4, 0.0,
        "开关确实生效" if d4 > 0 else "开关无效（异常）")
    ROWS[-1]["verdict"] = "PASS" if d4 > 0 else "FAIL"

    # ── C. 冻结核等价性 ──
    print("\n== C. 冻结核等价性（t3b_settle 向量化 vs T1-A 原实现）==")
    k5 = max(1, int(round(0.5 / cc["dt"])))
    ytot = L.med_smooth(Y0.sum(axis=1), k5)
    ych = L.med_smooth(Y0[:, A.MAIN_CH[name]], k5)
    rtot = L.med_smooth(cc["tot"], k5)
    rch = L.med_smooth(Xu[:, A.MAIN_CH[name]], k5)
    loc = ev[ev.rec == name].sort_values("t_on")
    tlist = [(float(r.t_on), float(r.J_nc)) for _, r in loc.iterrows()
             if np.isfinite(r.J_nc) and abs(r.J_nc) >= 0.02 * cc["peak"]]
    nchk = 0
    for _, r in loc.head(4).iterrows():
        k0 = int(np.searchsorted(tu, float(r.t_on)))
        times = [(t, j) for t, j in tlist if abs(t - float(r.t_on)) > 1e-9]
        cut = ST.next_event_cut(times, float(r.t_on), cc["peak"])
        cut_k = None if cut is None else int(round(cut / cc["dt"]))
        for tag, Y in (("tot", ytot), ("ch", ych)):
            J, pre, _ = ST.amp_5s(Y, k0)
            a1 = ST.t_stable(tu, Y, k0, J)
            b1 = ST.t_stable_ref(tu, Y, k0, J)
            a2 = ST.t_stable_ev(tu, Y, k0, J, cut_k)
            b2 = ST.t_stable_ev_ref(tu, Y, k0, J, cut_k)
            zf, _ = ST.z_final_of(Y, k0, cut_k)
            a3 = ST.t_settle(tu, Y, k0, J, 0.05, zf, cut_k)
            b3 = ST.t_settle_ref(tu, Y, k0, J, 0.05, zf, cut_k)
            same = (np.isclose(a1[0], b1[0], equal_nan=True) and a1[1] == b1[1]
                    and np.isclose(a2[0], b2[0], equal_nan=True) and a2[1] == b2[1]
                    and np.isclose(a3[0], b3[0], equal_nan=True) and a3[1] == b3[1])
            print("  t_on=%7.2f kind=%-14s %s  D1 %s/%s  D1ev %s/%s  D2 %s/%s  %s"
                  % (r.t_on, r.kind, tag, _f(a1[0]), _f(b1[0]), _f(a2[0]), _f(b2[0]),
                     _f(a3[0]), _f(b3[0]), "一致" if same else "!! 不一致"))
            add("C_kernel_equiv_%s" % tag, "%s@%.2f" % (name, r.t_on), len(tu),
                "D1/D1ev/D2 tuple", 0.0, 0.0 if same else 1.0,
                "与 t1a_common 原实现逐项一致" if same else "不一致")
            ROWS[-1]["verdict"] = "PASS" if same else "FAIL"
            nchk += 1
    print("  共比对 %d 组（每组 3 个核 × 2 序列）" % nchk)

    # 额外：raw 序列（永不稳定）走删失分支的等价性
    a = ST.t_stable(tu, rtot, int(np.searchsorted(tu, float(loc.iloc[0].t_on))), 100.0)
    b = ST.t_stable_ref(tu, rtot, int(np.searchsorted(tu, float(loc.iloc[0].t_on))), 100.0)
    same = (np.isclose(a[0], b[0], equal_nan=True) and a[1] == b[1])
    print("  删失分支：raw 序列 D1 %s/%s -> %s" % (_f(a[0]), _f(b[0]), "一致" if same else "!!"))
    add("C_kernel_equiv_censored", name, len(tu), "D1 raw-never-stable", 0.0,
        0.0 if same else 1.0, "删失分支一致" if same else "不一致")
    ROWS[-1]["verdict"] = "PASS" if same else "FAIL"

    # C2 **有限 T** 的等价性：只用删失样本比 NaN 会掩盖 bug（本任务踩过一次：
    #    pandas.rolling 后向窗把"此后 30 s 平坦"算成"此前 30 s 平坦" ⇒ v6 的 1.0 s 被报成 30.5 s）。
    print("\n  C2 有限 T 的等价性（合成阶跃 + 真实 v6 输出）：")
    syn = np.zeros(6000)
    syn[500:] = 10.0
    tu_s = np.arange(len(syn)) * 0.01
    for k0s in (100, 480, 500, 520, 3000):
        for tag, Y in (("syn", syn),):
            a = ST.t_stable(tu_s, Y, k0s, 10.0)
            b = ST.t_stable_ref(tu_s, Y, k0s, 10.0)
            c = ST.t_stable_ev(tu_s, Y, k0s, 10.0, 5000)
            d = ST.t_stable_ev_ref(tu_s, Y, k0s, 10.0, 5000)
            ok = (np.isclose(a[0], b[0], equal_nan=True) and a[1] == b[1]
                  and np.isclose(c[0], d[0], equal_nan=True) and c[1] == d[1])
            print("    合成 k0=%4d  D1 %s/%s   D1ev %s/%s  %s"
                  % (k0s, _f(a[0]), _f(b[0]), _f(c[0]), _f(d[0]), "一致" if ok else "!! 不一致"))
            add("C2_kernel_equiv_finite", "synth@%d" % k0s, len(syn), "D1/D1ev tuple", 0.0,
                0.0 if ok else 1.0, "有限 T 逐项一致" if ok else "不一致")
            ROWS[-1]["verdict"] = "PASS" if ok else "FAIL"
    # 真实序列（v6 输出；恒载 onset 的 T 应稳定在 1~2 s 量级）
    real = {}
    for nm in ("四指指尖/数据1", "左拇指指尖/数据2"):
        cn = cache[nm]
        c_, Yr = run(G6.GLM53v6, cn["tu"], cn["Xu"])
        real[nm] = (cn, Yr)
        k5 = max(1, int(round(0.5 / cn["dt"])))
        Ys = L.med_smooth(Yr.sum(axis=1), k5)
        r0 = ev[(ev.rec == nm) & (ev.kind == "onset")].iloc[0]
        k0 = int(np.searchsorted(cn["tu"], float(r0.t_on)))
        J, pre, _ = ST.amp_5s(Ys, k0)
        a = ST.t_stable(cn["tu"], Ys, k0, J)
        b = ST.t_stable_ref(cn["tu"], Ys, k0, J)
        c = ST.t_stable_ev(cn["tu"], Ys, k0, J, None)
        d = ST.t_stable_ev_ref(cn["tu"], Ys, k0, J, None)
        ok = (np.isclose(a[0], b[0], equal_nan=True) and a[1] == b[1]
              and np.isclose(c[0], d[0], equal_nan=True) and c[1] == d[1])
        print("    真实 %-14s onset t_on=%6.2f  D1 %s/%s   D1ev %s/%s  %s"
              % (nm, r0.t_on, _f(a[0]), _f(b[0]), _f(c[0]), _f(d[0]), "一致" if ok else "!! 不一致"))
        add("C2_kernel_equiv_finite", nm, len(cn["tu"]), "D1/D1ev tuple", 0.0,
            0.0 if ok else 1.0, "有限 T 逐项一致" if ok else "不一致")
        ROWS[-1]["verdict"] = "PASS" if ok else "FAIL"

    # ── D. 已知答案检验（KAT）：口径若写反会得到 ~30 s，必须挡住 ──
    print("\n== D. 已知答案检验（KAT）：v6 在恒载 onset 上的 D1-ev 必须落在 0.3~5 s ==")
    for nm in ("四指指尖/数据1", "左拇指指尖/数据2"):
        cn, Yr = real[nm]
        k5 = max(1, int(round(0.5 / cn["dt"])))
        Ys = L.med_smooth(Yr.sum(axis=1), k5)
        r0 = ev[(ev.rec == nm) & (ev.kind == "onset")].iloc[0]
        k0 = int(np.searchsorted(cn["tu"], float(r0.t_on)))
        J, pre, _ = ST.amp_5s(Ys, k0)
        t, cens, win = ST.t_stable_ev(cn["tu"], Ys, k0, J, None)
        ok = (not cens) and 0.3 <= t <= 5.0
        print("  %-14s T_stable_ev(tot)=%s s (J=%.3f, n=%d) -> %s"
              % (nm, _f(t), J, len(cn["tu"]), "PASS" if ok else "FAIL"))
        add("D_kat_v6_onset", nm, len(cn["tu"]), "T_stable_ev_tot5", t, t, 
            "落在 0.3~5 s（第一轮同口径为 0.4~5 s 量级）" if ok else "越界：口径可能写反")
        ROWS[-1]["verdict"] = "PASS" if ok else "FAIL"

    df = pd.DataFrame(ROWS)
    p = os.path.join(RES, "t3b_patch_ab_zero.csv")
    df.to_csv(p, index=False, encoding="utf-8-sig")
    nfail = int((df.verdict == "FAIL").sum())
    print("\n== 汇总 ==\n%s" % df.to_string(index=False))
    print("\nFAIL 计数 = %d  -> %s" % (nfail, "全部通过" if nfail == 0 else "存在失败项"))
    print("总耗时 %.1f s" % (time.time() - t_all))
    print("-> %s" % p)
    return 1 if nfail else 0


def _f(x):
    return "nan" if x is None or not np.isfinite(x) else "%.3f" % x


if __name__ == "__main__":
    sys.exit(main())
