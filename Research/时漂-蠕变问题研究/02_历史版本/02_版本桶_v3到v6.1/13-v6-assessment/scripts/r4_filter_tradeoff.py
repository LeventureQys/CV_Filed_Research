# -*- coding: utf-8 -*-
"""r4：① 稳定时间目标（1~2 s）能不能达成；② 对 v6 输出做（因果）低通滤波能否治过充。

做法：在真实 onset 上跑三臂 —— raw / v5.1(无责3s) / v6 —— 再对 v6 的输出串上不同 τ 的因果 EMA，
测「首次稳定时刻 T_stable」与「加载后的最大正误差（过充）」两个指标，得到权衡曲线。
T_stable 口径与论文（11-paper-v6）一致：显示首次"停下"（此后 hold 30 s 内相对该时刻自身漂移 ≤5%×阶跃）。
产出 results/settle_arms.csv、results/filter_pareto.csv、figures/r4_filter_pareto.png
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
FLASH = os.path.dirname(os.path.dirname(OUT))
TEMP = os.path.dirname(FLASH)
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.join(FLASH, "progress", "04-v5", "scripts"))
import ad_lib as L  # noqa: E402
sys.path.insert(0, os.path.join(FLASH, "progress", "05-v5.1", "scripts"))
from glm53_v51 import GLM53v51  # noqa: E402
sys.path.insert(0, os.path.join(FLASH, "progress", "07-v6", "scripts"))
from glm53_v6 import GLM53v6  # noqa: E402

B = os.path.join(TEMP, "变化负载")
CASES = [(f"{loc}/数据{i}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"),
          {"右拇指指尖": 17, "左拇指指尖": 18, "四指指尖": 11}[loc])
         for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]
CH = {"切换负载-快相无责": 4, "再切换负载": 15, "中途切换-1d9493": 6, "中途切换-13ffca": 11}
CASES += [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                            "20260917_133923_single_device_ee20bc", "device_001_seg000.csv"), 4),
          ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv"), 15),
          ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv"), 6),
          ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                           "最终测试目标", "device_001_seg000.csv"), 11)]
TAUS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0]


def unload_index(y, k, dt):
    """加载沿之后的最大负跳变（卸载沿）；用于把「蠕变含在内的总幅度」定出来。"""
    w = max(1, int(0.10 / dt))
    j0, j1 = k + int(3.0 / dt), len(y) - 1
    d = y[j0 + w:j1] - y[j0:j1 - w]
    if len(d) == 0:
        return None
    return j0 + int(np.argmin(d))


def stable_time(tu, Y, i0, step, hold_s=30.0, tol_frac=0.05):
    """显示首次「停下」：此后 hold_s 内相对该时刻自身的漂移 ≤ tol×阶跃。"""
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


def onset_index(y, dt):
    """最陡上升帧（真沿）。"""
    w = max(1, int(0.10 / dt))
    ys = L.med_smooth(y, w)
    d = np.zeros_like(ys)
    d[w:-w] = ys[2 * w:] - ys[:-2 * w]
    return int(np.argmax(d))


def main():
    rows, pareto = [], []
    for name, path, ch in CASES:
        if not os.path.isfile(path):
            print("!! 缺:", path)
            continue
        d = L.prep(path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
        y = Xu[:, ch]
        k = onset_index(y, dt)
        i0 = k
        pre = float(np.median(y[max(0, k - int(2.0 / dt)):k]))
        tgt = float(np.median(y[min(len(y) - 1, k + int(5.0 / dt)):
                                 min(len(y), k + int(7.0 / dt))]))
        step = tgt - pre
        if step <= 0:
            continue
        Y_raw = y
        Y_51, _ = L.run_algo(tu, Xu, GLM53v51)
        Y_v6, c6 = L.run_algo(tu, Xu, GLM53v6)
        arms = {"raw": y, "v5.1(免责3s)": Y_51[:, ch], "v6": Y_v6[:, ch]}
        tot_arms = {"raw": Xu.sum(axis=1), "v5.1(免责3s)": Y_51.sum(axis=1), "v6": Y_v6.sum(axis=1)}
        dom = "ADC域" if name in CH else "显示域"
        ju = unload_index(y, k, dt)
        step_end = (float(np.median(y[max(k, ju - int(5.0 / dt)):ju]) - pre)
                    if ju else np.nan)
        for a, Y in arms.items():
            tS = stable_time(tu[k - int(1.0 / dt):], Y[k - int(1.0 / dt):], int(1.0 / dt), step)
            tStot = stable_time(tu[k - int(1.0 / dt):], tot_arms[a][k - int(1.0 / dt):], int(1.0 / dt),
                                float(np.median(tot_arms[a][k + int(5.0 / dt):k + int(7.0 / dt)])
                                      - np.median(tot_arms[a][max(0, k - int(2.0 / dt)):k])))
            # 用「含蠕变的总幅度」当参考口径（更松的稳定判据）
            tS_end = (stable_time(tu[k - int(1.0 / dt):], Y[k - int(1.0 / dt):], int(1.0 / dt), step_end)
                      if step_end == step_end and step_end > 0 else np.nan)
            seg = Y[k:k + int(10.0 / dt)]
            over = float(np.max(seg - (pre + step))) / step * 100
            err1 = float(np.median(Y[k + int(0.9 / dt):k + int(1.1 / dt)]) - (pre + step)) / step * 100
            err2 = float(np.median(Y[k + int(1.9 / dt):k + int(2.1 / dt)]) - (pre + step)) / step * 100
            rows.append(dict(rec=name, dom=dom, arm=a, step=round(step, 2),
                             T_stable=round(tS, 2) if tS == tS else np.nan,
                             T_stable_tot=round(tStot, 2) if tStot == tStot else np.nan,
                             T_stable_含蠕变口径=round(tS_end, 2) if tS_end == tS_end else np.nan,
                             over_pct=round(over, 2), err_1s_pct=round(err1, 2),
                             err_2s_pct=round(err2, 2)))
        # v6 + 输出端因果 EMA
        for tau in TAUS:
            if tau == 0:
                Yf = Y_v6[:, ch].copy()
            else:
                a = dt / tau
                Yf = np.empty(len(tu))
                acc = Y_v6[0, ch]
                for i in range(len(tu)):
                    acc += a * (Y_v6[i, ch] - acc)
                    Yf[i] = acc
            tS = stable_time(tu[k - int(1.0 / dt):], Yf[k - int(1.0 / dt):], int(1.0 / dt), step)
            seg = Yf[k:k + int(10.0 / dt)]
            over = float(np.max(seg - (pre + step))) / step * 100
            pareto.append(dict(rec=name, tau=tau, T_stable=tS if tS == tS else np.nan,
                               over_pct=over,
                               err_1s_pct=float(np.median(
                                   Yf[k + int(0.9 / dt):k + int(1.1 / dt)]) - (pre + step)) / step * 100))
    r = pd.DataFrame(rows)
    p = pd.DataFrame(pareto)
    r.to_csv(os.path.join(RES, "settle_arms.csv"), index=False, encoding="utf-8-sig")
    p.to_csv(os.path.join(RES, "filter_pareto.csv"), index=False, encoding="utf-8-sig")

    print("== 各臂的稳定时间与精度（13 份 onset；T_stable 越小越好，over 为正=过充）==")
    print(r.groupby("arm").agg(T_stable_med=("T_stable", "median"), T_stable_p90=("T_stable", lambda s: s.quantile(.9)),
                               T_stable总通道=("T_stable_tot", "median"),
                               over_med=("over_pct", "median"), over_max=("over_pct", "max"),
                               err1_med=("err_1s_pct", "median"), err2_med=("err_2s_pct", "median")
                               ).round(2).to_string())
    print("\n== 按数据域拆（主通道 T_stable / 过充）==")
    print(r.groupby(["dom", "arm"]).agg(T_stable_med=("T_stable", "median"),
                                        over_med=("over_pct", "median"),
                                        over_max=("over_pct", "max"),
                                        T_stable总通道=("T_stable_tot", "median")).round(2).to_string())
    print("\n== v6 输出加因果 EMA 的权衡（τ=0 即原样）==")
    print(p.groupby("tau").agg(T_stable_med=("T_stable", "median"), T_stable_p90=("T_stable", lambda s: s.quantile(.9)),
                               over_med=("over_pct", "median"), over_max=("over_pct", "max"),
                               err1_med=("err_1s_pct", "median")).round(2).to_string())
    # 图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        g = p.groupby("tau").agg(t=("T_stable", "median"), o=("over_pct", "median"),
                                 om=("over_pct", "max"), e=("err_1s_pct", "median"))
        ax[0].plot(g.t, g.o, "o-", label="中位过充")
        ax[0].plot(g.t, g.om, "s--", label="最大过充")
        for tau, row in g.iterrows():
            ax[0].annotate("τ=%.1f" % tau, (row.t, row.o), fontsize=8)
        ax[0].axvspan(1.0, 2.0, color="#c8e6c9", alpha=.6, label="1~2 s 目标区")
        ax[0].set_xlabel("T_stable（s）"); ax[0].set_ylabel("加载后最大正误差（%×阶跃）")
        ax[0].set_title("v6 输出加低通：治过充要以稳定时间为代价"); ax[0].legend(fontsize=8)
        ax[1].plot(g.t, g.e, "o-")
        ax[1].axvspan(1.0, 2.0, color="#c8e6c9", alpha=.6)
        ax[1].axhline(0, color="k", lw=.6)
        ax[1].set_xlabel("T_stable（s）"); ax[1].set_ylabel("1 s 时刻误差（%×阶跃）")
        ax[1].set_title("1 s 时刻的显示误差")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "r4_filter_pareto.png"), dpi=120)
        print("\n-> figures/r4_filter_pareto.png")
    except Exception as e:                                                        # noqa: BLE001
        print("绘图失败:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
