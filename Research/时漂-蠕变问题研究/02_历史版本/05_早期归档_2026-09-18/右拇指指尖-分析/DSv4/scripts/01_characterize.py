# -*- coding: utf-8 -*-
"""01 数据特征分析：时漂/零漂/噪声/空间分布，并输出基础图件。

输出：
    DSv4/results/characterization_metrics.json
    DSv4/figures/fig01_overview.png
    DSv4/figures/fig02_drift_models.png
    DSv4/figures/fig03_zero_drift.png
    DSv4/figures/fig04_spatial_maps.png
    DSv4/figures/fig05_common_mode.png
"""
import os
import sys
import json

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))
from common import (BASE, DATASETS, FIG, RES, load_dataset, segment_total,
                    channels_to_map, active_positions, setup_mpl)

plt = setup_mpl()


def exp_model(t, a, tau, c):
    return a * (1.0 - np.exp(-t / tau)) + c


def log_model(t, a, b):
    return a * np.log1p(t) + b


def rec_model(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def fit_r2(y, yhat):
    ss = np.sum((y - yhat) ** 2)
    return 1.0 - ss / np.sum((y - np.mean(y)) ** 2)


def load_window(d, s0, dt0=0.5, dt1=2.0):
    """负载起始响应参考：onset 后 dt0~dt1 秒的中位数。"""
    fs = d["fs"]
    i0 = min(s0 + int(round(dt0 * fs)), len(d["t"]) - 1)
    i1 = min(s0 + int(round(dt1 * fs)), len(d["t"]))
    return i0, i1


def characterize():
    R = {}
    for name in DATASETS:
        d = load_dataset(name)
        t, X, Xk, total = d["t"], d["X"], d["Xk"], d["total"]
        s0, s1 = segment_total(total)
        fs = d["fs"]

        # 参考窗
        pre_ref = X[max(0, s0 - int(3 * fs)):s0]
        iw0, iw1 = load_window(d, s0)
        post_imm_ref = X[s1 + int(2 * fs):s1 + int(5 * fs)]
        post_steady_ref = X[-int(3 * fs):]

        pre_med = np.median(pre_ref, axis=0)
        onset_med = np.median(X[iw0:iw1], axis=0)
        post_imm_med = np.median(post_imm_ref, axis=0)
        post_med = np.median(post_steady_ref, axis=0)
        response = onset_med - pre_med

        # 主通道 = 负载段均值最大（三组数据均为 ch17，保证跨组可比）
        main_ch = int(np.argmax(X[s0:s1].mean(axis=0)))
        loaded_mask = response > 0.15
        idle_mask = response < 0.02

        # 负载尾参考
        tail_med = np.median(X[s1 - int(3 * fs):s1], axis=0)
        drift_per_ch = tail_med - onset_med

        tl = t[s0:s1] - t[s0]
        Xload = X[s0:s1]

        # 每通道线性斜率
        slopes = np.array([np.polyfit(tl, Xload[:, j], 1)[0]
                           for j in range(X.shape[1])])

        # 主通道时漂形态拟合
        sig = X[:, main_ch]
        yload = sig[s0:s1]
        p0_lin = np.polyfit(tl, yload, 1)
        ylin = np.polyval(p0_lin, tl)
        try:
            popt_exp, _ = curve_fit(exp_model, tl, yload,
                                    p0=[yload[-1] - yload[0],
                                        max(tl[-1] / 3.0, 1.0), yload[0]],
                                    maxfev=50000)
            yexp = exp_model(tl, *popt_exp)
            r_exp = fit_r2(yload, yexp)
        except Exception:
            popt_exp, yexp, r_exp = None, None, np.nan
        A = np.vstack([np.log1p(tl), np.ones_like(tl)]).T
        (a_log, b_log), *_ = np.linalg.lstsq(A, yload, rcond=None)
        ylog = a_log * np.log1p(tl) + b_log
        r_lin = fit_r2(yload, ylin)
        r_log = fit_r2(yload, ylog)

        # 空载趋势（前/后），逐 1s bin 看是暖机还是长期零漂
        def bin_trend(x, tt, win=1.0):
            tt = tt - tt[0]
            n = max(1, int(round(len(tt) / (tt[-1] / win))) if tt[-1] > 0 else 1)
            edges = np.linspace(0, len(x), min(len(x), max(n, 2)) + 1).astype(int)
            return tt[edges[:-1]], np.array([np.mean(x[a:b]) for a, b in zip(edges[:-1], edges[1:])])

        pre_bin_t, pre_bin = bin_trend(sig[:s0], t[:s0])
        post_bin_t, post_bin = bin_trend(sig[s1:], t[s1:])
        if len(pre_bin) >= 2:
            k_pre, b_pre = np.polyfit(pre_bin_t, pre_bin, 1)
        else:
            k_pre = np.nan
        if len(post_bin) >= 2:
            k_post, b_post = np.polyfit(post_bin_t, post_bin, 1)
        else:
            k_post = np.nan

        # 卸载恢复拟合（主通道，相对 pre）
        yrec = sig[s1:] - pre_med[main_ch]
        trec = t[s1:] - t[s1]
        try:
            popt_rec, _ = curve_fit(rec_model, trec, yrec,
                                    p0=[yrec[0] - yrec[-1],
                                        max(1.0, trec[-1] / 10.0), yrec[-1]],
                                    maxfev=50000)
            tau_rec = popt_rec[1]
            rec_c = popt_rec[2] + pre_med[main_ch]
            r_rec = fit_r2(yrec, rec_model(trec, *popt_rec))
        except Exception:
            tau_rec, rec_c, r_rec = np.nan, np.nan, np.nan

        # 零漂：前空载偏移(相对0)、后空载-前空载
        pre_mean = X[:s0].mean(axis=0)
        post_mean = X[-int(5 * fs):].mean(axis=0)
        zero_shift = post_med - pre_med
        # 负载尾均值相对起始响应
        drift_load = drift_per_ch[main_ch]

        # 噪声（前空载末尾 3s，量化台阶明显）
        noise = np.std(pre_ref, axis=0)
        dpos = np.diff(np.sort(X[:, main_ch]))
        dpos = dpos[dpos > 0.00001]
        quant = float(np.median(dpos)) if len(dpos) else np.nan
        nonzero_q = np.mean(np.abs(np.diff(X[:, main_ch])) > 0.00049)

        # 未受载参考通道在负载段斜率
        ref_idle = X[:, idle_mask].mean(axis=1) if idle_mask.sum() else np.zeros_like(total)
        k_ref_load = np.polyfit(tl, ref_idle[s0:s1], 1)[0]
        # 漂移斜率与响应幅度的相关性
        if loaded_mask.sum() >= 4:
            corr_slope_resp = np.corrcoef(slopes[loaded_mask],
                                          response[loaded_mask])[0, 1]
        else:
            corr_slope_resp = np.nan

        # 主机现有 Kalman 文件的关键指标
        pre_med_k = np.median(Xk[max(0, s0 - int(3 * fs)):s0], axis=0)
        onset_med_k = np.median(Xk[iw0:iw1], axis=0)
        tail_med_k = np.median(Xk[s1 - int(3 * fs):s1], axis=0)
        post_med_k = np.median(Xk[s1 + int(2 * fs):s1 + int(5 * fs)], axis=0)

        R[name] = dict(
            n=len(t), fs=fs, dur=float(t[-1]),
            s0=int(s0), s1=int(s1),
            main_ch=main_ch, ch_main=d["ch_cols"][main_ch],
            ch_cols=d["ch_cols"],
            loaded_channels=int(loaded_mask.sum()),
            idle_channels=int(idle_mask.sum()),
            loaded_channel_names=[d["ch_cols"][i] for i in np.where(loaded_mask)[0]],
            idle_channel_names=[d["ch_cols"][i] for i in np.where(idle_mask)[0]],
            response=response.tolist(),
            pre_med=pre_med.tolist(),
            onset_med=onset_med.tolist(),
            tail_med=tail_med.tolist(),
            post_med=post_med.tolist(),
            post_imm_med=post_imm_med.tolist(),
            pre_mean=pre_mean.tolist(),
            post_mean=post_mean.tolist(),
            zero_shift=zero_shift.tolist(),
            drift_per_ch=drift_per_ch.tolist(),
            slopes=slopes.tolist(),
            step=float(response[main_ch]),
            drift_load=float(drift_load),
            drift_ratio=float(drift_load / response[main_ch]),
            k_lin=float(p0_lin[0]), b_lin=float(p0_lin[1]), r_lin=float(r_lin),
            r_log=float(r_log), a_log=float(a_log), b_log=float(b_log),
            exp_tau=float(popt_exp[1]) if popt_exp is not None else np.nan,
            exp_amp=float(popt_exp[0]) if popt_exp is not None else np.nan,
            exp_c=float(popt_exp[2]) if popt_exp is not None else np.nan,
            r_exp=float(r_exp),
            k_pre=float(k_pre), k_post=float(k_post),
            pre_bin_t=pre_bin_t.tolist(), pre_bin=pre_bin.tolist(),
            post_bin_t=post_bin_t.tolist(), post_bin=post_bin.tolist(),
            tau_rec=float(tau_rec), rec_c=float(rec_c), r_rec=float(r_rec),
            noise_median=float(np.median(noise)),
            noise_max=float(np.max(noise)),
            quant_step=float(quant),
            nonzero_q_ratio=float(nonzero_q),
            k_ref_load=float(k_ref_load),
            corr_slope_resp=float(corr_slope_resp),
            kalman_step=float(onset_med_k[main_ch] - pre_med_k[main_ch]),
            kalman_drift=float(tail_med_k[main_ch] - onset_med_k[main_ch]),
            kalman_pre=float(pre_med_k[main_ch]),
            kalman_post=float(post_med_k[main_ch]),
        )

        print(f"\n===== {name} =====")
        print(f"帧数={len(t)} 时长={t[-1]:.1f}s fs≈{fs:.1f}Hz")
        print(f"负载段 {t[s0]:.1f}~{t[s1]:.1f}s ({t[s1]-t[s0]:.1f}s)")
        print(f"主通道 {d['ch_cols'][main_ch]}: 前基线={pre_med[main_ch]:.4f} 阶跃={response[main_ch]:.4f} "
              f"末段漂移={drift_load:.4f} ({100*drift_load/response[main_ch]:.1f}%) 尾均值={tail_med[main_ch]:.4f}")
        print(f"时漂拟合R²: 线性={r_lin:.3f} 指数(τ={popt_exp[1]:.0f}s)={r_exp:.3f} 对数={r_log:.3f}")
        print(f"漂移斜率={p0_lin[0]*1000:.3f}e-3 N/s; 受载通道斜率与响应相关={corr_slope_resp:.3f}")
        print(f"零漂: 前基线={pre_med[main_ch]:.4f} 卸载后2~5s={post_imm_med[main_ch]:.4f} "
              f"稳态后基线={post_med[main_ch]:.4f} 稳态零漂={zero_shift[main_ch]:+.4f} "
              f"({100*zero_shift[main_ch]/response[main_ch]:+.1f}% of step)")
        print(f"前空载bin斜率={k_pre*1000:.3f}e-3/s 后空载bin斜率={k_post*1000:.3f}e-3/s")
        print(f"卸载恢复: τ={tau_rec:.1f}s, 稳态={rec_c:.4f}, R²={r_rec:.3f}")
        print(f"噪声std 前3s: 中位={np.median(noise):.4f} max={np.max(noise):.4f}; 量化台阶={quant:.4f}")
        print(f"未受载通道({idle_mask.sum()}): {[d['ch_cols'][i] for i in np.where(idle_mask)[0]]} "
              f"负载段共模斜率={k_ref_load*1000:.3f}e-3/s")
        print(f"主机Kalman: 阶跃={R[name]['kalman_step']:.4f} 漂移={R[name]['kalman_drift']:.4f} "
              f"前基线={R[name]['kalman_pre']:.4f} 后基线={R[name]['kalman_post']:.4f}")

    with open(RES / "characterization_metrics.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2, allow_nan=True)
    return R


def make_figures(R):
    pos = active_positions()

    # ---------- fig01 总览 ----------
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 10.5), constrained_layout=True)
    for r, name in enumerate(DATASETS):
        d = load_dataset(name)
        t, X, total = d["t"], d["X"], d["total"]
        r_ = R[name]
        s0, s1 = r_["s0"], r_["s1"]
        axes[r, 0].plot(t, total, lw=0.5, color="tab:blue")
        axes[r, 0].axvspan(t[s0], t[s1], color="orange", alpha=0.16)
        axes[r, 0].set_title(f"{name} 31通道总和 (N)")
        axes[r, 0].set_xlabel("时间 (s)"); axes[r, 0].set_ylabel("ΣForce (N)")
        axes[r, 0].grid(alpha=0.3)
        axes[r, 1].plot(t, X[:, r_["main_ch"]], lw=0.5, color="tab:red")
        axes[r, 1].axvspan(t[s0], t[s1], color="orange", alpha=0.16)
        axes[r, 1].axhline(r_["pre_med"][r_["main_ch"]], color="green", ls="--", lw=0.9)
        axes[r, 1].set_title(f"{name} 主通道 {r_['ch_main']} (N)")
        axes[r, 1].set_xlabel("时间 (s)"); axes[r, 1].set_ylabel("Force (N)")
        axes[r, 1].grid(alpha=0.3)
        # 空间响应图
        resp = channels_to_map(r_["response"])
        im = axes[r, 2].imshow(resp, cmap="hot", aspect="auto", origin="upper")
        axes[r, 2].set_title(f"{name} 阶跃响应空间分布 (N)")
        axes[r, 2].set_xticks([]); axes[r, 2].set_yticks([])
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046)
        # 标注物理位置
        for k in range(31):
            axes[r, 2].text(pos[k, 1], pos[k, 0], r_["ch_cols"][k],
                            ha="center", va="center", fontsize=6)
    fig.suptitle("图1  数据总览：空载(蓝区外) — 恒定负载(橙区) — 空载", fontsize=13)
    fig.savefig(FIG / "fig01_overview.png", dpi=150)
    plt.close(fig)

    # ---------- fig02 时漂形态 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    for ax, name in zip(axes, DATASETS):
        d = load_dataset(name)
        t = d["t"]; X = d["X"]
        r_ = R[name]
        s0, s1 = r_["s0"], r_["s1"]
        j = r_["main_ch"]
        tl = t[s0:s1] - t[s0]
        y = X[s0:s1, j]
        ax.plot(tl, y, lw=0.45, color="gray", alpha=0.85, label="原始")
        ax.plot(tl, np.polyval([r_["k_lin"], r_["b_lin"]], tl), lw=1.3,
                label=f"线性 R²={r_['r_lin']:.3f}")
        if np.isfinite(r_["exp_tau"]):
            tau = r_["exp_tau"]; amp = r_["exp_amp"]; c = r_["exp_c"]
            ax.plot(tl, exp_model(tl, amp, tau, c), lw=1.5,
                    label=f"指数 τ={tau:.1f}s R²={r_['r_exp']:.3f}")
        ax.plot(tl, log_model(tl, r_["a_log"], r_["b_log"]), lw=1.3, ls="--",
                label=f"对数 R²={r_['r_log']:.3f}")
        ax.set_title(f"{name} 主通道 {r_['ch_main']} 负载段")
        ax.set_xlabel("负载持续时间 (s)"); ax.set_ylabel("Force (N)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("图2  恒定负载段时漂形态与线性/指数/对数模型拟合", fontsize=13)
    fig.savefig(FIG / "fig02_drift_models.png", dpi=150)
    plt.close(fig)

    # ---------- fig03 零漂 ----------
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.6), constrained_layout=True)
    for k, name in enumerate(DATASETS):
        d = load_dataset(name)
        t = d["t"]; X = d["X"]
        r_ = R[name]
        j = r_["main_ch"]; s0, s1 = r_["s0"], r_["s1"]
        sig = X[:, j]
        ax = axes[0, k]
        ax.plot(t[:s0], sig[:s0], lw=0.5, color="tab:blue")
        ax.plot(t[s1:], sig[s1:], lw=0.5, color="tab:green")
        ax.axhline(r_["pre_med"][j], color="tab:blue", ls="--", lw=1)
        ax.axhline(r_["post_med"][j], color="tab:green", ls="--", lw=1)
        ax.set_title(f"{name} {r_['ch_main']} 空载段 前基线={r_['pre_med'][j]:.3f} "
                     f"后基线={r_['post_med'][j]:.3f}")
        ax.set_xlabel("时间 (s)"); ax.set_ylabel("Force (N)"); ax.grid(alpha=0.3)
        # 归一化空载趋势：把前/后空载按各自起点排齐
        ax2 = axes[1, k]
        tp = t[:s0] - t[0]
        tq = t[s1:] - t[s1]
        ax2.plot(tp, sig[:s0] - r_["pre_med"][j], lw=0.6, color="tab:blue",
                 label=f"前空载(斜率{r_['k_pre']*1000:.2f}e-3/s)")
        ax2.plot(tq, sig[s1:] - r_["post_med"][j], lw=0.6, color="tab:green",
                 label=f"后空载(斜率{r_['k_post']*1000:.2f}e-3/s)")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_title(f"{name} 空载段相对各自均值的趋势（暖机/回零检验）")
        ax2.set_xlabel("相对段起点时间 (s)"); ax2.set_ylabel("Force-均值 (N)")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.suptitle("图3  零漂：前空载基线、卸载后基线与段内趋势", fontsize=13)
    fig.savefig(FIG / "fig03_zero_drift.png", dpi=150)
    plt.close(fig)

    # ---------- fig04 空间图 ----------
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 14.5), constrained_layout=True)
    map_titles = ["负载阶跃响应", "负载末-初漂移", "前空载零偏", "卸载后残余(后-前)"]
    for row, key in enumerate(["response", "drift_per_ch", "pre_mean", "zero_shift"]):
        for k, name in enumerate(DATASETS):
            r_ = R[name]
            ax = axes[row, k]
            grid = channels_to_map(r_[key])
            cmap = "hot" if row == 0 else ("coolwarm" if row != 2 else "hot")
            vlim = np.nanmax(np.abs(grid)) if row in (1, 3) else None
            im = ax.imshow(grid, cmap=cmap, aspect="auto",
                           vmin=-vlim if vlim else None, vmax=vlim if vlim else None)
            ax.set_title(f"{name} {map_titles[row]}")
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("图4  阵列空间特征：响应、时漂、零偏与卸载残余", fontsize=13)
    fig.savefig(FIG / "fig04_spatial_maps.png", dpi=150)
    plt.close(fig)

    # ---------- fig05 共模/参考通道 ----------
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), constrained_layout=True)
    for k, name in enumerate(DATASETS):
        d = load_dataset(name)
        t, X = d["t"], d["X"]
        r_ = R[name]
        s0, s1 = r_["s0"], r_["s1"]
        loaded = np.array([d["ch_cols"].index(c) for c in r_["loaded_channel_names"]])
        idle = np.array([d["ch_cols"].index(c) for c in r_["idle_channel_names"]])
        ax = axes[0, k]
        ax.plot(t, X[:, loaded].mean(axis=1), lw=0.6, color="tab:red",
                label=f"受载组均值({len(loaded)}ch)")
        ax2 = ax.twinx()
        ax2.plot(t, X[:, idle].mean(axis=1), lw=0.9, color="tab:blue",
                 label=f"未受载组均值({len(idle)}ch)")
        ax.axvspan(t[s0], t[s1], color="orange", alpha=0.12)
        ax.set_title(f"{name} 受载组 vs 未受载参考通道")
        ax.set_xlabel("时间 (s)"); ax.set_ylabel("受载组 (N)", color="tab:red")
        ax2.set_ylabel("未受载组 (N)", color="tab:blue")
        ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
        # 漂移斜率 vs 阶跃响应散点
        ax = axes[1, k]
        resp = np.array(r_["response"])
        sl = np.array(r_["slopes"]) * 1000
        ax.scatter(resp[idle], sl[idle], s=28, color="tab:blue", label="未受载")
        ax.scatter(resp[loaded], sl[loaded], s=28, color="tab:red", label="受载")
        for j in loaded:
            if resp[j] > 0.5:
                ax.annotate(d["ch_cols"][j], (resp[j], sl[j]), fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(f"{name} 各通道漂移斜率 vs 负载响应 "
                     f"(相关={r_['corr_slope_resp']:.2f})")
        ax.set_xlabel("阶跃响应 (N)"); ax.set_ylabel("负载段线性斜率 (e-3 N/s)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("图5  时漂的共模性检验与响应-漂移关系", fontsize=13)
    fig.savefig(FIG / "fig05_common_mode.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    R = characterize()
    make_figures(R)
    print(f"\nfigures saved to {FIG}")
    print(f"metrics saved to {RES / 'characterization_metrics.json'}")
