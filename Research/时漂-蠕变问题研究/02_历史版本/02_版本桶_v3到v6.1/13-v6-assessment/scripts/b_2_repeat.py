# -*- coding: utf-8 -*-
"""b2：v6 的可重复性（问题 2）。

用「同一传感器同一负载的 3 次重复录制」（右/左拇指、四指 数据1/2/3，共 9 份恒载）测组内离散度。
臂指标直接复用 11-paper-v6/results/{metrics_all,metrics_settle}.csv（同一次运行），
另从 cache/*.npz 取逐帧曲线与 epoch 台账，从 CSV 全速率取原始读数做「形状先验」证据。

产出：
  results/b_repeat_records.csv   每（录制 × 臂）一行的全部可重复性指标
  results/b_repeat_disp.csv      每（传感器组 × 臂 × 指标）的组内离散度（std/极差/CV）
  results/b_repeat_shape.csv     pin 语义的形状先验证据（Â 预测 vs 实测平台误差）
  results/_b2_repeat.log         日志
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_common as B                                          # noqa: E402

LOG = []
HOLD = [t for t, _ in B.ALL if B.KIND[t] == "恒载"]
GROUP = {"右拇指指尖": "右拇指", "左拇指指尖": "左拇指", "四指指尖": "四指"}


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def _corr(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan, np.nan, np.nan, np.nan, int(m.sum())
    rs, ps = stats.spearmanr(x[m], y[m])
    rp, pp = stats.pearsonr(x[m], y[m])
    return float(rs), float(ps), float(rp), float(pp), int(m.sum())


def main():
    B.log_reconfigure()
    ma = pd.read_csv(os.path.join(B.PAPER_RES, "metrics_all.csv"))
    ms = pd.read_csv(os.path.join(B.PAPER_RES, "metrics_settle.csv"))
    rows, shape_rows = [], []
    for tag, path in B.ALL:
        if tag not in HOLD:
            continue
        d = B.load_csv(tag, path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
        tot = Xu.sum(axis=1)
        ts1 = B.med(tot, dt, 0.1)
        fo = B.C.first_onset(tu, tot, dt)
        i_e, pre = fo[0], fo[1]
        a_step = float(ts1[min(len(tu) - 1, i_e + int(0.2 / dt))] - pre)
        inc5 = float(ts1[min(len(tu) - 1, i_e + int(5.0 / dt))] - pre)
        Ahat, Ahat_i5, mshape = B.shape_mismatch(tu, tot, i_e, dt)
        z = np.load(B.cache_path(tag), allow_pickle=True)
        tuc = z["tu"]
        grp = GROUP[tag.split("/")[0]]
        kl_all = {}
        for arm in B.ARMS:
            kl = [str(s) for s in z.get("kindlog_" + arm, np.array([], dtype=object))]
            ev = [(float(s.split("|")[0]), s.split("|")[1]) for s in kl if "|" in s]
            kl_all[arm] = ev
        n_pre_var = len(set(kl_all["e3s"]) ^ set(kl_all["v6"]))
        for arm in B.ARMS:
            # 全部臂都在缓存栅格（25 Hz）上比较：Y_* = 0.5 s 中位平滑的显示总量
            dis = z["Y_" + arm]
            dis_raw = z["Yraw_" + arm].sum(axis=1)
            lvl = float(np.median(dis[(tuc >= tu[i_e] + 40.0) & (tuc <= tu[i_e] + 60.0)])) \
                if tu[i_e] + 60.0 <= tuc[-1] else float("nan")
            raw_lvl = float(np.median(z["tot_s"][(tuc >= tu[i_e] + 40.0) & (tuc <= tu[i_e] + 60.0)]))
            tgt = pre + a_step
            tgt5 = pre + inc5                      # 真值代理：加载沿 +5 s 的原始电平
            win = (tuc >= tu[i_e] + 5.0) & (tuc <= tuc[-1])
            max_dev = float(np.max(np.abs(dis[win] - tgt5))) if win.any() else np.nan
            w2 = (tuc >= tu[i_e] + 30.0) & (tuc <= tu[i_e] + 60.0)
            flat = float(np.max(dis[w2]) - np.min(dis[w2])) if (tu[i_e] + 60.0 <= tuc[-1]) else np.nan
            mv = (float(np.median(dis[(tuc >= tu[i_e] + 50.0) & (tuc <= tu[i_e] + 60.0)]))
                  - float(np.median(dis[(tuc >= tu[i_e] + 30.0) & (tuc <= tu[i_e] + 40.0)]))) \
                if (tu[i_e] + 60.0 <= tuc[-1]) else np.nan
            mrows = ma[(ma.dataset == tag) & (ma.algo == arm)]
            srows = ms[(ms.dataset == tag) & (ms.algo == arm)]
            ev = kl_all[arm]
            n_pre = sum(1 for t, _ in ev if t < tu[i_e] - 0.3)
            rows.append(dict(
                rec=tag, group=grp, arm=arm, t_e=float(tu[i_e]), pre=pre,
                a_step=a_step, inc5=inc5, Ahat=Ahat, m_shape=mshape,
                err_pred_pct=100.0 * (Ahat / a_step - 1.0) if abs(a_step) > 1e-12 else np.nan,
                plat_lvl=lvl, plat_raw=raw_lvl,
                err5s=float(srows.err5s.iloc[0]) if len(srows) else np.nan,
                err_plat5_pct=100.0 * ((lvl - pre) / inc5 - 1.0) if abs(inc5) > 1e-12 else np.nan,
                err_platstep_pct=100.0 * (lvl - tgt) / a_step if abs(a_step) > 1e-12 else np.nan,
                gain_plat=(1.0 - (lvl - pre) / (raw_lvl - pre)
                           if abs(raw_lvl - pre) > 1e-12 else np.nan),
                flat_plat=flat, flat5_pct=100.0 * flat / max(abs(inc5), 1e-12),
                move3060_pct=100.0 * mv / max(abs(inc5), 1e-12),
                max_dev=max_dev, max_dev5_pct=100.0 * max_dev / max(abs(inc5), 1e-12),
                plat_pred=pre + Ahat,
                err_vs_pred_pct=100.0 * (lvl - pre - Ahat) / max(abs(inc5), 1e-12),
                drift_slow=float(mrows.drift_slow.iloc[0]) if len(mrows) else np.nan,
                drift_main=float(mrows.drift_main.iloc[0]) if len(mrows) else np.nan,
                t_stable=float(srows.t_stable.iloc[0]) if len(srows) else np.nan,
                t_settle=float(srows.t_settle.iloc[0]) if len(srows) else np.nan,
                t_flat30=float(srows.t_flat30.iloc[0]) if len(srows) else np.nan,
                epoch_total=int(mrows.epoch.iloc[0]) if len(mrows) else -1,
                epoch_pre=n_pre,
                epochs_before=";".join(f"{t:.2f}:{k}" for t, k in ev if t < tu[i_e] - 0.3),
                epochs_after=";".join(f"{t:.2f}:{k}" for t, k in ev if t >= tu[i_e] - 0.3),
                a_max=float(mrows.a_max.iloc[0]) if len(mrows) else np.nan))
        shape_rows.append(dict(rec=tag, group=grp, t_e=float(tu[i_e]), pre=pre,
                               a_step=a_step, inc5=inc5, Ahat=Ahat, m_shape=mshape,
                               err_pred_pct=100.0 * (Ahat / a_step - 1.0),
                               n_epoch_v6=len(kl_all["v6"]),
                               n_pre_v6=sum(1 for t, _ in kl_all["v6"] if t < tu[i_e] - 0.3),
                               n_epoch_v51=len(kl_all["e3s"]),
                               v6_epochs=";".join(f"{t:.2f}:{k}" for t, k in kl_all["v6"])))
    rec = pd.DataFrame(rows)
    shp = pd.DataFrame(shape_rows)
    rec.to_csv(os.path.join(B.RES, "b_repeat_records.csv"), index=False, encoding="utf-8-sig")
    shp.to_csv(os.path.join(B.RES, "b_repeat_shape.csv"), index=False, encoding="utf-8-sig")

    # ── 组内离散度 ──
    metrics = [("err5s", "5 s 电平误差(%)"), ("err_plat5_pct", "平台误差@40-60s(%,基准=5s电平)"),
               ("plat_lvl", "平台绝对电平"), ("gain_plat", "蠕变扣除比例"),
               ("max_dev5_pct", "全程最大偏差(%·5s电平)"), ("flat5_pct", "平台内极差(%)"),
               ("move3060_pct", "平台 30→60s 移动(%)"),
               ("drift_slow", "慢相段时漂(%)"), ("t_stable", "T_stable(s)"),
               ("epoch_total", "epoch 总数"), ("epoch_pre", "真沿前 epoch 数")]
    drow = []
    for (grp, arm), g in rec.groupby(["group", "arm"]):
        for key, lbl in metrics:
            v = g[key].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            drow.append(dict(group=grp, arm=arm, metric=key, label=lbl, n=len(v),
                             mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                             rng=float(np.max(v) - np.min(v)) if len(v) > 1 else np.nan,
                             cv=(100.0 * float(np.std(v, ddof=1)) / abs(float(np.mean(v)))
                                 if len(v) > 1 and abs(float(np.mean(v))) > 1e-12 else np.nan),
                             values="|".join(f"{x:.3f}" for x in v)))
    disp = pd.DataFrame(drow)
    disp.to_csv(os.path.join(B.RES, "b_repeat_disp.csv"), index=False, encoding="utf-8-sig")

    # 跨 3 个传感器组的「离散度中位」（作为该臂的可重复性总评）
    ag = disp.groupby(["arm", "metric", "label"]).agg(
        std_med=("std", "median"), rng_med=("rng", "median"),
        cv_med=("cv", "median"), mean_med=("mean", "median")).reset_index()

    p("=" * 120)
    p("表 1  可重复性：3 次重复录制 × 4 臂 的关键指标（原始单位；err5s/err_plat 已由台阶归一）")
    p("=" * 120)
    show = ["rec", "arm", "a_step", "inc5", "err5s", "err_plat5_pct", "plat_lvl", "plat_pred",
            "err_vs_pred_pct", "gain_plat", "max_dev5_pct", "flat5_pct", "move3060_pct",
            "drift_slow", "t_stable", "epoch_total", "epoch_pre", "Ahat", "m_shape"]
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        p(rec[show].round(3).to_string(index=False))
    p("")
    p("=" * 120)
    p("表 2  组内离散度（3 次重复；rng=极差）——分组明细")
    p("=" * 120)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        p(disp[["group", "arm", "label", "n", "mean", "std", "rng", "cv", "values"]]
          .round(3).to_string(index=False))
    p("")
    p("=" * 120)
    p("表 3  离散度跨 3 个传感器组取中位（越小越可重复）")
    p("=" * 120)
    piv_std = ag.pivot_table(index=["metric", "label"], columns="arm", values="std_med")
    piv_rng = ag.pivot_table(index=["metric", "label"], columns="arm", values="rng_med")
    piv_cv = ag.pivot_table(index=["metric", "label"], columns="arm", values="cv_med")
    for nm, piv in (("std 中位", piv_std), ("极差 中位", piv_rng), ("CV% 中位", piv_cv)):
        p(f"  [{nm}]")
        with pd.option_context("display.width", 200):
            p(piv[list(B.ARMS)].round(3).to_string())
        p("")
    p("  均值中位（同口径的偏差，用于区分「偏」与「不稳」）：")
    piv_mean = ag.pivot_table(index=["metric", "label"], columns="arm", values="mean_med")
    with pd.option_context("display.width", 200):
        p(piv_mean[list(B.ARMS)].round(3).to_string())
    p("  各臂「离散度最大」的指标个数（按 std 中位比较）：")
    worst = piv_std[list(B.ARMS)].idxmax(axis=1).value_counts()
    p("    " + worst.to_string().replace("\n", "\n    "))
    p("")
    p("=" * 120)
    p("表 4  pin 语义的形状先验证据：Â（形状反演）× 形状匹配度 × 实沿前 epoch 数")
    p("=" * 120)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        p(shp[["rec", "group", "a_step", "inc5", "Ahat", "m_shape", "err_pred_pct",
               "n_epoch_v6", "n_pre_v6", "n_epoch_v51", "v6_epochs"]].round(3).to_string(index=False))
    p("")
    obs = rec[rec.arm == "v6"].set_index("rec")
    for tgt in ("err_plat5_pct", "err5s"):
        x = [obs.loc[r, "err_pred_pct"] for r in shp.rec]
        y = [obs.loc[r, tgt] for r in shp.rec]
        rs, ps, rp, pp, n = _corr(x, y)
        p(f"  Â 预测误差 vs 实测 {tgt}：Spearman ρ={rs:.3f}(p={ps:.3f})  "
          f"Pearson r={rp:.3f}(p={pp:.3f})  n={n}")
    for metric in ("err_plat5_pct", "err5s", "plat_lvl", "gain_plat", "max_dev5_pct",
                   "flat5_pct", "move3060_pct", "drift_slow", "t_stable", "epoch_total"):
        x = [obs.loc[r, "epoch_pre"] for r in shp.rec]
        y = [obs.loc[r, metric] for r in shp.rec]
        rs, ps, rp, pp, n = _corr(x, y)
        p(f"  真沿前 epoch 数 vs v6 {metric}：ρ={rs:.3f}(p={ps:.3f})  r={rp:.3f}(p={pp:.3f})")
    with open(os.path.join(B.RES, "_b2_repeat.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    print("\n-> results/b_repeat_records.csv / b_repeat_disp.csv / b_repeat_shape.csv / "
          "_b2_repeat.log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
