# -*- coding: utf-8 -*-
"""t4b_q9_loadlevel.py —— T4-Q9：跨载荷量级的缺口声明与可做的替代检验。

两层结论：
 ① **绝对量级（N）依赖性：缺数据，无法判定**。判断依据有三条免假设的证据：
    (a) 13 份录制里没有任何一份带"载荷量（N/gf）"标注文件（脚本自动检查 session.json 的
        calibration 字段与录制目录下的文件清单）；
    (b) ADC 域录制的总量峰值在 4 份之间差数倍（不同手指/不同传感器/不同灵敏度），
        没有可比的物理标度 ⇒ 跨录制比"绝对幅度"无意义；
    (c) 恒载 9 组是 processed_display 域且是单位数，更不可换算。
 ② **相对量级（占本次满量程的比例）依赖性：有数据、可判定**。用"平台电平占该录制峰值的比例"
    作为相对量级代理，看：不同相对量级的加载事件，其快相归一化形状是否不同。

产物：results/t4b_loadlevels.csv（每事件：相对量级 + 形状）、
      results/t4b_loadlevel_shape.csv（按相对量级分档的形状对比 + AUC）、
      results/t4b_loadlevel_probe.csv（量级标注缺失的自动检查结果）
运行：python scripts/t4b_q9_loadlevel.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t4b_common as C  # noqa: E402


def auc_mw(x, y):
    x = np.asarray([v for v in x if v == v], float)
    y = np.asarray([v for v in y if v == v], float)
    if len(x) < 3 or len(y) < 3:
        return np.nan
    allv = np.concatenate([x, y])
    r = pd.Series(allv).rank().to_numpy()
    rx = r[:len(x)].sum()
    return float((rx - len(x) * (len(x) + 1) / 2.0) / (len(x) * len(y)))


def probe_annotations():
    """检查是否有任何载荷量（N/gf）标注：录制的 session.json 字段 + 目录内文件清单。"""
    rows = []
    for name, path, dom in C.RECS:
        d = os.path.dirname(path)
        sj = os.path.join(d, "session.json")
        keys, cal, unit_hits, extra = [], {}, [], {}
        if os.path.isfile(sj):
            try:
                with open(sj, "r", encoding="utf-8", errors="ignore") as f:
                    j = json.load(f)
                keys = sorted(j.keys())
                cal = j.get("calibration", {}) if isinstance(j.get("calibration"), dict) else {}
                # 量纲可用性的决定性字段（不是"关键字出现在文本里"，而是具体取值）
                for k in ("force_conversion_active", "display_mode", "params",
                          "reported_force_unit", "display_force_unit"):
                    if k in cal:
                        extra[k] = cal[k]
                if not extra.get("force_conversion_active"):
                    unit_hits.append("force_conversion_active=False⇒无物理标度")
            except Exception as e:      # noqa: BLE001
                extra = {"_error": str(e)}
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        rows.append(dict(rec=name, dom=dom, session_json=os.path.isfile(sj),
                         top_keys="|".join(str(k) for k in keys[:12]),
                         force_conversion_active=str(extra.get("force_conversion_active")),
                         calib_params=str(extra.get("params")),
                         reported_force_unit=str(extra.get("reported_force_unit")),
                         unit_keyword_hits="|".join(unit_hits),
                         files="|".join(files[:8])))
    return pd.DataFrame(rows)


def main():
    ev, cache = C.build_events(verbose=False)
    rise = ev[ev.kind.isin(["onset", "restep"])].copy()

    # ── ① 标注检查 ──
    pr = probe_annotations()
    pr.to_csv(os.path.join(C.RES, "t4b_loadlevel_probe.csv"), index=False, encoding="utf-8-sig")
    print("== ① 载荷量标注检查（13 份录制）==")
    print(pr[["rec", "dom", "session_json", "force_conversion_active", "calib_params",
              "reported_force_unit", "unit_keyword_hits"]].to_string(index=False))
    hit = pr.unit_keyword_hits.ne("").sum()
    print("\nforce_conversion_active 为假（⇒ ADC 无物理标度）的录制数 = %d / %d"
          % (hit, len(pr)))
    print("calibration.params 非空的录制数 = %d / %d"
          % (pr.calib_params.ne("{}").sum(), len(pr)))
    print("任何一份录制带载荷量（N/gf）数值标注的：%s"
          % ("无" if pr.calib_params.ne("{}").sum() == 0 else "有"))

    # ── ② 相对量级：平台电平 ÷ 录制峰值 ──
    rows = []
    for name, cc in cache.items():
        sub = rise[rise.rec == name]
        if not len(sub):
            continue
        peak = cc["peak"]
        ts, dt, tot = cc["tu"], cc["dt"], cc["tot"]
        for _, r in sub.iterrows():
            k = int(np.searchsorted(ts, r.t_on))
            a = k + int(4.0 / dt)
            b = k + int(6.0 / dt)
            post = float(np.median(tot[a:b])) if b < len(tot) else np.nan
            rows.append(dict(rec=name, t_on=r.t_on, kind=r.kind, arm=r.arm,
                             pre_frac_nc=r.pre_frac_nc, post_frac_rec=post / max(peak, 1e-9),
                             J_frac_rec=r.J_frac_nc,
                             J_over_post=float(r.J_nc) / post if post > 1e-9 else np.nan,
                             **{c: r[c] for c in rise.columns if c.startswith("sh_") and c.endswith("b")}))
    ll = pd.DataFrame(rows)
    ll.to_csv(os.path.join(C.RES, "t4b_loadlevels.csv"), index=False, encoding="utf-8-sig")
    print("\n== ② 事件落点的相对量级（占录制峰值；仅加载类事件 n=%d）==" % len(ll))
    print(ll.groupby("kind")[["pre_frac_nc", "post_frac_rec", "J_frac_rec"]].describe()
          .loc[:, (slice(None), ["min", "50%", "max"])].to_string())

    # 分档：加载后电平 < 0.55 峰值 = "半量级"，≥ 0.55 = "满量级"
    ll["tier"] = np.where(ll.post_frac_rec < 0.55, "half_scale", "full_scale")
    print("\n分档计数：", dict(ll.tier.value_counts()))
    print("每档的录制来源：")
    print(ll.groupby("tier").rec.apply(lambda s: "|".join(sorted(set(s)))).to_string())

    # ── 形状是否随相对量级变 ──
    cols = [c for c in ll.columns if c.startswith("sh_") and c.endswith("b")]
    rows = []
    for c in cols:
        half = ll[(ll.tier == "half_scale")][c].dropna()
        full = ll[(ll.tier == "full_scale")][c].dropna()
        # 同族内（onset）也看一遍，避免与形态混淆
        oh = ll[(ll.tier == "half_scale") & (ll.kind == "onset")][c].dropna()
        of = ll[(ll.tier == "full_scale") & (ll.kind == "onset")][c].dropna()
        rows.append(dict(tau=int(c[3:6]) / 100,
                         n_half=len(half), n_full=len(full),
                         med_half=half.median(), med_full=full.median(),
                         d_pt=100 * (half.median() - full.median()) if len(half) and len(full) else np.nan,
                         auc_mixed=auc_mw(half, full),
                         n_onset_half=len(oh), n_onset_full=len(of),
                         med_onset_half=oh.median(), med_onset_full=of.median(),
                         d_pt_onset=100 * (oh.median() - of.median()) if len(oh) and len(of) else np.nan,
                         auc_onset=auc_mw(oh, of)))
    ls = pd.DataFrame(rows)
    ls.to_csv(os.path.join(C.RES, "t4b_loadlevel_shape.csv"), index=False, encoding="utf-8-sig")
    print("\n== 形状随相对量级的变化（sh_*b 归一到各自 J(5s)）==")
    print(ls.to_string(index=False))

    # ── 连续视角：形状 vs 加载后相对量级的秩相关（比二分档更有力）──
    from scipy.stats import spearmanr   # noqa: E402
    print("\n== 形状 vs 加载后相对量级 post_frac_rec 的 Spearman 秩相关 ==")
    cor = []
    for c in cols:
        for sub, tag in ((ll, "全部加载事件"), (ll[ll.kind == "onset"], "仅 onset")):
            x, y = sub.post_frac_rec.to_numpy(float), sub[c].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 5:
                continue
            rho, p = spearmanr(x[m], y[m])
            cor.append(dict(tau=int(c[3:6]) / 100, subset=tag, n=int(m.sum()),
                            rho=round(float(rho), 4), p=round(float(p), 4)))
    cr = pd.DataFrame(cor)
    cr.to_csv(os.path.join(C.RES, "t4b_loadlevel_corr.csv"), index=False, encoding="utf-8-sig")
    print(cr.pivot_table(index="tau", columns="subset", values=["rho", "p"]).to_string())

    # ── 档位清单（T4-Q9 的"有哪些量级"直接证据）──
    pl = pd.read_csv(os.path.join(C.RES, "t4b_plateaus.csv"))
    adc = pl[pl.dom == "ADC域"].copy()
    adc = adc[adc.level_frac_peak > 0.02]
    tier = (adc.groupby(["rec", "level_frac_peak"])
            .agg(dur_tot=("dur", "sum"), dur_max=("dur", "max")).reset_index())
    tier.to_csv(os.path.join(C.RES, "t4b_loadlevel_tiers.csv"), index=False,
                encoding="utf-8-sig")
    print("\n== ADC 录制里的载荷档位（平台电平/录制峰值，只列持续 >3 s 的真档位）==")
    print(adc[adc.dur > 3][["rec", "level", "level_frac_peak", "t0", "t1", "dur"]]
          .sort_values(["rec", "level_frac_peak"]).to_string(index=False))

    # ── 跨录制的"物理标度"是否可比 ──
    print("\n== ③ 跨录制的量级可比性检查（总量峰值，ADC/显示单位）==")
    pk = pd.DataFrame([dict(rec=n, dom=cc["dom"], peak=cc["peak"],
                            n_ch=cc["Xu"].shape[1]) for n, cc in cache.items()])
    print(pk.to_string(index=False))
    adc = pk[pk.dom == "ADC域"]
    print("\nADC 域 4 份峰值：%.0f ~ %.0f（差 %.2f 倍）⇒ 无物理标度不可跨录制比绝对量级"
          % (adc.peak.min(), adc.peak.max(), adc.peak.max() / max(adc.peak.min(), 1e-9)))
    print("\n-> results/t4b_loadlevels.csv / t4b_loadlevel_shape.csv / t4b_loadlevel_probe.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
