# -*- coding: utf-8 -*-
"""T7-B 步骤 1：方向统一事件指标 + 卸载/部分卸载形状 + 幅度-回复关系（T7-Q6 数据面）。

输入（只读）：
  T4-A 冻结事件表 results/t4a_morphology.csv（60 事件，**不重新检测事件时刻**）
  T4-A 等效输入斜坡表 results/t4a_input_recover.csv（加载侧 T_ramp，用于分层）
  temp/ 下 13 份录制（4 份 ADC 域实录 + 9 份显示域恒载）

方向统一口径（§2 明写）：
  pre = median Z[t0-2,t0)；post = median Z[t0+4,t0+6]（对齐 00_共享/指标字典 §2.1）
  D = post - pre（带符号）；f(τ) = (Z(t0+τ) - pre)/D；t_α = f 首次 ≥ α/100
  ⇒ 加载与卸载同式，符号差异不会被误当机制差异。

输出：
  results/t7b_events_metrics.csv       60 事件方向统一指标（含 ±1 包、信噪比、窗污染旗标）
  results/t7b_shape_long.csv           60 事件归一化轮廓（逐事件 × τ 网格）
  results/t7b_shape_summary.csv        分组轮廓中位 + p10~p90
  results/t7b_decrement_inventory.csv  T7-B 自建减重扫描（补样本量，口径另标）
  results/t7b_partial_amp_response.csv 减重事件统一表：幅度 → 回复速度 + **实测重新分档**
  results/t7b_partial_shape.csv        减重事件归一化轮廓（用自适应 D）
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t7b_ad_lib import (RECS, DOMAIN, rec_path, load_rec, pkt_dt, to_grid,  # noqa: E402
                        read_frozen_events, read_input_recover, event_metrics,
                        TAU_GRID, FS, RESULTS, ensure_dirs)

ensure_dirs()
LOG = os.path.join(RESULTS, "_t7b_01_shape.log")
IDX = {float(t): i for i, t in enumerate(TAU_GRID)}


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.f.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.f.flush()


_CACHE = {}


def grid_of(key):
    if key not in _CACHE:
        t, X, meta = load_rec(rec_path(key))
        tu, Xu = to_grid(t, X)
        _CACHE[key] = dict(tu=tu, zu=Xu.sum(1), Xu=Xu, pkt=pkt_dt(t),
                           span=float(t[-1] - t[0]), zpeak=float(Xu.sum(1).max()))
    return _CACHE[key]


def adaptive_post_win(t0, t_next):
    """自适应 post 窗：默认 [t0+4,t0+6]；被后续事件压到时缩短到 [t0+1, t_next-1]。"""
    if np.isfinite(t_next) and t_next < t0 + 7.0:
        hi = max(t0 + 2.0, t_next - 1.0)
        lo = max(t0 + 1.0, hi - 2.0)
        if lo >= hi:
            lo, hi = t0 + 1.0, t0 + 2.0
        return lo - t0, hi - t0, "shortened"
    return 4.0, 6.0, "clean"


def classify(drop_frac):
    """按 00_共享/指标字典 §2.1 的实测幅度重新分档。"""
    if not np.isfinite(drop_frac):
        return "n/a"
    if drop_frac >= 0.80:
        return "unload"
    if drop_frac >= 0.20:
        return "partial_unload"
    return "decrement_small"


# ── T7-B 自建减重扫描（口径另标：与冻结表检测口径不同，仅用于补样本量/召回交叉验证）──
def decrement_inventory(key, zs, tu, min_frac=0.05, dead_s=6.0, trig_frac=0.03,
                        amp_frac_peak=0.02):
    z = pd.Series(zs)
    zpeak = float(np.max(zs))
    pre = z.rolling(int(2 * FS)).median().shift(1).to_numpy()
    post = z.rolling(int(2 * FS)).median().shift(-int(6 * FS)).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        drop = (pre - post) / np.maximum(np.abs(pre), 1.0)
    cand = np.where(np.isfinite(drop) & (drop >= min_frac))[0]
    dz = np.diff(zs, prepend=zs[0])
    jump_ok = np.abs(dz) >= trig_frac * np.maximum(np.abs(pre), 1.0)
    cand = cand[jump_ok[cand]] if len(cand) else cand
    out = []
    if len(cand) == 0:
        return out
    order = cand[np.argsort(-drop[cand])]
    taken = []
    for i in order:
        if any(abs(i - j) < dead_s * FS for j in taken):
            continue
        taken.append(i)
    for i in sorted(taken):
        lo = max(0, i - int(0.5 * FS))
        hi = min(len(zs) - 1, i + int(0.5 * FS))
        seg = np.arange(lo, hi + 1)
        if len(seg) == 0:
            continue
        i0 = int(seg[np.argmin(dz[lo:hi + 1])])
        t0 = tu[i0]
        a = (tu >= t0 - 2.0) & (tu < t0)
        b = (tu >= t0 + 4.0) & (tu < t0 + 6.0)
        if a.sum() < 20 or b.sum() < 20:
            continue
        pa, pb = zs[a], zs[b]
        mad_a = float(np.median(np.abs(pa - np.median(pa))))
        mad_b = float(np.median(np.abs(pb - np.median(pb))))
        pre_m, post_m = float(np.median(pa)), float(np.median(pb))
        J = post_m - pre_m
        sigma = 1.4826 * max(mad_a, mad_b)
        amp_min = max(amp_frac_peak * zpeak, 5.0 * sigma)
        if abs(J) < amp_min:
            continue
        tol = max(0.02 * max(abs(pre_m), abs(post_m)), 0.05 * abs(J))
        if mad_a > tol or mad_b > tol:
            continue
        m = event_metrics(zs, tu, t0)
        if m is None or m["J"] >= 0:
            continue
        m.update(dict(key=key, t_on=t0, drop_frac=float(drop[i]), noise_sigma=sigma,
                      amp_min=amp_min))
        out.append(m)
    return out


def main():
    print("CMD: python scripts/t7b_01_shape.py")
    print("=" * 78)
    ev = read_frozen_events()
    rc = read_input_recover()[["key", "t_on", "kind", "T_ramp", "gain_ramp",
                              "rmse_ramp_pct", "rmse_step_pct"]].copy()
    rc["t_on_r"] = rc["t_on"].round(2)
    ev["t_on_r"] = ev["t_on"].round(2)
    ev = ev.merge(rc.drop(columns=["t_on", "kind"]), on=["key", "t_on_r"], how="left")

    rows, shape_rows = [], []
    for _, r in ev.iterrows():
        key = r["key"]
        g = grid_of(key)
        m = event_metrics(g["zu"], g["tu"], float(r["t_on"]))
        if m is None:
            print("[SKIP] %s t=%.2f（窗越界/退化的截断事件）" % (key, r["t_on"]))
            continue
        eid = "%s@%.2f" % (key, r["t_on"])
        others = ev[(ev["key"] == key)]["t_on"].to_numpy(float)
        pre_ok = not ((others > r["t_on"] - 8.0) & (others < r["t_on"] - 1.0)).any()
        post_ok = not ((others > r["t_on"] + 1.5) & (others < r["t_on"] + 8.5)).any()
        mpre = (g["tu"] >= float(r["t_on"]) - 2.0) & (g["tu"] < float(r["t_on"]))
        pa = g["zu"][mpre]
        nsig = 1.4826 * float(np.median(np.abs(pa - np.median(pa)))) if mpre.sum() > 20 \
            else np.nan
        amp_min = max(0.02 * g["zpeak"], 5.0 * nsig) if np.isfinite(nsig) \
            else 0.02 * g["zpeak"]
        for tag, sh in (("m1", -g["pkt"]), ("p1", +g["pkt"])):
            mm = event_metrics(g["zu"], g["tu"], float(r["t_on"]) + sh)
            if mm is not None:
                m["t90_%s" % tag] = mm["t90"]
                m["t50_%s" % tag] = mm["t50"]
        rows.append(dict(
            event_id=eid, key=key, domain=DOMAIN[key], rec=r["rec"],
            t_on=float(r["t_on"]), kind=r["kind"], clean=bool(r["clean"]),
            pre_win_clean=bool(pre_ok), post_win_clean=bool(post_ok),
            noise_sigma=nsig, z_peak=g["zpeak"], amp_min=amp_min,
            amp_ok=bool(m["absJ"] >= amp_min),
            snr=(m["absJ"] / nsig) if (np.isfinite(nsig) and nsig > 1e-12) else np.nan,
            pre=m["pre"], post=m["post"], J=m["J"], absJ=m["absJ"],
            jump_t4a=r.get("jump"), t90_t4a=r.get("t90"), t50_t4a=r.get("t50"),
            T_ramp=r.get("T_ramp"), gain_ramp=r.get("gain_ramp"),
            rmse_ramp_pct=r.get("rmse_ramp_pct"), rmse_step_pct=r.get("rmse_step_pct"),
            t10=m["t10"], t50=m["t50"], t80=m["t80"], t90=m["t90"], t95=m["t95"],
            t50_m1=m.get("t50_m1"), t90_m1=m.get("t90_m1"),
            t50_p1=m.get("t50_p1"), t90_p1=m.get("t90_p1"),
            step_frame_frac=m["step_frame_frac"], top3_frame_frac=m["top3_frame_frac"],
            overshoot_pct=m["overshoot_pct"], undershoot_pct=m["undershoot_pct"],
            pkt_dt=g["pkt"], drop_frac_meas=abs(m["J"]) / max(abs(m["pre"]), 1.0),
            f_005=m["f_grid"][IDX[0.05]], f_01=m["f_grid"][IDX[0.10]],
            f_02=m["f_grid"][IDX[0.20]], f_05=m["f_grid"][IDX[0.50]],
            f_10=m["f_grid"][IDX[1.00]],
        ))
        for tau, f in zip(TAU_GRID, m["f_grid"]):
            shape_rows.append(dict(event_id=eid, key=key, kind=r["kind"], tau=tau, f=f,
                                   absJ=m["absJ"], src="t4a_frozen"))
    met = pd.DataFrame(rows)
    met.to_csv(os.path.join(RESULTS, "t7b_events_metrics.csv"), index=False,
               encoding="utf-8-sig")
    shp = pd.DataFrame(shape_rows)
    shp.to_csv(os.path.join(RESULTS, "t7b_shape_long.csv"), index=False,
               encoding="utf-8-sig")

    print("\n[1] 冻结事件统一口径指标（中位；含 60 事件中 1 个截断事件的剔除）")
    print(met.groupby("kind")[["absJ", "t50", "t90", "f_02", "step_frame_frac",
                               "top3_frame_frac"]].median().to_string(float_format="%.3f"))
    ck = met.dropna(subset=["t90_t4a", "t90"])
    ck = ck.assign(diff=(ck["t90"] - ck["t90_t4a"]).abs())
    print("  口径交叉核对（T4-A t90 vs T7-B t90）：n=%d，|差| 中位 %.3f s，最大 %.3f s"
          % (len(ck), ck["diff"].median(), ck["diff"].max()))
    print("    分域最大 |Δ|：%s" % ck.groupby("domain")["diff"].max().to_dict())
    print("    分歧 >0.05 s 的 %d 个（全部落在 ADC 域的加载侧）" % int((ck["diff"] > 0.05).sum()))

    # ── 自建减重扫描 ──
    inv = []
    for key in RECS:
        g = grid_of(key)
        inv += decrement_inventory(key, g["zu"], g["tu"])
    invdf = pd.DataFrame(inv)
    invdf["domain"] = invdf["key"].map(DOMAIN)
    invdf["drop_frac"] = invdf["absJ"] / np.maximum(invdf["pre"].abs(), 1.0)
    invdf = invdf[["key", "domain", "t_on", "pre", "post", "J", "absJ", "drop_frac",
                   "noise_sigma", "amp_min", "t50", "t80", "t90", "t95",
                   "step_frame_frac", "top3_frame_frac", "overshoot_pct",
                   "undershoot_pct"]]
    invdf.to_csv(os.path.join(RESULTS, "t7b_decrement_inventory.csv"), index=False,
                 encoding="utf-8-sig")
    frozen_dec = met[met["kind"].isin(["unload", "partial_unload"])]
    found = sum(1 for _, e in frozen_dec.iterrows()
                if ((invdf["key"] == e["key"]) & ((invdf["t_on"] - e["t_on"]).abs() <= 1.0)).any())
    print("\n[2] 自建减重扫描：n=%d（drop_frac≥0.8 的 %d、0.2~0.8 的 %d、<0.2 的 %d）"
          % (len(invdf), int((invdf.drop_frac >= 0.8).sum()),
             int(((invdf.drop_frac >= 0.2) & (invdf.drop_frac < 0.8)).sum()),
             int((invdf.drop_frac < 0.2).sum())))
    print("    对冻结表减重事件的召回：%d/%d（漏检：%s）"
          % (found, len(frozen_dec),
             [e["event_id"] for _, e in frozen_dec.iterrows()
              if not ((invdf["key"] == e["key"]) &
                      ((invdf["t_on"] - e["t_on"]).abs() <= 1.0)).any()]))
    # 漏检诊断：给出漏检事件 pre/post 窗的 MAD 与门限（说明是门限还是检出问题）
    for _, e in frozen_dec.iterrows():
        if ((invdf["key"] == e["key"]) & ((invdf["t_on"] - e["t_on"]).abs() <= 1.0)).any():
            continue
        g = grid_of(e["key"])
        t0 = float(e["t_on"])
        a = (g["tu"] >= t0 - 2.0) & (g["tu"] < t0)
        b = (g["tu"] >= t0 + 4.0) & (g["tu"] < t0 + 6.0)
        pa, pb = g["zu"][a], g["zu"][b]
        ma, mb = float(np.median(np.abs(pa - np.median(pa)))), \
            float(np.median(np.abs(pb - np.median(pb))))
        J = float(np.median(pb) - np.median(pa))
        tol = max(0.02 * max(abs(np.median(pa)), abs(np.median(pb))), 0.05 * abs(J))
        print("      漏检诊断 %s：pre MAD=%.0f / post MAD=%.0f，稳定门限 tol=%.0f ⇒ %s"
              % (e["event_id"], ma, mb, tol,
                 "被稳定门限剔除" if max(ma, mb) > tol else "其它原因"))

    # ── 统一减重事件表（冻结 19 + 扫描补充），自适应 post 窗 ──
    dec_rows = []
    for _, r in frozen_dec.iterrows():
        dec_rows.append(dict(event_id=r["event_id"], key=r["key"], t_on=float(r["t_on"]),
                             kind_t4a=r["kind"], src="t4a_frozen",
                             drop_frac_frozen=r["drop_frac_meas"],
                             post_win_clean=bool(r["post_win_clean"]),
                             amp_ok=bool(r["amp_ok"]), pkt_dt=r["pkt_dt"],
                             t90_frozen=r["t90"], step_frame_frac=r["step_frame_frac"]))
    for _, r in invdf.iterrows():
        if ((frozen_dec["key"] == r["key"]) &
                ((frozen_dec["t_on"] - r["t_on"]).abs() <= 1.0)).any():
            continue
        dec_rows.append(dict(event_id="%s@%.2f(scan)" % (r["key"], r["t_on"]), key=r["key"],
                             t_on=float(r["t_on"]), kind_t4a="scan_only", src="t7b_scan",
                             drop_frac_frozen=r["drop_frac"], post_win_clean=False,
                             amp_ok=True, pkt_dt=grid_of(r["key"])["pkt"],
                             t90_frozen=np.nan, step_frame_frac=r["step_frame_frac"]))
    dec = pd.DataFrame(dec_rows)
    # post 窗污染只看"后续是否有任何事件"（加载或卸载都算），故用冻结全表 + 扫描事件
    all_ev = {}
    for _, r in met.iterrows():
        all_ev.setdefault(r["key"], []).append(float(r["t_on"]))
    for _, r in invdf.iterrows():
        all_ev.setdefault(r["key"], []).append(float(r["t_on"]))
    for k in all_ev:
        all_ev[k] = np.sort(np.array(all_ev[k]))
    tnext = []
    for _, r in dec.iterrows():
        cand = all_ev.get(r["key"], np.array([]))
        cand = cand[cand > r["t_on"] + 0.2]
        tnext.append(float(cand.min()) if len(cand) else np.nan)
    dec["t_next"] = tnext
    dec["domain"] = dec["key"].map(DOMAIN)

    out_rows, prof_rows = [], []
    for _, r in dec.iterrows():
        g = grid_of(r["key"])
        wa, wb, wtag = adaptive_post_win(float(r["t_on"]), r["t_next"])
        m = event_metrics(g["zu"], g["tu"], float(r["t_on"]), post_win=(wa, wb))
        if m is None:
            continue
        drop_eff = m["absJ"] / max(abs(m["pre"]), 1.0)
        kind_eff = classify(drop_eff)
        for tag, sh in (("m1", -g["pkt"]), ("p1", +g["pkt"])):
            mm = event_metrics(g["zu"], g["tu"], float(r["t_on"]) + sh, post_win=(wa, wb))
            if mm is not None:
                m["t90_%s" % tag] = mm["t90"]
                m["t50_%s" % tag] = mm["t50"]
        lo = float(r["t_on"]) + wa
        hi = float(r["t_on"]) + wb
        seg = g["zu"][int(round(float(r["t_on"]) * FS)):int(round(hi * FS)) + 1]
        out_rows.append(dict(
            event_id=r["event_id"], key=r["key"], domain=r["domain"], t_on=float(r["t_on"]),
            kind_t4a=r["kind_t4a"], kind_eff=kind_eff, src=r["src"],
            post_win_clean=r["post_win_clean"], post_win="[t0+%.1f,t0+%.1f]" % (wa, wb),
            post_win_kind=wtag, t_next=r["t_next"], amp_ok=r["amp_ok"],
            pre_ad=m["pre"], post_ad=m["post"], absJ_ad=m["absJ"], drop_frac_eff=drop_eff,
            drop_frac_frozen=r["drop_frac_frozen"],
            t50_ad=m["t50"], t80_ad=m["t80"], t90_ad=m["t90"], t95_ad=m["t95"],
            t90_frozen=r["t90_frozen"],
            t50_ad_m1=m.get("t50_m1"), t90_ad_m1=m.get("t90_m1"),
            t50_ad_p1=m.get("t50_p1"), t90_ad_p1=m.get("t90_p1"),
            f_005_ad=m["f_grid"][IDX[0.05]], f_01_ad=m["f_grid"][IDX[0.10]],
            f_02_ad=m["f_grid"][IDX[0.20]], f_05_ad=m["f_grid"][IDX[0.50]],
            f_10_ad=m["f_grid"][IDX[1.00]],
            rebound_02_pct=100.0 * (m["f_grid"][IDX[0.20]] - 1.0),
            rebound_05_pct=100.0 * (m["f_grid"][IDX[0.50]] - 1.0),
            dip_vs_post_pct=100.0 * (m["post"] - float(np.min(seg))) / max(m["absJ"], 1)
            if len(seg) else np.nan,
            step_frame_frac=m["step_frame_frac"], top3_frame_frac=m["top3_frame_frac"],
            pkt_dt=g["pkt"],
        ))
        for tau, f in zip(TAU_GRID, m["f_grid"]):
            prof_rows.append(dict(event_id=r["event_id"], key=r["key"], tau=tau, f=f,
                                  absJ=m["absJ"], kind_eff=kind_eff, src=r["src"],
                                  domain=r["domain"], drop_frac_eff=drop_eff))
    ampr = pd.DataFrame(out_rows)
    ampr.to_csv(os.path.join(RESULTS, "t7b_partial_amp_response.csv"), index=False,
                encoding="utf-8-sig")
    pro = pd.DataFrame(prof_rows)
    pro.to_csv(os.path.join(RESULTS, "t7b_partial_shape.csv"), index=False,
               encoding="utf-8-sig")

    print("\n[3] 减重事件实测重新分档（00_共享/指标字典 §2.1：<20% / 20~80% / ≥80%）")
    conf = ampr.groupby(["kind_t4a", "kind_eff"]).size().unstack(fill_value=0)
    print(conf.to_string())
    recl = ampr[(ampr["kind_t4a"] == "partial_unload") & (ampr["kind_eff"] != "partial_unload")]
    for _, r in recl.iterrows():
        print("    ⚠ 重分类：T4-A 标 %s@%.2f 为 partial_unload，实测幅度 %.1f%% 电平 ⇒ %s"
              "（冻结 post 窗被后续事件压到：窗= %s，自适应窗= %s）"
              % (r["key"], r["t_on"], 100 * r["drop_frac_eff"], r["kind_eff"],
                 "污染" if not r["post_win_clean"] else "干净", r["post_win"]))
    print("\n    分档计数：%s" % ampr["kind_eff"].value_counts().to_dict())

    print("\n[4] 减重幅度 → 回复速度（Q6；含 ±1 包敏感性）")
    print(ampr[["event_id", "kind_eff", "drop_frac_eff", "absJ_ad", "post_win",
                "t50_ad", "t90_ad", "t90_ad_m1", "t90_ad_p1", "f_02_ad",
                "rebound_02_pct", "step_frame_frac"]]
          .to_string(index=False, float_format="%.3f"))
    r2 = ampr.dropna(subset=["drop_frac_eff", "t90_ad"])
    if len(r2) >= 4:
        print("  Spearman(drop_frac_eff, t90_ad) = %.3f (n=%d)"
              % (r2["drop_frac_eff"].corr(r2["t90_ad"], method="spearman"), len(r2)))
        nb = r2[r2["kind_eff"] == "unload"]
        print("  仅全卸载（n=%d）：t90 中位 %.3f s，f_02 中位 %.3f"
              % (len(nb), nb["t90_ad"].median(), nb["f_02_ad"].median()))
        np_ = r2[r2["kind_eff"].isin(["partial_unload", "decrement_small"])]
        print("  非全卸载（n=%d）：t90 中位 %.3f s，f_02 中位 %.3f（%s）"
              % (len(np_), np_["t90_ad"].median(), np_["f_02_ad"].median(),
                 "仅作定性参考" if len(np_) <= 3 else ""))

    # ── 分组轮廓汇总 ──
    out = []
    for (grp, src), gg in pro.groupby(["kind_eff", "src"]):
        for tau in TAU_GRID:
            v = gg[gg["tau"] == tau]["f"].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v):
                out.append(dict(group=grp, src=src, n=len(v), tau=tau,
                                f_med=float(np.median(v)),
                                f_p10=float(np.percentile(v, 10)),
                                f_p90=float(np.percentile(v, 90))))
    for grp in ("onset", "restep"):
        gg = shp[shp["kind"] == grp]
        for tau in TAU_GRID:
            v = gg[gg["tau"] == tau]["f"].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v):
                out.append(dict(group=grp, src="t4a_frozen", n=len(v), tau=tau,
                                f_med=float(np.median(v)),
                                f_p10=float(np.percentile(v, 10)),
                                f_p90=float(np.percentile(v, 90))))
    smm = pd.DataFrame(out)
    smm.to_csv(os.path.join(RESULTS, "t7b_shape_summary.csv"), index=False,
               encoding="utf-8-sig")
    print("\n[5] 分组轮廓（f 中位 @ τ）")
    piv = smm.pivot_table(index=["group", "src"], columns="tau", values="f_med")
    ncol = smm.groupby(["group", "src"])["n"].max().rename("n")
    show = [c for c in (0.05, 0.1, 0.2, 0.5, 1.0, 3.0, 5.0) if c in piv.columns]
    print(pd.concat([ncol, piv[show]], axis=1).to_string(float_format="%.3f"))
    print("\nWROTE results/t7b_events_metrics.csv, t7b_shape_long.csv, t7b_shape_summary.csv,")
    print("      results/t7b_decrement_inventory.csv, t7b_partial_amp_response.csv,")
    print("      results/t7b_partial_shape.csv")


if __name__ == "__main__":
    tee = Tee(LOG)
    _o = sys.stdout
    sys.stdout = tee
    try:
        main()
    finally:
        sys.stdout = _o
        tee.flush()
        tee.f.close()
