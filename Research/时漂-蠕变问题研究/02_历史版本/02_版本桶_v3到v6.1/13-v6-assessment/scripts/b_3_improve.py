# -*- coding: utf-8 -*-
"""b3：改进空间的前后对比（问题 3）。仅跑 9 份恒载（可重复性口径），变体都用 v6 原型派生。

变体（每个只改 1~2 个既有开关，不改算法结构）：
  P1 unload_fast : 卸载沿不再「抱死扣除」——检出减重且新电平已接近空载时直接退出补偿态
                   （新增 _reanchor 覆写 + DECREASE_SETTLE=0）
  P2 idle_guard  : 空载 onset 门限从 10%·历史最大 提到 25%，静默期 0.5→2.0 s（少建假 epoch）
  P3 trim_1pct   : A 慢修正死区 2.5%→1.0%、速率 0.2%→0.4%/s（把「绝对准」压上去，看「绝对平」代价）
  P4 anchor_meas : ANCHOR_MODE pin→measured（交接时把 A 钉到实测电平，平台精度不再继承形状先验）

产出：results/b_improve_records.csv / b_improve_summary.csv / b_improve_unload.csv / _b3_improve.log
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


class P1(B.PR.V6Trace):
    """卸载沿不再抱死扣除。"""
    DECREASE_SETTLE = 0.0

    def _reanchor(self, ts, v):
        vs = self._mat_mean(self._bvx, ts - self.REANCHOR_SMOOTH_S, ts)
        a_old = float(self.A.max()) if self.A.size else 0.0
        new = float(np.max(vs)) if vs is not None else float(np.max(v))
        if a_old > 1e-9 and new < 0.15 * a_old:
            self.hold_comp = None
            self._to_idle()
            return
        super()._reanchor(ts, v)


class P2(B.PR.V6Trace):
    """空载 onset 门限提高 + 静默期加长。"""
    DET_IDLE_FRAC = 0.25
    IDLE_SETTLE = 2.0


class P3(B.PR.V6Trace):
    """A 慢修正死区收紧。"""
    TRIM_RATE = 0.004
    TRIM_DEAD_FRAC = 0.010


class P4(B.PR.V6Trace):
    """交接锚定改成实测电平。"""
    ANCHOR_MODE = "measured"


class P1b(B.PR.V6Trace):
    """P1 修正版：卸载沿一旦判出「负载已去掉 >35%·事件前电平」立刻退出补偿态。"""
    DECREASE_SETTLE = 0.0

    def _reanchor(self, ts, v):
        vs = self._mat_mean(self._bvx, ts - self.REANCHOR_SMOOTH_S, ts)
        a_old = float(self.A.max()) if self.A.size else 0.0
        ev = self.ev
        ref = float(np.max(ev["v0"])) if (ev is not None and "v0" in ev) else a_old
        new = float(np.max(vs)) if vs is not None else float(np.max(v))
        if a_old > 1e-9 and ref > 1e-9 and (ref - new) / ref > 0.35:
            self.hold_comp = None
            self._to_idle()
            return
        super()._reanchor(ts, v)


VARIANTS = [("v6（基线）", B.PR.V6Trace), ("P1 卸载不抱死", P1), ("P2 空载门限+静默", P2),
            ("P3 trim 1%死区", P3), ("P4 A=实测电平", P4)]


def variants_for(filter_key):
    """命令行开关：p1b 只跑 基线 + P1b，写到 b_improve_p1b_*。"""
    if filter_key == "p1b":
        return [("v6（基线）", B.PR.V6Trace), ("P1b 卸载即刻回原始", P1b)], "b_improve_p1b"
    return VARIANTS, "b_improve"


def metrics_from_display(tu, dis, i_e, inc5, pre, fall_idx=None, drop=None):
    ds = B.med(dis, tu[1] - tu[0], 0.5)
    t_e = tu[i_e]
    tgt5 = pre + inc5
    lvl = float(np.median(ds[(tu >= t_e + 40.0) & (tu <= t_e + 60.0)]))
    w = (tu >= t_e + 30.0) & (tu <= t_e + 60.0)
    flat = float(np.max(ds[w]) - np.min(ds[w]))
    mv = float(np.median(ds[(tu >= t_e + 50.0) & (tu <= t_e + 60.0)])) - \
        float(np.median(ds[(tu >= t_e + 30.0) & (tu <= t_e + 40.0)]))
    win = (tu >= t_e + 5.0) & (tu <= tu[-1])
    maxdev = float(np.max(np.abs(ds[win] - tgt5)))
    seg = (tu >= t_e + 4.0) & (tu <= t_e + 12.0)
    jump = float(np.max(np.abs(np.diff(ds[seg])))) if seg.sum() > 2 else np.nan
    out = dict(plat_lvl=lvl, err_plat5_pct=100.0 * ((lvl - pre) / inc5 - 1.0),
               flat5_pct=100.0 * flat / abs(inc5), move3060_pct=100.0 * mv / abs(inc5),
               max_dev5_pct=100.0 * maxdev / abs(inc5),
               handoff_jump_pct=100.0 * jump / abs(inc5))
    if fall_idx is not None and drop:
        j0 = fall_idx + 2
        j1 = min(len(tu) - 1, fall_idx + int(2.0 / (tu[1] - tu[0])))
        post = float(np.median(ds[(tu >= tu[fall_idx] + 2.0) & (tu <= tu[fall_idx] + 18.0)]))
        out["dip"] = post - float(np.min(ds[j0:j1]))
        out["dip_pct"] = 100.0 * out["dip"] / abs(drop)
        out["pin0_s"] = float(np.sum(ds[j0:j1] <= 0.002 * abs(drop)) * (tu[1] - tu[0]))
    return out


def main():
    B.log_reconfigure()
    VARIANTS_USE, prefix = variants_for(sys.argv[1] if len(sys.argv) > 1 else None)
    base = pd.read_csv(os.path.join(B.RES, "b_repeat_records.csv"))
    rows, ul_rows = [], []
    for tag, path in B.ALL:
        if tag not in HOLD:
            continue
        d = B.load_csv(tag, path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
        tot = Xu.sum(axis=1)
        ts1 = B.med(tot, dt, 0.1)
        fo = B.C.first_onset(tu, tot, dt)
        i_e, pre = fo[0], fo[1]
        inc5 = float(ts1[min(len(tu) - 1, i_e + int(5.0 / dt))] - pre)
        evs, _, _, _ = B.unload_events(d)
        ev = evs[0] if evs else None
        grp = GROUP[tag.split("/")[0]]
        p(f"[{tag}] inc5={inc5:.2f} 卸载沿 t={ev['t_dn']:.1f} 正在跑 5 个变体")
        for lbl, cls in VARIANTS_USE:
            vdis = None
            if cls is None:                       # （保留的缓存分支，当前未启用）
                z = np.load(B.cache_path(tag), allow_pickle=True)
                tuf = z["tu"]
                vdis = z["Yraw_v6"].sum(axis=1)
                k = int(np.searchsorted(tuf, tu[i_e]))
                n_ep = len([s for s in z["kindlog_v6"] if "|" in str(s)])
                n_pre = len([s for s in z["kindlog_v6"]
                             if "|" in str(s) and float(str(s).split("|")[0]) < tu[i_e] - 0.3])
                mm = metrics_from_display(tuf, vdis, k, inc5, pre)
                mm["t_stable"] = float(base[(base.rec == tag)
                                            & (base.arm == "v6")].t_stable.iloc[0])
            else:
                c = cls(Xu.shape[1])
                Y = np.empty_like(Xu)
                for i in range(len(tu)):
                    Y[i] = c.process(tu[i], Xu[i])
                vdis = Y.sum(axis=1)
                n_ep = len(c.epoch_t)
                n_pre = sum(1 for t in c.epoch_t if t < tu[i_e] - 0.3)
                mm = metrics_from_display(tu, vdis, i_e, inc5, pre)
                mm["t_stable"] = B.C.stable_time(tu, Y.sum(axis=1), i_e, inc5, dt)
            mm.update(rec=tag, group=grp, variant=lbl, inc5=inc5, epoch_total=n_ep,
                      epoch_pre=n_pre)
            rows.append(mm)
            if ev is not None and cls is not None:      # 卸载沿：同一口径算下冲/贴零
                k0 = ev["i_fall"]
                j0, j1 = k0 + 2, min(len(tu) - 1, k0 + int(2.0 / dt))
                post = float(np.median(vdis[(tu >= ev["t_dn"] + 2.0)
                                            & (tu <= min(ev["post_end"], ev["t_dn"] + 18.0))]))
                ul_rows.append(dict(rec=tag, variant=lbl, t_dn=ev["t_dn"], drop=ev["drop"],
                                    post=post, min_display=float(np.min(vdis[j0:j1])),
                                    dip=post - float(np.min(vdis[j0:j1])),
                                    dip_pct=100.0 * (post - float(np.min(vdis[j0:j1])))
                                    / abs(ev["drop"]),
                                    pin0_s=float(np.sum(vdis[j0:j1] <= 0.002 * abs(ev["drop"]))
                                                 * dt)))
    rec = pd.DataFrame(rows)
    rec.to_csv(os.path.join(B.RES, prefix + "_records.csv"), index=False, encoding="utf-8-sig")
    ul = pd.DataFrame(ul_rows)
    ul.to_csv(os.path.join(B.RES, prefix + "_unload_var.csv"), index=False, encoding="utf-8-sig")
    # 基线卸载沿（原始 + v6）
    rawul = pd.read_csv(os.path.join(B.RES, "b_unload_raw.csv"))
    ev6 = pd.read_csv(os.path.join(B.RES, "b_unload_events.csv"))
    for tag in HOLD:
        r = rawul[rawul.rec == tag]
        if len(r):
            ul_rows.append(dict(rec=tag, variant="raw（无补偿）", t_dn=float(r["t_dn"].iloc[0]),
                                drop=float(r["drop"].iloc[0]), dip=float(r["dip_post"].iloc[0]),
                                dip_pct=float(r["dip_post_pct"].iloc[0]), pin0_s=np.nan))
        v = ev6[(ev6.rec == tag) & (ev6.arm == "v6")]
        if len(v):
            ul_rows.append(dict(rec=tag, variant="v6（基线）", t_dn=float(v.t_dn.iloc[0]),
                                drop=float(v.drop_ev.iloc[0]), dip=float(v.dip_vs_post.iloc[0]),
                                dip_pct=float(v.dip_vs_post.iloc[0]) / abs(float(v.drop_ev.iloc[0])) * 100,
                                pin0_s=float(v.pin0_s.iloc[0])))
    ul = pd.DataFrame(ul_rows)
    ul.to_csv(os.path.join(B.RES, prefix + "_unload.csv"), index=False, encoding="utf-8-sig")

    mets = [("err_plat5_pct", "平台误差@40-60s(%)"), ("plat_lvl", "平台绝对电平"),
            ("flat5_pct", "平台内极差(%)"), ("move3060_pct", "平台 30→60s 移动(%)"),
            ("max_dev5_pct", "全程最大偏差(%)"), ("handoff_jump_pct", "交接处最大单帧外露(%)"),
            ("t_stable", "T_stable(s)"), ("epoch_total", "epoch 总数"),
            ("epoch_pre", "真沿前 epoch 数")]
    srows = []
    for (grp, var), g in rec.groupby(["group", "variant"]):
        for key, lbl in mets:
            v = g[key].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v):
                srows.append(dict(group=grp, variant=var, metric=key, label=lbl, n=len(v),
                                  mean=float(np.mean(v)), std=float(np.std(v, ddof=1)),
                                  rng=float(np.max(v) - np.min(v)),
                                  values="|".join(f"{x:.2f}" for x in v)))
    sm = pd.DataFrame(srows)
    sm.to_csv(os.path.join(B.RES, prefix + "_summary.csv"), index=False, encoding="utf-8-sig")
    agg = sm.groupby(["variant", "metric", "label"]).agg(
        std_med=("std", "median"), rng_med=("rng", "median"),
        mean_med=("mean", "median")).reset_index()

    p("")
    p("=" * 120)
    p("表 1  逐录制明细（恒载 9 组）")
    p("=" * 120)
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        p(rec[["rec", "variant", "inc5", "plat_lvl", "err_plat5_pct", "flat5_pct",
               "move3060_pct", "max_dev5_pct", "handoff_jump_pct", "t_stable", "epoch_total",
               "epoch_pre"]].round(3).to_string(index=False))
    p("")
    p("=" * 120)
    p("表 2  变体汇总：组内离散度（跨 3 个传感器组取中位）与均值")
    p("=" * 120)
    for key, lbl in mets:
        sub = agg[agg.metric == key].set_index("variant")
        sub = sub.reindex([v for v, _ in VARIANTS_USE])
        p(f"  [{lbl}]   std中位 / 均值中位")
        with pd.option_context("display.width", 200):
            p(pd.DataFrame({"std_med": sub.std_med, "mean_med": sub.mean_med}).round(3).to_string())
        p("")
    p("=" * 120)
    p("表 3  卸载沿下冲（恒载 9 组的卸载事件；dip 相对卸载后稳态）")
    p("=" * 120)
    ug = ul.groupby("variant").agg(n=("dip_pct", "size"), dip_pct_med=("dip_pct", "median"),
                                   dip_pct_max=("dip_pct", "max"),
                                   pin0_med=("pin0_s", "median"),
                                   pin0_max=("pin0_s", "max")).round(3)
    with pd.option_context("display.width", 200):
        p(ug.to_string())
    with open(os.path.join(B.RES, "_b3_" + prefix + ".log"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    print("\n-> results/b_improve_records.csv / b_improve_summary.csv / b_improve_unload.csv / "
          "_b3_" + prefix + ".log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
