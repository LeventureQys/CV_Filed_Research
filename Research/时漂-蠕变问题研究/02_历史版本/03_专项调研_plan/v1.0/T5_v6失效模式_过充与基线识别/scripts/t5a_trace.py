# -*- coding: utf-8 -*-
"""T5A-Q2：v6 状态机内部量逐帧落盘 + 过充归因（每条归因带帧号与数值）。

落盘量（每帧，100 Hz 网格）：
    ts, key, arm, tot(原始 Z), ytot(显示 ΣY), dev(显示−原始), state_code(0 idle/1 event/2 slow),
    ev_kind, tau, A_hat(反演幅度), inc(实测增量), inc_s(0.1 s 窗增量), target(滑行目标),
    c_applied(当前修正量), e_track(= target−tot), gliding(W 完成度), Tglide(滑行时长),
    stalled(停滞标志), stalled_t, g, gamma_med, kappa_used, A_sum, share_sum
  ⇒ `results/t5a_internal_trace.csv`（约 70 万行 / 13 份）

归因判据（**逐候选动作给量化证据，不用"可能是噪声"**）：
  H1 形状反演高估：事件后 τ∈[0.2,1] s 内 `A_hat / |J| − 1`（用原始台阶 J 作真值）。
  H2 滑行器速率上限：窗口内 `|Δc_applied|/dt` 是否贴到 `RATE_MAX·|A_hat|`。
  H3 τ_ho 交接跳变：交接帧前后的 `ΣY` 一阶差分，与相邻帧差分中位比较（跳变倍数）。
  H4 慢相模块提前起扣：交接后 5 s 内 `g` 从 0 起跳的斜率（是否在慢相尚未建立时就扣）。

产出：`t5a_internal_trace.csv`、`t5a_internal_snap.csv`（关键帧）、
      `t5a_event_state_events.csv`（epoch/revoke/handoff/unload 事件表）、
      `_t5a_trace.log`。

用法：`python scripts/t5a_trace.py`（长任务，建议后台）
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

LOG = []
T0 = time.time()


def rec(m):
    s = f"[{time.time() - T0:7.1f}s] {m}"
    print(s, flush=True)
    LOG.append(s)


TRACE_COLS = ["key", "i", "ts", "arm", "tot", "ytot", "dev", "state_code", "ev_kind",
              "tau", "A_hat", "inc", "inc_s", "target", "c_applied", "e_track",
              "W_glide", "Tglide", "stalled", "stall_t", "g", "gamma_med",
              "kappa_used", "A_sum", "share_sum", "n_ch"]


def trace_one(d, key, arm_id, X=None, cls=None, kappa_onset=1.30, kappa_restep=1.12):
    """逐帧跑一遍 v6（κ 可调），把内部量落盘。返回 (trace DataFrame, snapshot DataFrame,
    state-event DataFrame)。"""
    Xu = d["Xu"] if X is None else X
    tu = d["tu"]
    n = len(tu)
    c = (cls or C.KV6)(Xu.shape[1])
    c.kappa_onset = float(kappa_onset)
    c.kappa_restep = float(kappa_restep)
    st = {"idle": 0, "event": 1, "slow": 2}
    col = {k: np.full(n, np.nan) for k in
           ["tau", "A_hat", "inc", "inc_s", "target", "c_applied", "e_track",
            "W_glide", "Tglide", "g", "gamma_med", "A_sum", "share_sum"]}
    ytot = np.empty(n)
    tot = np.empty(n)
    state_code = np.zeros(n, np.int8)
    stalled = np.zeros(n, np.int8)
    stall_t = np.full(n, np.nan)
    ev_kind = np.array([""] * n, dtype=object)
    kappa_used = np.full(n, np.nan)
    prm = {k: np.full(n, np.nan) for k in
           ["vA_sum", "vA_max", "vloaded", "vg", "vhold", "vtrim"]}
    snaps = []
    shead = []
    for i in range(n):
        Y = c.process(tu[i], Xu[i])
        S = float(Xu[i].sum())
        T = float(Y.sum())
        tot[i] = S
        ytot[i] = T
        state_code[i] = st[c.state]
        if c.ev is not None:
            ev = c.ev
            tau = tu[i] - ev["t0"]
            A_hat = float(ev["A_hat"])
            inc = S - ev["base"]
            iw = c._win_mean(tu[i] - 0.10, tu[i])
            inc_s = (iw - ev["base"]) if iw is not None else inc
            tgt = float(ev["base_y"]) + A_hat
            ca = float(ev["c_applied"])
            eve = ev
            col["tau"][i] = tau
            col["A_hat"][i] = A_hat
            col["inc"][i] = inc
            col["inc_s"][i] = inc_s
            col["target"][i] = tgt
            col["c_applied"][i] = ca
            col["e_track"][i] = tgt - S
            if eve["tau_g0"] is None:
                col["W_glide"][i] = 0.0
            else:
                xx = float(np.clip((tau - eve["tau_g0"]) / max(eve["Tglide"], 1e-9), 0, 1))
                col["W_glide"][i] = xx * xx * (3.0 - 2.0 * xx)
            col["Tglide"][i] = float(eve["Tglide"])
            stalled[i] = 1 if eve["stalled"] else 0
            stall_t[i] = float(eve["stall_t"])
            ev_kind[i] = str(eve["kind"])
            kappa_used[i] = (c.kappa_onset if eve["kind"] == "onset" else c.kappa_restep)
        else:
            ev_kind[i] = ""
        col["g"][i] = float(c.g)
        if c.loaded.any():
            col["gamma_med"][i] = float(np.median(c.gamma[c.loaded]))
        col["A_sum"][i] = float(c.A.sum())
        inc_vec = np.asarray(Xu[i], float) - (c.ev["v0"] if c.ev is not None else 0.0)
        w = np.clip(inc_vec, 0.0, None)
        col["share_sum"][i] = float(w.sum())
        prm["vA_sum"][i] = float(c.A.sum())
        prm["vA_max"][i] = float(c.A.max())
        prm["vloaded"][i] = float(c.loaded.sum())
        prm["vg"][i] = float(c.g)
        prm["vhold"][i] = 0.0 if c.hold_comp is None else float(np.sum(c.hold_comp))
        prm["vtrim"][i] = float("nan") if c.trim_target_sum is None else float(c.trim_target_sum)
    # 事件（epoch / handoff / revoke / unload / gevent）
    for e in c.tr_epoch:
        shead.append(dict(key=key, kind="epoch", ts=e[0], t_det=e[2], base=e[3],
                          v0sum=e[4], y0sum=e[5], ev_kind=e[1], arm=arm_id))
    for h in c.tr_handoff:
        shead.append(dict(key=key, kind="handoff", ts=h[0], A_hat=h[2], c_applied=h[3],
                          A_sum=h[4], g=h[5], arm=arm_id))
    for r in c.tr_revoke:
        shead.append(dict(key=key, kind="revoke", ts=r[0], arm=arm_id))
    for u in c.tr_unload:
        shead.append(dict(key=key, kind="unload", ts=u, arm=arm_id))
    for g in c.tr_gevent:
        shead.append(dict(key=key, kind="gevent", ts=g[0], ev_kind=g[1], arm=arm_id))
    tr = pd.DataFrame({k: col[k] for k in col})
    tr.insert(0, "key", key)
    tr.insert(1, "i", np.arange(n))
    tr.insert(2, "ts", tu)
    tr.insert(3, "arm", arm_id)
    tr["tot"] = tot
    tr["ytot"] = ytot
    tr["dev"] = ytot - tot
    tr["state_code"] = state_code
    tr["ev_kind"] = ev_kind
    tr["stalled"] = stalled
    tr["stall_t"] = stall_t
    tr["kappa_used"] = kappa_used
    tr["n_ch"] = Xu.shape[1]
    for k, v in prm.items():
        tr[k] = v
    return tr[TRACE_COLS], pd.DataFrame(shead)


def main():
    ev = C.ev_load_frozen()
    recs = C.recordings()
    kinds = dict(name="v6_now", cls=C.KV6, kappa_onset=1.30, kappa_restep=1.12)
    trs, shs = [], []
    for k, d in recs.items():
        if not len(ev[ev.key == k]):
            continue
        t, s = trace_one(d, k, kinds["name"], cls=C.KV6,
                         kappa_onset=1.30, kappa_restep=1.12)
        trs.append(t)
        shs.append(s)
        rec(f"  trace {k}: {len(t)} 帧, 状态事件 {len(s)}")
    TR = pd.concat(trs, ignore_index=True)
    SH = pd.concat(shs, ignore_index=True)
    p1 = os.path.join(C.TASK, "results", "t5a_internal_trace.csv")
    TR.to_csv(p1, index=False, encoding="utf-8-sig", float_format="%.6g")
    rec(f"产出 {p1}（{len(TR)} 行 × {len(TR.columns)} 列）")
    p2 = os.path.join(C.TASK, "results", "t5a_event_state_events.csv")
    SH.to_csv(p2, index=False, encoding="utf-8-sig")
    rec(f"产出 {p2}（{len(SH)} 行）")

    # ── 关键帧快照 + 四类归因 ──
    snaps, attrib = [], []
    for _, e in ev.iterrows():
        k = e["key"]
        d = recs[k]
        t = TR[(TR.key == k)]
        if not len(t):
            continue
        t0 = float(e["t_on"])
        i0 = int(np.searchsorted(t["ts"].to_numpy(), t0))
        span = d["span"]
        pre, post, J = C.j_and_pre(d["Z"], d["tu"], t0, span)
        # 事件窗：找该事件对应的 epoch（最近的、ts ≤ t0+0.6 的 epoch）
        sub = SH[(SH.key == k) & (SH.kind == "epoch")]
        if len(sub):
            cand = sub[(sub.ts >= t0 - 1.0) & (sub.ts <= t0 + 1.0)]
            ep = cand.iloc[0] if len(cand) else None
        else:
            ep = None
        ho = None
        hsub = SH[(SH.key == k) & (SH.kind == "handoff") & (SH.ts >= t0)]
        if len(hsub):
            ho = hsub.iloc[0]
        # 窗口 [t0, t0+6]
        j0 = i0
        j1 = int(np.searchsorted(t["ts"].to_numpy(), t0 + 6.0))
        win = t.iloc[j0:j1]
        if not len(win):
            continue
        # H1 形状反演高估（A_hat / |J| − 1，τ∈[0.2,1]）
        w1 = win[(win.tau >= 0.2) & (win.tau <= 1.0)]
        h1 = np.nan
        if len(w1) and abs(J) > 1e-9:
            h1 = float((w1.A_hat / abs(J) - 1.0).median()) * 100.0
        ahat_1s = float(win[(win.tau >= 0.9) & (win.tau <= 1.0)].A_hat.median()) if len(w1) else np.nan
        # H2 滑行器速率上限：|Δc|/dt 与 RATE_MAX·|A_hat| 的比
        h2_hits, h2_ratio = 0, np.nan
        if len(win) > 3:
            dt = np.diff(win["ts"].to_numpy())
            dcv = np.diff(win["c_applied"].to_numpy())
            ok = np.isfinite(dcv) & (dt > 1e-6)
            if ok.sum() > 2:
                rate = np.abs(dcv[ok]) / dt[ok]
                amp = np.abs(win["A_hat"].to_numpy()[1:][ok])
                lim = C.KV6.RATE_MAX * np.maximum(amp, 1e-9)
                h2_hits = int((rate > 0.985 * lim).sum())
                h2_ratio = float(np.median(rate / np.maximum(lim, 1e-9)))
        # H3 交接跳变：交接帧的 |ΔΣY| / 邻域 |ΔΣY| 的**IQR**（比中位稳健：中位在平稳段趋 0）
        h3_jump = np.nan
        h3_tau = np.nan
        h3_abs = np.nan
        h3_down = np.nan
        if ho is not None:
            ih = int(np.searchsorted(t["ts"].to_numpy(), ho["ts"]))
            if 20 < ih < len(t) - 20:
                yv = t["ytot"].to_numpy()
                dloc = np.abs(np.diff(yv[ih - 20:ih + 21]))
                if len(dloc) > 10:
                    q = float(np.percentile(dloc, 75) - np.percentile(dloc, 25))
                    dj = np.diff(yv[ih - 1:ih + 2])
                    jump = float(np.abs(dj).max())
                    h3_abs = float(dj[np.argmax(np.abs(dj))])
                    h3_down = float(dj.min())
                    h3_jump = jump / max(q, 1e-9)
                    h3_tau = float(ho["ts"] - t0)
        # H4 交接后慢相提前起扣：交接后 5 s 内 g 的斜率
        h4_gslope = np.nan
        h4_gmax = np.nan
        if ho is not None:
            w4 = t[(t.ts >= ho["ts"]) & (t.ts <= ho["ts"] + 5.0)]
            if len(w4) > 20:
                gg = w4["g"].to_numpy()
                dur = float(w4["ts"].iloc[-1] - w4["ts"].iloc[0])
                h4_gslope = float((gg[-1] - gg[0]) / max(dur, 1e-9))
                h4_gmax = float(np.nanmax(np.abs(gg)))
        attrib.append(dict(
            key=k, rec=e["rec"], dom=e["dom"], kind=e["kind"], t_on=t0, clean=bool(e["clean"]),
            J=J, ev_kind_epoch=(ep["ev_kind"] if ep is not None else None),
            epoch_ts=(ep["ts"] if ep is not None else np.nan),
            epoch_base=(ep["base"] if ep is not None else np.nan),
            epoch_det_lag=(ep["t_det"] - ep["ts"] if ep is not None else np.nan),
            handoff_tau=h3_tau, handoff_A_hat=(ho["A_hat"] if ho is not None else np.nan),
            handoff_c=(ho["c_applied"] if ho is not None else np.nan),
            handoff_g=(ho["g"] if ho is not None else np.nan),
            H1_ahat_over_J_pct=h1, H1_ahat_at_1s=ahat_1s,
            H1_ahat_over_J_at1s_pct=(ahat_1s / abs(J) * 100.0 - 100.0) if abs(J) > 1e-9 else np.nan,
            H2_rate_ratio_med=h2_ratio, H2_rate_sat_frames=h2_hits,
            H3_handoff_jump_ratio=h3_jump, H3_handoff_jump_abs=h3_abs,
            H3_handoff_jump_down=h3_down,
            H4_g_slope_per_s=h4_gslope, H4_g_absmax=h4_gmax,
            n_stall_frames=int(win["stalled"].sum()),
            reached_stall=bool(win["stalled"].max() > 0) if len(win) else False,
        ))
        # 关键帧快照（每事件 6 帧：t0 / +0.2 / +0.5 / +1 / +τ_ho / +τ_ho+0.5）
        picks = [t0, t0 + 0.2, t0 + 0.5, t0 + 1.0]
        if ho is not None:
            picks += [float(ho["ts"]), float(ho["ts"]) + 0.5]
        for pt in picks:
            ii = int(np.searchsorted(t["ts"].to_numpy(), pt))
            if ii >= len(t):
                continue
            r = t.iloc[ii]
            snaps.append(dict(key=k, t_on=t0, kind=e["kind"], ts=float(r["ts"]),
                              rel_s=float(r["ts"] - t0), frame=int(r["i"]),
                              dev=float(r["dev"]), state_code=int(r["state_code"]),
                              ev_kind=r["ev_kind"], tau=r["tau"], A_hat=r["A_hat"],
                              inc=r["inc"], target=r["target"], c_applied=r["c_applied"],
                              e_track=r["e_track"], W_glide=r["W_glide"],
                              Tglide=r["Tglide"], g=r["g"], kappa_used=r["kappa_used"]))
    AT = pd.DataFrame(attrib)
    SN = pd.DataFrame(snaps)
    p3 = os.path.join(C.TASK, "results", "t5a_internal_snap.csv")
    SN.to_csv(p3, index=False, encoding="utf-8-sig", float_format="%.6g")
    p4 = os.path.join(C.TASK, "results", "t5a_attribution.csv")
    AT.to_csv(p4, index=False, encoding="utf-8-sig", float_format="%.6g")
    rec(f"产出 {p3}（{len(SN)} 行）、{p4}（{len(AT)} 行）")

    # ── 汇总打印 ──
    lines = ["\n===== T5A-Q2 归因汇总（按事件类别）====="]
    for kind in ["onset", "restep", "unload", "partial_unload"]:
        s = AT[AT.kind == kind]
        if not len(s):
            continue
        lines.append(
            f"{kind:15s} n={len(s):2d}  H1(A_hat/|J|−1)中位 {s.H1_ahat_over_J_pct.median():7.2f}%  "
            f"p10~p90 {s.H1_ahat_over_J_pct.quantile(.1):7.2f}~{s.H1_ahat_over_J_pct.quantile(.9):7.2f}  "
            f"| H2 速率饱和帧中位 {s.H2_rate_sat_frames.median():4.0f}  "
            f"| H3 交接跳变比中位 {s.H3_handoff_jump_ratio.median():6.2f}  "
            f"| H4 g 斜率中位 {s.H4_g_slope_per_s.median():8.5f}  "
            f"| 停滞事件 {int(s.reached_stall.sum())}/{len(s)}")
    txt = "\n".join(lines)
    rec(txt)
    C.write_log(os.path.join(C.TASK, "results", "_t5a_trace.log"),
                "python scripts/t5a_trace.py\n\n" + "\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
