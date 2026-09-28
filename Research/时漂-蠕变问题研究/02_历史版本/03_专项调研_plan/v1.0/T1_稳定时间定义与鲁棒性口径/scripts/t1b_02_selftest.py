# -*- coding: utf-8 -*-
"""T1-B / 02：`t1_common.py` 扰动注入器**自检**（产物：results/t1b_injector_selftest.csv）。

自检项（每项一行，含实测值与判据、PASS/FAIL）：
  load_rec        ##Data 自动定位 + 帧数与 `t1b_probe.csv` 一致
  packets         包结构解析（包数、包周期、包内帧数）
  to_grid         100 Hz 网格步长/长度
  perturb_white   总量 RMS = A
  perturb_band    总量 RMS = A 且频谱被限制在 [f_lo, f_hi]
  perturb_common  总量 RMS = A 且通道间相关系数 = 1（完全共模）
  perturb_corr    部分相关档的通道间相关系数 ≈ corr
  tap_peak        总量峰值增量 = amp，且形态 50/100/50 ms
  step_level      台阶稳态增量 = amp
  jitter_*        ±k 包抖动：包结构保留、时间偏移 ≤ k·T_pkt、时间轴严格递增
  dropout_frame   单帧丢包比例 ≈ frac
  dropout_packet  整包丢包
  aggregate_k     包聚合后唯一时间戳数 ≈ ceil(n/k)
  metric_stable   对合成"阶跃+已知稳定"信号，t_stable 能测出人为设定的稳定时刻（机制对照）
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

import t1_common as C          # noqa: E402


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()


def main():
    log = Tee(os.path.join(RES, "_t1b_02_selftest.log"))
    log("=== T1-B injector self-test ===")
    rows = []

    def add(item, value, crit, ok, note=""):
        rows.append(dict(item=item, value=value, criterion=crit, pass_=bool(ok), note=note))
        log(f"  [{'PASS' if ok else 'FAIL'}] {item}: {value}  (判据 {crit}) {note}")

    # ── 数据层 ──
    rec = C.REC["零负载-切换负载-零负载-再切换负载"]
    d = C.load_uniform(rec)
    t, X = d["t"], d["X"]
    add("load_rec_frames", len(t), "== 7128 (t1b_probe)", len(t) == 7128)
    add("load_rec_ch", X.shape[1], "== 21", X.shape[1] == 21)
    pk = C.packets(t)
    add("packets_pkt_s_ms", round(pk["pkt_s"] * 1000, 3), "≈40 ms（实录）",
        30 < pk["pkt_s"] * 1000 < 45, f"n_pkt={pk['n_pkt']}")
    add("packets_frames_per_pkt", int(pk["frames_per_pkt"].max()), "<= 20",
        pk["frames_per_pkt"].max() <= 20)
    add("to_grid_n", len(d["tu"]), "≈ span*100", abs(len(d["tu"]) - d["span"] * 100) < 5)
    add("to_grid_dt", round(float(np.diff(d["tu"])[0]), 6), "== 0.01",
        abs(np.diff(d["tu"])[0] - 0.01) < 1e-9)

    rng = np.random.default_rng(7)
    Xg = d["Xu"]

    # ── 加性扰动 ──
    for kind in ("white", "band", "common"):
        A = 500.0
        E = C.make_perturb(Xg, A, kind, rng, f_lo=0.3, f_hi=40.0)
        rms = float(E.sum(axis=1).std())
        add(f"perturb_{kind}_rms", round(rms, 6), f"== {A}",
            abs(rms - A) / A < 1e-6)
        if kind == "common":
            cc = np.corrcoef(E[:, 0], E[:, 5])[0, 1]
            add("perturb_common_ch_corr", round(float(cc), 6), "== 1.0", abs(cc - 1) < 1e-9)
        if kind == "band":
            W = np.abs(np.fft.rfft(E.sum(axis=1))) ** 2
            f = np.fft.rfftfreq(len(W) * 2 - 2, 0.01)
            tot = W.sum()
            inband = W[(f >= 0.3) & (f <= 40.0)].sum()
            add("perturb_band_inband_energy", round(float(inband / tot), 6),
                ">= 0.999", inband / tot >= 0.999)
        if kind == "white":
            W = np.abs(np.fft.rfft(E.sum(axis=1))) ** 2
            f = np.fft.rfftfreq(len(W) * 2 - 2, 0.01)
            hi = W[f > 45.0].sum() / W.sum()
            add("perturb_white_energy_above45Hz", round(float(hi), 4), "> 0.1", hi > 0.1)
    E5 = C.make_perturb(Xg, 500.0, "white", rng, corr=0.5)
    cm = np.corrcoef(E5.T)
    off = cm[np.triu_indices_from(cm, 1)]
    add("perturb_corr0.5_mean_ch_corr", round(float(off.mean()), 4), "≈ 0.5",
        abs(off.mean() - 0.5) < 0.05)

    # ── 形态注入 ──
    i0 = 3000
    Y, nr = C.add_tap(Xg, i0, 3000.0, rise_ms=50, hold_ms=100, fall_ms=50)
    dtot = Y.sum(axis=1) - Xg.sum(axis=1)
    add("tap_peak_total", round(float(dtot.max()), 3), "== 3000", abs(dtot.max() - 3000) < 1)
    add("tap_len_ms", round(1000 * len(dtot[dtot > 1.0]) / 100, 1), "≈ 200 ms（50+100+50）",
        150 < 1000 * len(dtot[dtot > 1.0]) / 100 < 260)
    add("tap_returns_to_zero", round(float(np.abs(dtot[i0 + 400:]).max()), 6), "== 0",
        np.abs(dtot[i0 + 400:]).max() < 1e-6)
    Y2 = C.add_step(Xg, i0, 3000.0, rise_ms=50)
    ds = Y2.sum(axis=1) - Xg.sum(axis=1)
    add("step_level_total", round(float(ds[-100:].mean()), 3), "== 3000",
        abs(ds[-100:].mean() - 3000) < 1)

    # ── 时序注入（帧级）──
    for k in (1, 2):
        t2, X2 = C.add_packet_jitter(t, X, k_pkt=k, rng=np.random.default_rng(3))
        pk2 = C.packets(t2)
        add(f"jitter{k}_monotonic", bool(np.all(np.diff(t2) >= 0)), "时间轴非降", np.all(np.diff(t2) >= 0))
        add(f"jitter{k}_n_frames", len(t2), "== n（帧数不变）", len(t2) == len(t))
        add(f"jitter{k}_merge_rate", round(1 - len(pk2["ts"]) / len(pk["ts"]), 4),
            "信息项（同刻到达=合并包，非判据）", True)
        add(f"jitter{k}_max_frame_shift_pkt",
            round(float(np.abs(t2 - t).max() / pk["pkt_s"]), 3),
            f"<= {k} + 0.01", float(np.abs(t2 - t).max() / pk["pkt_s"]) <= k + 0.01)
        add(f"jitter{k}_cum_shift_pkt", round(float(abs(t2[-1] - t[-1]) / pk["pkt_s"]), 3),
            f"<= {k} + 0.01（无累计漂移）", abs(t2[-1] - t[-1]) / pk["pkt_s"] <= k + 0.01)
    t2d, X2d = C.add_packet_jitter(t, X, k_pkt=1, rng=np.random.default_rng(5), mode="delay")
    add("jitter_delay_only_forward", round(float(np.min(t2d - t)), 6),
        ">= −0.001 s（包内原始时间戳微差不计）", float(np.min(t2d - t)) >= -0.001)
    add("jitter_shift_origin", C.shift_event_origin(10.0, 2, pk["pkt_s"]),
        "== 10.0 + 2*T_pkt", abs(C.shift_event_origin(10.0, 2, pk["pkt_s"]) - 10.0 - 2 * pk["pkt_s"]) < 1e-9)
    for frac in (0.01, 0.05):
        t3, X3 = C.add_dropout(t, X, frac=frac, rng=np.random.default_rng(11))
        got = 1 - len(t3) / len(t)
        add(f"dropout_frame_{frac}_realized", round(got, 4), f"≈ {frac}", abs(got - frac) < 0.01)
    t4, X4 = C.add_dropout(t, X, frac=0.05, rng=np.random.default_rng(11), whole_packet=True)
    pk4 = C.packets(t4)
    add("dropout_packet_0.05_n_pkt", pk4["n_pkt"], f"≈ {int(pk['n_pkt']*0.95)}",
        abs(pk4["n_pkt"] - pk["n_pkt"] * 0.95) < pk["n_pkt"] * 0.03)
    ts_syn = np.arange(0.0, 10.0, 0.01)
    Xsyn = np.zeros((len(ts_syn), 3))
    for k in (2, 4):
        t5, X5 = C.add_packet_aggregate(ts_syn, Xsyn, k=k)
        nu = len(np.unique(np.round(t5, 4)))
        add(f"aggregate_synthetic_{k}_n_unique_ts", nu,
            f"== ceil(n/{k})={int(np.ceil(len(ts_syn)/k))}",
            nu == int(np.ceil(len(ts_syn) / k)))
    t6, _ = C.add_packet_aggregate(t, X, k=16)
    pk6 = C.packets(t6)
    add("aggregate_real_k16_n_pkt_vs_orig", f"{pk6['n_pkt']} vs {pk['n_pkt']}",
        "信息项（实录本已 4 帧/包，k=16 ⇒ 4 包并 1）", pk6["n_pkt"] < pk["n_pkt"])

    # ── 指标机制对照（合成信号，仅用于自检口径实现，不作为主结论）──
    # 口径说明：字典 §3 的 T_stable 是"5%·J 带内首次不再移动"，因此**读数一进 ±5%·J 带
    # 且此后 30 s 不再出去**就算稳定 —— 下面第 1 个信号故意让快相在 2 s 处才进带。
    fs, dtm = 100.0, 0.01
    tu = np.arange(0, 120, dtm)
    t_on, J = 10.0, 100.0
    Z = np.zeros_like(tu)
    m = tu >= t_on
    tau_fast = 1.0
    Z[m] = J * (1.0 - np.exp(-(tu[m] - t_on) / tau_fast))
    Z[tu >= t_on + 2.0] = J * (1.0 - np.exp(-2.0 / tau_fast))       # 2 s 起完全平
    zc = J * (1.0 - np.exp(-2.0 / tau_fast))
    tau_exp = -tau_fast * np.log(1.0 - (zc - 0.05 * J) / J)          # 首次进 ±5%·J 带的时刻
    ts = C.t_stable(tu, Z, t_on, J, dtm=dtm)
    add("metric_t_stable_on_synthetic", round(float(ts["tau_v1"]), 3),
        f"≈ {tau_exp:.2f}（首次进 5%·J 带，±0.35）", abs(ts["tau_v1"] - tau_exp) < 0.35)
    Z3 = np.zeros_like(tu)                                            # 持续蠕变（永不稳定）
    Z3[m] = J * (1.0 - np.exp(-(tu[m] - t_on) / 40.0))
    ts3 = C.t_stable(tu, Z3, t_on, J, dtm=dtm, horizon=30.0)
    add("metric_t_stable_on_creep", f"tau_v1={ts3['tau_v1']:.2f}, ok={ts3['ok']}",
        "τ_slow=40 s 的蠕变：应 >20 s 或不可测", (not ts3["ok"]) or ts3["tau_v1"] > 20.0)
    Z4 = np.zeros_like(tu)
    Z4[m] = J * np.ones(m.sum())
    ts4 = C.t_stable(tu[:3000], Z4[:3000], t_on, J, dtm=dtm)          # 记录太短
    add("metric_t_stable_short_record", f"ok={ts4['ok']}", "记录不足 30 s ⇒ ok=False",
        ts4["ok"] is False)
    osd = C.overshoot(tu, Z, t_on, J, z_final=None, dtm=dtm)
    add("metric_os_on_synthetic", round(float(osd["os_pct"]), 3), "≈ 0（无过冲）",
        abs(osd["os_pct"]) < 0.5)
    Z2 = Z.copy()
    Z2[(tu >= t_on + 3.0) & (tu < t_on + 4.0)] += 0.15 * J            # 平台段 1 s 宽、15% 过冲
    osd2 = C.overshoot(tu, Z2, t_on, J, z_final=None, dtm=dtm)
    add("metric_os_on_synthetic_over", round(float(osd2["os_pct"]), 3), "≈ 15%（±1.0）",
        abs(osd2["os_pct"] - 15.0) < 1.0)
    add("metric_zfin_src", osd2["zfin_src"], "短记录退回 plateau 口径", osd2["zfin_src"] in ("ref60", "plateau"))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "t1b_injector_selftest.csv"), index=False, encoding="utf-8-sig")
    nfail = int((~df["pass_"]).sum())
    log(f"--- {len(df)} 项，FAIL {nfail} ---")
    log.f.close()
    return nfail


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
