# -*- coding: utf-8 -*-
"""T5-B / 03：T5B-Q2 失效机制链（逐帧内部状态落盘 + 带帧号的归因）

产出
    results/t5b_internal_trace.csv   逐帧内部量（含 d/gate/σ_d/state/A_hat/c_applied/ΣA/g/γ 数）
                                     每个 case 取事件前后 ±6 s
    results/t5b_mech_chain.csv       机制链（每个 case 一串"动作 + 帧号 + 数值"）
    results/_t5b_03.log

案例（覆盖用户现象的两条路径 + 真实扰动对照 + 真实阶跃对照）
    C1 实录真实瞬态（SW4 @99.56 s，实测人手拍/冲击类；**未注入**）
    C2 注入拍击 50% 电平 / 100 ms
    C3 注入拍击 100% 电平 / 500 ms（慢而长的拍击：跨过 0.40 s 撤销窗）
    C4 注入白噪 1000 ADC（复现"检测器被抖动堵住 ⇒ 真实沿被漏 ⇒ 锚点错位"）
    C5 注入真实阶跃 +4000 ADC（对照：同样幅度但不回落 ⇒ 应当成功建事件并交接）
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

from t5b_core import (TraceV6, run_full, tr_frame, load_window, perturb_window,  # noqa: E402
                      gt_in_window, load_gt, WIN, DT, REC, STATE_CODE)
from t5b_a_common import load_uniform                                            # noqa: E402

CASES = [
    dict(name="C1_real_transient_SW4_99p56", rec="中途切换-最终测试目标", t0=94.0, t1=108.0,
         kind="none", t_ev=99.56, note="实录真实瞬态（未注入），峰值 1896 ADC / 电平 29335"),
    dict(name="C2_tap_50pct_100ms", win="W1_1w", kind="tap", amp_pct=50.0, dur_ms=100,
         site=6.0, t_ev=32.0, note="注入拍击 6051 ADC 峰值 / 100 ms"),
    dict(name="C3_tap_100pct_500ms", win="W1_1w", kind="tap", amp_pct=100.0, dur_ms=500,
         site=6.0, t_ev=32.0, note="注入慢拍击 12101 ADC / 500 ms（回落沿晚于 0.40 s 撤销窗）"),
    dict(name="C4_white1000_reaIevent", win="W1_1w", kind="white", amp=1000.0, seed=7,
         t_ev=28.63, note="注入白噪 1000 ADC RMS；观察窗内真实沿 28.63 s"),
    dict(name="C5_step_4000", win="W1_1w", kind="step", amp_abs=4000.0, site=36.0, t_ev=36.0,
         note="注入真实阶跃 +4000 ADC（不回落）"),
]


def get_ctx(c):
    """返回 (tu, Xu, X_used, t_ev_w, level, w_t0, gt_w, note)。"""
    if "rec" in c:                                     # 直接用录制的一小段
        d = load_uniform(REC[c["rec"]], tmax=c["t1"])
        i0 = int(round(c["t0"] / DT))
        sl = slice(i0, len(d["tu"]))
        tu = d["tu"][sl] - c["t0"]
        Xu = d["Xu"][sl]
        return tu, Xu, Xu, c["t_ev"] - c["t0"], float(np.median(Xu.sum(axis=1))), c["t0"], None, c
    W = WIN[c["win"]]
    tu, Xu, W = load_window(W)
    lvl = float(np.median(Xu.sum(axis=1)))
    t_ev = c["t_ev"] - W["t0"]
    if c["kind"] == "none":
        X = Xu
    elif c["kind"] == "tap":
        amp = c["amp_pct"] / 100.0 * lvl
        site = c["site"]
        X, _ = perturb_window(tu, Xu, "tap", amp, 7, t_inj=site,
                              rise_ms=max(10, c["dur_ms"] // 4),
                              hold_ms=max(10, c["dur_ms"] // 2),
                              fall_ms=max(10, c["dur_ms"] // 4))
        t_ev = site + 0.05
    elif c["kind"] == "white":
        X, _ = perturb_window(tu, Xu, "white", c["amp"], c["seed"])
    elif c["kind"] == "step":
        X, _ = perturb_window(tu, Xu, "step", c["amp_abs"], 7, t_inj=c["site"])
        t_ev = c["site"] + 0.05
    gt = load_gt()
    gt_w = gt_in_window(gt, W["rec"], W["t0"], W["t1"])
    return tu, Xu, X, t_ev, lvl, W["t0"], gt_w, c


def chain(case, tr, ts_ev, t_end, run=None):
    """从逐帧 trace 里抽机制链（带帧号与数值）。`run` 提供 epoch/revoke/handoff/unload 时刻。"""
    n = len(tr)
    ts = tr["ts"].to_numpy()
    rows = []

    def add(step, i, event, v1, v2, v3, note):
        if i is None or i < 0 or i >= n:
            return
        r = tr.iloc[i]
        rows.append(dict(case=case, step=step, frame=int(i), t=round(float(r["ts"]), 3),
                         event=event, val1=(round(float(v1), 3) if v1 == v1 else None),
                         val2=(round(float(v2), 3) if v2 == v2 else None),
                         val3=(round(float(v3), 3) if v3 == v3 else None),
                         d=round(float(r["d"]), 1), gate=round(float(r["gate"]), 1),
                         sigma_d=round(float(r["sig_d"]), 1), state=str(r["state_pre"]),
                         ev_kind=str(r["ev_now"]), sumA=round(float(r["sumA"]), 1),
                         A_hat=(round(float(r["A_hat"]), 1) if r["A_hat"] == r["A_hat"] else None),
                         g=round(float(r["g"]), 4), note=note))
    hit = tr["raw_hit"].to_numpy(bool)
    hr = tr["hit_run"].to_numpy(int)
    idx = np.where(hit)[0]
    if len(idx):
        i_first = int(idx[0])
        add(1, i_first, "首次 |d|>gate", tr["d"].iloc[i_first], tr["gate"].iloc[i_first],
            float(tr["d"].iloc[i_first] / max(tr["gate"].iloc[i_first], 1e-9)),
            "raw_hit=真，hit_run=1")
        j = None
        for k in idx:
            if hr[k] >= 3:
                j = int(k)
                break
        add(2, j, "满足连续 3 帧（现行判据决策点）", tr["d"].iloc[j] if j is not None else np.nan,
            tr["gate"].iloc[j] if j is not None else np.nan,
            tr["armed_pre"].iloc[j] if j is not None else np.nan,
            "armed_pre 决定是否真正允许建事件")
    i_pk = int(np.argmax(np.abs(tr["d"].to_numpy())))
    add(3, i_pk, "|d| 峰值", tr["d"].iloc[i_pk], tr["gate"].iloc[i_pk],
        float(tr["d"].iloc[i_pk]) / max(float(tr["gate"].iloc[i_pk]), 1e-9),
        "扰动偏离最大处")

    def idx_of(t):
        return int(np.argmin(np.abs(ts - t)))

    if run is not None:
        k = 4
        for e in run["epoch"]:
            add(k, idx_of(e[2]), "建事件 t_det", e[2], e[3], e[0],
                f"kind={e[1]} 回溯 t0={e[0]:.2f} base={e[3]:.0f}")
            k += 1
        for v in run["revoke"]:
            add(k, idx_of(v[0]), "撤销事件（回到上一状态）", v[0], np.nan, np.nan,
                "tau<0.40s 且 inc_s<0.5·inc_max ⇒ 判定为瞬态")
            k += 1
        for h in run["handoff"]:
            add(k, idx_of(h[0]), "交接给慢相模块（锚定 A）", h[2], h[3], h[4],
                f"Â={h[2]:.0f} c_applied={h[3]:.0f} ΣA={h[4]:.0f} g={h[5]:.4f}")
            k += 1
        for u in run["unload"]:
            add(k, idx_of(u), "回空载 _to_idle（A/g/loaded 全部清零）", u, np.nan, np.nan,
                "SUM_A=0, g=0, loaded=False ⇒ **基线被清零**")
            k += 1
        add(k, n - 1, "窗末状态", tr["sumA"].iloc[-1], tr["dA"].iloc[-1] if "dA" in tr else np.nan,
            np.nan, f"state={tr['state_post'].iloc[-1]}")
    return rows


def main():
    all_tr, all_chain, summary = [], [], []
    for c in CASES:
        tu, Xu, X, t_ev, lvl, w_t0, gt_w, cc = get_ctx(c)
        r = run_full(tu, X, TraceV6)
        ref = run_full(tu, Xu, TraceV6)
        df = tr_frame(r["tr"])
        df.insert(0, "case", c["name"])
        df["dA"] = r["tr"]["sumA"] - ref["tr"]["sumA"]
        df["dev"] = r["Ysum"] - ref["Ysum"]
        df["Zdev"] = r["Z"] - ref["Z"]
        df["state_code"] = df["state_pre"].map(STATE_CODE)
        i_ev = int(np.argmin(np.abs(tu - t_ev)))
        a, b = max(0, i_ev - int(6 / DT)), min(len(tu), i_ev + int(8 / DT))
        all_tr.append(df.iloc[a:b])
        ch = chain(c["name"], df, t_ev, tu[-1], run=r)
        all_chain += ch
        # 另外记录整段运行的窗口外事件（epoch/revoke/unload 全量）
        for e in r["epoch"]:
            all_chain.append(dict(case=c["name"], step=90, frame=int(np.argmin(np.abs(tu - e[2]))),
                                  t=round(e[2], 3), event="epoch_all", val1=e[0], val2=e[3],
                                  val3=e[2], d=np.nan, gate=np.nan, sigma_d=np.nan,
                                  state="", ev_kind=e[1], sumA=np.nan, A_hat=np.nan, g=np.nan,
                                  note="全量 epoch 清单"))
        for v in r["revoke"]:
            all_chain.append(dict(case=c["name"], step=91, frame=int(np.argmin(np.abs(tu - v[0]))),
                                  t=round(v[0], 3), event="revoke_all", val1=v[0], val2=np.nan,
                                  val3=np.nan, d=np.nan, gate=np.nan, sigma_d=np.nan, state="",
                                  ev_kind="", sumA=np.nan, A_hat=np.nan, g=np.nan,
                                  note="全量 revoke 清单"))
        for u in r["unload"]:
            all_chain.append(dict(case=c["name"], step=92, frame=int(np.argmin(np.abs(tu - u))),
                                  t=round(u, 3), event="to_idle_all", val1=u, val2=np.nan,
                                  val3=np.nan, d=np.nan, gate=np.nan, sigma_d=np.nan, state="",
                                  ev_kind="", sumA=np.nan, A_hat=np.nan, g=np.nan,
                                  note="全量 _to_idle 清单（基线清零）"))
        # 关键量
        ep = [e for e in r["epoch"] if abs(e[0] - t_ev) <= 2.0 or abs(e[2] - t_ev) <= 2.5]
        rev = [v for v in r["revoke"] if abs(v[0] - t_ev) <= 3.0]
        ho = [h for h in r["handoff"] if abs(h[0] - t_ev) <= 7.0]
        summary.append(dict(
            case=c["name"], note=c["note"], level=round(lvl, 1), t_ev=round(t_ev, 3),
            n_epoch_all=len(r["epoch"]),
            n_epoch_near=len(ep),
            epoch_near="|".join(f"{e[1]}@t0={e[0]:.2f}/det={e[2]:.2f}/base={e[3]:.0f}/Â={e[6] if len(e)>6 else ''}"
                                for e in ep),
            n_revoke_near=len(rev),
            revoke_ts="|".join(f"{v[0]:.2f}" for v in rev),
            n_handoff_near=len(ho),
            handoff_ts="|".join(f"{h[0]:.2f}Â={h[2]:.0f}" for h in ho),
            dA_at_end=round(float(r["tr"]["sumA"][-1] - ref["tr"]["sumA"][-1]), 1),
            sumA_end=round(float(r["tr"]["sumA"][-1]), 1), sumA_clean_end=round(float(ref["tr"]["sumA"][-1]), 1),
            dev_max=round(float(np.abs(r["Ysum"] - ref["Ysum"]).max()), 1),
            dev_rel=round(float(np.abs(r["Ysum"] - ref["Ysum"]).max()) / max(lvl, 1.0), 4),
        ))
        print(f"[{c['name']}] 电平={lvl:.0f} 附近epoch={len(ep)} 撤销={len(rev)} 交接={len(ho)} "
              f"末ΔΣA={summary[-1]['dA_at_end']:.0f} 最大显示偏差={summary[-1]['dev_max']:.0f}",
              flush=True)
    pd.concat(all_tr, ignore_index=True).to_csv(
        os.path.join(RES, "t5b_internal_trace.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(all_chain).to_csv(
        os.path.join(RES, "t5b_mech_chain.csv"), index=False, encoding="utf-8-sig")
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(RES, "t5b_mech_summary.csv"), index=False, encoding="utf-8-sig")
    print("\n=== 机制链汇总 ===")
    print(sm.to_string())


if __name__ == "__main__":
    main()
