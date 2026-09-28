# -*- coding: utf-8 -*-
"""v2.0 A1 · 回放诊断主体：Q1~Q5 的可复算数字。

运行：python temp\\v4.1flash\\plan\\v2.0\\scripts\\v20_a1_probe.py
只读：不改任何源码、不构建、不跑测试。产物：
  results/a1_probe.txt          （本脚本全部 stdout）
  results/a1_replay_frames.csv  （当前参数集逐帧内部状态）
  results/a1_replay_events.csv  （全部建事件/状态迁移日志）
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import v20_lib as L          # noqa: E402
import v20_replay as R       # noqa: E402

OUT = os.path.abspath(os.path.join(_HERE, "..", "results"))
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "a1_probe.txt"), "w", encoding="utf-8")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")


def med(x):
    x = np.asarray(x, float)
    return float(np.median(x)) if x.size else float("nan")


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def merge_segments(segs, min_dur=1.0):
    out = []
    for s in segs:
        if out and (s[2] - s[1]) < min_dur * 100:
            out[-1] = (out[-1][0], out[-1][1], s[2])
        else:
            out.append(s)
    return out


def run(pre, **kw):
    rp = R.V6Replay(L.NCH, **kw)
    t0 = time.time()
    for ts, v in zip(pre["ts"], pre["V"]):
        rp.process(float(ts), v)
    return rp, time.time() - t0


def main():
    t_start = time.time()
    ds = L.load_dataset(L.DS_ZERO)
    pre, main = ds["pre"], ds["main"]
    el = pre["el"]
    tp = pre["V"].sum(1)
    tm = main["V"].sum(1)
    nfr = pre["n"]
    off_field = tm - tp
    p(f"[data] DS_ZERO={L.DS_ZERO}")
    p(f"[data] frames={nfr} span={el[-1]-el[0]:.2f}s fps={nfr/(el[-1]-el[0]):.2f}")
    p(f"[data] tot_pre  min={tp.min():.0f} max={tp.max():.0f}  逐通道数={L.NCH}")
    p(f"[data] tot_main min={tm.min():.0f} max={tm.max():.0f}")
    p(f"[data] 现场偏移 main-pre: 第一帧={off_field[0]:.0f} 末帧={off_field[-1]:.0f} "
      f"中位={med(off_field):.0f} RMS={rms(off_field):.0f} 峰谷="
      f"{off_field.min():.0f}/{off_field.max():.0f}")
    p(f"[data] session.algorithm.params={ds['sess']['algorithm']['params']}")

    # ── 四个回放变体 ───────────────────────────────────────────────────
    variants = {
        "A_cur(κ1.05,ho3.5,hold3)": dict(**R.PARAMS_CURRENT, med_lag=0),
        "B_cur+C++med_lag1": dict(**R.PARAMS_CURRENT, med_lag=1),
        "C_proto(κ1.30,ho5.0,hold1)": dict(**R.PARAMS_PROTO_RAW, med_lag=0),
        "D_proto+C++med_lag1": dict(**R.PARAMS_PROTO_RAW, med_lag=1),
    }
    reps = {}
    for name, kw in variants.items():
        rp, dt = run(pre, **kw)
        reps[name] = rp
        st = [r["state"] for r in rp.frames]
        nev = len([e for e in rp.events if e["ev"] == "NEW_EVENT"])
        nho = len([e for e in rp.events if e["ev"] == "HANDOFF"])
        nre = len([e for e in rp.events if e["ev"] == "REANCHOR"])
        nid = len([e for e in rp.events if e["ev"] == "TO_IDLE"])
        nrv = len([e for e in rp.events if e["ev"] == "REVOKE"])
        p(f"[replay] {name:<28} {dt:5.1f}s  事件={nev:3d} 交接={nho:3d} 重锚={nre:3d} "
          f"ToIdle={nid:3d} revoke={nrv:3d}  idle/event/slow="
          f"{st.count('idle')/100:.0f}/{st.count('event')/100:.0f}/{st.count('slow')/100:.0f}s")

    rp = reps["A_cur(κ1.05,ho3.5,hold3)"]
    R.dump_frames(rp, R.FRAMES_CSV)
    R.dump_events(rp, R.EVENTS_CSV)
    p(f"[replay] wrote {R.FRAMES_CSV}")
    p(f"[replay] wrote {R.EVENTS_CSV}")
    proto = np.array([r["out_total"] for r in rp.frames])
    states = [r["state"] for r in rp.frames]

    # ═══════════════════════ 平台切片 ═══════════════════════
    segs = merge_segments(L.plateau_segments(tp, el, hyst_frac=0.25, min_dur=0.25))
    idle_m = np.zeros(nfr, bool)
    load_m = np.zeros(nfr, bool)
    for k, a, b in segs:
        if (b - a) < 50:
            continue
        (idle_m if k == "idle" else load_m)[a:b + 1] = True

    # ═══════════════════════ Q1 ═══════════════════════
    p("\n" + "=" * 78)
    p("Q1  原型回放(pre 流) vs 现场实测算法输出(main 流)；口径=总量 ADC")
    p("=" * 78)
    d = proto - tm
    p(f"[Q1] 逐帧差 proto−main: 中位={med(d):.1f} 均值={d.mean():.1f} RMS={rms(d):.1f} "
      f"min={d.min():.1f} max={d.max():.1f} |范围|={d.max()-d.min():.1f} ADC")
    p(f"[Q1] 参照 现场偏移 main−pre 中位={med(off_field):.1f} RMS={rms(off_field):.1f} ADC")
    p(f"[Q1] 参照 原型偏移 proto−pre 中位={med(proto-tp):.1f} RMS={rms(proto-tp):.1f} ADC")
    p(f"{'#':>3}{'kind':>8}{'t0':>8}{'t1':>8}{'dur':>7}{'pre':>8}{'main':>8}"
      f"{'proto':>8}{'off_main':>9}{'off_proto':>10}{'d=proto-main':>14}{'RMS_d':>9}")
    for i, (k, a, b) in enumerate(segs):
        sl = slice(a, b + 1)
        p(f"{i:>3}{k:>8}{el[a]:8.2f}{el[b]:8.2f}{el[b]-el[a]:7.1f}{np.mean(tp[sl]):8.0f}"
          f"{np.mean(tm[sl]):8.0f}{np.mean(proto[sl]):8.0f}"
          f"{np.mean(tm[sl]-tp[sl]):9.0f}{np.mean(proto[sl]-tp[sl]):10.0f}"
          f"{np.mean(proto[sl]-tm[sl]):14.0f}{rms(proto[sl]-tm[sl]):9.1f}")
    p(f"[Q1] 空载帧={idle_m.sum()} 受载帧={load_m.sum()}")
    for nm, m in (("空载", idle_m), ("受载", load_m)):
        p(f"[Q1] {nm}: d 中位={med(proto[m]-tm[m]):.1f} RMS={rms(proto[m]-tm[m]):.1f} "
          f"| off_proto 中位={med(proto[m]-tp[m]):.0f} off_main 中位={med(tm[m]-tp[m]):.0f} "
          f"| 电平 proto={np.mean(proto[m]):.0f} main={np.mean(tm[m]):.0f} pre={np.mean(tp[m]):.0f}")
    b0 = float(np.median(tp[idle_m]))
    for nm, m in (("空载", idle_m), ("受载", load_m)):
        p(f"[Q1] 增量口径(以空载中位 {b0:.0f} 为零): {nm} "
          f"Δmain 中位={med(tm[m]-b0):.0f} Δproto 中位={med(proto[m]-b0):.0f} "
          f"Δ(proto−main) 中位={med(proto[m]-tm[m]):.1f} RMS={rms(proto[m]-tm[m]):.1f}")

    p("[Q1] 各变体逐帧差（proto−main）与各自偏移：")
    for name, rpv in reps.items():
        pv = np.array([r["out_total"] for r in rpv.frames])
        p(f"[Q1]   {name:<28} d 中位={med(pv-tm):8.1f} RMS={rms(pv-tm):8.1f} "
          f"off 中位={med(pv-tp):8.1f} RMS={rms(pv-tp):8.1f} ADC")

    # 事件序列对照
    p("[Q1] 原型(A) 建事件序列（t0/kind/触发帧 ts/base/d/thr_d/tail_gate）：")
    for e in [x for x in rp.events if x["ev"] == "NEW_EVENT"]:
        p(f"      t0={e['t0']:9.3f} {R.KIND_CN.get(e['kind'], e['kind']):<13}"
          f"ts={e['ts']:9.3f} base={e['base']:9.1f} d={e['d']:8.1f} "
          f"thr_d={e['thr_d']:7.1f} tail={e['tail_gate']:7.1f} hist={e['hist_len']}")
    doff = np.diff(off_field, prepend=off_field[0])
    thr_ev = 20.0
    cand = np.where(np.abs(doff) > thr_ev)[0]
    groups = []
    for i in cand:
        if groups and i - groups[-1][-1] <= 30:
            groups[-1].append(i)
        else:
            groups.append([i])
    p(f"[Q1] 现场输出侧偏移跳变簇数={len(groups)}（|Δoff/frame|>{thr_ev} ADC，30 帧合并）：")
    p("      t=" + " ".join(f"{el[g[np.argmax(np.abs(doff[g]))]]:.2f}" for g in groups))

    # ═══════════════════════ Q2 ═══════════════════════
    p("\n" + "=" * 78)
    p("Q2  44 s 附近 ±5 s（39~49 s）逐帧明细 + 该处算法动作")
    p("=" * 78)
    p(f"{'t':>7}{'pre':>8}{'main':>8}{'proto':>8}{'off_f':>8}{'off_p':>8}"
      f"{'state':>7}{'kind':>13}{'tau':>7}{'A_hat':>9}{'kAP':>6}{'g':>10}"
      f"{'A_sum':>10}{'idle':>5}{'min_ts':>9}{'lev_ref':>9}{'C6':>4}")
    step = 0.1
    tq = 39.0
    while tq < 49.0:
        sl = (el >= tq) & (el < tq + step)
        if sl.any():
            r = rp.frames[int(np.where(sl)[0][0])]
            k = R.KIND_CN.get(r["kind"], r["kind"])
            p(f"{tq:7.2f}{np.mean(tp[sl]):8.0f}{np.mean(tm[sl]):8.0f}"
              f"{np.mean(proto[sl]):8.0f}{np.mean(off_field[sl]):8.0f}"
              f"{np.mean(proto[sl]-tp[sl]):8.0f}{r['state']:>7}{k:>13}"
              f"{(r['tau'] if r['tau'] != '' else float('nan')):7.2f}"
              f"{(r['A_hat'] if r['A_hat'] != '' else float('nan')):9.1f}"
              f"{(r['c_applied'] if r['c_applied'] != '' else float('nan')):6.0f}"
              f"{r['g']:10.5f}{r['A_sum']:10.0f}{r['idle_now']:5d}"
              f"{r['min_ts']:9.0f}{r['level_ref']:9.0f}{r['C6']:4d}")
        tq += step
    p("[Q2] 39~49 s 内原型全部状态迁移日志：")
    for e in rp.events:
        try:
            ets = float(e["ts"])
        except (TypeError, ValueError):
            continue
        if 38.5 <= ets <= 49.5:
            p("      " + " ".join(f"{k}={e[k]}" for k in
                                  ("ev", "ts", "t0", "kind", "base", "A_hat", "A_sum_before",
                                   "A_sum_after", "g_before", "g_after", "state_before",
                                   "state_after", "c0_inherit", "C", "C6") if k in e))

    # ═══════════════════════ Q3 ═══════════════════════
    p("\n" + "=" * 78)
    p("Q3  偏移时间线：通路(a) 事件期滑行器 share·c_target  vs  通路(b) 慢相 γ·A·g")
    p("     口径：off = main − pre = −(总扣除量)；两条通路的扣除量互斥（逐帧只走一条）")
    p("=" * 78)
    off_proto = proto - tp
    p(f"{'#':>3}{'kind':>8}{'t0':>8}{'t1':>8}{'off_field':>10}{'off_proto':>10}"
      f"{'a_glide_med':>12}{'b_slow_med':>11}{'a+b_med':>9}{'a_share':>8}{'b_share':>8}"
      f"{'n_evt帧':>8}{'n_slow帧':>9}")
    for i, (k, a, b) in enumerate(segs):
        sl = slice(a, b + 1)
        gd = np.array([rp.frames[j]["off_glide"] for j in range(a, b + 1)])
        sd = np.array([rp.frames[j]["off_slow"] for j in range(a, b + 1)])
        gtot = np.median(np.abs(gd)) if np.any(gd != 0) else 0.0
        stot = np.median(np.abs(sd)) if np.any(sd != 0) else 0.0
        sm = gtot + stot
        p(f"{i:>3}{k:>8}{el[a]:8.2f}{el[b]:8.2f}{np.median(off_field[sl]):10.0f}"
          f"{np.median(off_proto[sl]):10.0f}{gtot:12.0f}{stot:11.0f}{sm:9.0f}"
          f"{(gtot/sm if sm else 0):8.2f}{(stot/sm if sm else 0):8.2f}"
          f"{int(np.sum(gd != 0)):8d}{int(np.sum(sd != 0)):9d}")
    gdall = np.array([r["off_glide"] for r in rp.frames])
    sdall = np.array([r["off_slow"] for r in rp.frames])
    p(f"[Q3] 全段: 通路a 非零帧={int(np.sum(gdall!=0))} 均值={gdall[gdall!=0].mean() if np.any(gdall!=0) else 0:.0f} "
      f"| 通路b 非零帧={int(np.sum(sdall!=0))} 均值={sdall[sdall!=0].mean() if np.any(sdall!=0) else 0:.0f}")
    p(f"[Q3] 现场偏移 vs 原型(通道路径和)逐帧残差 RMS="
      f"{rms(off_field-(gdall+sdall)):.1f} 中位={med(off_field-(gdall+sdall)):.1f} ADC")

    # ═══════════════════════ Q4 ═══════════════════════
    p("\n" + "=" * 78)
    p("Q4  min_ts / level_ref / g / A.sum() 全程轨迹（每 5 s 一点）")
    p("=" * 78)
    p(f"{'t':>7}{'state':>7}{'min_ts':>9}{'level_ref':>10}{'g':>10}{'A_sum':>10}"
      f"{'A_max':>8}{'idle':>5}{'pre':>8}{'proto':>8}{'off_f':>8}")
    tq = 0.0
    while tq < el[-1]:
        idx = int(np.searchsorted(el, tq))
        if idx < nfr:
            r = rp.frames[idx]
            p(f"{tq:7.1f}{r['state']:>7}{r['min_ts']:9.0f}{r['level_ref']:10.0f}"
              f"{r['g']:10.5f}{r['A_sum']:10.0f}{'':>8}{r['idle_now']:5d}"
              f"{tp[idx]:8.0f}{proto[idx]:8.0f}{off_field[idx]:8.0f}")
        tq += 5.0
    gs = np.array([r["g"] for r in rp.frames])
    minss = np.array([r["min_ts"] for r in rp.frames])
    lrs = np.array([r["level_ref"] for r in rp.frames])
    As = np.array([r["A_sum"] for r in rp.frames])
    p(f"[Q4] min_ts: 全程最小={minss.min():.0f} 出现在 t={el[int(np.argmin(minss))]:.2f}s；"
      f"末值={minss[-1]:.0f}")
    p(f"[Q4] level_ref: 最小={lrs.min():.0f} 最大={lrs.max():.0f} 末值={lrs[-1]:.0f}")
    p(f"[Q4] g: 最小={gs.min():.5f} 最大={gs.max():.5f} 末值={gs[-1]:.5f} "
      f"g<0 帧占比={np.mean(gs<0)*100:.1f}%")
    p(f"[Q4] A.sum(): 最小={As.min():.0f} 最大={As.max():.0f} 末值={As[-1]:.0f}")
    p("[Q4] 1.5·min_ts 判据：全程 1.5·min_ts 最大="
      f"{1.5*minss.max():.0f}（min_ts 最大值 {minss.max():.0f}）")
    for name, rpv in reps.items():
        f2 = rpv.frames
        a2 = np.array([r["A_sum"] for r in f2])
        g2 = np.array([r["g"] for r in f2])
        m2 = np.array([r["min_ts"] for r in f2])
        p(f"[Q4] {name:<28} min_ts 最小={m2.min():8.0f} 末={m2[-1]:8.0f} | "
          f"A_sum 末={a2[-1]:9.0f} g 末={g2[-1]:9.5f}")

    # ═══════════════════════ Q5 ═══════════════════════
    p("\n" + "=" * 78)
    p("Q5  与 T9 案例对比：never-idle 失效域判据")
    p("=" * 78)
    for nm, r in (("A", rp), ("B(C++lag)", reps["B_cur+C++med_lag1"])):
        idn = np.array([x["idle_now"] for x in r.frames], bool)
        st = np.array([x["state"] for x in r.frames])
        p(f"[Q5] 变体{nm}: idle_now 真帧={idn.sum()}（{idn.mean()*100:.2f}%）"
          f" 假帧={(~idn).sum()}（{(~idn).mean()*100:.2f}%）")
        for s in ("idle", "event", "slow"):
            m = (st == s)
            if m.any():
                p(f"[Q5]   状态 {s:<6} 帧={int(m.sum()):6d}（{m.mean()*100:5.1f}%）"
                  f" 其中 idle_now 真={int(idn[m].sum())}（{idn[m].mean()*100:5.1f}%）")
    base_lo = float(np.percentile(tp, 2))
    base_hi = float(np.percentile(tp, 20))
    lo_p = np.percentile(tp, 3)
    hi_p = np.percentile(tp, 97)
    amp = hi_p - lo_p
    idleref = float(np.median(tp[tp < lo_p + 0.25 * amp]))
    loadref = float(np.median(tp[tp > hi_p - 0.25 * amp]))
    p(f"[Q5] 本录制: 空载平台中位={idleref:.0f} 受载平台中位={loadref:.0f} "
      f"增量={loadref-idleref:.0f} 增量/基线={(loadref-idleref)/idleref:.3f}")
    p(f"[Q5] 本录制: 0.10·level_ref 判据需 ts_smooth < "
      f"{0.10*float(np.max(lrs)):.0f}（level_ref 全程最大 {lrs.max():.0f}）")
    p(f"[Q5] 本录制: 1.5·min_ts 判据需 ts_smooth < {1.5*minss.min():.0f}"
      f"（min_ts 全程最小 {minss.min():.0f}）")
    try:
        ds9 = L.load_dataset(L.DS_T9)
        t9p = ds9["pre"]["V"].sum(1)
        t9m = ds9["main"]["V"].sum(1)
        e9 = ds9["pre"]["el"]
        p(f"[Q5] T9 案例: frames={ds9['pre']['n']} span={e9[-1]-e9[0]:.1f}s "
          f"pre min={t9p.min():.0f} max={t9p.max():.0f} 中位={med(t9p):.0f}")
        l9 = float(np.percentile(t9p, 3))
        h9 = float(np.percentile(t9p, 97))
        ir9 = float(np.median(t9p[t9p < l9 + 0.25*(h9-l9)]))
        lr9 = float(np.median(t9p[t9p > h9 - 0.25*(h9-l9)]))
        p(f"[Q5] T9 案例: 空载中位={ir9:.0f} 受载中位={lr9:.0f} 增量={lr9-ir9:.0f} "
          f"增量/基线={(lr9-ir9)/ir9:.3f} | 偏移 中位={med(t9m-t9p):.0f} "
          f"范围={ (t9m-t9p).min():.0f}/{ (t9m-t9p).max():.0f}")
    except Exception as exc:  # noqa: BLE001
        p(f"[Q5] T9 读取失败: {type(exc).__name__}: {exc}")

    p(f"\n[ok] 用时 {time.time()-t_start:.1f}s")
    LOG.close()


if __name__ == "__main__":
    main()
