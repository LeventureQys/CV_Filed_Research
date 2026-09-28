# -*- coding: utf-8 -*-
"""T6-B：L3 路径抖动 + 状态机事件序列抖动（T6-Q2 / T6-Q3）——本任务最有诊断价值的实验。

原理：**同一份输入样本、只改到达时刻或加微扰**，算法输出与状态机事件的差异就**纯因果地**
      归因于算法自身，与"真实加载每次不同"（L1）完全无关。

三个场景（`--scene a|b`）：
  a = 实录 `中途切换-1d9493`（ADC 域，21 ch，63.85 s，含 onset/restep/unload 多事件）
  b = 恒载 `四指指尖/数据1` 前 80 s（力域，21 ch，含 onset + 平台 + 卸载）
包结构（实测，t6_probe.csv）：实录 40.0 ms/包、4.00 帧/包；指尖 16.7 ms/包、1.72 帧/包。

扰动条件（每条 ≥30 个不同随机种子）：
  timing_JxxP : **整包平移** δ_k~U(−J,J)，J = xx%·包周期，包内帧间隔不变，cummax 保序
  noise_1ADC / noise_5ADC : 带限白噪 [0.3,40] Hz，**总量 RMS = 1 / 5 ADC**
  dropout_1frame : 随机 1 帧丢包（零阶保持）

口径：100 Hz 均匀网格（timestamp 列）；输出差异 = |Y_jit.sum(1) − Y_base.sum(1)| 的逐帧 RMS，
      天然量用 ADC，相对量用 `%·记录中位电平` 与 `%·机械台阶 a_step`（两种都报，因为跨域不可比绝对）。
      `T_stable` 用 legacy 口径（tol 5%·step / hold 30 s；T1-A 未交付，**不可与 T1 绝对横比**）。

产出：results/t6_jitter_paths.csv、t6_jitter_summary.csv、t6_event_sequence.csv、
      results/_t6b_jitter_<scene>.log
"""
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import t6_common as C                                              # noqa: E402
import t6_ad_lib as AL                                             # noqa: E402
import t6_a_common as AC                                            # noqa: E402
from t6_glm53_v51 import GLM53v51                                  # noqa: E402
from t6_glm53_v6 import GLM53v6                                    # noqa: E402
from t6_glm53_v61 import GLM53v61                                  # noqa: E402

RES = C.RES
LOG = []
TRACED_V6 = AC.make_traced(GLM53v6)
TRACED_V61 = AC.make_traced(GLM53v61)
TRACED_V51 = AL.make_traced(GLM53v51)
IMPLS = ["v5.1", "v6", "v6.1"]
N_SEED = 30
SCENES = {
    "a": dict(tag="中途切换-1d9493", tmax=None, conds=[
        ("timing_J0P", 0.00), ("timing_J10P", 0.10), ("timing_J25P", 0.25),
        ("timing_J50P", 0.50),
        ("timing_J100P", 1.00), ("noise_1ADC", None), ("noise_5ADC", None),
        ("dropout_1frame", None)]),
    "b": dict(tag="四指指尖/数据1", tmax=80.0, conds=[
        ("timing_J0P", 0.00), ("timing_J25P", 0.25), ("timing_J100P", 1.00),
        ("noise_1ADC", None), ("noise_5ADC", None), ("dropout_1frame", None)]),
}


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def run_impl(impl, tu, Xu):
    """返回 (Y_tot, ledger)。ledger["seq"] = 时间排序的事件 token 列表。"""
    if impl == "v5.1":
        c = TRACED_V51(Xu.shape[1])
        Y = np.empty_like(Xu)
        for i in range(len(tu)):
            Y[i] = c.process(tu[i], Xu[i])
        ev = [(float(t), "E:" + str(k)) for t, k in
              zip(c.epoch_t, ["v5_begin"] * len(c.epoch_t))]
        return Y.sum(axis=1), ev, dict(n_epoch=len(c.epoch_t), n_revoke=0,
                                       n_handoff=0, n_unload=0)
    Cls = TRACED_V6 if impl == "v6" else TRACED_V61
    r = AC.run_traced(Cls, tu, Xu)
    ev = []
    for e in r["epoch"]:
        ev.append((float(e[0]), "E:" + str(e[1])))
    for e in r["revoke"]:
        ev.append((float(e[0]), "R"))
    for e in r["handoff"]:
        ev.append((float(e[0]), "H:" + str(e[1])))
    for t in r["unload"]:
        ev.append((float(t), "U"))
    ev.sort(key=lambda z: z[0])
    return r["Y"].sum(axis=1), ev, dict(n_epoch=len(r["epoch"]), n_revoke=len(r["revoke"]),
                                        n_handoff=len(r["handoff"]), n_unload=len(r["unload"]))


def lev(a, b):
    """token 序列的编辑距离（路径级比对，不看指标）。"""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
        prev = cur
    return prev[m]


def match_tokens(base, jit, tol=0.25):
    """按「类型相同 + 时刻差 ≤ tol s」贪心配对，返回 TP/FP/FN。"""
    used = [False] * len(jit)
    tp = 0
    for tb, kb in base:
        best, bd = -1, tol
        for i, (tj, kj) in enumerate(jit):
            if used[i] or kj != kb:
                continue
            dd = abs(tj - tb)
            if dd <= bd:
                best, bd = i, dd
        if best >= 0:
            used[best] = True
            tp += 1
    fp = int(np.sum(~np.array(used))) if used else len(jit)
    return tp, fp, len(base) - tp


def main():
    scene = "a"
    for k in ("--scene", "-s"):
        if k in sys.argv:
            scene = sys.argv[sys.argv.index(k) + 1]
    n_seed = N_SEED
    if "--seeds" in sys.argv:
        n_seed = int(sys.argv[sys.argv.index("--seeds") + 1])
    only = None
    if "--cond" in sys.argv:
        only = sys.argv[sys.argv.index("--cond") + 1]
    sc = SCENES[scene]
    tag, tmax = sc["tag"], sc["tmax"]
    conds = [c for c in sc["conds"] if only is None or c[0] == only]
    path = C.ALL[tag]
    d = C.grid(path, tmax=tmax)
    tu, Xu, dt = d["tu"], d["Xu"], d["dt"]
    tot = Xu.sum(axis=1)
    pid_raw, n_pkt, P, fpp = C.packet_layout(d["t"])
    pkt = C.grid_packet_id(tu, d["t"], pid_raw)
    dom = C.onset_geometry(tu, tot, dt) if C.KIND[tag] == "恒载" else None
    lvl_med = float(np.median(tot))
    a_step = float(dom["a_step"]) if dom else float(np.nan)
    p(f"scene={scene} tag={tag} n={len(tu)} ch={Xu.shape[1]} span={d['span']:.2f}s "
      f"packet_dt={P*1000:.2f}ms frames/pkt={fpp:.2f} n_pkt={n_pkt} lvl_med={lvl_med:.1f} "
      f"a_step={a_step} rank={C.DOMAIN[tag]}")

    base = {}
    for impl in IMPLS:
        y, ev, cnt = run_impl(impl, tu, Xu)
        base[impl] = dict(y=y, ev=ev, cnt=cnt)
        p(f"  baseline {impl:>5}: events={len(ev)} {cnt}")

    rows, seq_rows = [], []
    for ci, (cname, Jrel) in enumerate(conds):
        for seed in range(n_seed):
            rng = np.random.default_rng(10000 + 1000 * ci + seed)
            reorder = np.nan
            drop_idx = ""
            if cname.startswith("timing"):
                tj, reorder = C.jitter_axis(tu, pkt, P, Jrel * P, rng)
                Xj = Xu
            elif cname.startswith("noise"):
                tj = tu
                A = 1.0 if "1ADC" in cname else 5.0
                Xj = C.add_bad_noise(Xu, A, "band", rng)
            else:
                tj = tu
                Xj, di = C.dropout_frame(Xu, rng, 1)
                drop_idx = ";".join(str(i) for i in di)
            for impl in IMPLS:
                y, ev, cnt = run_impl(impl, tj, Xj)
                yb, evb, cntb = base[impl]["y"], base[impl]["ev"], base[impl]["cnt"]
                dd = y - yb
                rms = float(np.sqrt(np.mean(dd * dd)))
                absd = np.abs(dd)
                # 诊断：分歧从哪里开始、是持续偏移还是瞬态
                k0 = int(evb[0][0] / dt) if len(evb) else 0
                k0 = min(max(k0, 1), len(tu) - 1)
                thr05 = 0.005 * abs(lvl_med)
                hit = np.where(absd > thr05)[0]
                t_first_div = float(tu[hit[0]]) if len(hit) else float("nan")
                rms_pre = float(np.sqrt(np.mean(dd[:k0] ** 2))) if k0 > 1 else np.nan
                rms_post = float(np.sqrt(np.mean(dd[k0:] ** 2))) if k0 < len(tu) - 1 else np.nan
                fin_off = float(np.mean(dd[-int(5.0 / dt):]))
                tp, fp, fn = match_tokens(evb, ev)
                kb = [k for _, k in evb]
                kj = [k for _, k in ev]
                tt = None
                if dom is not None and not np.isnan(a_step):
                    tt = C.stable_time(tu, y, dom["i_e"], dom["step"], dt)
                rows.append(dict(
                    scene=scene, tag=tag, domain=C.DOMAIN[tag], impl=impl, cond=cname,
                    J_rel=Jrel if Jrel is not None else np.nan, seed=seed,
                    reorder_frac=reorder, drop_idx=drop_idx,
                    rms_adc=rms, rms_pct_lvl=100.0 * rms / max(abs(lvl_med), 1e-9),
                    rms_pre_event=rms_pre, rms_post_event=rms_post,
                    t_absmax=float(tu[int(np.argmax(absd))]),
                    t_first_div=t_first_div, final_offset=fin_off,
                    final_off_pct_lvl=100.0 * fin_off / max(abs(lvl_med), 1e-9),
                    rms_pct_step=(100.0 * rms / abs(a_step) if np.isfinite(a_step)
                                  and abs(a_step) > 1e-9 else np.nan),
                    max_abs_adc=float(absd.max()),
                    p95_abs_adc=float(np.percentile(absd, 95)),
                    max_abs_pct_lvl=100.0 * float(absd.max()) / max(abs(lvl_med), 1e-9),
                    T_stable_jit=tt,
                    T_stable_base=C.stable_time(tu, yb, dom["i_e"], dom["step"], dt)
                    if dom is not None else np.nan,
                    plat_jit=float(np.median(y[(tu >= tu[dom["i_e"]] + 40) &
                                               (tu <= tu[dom["i_e"]] + 60)])) if dom else np.nan,
                    plat_base=float(np.median(yb[(tu >= tu[dom["i_e"]] + 40) &
                                                 (tu <= tu[dom["i_e"]] + 60)])) if dom else np.nan,
                    n_epoch=cnt["n_epoch"], n_epoch_base=cntb["n_epoch"],
                    n_revoke=cnt["n_revoke"], n_revoke_base=cntb["n_revoke"],
                    n_handoff=cnt["n_handoff"], n_handoff_base=cntb["n_handoff"],
                    n_unload=cnt["n_unload"], n_unload_base=cntb["n_unload"],
                    seq_lev=lev(kb, kj), seq_same=int(kb == kj),
                    seq_tp=tp, seq_fp=fp, seq_fn=fn,
                    seq_base=";".join(f"{t:.2f}{k}" for t, k in evb),
                    seq_jit=";".join(f"{t:.2f}{k}" for t, k in ev)))
        sub = [r for r in rows if r["cond"] == cname]
        p(f"  cond={cname:>15} reorder={np.nanmean([r['reorder_frac'] for r in sub]):.3f} "
          + " | ".join(
              f"{im}: rms={np.median([r['rms_adc'] for r in sub if r['impl'] == im]):.1f}ADC "
              f"lev={np.median([r['seq_lev'] for r in sub if r['impl'] == im]):.1f}"
              for im in IMPLS))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, f"t6_jitter_paths_{scene}.csv"), index=False,
              encoding="utf-8-sig")
    # 事件序列比对表（路径级）
    sc_cols = ["scene", "tag", "impl", "cond", "seed", "seq_base", "seq_jit",
               "seq_lev", "seq_same", "seq_tp", "seq_fp", "seq_fn",
               "n_epoch", "n_epoch_base", "n_revoke", "n_revoke_base",
               "n_handoff", "n_handoff_base", "n_unload", "n_unload_base"]
    df[sc_cols].to_csv(os.path.join(RES, f"t6_event_sequence_{scene}.csv"), index=False,
                       encoding="utf-8-sig")

    # 汇总（抖动-输出差异曲线）
    agg = []
    for (im, cd), g in df.groupby(["impl", "cond"]):
        a = dict(scene=scene, impl=im, cond=cd, n=len(g),
                 J_rel=float(g.J_rel.iloc[0]) if np.isfinite(g.J_rel.iloc[0]) else np.nan)
        for k in ("rms_adc", "rms_pct_lvl", "rms_pct_step", "max_abs_pct_lvl",
                  "p95_abs_adc", "T_stable_jit", "n_epoch", "n_revoke", "n_handoff",
                  "n_unload", "seq_lev", "seq_fp", "seq_fn", "reorder_frac",
                  "rms_pre_event", "rms_post_event", "t_first_div", "final_offset",
                  "final_off_pct_lvl"):
            v = g[k].to_numpy(float)
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            a[k + "_med"] = float(np.median(v))
            a[k + "_p10"] = float(np.percentile(v, 10))
            a[k + "_p90"] = float(np.percentile(v, 90))
        a["rms_adc_mean"] = float(g.rms_adc.mean())
        a["rms_adc_std"] = float(g.rms_adc.std(ddof=1))
        a["seq_same_rate"] = float(g.seq_same.mean())
        a["n_epoch_changed_rate"] = float((g.n_epoch != g.n_epoch_base).mean())
        a["n_revoke_changed_rate"] = float((g.n_revoke != g.n_revoke_base).mean())
        a["n_handoff_changed_rate"] = float((g.n_handoff != g.n_handoff_base).mean())
        a["n_unload_changed_rate"] = float((g.n_unload != g.n_unload_base).mean())
        agg.append(a)
    ad = pd.DataFrame(agg)
    ad.to_csv(os.path.join(RES, f"t6_jitter_summary_{scene}.csv"), index=False,
              encoding="utf-8-sig")
    p("")
    p("=" * 118)
    p(f"表 L3（scene={scene}）  微扰 → 输出轨线差异 / 事件序列改动（各条件 n={n_seed}）")
    p("=" * 118)
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        p(ad[["impl", "cond", "n", "rms_adc_med", "rms_adc_p90", "rms_pct_lvl_med",
              "max_abs_pct_lvl_med", "seq_lev_med", "seq_same_rate",
              "n_epoch_changed_rate", "n_revoke_changed_rate", "n_handoff_changed_rate",
              "n_unload_changed_rate"]].round(4).to_string(index=False))
    with io.open(os.path.join(RES, f"_t6b_jitter_{scene}.log"), "w", encoding="utf-8") as fh:
        fh.write(f"T6-B jitter log (scene={scene})\n" + "\n".join(LOG) + "\n")
    p(f"\n-> results/t6_jitter_paths_{scene}.csv / t6_jitter_summary_{scene}.csv / "
      f"t6_event_sequence_{scene}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
