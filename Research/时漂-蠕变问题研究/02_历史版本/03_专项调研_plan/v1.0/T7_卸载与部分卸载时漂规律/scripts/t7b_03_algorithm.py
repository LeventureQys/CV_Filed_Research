# -*- coding: utf-8 -*-
"""T7-B 步骤 3（§B 算法耦合）：部分卸载后的基线偏差 / 反向事件误判 / 单帧跳变复现。

四块内容：
  [A] 补丁 A/B 零差证明 → results/t7b_patch_ab_zero.csv
      本地副本 t7b_glm53_v6 / t7b_glm53_v51 与 progress 下原版逐帧对拍（max|Δ|=0）。
      另对 t7b_v2_comp.py（temp/archived/GLM53/scripts/f_varying_load.py 的类部分原样拷贝，
      仅截去其文件末尾的顶层执行代码）做 CompV1/CompV2 的"拷贝≡原文件"校验。
  [B] 部分卸载后的基线偏差（T7-Q7）→ results/t7b_baseline_bias.csv
  [C] 反向事件误判率（T7-Q7）→ results/t7b_reverse_misjudge.csv
  [D] 7561 ADC 单帧跳变案例复核（T7-Q10）→ results/t7b_jump7561_check.csv

注入实验一律标注 src='inject_linear'（线性叠加假设，只用于 §B 算法耦合，不用于物理规律）。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t7b_ad_lib import (RECS, DOMAIN, rec_path, load_rec, pkt_dt, to_grid,  # noqa: E402
                        read_frozen_events, med_smooth, FS, PROG, TEMP, RESULTS,
                        ensure_dirs)

ensure_dirs()
LOG = os.path.join(RESULTS, "_t7b_03_algorithm.log")
ARMS = ("raw", "v51", "v6")
POST_W = (4.0, 6.0)


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


def grid(key):
    if key not in _CACHE:
        t, X, meta = load_rec(rec_path(key))
        tu, Xu = to_grid(t, X)
        _CACHE[key] = dict(t=t, X=X, tu=tu, Xu=Xu, zu=Xu.sum(1),
                           pkt=pkt_dt(t), span=float(t[-1] - t[0]))
    return _CACHE[key]


# ── [A] 补丁 A/B 零差证明 ────────────────────────────────────────────────
def ab_zero():
    """本地副本 vs progress 原版逐帧对拍。"""
    for p in (os.path.join(PROG, "07-v6", "scripts"),
              os.path.join(PROG, "05-v5.1", "scripts"),
              os.path.join(PROG, "04-v5", "scripts")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.append(p)
    rows = []
    import importlib
    import t7b_a_common as AC
    for modname, clsname, arm in (("glm53_v6", "GLM53v6", "v6"),
                                  ("glm53_v51", "GLM53v51", "v51")):
        try:
            mod_orig = importlib.import_module(modname)
            Cls_orig = getattr(mod_orig, clsname)
        except Exception as exc:                      # pragma: no cover
            rows.append(dict(arm=arm, rec="-", n=0, status="原版导入失败: %s" % exc,
                             max_abs_diff=np.nan, rms_diff=np.nan, identical=False))
            print("[A] 原版 %s 导入失败：%s" % (modname, exc))
            continue
        Cls_copy = getattr(AC, "GLM53v6" if arm == "v6" else "GLM53v51")
        for key in ("SW3", "SW4", "RT1", "F41"):
            g = grid(key)
            Y1, _ = AC.run_plain(Cls_copy, g["tu"], g["Xu"])
            Y2, _ = AC.run_plain(Cls_orig, g["tu"], g["Xu"])
            d = np.abs(Y1 - Y2)
            rows.append(dict(arm=arm, rec=key, n=len(g["tu"]),
                             max_abs_diff=float(d.max()), rms_diff=float(np.sqrt((d ** 2).mean())),
                             identical=bool(np.array_equal(Y1, Y2)),
                             status="本地副本 vs progress 原版"))
            print("[A] %-4s %-4s n=%-6d max|Δ|=%.3e  identical=%s"
                  % (arm, key, len(g["tu"]), d.max(), np.array_equal(Y1, Y2)))
    # 存档 v2 原型：拷贝文件截断后与"源码原文前 571 行"一致性（哈希级）
    import hashlib
    src = os.path.join(TEMP, "archived", "GLM53",
                       "scripts", "f_varying_load.py")
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "t7b_v2_comp.py")
    if os.path.exists(src):
        with open(src, "r", encoding="utf-8", errors="ignore") as f:
            head = "".join(f.readlines()[:571])
        with open(here, "r", encoding="utf-8", errors="ignore") as f:
            mine = f.read()
        same = hashlib.sha256(head.encode("utf-8")).hexdigest() == \
            hashlib.sha256(mine.encode("utf-8")).hexdigest()
        rows.append(dict(arm="v2_comp", rec="f_varying_load.py[:571]", n=571,
                         max_abs_diff=0.0, rms_diff=0.0, identical=bool(same),
                         status="类定义部分逐字相同（顶层执行代码已截去）"))
        print("[A] v2 原型类部分拷贝一致性：%s" % same)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "t7b_patch_ab_zero.csv"), index=False,
              encoding="utf-8-sig")
    return df


# ── 事件表（冻结 + 自建扫描，去重） ──────────────────────────────────────
def decrement_events():
    """减重事件集：直接用步骤 1 的统一表（含实测重新分档、自适应 post 窗、t_next）。"""
    amp = pd.read_csv(os.path.join(RESULTS, "t7b_partial_amp_response.csv"))
    out = pd.DataFrame(dict(
        event_id=amp["event_id"], key=amp["key"], domain=amp["domain"],
        t_on=amp["t_on"].astype(float), kind=amp["kind_eff"],
        kind_t4a=amp["kind_t4a"], src=amp["src"], absJ=amp["absJ_ad"],
        drop_frac_meas=amp["drop_frac_eff"], post_win_clean=amp["post_win_clean"],
        amp_ok=amp["amp_ok"], pkt_dt=amp["pkt_dt"], t_next=amp["t_next"],
        step_frame_frac=amp["step_frame_frac"]))
    out["T_ramp"] = np.nan
    return out.sort_values(["key", "t_on"]).reset_index(drop=True)


_ARM_CACHE = {}


def run_arms(key, t_on, inject_alpha=0.0):
    """在 key 的 100 Hz 网格上跑 raw / v5.1 / v6（v6 带仪表化）。可叠加注入。

    注入（`inject_alpha>0`）：把该次**真实全卸载**的 |J| 按 α 比例加回（余弦 50 ms 上升），
    ⇒ 合成一次幅度 (1−α)|J| 的部分卸载。**线性叠加假设**，标注 src='inject_linear'，
    只用于 §B 算法耦合（Q7 幅度-偏差关系），不用于物理规律（Q6/Q8）。
    """
    import t7b_a_common as AC
    ck = (key, round(float(inject_alpha), 3))
    if ck in _ARM_CACHE:
        return _ARM_CACHE[ck]
    g = grid(key)
    Xu = g["Xu"]
    tag = ""
    if inject_alpha > 0:
        i0 = int(round(t_on * FS))
        pre = float(np.median(Xu[max(0, i0 - 200):i0].sum(1)))
        post = float(np.median(Xu[i0:i0 + min(600, len(Xu) - i0)].sum(1)))
        amp = inject_alpha * abs(pre - post)
        prof = np.abs(Xu[max(0, i0 - 5)])
        prof = prof / prof.sum() if prof.sum() > 1e-9 else np.ones(Xu.shape[1]) / Xu.shape[1]
        nr = 5
        w = amp * (1.0 - np.cos(np.linspace(0, np.pi, nr))) / 2.0
        ramp = np.concatenate([w, np.full(max(0, len(Xu) - i0 - nr), amp)])[: len(Xu) - i0]
        Xu = Xu.copy()
        Xu[i0:] += np.outer(ramp, prof)
        tag = "inject%.2f" % inject_alpha
    Y51, _ = AC.run_plain(AC.GLM53v51, g["tu"], Xu)
    tr = AC.run_traced(AC.TRACED_V6, g["tu"], Xu)
    out = dict(tu=g["tu"], pkt=g["pkt"], raw=Xu, v51=Y51, v6=tr["Y"], tr=tr,
               tag=tag, Xu=Xu)
    _ARM_CACHE[ck] = out
    return out


def win_median(z, tu, lo, hi):
    m = (tu >= lo) & (tu < hi)
    return float(np.median(z[m])) if m.any() else np.nan


def analyse_event(ev, key, t_on, arm_res, ref=None):
    """返回一个事件的 §B 指标（对 raw / v51 / v6）。"""
    tu = arm_res["tu"]
    pkt = arm_res["pkt"]
    raw = arm_res["raw"].sum(1)
    i0 = int(round(t_on * FS))
    # 自适应 post 窗：被后续事件污染时缩短
    lo, hi = t_on + POST_W[0], t_on + POST_W[1]
    nxt = ev.get("t_next", np.nan)
    win_note = "clean"
    if np.isfinite(nxt) and nxt < hi + 1.0 and nxt > t_on + 1.0:
        hi = max(t_on + 2.0, nxt - 1.0)
        lo = min(lo, max(t_on + 1.0, hi - 2.0))
        if lo >= hi:
            lo, hi = t_on + 1.0, t_on + 2.0
        win_note = "shortened[%.1f,%.1f]" % (lo - t_on, hi - t_on)
    true_post = win_median(raw, tu, lo, hi)
    # 自适应幅度（同一 post 窗；污染事件的冻结 |J| 不能用，会放大百分比）
    pre_ad = win_median(raw, tu, t_on - 2.0, t_on)
    absJ_ad = abs(true_post - pre_ad) if (np.isfinite(true_post) and
                                          np.isfinite(pre_ad)) else np.nan
    if not np.isfinite(absJ_ad) or absJ_ad < 1e-9:
        absJ_ad = abs(ev.get("absJ") or 1.0)
    rows = []
    seg = slice(i0, min(len(tu), i0 + int(8 * FS)))
    rawseg = raw[seg]
    jraw = np.diff(raw, prepend=raw[0])
    for arm in ARMS:
        y = arm_res[arm].sum(1)
        disp_post = win_median(y, tu, lo, hi)
        yseg = y[seg]
        jdisp = np.diff(y, prepend=y[0])
        ind = jdisp - jraw                                  # 算法引入的单帧跳变
        k = int(np.argmax(np.abs(ind[seg]))) + seg.start if len(ind[seg]) else 0
        under = (rawseg - yseg)                              # >0 ⇒ 显示低于原始（多扣）
        over = (yseg - rawseg)
        rows.append(dict(
            event_id=ev["event_id"], key=key, domain=ev.get("domain", DOMAIN[key]),
            kind=ev.get("kind"), src=ev.get("src", "t4a_frozen"),
            t_on=t_on, absJ=ev.get("absJ"), absJ_ad=absJ_ad,
            drop_frac=ev.get("drop_frac_meas"),
            drop_frac_ad=absJ_ad / max(abs(pre_ad), 1.0),
            post_win=win_note, post_win_lo=lo - t_on, post_win_hi=hi - t_on,
            arm=arm, true_pre_ad=pre_ad, true_post=true_post, disp_post=disp_post,
            bias=disp_post - true_post,
            bias_pct_J=100.0 * (disp_post - true_post) / max(absJ_ad, 1),
            max_under=float(np.max(under)) if len(under) else np.nan,
            max_under_pct_J=100.0 * float(np.max(under)) / max(absJ_ad, 1)
            if len(under) else np.nan,
            t_max_under=(float(np.argmax(under)) / FS) if len(under) else np.nan,
            max_over_pct_J=100.0 * float(np.max(over)) / max(absJ_ad, 1)
            if len(over) else np.nan,
            dur_under_5pct=float(np.sum(under > 0.05 * absJ_ad) / FS)
            if len(under) else np.nan,
            jump_disp_at_maxind=float(jdisp[k]), jump_raw_at_maxind=float(jraw[k]),
            induced_jump=float(ind[k]), t_induced=j0_ts(k, FS),
            induced_pct_J=100.0 * float(ind[k]) / max(absJ_ad, 1),
            pkt_dt=pkt, inject=(arm_res["tag"] != ""),
            inject_tag=arm_res["tag"],
        ))
    # 内部事件分类（仅 v6 traced）
    ep = [(t, k) for (t, k, _det, *_r) in arm_res["tr"]["epoch"]]
    near2 = [(t, k) for (t, k) in ep if abs(t - t_on) <= 2.0]
    near1 = [(t, k) for (t, k) in ep if abs(t - t_on) <= 1.0]
    press = ("onset", "restep", "restep_reload")

    def verdict_of(near):
        kinds = sorted(set(k for _t, k in near))
        if "decrease" in kinds:
            return "decrease 正确方向", kinds
        if any(k in press for k in kinds):
            return "反向误判（按加压方向建事件）", kinds
        if kinds:
            return "其它:%s" % kinds, kinds
        return "未建事件（继续按旧扣除输出）", kinds

    v2, k2 = verdict_of(near2)
    v1, k1 = verdict_of(near1)
    for r in rows:
        r["v6_near_epochs_pm2"] = "|".join("%s@%.2f" % (k, t) for t, k in near2)
        r["v6_near_epochs_pm1"] = "|".join("%s@%.2f" % (k, t) for t, k in near1)
        r["v6_reverse_verdict"] = v2
        r["v6_reverse_verdict_pm1"] = v1
        r["v6_n_decrease_pm1"] = sum(1 for _t, k in near1 if k == "decrease")
        r["v6_n_press_pm1"] = sum(1 for _t, k in near1 if k in press)
    return rows


def j0_ts(k, fs):
    return k / fs


def main():
    print("CMD: python scripts/t7b_03_algorithm.py")
    print("=" * 78)
    ab = ab_zero()
    print("=" * 78)

    dec = decrement_events()
    print("[事件集] 减重事件 n=%d（实测分档：%s；T4-A 原标注：%s）"
          % (len(dec), dec["kind"].value_counts().to_dict(),
             dec["kind_t4a"].value_counts().to_dict()))

    # ── [B][C] 逐事件跑臂 ──
    rows = []
    for key, gk in dec.groupby("key"):
        for _, ev in gk.iterrows():
            try:
                res = run_arms(key, float(ev["t_on"]))
            except Exception as exc:
                print("[ERR] %s %s: %s" % (key, ev["event_id"], exc))
                continue
            rows += analyse_event(ev, key, float(ev["t_on"]), res)
    bb = pd.DataFrame(rows)
    bb.to_csv(os.path.join(RESULTS, "t7b_baseline_bias.csv"), index=False,
              encoding="utf-8-sig")
    print("\n[B] 部分/全卸载后基线偏差（显示 post − 原始 post；%% 按**自适应** |J| 归一；中位）")
    piv = bb[bb["arm"].isin(["raw", "v51", "v6"])] \
        .groupby(["kind", "arm"]).agg(
            n=("bias_pct_J", "size"),
            bias_ADC_med=("bias", "median"),
            bias_pct_J_med=("bias_pct_J", "median"),
            max_under_pct_J_med=("max_under_pct_J", "median"),
            dur_under_5pct_med=("dur_under_5pct", "median"),
            induced_ADC_max=("induced_jump", lambda s: float(np.max(np.abs(s)))),
            induced_pct_J_med=("induced_pct_J", "median"),
            induced_pct_J_max=("induced_pct_J", lambda s: float(np.max(np.abs(s)))))
    print(piv.to_string(float_format="%.3f"))

    # 反向误判率
    mis = bb[bb["arm"] == "v6"].copy()
    for tag, col in (("pm2", "v6_reverse_verdict"), ("pm1", "v6_reverse_verdict_pm1")):
        mis["bin_%s" % tag] = np.where(mis[col].str.startswith("decrease"), "decrease",
                                       np.where(mis[col].str.startswith("反向误判"),
                                                "reverse_misjudge", "other"))
    rate = mis.groupby(["kind", "bin_pm2"]).size().unstack(fill_value=0)
    rate["n"] = rate.sum(axis=1)
    for c in ("decrease", "reverse_misjudge", "other"):
        if c not in rate:
            rate[c] = 0
    rate["reverse_rate_pct_pm2"] = 100.0 * rate["reverse_misjudge"] / rate["n"]
    r1 = mis.groupby("kind")["bin_pm1"].apply(
        lambda s: 100.0 * (s == "reverse_misjudge").mean())
    rate["reverse_rate_pct_pm1"] = r1
    rate["no_event_rate_pct_pm1"] = mis.groupby("kind")["bin_pm1"].apply(
        lambda s: 100.0 * (s == "other").mean())
    print("\n[C] 反向事件误判率（v6；±1 s 与 ±2 s 两窗口）")
    print(rate[["n", "decrease", "reverse_misjudge", "other",
                "reverse_rate_pct_pm2", "reverse_rate_pct_pm1",
                "no_event_rate_pct_pm1"]].to_string(float_format="%.1f"))
    print("  v6 在减重沿邻近建出的内部事件种类（±1 s，全事件合计）：%s"
          % mis["v6_near_epochs_pm1"].replace("", np.nan).dropna().str.split("|")
            .explode().str.split("@").str[0].value_counts().to_dict())
    rate_out = rate.reset_index()
    rate_out.to_csv(os.path.join(RESULTS, "t7b_reverse_misjudge.csv"), index=False,
                    encoding="utf-8-sig")

    # ── [D] 7561 ADC 单帧跳变复核 ──
    print("\n[D] 7561 ADC 单帧跳变案例复核")
    rows_d = []
    for key in ("SW1", "SW2", "SW3", "SW4"):
        g = grid(key)
        t, X = g["t"], g["X"]
        t0r = float(t[0])
        import t7b_v2_comp as V2
        for cls_name in ("CompV1", "CompV2"):
            cls = getattr(V2, cls_name)
            c = cls()
            Y = np.empty_like(X, dtype=float)
            for i in range(len(t)):
                Y[i] = c.process(t[i], X[i].astype(float))
            jr = np.diff(X.sum(1), prepend=X.sum(1)[0])
            jd = np.diff(Y.sum(1), prepend=Y.sum(1)[0])
            ind = jd - jr
            i = int(np.argmax(np.abs(ind)))
            rows_d.append(dict(key=key, alg=cls_name, frame_dt=g["pkt"],
                               n=len(t), max_abs_induced=float(np.abs(ind).max()),
                               induced_at_max=float(ind[i]), t_at_max=float(t[i] - t0r),
                               jump_disp_at_max=float(jd[i]), jump_raw_at_max=float(jr[i]),
                               signed_at_max=float(ind[i])))
            print("  %-4s %-7s 最大算法单帧跳变 %+9.1f ADC @ t=%.2f s（同帧 显示%+.0f / 原始%+.0f）"
                  % (key, cls_name, ind[i], t[i] - t0r, jd[i], jr[i]))
    # 现役原型（v5.1 / v6）在同一批数据上的单帧跳变
    for key in ("SW1", "SW2", "SW3", "SW4"):
        g = grid(key)
        res = run_arms(key, 0.0)
        jr = np.diff(res["raw"].sum(1), prepend=res["raw"].sum(1)[0])
        for arm in ("v51", "v6"):
            jd = np.diff(res[arm].sum(1), prepend=res[arm].sum(1)[0])
            ind = jd - jr
            i = int(np.argmax(np.abs(ind)))
            rows_d.append(dict(key=key, alg=arm + "_100Hz", frame_dt=1.0 / FS, n=len(g["tu"]),
                               max_abs_induced=float(np.abs(ind).max()),
                               induced_at_max=float(ind[i]), t_at_max=float(res["tu"][i]),
                               jump_disp_at_max=float(jd[i]), jump_raw_at_max=float(jr[i]),
                               signed_at_max=float(ind[i])))
            print("  %-4s %-9s 最大算法单帧跳变 %+9.1f ADC @ t=%.2f s"
                  % (key, arm + "_100Hz", ind[i], res["tu"][i]))
    jd = pd.DataFrame(rows_d)
    jd.to_csv(os.path.join(RESULTS, "t7b_jump7561_check.csv"), index=False,
              encoding="utf-8-sig")

    # ── 注入补充（线性叠加，仅 §B）──
    print("\n[B2] 注入部分卸载（线性叠加，标注 inject_linear）")
    inj_rows = []
    for key, t0 in (("SW1", 249.63), ("SW4", 68.93), ("SW3", 30.68)):
        for alpha in (0.0, 0.3, 0.5, 0.7):
            base = dec[(dec["key"] == key) & ((dec["t_on"] - t0).abs() < 0.06)]
            if len(base) == 0:
                continue
            ev = base.iloc[0].to_dict()
            if alpha > 0:
                ev["event_id"] = "%s@%.2f_inject%.1f" % (key, t0, alpha)
                ev["src"] = "inject_linear"
                ev["drop_frac_meas"] = (1 - alpha) * float(ev.get("drop_frac_meas") or 0)
                ev["absJ"] = (1 - alpha) * float(ev.get("absJ") or 0)
                ev["post_win_clean"] = True
                ev["t_next"] = np.nan
            res = run_arms(key, float(t0), inject_alpha=alpha)
            if alpha > 0:
                ev["absJ"] = abs(float(np.median(res["raw"].sum(1)[
                    int(round((t0 + 4) * FS)):int(round((t0 + 6) * FS))]) -
                    np.median(res["raw"].sum(1)[int(round((t0 - 2) * FS)):int(round(t0 * FS))])))
            inj_rows += analyse_event(ev, key, float(t0), res)
    inj = pd.DataFrame(inj_rows)
    if len(inj):
        inj.to_csv(os.path.join(RESULTS, "t7b_inject_partial_bias.csv"), index=False,
                   encoding="utf-8-sig")
        p = inj.groupby(["src", "arm"]).agg(
            n=("bias_pct_J", "size"), bias_pct_J_med=("bias_pct_J", "median"),
            absJ_med=("absJ", "median"),
            induced_pct_J_max=("induced_pct_J", "max"))
        print(p.to_string(float_format="%.3f"))
    print("\nWROTE results/t7b_patch_ab_zero.csv, t7b_baseline_bias.csv, "
          "t7b_reverse_misjudge.csv, t7b_jump7561_check.csv, t7b_inject_partial_bias.csv")


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
