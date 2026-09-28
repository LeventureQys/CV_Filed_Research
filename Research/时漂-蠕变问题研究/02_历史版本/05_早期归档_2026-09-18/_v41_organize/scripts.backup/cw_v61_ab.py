# -*- coding: utf-8 -*-
"""v6 / v6.1 A/B：13 份录制 × 多档参数，指标口径与 `ci_v6_vs_1s_3s.py`、`pv_run.py` 逐条一致
（直接复用 `paper_v6/scripts/pv_common.py` 的指标函数），另加"尖峰"专属口径。

用法：
  python cw_v61_ab.py                 # v6 vs v6.1(默认参数)
  python cw_v61_ab.py --sweep rom     # ROM_SCALE 扫描
  python cw_v61_ab.py --sweep down    # RATE_DOWN 扫描

产物：results/v61_ab_<tag>.csv、results/_v61_ab_<tag>.log
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(OUT)
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(OUT, "paper_v6", "scripts"))
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                          # noqa: E402
import pv_common as C                                       # noqa: E402
from glm53_v6 import GLM53v6                                # noqa: E402
from glm53_v61 import GLM53v61                              # noqa: E402

B = os.path.join(TEMP, "变化负载")


def run_arm(cls, tu, Xu, **kw):
    c = cls(Xu.shape[1])
    for k, val in kw.items():
        setattr(c, k, val)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    c.A = np.full(Xu.shape[1], c.A_peak)
    return Y, c


def spike_metrics(tag, kind, tu, dtm, c, rs, ys, raw_inst=None):
    """阶跃后的"冲到真值之上 + 之后回落"口径（同 cv_v61_overshoot.py）。
    只统计"真阶跃"：step ≥ max(5% × 该录制峰值, 3 × 噪声) —— 排除录制起点附近的伪事件
    （录制开始时已带载，t0 处 step≈0，占比会被放大成几百 %）。"""
    n = len(tu)
    peak_all = float(rs.max())
    min_step = 0.05 * peak_all
    rows = []
    for t0, kd in c.kind_log:
        i0 = int(np.searchsorted(tu, t0))
        i_pre = max(0, i0 - int(0.3 / dtm))
        pre = float(np.median(rs[i_pre:max(i_pre + 1, i0)]))
        a = min(n - 1, i0 + int(4.6 / dtm))
        b = min(n, i0 + int(5.4 / dtm))
        if b - a < 3:
            continue
        P = float(np.median(rs[a:b]))
        step = P - pre
        if step < min_step:
            continue
        j5 = min(n - 1, i0 + int(5.0 / dtm))
        seg = ys[i0:j5 + 1]
        if len(seg) < 5:
            continue
        k = int(np.argmax(seg))
        peak = float(seg[k])
        # 回落窗：峰后 6 s，但**卸载沿即截止**（瞬时原始掉到窗内峰值 85% 以下 = 载荷在卸），
        # 否则窗口里"显示跟着掉到 0"会把回落量算成几百 %（与尖峰无关）
        e = min(n, i0 + k + int(6.0 / dtm))
        if raw_inst is not None:
            win = np.asarray(raw_inst, float)[i0 + k:e]
            if len(win):
                cut = np.where(win < 0.85 * win.max())[0]
                if len(cut):
                    e = max(i0 + k + int(cut[0]), i0 + k + 1)
        else:
            cut = np.where(rs[i0 + k:e] < 0.7 * P)[0]
            if len(cut):
                e = max(i0 + k + int(cut[0]), i0 + k + 1)
        trough = float(np.min(ys[i0 + k:e])) if e > i0 + k else peak
        rows.append(dict(t0=t0, ev=kd, step=step,
                         over_peak=100 * (peak - P) / step,
                         over_5s=100 * (float(ys[j5]) - P) / step,
                         transient=100 * (peak - trough) / step))
    df = pd.DataFrame(rows)
    if not len(df):
        return dict(sp_over_max=np.nan, sp_over_med=np.nan, sp_tr_max=np.nan,
                    sp_tr_med=np.nan, sp_over_onset_max=np.nan, sp_over_onset_med=np.nan,
                    sp_tr_onset_max=np.nan)
    on = df[df.ev == "onset"]
    return dict(sp_over_max=float(df.over_peak.max()), sp_over_med=float(df.over_peak.median()),
                sp_tr_max=float(df.transient.max()), sp_tr_med=float(df.transient.median()),
                sp_over_onset_max=float(on.over_peak.max()) if len(on) else np.nan,
                sp_over_onset_med=float(on.over_peak.median()) if len(on) else np.nan,
                sp_tr_onset_max=float(on.transient.max()) if len(on) else np.nan)


def eval_variant(name, cls, kw, data):
    rows, settle = [], []
    for tag, d, kind in data:
        tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
        tot_s, peak = d["tot_s"], d["peak"]
        Y, c = run_arm(cls, tu, Xu, **kw)
        mm = (C.hold_metrics(Y, c, tu, Xu, dtm, tot_s, peak) if kind == "恒载"
              else C.vary_metrics(Y, c, tu, Xu, dtm, tot_s, peak, d))
        rs = L.med_smooth(Xu.sum(axis=1), 0.5 / dtm)
        ys = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
        mm.update(spike_metrics(tag, kind, tu, dtm, c, rs, ys, Xu.sum(axis=1)))
        mm.update(dataset=tag, kind=kind, algo=name)
        rows.append(mm)
        if d["i0"] is not None:
            i0, tgt, step = d["i0"], d["tgt"], d["step"]
            settle.append(dict(dataset=tag, kind=kind, algo=name,
                               t_settle=C.settle_time(tu, Y.sum(axis=1), i0, tgt, step, dtm),
                               t_stable=C.stable_time(tu, Y.sum(axis=1), i0, step, dtm),
                               err5s=float(Y.sum(axis=1)[i0 + int(5.0 / dtm)] / tgt * 100 - 100)))
        print(f"    {name:>14} | {tag:>18} [{kind}] ok", flush=True)
    dfo = pd.DataFrame(rows)
    sto = pd.DataFrame(settle)
    dfo.to_csv(os.path.join(RES, f"v61_arm_{_slug(name)}.csv"), index=False, encoding="utf-8-sig")
    sto.to_csv(os.path.join(RES, f"v61_arm_{_slug(name)}_settle.csv"), index=False, encoding="utf-8-sig")
    return dfo, sto


def _slug(name):
    return name.replace(".", "").replace("/", "_")


def load_cached(name):
    p = os.path.join(RES, f"v61_arm_{_slug(name)}.csv")
    q = os.path.join(RES, f"v61_arm_{_slug(name)}_settle.csv")
    if os.path.exists(p) and os.path.exists(q):
        return pd.read_csv(p), pd.read_csv(q)
    return None, None


def summarize(df, st):
    h, v = df[df.kind == "恒载"], df[df.kind == "实采"]
    a = h.groupby("algo").agg(**{
        "drift_main": ("drift_main", lambda s: s.abs().mean()),
        "drift_slow": ("drift_slow", lambda s: s.abs().mean()),
        "drift_loaded": ("drift_loaded", lambda s: s.abs().mean()),
        "flat": ("flat", "mean"), "step_ratio": ("step_ratio", "mean"),
        "noise_ratio": ("noise_ratio", "mean"), "epoch": ("epoch", "sum"),
        "sp_over_onset_med": ("sp_over_onset_med", "median"),
        "sp_over_onset_max": ("sp_over_onset_max", "max"),
        "sp_tr_onset_max": ("sp_tr_onset_max", "max")}).round(3)
    b = v.groupby("algo").agg(**{
        "max_gap": ("max_gap", "median"), "gap_med": ("gap_med", "median"),
        "cap_med": ("cap_med", "mean"), "cap_min": ("cap_min", "min"),
        "epoch": ("epoch", "sum"),
        "sp_over_max": ("sp_over_max", "max"), "sp_tr_max": ("sp_tr_max", "max")}).round(3)
    s = (st.assign(a5=st.err5s.abs()).groupby(["kind", "algo"])
         .agg(T_stable=("t_stable", "median"), T_band=("t_settle", "median"),
              err5_med=("a5", "median"), err5_max=("a5", "max")))
    return a, b, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=None, choices=[None, "rom", "down", "stalltail", "hold"])
    ap.add_argument("--rom", default=None, help="逗号分隔的 ROM_SCALE 列表")
    ap.add_argument("--down", default=None, help="逗号分隔的 RATE_DOWN 列表")
    ap.add_argument("--onsetonly", default=None, help="逗号分隔的 0/1（只对 onset 保守）")
    ap.add_argument("--hold", default=None, help="逗号分隔的 STALL_HOLD_S")
    ap.add_argument("--tail", default=None, help="逗号分隔的 STALL_TAIL_KEEP")
    ap.add_argument("--arms", default="v6", help="v6 / v61 / both")
    ap.add_argument("--tag", default=None, help="输出文件名后缀")
    ap.add_argument("--nocache", action="store_true", help="忽略逐档缓存，强制重算")
    args = ap.parse_args()

    data = []
    for tag, path in C.ALL:
        if not os.path.exists(path):
            print(f"[skip] 缺文件 {tag}")
            continue
        d = L.prep(path)
        tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
        tot = Xu.sum(axis=1)
        d["tot_s"] = L.med_smooth(tot, 0.5 / dtm)
        d["peak"] = float(tot.max())
        fo = C.first_onset(tu, tot, dtm)
        d["i0"] = fo[0] if fo else None
        d["tgt"] = float(np.median(tot[d["i0"] + int(4.6 / dtm):d["i0"] + int(5.4 / dtm)])) if fo else np.nan
        d["step"] = (d["tgt"] - fo[1]) if fo else np.nan
        data.append((tag, d, C.KIND[tag]))
    print(f"数据 {len(data)} 份，逐档复算…")

    variants = []
    if args.arms in ("v6", "both"):
        variants.append(("v6", GLM53v6, {}))
    if args.rom and args.onsetonly:
        for s in [float(x) for x in args.rom.split(",")]:
            for o in [int(x) for x in args.onsetonly.split(",")]:
                variants.append((f"v61_s{s:.2f}_o{o}", GLM53v61,
                                 dict(ROM_SCALE=s, RATE_DOWN=0.25,
                                      CONSERVATIVE_ONSET_ONLY=bool(o))))
    elif args.rom:
        for s in [float(x) for x in args.rom.split(",")]:
            variants.append((f"v61_s{s:.2f}", GLM53v61, dict(ROM_SCALE=s, RATE_DOWN=0.25)))
    elif args.onsetonly:
        for o in [int(x) for x in args.onsetonly.split(",")]:
            variants.append((f"v61_o{o}", GLM53v61,
                             dict(ROM_SCALE=1.08, RATE_DOWN=0.25,
                                  CONSERVATIVE_ONSET_ONLY=bool(o))))
    if args.down:
        for r in [float(x) for x in args.down.split(",")]:
            variants.append((f"v61_r{r:.2f}", GLM53v61, dict(ROM_SCALE=1.08, RATE_DOWN=r)))
    if args.hold:
        for hs in [float(x) for x in args.hold.split(",")]:
            variants.append((f"v61_h{hs:.2f}", GLM53v61,
                             dict(ROM_SCALE=1.08, RATE_DOWN=0.25, STALL_HOLD_S=hs)))
    if args.tail:
        for k in [float(x) for x in args.tail.split(",")]:
            variants.append((f"v61_k{k:.2f}", GLM53v61,
                             dict(ROM_SCALE=1.08, RATE_DOWN=0.25, STALL_TAIL_KEEP=k)))
    if args.sweep == "rom":
        for s in (1.00, 1.06, 1.12):
            variants.append((f"v61_s{s:.2f}", GLM53v61, dict(ROM_SCALE=s, RATE_DOWN=0.25)))
    elif args.sweep == "down":
        for r in (0.0, 0.12, 0.25, 0.50):
            variants.append((f"v61_d{r:.2f}", GLM53v61, dict(ROM_SCALE=1.08, RATE_DOWN=r)))
    elif args.sweep == "stalltail":
        for k in (0.0, 0.05, 0.10, 0.20):
            variants.append((f"v61_k{k:.2f}", GLM53v61,
                             dict(ROM_SCALE=1.08, RATE_DOWN=0.25, STALL_TAIL_KEEP=k)))
    elif args.sweep == "hold":
        for hs in (0.45, 0.35, 0.25):
            variants.append((f"v61_h{hs:.2f}", GLM53v61,
                             dict(ROM_SCALE=1.08, RATE_DOWN=0.25, STALL_HOLD_S=hs)))
    elif not args.rom and not args.down and not args.onsetonly and args.arms != "v6":
        variants.append(("v61", GLM53v61, {}))

    tag_out = args.tag or args.sweep or "default"
    alld, alls = [], []
    for name, cls, kw in variants:
        cached, caches = (None, None) if args.nocache else load_cached(name)
        if cached is not None:
            print(f"  [cache] {name}", flush=True)
            df, st = cached, caches
        else:
            df, st = eval_variant(name, cls, kw, data)
        alld.append(df)
        alls.append(st)
    df = pd.concat(alld, ignore_index=True)
    st = pd.concat(alls, ignore_index=True)
    a, b, s = summarize(df, st)
    order = [v[0] for v in variants]
    a, b = a.reindex(order), b.reindex(order)
    print("\n" + "=" * 120)
    print("表 A 恒载 9 组（|·| 均值；sp_* = 尖峰口径，仅 onset 事件）")
    print("=" * 120)
    print(a.rename(columns={"drift_main": "全段时漂%", "drift_slow": "慢相段%",
                            "drift_loaded": "受载中位%", "flat": "平坦度%",
                            "step_ratio": "阶跃保真", "noise_ratio": "噪声比", "epoch": "epoch数",
                            "sp_over_onset_med": "超调中位%", "sp_over_onset_max": "超调max%",
                            "sp_tr_onset_max": "回落max%"}).to_string())
    print("\n" + "=" * 120)
    print("表 B 实采 4 份")
    print("=" * 120)
    print(b.rename(columns={"max_gap": "全程偏差中位", "gap_med": "变载窗偏差中位",
                            "cap_med": "捕获比中位", "cap_min": "捕获比最小", "epoch": "epoch数",
                            "sp_over_max": "超调max%", "sp_tr_max": "回落max%"}).to_string())
    print("\n" + "=" * 120)
    print("表 C 首次加载响应（中位，秒；err5 = 加载沿 +5 s 相对误差 %）")
    print("=" * 120)
    print(s.round(2).to_string())
    print("\n" + "=" * 120)
    print("表 D 逐份实采：全程最大偏差 / 阶跃超调 / 回落")
    print("=" * 120)
    v = df[df.kind == "实采"]
    print(v.pivot_table(index="dataset", columns="algo", values="sp_over_max").reindex(columns=order).round(1).to_string())
    print("\n每份实采的全程最大偏差（ADC）：")
    print(v.pivot_table(index="dataset", columns="algo", values="max_gap").reindex(columns=order).round(0).to_string())
    print("\n每份实采的台阶捕获比（中位）：")
    print(v.pivot_table(index="dataset", columns="algo", values="cap_med").reindex(columns=order).round(2).to_string())

    df.to_csv(os.path.join(RES, f"v61_ab_{tag_out}.csv"), index=False, encoding="utf-8-sig")
    st.to_csv(os.path.join(RES, f"v61_ab_{tag_out}_settle.csv"), index=False, encoding="utf-8-sig")
    print(f"\n产物：results/v61_ab_{tag_out}.csv  v61_ab_{tag_out}_settle.csv")


if __name__ == "__main__":
    main()
