#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三路对比：免责 1 s / 免责 3 s / v6c(改进版)，全部 13 份录制。

指标函数与 `temp/v4.1flash/scripts/ci_v6_vs_1s_3s.py` 完全一致，保证可比性。

产出：
  temp/Claude/results/v6c_comparison.csv        每份数据 × 每档的全部指标
  temp/Claude/results/v6c_settle.csv            首次加载的稳定时刻（T_band / T_stable）
  temp/Claude/figures/C1_v6c_overview_all.png   小倍数总览（13 格：raw / 1s / 3s / v6c）
  temp/Claude/figures/C2_v6c_metrics_all.png    指标对比 4 面板
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEMP = os.path.dirname(HERE)                          # temp/
V4SCRIPTS = os.path.join(TEMP, "v4.1flash", "scripts")
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
for _p in (V4SCRIPTS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import ad_lib as L                                     # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

try:
    from glm53_v6c import GLM53v6c                     # noqa: E402
except ImportError as _e:                              # 允许 v6c 未落地时导入本模块做链路自检
    GLM53v6c = None
    _V6C_ERR = str(_e)
else:
    _V6C_ERR = None

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

KS = ["1s", "3s", "v6c"]
FASTOF = {"1s": 1.0, "3s": 3.0}
LBL = {"1s": "免责 1s", "3s": "免责 3s", "v6c": "v6c"}
COL = {"raw": "0.6", "1s": "#1f77b4", "3s": "#ff7f0e", "v6c": "#2ca02c"}

B = os.path.join(TEMP, "变化负载")
HOLD = [(f"{loc}/数据{i}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"))
        for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]
VARY = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]
HOLDTAGS = [t for t, _ in HOLD]


# ═══════════════════════════ 算法运行 ═══════════════════════════
def run_v5(tu, Xu, fast):
    """运行 v5.1 并指定免责期长度（1 s / 3 s）。"""
    class _T(GLM53v51):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    c = _T(Xu.shape[1])
    c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = fast, fast / 3.0, fast
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


def run_v6c(tu, Xu):
    """运行 v6c（改进版）。"""
    if GLM53v6c is None:
        raise ImportError(f"找不到 glm53_v6c.py（应位于 {HERE}）：{_V6C_ERR}")
    c = GLM53v6c(Xu.shape[1])
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c


def run(k, tu, Xu):
    if k == "v6c":
        return run_v6c(tu, Xu)
    return run_v5(tu, Xu, FASTOF[k])


def n_epoch(c):
    """epoch 计数：优先用 _begin/_restep 记录；v6c 自带 epoch_t 时直接用。"""
    et = getattr(c, "epoch_t", None)
    if et is not None and len(et):
        return len(et)
    return int(getattr(c, "n_epoch", 0))


def a_max_of(c):
    """A 的幅度读数：v6c 记 A_peak，v5 用 A.max()。"""
    ap = getattr(c, "A_peak", None)
    if ap is not None and np.isfinite(ap) and ap > 0:
        return float(ap)
    try:
        return float(np.asarray(c.A).max())
    except Exception:
        return np.nan


# ═══════════════════════════ 指标函数 ═══════════════════════════
def find_segment(total, frac=0.15):
    thr = frac * total.max()
    ld = total > thr
    dd = np.diff(ld.astype(int))
    s = list(np.where(dd == 1)[0] + 1)
    e = list(np.where(dd == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)


def hold_metrics(Y, c, tu, Xu, dtm, tot_s, peak, d=None):
    """恒载指标：时漂残余 / 噪声比 / 平坦度 / 阶跃保真 / 首扣时延。"""
    s0r, s1r = find_segment(Xu.sum(axis=1))[0]
    s0 = int(s0r)
    s1 = min(int(s1r), len(tu) - 1)
    amp_v = Xu[s0:s1].mean(axis=0) - Xu[:max(1, s0)].mean(axis=0)
    m = int(np.argmax(amp_v))
    amp = amp_v[m]
    loaded = amp_v > 0.10 * amp_v.max()
    base = float(np.median(tot_s[max(0, s0 - int(2 / dtm)):s0]))
    n_on = next((i for i in range(s0, min(s0 + int(5 / dtm), s1))
                 if tot_s[i] > base + 0.05 * (tot_s[s0:s1].max() - base)), s0)
    nL = max(1, s1 - s0)
    seg = Y[s0:s1]
    by, bx = Y[:s0, m].mean(), Xu[:s0, m].mean()
    dr = seg[-nL // 10:].mean(axis=0) - seg[:nL // 10].mean(axis=0)
    a5 = max(s0, n_on + int(5.0 / dtm))
    n5 = max(1, s1 - a5)
    dr5 = Y[a5:s1][-n5 // 10:].mean(axis=0) - Y[a5:s1][:n5 // 10].mean(axis=0)
    i1, i2 = s0 + int(0.5 / dtm), s0 + int(2.5 / dtm)
    sx = Xu[i1:i2, m].mean() - bx
    step = (Y[i1:i2, m].mean() - by) / sx if abs(sx) > 1e-9 else np.nan
    tq = tu[s0:s1] - tu[s0]

    def dstd(sig, t):
        k, b0 = np.polyfit(t, sig, 1)
        return (sig - (k * t + b0)).std()

    ny = dstd(seg[10:, m], tq[10:])
    nx = dstd(Xu[s0 + 10:s1, m], tq[10:])
    ded = (Xu - Y).sum(axis=1)
    hit = np.where(ded[s0:s1] > 0.005 * abs(tot_s[min(s1, len(tu) - 1)] - base))[0]
    return dict(drift_main=100 * dr[m] / amp, drift_slow=100 * dr5[m] / amp,
                drift_loaded=100 * np.median(dr[loaded] / amp),
                noise_ratio=ny / nx if nx > 1e-12 else np.nan, flat=100 * ny / amp,
                step_ratio=step,
                ded_delay=float(hit[0] * dtm) if len(hit) else np.nan,
                a_max=a_max_of(c), g_end=float(c.g), epoch=n_epoch(c),
                max_gap=np.nan, pct=np.nan, cap_med=np.nan, cap_min=np.nan,
                n_event=0, gap_med=np.nan, gap_max=np.nan)


def vary_metrics(Y, c, tu, Xu, dtm, tot_s, peak, d):
    """实采指标：全程最大偏差 / 台阶捕获比 / 变载窗偏差。"""
    d["Ys"] = {"a": Y}
    d["periods"] = L.find_periods(Xu.sum(axis=1), dtm)
    d["events"] = [e for e, _ in L.detect_events(Xu.sum(axis=1), dtm)]
    y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
    gap = np.abs(y - tot_s)
    ev = L.event_table(d, d["events"], {"a": Y}, [], gain_s=6.0, algos=["a"])
    ev["big"] = ev["jump"].abs() >= 2000.0
    thr = 0.30 * ev["pre"].max() if len(ev) else 0
    ev["mid"] = (ev["pre"] > thr) & (ev["post"] > thr) if len(ev) else False
    ml = ev[ev.mid & ev.big] if len(ev) else ev
    cap = (ml["gain_a"] / ml["raw_gain"]).replace([np.inf, -np.inf], np.nan).dropna() \
        if len(ml) else pd.Series(dtype=float)
    return dict(drift_main=np.nan, drift_slow=np.nan, drift_loaded=np.nan, noise_ratio=np.nan,
                flat=np.nan, step_ratio=np.nan, ded_delay=np.nan, a_max=a_max_of(c),
                g_end=float(c.g), epoch=n_epoch(c), max_gap=float(gap.max()),
                pct=100 * float(gap.max()) / peak,
                cap_med=float(cap.median()) if len(cap) else np.nan,
                cap_min=float(cap.min()) if len(cap) else np.nan, n_event=int(len(ml)),
                gap_med=float(ml["gap_a"].median()) if len(ml) else np.nan,
                gap_max=float(ml["gap_a"].max()) if len(ml) else np.nan)


def first_onset(tu, tot, dtm):
    """首个真实加载沿（回溯到前置电平 + 5% 跳变处）。"""
    peak = float(np.percentile(tot, 99.5))
    idx = np.where(tot > 0.5 * peak)[0]
    if not len(idx):
        return None
    i = int(idx[0])
    pre = float(np.median(tot[max(0, i - int(1.5 / dtm)):max(1, i - int(0.3 / dtm))]))
    j = i
    while j > 0 and tot[j] > pre + 0.05 * (tot[i] - pre):
        j -= 1
    return j + 1, pre


def settle_time(tu, Ytot, i0, target, step, dtm, hold_s=30.0):
    """① T_band：显示首次进入 ±5%×阶跃带（相对真值）、且此后 hold_s 内不再离开。
       注意它把"准"和"快"混在一起：v6c 是钉在 Â 上的，Â 偏 >5% 就永远进不了带。"""
    if step <= 0:
        return np.nan
    lo, hi = target - 0.05 * step, target + 0.05 * step
    ok = (Ytot >= lo) & (Ytot <= hi)
    n = len(Ytot)
    H = int(hold_s / dtm)
    for k in range(i0, n):
        e = min(n, k + H)
        if e - k < min(H, n - i0):            # 剩余段不足：只要求到段末
            if ok[k:e].all():
                return float(tu[k] - tu[i0])
            break
        if ok[k:e].all():
            return float(tu[k] - tu[i0])
    return np.nan


def stable_time(tu, Ytot, i0, step, dtm, hold_s=30.0, tol_frac=0.05):
    """② T_stable：显示首次"停下"——此后 hold_s 内相对该时刻自身的漂移 ≤ tol_frac×阶跃。
       与绝对精度无关，量的就是用户要的「平稳滑行」。"""
    if step <= 0:
        return np.nan
    n = len(Ytot)
    H = int(hold_s / dtm)
    tol = tol_frac * step
    for k in range(i0, n):
        e = min(n, k + H)
        if e - k < min(H, n - i0):
            break
        if np.max(np.abs(Ytot[k:e] - Ytot[k])) <= tol:
            return float(tu[k] - tu[i0])
    return np.nan


# ═══════════════════════════ 图结构自检 ═══════════════════════════
def figcheck(fig, path):
    """非视觉结构自检：面板数 / 空白面板 / artist 越界 / 文本重叠。"""
    def _inter(b1, b2):
        x0, x1 = max(b1.x0, b2.x0), min(b1.x1, b2.x1)
        y0, y1 = max(b1.y0, b2.y0), min(b1.y1, b2.y1)
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    print(f"\n[figcheck] {os.path.basename(path)}  canvas={fb.width:.0f}x{fb.height:.0f}px  "
          f"axes={len(fig.axes)}")
    empty, oob_tot, ov_tot = [], 0, 0
    for i, a in enumerate(fig.axes):
        has_data = bool(a.lines or a.patches or a.collections or a.images)
        if not has_data:
            empty.append(i)
        oob = []
        arts = list(a.lines) + list(a.patches) + list(a.texts)
        if a.title.get_text():
            arts.append(a.title)
        for art in arts:
            if not art.get_visible():
                continue
            try:
                bb = art.get_window_extent(r)
            except Exception:
                continue
            area = bb.width * bb.height
            if area <= 0:
                continue
            if _inter(bb, fb) / area < 0.85:
                oob.append(type(art).__name__)
        items = [(t.get_text().strip()[:26], t.get_window_extent(r))
                 for t in a.texts if t.get_text().strip()]
        if a.title.get_text().strip():
            items.append(("<标题>", a.title.get_window_extent(r)))
        ov = []
        for j in range(len(items)):
            for k in range(j + 1, len(items)):
                b1, b2 = items[j][1], items[k][1]
                if _inter(b1, b2) > 0.12 * min(b1.width * b1.height, b2.width * b2.height):
                    ov.append((items[j][0], items[k][0]))
        oob_tot += len(oob)
        ov_tot += len(ov)
        print(f"  ax{i:2d}  lines={len(a.lines):2d} patches={len(a.patches):3d} "
              f"texts={len(a.texts):2d} 有内容={has_data} 越界={len(oob)} 文本重叠={len(ov)}")
    print(f"  -> 空白面板 {empty if empty else '无'}；artist 越界合计 {oob_tot}；文本重叠合计 {ov_tot}")
    return dict(axes=len(fig.axes), empty=empty, oob=oob_tot, overlap=ov_tot)


class _RawC:
    """raw 下界的伪控制器：只为复用 hold_metrics 里 c.A.max() 这一项。"""
    def __init__(self, n):
        self.A = np.zeros(n)
        self.g = 0.0
        self.epoch_t = []


def raw_hold_drifts():
    """把原始曲线当"显示"喂给同一套指标函数，得到无补偿下界（图 2 (1) 参考线）。"""
    out = []
    for _tag, path in HOLD:
        if not os.path.exists(path):
            continue
        d = L.prep(path)
        tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
        tot_s = L.med_smooth(Xu.sum(axis=1), 0.5 / dtm)
        out.append(hold_metrics(Xu, _RawC(Xu.shape[1]), tu, Xu, dtm, tot_s,
                                float(Xu.sum(axis=1).max()), d))
    full = float(np.mean([abs(m["drift_main"]) for m in out])) if out else np.nan
    slow = float(np.mean([abs(m["drift_slow"]) for m in out])) if out else np.nan
    print(f"\n原始（无补偿）下界（恒载 {len(out)} 组）："
          f"时漂残余 全段 {full:.2f}% / 慢相段 {slow:.2f}%")
    return full, slow


# ═══════════════════════════ 主流程 ═══════════════════════════
def main():
    """跑完全部 13 份录制：指标 → CSV → 摘要表 → 两张图。"""
    if GLM53v6c is None:
        print(f"[致命] 找不到 v6c 模块 glm53_v6c.py（应位于 {HERE}）：{_V6C_ERR}")
        return 2

    rows, curves, settle = [], {}, []
    skipped = []
    print("=" * 122)
    print("三路对比 · 免责 1 s / 免责 3 s / v6c（改进版），全部 13 份录制")
    print("=" * 122)
    for tag, path in HOLD + VARY:
        if not os.path.exists(path):
            print(f"[skip] 缺文件 {tag} -> {path}")
            skipped.append(tag)
            continue
        d = L.prep(path)
        tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
        tot_s = L.med_smooth(Xu.sum(axis=1), 0.5 / dtm)
        peak = float(Xu.sum(axis=1).max())
        kind = "恒载" if tag in HOLDTAGS else "实采"
        curves[tag] = dict(tu=tu, tot=tot_s, Y={}, kind=kind)
        fo = first_onset(tu, Xu.sum(axis=1), dtm)
        i0 = fo[0] if fo else None
        tgt = (float(np.median(Xu.sum(axis=1)[i0 + int(4.6 / dtm):i0 + int(5.4 / dtm)]))
               if fo else np.nan)
        step = (tgt - fo[1]) if fo else np.nan
        line = f"{tag:>18} [{kind}] {d['span']:5.1f}s ch={Xu.shape[1]:2d}"
        for k in KS:
            Y, c = run(k, tu, Xu)
            mm = (hold_metrics if kind == "恒载" else vary_metrics)(
                Y, c, tu, Xu, dtm, tot_s, peak, d)
            mm.update(dataset=tag, kind=kind, algo=k, span=d["span"], nch=Xu.shape[1])
            rows.append(mm)
            curves[tag]["Y"][k] = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
            if fo:
                tb = settle_time(tu, Y.sum(axis=1), i0, tgt, step, dtm)
                ts_ = stable_time(tu, Y.sum(axis=1), i0, step, dtm)
                err5 = float(Y.sum(axis=1)[i0 + int(5.0 / dtm)] / tgt * 100 - 100)
                settle.append(dict(dataset=tag, kind=kind, algo=k, t_edge=float(tu[i0]),
                                   target=tgt, pre=fo[1], step=step, t_settle=tb,
                                   t_stable=ts_, err5s=err5))
        if kind == "恒载":
            line += "  | 全段/慢相段: " + "  ".join(
                f"{k} {rows[-3 + KS.index(k)]['drift_main']:+.2f}"
                f"/{rows[-3 + KS.index(k)]['drift_slow']:+.2f}%" for k in KS)
        else:
            line += "  | 全程偏差: " + "  ".join(
                f"{k} {rows[-3 + KS.index(k)]['max_gap']:.0f}" for k in KS)
            line += "  | 捕获中位: " + "  ".join(
                f"{k} {rows[-3 + KS.index(k)]['cap_med']:.2f}" for k in KS)
        print(line)

    if not rows:
        print("\n[致命] 没有任何可用录制，退出。")
        return 1
    if skipped:
        print(f"\n[注意] 跳过 {len(skipped)} 份缺失录制：{', '.join(skipped)}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "v6c_comparison.csv"), index=False, encoding="utf-8-sig")
    st = pd.DataFrame(settle)
    st.to_csv(os.path.join(RES, "v6c_settle.csv"), index=False, encoding="utf-8-sig")

    hold = df[df.kind == "恒载"]
    vary = df[df.kind == "实采"]

    print("\n----- 恒载 9 组（|·| 均值 / 均值）-----")
    print(hold.groupby("algo").agg(时漂残余_全段=("drift_main", lambda s: s.abs().mean()),
                                  时漂残余_慢相段=("drift_slow", lambda s: s.abs().mean()),
                                  受载中位=("drift_loaded", lambda s: s.abs().mean()),
                                  噪声比=("noise_ratio", "mean"), 平坦度=("flat", "mean"),
                                  阶跃保真=("step_ratio", "mean"), 首扣时延=("ded_delay", "mean"),
                                  A占幅度=("a_max", "mean"), g末端=("g_end", "mean"),
                                  epoch数=("epoch", "sum")).reindex(KS).round(2).to_string())
    print("\n----- 实采 4 份 -----")
    print(vary.groupby("algo").agg(全程偏差中位=("max_gap", "median"),
                                  占峰值中位=("pct", "median"),
                                  变载窗偏差中位=("gap_med", "median"),
                                  捕获比中位=("cap_med", "mean"),
                                  捕获比最小=("cap_min", "min"), 有效事件=("n_event", "sum"),
                                  epoch数=("epoch", "sum")).reindex(KS).round(2).to_string())
    print("\n每份实录的全程最大偏差（ADC）")
    print(vary.pivot_table(index="dataset", columns="algo", values="max_gap")
          .reindex(columns=KS).round(0).to_string())
    print("\n每份实录的台阶捕获比（中位）")
    print(vary.pivot_table(index="dataset", columns="algo", values="cap_med")
          .reindex(columns=KS).round(2).to_string())

    print("\n----- 首次加载稳定时刻 T_band（进入 ±5%×阶跃带并保持 30 s，秒）-----")
    print(st.pivot_table(index="dataset", columns="algo", values="t_settle")
          .reindex(columns=KS).round(2).to_string())
    print("\n各组中位：")
    print(st.groupby(["kind", "algo"])["t_settle"].median().unstack()
          .reindex(columns=KS).round(2).to_string())
    print("\n----- T_stable（显示首次停下：此后 30 s 相对该时刻自身漂移 ≤5%×阶跃，秒）-----")
    print(st.pivot_table(index="dataset", columns="algo", values="t_stable")
          .reindex(columns=KS).round(2).to_string())
    print("\n各组中位：")
    print(st.groupby(["kind", "algo"])["t_stable"].median().unstack()
          .reindex(columns=KS).round(2).to_string())
    print("\n加载沿 +5 s 的相对误差（显示/真值−1，%）")
    print(st.pivot_table(index="dataset", columns="algo", values="err5s")
          .reindex(columns=KS).round(1).to_string())
    print("\n各组 |误差| 中位：")
    tmp = st.assign(abs5=st.err5s.abs())
    print(tmp.groupby(["kind", "algo"])["abs5"].median().unstack()
          .reindex(columns=KS).round(2).to_string())

    RAW_FULL, RAW_SLOW = raw_hold_drifts()

    # ═══════════════════ 图 1：小倍数总览 ═══════════════════
    order = HOLDTAGS + [t for t, _ in VARY if t in curves]
    n = len(order)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(21, 2.7 * nrow))
    fig.suptitle("全量 13 份数据 · 免责 1 s / 免责 3 s / v6c 效果总览"
                 "（灰=原始，蓝=免责1s，橙=免责3s，绿=v6c）", fontsize=15)
    for ax, tag in zip(axes.ravel(), order):
        cu = curves[tag]
        ax.plot(cu["tu"], cu["tot"], color=COL["raw"], lw=0.9, label="原始")
        for k in KS:
            ax.plot(cu["tu"], cu["Y"][k], color=COL[k], lw=1.1, label=LBL[k])
        g = df[df.dataset == tag].set_index("algo")
        if cu["kind"] == "恒载":
            ttl = (f"{tag}｜慢相段时漂 1s {g.loc['1s', 'drift_slow']:+.2f}%"
                   f" / 3s {g.loc['3s', 'drift_slow']:+.2f}%"
                   f" / v6c {g.loc['v6c', 'drift_slow']:+.2f}%")
        else:
            ttl = (f"{tag}｜全程偏差 1s {g.loc['1s', 'max_gap']:.0f}"
                   f" / 3s {g.loc['3s', 'max_gap']:.0f}"
                   f" / v6c {g.loc['v6c', 'max_gap']:.0f} ADC"
                   f"（占峰值 {g.loc['1s', 'pct']:.1f}/{g.loc['3s', 'pct']:.1f}"
                   f"/{g.loc['v6c', 'pct']:.1f}%）")
        ss = st[st.dataset == tag].set_index("algo")["t_stable"]
        if len(ss) == len(KS):
            ttl += (f"\n首次平稳 T_stable：1s {ss['1s']:.1f}s / 3s {ss['3s']:.1f}s"
                    f" / v6c {ss['v6c']:.2f}s")
        ax.set_title(ttl, fontsize=8.5)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8, loc="lower left")
    fig.subplots_adjust(left=0.045, right=0.99, top=0.93, bottom=0.035,
                        hspace=0.46, wspace=0.18)
    f1 = os.path.join(FIG, "C1_v6c_overview_all.png")
    fig.savefig(f1, dpi=105)
    figcheck(fig, f1)
    plt.close(fig)
    print(f"\n图已保存：{f1}")

    # ═══════════════════ 图 2：指标对比 ═══════════════════
    fig2, axs = plt.subplots(2, 2, figsize=(19.5, 11.0))
    fig2.suptitle("全量数据 · 免责 1 s / 免责 3 s / v6c（改进版）指标对比", fontsize=15)

    ax = axs[0, 0]
    g = hold.groupby("algo").agg(full=("drift_main", lambda s: s.abs().mean()),
                                 slow=("drift_slow", lambda s: s.abs().mean())).reindex(KS)
    xs = np.arange(2)
    for j, k in enumerate(KS):
        b = ax.bar(xs + (j - 1) * 0.26, [g.loc[k, "full"], g.loc[k, "slow"]], 0.25,
                   color=COL[k], label=LBL[k])
        ax.bar_label(b, fmt="%.2f", fontsize=9)
    for i, rv in enumerate((RAW_FULL, RAW_SLOW)):
        if np.isfinite(rv):
            ax.plot([i - 0.42, i + 0.42], [rv, rv], color="0.45", ls=":", lw=1.3)
            ax.text(i + 0.44, rv, f"原始 {rv:.1f}%", fontsize=9, color="0.35", va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(["时漂残余 全段", "时漂残余 慢相段"])
    ax.set_ylabel("恒载 9 组 均值 (%)")
    _ymax = max([18.0] + ([RAW_FULL, RAW_SLOW] if np.isfinite(RAW_FULL) else [])
                + list(g[["full", "slow"]].to_numpy().ravel()))
    ax.set_ylim(0, 1.18 * _ymax)
    ax.set_title("(1) 恒载 9 组：时漂残余（越小越好）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axs[0, 1]
    pv = vary.pivot_table(index="dataset", columns="algo", values="max_gap").reindex(columns=KS)
    xs = np.arange(len(pv))
    for j, k in enumerate(KS):
        b = ax.bar(xs + (j - 1) * 0.26, pv[k].values, 0.25, color=COL[k], label=LBL[k])
        ax.bar_label(b, fmt="%.0f", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("-", "-\n", 1) for s in pv.index], fontsize=9)
    ax.set_ylabel("全程 max|显示−原始| (ADC)")
    ax.set_title("(2) 实采 4 份：全程最大偏差（越小越好）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axs[1, 0]
    pv2 = vary.pivot_table(index="dataset", columns="algo", values="cap_med").reindex(columns=KS)
    pv3 = vary.pivot_table(index="dataset", columns="algo", values="gap_med").reindex(columns=KS)
    xs = np.arange(len(pv2))
    for j, k in enumerate(KS):
        ax.bar(xs + (j - 1) * 0.26, pv2[k].values, 0.25, color=COL[k], label=LBL[k])
        ax.bar_label(ax.containers[-1], fmt="%.2f", fontsize=8)
    ax.axhline(1.0, color="0.4", ls=":", lw=1.2)
    ax.axhline(0.9, color="0.6", ls="--", lw=1.0)
    ax.text(len(pv2) - 0.5, 1.01, "1.0 = 台阶完整透传", fontsize=8.5, color="0.35", ha="right")
    ax2 = ax.twinx()
    for j, k in enumerate(KS):
        ax2.plot(xs + (j - 1) * 0.26, pv3[k].values, marker="v", ms=6, color=COL[k],
                 ls="none", alpha=0.85)
    ax2.set_ylabel("变载窗最大偏差 中位 (ADC)（▽）")
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("-", "-\n", 1) for s in pv2.index], fontsize=9)
    ax.set_ylabel("台阶捕获比（柱）")
    ax.set_ylim(0, 1.35)
    ax.set_title("(3) 实采：变载台阶捕获比（柱）与变载窗偏差（▽）")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    ax = axs[1, 1]
    pv4 = st.pivot_table(index="dataset", columns="algo", values="t_stable").reindex(columns=KS)
    order2 = HOLDTAGS + [t for t, _ in VARY if t in pv4.index]
    pv4 = pv4.reindex(order2)
    xs = np.arange(len(pv4))
    w = 0.26
    for j, k in enumerate(KS):
        vals = pv4[k].values
        ax.bar(xs + (j - 1) * w, np.nan_to_num(vals), w, color=COL[k], label=LBL[k])
        for x, v in zip(xs + (j - 1) * w, vals):
            if np.isfinite(v):
                ax.text(x, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.5)
            else:
                ax.text(x, 0.15, "n/a", ha="center", fontsize=7)
    ax.axhline(1.0, color="#d62728", ls="--", lw=1.2)
    ax.text(len(pv4) - 0.5, 1.15, "1 s 目标线", fontsize=9, color="#d62728", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("/", "\n") for s in pv4.index], fontsize=7.5)
    ax.set_ylabel("T_stable：显示首次停下并保持 30 s (s)")
    ax.set_title("(4) 首次加载平稳时刻 T_stable（越小越好）")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig2.subplots_adjust(left=0.055, right=0.94, top=0.92, bottom=0.10,
                         wspace=0.28, hspace=0.34)
    f2 = os.path.join(FIG, "C2_v6c_metrics_all.png")
    fig2.savefig(f2, dpi=115)
    figcheck(fig2, f2)
    plt.close(fig2)
    print(f"图已保存：{f2}")

    print("\n注：'首扣时延' 对 v6c 语义可能不同 —— 若 v6c 前若干秒显示在原始之上（前置），"
          "该列指的是'首次变成向下扣除'的时刻，不是'首次修正'。")
    print("注：'T_band' = 显示首次进入 ±5%×阶跃带、且此后 30 s 不再离开的时刻；"
          "v5 两档在带内先停留（原始透传）再被蠕变推出，故 T_band 偏大。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
