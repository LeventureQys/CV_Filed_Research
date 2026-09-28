# -*- coding: utf-8 -*-
"""免责期（「无责」窗）1s / 3s / 5s 同图对比 + 「1s 能不能救」的两因子拆解。

动机（用户提问）：把「无责」从 3s 压到 1s 会怎样？请与 3s / 5s 放进同一张图对比。

参数口径（与 C++ 完全一致的三处联动）：`DriftCompensator::SetFastPhase(fast)` 派生
    FAST_S      = fast            免责期长度
    EXEMPT_AWIN = fast / 3        A 采集窗长度（紧贴免责期末端 → 窗 = epoch+[fast−fast/3, fast]）
    LEV_ARM_S   = fast            短滞后电平判据的武装延时
三档都按该规则设参，**其余参数一律不动**（STEP_PERSIST=2.5s、STEP_SUPPRESS=6s、TAU_G=3s、
门限与限幅全部保持）。算法本体 = `glm53_v51.py`（v5.1，与 src/domain/drift 的 C++ 逐帧对拍过）。

时间口径（第一性，务必分清）：
    epoch 起点 = 真实台阶 + STEP_PERSIST(2.5s)
    ⇒ 「真实加载沿 → 首帧开始扣」理论时延 = 2.5 + fast = 3.5 / 5.5 / 7.5 s
    实测口径另给「首个可见扣除(|扣除| > 0.5%×本段幅度)」的时刻（g 从 0 起以 τ=3s 爬升）。

两因子拆解（Ablation）：把 FAST_S 压到 1s 会同时改动两件事，必须分开看——
    ① A 采集窗时刻被一起提前（`EXEMPT_AWIN = FAST_S/3`，且窗必须在免责期内、
       免责期一结束当帧就采 A）⇒ **在现行结构里无法与免责期解耦**：
       「免责 1s 但 A 仍取 epoch+[2,3]s」等价于「免责 3s」（3s 内都不扣），
       所以该因子无法单独剥离，只能作为"1s 的不可分割代价"陈述；
    ② 短滞后电平判据的武装延时 `LEV_ARM_S` 被一起压到 1s ⇒ epoch 起点后 1s
       （= 真实台阶后 3.5s）快相尾巴还在爬，会被判成变载 → pending/hold 冻结补偿、
       并在 u>6s 时被当成 restep 重锚（假 epoch）。该因子可以单独剥离。
    故拆解矩阵 = 免责期 {1s, 3s} × 武装延时 {1s, 3s}，外加 5s 现行档：
        免责1s|ARM1   = 用户设想的 1s 档（现状）
        免责1s|ARM3   = 只把武装延时改回 3s
        免责3s|ARM1   = 只把武装延时压到 1s
        免责3s|ARM3   = 现行 3s 档
        免责5s|ARM5   = 现行 5s 档

产出：
    results/exempt_sweep_static9.csv / _static9_summary.csv   恒载 9 组
    results/exempt_sweep_recs.csv                             4 份实录汇总 + epoch 数
    results/exempt_sweep_events.csv                           逐个负载内变载事件
    results/exempt_sweep_deddelay.csv                         每个负载段的首扣时延
    results/exempt_sweep_awin.csv                             A 窗位置的原始电平（机制）
    results/exempt_sweep_mech.csv                             pending/hold 占空比（机制）
    results/exempt_sweep_ablation.csv                         两因子拆解
    results/exempt_sweep.npz                                  作图用时序
    figures/H1_exempt_1s_3s_5s.png                            主对比图（3×3）
    figures/H2_exempt_1s_ablation.png                         1s 拆解图（1×3）
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
FIG = os.path.join(OUT, "figures")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v3 import GLM53v3                           # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

B = os.path.join(TEMP, "变化负载")
RECS = [("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                          "最终测试目标", "device_001_seg000.csv"))]
STATIC = [(loc, f"数据{i}") for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)]

MAIN = [("raw", None, {}),
        ("1s", GLM53v51, dict(FAST_S=1.0, EXEMPT_AWIN=1.0 / 3.0, LEV_ARM_S=1.0)),
        ("3s", GLM53v51, dict(FAST_S=3.0, EXEMPT_AWIN=1.0, LEV_ARM_S=3.0)),
        ("5s", GLM53v51, dict(FAST_S=5.0, EXEMPT_AWIN=5.0 / 3.0, LEV_ARM_S=5.0))]
REF = [("v3", GLM53v3, {})]
# 两因子拆解：免责期长度 × 电平判据武装延时（A 窗时刻无法与免责期解耦，见文件头说明）
ABL = [("免责1s|ARM1", GLM53v51, dict(FAST_S=1.0, EXEMPT_AWIN=1.0 / 3.0, LEV_ARM_S=1.0)),
       ("免责1s|ARM3", GLM53v51, dict(FAST_S=1.0, EXEMPT_AWIN=1.0 / 3.0, LEV_ARM_S=3.0)),
       ("免责3s|ARM1", GLM53v51, dict(FAST_S=3.0, EXEMPT_AWIN=1.0, LEV_ARM_S=1.0)),
       ("免责3s|ARM3", GLM53v51, dict(FAST_S=3.0, EXEMPT_AWIN=1.0, LEV_ARM_S=3.0)),
       ("免责5s|ARM5", GLM53v51, dict(FAST_S=5.0, EXEMPT_AWIN=5.0 / 3.0, LEV_ARM_S=5.0))]
VK = [k for k, _, _ in MAIN]
KWV = {k: kw for k, _, kw in MAIN}
ABLK = [k for k, _, _ in ABL]
STYLE = {"raw": dict(color="0.55", lw=1.0, ls="-"),
         "1s": dict(color="#1f77b4", lw=1.4, ls="-"),
         "3s": dict(color="#ff7f0e", lw=1.4, ls="-"),
         "5s": dict(color="#d62728", lw=1.4, ls="-"),
         "v3": dict(color="0.3", lw=1.0, ls="--")}
ABL_STYLE = {"免责1s|ARM1": "#1f77b4", "免责1s|ARM3": "#2ca02c", "免责3s|ARM1": "#9467bd",
             "免责3s|ARM3": "#ff7f0e", "免责5s|ARM5": "#d62728"}


def make_traced(cls):
    """给任意算法类套一层：记录 epoch 轨迹与逐帧状态（供机制分析）。"""

    class _Traced(cls):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []
            self.st = []

        def process(self, ts, v):
            y = super().process(ts, v)
            self.st.append((float(ts), int(self.in_load), int(self.pending), int(self.hold),
                            float(self.A.max()) if self.A.size else 0.0, float(self.g)))
            return y

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    return _Traced


def run_variant(cls, kw, tu, Xu):
    if cls is None:
        return Xu.copy(), None, None
    c = make_traced(cls)(Xu.shape[1])
    for k, v in kw.items():
        setattr(c, k, v)
    Y = np.empty_like(Xu)
    for i in range(len(tu)):
        Y[i] = c.process(tu[i], Xu[i])
    return Y, c, np.array(c.st)


def ded_total(Y, X):
    """逐帧已生效总扣除量（输出定义式：显示 = 读数 − 扣除，含输出封顶）。"""
    return (X - Y).sum(axis=1)


def first_ded_delay(Y, X, dtm, s0, amp, frac=0.005):
    ded = ded_total(Y, X)[s0:]
    idx = np.where(ded > frac * amp)[0]
    return float(idx[0] * dtm) if len(idx) else float("nan")


# ═══════════════════════════ A. 恒载 9 组 ═══════════════════════════
def find_segment(total, frac=0.15):
    thr = frac * total.max()
    ld = total > thr
    d = np.diff(ld.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if ld[0]:
        s = np.r_[0, s]
    if ld[-1]:
        e = np.r_[e, len(ld)]
    return sorted(zip(s, e), key=lambda z: z[1] - z[0], reverse=True)


def static_metrics(Y, X, tt, s0, s1, amp, loaded, m, dtm, n_on):
    """口径同 Document/03 与 bd_v5_static9.py（不另立标准）。"""
    nL = s1 - s0
    seg = Y[s0:s1]
    by, bx = Y[:s0, m].mean(), X[:s0, m].mean()
    dr = seg[-nL // 10:].mean(axis=0) - seg[: nL // 10].mean(axis=0)
    a5 = max(s0, n_on + int(5.0 / dtm))
    L5 = Y[a5:s1]
    n5 = max(1, len(L5))
    dr5 = L5[-n5 // 10:].mean(axis=0) - L5[: n5 // 10].mean(axis=0)
    i1, i2 = s0 + int(0.5 / dtm), s0 + int(2.5 / dtm)
    sx = X[i1:i2, m].mean() - bx
    step = ((Y[i1:i2, m].mean() - by) / sx) if abs(sx) > 1e-9 else np.nan
    tts = tt[s0:s1] - tt[s0]

    def dstd(sig, tq):
        k, b0 = np.polyfit(tq, sig, 1)
        return (sig - (k * tq + b0)).std()

    ny = dstd(seg[10:, m], tts[10:])
    nx = dstd(X[s0 + 10:s1, m], tts[10:])
    dY, dX = np.diff(Y[:, m]), np.diff(X[:, m])
    jump = float(np.abs(dY - dX)[s0:s1].max()) if s1 > s0 else np.nan
    return dict(drift_main=100 * dr[m] / amp, drift_slow=100 * dr5[m] / amp,
                drift_loaded=100 * np.median(dr[loaded] / amp),
                noise_ratio=(ny / nx) if nx > 1e-12 else np.nan,
                flat_main=100 * ny / amp, step_ratio=step, jump_excess=jump,
                ded_delay_s=first_ded_delay(Y, X, dtm, s0, amp))


srows, arows = [], []
print("=" * 122)
print("A. 恒载 9 组（口径同 Document/03；时漂残余为 |末10% − 首10%| / 电平 %）")
print("=" * 122)
for loc, name in STATIC:
    p = os.path.join(TEMP, loc, name, "device_001_seg000.csv")
    if not os.path.exists(p):
        print(f"[skip] 缺文件 {p}")
        continue
    d = L.prep(p)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    s0r, s1r = find_segment(d["tot"])[0]
    s0 = int(np.searchsorted(tu, tu[min(s0r, len(tu) - 1)]))
    s1 = int(np.searchsorted(tu, min(d["t"][s1r], d["span"])))
    amp_v = Xu[s0:s1].mean(axis=0) - Xu[:max(1, s0)].mean(axis=0)
    m = int(np.argmax(amp_v))
    amp = amp_v[m]
    loaded = amp_v > 0.10 * amp_v.max()
    b0 = d["tot"][:max(1, s0)].mean()
    n_on = next((i for i in range(s0, min(s0 + 400, len(tu) - 1))
                 if d["tot"][i] > b0 + 0.05 * (d["tot"][s0:s1].max() - b0)), s0)
    line = f"{loc}/{name} ch{m} amp={amp:7.3f}"
    for k, cls, kw in MAIN + REF:
        Y, c, st = run_variant(cls, kw, tu, Xu)
        mm = static_metrics(Y, Xu, tu, s0, s1, amp, loaded, m, dtm, n_on)
        mm.update(location=loc, dataset=name, algo=k,
                  a_over_amp=(c.A[m] / amp if c is not None and abs(amp) > 1e-9 else np.nan),
                  g_end=(c.g if c is not None else np.nan),
                  loaded_n=(int(c.loaded.sum()) if c is not None else np.nan))
        srows.append(mm)
        line += f" | {k}:{mm['drift_main']:+6.2f}/{mm['drift_slow']:+6.2f}%" \
                f"(A/幅={mm['a_over_amp']:.2f} 首扣={mm['ded_delay_s']:.1f}s)"
    for k, cls, kw in ABL:
        Y, c, st = run_variant(cls, kw, tu, Xu)
        mm = static_metrics(Y, Xu, tu, s0, s1, amp, loaded, m, dtm, n_on)
        arows.append(dict(scope="static", dataset=f"{loc}/{name}", algo=k,
                          drift_main=mm["drift_main"], drift_slow=mm["drift_slow"],
                          noise_ratio=mm["noise_ratio"], step_ratio=mm["step_ratio"],
                          max_abs_gap=np.nan, cap_med=np.nan, cap_min=np.nan,
                          pesc=np.nan, ded_delay_s=mm["ded_delay_s"], epochs=np.nan))
    print(line)

dfs = pd.DataFrame(srows)
dfs.to_csv(os.path.join(RES, "exempt_sweep_static9.csv"), index=False, encoding="utf-8-sig")
agg = dfs.groupby("algo").agg(
    时漂残余_全段=("drift_main", lambda s: s.abs().mean()),
    时漂残余_慢相段=("drift_slow", lambda s: s.abs().mean()),
    时漂残余_受载中位=("drift_loaded", lambda s: s.abs().mean()),
    噪声比=("noise_ratio", "mean"), 平坦度=("flat_main", "mean"),
    阶跃保真=("step_ratio", "mean"), 事件跳变超额=("jump_excess", "max"),
    A占幅度=("a_over_amp", "mean"), g_末端=("g_end", "mean"),
    首扣时延s=("ded_delay_s", "mean"),
).reindex(VK + [k for k, _, _ in REF])
print("\n----- 恒载 9 组汇总（v3 = 现役基线参考）-----")
print(agg.round(2).to_string())
agg.to_csv(os.path.join(RES, "exempt_sweep_static9_summary.csv"), encoding="utf-8-sig")

# ═══════════════════════════ B. 4 份实采录制 ═══════════════════════════
rrows, erows, drows, awin_rows, mech_rows = [], [], [], [], []
store = {}
for tag, path in RECS:
    if not os.path.exists(path):
        print(f"[skip] 缺文件 {path}")
        continue
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    d["periods"] = L.find_periods(tot, dtm)
    d["events"] = [e for e, _ in L.detect_events(tot, dtm)]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    peak = float(tot.max())
    print(f"\n[{tag}] {d['span']:.1f}s / {len(tu)} 帧 / 负载段 {len(d['periods'])} / 事件 {len(d['events'])}")

    Ys, eps, sts = {}, {}, {}
    for k, cls, kw in MAIN + REF:
        Ys[k], c, st = run_variant(cls, kw, tu, Xu)
        eps[k], sts[k] = (c.epoch_t if c is not None else []), st
        if c is not None:
            print(f"   [{k:3s}] A_max={c.A.max():8.2f} loaded={int(c.loaded.sum()):3d} "
                  f"g_end={c.g:+.4f} epoch={len(c.epoch_t)}")
    d["Ys"] = Ys
    periods = [(int(a), min(int(b), len(tu) - 1)) for a, b in d["periods"]]
    for (a, b) in periods:
        if (b - a) * dtm < 3.0:
            continue
        base = float(np.median(tot_s[max(0, a - int(2 / dtm)):a])) if a > 0 else float(tot_s[0])
        onset = next((i for i in range(a, min(a + int(5 / dtm), b))
                      if tot_s[i] > base + 0.05 * (tot_s[a:b].max() - base)), a)
        amp = float(tot_s[a:b].max() - base)
        row = dict(rec=tag, t_onset=float(tu[onset]), amp=amp, dur_s=(b - a) * dtm)
        for k in VK:
            if k == "raw":
                continue
            ep = [e for e in eps[k] if -0.5 <= e - float(tu[onset]) <= 8.0]
            row[f"epoch_{k}"] = (ep[0] - float(tu[onset])) if ep else np.nan
            row[f"theo_{k}"] = (row[f"epoch_{k}"] + KWV[k]["FAST_S"]) if ep else np.nan
            row[f"ded_{k}"] = first_ded_delay(Ys[k], Xu, dtm, onset, amp)
        drows.append(row)
        print(f"   负载段 @{tu[onset]:6.1f}s amp={amp:6.0f} " + " ".join(
            f"| {k}: epoch+{row[f'epoch_{k}']:.1f}s 理论首扣 {row[f'theo_{k}']:.1f}s "
            f"实测首扣 {row[f'ded_{k}']:.1f}s" for k in ("1s", "3s", "5s")))
        for k in ("1s", "3s", "5s"):
            if not np.isfinite(row[f"epoch_{k}"]):
                continue
            fast, awin = KWV[k]["FAST_S"], KWV[k]["EXEMPT_AWIN"]
            t0w = float(tu[onset]) + row[f"epoch_{k}"] + (fast - awin)
            t1w = float(tu[onset]) + row[f"epoch_{k}"] + fast
            w = (tu > t0w) & (tu <= t1w)
            awin_rows.append(dict(rec=tag, t_onset=float(tu[onset]), variant=k, amp=amp,
                                  win_start=float(t0w - tu[onset]), win_end=float(t1w - tu[onset]),
                                  level=float(tot_s[w].mean() - base) / max(amp, 1e-9) if w.any() else np.nan))
        # 机制：负载沿 → 首个真实变载之前的窗内 pending / hold 占空比 + epoch 重启数
        # （只取"快相刚过、真实变载之前"这段，才能把误触发与真实事件分开）
        ev_after = [e for e in d["events"] if tu[e] > tu[onset] + 1.0]
        w_end = float(tu[min(ev_after)]) if ev_after else float(tu[onset]) + 40.0
        w_end = min(w_end, float(tu[onset]) + 40.0)
        for k in ("1s", "3s", "5s"):
            st = sts[k]
            mm = (st[:, 0] >= tu[onset]) & (st[:, 0] <= w_end)
            if mm.sum() < 10:
                continue
            mech_rows.append(dict(rec=tag, t_onset=float(tu[onset]), variant=k,
                                  win_s=w_end - float(tu[onset]),
                                  pending_pct=100 * st[mm, 2].mean(), hold_pct=100 * st[mm, 3].mean(),
                                  epoch_restarts=int(sum(1 for e in eps[k]
                                                         if tu[onset] < e <= w_end)),
                                  ded_peak=float(ded_total(Ys[k], Xu)[mm].max())))

    ev = L.event_table(d, d["events"], Ys, eps["3s"], gain_s=6.0, algos=[k for k in VK if k != "raw"] + ["v3"])
    if len(ev):
        ev["valid_gain"] = ev["raw_gain"].abs() > 0.5 * ev["jump"].abs()
        ev["big_event"] = ev["jump"].abs() >= 2000.0     # 排除 818/1276 ADC 的噪声级小台阶
        ev.insert(0, "rec", tag)
        erows.append(ev)
    for k in VK[1:] + ["v3"]:
        y = L.med_smooth(Ys[k].sum(axis=1), 0.5 / dtm)
        gap = np.abs(y - tot_s)
        ml = L.mid_load_events(ev) if len(ev) else ev
        if len(ml):
            mlb = ml[ml.big_event]
            cap = (mlb[f"gain_{k}"] / mlb["raw_gain"]).replace([np.inf, -np.inf], np.nan).dropna()
        else:
            cap = pd.Series(dtype=float)
        rrows.append(dict(rec=tag, algo=k, max_abs_gap=float(gap.max()),
                          pct_of_peak=100 * float(gap.max()) / peak,
                          events=int(len(ml)), big_events=int(len(mlb)) if len(ml) else 0,
                          cap_med=float(cap.median()) if len(cap) else np.nan,
                          cap_min=float(cap.min()) if len(cap) else np.nan,
                          gap_med=float(ml[f"gap_{k}"].median()) if len(ml) else np.nan,
                          epochs=len(eps[k])))
    store[tag] = dict(tu=tu, Xu=Xu, tot=tot, tot_s=tot_s, dtm=dtm, peak=peak,
                      periods=periods, events=d["events"], Y={k: Ys[k] for k in VK},
                      eps={k: eps[k] for k in VK if k != "raw"}, st=sts)

    # 两因子拆解（实录口径）+ 首个负载段的 pending/hold 机制
    for k, cls, kw in ABL:
        Y, c, st = run_variant(cls, kw, tu, Xu)
        y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
        gap = np.abs(y - tot_s)
        ev2 = L.event_table(d, d["events"], {**Ys, k: Y}, eps["3s"], gain_s=6.0, algos=[k])
        ml2 = L.mid_load_events(ev2)
        ml2 = ml2[ml2.jump.abs() >= 2000.0] if len(ml2) else ml2
        cap2 = (ml2[f"gain_{k}"] / ml2["raw_gain"]).replace([np.inf, -np.inf], np.nan).dropna() \
            if len(ml2) else pd.Series(dtype=float)
        if periods:
            a0, b0_ = periods[0]
            bs = float(np.median(tot_s[max(0, a0 - int(2 / dtm)):a0])) if a0 > 0 else float(tot_s[0])
            on0 = next((i for i in range(a0, min(a0 + int(5 / dtm), b0_))
                        if tot_s[i] > bs + 0.05 * (tot_s[a0:b0_].max() - bs)), a0)
            ev_after = [e for e in d["events"] if tu[e] > tu[on0] + 1.0]
            w_end = min(float(tu[min(ev_after)]) if ev_after else float(tu[on0]) + 40.0,
                        float(tu[on0]) + 40.0)
            mm = (st[:, 0] >= tu[on0]) & (st[:, 0] <= w_end)
            if mm.sum() >= 10:
                mech_rows.append(dict(rec=tag, t_onset=float(tu[on0]), variant=k,
                                      win_s=w_end - float(tu[on0]),
                                      pending_pct=100 * st[mm, 2].mean(),
                                      hold_pct=100 * st[mm, 3].mean(),
                                      epoch_restarts=int(sum(1 for e in c.epoch_t
                                                             if tu[on0] < e <= w_end)),
                                      ded_peak=float(ded_total(Y, Xu)[mm].max())))
        arows.append(dict(scope="rec", dataset=tag, algo=k,
                          drift_main=np.nan, drift_slow=np.nan, noise_ratio=np.nan,
                          step_ratio=np.nan, max_abs_gap=float(gap.max()),
                          cap_med=float(cap2.median()) if len(cap2) else np.nan,
                          cap_min=float(cap2.min()) if len(cap2) else np.nan,
                          pesc=100 * float(gap.max()) / peak, ded_delay_s=np.nan,
                          epochs=(len(c.epoch_t) if c is not None else np.nan)))

dfr = pd.DataFrame(rrows)
dfr.to_csv(os.path.join(RES, "exempt_sweep_recs.csv"), index=False, encoding="utf-8-sig")
dfe = pd.concat(erows, ignore_index=True) if erows else pd.DataFrame()
if len(dfe):
    dfe.to_csv(os.path.join(RES, "exempt_sweep_events.csv"), index=False, encoding="utf-8-sig")
dfd = pd.DataFrame(drows)
dfd.to_csv(os.path.join(RES, "exempt_sweep_deddelay.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame(awin_rows).to_csv(os.path.join(RES, "exempt_sweep_awin.csv"),
                               index=False, encoding="utf-8-sig")
dfm = pd.DataFrame(mech_rows)
dfm.to_csv(os.path.join(RES, "exempt_sweep_mech.csv"), index=False, encoding="utf-8-sig")
dfa = pd.DataFrame(arows)
dfa.to_csv(os.path.join(RES, "exempt_sweep_ablation.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 122)
print("B. 4 份实录汇总（台阶捕获比 = 事件后 6s 显示增量 ÷ 原始增量；仅 |台阶| ≥ 2000 ADC 的有效事件）")
print("=" * 122)
print(dfr.groupby("algo").agg(
    有效事件=("big_events", "sum"),
    台阶捕获比中位=("cap_med", "mean"), 台阶捕获比最小=("cap_min", "min"),
    变载窗最大偏差中位=("gap_med", "median"), 全程最大偏差中位=("max_abs_gap", "median"),
    全程最大偏差占峰值=("pct_of_peak", "median"), epoch数=("epochs", "sum"),
).reindex(VK[1:] + ["v3"]).round(2).to_string())

print("\n----- 首扣时延（相对真实加载沿，s）-----")
print(dfd.groupby("rec")[["theo_1s", "theo_3s", "theo_5s", "ded_1s", "ded_3s", "ded_5s"]]
      .mean().round(2).to_string())
print("全体负载段均值：" + "  ".join(
    f"{k}: 理论{dfd[f'theo_{k}'].mean():.2f}s / 实测{dfd[f'ded_{k}'].mean():.2f}s"
    for k in ("1s", "3s", "5s")))

print("\n----- 机制：加载沿 → 首个真实变载之前（≤40s）窗内的 pending / hold 占空比 -----")
print("[三档主对比]")
print(dfm[dfm.variant.isin(["1s", "3s", "5s"])].groupby("variant")
      [["win_s", "pending_pct", "hold_pct", "epoch_restarts", "ded_peak"]].mean().round(2).to_string())
print("[两因子拆解]")
print(dfm[dfm.variant.isin(ABLK)].groupby("variant")
      [["win_s", "pending_pct", "hold_pct", "epoch_restarts", "ded_peak"]].mean().round(2).to_string())

print("\n----- A 采集窗内的原始电平（占本段幅度；越接近 1.0 说明触面越接近建立）-----")
aw = pd.DataFrame(awin_rows)
print(aw.groupby("variant")[["win_start", "win_end", "level"]].mean().round(3).to_string())
lvl = aw.groupby("variant")["level"].mean()
print(f"相对 5s 档：A(1s)/A(5s) = {lvl['1s'] / lvl['5s']:.3f}，A(3s)/A(5s) = {lvl['3s'] / lvl['5s']:.3f}")

print("\n----- 两因子拆解（免责期长度 × 武装延时；实录与恒载口径）-----")
abl = dfa.copy()
rec_only = abl[abl.scope == "rec"].groupby("algo").agg(
    全程最大偏差=("max_abs_gap", "median"), 占峰值=("pesc", "median"),
    捕获比中位=("cap_med", "mean"), 捕获比最小=("cap_min", "min"),
    epoch数=("epochs", "sum")).reindex(ABLK)
st_only = abl[abl.scope == "static"].groupby("algo").agg(
    时漂残余_全段=("drift_main", lambda s: s.abs().mean()),
    时漂残余_慢相段=("drift_slow", lambda s: s.abs().mean()),
    噪声比=("noise_ratio", "mean")).reindex(ABLK)
print(pd.concat([rec_only, st_only], axis=1).round(2).to_string())

RIDX = {tag: f"r{i}" for i, tag in enumerate(store)}
print("\nnpz 键名对照：" + "  ".join(f"{v}={k}" for k, v in RIDX.items()))
np.savez_compressed(os.path.join(RES, "exempt_sweep.npz"),
                    **{f"{RIDX[tag]}|{k}": store[tag]["Y"][k] for tag in store for k in VK},
                    **{f"{RIDX[tag]}|st|{k}": store[tag]["st"][k] for tag in store
                       for k in VK if k != "raw"},
                    **{f"{RIDX[tag]}|{k}": store[tag][k] for tag in store
                       for k in ("tu", "tot", "tot_s", "dtm", "peak")})

# ═══════════════════════════ C. H1：主对比图（3×3）═══════════════════════════
PRIMARY = "切换负载-快相无责" if "切换负载-快相无责" in store else list(store)[0]
S = store[PRIMARY]
tu, tot, tot_s, dtm, peak = S["tu"], S["tot"], S["tot_s"], S["dtm"], S["peak"]
per = [p for p in S["periods"] if (p[1] - p[0]) * dtm >= 3.0]
on = None
for a, b in per:
    base = float(np.median(tot_s[max(0, a - int(2 / dtm)):a])) if a > 0 else float(tot_s[0])
    cand = next((i for i in range(a, min(a + int(5 / dtm), b))
                 if tot_s[i] > base + 0.05 * (tot_s[a:b].max() - base)), a)
    if tu[cand] > 5.0:
        on = int(cand)
        break
on = int(on if on is not None else per[0][0])
big = dfe[dfe.big_event] if len(dfe) else dfe

fig, axes = plt.subplots(3, 3, figsize=(19, 14))
fig.suptitle("快相免责期 1s / 3s / 5s 同图对比（算法 = drift v5.1；三处参数按 C++ SetFastPhase 联动）\n"
             f"主记录：{PRIMARY}（{S['Xu'].shape[1]}ch，{tu[-1]:.0f}s，首个可分析负载沿 @{tu[on]:.1f}s）",
             fontsize=14)

ax = axes[0, 0]
ax.plot(tu, tot_s, label="原始", **STYLE["raw"])
for k in ("1s", "3s", "5s"):
    ax.plot(tu, L.med_smooth(S["Y"][k].sum(axis=1), 0.5 / dtm), label=f"免责 {k}", **STYLE[k])
ax.axvline(tu[on], color="k", lw=0.8, ls=":")
ax.set_title("(1) 全长时序（总显示量）")
ax.set_xlabel("时间 (s)"); ax.set_ylabel("总量 (ADC)"); ax.legend(fontsize=8)

ax = axes[0, 1]
w0, w1 = max(0, on - int(4 / dtm)), min(len(tu) - 1, on + int(16 / dtm))
ax.plot(tu[w0:w1], tot_s[w0:w1], label="原始", **STYLE["raw"])
for j, k in enumerate(("1s", "3s", "5s")):
    yv = L.med_smooth(S["Y"][k].sum(axis=1), 0.5 / dtm)
    ax.plot(tu[w0:w1], yv[w0:w1], label=f"免责 {k}", **STYLE[k])
    st = S["st"][k]
    m = (st[:, 0] >= tu[w0]) & (st[:, 0] <= tu[w1])
    tsm, pend, hold = st[m, 0], st[m, 2].astype(bool), st[m, 3].astype(bool)
    ymin, h = 0.03 + j * 0.055, 0.045
    for arr, col in ((pend, STYLE[k]["color"]), (hold, "crimson")):
        idx = np.where(arr)[0]
        if len(idx):
            brk = np.where(np.diff(idx) > 1)[0]
            for seg in np.split(idx, brk + 1):
                ax.axvspan(tsm[seg[0]], tsm[seg[-1]], ymin=ymin, ymax=ymin + h,
                           color=col, alpha=0.55 if col == "crimson" else 0.28, lw=0)
    ax.text(tu[w0] + 0.3, ymin + h / 2, f"{k}  ", fontsize=8, va="center", ha="left")
    ep = [e for e in S["eps"][k] if 0 <= e - tu[on] <= 20]
    fast, awin = KWV[k]["FAST_S"], KWV[k]["EXEMPT_AWIN"]
    if ep:
        ax.axvline(ep[0] + fast, color=STYLE[k]["color"], lw=1.0, ls="--")
        ax.axvspan(ep[0] + fast - awin, ep[0] + fast, color=STYLE[k]["color"], alpha=0.15)
ax.set_title("(2) 加载沿放大：虚线=首帧开始扣，色块=A 采集窗，下方条带=逐档 pending(浅)/hold(红)")
ax.set_xlabel("时间 (s)"); ax.set_ylabel("总量 (ADC)"); ax.legend(fontsize=8, loc="lower right")

ax = axes[0, 2]
xs = np.arange(3)
for j, k in enumerate(("1s", "3s", "5s")):
    ax.bar(xs[j] - 0.18, dfd[f"theo_{k}"].mean(), 0.34, color=STYLE[k]["color"], alpha=0.9,
           label="理论 = 2.5s 确认 + 免责期" if j == 0 else None)
    ax.bar(xs[j] + 0.18, dfd[f"ded_{k}"].mean(), 0.34, color=STYLE[k]["color"], alpha=0.4,
           label="实测：首个可见扣除" if j == 0 else None)
    ax.scatter(np.full(len(dfd), xs[j]) + 0.18, dfd[f"ded_{k}"], s=20, color="k", zorder=5)
    ax.annotate(f"{dfd[f'ded_{k}'].mean():.1f}s",
                (xs[j], max(dfd[f"theo_{k}"].mean(), dfd[f"ded_{k}"].mean())), fontsize=9,
                ha="center", va="bottom")
ax.set_xticks(xs); ax.set_xticklabels(["免责 1s", "免责 3s", "免责 5s"])
ax.set_ylabel("从真实加载沿起算 (s)")
ax.set_title(f"(3) 首扣时延（{len(dfd)} 个负载段；黑点 = 各段实测）")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

ax = axes[1, 0]
keys = ["时漂残余_全段", "时漂残余_慢相段"]
xs = np.arange(len(keys))
rawv = [dfs[dfs.algo == "raw"]["drift_main"].abs().mean(), dfs[dfs.algo == "raw"]["drift_slow"].abs().mean()]
for j, k in enumerate(("1s", "3s", "5s")):
    v = [agg.loc[k, "时漂残余_全段"], agg.loc[k, "时漂残余_慢相段"]]
    b = ax.bar(xs + (j - 1) * 0.26, v, 0.25, color=STYLE[k]["color"], label=f"免责 {k}")
    ax.bar_label(b, fmt="%.2f", fontsize=8)
for i, rv in enumerate(rawv):
    ax.plot([i - 0.42, i + 0.42], [rv, rv], color="0.45", ls=":", lw=1.2)
    ax.text(i + 0.44, rv, f"原始 {rv:.1f}%", fontsize=8, color="0.35", va="center")
ax.axhline(agg.loc["v3", "时漂残余_慢相段"], color="0.2", ls="--", lw=1.0)
ax.set_xticks(xs); ax.set_xticklabels(keys)
ax.set_ylabel("时漂残余 (%)  越小越好")
ax.set_title("(4) 恒载 9 组：时漂残余（虚线 = v3 慢相段 %.2f%%）" % agg.loc["v3", "时漂残余_慢相段"])
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

ax = axes[1, 1]
ks = ["1s", "3s", "5s"]
gg = dfr.groupby("algo")
cap_med = [gg.get_group(k)["cap_med"].mean() for k in ks]
cap_min = [gg.get_group(k)["cap_min"].min() for k in ks]
gap_med = [gg.get_group(k)["gap_med"].median() for k in ks]
b1 = ax.bar(np.arange(3) - 0.21, cap_med, 0.2, color=[STYLE[k]["color"] for k in ks], label="捕获比中位")
b2 = ax.bar(np.arange(3), cap_min, 0.2, color=[STYLE[k]["color"] for k in ks], alpha=0.5, label="捕获比最小")
ax.set_ylim(0, 1.2); ax.set_ylabel("台阶捕获比（1.0 = 完整透传）")
ax2 = ax.twinx()
b3 = ax2.bar(np.arange(3) + 0.23, gap_med, 0.2, color="0.35", label="变载窗最大偏差中位")
ax2.set_ylabel("变载窗最大偏差中位 (ADC)  越小越好")
ax.set_xticks(np.arange(3)); ax.set_xticklabels([f"免责 {k}" for k in ks])
ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
ax2.bar_label(b3, fmt="%.0f", fontsize=8)
ax.set_title(f"(5) 变载跟踪（4 份实录，|台阶| ≥ 2000 ADC 的有效事件 {int(dfr[dfr.algo=='3s'].big_events.sum())} 个）")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")

ax = axes[1, 2]
w0, w1 = max(0, on - int(2 / dtm)), min(len(tu) - 1, on + int(60 / dtm))
span = tot_s[on:on + int(60 / dtm)].max() - tot_s[max(0, on - int(2 / dtm))]
for k in ("1s", "3s", "5s"):
    ax.plot(tu[w0:w1] - tu[on], L.med_smooth(ded_total(S["Y"][k], S["Xu"]), 0.5 / dtm)[w0:w1],
            label=f"免责 {k}", **STYLE[k])
ax.axhline(0.16 * span, color="0.5", ls=":", lw=1.0)
ax.text(59, 0.16 * span, "快相占增量下限 16%", fontsize=8, color="0.35", va="bottom", ha="right")
ax.set_title("(6) 已生效扣除量（同一负载段；0 = 加载沿）")
ax.set_xlabel("加载后时间 (s)"); ax.set_ylabel("扣除总量 (ADC)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2, 0]
tt = np.arange(0.0, 9.0, 0.05)
prof = []
for t2, St in store.items():
    t_u, t_s, dd = St["tu"], St["tot_s"], St["dtm"]
    for a, b in [p for p in St["periods"] if (p[1] - p[0]) * dd >= 8.0]:
        if a <= 0:
            continue
        base = float(np.median(t_s[max(0, a - int(2 / dd)):a]))
        pk = float(t_s[a:b].max() - base)
        if pk <= 0:
            continue
        seg = np.interp(a * dd + tt, t_u, t_s, left=np.nan, right=np.nan)
        prof.append((seg - base) / pk)
P = np.vstack([p for p in prof if np.isfinite(p).all()])
ax.plot(tt, np.median(P, axis=0), color="0.2", lw=1.6, label=f"原始归一化轮廓（{len(P)} 段中位）")
ax.fill_between(tt, np.percentile(P, 25, axis=0), np.percentile(P, 75, axis=0), color="0.7", alpha=0.35)
lvl = aw.groupby("variant")["level"].mean()
for k in ("1s", "3s", "5s"):
    fast, awin = KWV[k]["FAST_S"], KWV[k]["EXEMPT_AWIN"]
    ax.axvspan(2.5 + fast - awin, 2.5 + fast, color=STYLE[k]["color"], alpha=0.25)
    ax.annotate(f"{k}: 窗内电平 {lvl[k]:.2f}", (2.5 + fast - awin / 2, 1.05),
                fontsize=9, color=STYLE[k]["color"], ha="center")
ax.axvline(2.5, color="k", ls=":", lw=0.9)
ax.text(2.56, 0.06, "epoch 起点\n= 台阶 + 2.5s", fontsize=8)
ax.set_xlabel("真实加载沿后时间 (s)"); ax.set_ylabel("归一化电平（÷ 本段幅度）")
ax.set_title("(7) 机制：A 采集窗落在快相的哪一段（实测窗内电平）")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

ax = axes[2, 1]
if len(big):
    for k in ("1s", "3s", "5s"):
        r = (big[f"gain_{k}"] / big["raw_gain"]).replace([np.inf, -np.inf], np.nan)
        ax.scatter(big["ratio"], r, s=30, color=STYLE[k]["color"], alpha=0.85, label=f"免责 {k}")
    ax.axhline(1.0, color="0.4", ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("事件台阶比（台阶 ÷ 变载前电平）")
    ax.set_ylabel("台阶捕获比（1.0 = 完整透传）")
    ax.set_title(f"(8) 逐个有效变载事件（{len(big)} 个）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2, 2]
for k in ("1s", "3s", "5s"):
    y = L.med_smooth(S["Y"][k].sum(axis=1), 0.5 / dtm)
    ax.plot(tu, np.abs(y - tot_s), label=f"免责 {k}", color=STYLE[k]["color"], lw=1.0)
ax.axhline(0.10 * peak, color="0.5", ls=":", lw=1.0)
ax.text(tu[-1], 0.10 * peak, "峰值 10%", fontsize=8, color="0.35", va="bottom", ha="right")
ax.set_title("(9) 全程 |显示 − 原始|（主记录）")
ax.set_xlabel("时间 (s)"); ax.set_ylabel("偏差 (ADC)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout(rect=(0, 0, 1, 0.935))
fp1 = os.path.join(FIG, "H1_exempt_1s_3s_5s.png")
fig.savefig(fp1, dpi=110)
plt.close(fig)
print(f"\n图已保存：{fp1}")

# ═══════════════════════════ D. H2：两因子拆解图 ═══════════════════════════
fig2, ax2s = plt.subplots(1, 3, figsize=(19, 5.6))
fig2.suptitle("免责期 1s 的两个连带因子拆解（主记录 " + PRIMARY + "）："
              "① 免责期一结束就采 A（A 窗无法与免责期解耦）  ② 电平判据武装延时 = 免责期", fontsize=13)
A = store[PRIMARY]
a_tu, a_tot, a_dtm = A["tu"], A["tot_s"], A["dtm"]
a_on = on
ax = ax2s[0]
ax.plot(a_tu, a_tot, color="0.55", lw=1.0, label="原始")
for j, (k, cls, kw) in enumerate(ABL):
    Y, c, st = run_variant(cls, kw, a_tu, A["Xu"])
    ax.plot(a_tu, L.med_smooth(Y.sum(axis=1), 0.5 / a_dtm), color=ABL_STYLE[k], lw=1.3, label=k)
    m = (st[:, 0] >= a_tu[max(0, a_on - int(1 / a_dtm))]) & \
        (st[:, 0] <= a_tu[min(len(a_tu) - 1, a_on + int(20 / a_dtm))])
    tsm, pend, hold = st[m, 0], st[m, 2].astype(bool), st[m, 3].astype(bool)
    ymin, h = 0.02 + (4 - j) * 0.05, 0.04
    for arr, col, al in ((pend, ABL_STYLE[k], 0.35), (hold, "crimson", 0.6)):
        idx = np.where(arr)[0]
        if len(idx):
            for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                ax.axvspan(tsm[seg[0]], tsm[seg[-1]], ymin=ymin, ymax=ymin + h,
                           color=col, alpha=al, lw=0)
    ax.text(a_tu[max(0, a_on - int(1 / a_dtm))] + 0.2, ymin + h / 2, k, fontsize=7.5,
            va="center", ha="left")
ax.set_xlim(a_tu[max(0, a_on - int(3 / a_dtm))], a_tu[min(len(a_tu) - 1, a_on + int(20 / a_dtm))])
ax.set_title("(a) 加载沿后 20s：下方条带 = pending(浅)/hold(红) 误触发窗口")
ax.set_xlabel("时间 (s)"); ax.set_ylabel("总量 (ADC)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)

ax = ax2s[1]
rec_only = dfa[dfa.scope == "rec"]
xs = np.arange(len(ABLK))
mg = [rec_only[rec_only.algo == k]["max_abs_gap"].median() for k in ABLK]
b = ax.bar(xs, mg, 0.62, color=[ABL_STYLE[k] for k in ABLK])
ax.bar_label(b, fmt="%.0f", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(ABLK, rotation=18, fontsize=8.5)
ax.set_ylabel("4 份实录 全程最大偏差 中位 (ADC)")
ax.set_title("(b) 全程最大偏差（越低越好）")
ax.grid(axis="y", alpha=0.3)

ax = ax2s[2]
st_only = dfa[dfa.scope == "static"]
mf = [st_only[st_only.algo == k]["drift_main"].abs().mean() for k in ABLK]
sl = [st_only[st_only.algo == k]["drift_slow"].abs().mean() for k in ABLK]
b1 = ax.bar(xs - 0.2, mf, 0.38, color=[ABL_STYLE[k] for k in ABLK], label="全段（含快相增量）")
b2 = ax.bar(xs + 0.2, sl, 0.38, color=[ABL_STYLE[k] for k in ABLK], alpha=0.5,
            label="慢相段（onset+5s 起）")
ax.bar_label(b1, fmt="%.2f", fontsize=8); ax.bar_label(b2, fmt="%.2f", fontsize=8)
ax.set_xticks(xs); ax.set_xticklabels(ABLK, rotation=18, fontsize=8.5)
ax.set_ylabel("恒载 9 组 时漂残余 (%)")
ax.set_title("(c) 恒载时漂残余")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

fig2.tight_layout(rect=(0, 0, 1, 0.90))
fp2 = os.path.join(FIG, "H2_exempt_1s_ablation.png")
fig2.savefig(fp2, dpi=110)
plt.close(fig2)
print(f"图已保存：{fp2}")
print(f"主记录 {PRIMARY}：首个可分析负载段 @{tu[on]:.1f}s（峰值 {peak:.0f} ADC）")
