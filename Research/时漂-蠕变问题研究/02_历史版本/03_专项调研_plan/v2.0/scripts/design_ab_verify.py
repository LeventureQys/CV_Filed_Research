# -*- coding: utf-8 -*-
"""v2.0 阶段二 · 设计验证（组合候选 A/B）——在原型上验证三项已批准修复的**组合**效果。

已批准的三项（问题清单 v2 置顶区裁决）：
  C  · 显示单侧限幅：out ≤ raw + α·|raw|，α = 0.005                         （治瞬态过充）
  F2 · epoch 内卸载出口：事件期回到"谷底"即出事件（不依赖 idle_now）          （治卸载被漏掉）
  F3 · Reanchor/ToIdle 加"回到谷底即归零"口子                                （治错误扣除被继承）
  F5 · 慢相 g 的符号/漂移保护                                                （治显示被长期抬高）

谷底判据（候选，全部为**滚动窗相对判据**，不依赖"空载读数≈0"）：
  w_min / w_max = 最近 W 秒 ts_smooth 的最小 / 最大值
  near_valley   = ts_smooth < w_min + VALLEY_FRAC·(w_max − w_min)
  enough_range  = (w_max − w_min) > RANGE_FRAC·w_max
  valley        = near_valley ∧ enough_range

本脚本输出：base（现役）vs 各组合臂的「过充 / 长时偏移 / 速度 / 捕获比」四联表，
并给出"改坏必非零"的阳性对照（故意关闭 F2/F3 时指标必须回到 base 量级）。
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SPEC = importlib.util.spec_from_file_location("glm53_v6", L.PROTO_V6)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)

# ── 现役参数集（必须显式覆盖：归档原型默认 κ_onset=1.30 / HO_MIN=5.0 是 plan-v1.0 之前的旧值）──
KAPPA_ONSET = 1.05       # = C++ kKappaOnset
KAPPA_RESTEP = 1.12      # = C++ kKappaRestep
HO_MIN = 3.5             # = C++ kHoMinS

# ── 候选参数（设计文档必须逐条引用这些名字与默认值） ──
CLAMP_ALPHA = 0.005      # C   显示单侧限幅系数
W_S = 40.0               # F2  谷底滚动窗（秒）
VALLEY_FRAC = 0.15       # F2  距窗口谷底的相对位置门
RANGE_FRAC = 0.05        # F2  "窗口内确实发生过加载"的门（相对 w_max）
REANCHOR_TO_IDLE_S = 0.20  # F3 重锚前要求"回到谷底"持续多久
G_NEG_FLOOR = -0.05      # F5  g 的负向下界（低于它即按此值扣除并使用）
G_VALLEY_RESET_S = 0.30  # F5  谷底持续多久把 g 归零


class Candidate(P.GLM53v6):
    """在归档原型上叠加三项候选（逐行只在必要处改写）。"""

    def __init__(self, n, clamp_alpha=None, f2=False, f3=False, f5=False,
                 w_s=W_S, valley_frac=VALLEY_FRAC, range_frac=RANGE_FRAC):
        super().__init__(n)
        self.KAPPA_ONSET = KAPPA_ONSET
        self.KAPPA_RESTEP = KAPPA_RESTEP
        self.HO_MIN = HO_MIN
        self.clamp_alpha = clamp_alpha
        self.f2 = f2
        self.f3 = f3
        self.f5 = f5
        self.w_s = w_s
        self.valley_frac = valley_frac
        self.range_frac = range_frac
        self._ts_hist = []      # (ts, ts_smooth) 滚动窗
        self.valley_now = False
        self.valley_run = 0.0
        self.n_exit_f2 = 0
        self.n_reset_f3 = 0
        self.n_reset_f5 = 0
        self.gmin_seen = 0.0

    # ── 谷底判据 ────────────────────────────────────────────────
    def _update_valley(self, ts):
        self._ts_hist.append((ts, self.ts_smooth))
        t_cut = ts - self.w_s
        while self._ts_hist and self._ts_hist[0][0] < t_cut:
            self._ts_hist.pop(0)
        arr = [v for _t, v in self._ts_hist]
        w_min = min(arr)
        w_max = max(arr)
        eps = 1e-6 * (1.0 + abs(w_max))
        near = self.ts_smooth < w_min + self.valley_frac * max(w_max - w_min, eps)
        enough = (w_max - w_min) > self.range_frac * max(abs(w_max), eps)
        self.valley_now = bool(near and enough)

    def process(self, ts, v):
        v = np.asarray(v, float).copy()
        raw = v.copy()
        out = np.asarray(super().process(ts, v), float).copy()
        self._update_valley(ts)
        if self.valley_now:
            self.valley_run += 0.01
        else:
            self.valley_run = 0.0
        if self.f5:
            if self.g < G_NEG_FLOOR:
                self.g = G_NEG_FLOOR
            self.gmin_seen = min(self.gmin_seen, self.g)
            if self.valley_now and self.valley_run >= G_VALLEY_RESET_S and self.g < 0.0:
                self.g = 0.0
                self.n_reset_f5 += 1
        if self.clamp_alpha is not None:
            out = np.minimum(out, raw + self.clamp_alpha * np.abs(raw))
        return out

    # ── F2：事件期卸载出口（不依赖 idle_now） ────────────────────
    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        if self.f2 and ev is not None:
            tau = ts - ev["t0"]
            if self.valley_now and tau > 0.60 and ev["inc_max"] > eps:
                self.n_exit_f2 += 1
                self.ev = None
                self.state = "idle"
                self.last_ev_ts = ts
                self.idle_since = ts
                self.ev_block_until = ts + self.UNLOAD_BLOCK
                self.hold_comp = None
                self.loaded[:] = False
                self.g = 0.0
                self.last_out = v.copy()
                return v
        return super()._run_event(ts, v, total, dt, eps, idle_now)

    # ── F3：Reanchor 加"回到谷底即归零" ─────────────────────────
    def _reanchor(self, ts, v):
        if self.f3 and self.valley_now and self.valley_run >= REANCHOR_TO_IDLE_S:
            self.n_reset_f3 += 1
            self._to_idle()
            return
        return super()._reanchor(ts, v)


# ── 指标 ────────────────────────────────────────────────────────
def load_edges(el, tot):
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(tot, 3), np.percentile(tot, 97)
    up, _ = L.edges_from_tot(np.convolve(tot, k, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in up:
        if not tu or el[i] - tu[-1] > 1.0:
            tu.append(float(el[i]))
    return tu


def metrics(el, pre_tot, out_tot, t_ups):
    rows = []
    for t_up in t_ups:
        m0 = (el >= t_up - 1.2) & (el < t_up - 0.35)
        if m0.sum() < 5:
            continue
        bp = float(np.median(pre_tot[m0]))
        t_end = min(t_up + 30.0, el[-1] - 0.2)
        m1 = (el >= t_end - 1.5) & (el <= t_end)
        ps, os_ = float(np.median(pre_tot[m1])), float(np.median(out_tot[m1]))
        A = ps - bp
        if abs(A) < 500:
            continue
        idx = np.where((el >= t_up - 0.05) & (el <= t_end))[0]
        lead = (out_tot[idx] - pre_tot[idx]) / abs(A)
        tail = (out_tot[idx] - os_) / abs(A)
        t5 = float("nan")
        for k in range(len(idx)):
            if abs(out_tot[idx[k]] - os_) <= 0.05 * abs(A):
                t5 = el[idx[k]] - t_up
                break
        rows.append(dict(t_up=t_up, A=A, lead_pk=float(np.max(lead)),
                         tail_pk=float(np.max(tail)),
                         t5=t5, bias=(os_ - ps) / abs(A)))
    return rows


def offset_curve(el, pre_tot, out_tot):
    """受载平台的偏移（逐帧相减后聚合到 0.5 s 桶），返回 (t[], off[])"""
    off = out_tot - pre_tot
    step = 0.5
    t = el[0]
    ts, vs = [], []
    while t < el[-1]:
        m = (el >= t) & (el < t + step)
        if m.any():
            ts.append(t)
            vs.append(float(np.mean(off[m])))
        t += step
    return np.asarray(ts), np.asarray(vs)


def idle_off_level(el, pre_tot, out_tot):
    """空载段（pre 低于 p10）的 |off| 最大与中位 —— 卸载后是否残留"""
    thr = np.percentile(pre_tot, 10)
    m = pre_tot < thr
    if m.sum() < 50:
        return float("nan"), float("nan")
    off = np.abs(out_tot[m] - pre_tot[m])
    return float(np.median(off)), float(np.max(off))


def replay(el, V, **kw):
    n, ch = V.shape
    c = Candidate(ch, **kw)
    out = np.empty((n, ch))
    for i in range(n):
        out[i] = c.process(float(el[i]), V[i])
    return c, out.sum(1)


def main():
    d = L.load_dataset(L.DS_ZERO)
    el = d["pre"]["el"]
    V = d["pre"]["V"]
    pre_tot = V.sum(1)
    t_ups = load_edges(el, pre_tot)
    print(f"数据集 = 新录制（33908 帧 / 337.5 s），加载沿 = {[round(t,2) for t in t_ups]}")
    print(f"候选参数：α={CLAMP_ALPHA}  W={W_S}s valley_frac={VALLEY_FRAC} "
          f"range_frac={RANGE_FRAC}  g_floor={G_NEG_FLOOR}\n")

    arms = [
        ("base(现役)", dict()),
        ("C 只限幅", dict(clamp_alpha=CLAMP_ALPHA)),
        ("F2 只卸载出口", dict(f2=True)),
        ("F3 只重锚归零", dict(f3=True)),
        ("F5 只 g 保护", dict(f5=True)),
        ("C+F2+F3+F5(推荐)", dict(clamp_alpha=CLAMP_ALPHA, f2=True, f3=True, f5=True)),
        ("阳性对照 F2F3 关掉", dict(clamp_alpha=CLAMP_ALPHA)),
    ]
    print(f"{'臂':22s}{'超前峰中位':>10}{'超前峰最大':>10}{'尾峰中位':>9}"
          f"{'T5%中位':>8}{'稳态偏差中位':>12}{'空载|off|中位':>12}{'空载|off|最大':>12}")
    results = {}
    for tag, kw in arms:
        c, osum = replay(el, V, **kw)
        rows = metrics(el, pre_tot, osum, t_ups)
        if not rows:
            print(f"{tag:22s}  无有效事件")
            continue
        lead = [r["lead_pk"] for r in rows]
        tail = [r["tail_pk"] for r in rows]
        t5 = [r["t5"] for r in rows if r["t5"] == r["t5"]]
        bias = [r["bias"] for r in rows]
        io_med, io_max = idle_off_level(el, pre_tot, osum)
        results[tag] = (c, osum, rows)
        print(f"{tag:22s}{100*np.median(lead):9.2f}%{100*max(lead):9.2f}%"
              f"{100*np.median(tail):8.2f}%"
              f"{(np.median(t5) if t5 else float('nan')):8.2f}"
              f"{100*np.median(bias):11.2f}%{io_med:12.0f}{io_max:12.0f}")
        print(f"{'':22s}  F2退出={c.n_exit_f2:3d}  F3归零={c.n_reset_f3:3d}  "
              f"F5归零={c.n_reset_f5:3d}  g_min={c.gmin_seen:+.3f}  末态={c.state}")

    # ── 长时偏移曲线：base vs 推荐组合 ──
    print("\n── 受载段偏移（显示−原始）随时间：base vs 推荐组合（1 s 桶，抽样）──")
    if "base(现役)" in results and "C+F2+F3+F5(推荐)" in results:
        ts_b, off_b = offset_curve(el, pre_tot, results["base(现役)"][1])
        ts_r, off_r = offset_curve(el, pre_tot, results["C+F2+F3+F5(推荐)"][1])
        print(f"{'t':>7}{'base':>10}{'推荐':>10}{'Δ':>9}")
        for i in range(0, len(ts_b), 10):
            if i < len(ts_r):
                print(f"{ts_b[i]:7.1f}{off_b[i]:10.0f}{off_r[i]:10.0f}"
                      f"{off_r[i]-off_b[i]:9.0f}")
        # 逐 30 s 桶：是否单调累积
        print("\n  30 s 桶偏移中位（判断是否单调累积）：")
        for t0 in range(0, 330, 30):
            mb = (ts_b >= t0) & (ts_b < t0 + 30)
            if mb.any():
                print(f"   t={t0:3d}~{t0+30:3d}s   base={np.median(off_b[mb]):8.0f}   "
                      f"推荐={np.median(off_r[mb]):8.0f}")


if __name__ == "__main__":
    main()
