# -*- coding: utf-8 -*-
"""v2.0 阶段二 · 最终候选组合的跨数据集不劣化回归（14 份数据 × 4 臂）。

臂：base / C（限幅 0.005）/ C+F2+F3（+谷底出口 +重锚归零）/ C+F2+F3+F5（+g 负向下界）
指标（越小越好或越接近 1 越好，逐数据集给出，并给出相对 base 的差值）：
  resid   受载段时漂残余 = 滑窗(5 s)极差中位 / 段中位
  lead    加载沿后 0.3~1.2 s 的超前量中位（占台阶幅度）
  T5%     进入自身稳态 ±5%·台阶 的时间中位
  G4_5    沿后 4~5 s 的台阶捕获比中位
  offabs  受载平台 |显示−原始| 的中位
"""
import importlib.util
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SPEC = importlib.util.spec_from_file_location("glm53_v6", L.PROTO_V6)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)

ALPHA = 0.005
W_S, VALLEY_FRAC, RANGE_FRAC = 40.0, 0.15, 0.05
G_FLOOR, G_RESET_S = -0.05, 0.30


class Arm(P.GLM53v6):
    def __init__(self, n, clamp=None, f2=False, f3=False, f5=False):
        super().__init__(n)
        self.KAPPA_ONSET, self.KAPPA_RESTEP, self.HO_MIN = 1.05, 1.12, 3.5
        self.clamp, self.f2, self.f3, self.f5 = clamp, f2, f3, f5
        self._win = []
        # 单调队列（O(1) 均摊）维护窗口最小/最大，替代每帧 min()/max()
        self._dq_lo = collections.deque()
        self._dq_hi = collections.deque()
        self._idx = 0
        self.valley = False
        self.valley_run = 0.0

    def _valley_update(self, ts, dt):
        v = self.ts_smooth
        self._win.append(v)
        i = self._idx
        self._idx += 1
        while self._dq_lo and self._win[self._dq_lo[-1]] >= v:
            self._dq_lo.pop()
        self._dq_lo.append(i)
        while self._dq_hi and self._win[self._dq_hi[-1]] <= v:
            self._dq_hi.pop()
        self._dq_hi.append(i)
        cut = i - 4000
        while self._dq_lo and self._dq_lo[0] <= cut:
            self._dq_lo.popleft()
        while self._dq_hi and self._dq_hi[0] <= cut:
            self._dq_hi.popleft()
        if len(self._win) < 4000:
            self.valley = False
            self.valley_run = 0.0
            return
        mn = self._win[self._dq_lo[0]]
        mx = self._win[self._dq_hi[0]]
        eps = 1e-6 * (1 + abs(mx))
        self.valley = bool(v < mn + VALLEY_FRAC * max(mx - mn, eps)
                           and (mx - mn) > RANGE_FRAC * max(abs(mx), eps))
        if self.valley:
            self.valley_run += max(dt, 0.0)
        else:
            self.valley_run = 0.0
        # 窗口裁剪（限制内存）
        if len(self._win) > 8000:
            drop = len(self._win) - 4000
            self._win = self._win[drop:]
            self._dq_lo = collections.deque(x - drop for x in self._dq_lo)
            self._dq_hi = collections.deque(x - drop for x in self._dq_hi)
            self._idx -= drop

    def process(self, ts, v):
        raw = np.asarray(v, float).copy()
        dt_raw = ts - self.last_ts if not self.first else 0.0
        dt = min(max(dt_raw, 0.0), 0.1)
        out = np.asarray(super().process(ts, raw.copy()), float).copy()
        self._valley_update(ts, dt)
        if self.f5:
            if self.g < G_FLOOR:
                self.g = G_FLOOR
            if self.valley and self.valley_run >= G_RESET_S and self.g < 0:
                self.g = 0.0
        if self.clamp is not None:
            out = np.minimum(out, raw + self.clamp * np.abs(raw))
        return out

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        if self.f2 and ev is not None:
            tau = ts - ev["t0"]
            if tau > 0.60 and self.valley:
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

    def _reanchor(self, ts, v):
        if self.f3 and self.valley and self.valley_run >= 0.20:
            self._to_idle()
            return
        return super()._reanchor(ts, v)


def run(el, V, **kw):
    c = Arm(V.shape[1], **kw)
    out = np.empty((len(el), V.shape[1]))
    for i in range(len(el)):
        out[i] = c.process(float(el[i]), V[i])
    return out.sum(1)


def read_simple(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    hdr = rows[di + 1].split(",")
    idx = [i for i, h in enumerate(hdr) if h.startswith("ch") and not h.startswith("ch_")]
    if not idx:
        ncol = len(rows[di + 2].split(","))
        idx = list(range(ncol - L.NCH, ncol))
    el, vals = [], []
    for r in rows[di + 2:]:
        if not r.strip():
            continue
        f = r.split(",")
        el.append(float(f[1]))
        vals.append([float(f[i]) for i in idx])
    return np.asarray(el), np.asarray(vals)


def edges(el, tot):
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(tot, 3), np.percentile(tot, 97)
    if hi - lo <= 0:
        return []
    up, _ = L.edges_from_tot(np.convolve(tot, k, mode="same"), lo + 0.5 * (hi - lo))
    tu = []
    for i in up:
        if not tu or el[i] - tu[-1] > 1.0:
            tu.append(float(el[i]))
    return tu


def evaluate(el, pre, out, tu):
    resid = float("nan")
    thr = np.percentile(pre, 60)
    m = pre > thr
    if m.sum() > 400:
        seg = out[m]
        w = 500
        step = max(w // 4, 1)
        if len(seg) > w:
            rs = [np.max(seg[i:i + w]) - np.min(seg[i:i + w])
                  for i in range(0, len(seg) - w, step)]
            resid = float(np.median(rs) / max(np.median(seg), 1e-9))
    lead, t5, g = [], [], []
    for t in tu:
        m0 = (el >= t - 1.2) & (el < t - 0.35)
        if m0.sum() < 5:
            continue
        bp, bo = float(np.median(pre[m0])), float(np.median(out[m0]))
        te = min(t + 30.0, el[-1] - 0.2)
        m1 = (el >= te - 1.5) & (el <= te)
        ps, os_ = float(np.median(pre[m1])), float(np.median(out[m1]))
        A = ps - bp
        if abs(A) < 500:
            continue
        m2 = (el >= t + 0.3) & (el <= t + 1.2)
        if m2.any():
            lead.append(float(np.median(out[m2] - pre[m2]) / abs(A)))
        idx = np.where((el >= t - 0.05) & (el <= te))[0]
        for k in range(len(idx)):
            if abs(out[idx[k]] - os_) <= 0.05 * abs(A):
                t5.append(el[idx[k]] - t)
                break
        m3 = (el >= t + 4.0) & (el <= t + 5.0)
        if m3.any():
            g.append((float(np.median(out[m3])) - bo) / A)
    offabs = float("nan")
    if m.sum() > 400:
        offabs = float(np.median(np.abs(out[m] - pre[m])))
    return dict(resid=resid,
                lead=np.median(lead) if lead else float("nan"),
                t5=np.median(t5) if t5 else float("nan"),
                G=np.median(g) if g else float("nan"),
                offabs=offabs)


def main():
    sel = os.environ.get("V20_CASES", "new").strip()
    want = set(x for x in sel.split(",") if x)
    cases = []
    if "new" in want or "all" in want:
        cases.append(("★新录制(ADC/大偏置)", os.path.join(L.DS_ZERO, "device_001_pre_seg0.csv")))
    base_dir = os.path.join(L.ROOT, "temp", "原始数据only")
    if "all" in want or "legacy" in want:
        for grp in ("四指指尖", "右拇指指尖", "左拇指指尖"):
            for d in ("数据1", "数据2", "数据3"):
                p = os.path.join(base_dir, grp, d, "device_001_seg000.csv")
                if os.path.exists(p):
                    cases.append((f"{grp}/{d}", p))
    if "all" in want or "vary" in want:
        root = os.path.join(base_dir, "变化负载")
        for d in sorted(os.listdir(root)):
            for dp, _dn, fn in os.walk(os.path.join(root, d)):
                if "device_001_seg000.csv" in fn:
                    cases.append((f"变化负载/{d}", os.path.join(dp, "device_001_seg000.csv")))
    if not cases:
        cases.append(("★新录制(ADC/大偏置)", os.path.join(L.DS_ZERO, "device_001_pre_seg0.csv")))

    arms = [("base", dict()), ("C", dict(clamp=ALPHA)),
            ("C+F2F3", dict(clamp=ALPHA, f2=True, f3=True)),
            ("C+F2F3F5", dict(clamp=ALPHA, f2=True, f3=True, f5=True))]

    print(f"{'数据集':30s}{'臂':>10}{'resid':>9}{'lead':>9}{'T5%':>8}{'G':>8}{'offabs':>9}",
          flush=True)
    summary = {tag: [] for tag, _ in arms}
    for name, path in cases:
        try:
            el, V = read_simple(path)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:30s}  读取失败 {exc}", flush=True)
            continue
        if len(el) < 500:
            continue
        pre = V.sum(1)
        tu = edges(el, pre)
        base_m = None
        for tag, kw in arms:
            out = run(el, V, **kw)
            m = evaluate(el, pre, out, tu)
            line = (f"{name:30s}{tag:>10}{m['resid']:9.4f}"
                    f"{100*m['lead']:8.2f}%{m['t5']:8.2f}{m['G']:8.3f}{m['offabs']:9.0f}")
            if base_m is None:
                base_m = m
            else:
                line += (f"   Δresid={m['resid']-base_m['resid']:+.4f}"
                         f" Δlead={100*(m['lead']-base_m['lead']):+.2f}pp"
                         f" ΔG={m['G']-base_m['G']:+.3f}")
            print(line, flush=True)
            summary[tag].append(m)
        print(flush=True)
    print("== 汇总（各臂相对 base 的中位变化）==", flush=True)
    for tag, _ in arms[1:]:
        d_r = np.nanmedian([m["resid"] - b["resid"] for m, b in
                            zip(summary[tag], summary["base"])])
        d_l = np.nanmedian([m["lead"] - b["lead"] for m, b in
                            zip(summary[tag], summary["base"])])
        d_g = np.nanmedian([m["G"] - b["G"] for m, b in
                            zip(summary[tag], summary["base"])])
        print(f"  {tag:10s} Δresid={d_r:+.5f}  Δlead={100*d_l:+.2f}pp  ΔG={d_g:+.4f}",
              flush=True)


if __name__ == "__main__":
    main()
