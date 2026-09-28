# -*- coding: utf-8 -*-
"""v2.0 PROBE-F：用原型 glm53_v6 回放新录制，打印 41~48 s 的内部状态（MainAgent 快速核验）。

与 A1 席位的完整骨架互补：本探针只做「一个区间 + 关键字段」的最小可读输出，
用于在写问题清单前先确认状态机在该区间发生了什么。
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SPEC = importlib.util.spec_from_file_location("glm53_v6", L.PROTO_V6)
PROTO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROTO)


class Probe(PROTO.GLM53v6):
    """记录每帧内部状态。"""

    def __init__(self, n):
        super().__init__(n)
        self.log = []

    def process(self, ts, v):
        out = super().process(ts, v)
        ev = self.ev
        self.log.append(dict(
            ts=ts, tot=float(np.sum(v)), out_tot=float(np.sum(out)),
            state=self.state,
            tau=(ts - ev["t0"]) if ev is not None else float("nan"),
            kind=(ev["kind"] if ev is not None else ""),
            a_hat=(ev["A_hat"] if ev is not None else float("nan")),
            c=(ev["c_applied"] if ev is not None else float("nan")),
            inc_max=(ev["inc_max"] if ev is not None else float("nan")),
            tglide=(ev["Tglide"] if ev is not None else float("nan")),
            tau_g0=(ev["tau_g0"] if ev is not None else float("nan")),
            stalled=(ev["stalled"] if ev is not None else False),
            base=(ev["base"] if ev is not None else float("nan")),
            g=self.g, a_sum=float(self.A.sum()),
            min_ts=self.min_ts, lvl=self.level_ref, ts_smooth=self.ts_smooth,
            inc_ref=(ev["inc_ref"] if ev is not None else float("nan")),
            stall_t=(ev["stall_t"] if ev is not None else float("nan")),
            target=((ev["base_y"] + ev["A_hat"]) if ev is not None else float("nan")),
        ))
        return out

    def ram_ratio(self, ts, tmp, ref, inc_s, eps):
        """复算停滞判据的两个比值（只读，不写状态）。"""
        ev = self.ev
        tau = ts - ev["t0"]
        amp = max(abs(ev["A_hat"]), eps)
        gr = float(PROTO.g_shape(self.TAU_REF))
        ratio_mod = (float(PROTO.g_shape(tau)) - gr) / gr
        ratio_obs = (inc_s - ref) / ref if ref else float("nan")
        return tau, amp, ratio_mod, ratio_obs, (ref > self.STALL_MIN_FRAC * amp)


def main():
    d = L.load_dataset(L.DS_ZERO)
    pre, main = d["pre"], d["main"]
    el = pre["el"]
    V = pre["V"]
    n = len(el)
    c = Probe(L.NCH)
    out = np.empty((n, L.NCH))
    for i in range(n):
        out[i] = c.process(float(el[i]), V[i])
    osum = out.sum(1)
    psum = V.sum(1)
    msum = main["V"].sum(1)

    print(f"回放完成: n={n}")
    print(f"原型 out 总量  : min={osum.min():.0f} max={osum.max():.0f}")
    print(f"现场 main 总量 : min={msum.min():.0f} max={msum.max():.0f}")
    print(f"逐帧 RMS(原型−现场) = {np.sqrt(np.mean((osum-msum)**2)):.1f} ADC ; "
          f"中位绝对差 = {np.median(np.abs(osum-msum)):.1f} ADC")
    print(f"事件序列(原型): {[(round(t,2),k) for t,k in c.kind_log]}")
    print(f"revoke 次数={c.n_revoke}  C5 次数={c.n_c5}  末态={c.state}  g={c.g:+.4f}")

    print("\n── 41.0~47.0 s 逐 20 帧（0.2 s）──")
    print(f"{'t':>7}{'pre':>8}{'现场main':>9}{'原型out':>9}{'off现场':>9}{'off原型':>9}"
          f"{'state':>7}{'kind':>13}{'tau':>7}{'a_hat':>9}{'c':>8}{'g':>8}{'A_sum':>9}{'st':>3}")
    for i in range(n):
        t = el[i]
        if not (41.0 <= t <= 47.0):
            continue
        if i % 20:
            continue
        r = c.log[i]
        print(f"{t:7.2f}{psum[i]:8.0f}{msum[i]:9.0f}{osum[i]:9.0f}"
              f"{msum[i]-psum[i]:9.0f}{osum[i]-psum[i]:9.0f}"
              f"{r['state']:>7}{r['kind']:>13}{r['tau']:7.2f}{r['a_hat']:9.0f}"
              f"{r['c']:8.0f}{r['g']:8.4f}{r['a_sum']:9.0f}"
              f"{'T' if r['stalled'] else '.':>3}")

    # 关键转移点
    print("\n── 状态/事件转移点（41~50 s）──")
    prev = None
    for i in range(n):
        r = c.log[i]
        key = (r["state"], r["kind"])
        if key != prev:
            if 41.0 <= r["ts"] <= 50.0:
                print(f"  t={r['ts']:8.2f}  state={r['state']:6s} kind={r['kind']:13s} "
                      f"tau={r['tau']:6.2f} a_hat={r['a_hat']:9.0f} c={r['c']:8.0f} "
                      f"g={r['g']:+.4f} A_sum={r['a_sum']:8.0f} base={r['base']:8.0f} "
                      f"target={r['target']:8.0f}")
            prev = key

    # ── 停滞判据复算：42.10 s 那个 restep 事件的 tau-ratio 明细 ──
    print("\n── 停滞判据复算（42.10 s restep 事件）──")
    i0 = None
    for i in range(n):
        r = c.log[i]
        if abs(r["ts"] - 42.10) < 0.02 and r["kind"] == "restep":
            i0 = i
            break
    if i0 is not None:
        ev_t0 = c.log[i0]["ts"] - c.log[i0]["tau"]
        base = c.log[i0]["base"]
        print(f"  t0={ev_t0:.3f} base={base:.0f}")
        inc_ref = None
        print(f"{'tau':>7}{'inc':>8}{'inc_s':>8}{'a_hat':>8}{'ratio_mod':>10}"
              f"{'ratio_obs':>10}{'stall_now':>10}{'stall_t':>9}")
        j = i0
        while j < n and el[j] - ev_t0 <= 5.0:
            tau = el[j] - ev_t0
            inc = psum[j] - base
            m = (el >= el[j] - 0.10) & (el <= el[j])
            inc_s = float(np.mean(psum[m])) - base if m.any() else inc
            if inc_ref is None and tau >= 0.20:
                inc_ref = inc
            gr = float(PROTO.g_shape(0.20))
            ratio_mod = (float(PROTO.g_shape(tau)) - gr) / gr
            ratio_obs = ((inc_s - inc_ref) / inc_ref) if inc_ref else float("nan")
            a_hat = c.log[j]["a_hat"]
            amp = max(abs(a_hat), 1e-6)
            ref_ok = inc_ref is not None and inc_ref > 0.30 * amp
            stall_now = bool(ratio_mod > 0.03 and ratio_obs < 0.5 * ratio_mod) and ref_ok
            if j % 10 == 0:
                rr = c.log[j]
                amp = max(abs(rr["a_hat"]), 1e-6)
                ir = rr["inc_ref"]
                ref_ok = (ir is not None) and (ir > 0.30 * amp)
                print(f"ENG {tau:6.2f} a_hat={rr['a_hat']:7.0f} "
                      f"inc_ref={('None' if ir is None else format(ir, '.0f')):>8s} "
                      f"ref_ok={ref_ok} stall_t={rr['stall_t']:.3f} stalled={rr['stalled']} "
                      f"ratio_obs={ratio_obs:.4f} ratio_mod={ratio_mod:.4f} | "
                      f"MY ref={('None' if inc_ref is None else format(inc_ref, '.0f')):>8s} "
                      f"inc_s={inc_s:7.0f} stall_now={stall_now}")
            j += 1


if __name__ == "__main__":
    main()
