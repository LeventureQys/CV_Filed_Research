# -*- coding: utf-8 -*-
"""v2.0 PROBE-G：验证「停滞判据 inc_ref 迟延锁存」修复的因果性与收益（离线 A/B）。

假设（来自 probe_f 的逐帧证据）：
  现役实现的 inc_ref 要到 τ≈0.65 s 才取到（因为 hist 在 τ=0.20 s 时只有 4 帧，
  而取参考需要 hist 里能定位到 τ_ref 的相邻样本对——C++ 侧是 `hist.size() >= 4` 的显式门），
  取到的是"已在爬升"的值 ⇒ ratio_obs 被抬高 ⇒ 停止判据永不成立 ⇒ 显示在高位驻留。

本探针：把事件期内的 inc_ref 取样改为「在 τ = kTauRef 处对 hist 线性插值」
（与原型 glm53_v6.py:471-474 的 np.interp 语义一致，但去掉 hist.size() 的迟延门），
对比 A/B 两臂在本录制与既有 13 份录制上的表现。

口径：
  · 显示偏移 off(t) = out_tot(t) − pre_tot(t)（本探测器的 pre 就是算法输入）
  · 驻留时长 = 事件内 off 超过「事件后 5 s 稳态 off」+ 2%·台阶 的连续帧时长
  · 台阶捕获比 G = (事件后 4~5 s 显示中位 − 事件前显示中位) / 台阶幅度
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
g_shape = PROTO.g_shape


class Fixed(PROTO.GLM53v6):
    """把 inc_ref 的取样改成「在 τ=kTauRef 处线性插值」，其余逐行沿用原型。"""

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        if ev is None:
            return super()._run_event(ts, v, total, dt, eps, idle_now)
        tau = ts - ev["t0"]
        # 在进入本帧前，若 τ 已跨过 TAU_REF 且 hist 覆盖到 TAU_REF，就直接锁存插值参考
        if ev["inc_ref"] is None and tau >= self.TAU_REF and len(ev["hist"]) >= 2:
            tt = np.array([h[0] for h in ev["hist"]])
            yy = np.array([h[1] for h in ev["hist"]])
            if tt.min() <= self.TAU_REF <= tt.max():
                ev["inc_ref"] = float(np.interp(self.TAU_REF, tt, yy))
        return super()._run_event(ts, v, total, dt, eps, idle_now)


def replay(cls, el, V):
    n = len(el)
    c = cls(L.NCH)
    out = np.empty((n, L.NCH))
    for i in range(n):
        out[i] = c.process(float(el[i]), V[i])
    return c, out.sum(1)


def metrics(el, pre_tot, out_tot, t_ups):
    """逐加载沿：驻留时长 / 过充峰值 / 台阶捕获比。"""
    rows = []
    for t_up in t_ups:
        m = (el >= t_up - 0.4) & (el <= min(t_up + 30.0, el[-1]))
        if m.sum() < 100:
            continue
        base_out = float(np.median(out_tot[(el >= t_up - 1.2) & (el < t_up - 0.35)]))
        base_pre = float(np.median(pre_tot[(el >= t_up - 1.2) & (el < t_up - 0.35)]))
        idx = np.where(m)[0]
        j0 = idx[0]
        # 台阶幅度：用事件后 25~30 s（或末段）的 pre 平台
        t_end = min(t_up + 30.0, el[-1] - 0.1)
        pre_ss = float(np.median(pre_tot[(el >= t_end - 1.5) & (el <= t_end)]))
        out_ss = float(np.median(out_tot[(el >= t_end - 1.5) & (el <= t_end)]))
        A = pre_ss - base_pre
        if abs(A) < 200:
            continue
        off = out_tot[idx] - pre_tot[idx]
        off_ss = out_ss - pre_ss
        # 驻留：off 超过 off_ss + 2%·A 的最长连续时长
        thr = off_ss + 0.02 * abs(A)
        over = off > thr
        run = best = 0
        tt = el[idx]
        for k in range(len(over)):
            if over[k]:
                run = (run + dt_of(tt, k)) if k else 0.0
                best = max(best, run)
            else:
                run = 0.0
        os_max = float(np.max(off - off_ss)) if len(off) else 0.0
        # 到带时间：显示进入 ±5%·A（围绕 pre 最终电平）
        t5 = float("nan")
        for k in range(len(idx)):
            if abs(out_tot[idx[k]] - pre_ss) <= 0.05 * abs(A):
                t5 = el[idx[k]] - t_up
                break
        rows.append(dict(t_up=t_up, A=A, off_ss=off_ss, hold=best,
                         os_max_frac=os_max / abs(A), t5=t5,
                         G=(float(np.median(out_tot[(el >= t_up + 4.0) & (el <= t_up + 5.0)]))
                            - base_out) / A))
    return rows


def dt_of(tt, k):
    if k == 0:
        return 0.0
    d = tt[k] - tt[k - 1]
    return d if d > 0 else 0.0


def load_legacy():
    base = os.path.join(L.ROOT, "temp", "原始数据only")
    cases = []
    for grp, sub in (("四指指尖", None), ("右拇指指尖", None), ("左拇指指尖", None)):
        for d in ("数据1", "数据2", "数据3"):
            p = os.path.join(base, grp, d, "device_001_seg000.csv")
            if os.path.exists(p):
                cases.append((f"{grp}/{d}", p))
    for d in os.listdir(os.path.join(base, "变化负载")):
        root = os.path.join(base, "变化负载", d)
        for dp, _dn, fn in os.walk(root):
            if "device_001_seg000.csv" in fn:
                cases.append((f"变化负载/{d}", os.path.join(dp, "device_001_seg000.csv")))
    return cases


def read_simple(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    hdr = rows[di + 1].split(",")
    idx = [i for i, h in enumerate(hdr) if h.startswith("ch")]
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


def main():
    # ── 本录制 ──
    d = L.load_dataset(L.DS_ZERO)
    el = d["pre"]["el"]
    V = d["pre"]["V"]
    psum = V.sum(1)
    k = np.ones(9) / 9.0
    lo, hi = np.percentile(psum, 3), np.percentile(psum, 97)
    ups, _ = L.edges_from_tot(np.convolve(psum, k, mode="same"), lo + 0.5 * (hi - lo))
    t_ups = []
    for i in ups:
        if not t_ups or el[i] - t_ups[-1] > 1.0:
            t_ups.append(float(el[i]))

    print("=== 数据集: 新录制（从零基线） ===")
    print(f"加载沿: {[round(t,2) for t in t_ups]}")
    for tag, cls in (("现役(inc_ref 迟延锁存)", PROTO.GLM53v6), ("修复(插值取参考)", Fixed)):
        c, osum = replay(cls, el, V)
        rows = metrics(el, psum, osum, t_ups)
        hold = [r["hold"] for r in rows if r["hold"] > 0]
        print(f"\n-- {tag} --  事件数={len(rows)}  末态={c.state}  g={c.g:+.4f}")
        print(f"{'t_up':>8}{'A':>8}{'off_ss':>8}{'驻留(s)':>9}{'过充峰值/A':>11}{'T5%(s)':>8}{'G':>7}")
        for r in rows:
            print(f"{r['t_up']:8.2f}{r['A']:8.0f}{r['off_ss']:8.0f}{r['hold']:9.2f}"
                  f"{100*r['os_max_frac']:10.2f}%{r['t5']:8.2f}{r['G']:7.3f}")
        if hold:
            print(f"   驻留中位={np.median(hold):.2f}s 最大={max(hold):.2f}s  "
                  f"过充峰值中位={100*np.median([r['os_max_frac'] for r in rows]):.2f}%  "
                  f"G 中位={np.median([r['G'] for r in rows]):.3f}")

    # ── 既有 13 份 ──
    print("\n=== 既有录制回归（力矩域/ADC 域混合，仅看是否劣化） ===")
    print(f"{'数据集':28s}{'事件':>5}{'驻留中位':>10}{'驻留最大':>10}"
          f"{'过充中位':>10}{'G中位':>8}   ← 现役 / 修复")
    for name, path in load_legacy():
        try:
            el2, V2 = read_simple(path)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:28s}  读取失败: {exc}")
            continue
        if len(el2) < 500:
            continue
        p2 = V2.sum(1)
        lo2, hi2 = np.percentile(p2, 3), np.percentile(p2, 97)
        if hi2 - lo2 < 0.5:
            print(f"{name:28s}  幅度过小，跳过（{hi2-lo2:.3f}）")
            continue
        ups2, _ = L.edges_from_tot(np.convolve(p2, k, mode="same"), lo2 + 0.5 * (hi2 - lo2))
        tu = []
        for i in ups2:
            if not tu or el2[i] - tu[-1] > 1.0:
                tu.append(float(el2[i]))
        if not tu:
            print(f"{name:28s}  无加载沿，跳过")
            continue
        line = f"{name:28s}{len(tu):5d}"
        for cls in (PROTO.GLM53v6, Fixed):
            _c, o2 = replay(cls, el2, V2)
            rows = metrics(el2, p2, o2, tu)
            if not rows:
                line += "        -        -        -       -"
                continue
            h = [r["hold"] for r in rows]
            line += (f"{np.median(h):10.2f}{max(h):10.2f}"
                     f"{100*np.median([r['os_max_frac'] for r in rows]):9.2f}%"
                     f"{np.median([r['G'] for r in rows]):8.3f}")
        print(line + "   ← 现役 / 修复")


if __name__ == "__main__":
    main()
