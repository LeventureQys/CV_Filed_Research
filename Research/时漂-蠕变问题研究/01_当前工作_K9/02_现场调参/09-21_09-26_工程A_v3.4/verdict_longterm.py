# -*- coding: utf-8 -*-
"""仅用「长期数据」4 段，判定调参前后孰优（现役 vs 最终5参）。

严格口径（只用这 4 段，不引用其他录制）：
  * 每段各自的加载沿 / 段末（卸载点）自动判定；
  * 稳态平台 = 该段受载帧后 40% 的显示均值；
  * 指标：
      过收敛(显示最低)   受载段显示最低点（越高=越不过扣）
      x2 起效            距加载沿的秒数
      Δ1/3/5/15s         各检查点「距平台」偏差（>0 此刻偏低→后续上漂）
      持载钉平           受载段后半程显示 std 与线性斜率
      卸载/低位窗震荡     低位窗（输入<30%台阶）显示 std 与最低
      补偿跟随           稳态后显示的段内变化 vs 输入变化

用法：python verdict_longterm.py
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_v34_on_csv import read_session_csv  # noqa: E402
from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402

P0 = Params()
P5 = replace(P0, r_fast=0.06, slow_confirm_s=4.0, soft_unfreeze_s=2.0,
             tau_r_slow_idle_s=2.0, tau_r_fast_s=0.5)
CKPTS = (1.0, 3.0, 5.0, 15.0)
SESSIONS = sorted((HERE / "长期数据").glob("*/device_001_seg000.csv"))


def run(p, t, V):
    c = CreepObserverK9(p)
    c._trace_frame = lambda *a, **k: None
    n = len(t)
    out = np.empty(n); x1 = np.empty(n); x2 = np.empty(n)
    for i in range(n):
        out[i] = c.process(float(t[i]), V[i]).sum()
        x1[i] = c.x_fast.sum(); x2[i] = c.x_slow.sum()
    return out, x1, x2


def at(t, a, tq):
    return float(a[min(int(np.searchsorted(t, tq)), len(a) - 1)])


def seg_of(tin, t):
    step = tin.max() - np.percentile(tin, 5)
    base5 = float(np.percentile(tin, 5))
    i_on = int(np.argmax(tin > 0.2 * step + base5))
    i_pk = next((i for i in range(i_on, len(tin)) if tin[i] > 0.8 * step + base5), i_on)
    i_un = next((i for i in range(i_pk + 1, len(tin)) if tin[i] < 0.3 * step + base5),
                len(tin) - 1)
    return step, base5, i_on, i_un


def metrics(t, tin, out, x2, step, base5, i_on, i_un):
    t_on, t_end = t[i_on], t[i_un]
    low = tin < 0.3 * step + base5
    m_plat = (t >= t_on + 0.6 * (t_end - t_on)) & (t <= t_end) & (tin > 0.5 * step + base5)
    plat = float(np.mean(out[m_plat])) if m_plat.sum() > 20 else np.nan
    above = x2 > 0.005 * step
    tx2 = float(t[int(np.argmax(above))] - t_on) if above.any() else float("nan")
    seg = (t >= t_on) & (t <= t_end)
    tail = (t >= t_on + 0.5 * (t_end - t_on)) & (t <= t_end)
    d = {}
    d["disp_min"] = float(out[seg].min())
    d["tail_std"] = float(np.std(out[tail])) if tail.sum() > 20 else np.nan
    d["tail_slope"] = float(np.polyfit(t[tail], out[tail], 1)[0]) if tail.sum() > 20 else np.nan
    d["low_std"] = float(np.std(out[low])) if low.any() else np.nan
    d["low_min"] = float(out[low].min()) if low.any() else np.nan
    d["x2_start"] = tx2
    d["x2_end"] = float(x2[-1])
    d["plat"] = plat
    d["ck"] = {}
    for c_ in CKPTS:
        tq = min(t_on + c_, t_end)
        d["ck"][c_] = float(plat - at(t, out, tq)) if np.isfinite(plat) else np.nan
    return d


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass

    res = []
    for csv in SESSIONS:
        d = read_session_csv(csv)
        t, V = d["t"], d["values"]
        tin = V.sum(axis=1)
        step, base5, i_on, i_un = seg_of(tin, t)
        o0, _, x20 = run(P0, t, V)
        o5, _, x25 = run(P5, t, V)
        m0 = metrics(t, tin, o0, x20, step, base5, i_on, i_un)
        m5 = metrics(t, tin, o5, x25, step, base5, i_on, i_un)
        res.append((csv.parent.name, t, tin, o0, o5, step, t[i_on], t[i_un], m0, m5))

    print("加载沿/段末：")
    for name, t, tin, o0, o5, step, t_on, t_end, m0, m5 in res:
        print(f"  {name}: t_on={t_on:.1f}s 段末={t_end:.1f}s 台阶={step:.0f} 时长={t[-1]-t[0]:.0f}s")

    print(f"\n=== A. 加载后检查点「距平台」偏差（>0 此刻偏低→后续上漂；<0 此刻偏高→后续下漂）===")
    print(f"{'会话':24s} {'方案':8s} " + " ".join(f"Δ{c_:>4.0f}s" for c_ in CKPTS) + "   x2起效")
    for name, t, tin, o0, o5, step, t_on, t_end, m0, m5 in res:
        for tag, m in (("现役", m0), ("最终5参", m5)):
            ck = " ".join(f"{m['ck'][c_]:+6.0f}" for c_ in CKPTS)
            xs = f"{m['x2_start']:6.1f}s" if np.isfinite(m["x2_start"]) else "   未起效"
            print(f"{name[:24]:24s} {tag:8s} {ck}  {xs}")
        print()

    print("=== B. 过收敛 / 钉平 / 卸载震荡 ===")
    print(f"{'会话':24s} {'方案':8s} {'显示最低':>9s} {'尾部std':>8s} {'尾部斜率':>10s} "
          f"{'低位std':>8s} {'低位min':>8s} {'x2末值':>7s}")
    for name, t, tin, o0, o5, step, t_on, t_end, m0, m5 in res:
        for tag, m in (("现役", m0), ("最终5参", m5)):
            print(f"{name[:24]:24s} {tag:8s} {m['disp_min']:9.0f} {m['tail_std']:8.0f} "
                  f"{m['tail_slope']:+9.1f}/s {m['low_std']:8.0f} {m['low_min']:8.0f} "
                  f"{m['x2_end']:7.0f}")
        print()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(res)
    fig, axes = plt.subplots(n, 2, figsize=(16, 3.2 * n),
                             gridspec_kw={"width_ratios": [2.0, 1.3]})
    for r, (name, t, tin, o0, o5, step, t_on, t_end, m0, m5) in enumerate(res):
        a1, a2 = axes[r]
        m = (t >= max(0, t_on - 3)) & (t <= t_end + 3)
        a1.plot(t[m], tin[m], color="#7f8c8d", lw=1.0, label="输入")
        a1.plot(t[m], o0[m], color="#c0392b", lw=1.2, label="现役")
        a1.plot(t[m], o5[m], color="#27ae60", lw=1.4, label="最终5参")
        if np.isfinite(m5["plat"]):
            a1.axhline(m5["plat"], color="#2c3e50", lw=0.9, ls="-.", alpha=0.7)
        for c_ in CKPTS:
            a1.axvline(min(t_on + c_, t_end), color="#2c3e50", lw=0.7, ls=":", alpha=0.5)
        a1.set_title(f"{name}", loc="left", fontsize=9.5)
        a1.set_ylabel("总量 (ADC)")
        a1.legend(loc="lower right", fontsize=8)
        a1.grid(alpha=0.25)

        # 右：距平台偏差曲线（现役 vs 最终5参）
        for tag, o, col in (("现役", o0, "#c0392b"), ("最终5参", o5, "#27ae60")):
            plat = m0["plat"] if tag == "现役" else m5["plat"]
            if np.isfinite(plat):
                a2.plot(t[m], plat - o[m], color=col, lw=1.2, label=f"{tag} 距平台")
        a2.axhline(0, color="k", lw=0.8)
        a2.set_title("距平台偏差（>0 偏低→后续上漂 / <0 偏高→后续下漂）", loc="left", fontsize=8.5)
        a2.set_ylabel("ADC")
        a2.legend(loc="upper right", fontsize=8)
        a2.grid(alpha=0.25)
        if r == n - 1:
            a2.set_xlabel("时间 (s)")

    fig.suptitle("仅长期数据 4 段：现役 vs 最终5参", y=0.998, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    png = HERE / "out" / "长期数据_判定_现役vs最终5参.png"
    fig.savefig(png, dpi=120)
    print(f"图已保存：{png}")


if __name__ == "__main__":
    main()
