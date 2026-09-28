# -*- coding: utf-8 -*-
"""150_b2d896_prep：会话 20260927_161747_b2d896（ADC 显示、算法关、单流）建工作区。

产出 temp/palm10/out/b2d896.npz（t / V / tot）与概要：
台阶时刻、E、快相时长与幅度、慢相速率、尾端速率、逐通道电平分布。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
OUT = TEMP / "palm10" / "out"
OUT.mkdir(parents=True, exist_ok=True)
SESS = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\手掌数据"
            r"\20260927_161747_single_device_b2d896")


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    di = next(i for i, ln in enumerate(lines) if ln.startswith("##Data"))
    hdr = lines[di + 1].split(",")
    assert hdr[0] == "时间戳" and hdr[2] == "帧序号", hdr[:4]
    rows = [ln.split(",") for ln in lines[di + 2:] if ln.strip()]
    a = np.array([[float(x) for x in r] for r in rows], dtype=np.float64)
    return a[:, 1], a[:, 3:]


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t, V = load_csv(SESS / "device_001_seg000.csv")
    t = t - t[0]
    tot = V.sum(1)
    np.savez(OUT / "b2d896.npz", t=t, V=V, tot=tot)
    fps = (len(t) - 1) / (t[-1] - t[0])
    print(f"帧数={len(t)} 通道={V.shape[1]} 时长={t[-1]:.1f}s fps={fps:.2f}")
    print(f"首帧总值={tot[0]:.0f} 末值={tot[-1]:.0f} 峰={tot.max():.0f}@{t[np.argmax(tot)]:.1f}s")
    # 台阶
    k = 30
    d = np.abs(tot[k:] - tot[:-k])
    hot = np.flatnonzero(d > 400)
    groups = [[int(hot[0])]]
    for i in hot[1:]:
        if i - groups[-1][-1] <= 60:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    for g in groups:
        i = g[int(np.argmax(d[g]))]
        print(f"  台阶 @{t[i]:.2f}s  Δ(0.3s)={tot[min(i+k, len(t)-1)]-tot[i]:+.0f}")
    i0 = groups[0][int(np.argmax(d[groups[0]]))]
    t0 = float(t[i0])
    m = (t >= t0 + 0.1) & (t <= t0 + 0.6)
    E = float(tot[m].min())
    print(f"\nE(台阶后0.1~0.6s最小)={E:.0f} ADC；台阶前电平={tot[max(0,i0-30):i0].mean():.0f} ADC")
    # 快相/慢相/尾端分段速率
    print(f"\n{'区间':>18s} {'输入均值':>9s} {'区间斜率 ADC/s':>14s} {'每秒占总E':>10s}")
    segs = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, 40.0),
            (40.0, 60.0), (60.0, 90.0), (90.0, 120.0), (120.0, 150.0)]
    last = t[-1]
    segs += [(last - 60, last - 30), (last - 30, last - 10), (last - 10, last)]
    for a, b in segs:
        ma = (t >= a) & (t <= b)
        if ma.sum() < 20:
            continue
        sl = np.polyfit(t[ma], tot[ma], 1)[0]
        print(f"{a:7.1f}~{b:6.1f}s {tot[ma].mean():9.0f} {sl:14.2f} "
              f"{sl/E*100:9.4f}%/s")
    # 尾端漂移绝对量
    for a in (10, 30, 60, 120):
        i = int(np.searchsorted(t, a))
        print(f"  {a}s→末：输入涨 {tot[-1]-tot[i]:+.0f} ADC "
              f"(+{(tot[-1]-tot[i])/E*100:.1f}% E)")
    # 逐通道电平
    Ech = V[m].min(axis=0)
    nz = Ech[Ech > 20]
    print(f"\n逐通道 E_ch：非零通道={nz.size} Σ={Ech.sum():.0f} "
          f"中位={np.median(nz):.0f} p90={np.percentile(nz,90):.0f} max={nz.max():.0f} ADC")
    print(f"  每通道蠕变总量（末−E）= {(V[-1]-Ech).mean():.1f} ADC 均值、"
          f"{(V[-1]-Ech).max():.1f} 最大")
    print(f"  尾端 10 s 每通道速率 p95 = "
          f"{np.percentile((V[-1]-V[int(np.searchsorted(t, last-10))])/10.0, 95):.3f} ADC/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
