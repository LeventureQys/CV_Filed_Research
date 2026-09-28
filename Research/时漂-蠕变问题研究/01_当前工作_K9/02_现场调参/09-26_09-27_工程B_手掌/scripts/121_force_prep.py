# -*- coding: utf-8 -*-
"""121_force_prep：力值会话 20260926_223756_1d925c（display=force、算法开、三条流）建工作区。

产物 temp/palm7/out/force_1d925c.npz（t/pre/seg/raw + 总值）与 force_1d925c.bin
（复算器输入：int32 n, int32 cols, 然后 n 行 (elapsed + 71 通道) float64）。
同时打印力值模式的结构性事实：逐通道弹性电平分布、illde 带宽（0.05·max(span,1)）、
逐通道爬升斜率 vs 斜率门（slope_gate_frac·max(e,1)）。
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
OUT = TEMP / "palm7" / "out"
OUT.mkdir(parents=True, exist_ok=True)
SESS = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\archived\手掌数据"
            r"\20260926_223756_single_device_1d925c")


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    di = next(i for i, ln in enumerate(lines) if ln.startswith("##Data"))
    hdr = lines[di + 1].split(",")
    assert hdr[0] == "时间戳" and hdr[2] == "帧序号", hdr[:4]
    rows = [ln.split(",") for ln in lines[di + 2:] if ln.strip()]
    a = np.array([[float(x) for x in r] for r in rows], dtype=np.float64)
    return a[:, 1], a[:, 3:]


def write_bin(path: Path, t: np.ndarray, V: np.ndarray) -> None:
    n, m = V.shape
    with path.open("wb") as f:
        f.write(struct.pack("<ii", n, m))
        buf = np.empty((n, m + 1), dtype="<f8")
        buf[:, 0] = t
        buf[:, 1:] = V
        f.write(buf.tobytes())


def steps(t: np.ndarray, tot: np.ndarray, thr: float, k: int = 30) -> list[tuple[float, int]]:
    d = np.abs(tot[k:] - tot[:-k])
    hot = np.flatnonzero(d > thr)
    if hot.size == 0:
        return []
    groups = [[int(hot[0])]]
    for i in hot[1:]:
        if i - groups[-1][-1] <= 60:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    out = []
    for g in groups:
        i = g[int(np.argmax(d[g]))]
        out.append((float(t[i]), i))
    return out


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    t, pre = load_csv(SESS / "device_001_pre_seg0.csv")
    t2, seg = load_csv(SESS / "device_001_seg000.csv")
    t3, raw = load_csv(SESS / "device_001_raw_seg000.csv")
    assert len(t) == len(t2) == len(t3), (len(t), len(t2), len(t3))
    assert np.allclose(t, t2) and np.allclose(t, t3)
    t = t - t[0]
    np.savez(OUT / "force_1d925c.npz", t=t, pre=pre, seg=seg, raw=raw,
             tot_pre=pre.sum(1), tot_seg=seg.sum(1), tot_raw=raw.sum(1))
    write_bin(OUT / "force_1d925c.bin", t, pre)
    fps = (len(t) - 1) / (t[-1] - t[0])
    tin, tout = pre.sum(1), seg.sum(1)
    print(f"帧数={len(t)} 通道={pre.shape[1]} 时长={t[-1]:.2f}s fps={fps:.2f}")
    print(f"输入总值：首={tin[0]:.3f} 峰={tin.max():.3f}@{t[np.argmax(tin)]:.2f}s 末={tin[-1]:.3f} N")
    print(f"录制显示：首={tout[0]:.3f} 峰={tout.max():.3f}@{t[np.argmax(tout)]:.2f}s 末={tout[-1]:.3f} N")
    print(f"输入首帧逐通道：min={pre[0].min():.4f} max={pre[0].max():.4f} mean={pre[0].mean():.4f} N")
    st = steps(t, tin, 1.0)
    print("\n台阶（Δ总值 > 1 N）：")
    for tt, i in st:
        print(f"  @{tt:6.2f}s Δ={tin[min(i+30, len(tin)-1)]-tin[i]:+8.3f} N")
    # 逐段弹性电平
    print("\n分段弹性电平 E（台阶后 0.1~0.6 s 最小值）与段末：")
    segs = []
    for j, (tt, i) in enumerate(st):
        a, b = tt + 0.1, tt + 0.6
        m = (t >= a) & (t <= b)
        E = float(tin[m].min())
        nxt = st[j + 1][0] if j + 1 < len(st) else t[-1]
        m2 = (t >= nxt - 0.6) & (t <= nxt)
        end = float(tin[m2].mean()) if m2.any() else float(tin[-1])
        creep = end - E
        segs.append((tt, nxt, E, end, creep))
        print(f"  段{j+1} {tt:6.2f}→{nxt:6.2f}s  E={E:8.3f}N  段末输入={end:8.3f}N  "
              f"蠕变={creep:+7.3f}N ({100*creep/max(E-tin[max(0,i-30)], 1e-9):+.1f}% 台阶)")
    print(f"\n逐通道弹性电平分布（第一段 E_ch）：")
    tt0, i0 = st[0]
    m = (t >= tt0 + 0.1) & (t <= tt0 + 0.6)
    Ech = pre[m].min(axis=0)
    for q in (50, 75, 90, 95, 99, 100):
        print(f"  p{q:<3d} = {np.percentile(Ech, q)*1000:8.1f} ADC  ({np.percentile(Ech, q):.4f} N)")
    print(f"  非零通道数(E_ch>0.005N)={int((Ech>0.005).sum())}  "
          f"E_ch>0.05N 的通道={int((Ech>0.05).sum())}  "
          f"E_ch>0.1N 的通道={int((Ech>0.1).sum())}")
    print(f"  ΣE_ch={Ech.sum():.3f} N")
    # 斜率：段内逐帧总值斜率与逐通道典型斜率
    print("\n加载后逐 0.5 s 的逐通道最大 |slope|（slope=(v−v_lp)/1s），与门 0.05 N/s 比：")
    for dt in np.arange(0.5, 12.0, 1.0):
        i = int(np.searchsorted(t, tt0 + dt))
        if i < 2 or i >= len(t):
            break
        sl = (pre[i] - pre[i - 1]) / max(t[i] - t[i - 1], 1e-9)
        tot_sl = (tin[i] - tin[i - 1]) / max(t[i] - t[i - 1], 1e-9)
        print(f"  t0+{dt:4.1f}s  总斜率={tot_sl:+7.3f} N/s  逐通道 max|slope|={np.abs(sl).max():8.4f} "
              f"p95={np.percentile(np.abs(sl), 95):8.4f} N/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
