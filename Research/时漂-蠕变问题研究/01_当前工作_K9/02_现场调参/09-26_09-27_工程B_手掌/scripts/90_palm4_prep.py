# -*- coding: utf-8 -*-
"""90_palm4_prep：20260927_101043（ADC 显示、算法开、71 通道、13.7 s）读表建工作区。

产出 temp/palm4/out/streams.npz：t / pre / seg / raw（逐通道）+ 各自总值。
CSV 结构：##Session 元数据块 + ##Data 表头（时间戳,经过时间,帧序号,71 通道）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEMP = HERE.parent
sys.path.insert(0, str(Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")))

SESS = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\手掌数据\20260927_101043_single_device_43da6f")
OUT = TEMP / "palm4" / "out"
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    di = next(i for i, ln in enumerate(lines) if ln.startswith("##Data"))
    hdr = lines[di + 1].split(",")
    assert hdr[0] == "时间戳" and hdr[2] == "帧序号", hdr[:4]
    rows = [ln.split(",") for ln in lines[di + 2:] if ln.strip()]
    a = np.array([[float(x) for x in r] for r in rows], dtype=np.float64)
    return a[:, 1], a[:, 3:]  # elapsed, channels


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
    t = t - t[0]
    np.savez(OUT / "streams.npz", t=t, pre=pre, seg=seg, raw=raw,
             tot_pre=pre.sum(axis=1), tot_seg=seg.sum(axis=1), tot_raw=raw.sum(axis=1))
    tin = pre.sum(axis=1)
    print(f"帧数={len(t)} 通道数={pre.shape[1]} 时长={t[-1]:.2f}s")
    print(f"输入总值：首帧={tin[0]:.1f} 峰={tin.max():.1f}@{t[np.argmax(tin)]:.2f}s "
          f"末={tin[-1]:.1f}")
    print(f"pre/seg 逐帧总差：均={np.abs(pre.sum(1)-seg.sum(1)).mean():.2f} "
          f"最大={np.abs(pre.sum(1)-seg.sum(1)).max():.2f} ADC")
    print(f"raw−pre 首帧均值差={raw[0].mean()-pre[0].mean():.1f}（判断调零口径）")
    # 台阶检测：输入总值相邻 0.3 s 差
    k = 30
    d = np.abs(tin[k:] - tin[:-k])
    hot = np.flatnonzero(d > 400)
    if hot.size:
        groups = [[int(hot[0])]]
        for i in hot[1:]:
            if i - groups[-1][-1] <= 60:
                groups[-1].append(int(i))
            else:
                groups.append([int(i)])
        for g in groups:
            i = g[int(np.argmax(d[g]))]
            print(f"台阶 @{t[i]:.2f}s  Δ总值={tin[min(i+k, len(tin)-1)]-tin[i]:+.1f} ADC "
                  f"→ 段均电平≈{tin[min(i+50, len(tin)-1):min(i+90, len(tin)-1)].mean():.1f}")
    else:
        print("未检测到明显台阶（缓坡或小载荷）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
