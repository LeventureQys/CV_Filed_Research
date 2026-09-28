# -*- coding: utf-8 -*-
"""只读诊断（续）：把 69.7~71.7 s 那段振荡放到**去重后的真实时间轴**上看。

背景：该录制的 `elapsed` 在同一时刻压了 2~4 帧（组内跨约 40 ms），
直接对 indices 做 FFT 会把"帧序"当成"时间"，得到假的 0.5 Hz。
这里改用 `timestamp`（微秒级、单调）重建统一时间网格后再做谱分析。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

DS = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况",
                  "20260919_134056_single_device_f40a1b")


def build_uniform(ts, V, t0, t1, fs=200.0):
    """按 timestamp 线性插值到 fs 的统一网格。"""
    m = (ts >= t0) & (ts <= t1)
    t = ts[m]
    X = V[m]
    o = np.argsort(t)
    t, X = t[o], X[o]
    # 去掉重复时间戳（保留均值）
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, X = t[keep], X[keep]
    g = np.arange(t[0], t[-1], 1.0 / fs)
    Y = np.empty((len(g), X.shape[1]))
    for k in range(X.shape[1]):
        Y[:, k] = np.interp(g, t, X[:, k])
    return g - g[0], Y


def main():
    pre = L.load_stream(DS, "device_001_pre_seg0.csv")
    ts = pre["ts"]
    V = pre["V"]
    print("timestamp 间隔统计（全录）：")
    d = np.diff(ts)
    print("  中位=%.3f ms  p95=%.2f ms  max=%.2f ms  重复(<=0)帧数=%d"
          % (np.median(d) * 1e3, np.percentile(d, 95) * 1e3, d.max() * 1e3,
             int((d <= 0).sum())))
    # 每帧的"真实时间步"
    print("  ⇒ 帧率≈%.1f Hz（按唯一时间戳计）" % (1.0 / np.median(d[np.argsort(d)[len(d) // 4:]])))

    for tag, a, b in (("振荡段 69.7~71.7 s", 69.7, 71.7),
                      ("安静段 108~118 s", 108.0, 110.0)):
        t0 = ts[np.searchsorted(pre["el"], a)]
        t1 = ts[np.searchsorted(pre["el"], b)]
        g, Y = build_uniform(ts, V, t0, t1, fs=200.0)
        # 去线性趋势
        A = np.vstack([g, np.ones_like(g)]).T
        Z = Y.copy()
        for k in range(Z.shape[1]):
            c, *_ = np.linalg.lstsq(A, Z[:, k], rcond=None)
            Z[:, k] -= A @ c
        Z -= Z.mean(0)
        tot = Z.sum(1)
        win = np.hanning(len(tot))
        F = np.fft.rfft(tot * win)
        fr = np.fft.rfftfreq(len(tot), g[1] - g[0])
        mag = np.abs(F)
        mag[0] = 0
        pk = int(mag.argmax())
        print(f"\n== {tag} ==  统一网格 %d 点 @%.0f Hz（来自 %d 帧）"
              % (len(g), 1.0 / (g[1] - g[0]), len(Y)))
        print("   总量去趋势 std=%.0f  主频=%.2f Hz（周期 %.3f s）  前 3 峰："
              % (tot.std(), fr[pk], 1.0 / fr[pk]))
        order = np.argsort(-mag)[:4]
        for k in order:
            if fr[k] > 0:
                print("      %.2f Hz  幅度 %.0f" % (fr[k], mag[k]))
        # 该主频处逐通道相位
        Fc = np.fft.rfft(Z * win[:, None], axis=0)
        ph_t = np.angle(Fc[pk].sum())
        rows = []
        for k in range(Z.shape[1]):
            amp = abs(Fc[pk, k])
            dph = ((np.degrees(np.angle(Fc[pk, k]) - ph_t) + 180) % 360) - 180
            rows.append((k, amp, dph))
        print("   逐通道在主频处的幅度/相位（幅度≥1000 才列）：")
        for k, amp, dph in sorted(rows, key=lambda r: -r[1]):
            if amp < 1000:
                continue
            print("      ch%-2d 幅度=%7.0f  Δ相位=%+7.1f°" % (k, amp, dph))
        pos = [k for k, a_, d_ in rows if a_ >= 1000 and abs(d_) < 90]
        neg = [k for k, a_, d_ in rows if a_ >= 1000 and abs(d_) >= 90]
        print("   同相簇 ch%s（共 %d 个）；反相簇 ch%s（共 %d 个）"
              % (pos, len(pos), neg, len(neg)))


if __name__ == "__main__":
    main()
