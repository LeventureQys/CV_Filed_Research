# -*- coding: utf-8 -*-
"""探针 D：用**录制口径**（pre = 算法输入, main = 算法结果）做逐通道归因。

定义（全部 ADC，21 通道）：
  ref_i      = 加载稳定后的「无蠕变电平」参考 := pre_i 在 [20,40] s 的均值
  creep_i(t) = pre_i(t) − ref_i            ← 传感器蠕变/时漂（应当被扣掉）
  ded_i(t)   = pre_i(t) − main_i(t)        ← 算法实际扣掉的量（录制口径，权威）
  resid_i(t) = creep_i(t) − ded_i(t) = main_i(t) − ref_i  ← 显示上残留的漂移

输出：results/v30_channel_attrib.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

TAG = "20260919_141824_single_device_110871"
DS = os.path.join(L.OVERVIEW, TAG)
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
OUT = os.path.join(RES, "v30_channel_attrib.txt")


def bucket(el, dt):
    out, i, n = [], 0, len(el)
    while i < n:
        j = i
        while j < n and el[j] < el[i] + dt:
            j += 1
        out.append((i, j))
        i = j
    return out


def main():
    ds = L.load_dataset(DS)
    el = ds["pre"]["el"]
    pre = ds["pre"]["V"]
    main = ds["main"]["V"]
    n = pre.shape[1]
    i20, i40 = int(20 * 100), int(40 * 100)
    ref = pre[i20:i40].mean(0)
    idle = pre[:430].mean(0)
    amp = ref - idle
    loaded = amp > 0.10 * amp.max()
    creep = pre - ref
    ded = pre - main
    resid = main - ref

    lines = []
    p = lines.append
    p("数据集 %s  (pre = 算法输入, main = 算法结果；ref = pre 在 [20,40] s 的均值)" % TAG)
    p("受载通道（amp > 10%% 峰，%d 个）: %s" %
      (int(loaded.sum()), ", ".join("ch%d" % k for k in np.nonzero(loaded)[0])))
    p("")
    p("=== ① 逐通道：蠕变 vs 实际扣除 vs 残留（末 5 s 均值）===")
    p("%-5s %9s %9s %9s %10s %10s %10s %8s" %
      ("ch", "idle", "ref(无蠕变)", "creep末", "ded末", "resid末", "扣除率", "受载?"))
    for k in range(n):
        if abs(amp[k]) < 1 and abs(creep[-500:, k].mean()) < 5:
            continue
        c = creep[-500:, k].mean()
        d = ded[-500:, k].mean()
        r = resid[-500:, k].mean()
        rate = (d / c) if abs(c) > 1e-9 else float("nan")
        p("%-5s %9.0f %9.0f %9.0f %10.0f %10.0f %9.0f%% %8s" %
          ("ch%d" % k, idle[k], ref[k], c, d, r, 100 * rate, "Y" if loaded[k] else "-"))
    p("")
    p("=== ② 总量口径 ===")
    tc = creep.sum(1)
    td = ded.sum(1)
    tr = resid.sum(1)
    for lbl, i0, i1 in (("首 5 s(4.3~9.3)", 430, 930), ("末 5 s", len(el) - 500, len(el))):
        p("  %-16s creep %+8.0f  ded %+8.0f  resid %+8.0f" %
          (lbl, tc[i0:i1].mean(), td[i0:i1].mean(), tr[i0:i1].mean()))
    p("")
    p("=== ③ 20 s 桶：扣除是否随时间变化（录制口径）===")
    p("%7s %9s %9s %9s %9s %8s %8s %8s" %
      ("t", "creep", "ded", "resid", "ded变化", "ded/creep", "受载ded", "未载ded"))
    prev = None
    for (i, j) in bucket(el, 20.0):
        if el[i] < 4:
            continue
        c = tc[i:j].mean()
        d = td[i:j].mean()
        r = tr[i:j].mean()
        ch = "" if prev is None else "%+.0f" % (d - prev)
        ld = np.nonzero(loaded)[0]
        ul = np.nonzero(~loaded)[0]
        p("%7.1f %9.0f %9.0f %9.0f %9s %7.1f%% %8.0f %8.0f" %
          (el[i], c, d, r, ch, 100 * d / c if abs(c) > 1 else 0,
           ded[i:j][:, ld].sum(1).mean(), ded[i:j][:, ul].sum(1).mean()))
        prev = d
    p("")
    p("=== ④ 蠕变的空间集中度（末 5 s）===")
    c_end = creep[-500:].mean(0)
    w = np.abs(c_end)
    o = np.argsort(-w)
    tot = w.sum()
    acc = 0.0
    for rank, k in enumerate(o[:10], 1):
        acc += w[k]
        p("  第%2d  ch%-3d creep %+8.0f  占蠕变总量 %5.1f%%  累计 %5.1f%%" %
          (rank, k, c_end[k], 100 * w[k] / tot, 100 * acc / tot))
    p("")
    p("=== ⑤ 扣除率 vs 通道（末 5 s）===")
    for k in np.nonzero(loaded)[0]:
        c = creep[-500:, k].mean()
        d = ded[-500:, k].mean()
        p("  ch%-3d creep %+8.0f  ded %+8.0f  扣除率 %6.1f%%" %
          (k, c, d, 100 * d / c if abs(c) > 1 else float("nan")))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
