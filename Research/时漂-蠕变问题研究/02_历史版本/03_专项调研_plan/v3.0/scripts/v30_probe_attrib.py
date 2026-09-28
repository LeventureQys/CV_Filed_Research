# -*- coding: utf-8 -*-
"""探针 C：稳定工况残漂的**通路归因**。

口径（全部 ADC，总量 = 21 通道求和）：
  offset(t) = 显示 − 算法输入        = 补偿量（负 = 显示被压低）
  creep(t)  = Σ_loaded (v_i − A_i)    = 慢相「应当扣除」的蠕变量（A 是无蠕变电平）
  resid(t)  = creep(t) − ded(t)       = 慢相没扣掉的残余漂移 = 显示相对 A 的漂移
  med_gap(t)= 需要的相对扣除 (A 加权均值) − 实际用的相对扣除 (median·γ)

输出：results/v30_attrib.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

TAG = "20260919_141824_single_device_110871"
DS = os.path.join(L.OVERVIEW, TAG)
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
OUT = os.path.join(RES, "v30_attrib.txt")


def bucket(el, dt=1.0):
    """按 dt 秒分桶的起止索引。"""
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
    X = np.load(os.path.join(RES, "v30_trace_%s.npy" % TAG))
    cols = open(os.path.join(RES, "v30_trace_%s.cols" % TAG), encoding="utf-8").read().split()
    D = {c: X[:, i] for i, c in enumerate(cols)}
    pre = ds["pre"]["V"]
    main = ds["main"]["V"]
    pre_tot, main_tot = pre.sum(1), main.sum(1)

    lines = []
    p = lines.append
    p("数据集 %s  帧 %d  时长 %.1f s" % (TAG, len(el), el[-1] - el[0]))
    p("")
    p("=== ① 总体（录制口径，与复算无关）===")
    p("  pre  总量: 首(4.3~9.3s 均值) %.0f  末(最后 5 s 均值) %.0f  漂移 %+.0f (%.2f%%)" %
      (pre_tot[int(4.3 * 100):int(9.3 * 100)].mean(), pre_tot[-500:].mean(),
       pre_tot[-500:].mean() - pre_tot[int(4.3 * 100):int(9.3 * 100)].mean(),
       100 * (pre_tot[-500:].mean() - pre_tot[int(4.3 * 100):int(9.3 * 100)].mean())
       / pre_tot[int(4.3 * 100):int(9.3 * 100)].mean()))
    p("  algo 总量: 首 %.0f  末 %.0f  漂移 %+.0f" %
      (main_tot[int(4.3 * 100):int(9.3 * 100)].mean(), main_tot[-500:].mean(),
       main_tot[-500:].mean() - main_tot[int(4.3 * 100):int(9.3 * 100)].mean()))
    p("  补偿量(显示−输入) = %+.0f → %+.0f" %
      ((main_tot - pre_tot)[int(4.3 * 100):int(9.3 * 100)].mean(),
       (main_tot - pre_tot)[-500:].mean()))
    p("")
    p("=== ② 逐 20 s 桶：录制口径与内部口径对照 ===")
    p("%7s %9s %9s %9s %9s %9s %8s %7s %7s %7s %7s %6s" %
      ("t", "pre", "algo", "offset", "creep", "ded", "resid", "g", "r_med", "r_wmn",
       "gam_med", "state"))
    for (i, j) in bucket(el, 20.0):
        t = 0.5 * (el[i] + el[min(j, len(el)) - 1])
        f = lambda k: float(np.median(D[k][i:j]))
        p("%7.1f %9.0f %9.0f %9.0f %9.0f %9.0f %8.0f %7.4f %7.4f %7.4f %7.3f %6.0f" %
          (t, np.median(pre_tot[i:j]), np.median(main_tot[i:j]),
           np.median((main_tot - pre_tot)[i:j]),
           f("ideal_ded"), f("ded_capped"), f("ideal_ded") - f("ded_capped"),
           f("g"), f("r_med"), f("r_wmean"), f("gam_med"), f("state")))
    p("")
    p("=== ③ 复算 vs 录制（复算的内部量只用于归因，绝对电平有约 2% 起始态偏差）===")
    off_rec = main_tot - pre_tot
    off_rep = D["comp_total"]
    p("  补偿量：录制 首 %.0f 末 %.0f | 复算 首 %.0f 末 %.0f" %
      (np.median(off_rec[430:930]), np.median(off_rec[-500:]),
       np.median(off_rep[430:930]), np.median(off_rep[-500:])))
    p("")
    p("=== ④ 残漂的机制拆解（用复算内部量，全程中位）===")
    need = D["r_wmean"]           # A 加权均值：把 loaded 通道蠕变全部扣掉所需的相对量
    used = D["g"] * D["gam_med"]  # 实际用的相对量（median 共识 × γ）
    gap = need - used
    m = D["state"] == 2
    p("  慢相帧占比 %.3f" % float(m.mean()))
    for name, arr in (("需要(A加权均值) r_wmean", need), ("实际 g·γ_med", used),
                      ("差 gap", gap)):
        v = arr[m]
        p("  %-22s 中位 %+.5f  p10 %+.5f  p90 %+.5f" %
          (name, float(np.median(v)), float(np.percentile(v, 10)),
           float(np.percentile(v, 90))))
    A_sum = np.median(D["A_sum"][m])
    p("  A_sum 中位 %.0f  ⇒ gap 折算 %.0f ADC" % (A_sum, np.median(gap[m]) * A_sum))
    p("  其中 median vs 加权均值（空间不均匀）: %.0f ADC" %
      (np.median((D["r_wmean"] - D["r_med"])[m]) * A_sum))
    p("  其中 γ 偏离 1:                        %.0f ADC" %
      (np.median((D["r_med"] * (1 - D["gam_med"]))[m]) * A_sum))
    p("")
    p("=== ⑤ γ 的分布（loaded 通道，慢相段）===")
    for name in ("gam_med", "gam_min", "gam_max"):
        v = D[name][m]
        p("  %-8s 中位 %.3f  p10 %.3f  p90 %.3f" %
          (name, float(np.median(v)), float(np.percentile(v, 10)),
           float(np.percentile(v, 90))))
    p("  loaded 通道数 中位 %.0f" % float(np.median(D["n_loaded"][m])))
    p("")
    p("=== ⑥ 逐通道漂移是否均匀（对 pre 直接测，载荷保持段）===")
    i0, i1 = int(60 * 100), len(el) - 1
    base = pre[i0:i0 + 500].mean(0)
    last = pre[-500:].mean(0)
    rel = (last - base) / np.maximum(np.abs(base), 1e-9)
    amp = base - pre[:430].mean(0)
    order = np.argsort(-np.abs(amp))
    p("  %-4s %10s %10s %10s %10s" % ("ch", "idle", "受载基线", "末值", "相对漂移%"))
    for k in order[:12]:
        p("  ch%-3d %10.0f %10.0f %10.0f %10.2f%%" %
          (k, pre[:430, k].mean(), base[k], last[k], 100 * rel[k]))
    p("  受载通道(amp>10%%峰) 数 %d" % int(np.sum(np.abs(amp) > 0.10 * np.abs(amp).max())))
    w = np.abs(amp)
    p("  A 加权相对漂移中位 %.2f%%   简单中位 %.2f%%   极差 %.2f%%" %
      (100 * np.sum(w * rel) / np.sum(w), 100 * np.median(rel), 100 * (rel.max() - rel.min())))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
