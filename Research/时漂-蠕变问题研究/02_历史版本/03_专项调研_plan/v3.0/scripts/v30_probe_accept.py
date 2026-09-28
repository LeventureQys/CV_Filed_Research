# -*- coding: utf-8 -*-
"""探针 H（验收证据）：目标录制 + 振荡工况的最终对照，落盘 results/v30_accept.txt。

口径：
  ① 目标录制总量轨迹 + 60→390 s / 6→390 s 漂移（输入 / 录制现役 / v2.0 复算 / v3.0 复算）
  ② 振荡工况专用口径（振荡段 std、极差、安静段偏移中位）
  ③ 逐通道末帧快照（v3.0）：A_k / 是否 loaded / 是否 PCT 生效 / pct 扣除量
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402
import v30_ab as A  # noqa: E402
import v30_probe_special as S  # noqa: E402

RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
OUT = os.path.join(RES, "v30_accept.txt")
TAG = "20260919_141824_single_device_110871"


def ch_dump(el, V, args):
    n = V.shape[1]
    lines = ["%d" % n] + ["%.6f " % t + " ".join("%.6f" % x for x in r)
                          for t, r in zip(el, V)]
    env = dict(os.environ)
    env["V30_DUMP_CH"] = "1"
    p = subprocess.run([A.RUNNERS["base"]] + args, input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    return [l for l in p.stdout.splitlines() if l.startswith("CH ")]


def main():
    lines = []
    p = lines.append
    ds = os.path.join(L.OVERVIEW, TAG)
    el, V = A.read_any(os.path.join(ds, "device_001_pre_seg0.csv"))
    tin = V.sum(1)
    trec = A.read_any(os.path.join(ds, "device_001_seg000.csv"))[1].sum(1)
    d20 = A.run_arm(A.RUNNERS["base"], ["--pct", "0"], el, V)["sum_out"]
    d30 = A.run_arm(A.RUNNERS["base"], [], el, V)["sum_out"]

    p("=== ① 目标录制 %s（总量口径，21 通道求和，ADC）===" % TAG)
    p("帧 %d  时长 %.1f s  采样 %.2f Hz" % (len(el), el[-1] - el[0],
                                          (len(el) - 1) / (el[-1] - el[0])))
    p("%6s %9s %11s %11s %11s" % ("t", "输入", "录制(现役)", "v2.0复算", "v3.0复算(PCT)"))
    for a in range(30, 400, 30):
        m = (el >= a) & (el < a + 2)
        p("%6d %9.0f %11.0f %11.0f %11.0f" %
          (a, np.median(tin[m]), np.median(trec[m]), np.median(d20[m]), np.median(d30[m])))
    g = lambda x, a, b: np.median(x[el >= b]) - np.median(x[(el >= a) & (el < a + 2)])
    p("")
    p("%-14s %10s %10s %10s %10s" % ("窗口", "输入", "录制(现役)", "v2.0复算", "v3.0复算"))
    for a, b, lbl in ((60, 389, "60→390 s"), (6, 389, "6→390 s"), (30, 389, "30→390 s")):
        p("%-14s %+10.0f %+10.0f %+10.0f %+10.0f" %
          (lbl, g(tin, a, b), g(trec, a, b), g(d20, a, b), g(d30, a, b)))
    p("")
    p("→ 60→390 s 漂移：v2.0 复算 %+.0f → v3.0 %+.0f（%.1f×）；现役录制 %+.0f" %
      (g(d20, 60, 389), g(d30, 60, 389),
       abs(g(d20, 60, 389) / g(d30, 60, 389)) if abs(g(d30, 60, 389)) > 1 else float("nan"),
       g(trec, 60, 389)))
    p("")
    p("=== ② 逐通道末帧快照（v3.0，末帧）===")
    ch = ch_dump(el, V, [])
    p("%-20s" % ch[0])
    for l in ch[1:]:
        f = l.split()
        if abs(float(f[4])) < 1e-9 and abs(float(f[6])) < 1: continue
        p(l)
    p("")
    p("=== ③ 振荡工况专用口径 ===")
    odd = os.path.join(L.ROOT, "temp", "算法数据&原始数据", "各种奇怪工况")
    if os.path.isdir(odd):
        d0 = sorted(os.listdir(odd))[0]
        eo, Vo = A.read_any(os.path.join(odd, d0, "device_001_pre_seg0.csv"))
        to = Vo.sum(1)
        o1 = A.run_arm(A.RUNNERS["base"], ["--pct", "0"], eo, Vo)["sum_out"]
        o2 = A.run_arm(A.RUNNERS["base"], [], eo, Vo)["sum_out"]
        p("数据 %s" % d0)
        p("%-16s %12s %12s %14s" % ("臂", "振荡段std", "振荡段极差", "安静段偏移中位"))
        for n, o in (("v2.0(PCT关)", o1), ("v3.0(PCT开)", o2)):
            m = S.osc_metrics(eo, to, o)
            p("%-16s %12.0f %12.0f %14.0f" %
              (n, m["osc_std"], m["osc_pkpk"], m["quiet_med"]))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print("\n-> %s" % OUT)


if __name__ == "__main__":
    main()
