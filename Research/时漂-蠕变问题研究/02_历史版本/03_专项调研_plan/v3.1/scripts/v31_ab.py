# -*- coding: utf-8 -*-
"""plan-v3.1 离线 A/B：**修复前 vs 修复后**（18 条录制，真实 C++ 本体）。

臂：
  v2.0(PCT关)      —— plan-v2.0 基线（--pct 0，修复对它逐位无影响）
  prefix(PCT开)    —— plan-v3.0 首版（PCT 开，但沉降窗/重锚漏了 PCT）★对照臂
  fixed(PCT开)     —— 本轮修复（DeductionVector 含 PCT；Handoff 清 pct；A_new 用总扣除）

指标（沿用 plan-v3.0 的口径，见 v30_ab.metrics）：
  保压偏离 = 受载平台内 |显示 − 加载后落定电平([1.5,4.5] s)| / 台阶 的中位
  早段残漂 = 同一参考窗 → 平台末段的漂移；后段残漂 = 平台中点 → 末段
  G中位    = 上升沿后 4~5 s 的 Δ显示/Δ输入
  ★ 新增 卸载台阶保真 Gd 与 卸载回跳 bounce（本轮的核心口径）

输出：results/v31_ab.txt
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

S30 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "v3.0", "scripts"))
sys.path.insert(0, S30)
import v30_ab as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
RUNNER_FIXED = os.path.join(HERE, "build", "v30_runner.exe")
RUNNER_PREFIX = os.path.join(HERE, "build", "v31_runner_prefix.exe")
ARMS = [("v2.0基线", RUNNER_PREFIX, ["--pct", "0", "--freeze", "0"]),
        ("v3.0首版", RUNNER_PREFIX, ["--freeze", "0"]),
        ("v3.1仅修#1", RUNNER_FIXED, ["--freeze", "0"]),
        ("v3.1(#1+#2)", RUNNER_FIXED, ["--freeze", "1"]),
        ("v3.1+单侧(P1)", RUNNER_FIXED, ["--freeze", "1", "--pct-mono", "1"])]


def wmed(el, x, t0, t1):
    m = (el >= t0) & (el < t1)
    return float(np.median(x[m])) if m.any() else float("nan")


def unload_metrics(el, tin, tout):
    """卸载沿（|Δin|>1500 的下降沿）的台阶保真与回跳。"""
    gs, bounces = [], []
    n = len(el)
    d = np.array([wmed(el, tin, t, t + 1.2) - wmed(el, tin, t - 0.4, t) for t in el])
    d = np.nan_to_num(d)
    hot = d < -900.0
    i = 0
    while i < n:
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < n and hot[j]:
            j += 1
        k = i + int(np.argmin(d[i:j]))
        t = float(el[k])
        di = wmed(el, tin, t + 3, t + 5) - wmed(el, tin, t - 2.5, t - 0.5)
        do = wmed(el, tout, t + 3, t + 5) - wmed(el, tout, t - 2.5, t - 0.5)
        if abs(di) > 1500:
            gs.append(do / di if abs(di) > 1 else float("nan"))
            # 回跳 = 沿后 [0.4,1.0] s 相对沿后 [0,0.15] s 的抬升（参考：输入同期变化）
            lo = wmed(el, tout, t, t + 0.15)
            hi = wmed(el, tout, t + 0.4, t + 1.0)
            ilo = wmed(el, tin, t, t + 0.15)
            ihi = wmed(el, tin, t + 0.4, t + 1.0)
            bounces.append((hi - lo) - (ihi - ilo))
        i = j
    return (float(np.median(gs)) if gs else float("nan"),
            float(np.median(bounces)) if bounces else float("nan"),
            float(np.max(np.abs(bounces))) if bounces else float("nan"),
            len(gs))


def main():
    items = A.collect()
    lines = []
    p = lines.append
    p("plan-v3.1 离线 A/B：%d 条录制 × %d 臂（修复前 vs 修复后）" % (len(items), len(ARMS)))
    p("")
    for name, runner, args in ARMS:
        p("── 臂 %s ──" % name)
        p("%-30s %9s %9s %9s %9s %8s %8s %8s %6s" %
          ("数据集", "保压偏离", "偏离p95", "后段残漂", "早段残漂", "G中位",
           "卸载Gd", "卸载回跳", "越界"))
        for lbl, path in items:
            el, V = A.read_any(path)
            tin = V.sum(1)
            try:
                D = A.run_arm(runner, args, el, V)
            except Exception as exc:  # noqa: BLE001
                p("%-30s  RUN FAIL %s" % (lbl, exc))
                continue
            m = A.metrics(el, V, D["sum_out"], tin, D["sum_out"], D["max_clamp_viol"])
            gd, bm, bx, nedge = unload_metrics(el, tin, D["sum_out"])
            p("%-30s %9.4f %9.4f %9.4f %9.4f %8.3f %8.3f %8.0f %6d" %
              (lbl, m["dev_hold"], m["dev_hold_p95"], m["drift_late"], m["drift_disp"],
               m["G_med"], gd, bm, m["viol"]))
        p("")
    with open(os.path.join(RES, "v31_ab.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> %s" % os.path.join(RES, "v31_ab.txt"))


if __name__ == "__main__":
    main()
