# -*- coding: utf-8 -*-
"""plan-v3.0 离线 A/B：用**真实 C++ 本体**（现役 vs 候选）跑全部实机录制。

臂（arm）= runner 的一组参数：
  base    现役交付态（无新增参数）
  pct     逐通道蠕变跟踪（新增）
  ...

指标（全部用**总量口径**，跨数据集可比）：
  M1 平台残漂   每个受载平台段内：末 3 s 中位 − 段内 [start+3,+6] s 中位，除以台阶幅度
                （小 = 显示在恒载下不飘；输入自身的漂移另列对照）
  M2 全程跟踪   全录 |显示 − 输入| / 台阶幅度的中位与 p95（小 = 显示贴着输入）
  M3 台阶保真   每个上升沿后 4~5 s 的 G = Δ显示/Δ输入（越接近 1 越准）
  M4 不变量     显示 > 输入×1.005 + 0.01 的格点数（现役硬不变量，必须 0）
  M5 末段偏移   末平台段内 显示 − 输入/台阶（有符号，用来看"冻结在错的地方"）

输出：results/v30_ab.txt + results/v30_ab_cases.csv
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(SCRIPTS, "..", "results"))
RUNNERS = {
    "base": os.path.join(SCRIPTS, "build", "v30_runner.exe"),
    "proto": os.path.join(SCRIPTS, "build", "v30_runner.exe"),
}
ARMS = {
    "v2.0(PCT关)":      ("proto", ["--pct", "0"]),
    "v3.0(PCT hi=2)":   ("proto", ["--pct", "10", "--pct-hi", "2"]),
    "v3.0(PCT hi=4)★":  ("proto", ["--pct", "10"]),
    "v3.0(PCT mono)":   ("proto", ["--pct", "10", "--pct-mono", "1"]),
}


def read_any(path):
    """读任意录制；表头缺通道名时按位置回退。返回 el, V(, n)。"""
    with open(path, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    hdr = rows[di + 1].split(",")
    idx = [i for i, h in enumerate(hdr) if h.startswith("ch") and not h.startswith("ch_")]
    data = [r.split(",") for r in rows[di + 2:] if r.strip()]
    if not idx:
        idx = list(range(3, len(data[0])))
    el = np.array([float(f[1]) for f in data])
    V = np.array([[float(f[c]) for c in idx] for f in data])
    return el, V


def collect():
    """数据集清单：数据根下**自动发现**的会话 + `temp/原始数据only` 的 B 组。"""
    items = []
    for lbl, d in L.discover_sessions():
        p = os.path.join(d, "device_001_pre_seg0.csv")
        if not os.path.isfile(p):
            p = os.path.join(d, "device_001_seg000.csv")
        items.append(("A|%s" % lbl, p))
    base = os.path.join(L.ROOT, "temp", "原始数据only")
    for grp in ("右拇指指尖", "左拇指指尖", "四指指尖"):
        for d in ("数据1", "数据2", "数据3"):
            p = os.path.join(base, grp, d, "device_001_seg000.csv")
            if os.path.exists(p):
                items.append(("B|%s/%s" % (grp, d), p))
    root = os.path.join(base, "变化负载")
    if os.path.isdir(root):
        for d in sorted(os.listdir(root)):
            for dp, _dn, fns in os.walk(os.path.join(root, d)):
                for fn in fns:
                    if fn == "device_001_seg000.csv":
                        items.append(("B|变化负载/%s" % d, os.path.join(dp, fn)))
    return items


def run_arm(runner, args, el, V):
    n = V.shape[1]
    lines = ["%d" % n]
    for t, row in zip(el, V):
        lines.append("%.6f " % float(t) + " ".join("%.6f" % x for x in row))
    p = subprocess.run([runner] + args, input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("runner rc=%d %s" % (p.returncode, p.stderr[:300]))
    rows = p.stdout.splitlines()
    frames = int(rows[1].split()[2])
    X = np.array([[float(x) for x in r.split()] for r in rows[2:2 + frames]])
    cols = rows[0].split()
    return {c: X[:, i] for i, c in enumerate(cols)}


def metrics(el, Vin, Vout, tin, tout, D_viol=None):
    """总量口径指标（tin/tout 为逐帧总量）。

    step 取「受载平台中位 − 空载平台中位」（不是分位数差，否则会随占空比漂）。
    """
    segs = L.hysteresis_segments(tin, min_frames=50)
    lv_idle = [float(np.median(tin[i0:i1 + 1])) for k, i0, i1 in segs if k == "idle"]
    lv_load = [float(np.median(tin[i0:i1 + 1])) for k, i0, i1 in segs if k == "loaded"]
    if lv_idle and lv_load:
        step = max(lv_load) - min(lv_idle)
    else:
        step = float(tin.max() - tin.min())
    step = max(step, 1.0)
    plat = []          # (drift_disp, drift_in, dur, off_end)
    late = []          # (drift_disp_late, drift_in_late)
    hold = []          # (dev_med, dev_p95, dev_in_med)
    for kind, i0, i1 in segs:
        if kind != "loaded":
            continue
        dur = el[i1] - el[i0]
        if dur < 12.0:
            continue
        # 参考窗 = 加载沿后 [1.5, 4.5] s（v6 的承诺就是"1 s 后稳定"）
        j0 = i0
        while j0 <= i1 and el[j0] < el[i0] + 1.5:
            j0 += 1
        j1 = j0
        while j1 <= i1 and el[j1] < el[i0] + 4.5:
            j1 += 1
        k0 = i1
        while k0 >= i0 and el[k0] > el[i1] - 3.0:
            k0 -= 1
        if j1 <= j0 or k0 >= i1:
            continue
        ref_d = float(np.median(tout[j0:j1]))
        ref_i = float(np.median(tin[j0:j1]))
        end_d = float(np.median(tout[k0:i1 + 1]))
        end_i = float(np.median(tin[k0:i1 + 1]))
        plat.append(((end_d - ref_d) / step, (end_i - ref_i) / step, dur,
                     (end_d - end_i) / step))
        # 后段残漂：平台中点 → 末段（避开加载瞬态与 rank-1 收敛段，跨数据集最可比）
        m0 = i0
        while m0 <= i1 and el[m0] < el[i0] + 0.5 * dur:
            m0 += 1
        m1 = m0
        while m1 <= i1 and el[m1] < el[i0] + 0.5 * dur + 3.0:
            m1 += 1
        if m1 > m0:
            ld = float(np.median(tout[m0:m1]))
            li = float(np.median(tin[m0:m1]))
            late.append(((end_d - ld) / step, (end_i - li) / step))
        # ★ 决定性指标：显示在保压期内相对「加载后落定电平」的**偏离**（中位 / p95）
        #   —— 这正是用户看到的“一直飘”；且不会被“先冲高再回落”的驼峰骗过。
        dev = np.abs(tout[j1:i1 + 1] - ref_d) / step
        dev_i = np.abs(tin[j1:i1 + 1] - ref_i) / step
        if dev.size:
            hold.append((float(np.median(dev)), float(np.percentile(dev, 95)),
                         float(np.median(dev_i))))
    # 台阶保真
    gs = []
    up, _dn = L.edge_indices(tin, float(np.median(tin)) )
    for i in up:
        a = max(0, i - 25)
        b = min(len(el) - 1, i + int(4.5 * 100))
        c = min(len(el) - 1, i + int(5.5 * 100))
        if c <= b:
            continue
        di = float(np.median(tin[b:c]) - np.median(tin[a:i - 1] if i - 1 > a else tin[a:i + 1]))
        do = float(np.median(tout[b:c]) - np.median(tout[a:i - 1] if i - 1 > a else tout[a:i + 1]))
        if abs(di) > 0.15 * step:
            gs.append(do / di)
    # 不变量（runner 逐帧给出的单格最大越界）
    viol = int(np.sum(D_viol > 0.01)) if D_viol is not None else -1
    r = np.abs(tout - tin) / step
    out = {
        "n_plat": len(plat),
        "drift_disp": float(np.median([p[0] for p in plat])) if plat else float("nan"),
        "drift_in": float(np.median([p[1] for p in plat])) if plat else float("nan"),
        "drift_worst": float(np.max([abs(p[0]) for p in plat])) if plat else float("nan"),
        "drift_late": float(np.median([p[0] for p in late])) if late else float("nan"),
        "dev_hold": float(np.median([p[0] for p in hold])) if hold else float("nan"),
        "dev_hold_p95": float(np.max([p[1] for p in hold])) if hold else float("nan"),
        "dev_hold_in": float(np.median([p[2] for p in hold])) if hold else float("nan"),
        "drift_late_in": float(np.median([p[1] for p in late])) if late else float("nan"),
        "drift_late_worst": float(np.max([abs(p[0]) for p in late])) if late else float("nan"),
        "off_end": float(np.median([p[3] for p in plat])) if plat else float("nan"),
        "track_med": float(np.median(r)), "track_p95": float(np.percentile(r, 95)),
        "G_med": float(np.median(gs)) if gs else float("nan"),
        "G_min": float(np.min(gs)) if gs else float("nan"),
        "viol": viol,
    }
    return out


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    items = collect()
    if only:
        keys = only.split(",")
        items = [it for it in items if any(k in it[0] for k in keys)]
    lines = []
    p = lines.append
    p("plan-v3.0 离线 A/B：%d 条录制 × %d 臂" % (len(items), len(ARMS)))
    p("臂 = %s" % ", ".join("%s(%s %s)" % (k, v[0], " ".join(v[1])) for k, v in ARMS.items()))
    p("")
    hdr = ("%-26s %8s %8s %8s %9s %8s %7s %6s" %
           ("数据集", "保压偏离", "偏离p95", "输入偏离", "后段残漂", "早段残漂", "G中位", "越界"))
    for name, (rk, args) in ARMS.items():
        p("── 臂 %s ──" % name)
        p(hdr)
        for lbl, path in items:
            el, V = read_any(path)
            try:
                D = run_arm(RUNNERS[rk], args, el, V)
            except Exception as exc:  # noqa: BLE001
                p("%-26s  RUN FAIL: %s" % (lbl, exc))
                continue
            m = metrics(el, V, D["sum_out"], V.sum(1), D["sum_out"], D["max_clamp_viol"])
            p("%-26s %8.4f %8.4f %8.4f %9.4f %8.4f %7.3f %6d" %
              (lbl, m["dev_hold"], m["dev_hold_p95"], m["dev_hold_in"],
               m["drift_late"], m["drift_disp"], m["G_med"], m["viol"]))
        p("")
    with open(os.path.join(RES, "v30_ab.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n-> results/v30_ab.txt")


if __name__ == "__main__":
    main()
