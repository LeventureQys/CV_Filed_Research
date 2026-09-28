# -*- coding: utf-8 -*-
"""v3.4 回放定因：用真实 C++ 本体（v3.1 代码）回放 pre 流，输出关键窗口的状态机时间线。"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(HERE, "build", "v34_runner.exe")
DS = os.path.join(L.DATA_ROOT, "working", "零基线-反复增减同一负载",
                  "20260919_160854_single_device_7b3977")

FIELDS = ("t sum_in sum_out state ev_valid ev_kind g A_sum n_loaded "
          "gam_med gam_min gam_max r_med r_wmean aggA_loaded num_loaded "
          "ded_unclamped ded_clamped ded_capped ideal_ded comp_total "
          "level_ref ts_smooth min_ts max_ts max_tot idle_now valley_now "
          "valley_run tau tau_g0 tglide A_hat inc_max c_applied stalled "
          "clamp_hits shape_hits trim_sum max_clamp_viol pct_sum n_pct").split()


def replay(extra=()):
    ds = L.load_dataset(DS)
    pre = ds["pre"]
    n = pre["V"].shape[1]
    lines = [str(n)]
    for i in range(pre["n"]):
        parts = ["%.6f" % pre["ts"][i]] + ["%.1f" % x for x in pre["V"][i]]
        lines.append(" ".join(parts))
    cmd = [RUNNER] + list(extra)
    p = subprocess.run(cmd, input="\n".join(lines), capture_output=True,
                       text=True, encoding="utf-8", cwd=HERE)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        raise SystemExit("runner failed")
    rows = []
    for ln in p.stdout.splitlines():
        f = ln.split()
        if not f or f[0] in ("t", "OK", "END", "CH"):
            if f and f[0] == "END":
                print("END:", ln)
            continue
        try:
            rows.append([float(x) for x in f])
        except ValueError:
            pass
    arr = np.array(rows)
    return pre, arr


def show(arr, t0, t1, step=0.5):
    hdr = FIELDS
    idx = {k: i for i, k in enumerate(hdr)}
    el = arr[:, 0]
    # 时间轴是 steady clock，转换成 elapsed 便于对读
    print("窗口 %.1f ~ %.1f（elapsed）" % (t0, t1))
    shown = -1e9
    prev_state = None
    for r in arr:
        pass
    # 先做 steady->elapsed 映射
    return idx


def main():
    pre, arr = replay()
    np.save(os.path.join(HERE, "..", "results", "v34_replay_default.npy"), arr)
    # steady -> elapsed 线性映射（用 pre 的两列）
    ts, el = pre["ts"], pre["el"]
    # 取首尾两点线性换算即可（同一时钟）
    a = (el[-1] - el[0]) / (ts[-1] - ts[0])
    b = el[0] - a * ts[0]
    arr_e = a * arr[:, 0] + b

    idx = {k: i for i, k in enumerate(FIELDS)}
    state_names = {0: "Idle", 1: "Event", 2: "Slow"}

    def w(t0, t1, dt=0.5):
        print("\n=== 窗口 %.1f ~ %.1f ===" % (t0, t1))
        m = (arr_e >= t0) & (arr_e <= t1)
        sub = arr[m]
        sub_e = arr_e[m]
        nxt = -1e9
        last = None
        for r, e in zip(sub, sub_e):
            cur = (r[idx["state"]], r[idx["ev_valid"]], r[idx["ev_kind"]])
            if last is not None and cur != last:
                print("  [%8.2f] 状态变化 -> %s ev=%d kind=%d" %
                      (e, state_names.get(int(cur[0]), int(cur[0])), cur[1], cur[2]))
            last = cur
            if e - nxt >= dt - 1e-9:
                nxt = e
                print("  %8.2f in=%7.0f out=%7.0f %s g=%+.3f A=%7.0f nld=%2d "
                      "ded=%7.0f pct=%7.0f idle=%d tau=%6.2f Ahat=%7.0f incm=%7.0f capp=%7.0f" %
                      (e, r[idx["sum_in"]], r[idx["sum_out"]],
                       state_names.get(int(r[idx["state"]])),
                       r[idx["g"]], r[idx["A_sum"]], int(r[idx["n_loaded"]]),
                       r[idx["ded_clamped"]], r[idx["pct_sum"]],
                       int(r[idx["idle_now"]]),
                       r[idx["tau"]], r[idx["A_hat"]], r[idx["inc_max"]],
                       r[idx["c_applied"]]))

    for t0, t1 in [(230.0, 234.5), (238.0, 242.0), (243.0, 248.0),
                   (250.0, 263.5)]:
        w(t0, t1, 1.0)


if __name__ == "__main__":
    main()
