# -*- coding: utf-8 -*-
"""v3.6 回放驱动：把录制喂给离线助手（真实 C++ 本体），取回逐帧内部状态。

用法（库）：
    from v36_replay import run_arm, EXE_V36, EXE_V31
    st = run_arm(ds, EXE_V36, ["--seed-gain", "0.7"])
`run_arm` 返回 dict：`t, sum_in, sum_out, ded, <每个内部字段> ...`
字段名见 `v36_runner.cpp` 的 kHeader（`HEADER` 常量）。
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v36_lib as K  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.abspath(os.path.join(HERE, "..", "build"))
EXE_V36 = os.path.join(B, "v36_runner.exe")
EXE_V31 = os.path.abspath(os.path.join(HERE, "..", "..", "v3.1", "scripts", "build",
                                       "v30_runner.exe"))
EXE_V32 = os.path.abspath(os.path.join(HERE, "..", "..", "v3.2", "build", "v32_runner.exe"))

HEADER = ("t sum_in sum_out state ev_valid ev_kind g A_sum n_loaded "
          "gam_med gam_min gam_max r_med r_wmean aggA_loaded num_loaded "
          "ded_unclamped ded_clamped ded_capped ideal_ded comp_total "
          "level_ref ts_smooth min_ts max_ts max_tot idle_now valley_now valley_run "
          "tau tau_g0 tglide A_hat inc_max c_applied stalled clamp_hits shape_hits trim_sum "
          "max_clamp_viol pct_sum n_pct creep_ratio seed_used w_seed ded1_sum ded_n").split()


def _feed(ds, nch):
    n = ds["pre"]["n"]
    el = ds["pre"]["el"]
    V = ds["pre"]["V"]
    lines = [str(nch)]
    for i in range(n):
        lines.append("%.6f " % el[i] + " ".join("%.6f" % x for x in V[i]))
    return "\n".join(lines) + "\n"


def run_arm(ds, exe, args=(), nch=21, timeout=600):
    payload = _feed(ds, nch)
    env = dict(os.environ)
    env.setdefault("V30_DUMP_CH", "0")
    p = subprocess.run([exe] + [str(a) for a in args], input=payload.encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                       timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError("%s 退出码 %d\n%s" % (exe, p.returncode,
                                                 p.stderr.decode("utf-8", "replace")[-2000:]))
    txt = p.stdout.decode("utf-8", "replace").splitlines()
    cols = txt[0].split() if txt else []
    if "sum_in" not in cols:
        raise RuntimeError("表头异常:\n%s" % txt[0][:300])
    nfr = int(txt[1].split()[2])
    data = np.array([[float(x) for x in ln.split()] for ln in txt[2:2 + nfr]])
    st = {k: data[:, i] for i, k in enumerate(cols)}
    st["frames"] = nfr
    st["ded"] = st["sum_in"] - st["sum_out"]
    end = [ln for ln in txt if ln.startswith("END ")]
    st["end"] = end[0] if end else ""
    return st


def align_live(ds, st):
    """把回放结果与现场显示对齐，返回中位 |Δ|（同口径自检）。"""
    a = ds["tot_out"]
    b = st["sum_out"]
    n = min(len(a), len(b))
    return float(np.median(np.abs(a[:n] - b[:n]))), float(np.max(np.abs(a[:n] - b[:n])))


if __name__ == "__main__":
    ds = K.load(K.DS_TARGET)
    for name, exe in (("v3.1", EXE_V31), ("v3.2", EXE_V32), ("v3.6", EXE_V36)):
        st = run_arm(ds, exe)
        md, mx = align_live(ds, st)
        print("%-6s frames=%d  回放 vs 现场 中位|Δ|=%.1f  max|Δ|=%.1f  END=%s"
              % (name, st["frames"], md, mx, st["end"]))
