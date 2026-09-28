# -*- coding: utf-8 -*-
"""plan-v3.0 诊断驱动：把算法输入流喂进内部状态导出器，解析成 numpy 结构。

输出：results/v30_trace_<tag>.csv（逐帧内部状态）+ 返回 dict
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.join(SCRIPTS, "build", "v30_runner.exe")
RES = os.path.abspath(os.path.join(SCRIPTS, "..", "results"))

COLS = ("t sum_in sum_out state ev_valid ev_kind g A_sum n_loaded "
        "gam_med gam_min gam_max r_med r_wmean aggA_loaded num_loaded "
        "ded_unclamped ded_clamped ded_capped ideal_ded comp_total "
        "level_ref ts_smooth min_ts max_ts max_tot idle_now valley_now valley_run "
        "tau tau_g0 tglide A_hat inc_max c_applied stalled clamp_hits shape_hits "
        "trim_sum").split()


def run_stream(el, V, legacy=False, channels=None):
    n = V.shape[1]
    idx = list(range(n)) if channels is None else list(channels)
    lines = ["%d" % len(idx)]
    for t, row in zip(el, V):
        lines.append("%.6f " % float(t) + " ".join("%.6f" % row[c] for c in idx))
    inp = "\n".join(lines) + "\n"
    args = [RUNNER] + (["--legacy"] if legacy else [])
    p = subprocess.run(args, input=inp, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError("runner rc=%d stderr=%s" % (p.returncode, p.stderr[:500]))
    rows = p.stdout.splitlines()
    assert rows and rows[0].startswith("t "), rows[0][:200]
    ok = rows[1].split()
    assert ok[0] == "OK", ok
    frames = int(ok[2])
    data = np.array([[float(x) for x in r.split()] for r in rows[2:2 + frames]])
    end = rows[2 + frames].split()
    meta = dict(zip(
        "clamp_hits shape_hits n_valley_exit n_reanchor_idle n_g_floor "
        "n_g_valley_reset in_event in_slow g".split(),
        [float(x) for x in end[1:]]))
    return dict(cols=COLS, X=data, meta=meta,
                D={c: data[:, i] for i, c in enumerate(COLS)})


def main():
    ds_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        L.OVERVIEW, "20260919_141824_single_device_110871")
    tag = os.path.basename(ds_dir.rstrip("\\/"))
    ds = L.load_dataset(ds_dir)
    el = ds["pre"]["el"]
    tr = run_stream(el, ds["pre"]["V"])
    D = tr["D"]
    main_tot = ds["main"]["V"].sum(1)
    d = D["sum_out"] - main_tot
    print("=== 校验：真实 C++ 复算 vs 录制算法结果 ===")
    print("  帧数 %d" % len(el))
    print("  总量差：max|Δ| %.3f  中位|Δ| %.3f  RMS %.3f" %
          (np.abs(d).max(), np.median(np.abs(d)), float(np.sqrt((d ** 2).mean()))))
    print("  逐帧完全相同(1e-6)的比例 %.4f" % float(np.mean(np.abs(d) < 1e-6)))
    print("  runner meta: %s" % tr["meta"])
    np.save(os.path.join(RES, "v30_trace_%s.npy" % tag), tr["X"])
    with open(os.path.join(RES, "v30_trace_%s.cols" % tag), "w", encoding="utf-8") as fh:
        fh.write(" ".join(COLS) + "\n")
    print("  -> results/v30_trace_%s.npy" % tag)


if __name__ == "__main__":
    main()
