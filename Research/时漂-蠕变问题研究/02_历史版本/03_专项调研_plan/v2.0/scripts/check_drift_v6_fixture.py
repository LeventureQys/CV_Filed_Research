# -*- coding: utf-8 -*-
"""校验 tests/fixtures/drift_v6 夹具与 100 Hz 全量录制的关键数字一致（装配后自检）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

FIX = os.path.join(L.ROOT, "tests", "fixtures", "drift_v6")


def main():
    print("== 夹具自检 ==")
    for sid in ("a_onset", "b_handoff", "c_cycle"):
        pre = L.load_stream(FIX, "seg_%s_pre.csv" % sid)
        main = L.load_stream(FIX, "seg_%s_main.csv" % sid)
        tp = pre["V"].sum(1)
        tm = main["V"].sum(1)
        el = pre["el"]
        off = tm - tp
        fs = pre["n"] / (el[-1] - el[0])
        print("%-10s n=%5d t=[%7.3f,%7.3f] fs=%6.2fHz pre[%6.0f,%6.0f] "
              "off_med=%7.1f off[%7.0f,%7.0f]"
              % (sid, pre["n"], el[0], el[-1], fs, tp.min(), tp.max(),
                 np.median(off), off.min(), off.max()))
        if sid == "b_handoff":
            for t in (39.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 48.0):
                m = (el >= t) & (el < t + 0.6)
                if m.any():
                    print("    t=%5.1f  pre=%7.0f  main=%7.0f  off=%7.0f"
                          % (t, np.mean(tp[m]), np.mean(tm[m]), np.mean(off[m])))

    # 与 100 Hz 全量的对照（同一时刻、0.6 s 窗）
    print("\n== 与 100 Hz 全量录制对照（b_handoff 段） ==")
    d = L.load_dataset(L.DS_ZERO)
    tp_f = d["pre"]["V"].sum(1)
    tm_f = d["main"]["V"].sum(1)
    el_f = d["pre"]["el"]
    pre = L.load_stream(FIX, "seg_b_handoff_pre.csv")
    main = L.load_stream(FIX, "seg_b_handoff_main.csv")
    el_s = pre["el"]
    tp_s = pre["V"].sum(1)
    tm_s = main["V"].sum(1)
    for t in (39.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 48.0):
        mf = (el_f >= t) & (el_f < t + 0.6)
        ms = (el_s >= t) & (el_s < t + 0.6)
        if mf.any() and ms.any():
            off_f = np.mean(tm_f[mf]) - np.mean(tp_f[mf])
            off_s = np.mean(tm_s[ms]) - np.mean(tp_s[ms])
            print("    t=%5.1f  全量 off=%7.0f  夹具 off=%7.0f  Δ=%5.0f"
                  % (t, off_f, off_s, off_s - off_f))


if __name__ == "__main__":
    main()
