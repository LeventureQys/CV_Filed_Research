# -*- coding: utf-8 -*-
"""探针 B：复算与录制结果的差异结构 + 录制流完整性（是否被录制频率限流）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import v30_lib as L  # noqa: E402

DS = os.path.join(L.OVERVIEW, "20260919_141824_single_device_110871")
RES = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))


def main():
    ds = L.load_dataset(DS)
    el = ds["pre"]["el"]
    fr = ds["pre"]["fr"]
    X = np.load(os.path.join(RES, "v30_trace_20260919_141824_single_device_110871.npy"))
    cols = open(os.path.join(RES, "v30_trace_20260919_141824_single_device_110871.cols"),
                encoding="utf-8").read().split()
    D = {c: X[:, i] for i, c in enumerate(cols)}
    pre_tot = ds["pre"]["V"].sum(1)
    main_tot = ds["main"]["V"].sum(1)
    raw_tot = ds["raw"]["V"].sum(1)

    print("frame_index: 首 %d 末 %d 连续? %s 唯一数 %d" %
          (fr[0], fr[-1], bool(np.all(np.diff(fr) == 1)), len(np.unique(fr))))
    print("elapsed 唯一数 %d / 帧数 %d  => 每包帧数 中位 %.1f 最大 %d" %
          (len(np.unique(el)), len(el),
           len(el) / len(np.unique(el)),
           int(np.max(np.diff(np.concatenate(([0], np.nonzero(np.diff(el))[0] + 1,
                                              [len(el)])))))))
    print("elapsed 跨度 %.2f s / 帧数 %d => %.2f Hz" % (el[-1] - el[0], len(el),
                                                     (len(el) - 1) / (el[-1] - el[0])))
    print("")
    print("pre vs raw 总量差：max %.3f 中位 %.3f" %
          (np.abs(pre_tot - raw_tot).max(), np.median(np.abs(pre_tot - raw_tot))))
    print("pre vs raw 逐格差：max %.3f  非零格占比 %.4f" %
          (np.abs(ds["pre"]["V"] - ds["raw"]["V"]).max(),
           float(np.mean(np.abs(ds["pre"]["V"] - ds["raw"]["V"]) > 1e-9))))
    print("")
    dd = D["sum_out"] - main_tot
    print("复算 vs 录制：Δ 中位 %.1f  中位绝对 %.1f  首帧 %.1f 末帧 %.1f" %
          (np.median(dd), np.median(np.abs(dd)), dd[0], dd[-1]))
    step = max(1, len(el) // 16)
    print("  Δ 轨迹: " + " ".join("%.0f" % v for v in dd[::step]))
    print("  t  轨迹: " + " ".join("%.0f" % v for v in el[::step]))
    print("")
    print("复算 sum_in vs pre 总量：max %.3f" % np.abs(D["sum_in"] - pre_tot).max())
    # 算法到底在什么状态
    for name, key in (("idle", "state==0"), ("event", "state==1"), ("slow", "state==2")):
        pass
    st = D["state"]
    print("状态占比：idle %.3f  event %.3f  slow %.3f" %
          (float(np.mean(st == 0)), float(np.mean(st == 1)), float(np.mean(st == 2))))


if __name__ == "__main__":
    main()
