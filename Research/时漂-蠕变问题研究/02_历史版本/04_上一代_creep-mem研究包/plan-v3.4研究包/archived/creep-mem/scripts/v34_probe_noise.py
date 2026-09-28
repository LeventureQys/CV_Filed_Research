# -*- coding: utf-8 -*-
"""v3.4 噪声退化定位：对疑点会话逐平台算 OFF/ON 的显示 std 与输入 std，
并统计 ON 相对 OFF 多出的运动能量集中在什么时刻。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v34_probe_oos import run_stream, load_input, plateaus
import v30_lib as L

HERE = os.path.dirname(os.path.abspath(__file__))

SESSIONS = [
    ("31b7f9 随机切换", "20260919_134726_single_device_31b7f9"),
    ("再切换负载(B组)", "再切换负载"),
    ("中途切换负载(B组)", "中途切换负载"),
    ("7b3977 目标", "20260919_160854_single_device_7b3977"),
    ("6b2e70 剧烈变化", "20260919_161435_single_device_6b2e70"),
]


def find(sub):
    for root in [L.DATA_ROOT, os.path.join(L.ROOT, "temp", "原始数据only")]:
        for label, d in L.discover_sessions(root, 5):
            if sub in label.replace("\\", "/"):
                return label.replace("\\", "/"), d
    raise SystemExit("not found " + sub)


def main():
    for tag, sub in SESSIONS:
        label, d = find(sub)
        name, s = load_input(d)
        el = s["el"]
        sin0, o0 = run_stream(s["ts"], s["V"], ["--mem", "0"])
        _, o1 = run_stream(s["ts"], s["V"], ["--mem", "120"])
        n = min(len(o0), len(el))
        el, sin0, o0, o1 = el[:n], sin0[:n], o0[:n], o1[:n]
        print("\n== %s  (%s)" % (tag, label))
        for a, b in plateaus(el, sin0):
            w0, w1 = a + 50, b - 50
            if w1 - w0 < 100:
                continue
            si = np.std(sin0[w0:w1])
            print("  平台 t=%6.1f~%6.1f  in_std=%6.0f  OFF_std=%6.0f  "
                  "ON_std=%6.0f  (in中位=%7.0f)" %
                  (el[min(a, n - 1)], el[min(b, n - 1)], si,
                   np.std(o0[w0:w1]), np.std(o1[w0:w1]),
                   np.median(sin0[a:b])))
        # ON 额外运动能量的时间分布（0.5s 差分）
        k = 50
        d0 = np.abs(np.diff(o0[::k])) 
        d1 = np.abs(np.diff(o1[::k]))
        extra = d1 - d0
        top = np.argsort(extra)[-5:][::-1]
        print("  ON 额外运动最大时刻(s):",
              ["%.0f:+%d" % (el[min(i * k, len(el) - 1)], extra[i]) for i in top])


if __name__ == "__main__":
    main()
