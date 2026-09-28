# -*- coding: utf-8 -*-
"""T2 步骤0：枚举全部可发现会话，报帧数/时长/量程/加载沿数，供挑选研究对象。"""
import os
import sys

import numpy as np

import t2_lib as T


def main():
    print("数据根:", T.DOC_ROOT)
    print("%-64s %7s %8s %10s %10s %6s %6s"
          % ("会话", "帧数", "时长s", "输入min", "输入max", "沿数", "事件数"))
    rows = []
    for label, d in T.all_sessions():
        try:
            s = T.load_pre(d)
        except Exception as exc:  # noqa: BLE001
            print("%-64s  读取失败: %s" % (label, exc))
            continue
        el, tot = s["el"], T.total(s)
        evs = T.load_events(el, tot)
        rows.append((label, s["n"], el[-1], tot.min(), tot.max(), len(evs)))
        print("%-64s %7d %8.0f %10.0f %10.0f %6s %6d"
              % (label, s["n"], el[-1], tot.min(), tot.max(),
                 "-", len(evs)))
    print("\n共 %d 个会话；含 >=1 实质加载事件的 %d 个"
          % (len(rows), sum(1 for r in rows if r[5] >= 1)))


if __name__ == "__main__":
    sys.exit(main())
