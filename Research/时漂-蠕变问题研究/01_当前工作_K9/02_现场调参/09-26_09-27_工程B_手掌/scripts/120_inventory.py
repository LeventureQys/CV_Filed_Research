# -*- coding: utf-8 -*-
"""120_inventory：手掌数据全会话盘点（模式/算法/参数/流/时长/台阶）。

扫描 data\手掌数据 与 data\archived\手掌数据，打印每会话可用于「力值模式整定」的关键事实：
display_mode、algorithm.enabled、algorithm.params（录制实际下发）、split_streams、CSV 流与时长。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOTS = [
    Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\手掌数据"),
    Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\data\archived\手掌数据"),
]


def csv_span(path: Path) -> tuple[int, float, int]:
    """返回 (帧数, 时长s, 通道数)；只读 ##Data 块，不整表解析。"""
    n = 0
    t_first = t_last = None
    m = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        in_data = False
        names = False
        for ln in f:
            if not in_data:
                if ln.startswith("##Data"):
                    in_data = True
                continue
            if not names:
                names = True
                m = max(0, len(ln.strip().split(",")) - 3)
                continue
            if not ln.strip():
                continue
            p = ln.split(",", 3)
            try:
                e = float(p[1])
            except Exception:
                continue
            if t_first is None:
                t_first = e
            t_last = e
            n += 1
    dur = (t_last - t_first) if (t_first is not None and t_last is not None) else 0.0
    return n, dur, m


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    for root in ROOTS:
        if not root.is_dir():
            continue
        print(f"\n===== {root} =====")
        for sess in sorted(root.iterdir()):
            sj = sess / "session.json"
            if not sj.is_file():
                continue
            j = json.loads(sj.read_text(encoding="utf-8-sig"))
            alg = j.get("algorithm", {})
            cal = j.get("calibration", {})
            streams = sorted(p.name for p in sess.glob("*.csv"))
            info = []
            for name in streams:
                n, dur, m = csv_span(sess / name)
                info.append(f"{name}: n={n} {dur:.1f}s m={m}")
            print(f"\n[{sess.name}]  mode={cal.get('display_mode')} "
                  f"force_conv={cal.get('force_conversion_active')} stage={cal.get('value_stage')}")
            print(f"  algo.enabled={alg.get('enabled')} id={alg.get('id')!r} "
                  f"split={alg.get('split_streams')}")
            p = alg.get("params") or {}
            keys = ["r_fast", "tau_c_fast_s", "slow_confirm_s", "soft_unfreeze_s",
                    "slope_cap_frac", "r_slow_max", "tau_r_fast_s", "tau_r_slow_idle_s"]
            if p:
                print("  params: " + ", ".join(f"{k}={p.get(k)}" for k in keys))
            else:
                print("  params: (空)")
            for s in info:
                print("  " + s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
