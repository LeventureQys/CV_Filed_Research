# -*- coding: utf-8 -*-
"""14 会话二进制预处理：把 16 份会话 CSV 转成 [elapsed, ch0..ch_{m-1}] 的 float64 二进制，
   供 C++ 扫参工具直接读取（每次调用省掉几秒 CSV 解析）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS, load  # noqa: E402


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    out = OUT_DIR / "_sess"
    out.mkdir(parents=True, exist_ok=True)
    for sensor, info in SENSORS.items():
        for tag, rel in info["sessions"].items():
            key = f"{sensor}/{tag}"
            d = load(rel)
            t = d["t"]
            V = d["v"] if "v" in d else d["V"]
            n, m = V.shape
            buf = np.empty((n, m + 1), dtype=np.float64)
            buf[:, 0] = t
            buf[:, 1:] = V
            f = out / (key.replace("/", "_") + ".bin")
            with open(f, "wb") as fh:
                fh.write(np.array([n, m], dtype=np.int32).tobytes())
                fh.write(np.ascontiguousarray(buf).tobytes())
            print(f"{key:<24s} {n} x {m}  {f.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
