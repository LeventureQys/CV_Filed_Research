# -*- coding: utf-8 -*-
"""17 出图自检：图片文件、尺寸、dptool 结构自检（空面板/越界/重叠）+ 图源数据数值抽查。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from sensor_common import DPTOOL_ROOT, OUT_DIR  # noqa: E402
from sweep_lib import PREP  # noqa: E402

FIG = OUT_DIR / "figures"


def png_size(p: Path):
    b = p.read_bytes()
    # PNG IHDR: 宽高在偏移 16..24（大端）
    w = int.from_bytes(b[16:20], "big")
    h = int.from_bytes(b[20:24], "big")
    return w, h


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    sys.path.insert(0, str(DPTOOL_ROOT))
    from dptool import api  # noqa: E402

    files = sorted(FIG.glob("*.png"))
    print(f"共 {len(files)} 张图：\n")
    bad = []
    for p in files:
        w, h = png_size(p)
        sz = p.stat().st_size / 1024
        flag = "" if (w > 800 and sz > 20) else "  ← 可疑"
        if flag:
            bad.append(p.name)
        print(f"  {p.name:<58s} {w}x{h}  {sz:7.1f} KB{flag}")

    # dptool 结构自检：拿 A 组一张图的数据重画一次（check=True）
    print("\n—— dptool 结构自检（重画 A 组首图）——")
    g = OUT_DIR / "_fig" / "四指指腹" / "d1_6a679f"
    dirs = [str(g / n) for n in sorted(x.name for x in g.iterdir())]
    res = api.plot_to_file(dirs, str(OUT_DIR / "17_出图结构自检示例.png"), mode="overlay",
                           series_by="dir", series_stream="out", signal="sum",
                           dpi=110, check=True, suptitle="结构自检（A 组示例）")
    chk = res.get("check")
    print(f"  ok={res.get('path') is not None}  n_panels={res.get('n_panels')} "
          f"curves={res.get('curves')}")
    print(f"  check={json.dumps(chk, ensure_ascii=False)}")

    # 数值抽查：B 组（右手掌/d1_A）三条曲线末值应为 输入≈100%、现役 86.2%、推荐 59.5%
    print("\n—— 图源数值抽查（右手掌/d1_A 残差图，单位 % 蠕变）——")
    gb = OUT_DIR / "_fig" / "B_右手掌" / "d1_A"
    for d in sorted(gb.iterdir()):
        arr = np.loadtxt(d / "device_001_seg000.csv", delimiter=",", skiprows=10,
                         encoding="utf-8")
        print(f"  {d.name:<22s} 末值={arr[-1, 3]:7.1f}  最小={arr[:, 3].min():7.1f}  "
              f"最大={arr[:, 3].max():7.1f}")
    print(f"  参考：该会话 Esum={PREP['右手掌/d1/A']['Esum_channels']:.1f} "
          f"蠕变={PREP['右手掌/d1/A']['creep_total']:.1f}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
