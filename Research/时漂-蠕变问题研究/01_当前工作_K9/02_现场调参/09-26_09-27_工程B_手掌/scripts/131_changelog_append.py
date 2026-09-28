# -*- coding: utf-8 -*-
"""131_changelog_append：向完整更新日志.md 追加本会话条目；超 100KB 则按规则整档归档。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

MAIN = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\Document\ChangingLog\完整更新日志.md")
ARCH = MAIN.parent / "Archived"
LIMIT = 102400

ENTRY = """
## 2026-09-27 手掌 161747 长保压（419 s）独立验证 v3.4 推荐参数 + 终版 8 项（离线复算，未改源码）

- 新会话 20260927_161747（ADC 显示、算法关、单流 42140 帧/419.3 s）：单台阶 E≈16502、快相 +6.9%、蠕变至 +15.6%、全程不卸载，作为 152928 推荐档的独立验证数据（91 号 Python 观测器离线复算，1 s 中值口径）。
- 结果：现参数（rf.04/τc1=1/conf1.5/soft1.5/cap.025/rsm.03）落点 +1.49 N 且 10s→末显示再爬 +0.58 N；推荐档（rf.06/τc1=2/conf2/soft1.5/cap.005/rsm.15）落点 +0.76 N、10 s 起显示冻结（−0.03 N）、下坠 0.16 N、过减 0。尾漂验证：x2/(rsm·e) 全程 ≤0.32，419 s 远未撞幅度顶，本类按压 rsm 不必升 0.6（升后落点仅 0.76→0.75）。
- 终版推荐（寄存器 201~208 = 6/200/200/150/50/600/5/15，即 rf=.06 τc1=2 conf=2 soft=1.5 τrsi=0.5 τr1=6 cap=.005 rsm=.15）；准度优先备选 201=8（落点 +0.4~0.5 N，下坠 ~0.3 N、恒压下坠 ~0.95 N）；rf 兑换律 ~1:1.15 复核成立。
- 产物：temp\\手掌_161747_长保压验证与终版参数.md、temp\\figures\\手掌_161747_长保压对照.png、temp\\palm8\\out\\{streams.npz,129_tune2.json}；脚本 temp\\scripts\\{128,129,130}_palm8*.py。未改任何源码、未构建、未跑测试。
"""


def split_entries(text: str):
    idxs = [m.start() for m in re.finditer(r"(?m)^## ", text)]
    if not idxs:
        return text, []
    header = text[: idxs[0]]
    ents = []
    for a, b in zip(idxs, idxs[1:] + [len(text)]):
        ents.append(text[a:b])
    return header, ents


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    raw = MAIN.read_bytes().decode("utf-8")
    raw = raw.rstrip("\n") + "\n" + ENTRY
    size = len(raw.encode("utf-8"))
    print(f"追加后大小={size} 上限={LIMIT}")
    if size <= LIMIT:
        MAIN.write_bytes(raw.encode("utf-8"))
        print("未超限，直接写入")
        return 0
    header, ents = split_entries(raw)
    print(f"共 {len(ents)} 条 `## ` 条目")
    # 从最旧端累计 90~110KB
    acc, cut = 0, 0
    for i, e in enumerate(ents):
        b = len(e.encode("utf-8"))
        if i < len(ents) - 1 and acc + b > 110 * 1024:
            break
        acc += b
        cut = i + 1
        if acc >= 90 * 1024 and i < len(ents) - 2:
            break
    mig, keep = ents[:cut], ents[cut:]
    print(f"迁出 {len(mig)} 条 {acc} 字节；主文件保留 {len(keep)} 条（含新条目）")
    dates = []
    for e in mig:
        m = re.match(r"## (\d{4}-\d{2}-\d{2})", e)
        if m:
            dates.append(m.group(1))
    d0, d1 = (min(dates), max(dates)) if dates else ("无日期条目", "无日期条目")
    name = f"完整更新日志_{d0}至{d1}.md"
    dest = ARCH / name
    n = 2
    while dest.exists():
        dest = ARCH / f"完整更新日志_{d0}至{d1}_{n}.md"
        n += 1
    body = "".join(mig)
    dest.write_text(
        f"# 完整更新日志 归档（{d0} 至 {d1}）\n\n"
        f"> 归档日期 2026-09-27：自主文件按「每档约 100KB」规则整批迁出 {len(mig)} 条（{acc} 字节），内容逐字未改。\n\n"
        + body, encoding="utf-8")
    new_main = header.rstrip("\n") + "\n\n" + "".join(keep)
    note = (f"> 归档（2026-09-27）：主文件写入新条目后达 {size} 字节、超 100KB（102400 字节）上限；"
            f"按「每档约 100KB」规则从最旧端整批迁出 {d0} ~ {d1} 的 {len(mig)} 条"
            f"（{acc} 字节）至 Archived/{dest.name}，主文件保留其后 {len(keep)} 条。归档清单见 Archived/归档索引.md。\n")
    first_note = header.find("\n> 归档")
    if first_note == -1:
        header2 = header.rstrip("\n") + "\n\n" + note
    else:
        header2 = header.rstrip("\n") + "\n" + note
    new_main = header2.rstrip("\n") + "\n\n" + "".join(keep)
    MAIN.write_text(new_main, encoding="utf-8")
    # 更新归档索引
    idx = ARCH / "归档索引.md"
    old = idx.read_text(encoding="utf-8") if idx.exists() else "# 归档索引\n"
    idx.write_text(old.rstrip("\n") + f"\n- {dest.name}（{d0} 至 {d1}，{acc} 字节，2026-09-27 归档）\n",
                   encoding="utf-8")
    print(f"归档={dest.name} 主文件新大小={len(new_main.encode('utf-8'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
