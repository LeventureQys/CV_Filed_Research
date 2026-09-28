# -*- coding: utf-8 -*-
"""v2.0 · 测试夹具（fixture）装配脚本：从实机录制中裁出 drift_v6 单元测试用的小样本。

为什么需要它：
  v6 的两个缺陷（① 加载瞬态超前量 ② 长时偏移逐周期累积）都只在**实机时序**下复现，
  纯合成信号无法覆盖。完整录制单个文件 8.9 MB，不适合进仓，故裁出三段关键窗口：

    seg_a_onset     0.00 ~   50.00 s   首次加载 + 1 kg 保压开头（瞬态过充 + 早期偏移）
    seg_b_handoff  39.00 ~   50.00 s   抬放 500 g 的复合事件（44 s 事件 + Handoff 钉值）
    seg_c_cycle   105.00 ~  152.00 s   完整卸载 → 再加载 → 短保压（偏移归零与再累积）

输出（每段两个文件，均为原格式、原列、原**相对**时间轴，只裁剪行并按 `--stride` 抽帧）：
    <out>/seg_<id>_pre.csv    算法输入（= device_001_pre_seg0.csv 的切片）
    <out>/seg_<id>_main.csv   算法输出（= device_001_seg000.csv 的切片，用于对照）

抽帧（`--stride`，默认 5）：原始 100.47 Hz → 约 20 Hz，用于控制进仓体积。
**20 Hz 夹具只用于"行为不回归"的单元测试**：所有按帧计的常量（`kDetPersist=3`、滑行 0.4~0.8 s）
在 20 Hz 下语义不变，但**绝对指标（T_stable/过充量）不可与 100 Hz 录制对表**；
需要绝对量时必须用全分辨率录制 + `temp/v4.1flash/plan/v2.0/scripts/probe_*.py`。

用法：
    python temp\\v4.1flash\\plan\\v2.0\\scripts\\make_drift_v6_fixture.py            # 写入 tests/fixtures/drift_v6
    python temp\\v4.1flash\\plan\\v2.0\\scripts\\make_drift_v6_fixture.py --out <dir>
    python temp\\v4.1flash\\plan\\v2.0\\scripts\\make_drift_v6_fixture.py --stride 1   # 全分辨率
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SEGS = [
    ("a_onset", 0.0, 50.0),
    ("b_handoff", 39.0, 50.0),
    ("c_cycle", 105.0, 152.0),
]

DEFAULT_OUT = os.path.join(L.ROOT, "tests", "fixtures", "drift_v6")


def slice_stream(stream, t0, t1, out_path, stride, extra_cfg=None):
    el = stream["el"]
    keep = [i for i in range(stream["n"]) if t0 <= el[i] <= t1]
    if not keep:
        raise SystemExit(f"空切片: {out_path}")
    # 直接复用原始行：重新读文件以保留原始字面值（避免浮点重排）
    src = stream["_path"]
    with open(src, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n").rstrip("\r") for r in fh]
    di = rows.index("##Data")
    head = rows[:di + 2]
    body = [r for r in rows[di + 2:] if r.strip()]
    lines = [body[i] for i in keep[::stride]]
    # 头部的 session-type 字段与裁剪无关，原样保留（下游按 ##Data 定位）
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(head + lines) + "\n")
    return len(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--dataset", default=L.DS_ZERO)
    ap.add_argument("--stride", type=int, default=5,
                    help="抽帧步长（1 = 全分辨率 100.47 Hz；默认 5 = 约 20 Hz）")
    args = ap.parse_args()
    assert args.stride >= 1

    d = L.load_dataset(args.dataset)
    pre, main = d["pre"], d["main"]
    pre["_path"] = os.path.join(args.dataset, "device_001_pre_seg0.csv")
    main["_path"] = os.path.join(args.dataset, "device_001_seg000.csv")

    manifest = []
    for sid, t0, t1 in SEGS:
        np_ = slice_stream(pre, t0, t1, os.path.join(args.out, f"seg_{sid}_pre.csv"),
                           args.stride)
        nm = slice_stream(main, t0, t1, os.path.join(args.out, f"seg_{sid}_main.csv"),
                          args.stride)
        assert np_ == nm, (sid, np_, nm)
        manifest.append((sid, t0, t1, np_))
        print(f"seg_{sid}: t=[{t0}, {t1}]  帧数={np_}（stride={args.stride}）  → {args.out}")

    fs = 100.47 / args.stride
    readme = os.path.join(args.out, "README.md")
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write(
            "# drift_v6 单元测试夹具（自动生成，勿手改）\n\n"
            f"来源录制：`{os.path.relpath(args.dataset, L.ROOT)}`\n"
            "（33908 帧 / 337.5 s / 100.47 Hz / 21 通道 / 显示域 adc / 参数集 plan-v1.0 A1+A4+A5a）\n\n"
            f"生成脚本：`temp/v4.1flash/plan/v2.0/scripts/make_drift_v6_fixture.py --stride {args.stride}`\n"
            f"**帧率：约 {fs:.2f} Hz（抽帧步长 {args.stride}）** ⇒ 只用于「行为不回归」的单元测试，\n"
            "绝对指标（T_stable / 过充量）不可与 100 Hz 录制对表。\n\n"
            "| 段 | 时间窗 (s) | 帧数 | 用途 |\n|---|---|---|---|\n"
            "| `seg_a_onset` | 0.00 ~ 50.00 | %d | 首次加载瞬态（过充）+ 1 kg 保压开头 |\n"
            "| `seg_b_handoff` | 39.00 ~ 50.00 | %d | 抬放 500 g 复合事件（44 s 处 + Handoff） |\n"
            "| `seg_c_cycle` | 105.00 ~ 152.00 | %d | 卸载 → 再加载 → 短保压（偏移归零与再累积） |\n\n"
            "文件格式与实机录制完全一致（含 `##Session` 头与 `##Data` 表头）；\n"
            "**`*_pre.csv` 的 `##Data` 表头缺 21 个通道名（已知交付缺陷），测试代码必须按列位置解析。**\n"
            "列布局（**0-based 索引**）：`0=timestamp, 1=elapsed, 2=frame_index, 3..23=ch0..ch20`（共 24 列）。\n\n"
            "## ⚠️ 使用限制（SubStage 1B/2A/2B 实测，必须遵守）\n\n"
            "1. **冷启动限制**：补偿器首帧会 `ResetFor` ⇒ 需要 ≥ ~40 s 前置历史才会产生事件与补偿。\n"
            "   因此 **`seg_b_handoff`（221 帧 / 11 s）在冷启动重放下全部走直通出口**（输出 ≡ 原始），\n"
            "   ⇒ **不要用它验限幅的触发、保压偏移或任何补偿量**，它只适合验证测试解析器与片段内容参考。\n"
            "   需要量测时用 `seg_a_onset` / `seg_c_cycle`，或先喂一段前置历史（0~39 s）再量。\n"
            "2. **末帧状态**：`seg_a_onset` 与 `seg_c_cycle` 是「事件进行中」的切片，跑完后 `in_event() == true` 属预期\n"
            "   （`seg_a` 的加载沿在**第 45 帧**，不是「前 100 帧空载」）。\n"
            "3. **口径**：`seg_*_main.csv` 才是完整的「事件 → 保压」段；量测保压偏移时以**总量（21 通道求和）**为准，\n"
            "   逐通道口径的数值完全不同（差约 50 倍），两种口径不得混报。\n"
            % tuple(m[3] for m in manifest))
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
