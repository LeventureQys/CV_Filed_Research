# -*- coding: utf-8 -*-
"""触觉阵列抗蠕变漂移分析 · 录制数据结构化盘点（只读，产出 CSV + 控制台汇总）。

目标：对 temp/ 下 11 组录制做统一口径的清单统计，重点核实时间戳重复率，
并为后续算法评测标注哪些字段可靠。

统计口径（与任务约定严格一致）：
  1. n_channels = session.json 的 devices[0].data_points；
     交叉核对：CSV 中 ch* 列数、layout_mask 中 '1' 的个数。
     另记 rows*cols 与 layout_mask 长度（物理格点总数，可能 != 激活通道数）。
  2. file_frames = CSV 数据行数（pd.read_csv(skiprows=24) 后的行数）。
  3. 时间轴一律用 `timestamp`（绝对秒，小数多）；t = timestamp - timestamp[0] 得相对秒。
     `elapsed` 量化到 1ms、存在大量重复，只用于对比统计，不用于时间轴。
  4. median_dt_ms = np.median(np.diff(timestamp)) * 1000。
  5. dup_timestamp_frames = np.sum(np.diff(timestamp) <= 0)；
     dup_ratio = dup_timestamp_frames / (file_frames - 1)（分母=diff 元素个数）；
     elapsed_dup_frames = np.sum(np.diff(elapsed) <= 0)（elapsed 单调不减，即 diff==0 的重复帧数）。
     本列是本次盘点最关键列：分别对 timestamp / elapsed 给出准确数字。
  6. value_min / value_max 按全部 ch* 列统计（跨通道全局最值）。
  7. main_channel = 负载段均值 − 前空载段均值 最大的通道；main_amp = 该差值。
  8. 负载段划分（与既有 find_segment 口径一致）：
     total = 所有通道之和；thr = 0.15 * max(total)；loaded = total > thr；
     取最长连续 loaded 区间为 [s0, s1)（s1 为开区间，即最后一个受载帧下标 +1）。
     baseline_pre_level = mean(total[0:s0])；load_plateau_level = mean(total[s0:s1])。
     load_onset_s = t[s0]；load_end_s = t[s1-1]（最后一个受载帧的时间）。
  9. raw_drift_pct = 主通道「负载段末 10% 均值 − 首 10% 均值」/ main_amp × 100。
"""
import os
import json
import sys
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))          # temp/v4.1flash/scripts
OUT = os.path.dirname(HERE)                                # temp/v4.1flash
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))                                # temp
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)

# (location, dataset) —— 11 组录制，路径约定：temp/<location>/<dataset>/device_001_seg000.csv
RECORDINGS = [
    ("右拇指指尖", "数据1"),
    ("右拇指指尖", "数据2"),
    ("右拇指指尖", "数据3"),
    ("左拇指指尖", "数据1"),
    ("左拇指指尖", "数据2"),
    ("左拇指指尖", "数据3"),
    ("四指指尖", "数据1"),
    ("四指指尖", "数据2"),
    ("四指指尖", "数据3"),
    ("变化负载", "零负载-切换负载-零负载-再切换负载"),
    ("变化负载", "零负载-中途切换负载-零负载-切换负载"),
]

COLUMNS = [
    "location", "dataset", "rows", "cols", "n_channels", "file_frames",
    "timestamp_t0", "timestamp_tend", "duration_s", "median_dt_ms",
    "dup_timestamp_frames", "dup_ratio", "elapsed_dup_frames",
    "value_min", "value_max", "main_channel", "main_amp",
    "load_onset_s", "load_end_s", "load_duration_s",
    "baseline_pre_level", "load_plateau_level", "raw_drift_pct",
]


def load_recording(csv_path):
    """读取 CSV + session.json，返回 (t, t_raw, elapsed, X, ch_cols, meta)。"""
    df = pd.read_csv(csv_path, skiprows=24)  # 前 24 行元数据 + ##Data，第 25 行为列名
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    t_raw = df["timestamp"].to_numpy(dtype=float)
    t = t_raw - t_raw[0]
    elapsed = df["elapsed"].to_numpy(dtype=float)
    X = df[ch_cols].to_numpy(dtype=float)
    with open(os.path.join(os.path.dirname(csv_path), "session.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return t, t_raw, elapsed, X, ch_cols, meta


def find_all_segments(total, frac=0.15):
    """返回 total > frac*max(total) 的所有连续区间 [(s0, s1), ...]，s1 为开区间。

    与既有 find_segment 完全同口径，只是返回全部区间而非仅最长区间。
    """
    thr = frac * total.max()
    loaded = total > thr
    d = np.diff(loaded.astype(int))
    s = np.where(d == 1)[0] + 1
    e = np.where(d == -1)[0] + 1
    if loaded[0]:
        s = np.r_[0, s]
    if loaded[-1]:
        e = np.r_[e, len(loaded)]
    segs = sorted(zip(s.tolist(), e.tolist()), key=lambda z: z[0])
    return segs


def main():
    rows_out = []
    cross_checks = []       # (location, dataset, message)
    layout_groups = {}      # (rows, cols, layout_mask) -> [(location, dataset), ...]
    varying_segments = {}   # dataset_name -> list of segments

    for location, dataset in RECORDINGS:
        csv_path = os.path.join(TEMP, location, dataset, "device_001_seg000.csv")
        t, t_raw, elapsed, X, ch_cols, meta = load_recording(csv_path)
        dev = meta["devices"][0]

        rows = int(dev["rows"])
        cols = int(dev["cols"])
        data_points = int(dev["data_points"])
        layout_mask = str(dev.get("layout_mask", ""))
        n_channels = data_points
        n_ch_csv = len(ch_cols)
        n_ones = layout_mask.count("1")
        mask_len = len(layout_mask)
        rows_times_cols = rows * cols
        file_frames = int(len(t_raw))

        # ---- 交叉核对（口径 1） ----
        layout_groups.setdefault((rows, cols, layout_mask), []).append((location, dataset))
        if not (n_channels == n_ch_csv == n_ones):
            cross_checks.append(
                (location, dataset,
                 f"data_points={n_channels} / ch列数={n_ch_csv} / mask'1'数={n_ones} 不一致"))
        if mask_len != rows_times_cols:
            cross_checks.append(
                (location, dataset,
                 f"rows*cols={rows_times_cols} != layout_mask长度={mask_len}"))
        elif n_channels != rows_times_cols:
            # 这是预期情况：物理格点数(rows*cols) > 激活通道数(data_points)，mask 中有 '0' 空洞
            pass

        # ---- 时间轴（口径 3/4/5） ----
        t0 = float(t_raw[0])
        tend = float(t_raw[-1])
        duration_s = float(tend - t0)
        dt = np.diff(t_raw)
        median_dt_ms = float(np.median(dt) * 1000.0)
        dup_timestamp_frames = int(np.sum(dt <= 0))
        dup_ratio = float(dup_timestamp_frames) / float(max(1, file_frames - 1))
        elapsed_dup_frames = int(np.sum(np.diff(elapsed) <= 0))

        # ---- 值域（口径 6） ----
        value_min = float(X.min())
        value_max = float(X.max())

        # ---- 负载段划分（口径 8，与 find_segment 一致） ----
        total = X.sum(axis=1)
        segs = find_all_segments(total, frac=0.15)
        if not segs:
            cross_checks.append((location, dataset, "未找到 total>0.15*max 的负载段"))
            s0, s1 = 0, file_frames
        else:
            s0, s1 = max(segs, key=lambda z: z[1] - z[0])  # 最长连续区间
        load_onset_s = float(t[s0])
        load_end_s = float(t[s1 - 1])
        load_duration_s = float(load_end_s - load_onset_s)
        baseline_pre_level = float(total[:s0].mean()) if s0 > 0 else float("nan")
        load_plateau_level = float(total[s0:s1].mean())

        # ---- 主通道（口径 7） ----
        load_mean = X[s0:s1].mean(axis=0)
        pre_mean = X[:s0].mean(axis=0) if s0 > 0 else np.zeros_like(load_mean)
        step = load_mean - pre_mean
        main_idx = int(np.argmax(step))
        main_channel = ch_cols[main_idx]
        main_amp = float(step[main_idx])

        # ---- 原始时漂（口径 9） ----
        y = X[s0:s1, main_idx]
        n10 = max(1, int(0.1 * len(y)))
        first10 = float(y[:n10].mean())
        last10 = float(y[len(y) - n10:].mean())
        if abs(main_amp) > 1e-12:
            raw_drift_pct = float((last10 - first10) / main_amp * 100.0)
        else:
            raw_drift_pct = float("nan")

        rows_out.append({
            "location": location, "dataset": dataset,
            "rows": rows, "cols": cols, "n_channels": n_channels,
            "file_frames": file_frames,
            "timestamp_t0": t0, "timestamp_tend": tend,
            "duration_s": duration_s, "median_dt_ms": median_dt_ms,
            "dup_timestamp_frames": dup_timestamp_frames, "dup_ratio": dup_ratio,
            "elapsed_dup_frames": elapsed_dup_frames,
            "value_min": value_min, "value_max": value_max,
            "main_channel": main_channel, "main_amp": main_amp,
            "load_onset_s": load_onset_s, "load_end_s": load_end_s,
            "load_duration_s": load_duration_s,
            "baseline_pre_level": baseline_pre_level,
            "load_plateau_level": load_plateau_level,
            "raw_drift_pct": raw_drift_pct,
        })

        # ---- 变化负载：记录全部分段（含电平） ----
        if location == "变化负载":
            varying_segments[dataset] = [
                (float(t[a]), float(t[b - 1]), float(b - a) / max(1.0, file_frames - 1),
                 float(total[a:b].mean()))
                for a, b in segs
            ]

    df = pd.DataFrame(rows_out, columns=COLUMNS)
    df.to_csv(os.path.join(RES, "dataset_inventory.csv"), index=False, encoding="utf-8-sig")

    # ---- 控制台输出 ----
    print("=" * 118)
    print("触觉阵列抗蠕变漂移 · 录制数据盘点（11 组）")
    print("=" * 118)
    print()
    print(">>> 交叉核对（口径 1）：data_points vs CSV ch 列数 vs layout_mask '1' 个数")
    if cross_checks:
        for loc, ds, msg in cross_checks:
            print(f"  [检查] {loc}/{ds}: {msg}")
    else:
        print("  全部 11 组：data_points == CSV ch 列数 == layout_mask '1' 个数（三者一致）")
    print()

    # 布局分组
    print(">>> 布局分组（rows×cols + layout_mask 相同即视为同一 layout）")
    for key, members in layout_groups.items():
        r, c, mask = key
        print(f"  {r}×{c} (激活={mask.count('1')}, mask长={len(mask)}) mask={mask}")
        for loc, ds in members:
            print(f"      - {loc}/{ds}")
    print()

    # 变化负载分段结构
    print(">>> 变化负载分段结构（total > 0.15*max(total) 的全部连续区间）")
    for ds, segs in varying_segments.items():
        print(f"  [{ds}]")
        for i, (a, b, fr, lvl) in enumerate(segs, 1):
            print(f"     段{i}: onset={a:8.3f}s  end={b:8.3f}s  时长={b-a:7.3f}s  "
                  f"帧占比={fr*100:5.1f}%  电平(mean total)={lvl:9.1f}")
    print()

    # 汇总表（可读性优先：宽表全字段）
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 400)
    pd.set_option("display.max_colwidth", 26)
    disp = df.copy()
    for c in ("timestamp_t0", "timestamp_tend", "duration_s", "median_dt_ms",
              "load_onset_s", "load_end_s", "load_duration_s"):
        disp[c] = disp[c].round(3)
    for c in ("dup_ratio",):
        disp[c] = disp[c].round(6)
    for c in ("value_min", "value_max", "main_amp", "baseline_pre_level",
              "load_plateau_level", "raw_drift_pct"):
        disp[c] = disp[c].round(4)
    print(">>> 汇总表（全字段）")
    print(disp.to_string(index=False))
    print()
    print(f">>> 已写入: {os.path.join(RES, 'dataset_inventory.csv')}")

    # 关键结论提示
    print()
    print(">>> 关键结论（时间戳重复率核实）")
    print("  dup_timestamp_frames (diff(timestamp)<=0): 见表，应接近 0（timestamp 微秒级严格递增）")
    print("  elapsed_dup_frames (diff(elapsed)<=0): 见表，elapsed 量化到 1ms 导致大量重复")
    for r in rows_out:
        et = r["elapsed_dup_frames"] / max(1, r["file_frames"] - 1)
        print(f"  {r['location']}/{r['dataset']}: ts重复={r['dup_timestamp_frames']}"
              f" ({r['dup_ratio']*100:.2f}%)  elapsed重复={r['elapsed_dup_frames']}"
              f" ({et*100:.2f}%)")


if __name__ == "__main__":
    main()
