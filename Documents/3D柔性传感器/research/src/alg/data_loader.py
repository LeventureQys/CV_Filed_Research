"""
data_loader.py — Load TactileSense CSV recording files into numpy arrays.

CSV format (from SessionRecorder):
  ##设备信息
  设备名称,...
  行数,<rows>
  列数,<cols>
  数据点数,<N>
  ...
  录制频率,<raw|N Hz>
  ##数据开始
  时间戳,通道1,...,通道96,总值,最小值,最大值,...
  <ts>,<v1>,...,<v96>,...
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np


def parse_csv_header(path: str) -> dict:
    """Parse the ##设备信息 header block.

    Returns
    -------
    dict with keys: device_name, rows, cols, data_points, display_mode,
                    recording_freq_str, recording_time_str
    """
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == "##数据开始":
                break
            if row[0].startswith("##"):
                continue
            if len(row) < 2:
                continue
            key = row[0].strip()
            val = row[1].strip()
            meta[key] = val
    return meta


def guess_sampling_freq(path: str) -> float:
    """Estimate median sampling frequency from timestamps.

    Returns
    -------
    Hz (float)
    """
    timestamps = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_data = False
        for row in reader:
            if not row:
                continue
            if row[0] == "##数据开始":
                in_data = True
                header_row = row
                continue
            if in_data and len(row) > 0 and row[0]:
                timestamps.append(row[0])

    if len(timestamps) < 2:
        return 0.0

    # Parse "14时00分27秒.924"
    def parse_ts(s: str) -> float:
        m = re.match(r"(\d+)时(\d+)分(\d+)秒\.?(\d*)", s)
        if not m:
            return 0.0
        h, m_, s_, ms = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
        ms = int(ms) if ms else 0
        return h * 3600.0 + m_ * 60.0 + s_ + ms / (10 ** len(str(ms)))

    t_sec = np.array([parse_ts(t) for t in timestamps], dtype=np.float64)
    diffs = np.diff(t_sec)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    median_diff = float(np.median(diffs))
    if median_diff <= 0:
        return 0.0
    return 1.0 / median_diff


def load_csv(path: str, max_frames: Optional[int] = None) -> tuple[np.ndarray, dict]:
    """Load a CSV recording file.

    Returns
    -------
    frames : ndarray, shape (N, C)
        ADC values. Only the per-channel columns (channel1..channelC) are kept;
        summary columns (总值,最小值,...) are discarded.
    meta : dict
        Parsed header metadata.
    """
    meta = parse_csv_header(path)
    rows_count = int(meta.get("行数", 0))
    cols_count = int(meta.get("列数", 0))

    frames = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_data = False
        data_cols = None
        for row in reader:
            if not row:
                continue
            if row[0] == "##数据开始":
                in_data = True
                # The actual column header is on the NEXT row
                try:
                    header_row = next(reader)
                except StopIteration:
                    header_row = []
                data_cols = []
                for col_name in header_row[1:]:
                    col_name = col_name.strip()
                    if col_name.startswith("通道"):
                        data_cols.append(col_name)
                    else:
                        break
                continue
            if in_data:
                if max_frames is not None and len(frames) >= max_frames:
                    break
                # Skip rows that don't have enough columns
                if len(row) < len(data_cols):
                    continue
                if len(row) > len(data_cols) + 1:
                    vals = row[1:1 + len(data_cols)]
                else:
                    vals = row[1:]
                try:
                    float_vals = [float(v) for v in vals]
                except (ValueError, IndexError):
                    continue
                frames.append(float_vals)

    frames_arr = np.array(frames, dtype=np.float64)
    meta["rows"] = rows_count
    meta["cols"] = cols_count
    meta["channels"] = frames_arr.shape[1]
    meta["n_frames"] = frames_arr.shape[0]

    # Estimate freq
    freq_str = meta.get("录制频率", "raw")
    if freq_str and freq_str.lower() != "raw":
        try:
            hz = float(freq_str.replace("Hz", "").strip())
        except ValueError:
            hz = 0.0
    else:
        hz = guess_sampling_freq(path)

    meta["sample_freq_hz"] = hz
    meta["path"] = os.path.basename(path)
    meta["device_label"] = f"{meta.get('设备名称', '?')} {rows_count}x{cols_count}"

    return frames_arr, meta


def load_dataset(base_dir: str, max_frames_per_file: Optional[int] = None):
    """Load all CSV files in a directory tree.

    Returns
    -------
    list of (frames, meta) for each CSV file found.
    """
    results = []
    for root, dirs, files in os.walk(base_dir):
        for fn in files:
            if fn.endswith(".csv"):
                path = os.path.join(root, fn)
                try:
                    frames, meta = load_csv(path, max_frames=max_frames_per_file)
                    results.append((frames, meta))
                except Exception as e:
                    print(f"[warn] Skipping {path}: {e}")
    return results
