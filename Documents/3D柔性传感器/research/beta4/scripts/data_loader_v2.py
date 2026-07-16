from __future__ import annotations

import csv
import importlib
import os
import re
import json
from pathlib import Path
from typing import Optional

import numpy as np

_this_dir = Path(__file__).resolve().parent
_research_src = str(_this_dir.parents[2] / "src")
if _research_src not in os.sys.path:
    os.sys.path.insert(0, _research_src)
_old_loader = importlib.import_module("alg.data_loader")
load_old_csv = _old_loader.load_csv


def _parse_new_header(path: str) -> dict:
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_data = False
        for row in reader:
            if not row:
                continue
            if row[0] == "##Data":
                in_data = True
                continue
            if in_data:
                break
            if row[0].startswith("##"):
                continue
            if len(row) < 2:
                continue
            key = row[0].strip()
            val = row[1].strip()
            meta[key] = val
    return meta


def _parse_new_data(path: str, max_frames: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    times = []
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        in_data = False
        for row in reader:
            if not row:
                continue
            if row[0] == "##Data":
                in_data = True
                header_row = next(reader, None)
                if not header_row:
                    break
                col_indices = []
                for i, name in enumerate(header_row):
                    name = name.strip()
                    if name.startswith("ch") and name[2:].isdigit():
                        col_indices.append(i)
                continue
            if in_data:
                if max_frames is not None and len(times) >= max_frames:
                    break
                try:
                    elapsed = float(row[1])
                    vals = [float(row[i]) for i in col_indices]
                except (ValueError, IndexError):
                    continue
                times.append(elapsed)
                frames.append(vals)
    return np.array(times, dtype=np.float64), np.array(frames, dtype=np.float64)


def _guess_freq_from_elapsed(times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    diffs = np.diff(times)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    median_diff = float(np.median(diffs))
    if median_diff <= 0:
        return 0.0
    return 1.0 / median_diff


def load_old_format(path: str, max_frames: Optional[int] = None) -> tuple[np.ndarray, dict]:
    return load_old_csv(path, max_frames=max_frames)


def load_new_format(session_dir: str, max_frames: Optional[int] = None) -> tuple[np.ndarray, dict]:
    session_dir = Path(session_dir)
    csv_path = session_dir / "device_001.csv"
    json_path = session_dir / "session.json"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing device CSV: {csv_path}")

    meta_raw = _parse_new_header(str(csv_path))
    times, frames = _parse_new_data(str(csv_path), max_frames=max_frames)

    rows = int(meta_raw.get("rows", 0))
    cols = int(meta_raw.get("cols", 0))

    meta = {
        "rows": rows,
        "cols": cols,
        "channels": frames.shape[1],
        "n_frames": frames.shape[0],
        "path": csv_path.name,
        "device_label": f"fabric {rows}x{cols}",
        "time_elapsed": times,
    }

    freq_str = meta_raw.get("record_frequency", "0")
    try:
        hz = float(freq_str) if freq_str and float(freq_str) > 0 else 0.0
    except ValueError:
        hz = 0.0

    if hz <= 0 and len(times) > 1:
        hz = _guess_freq_from_elapsed(times)

    meta["sample_freq_hz"] = max(hz, 0.1)
    return frames, meta


def is_old_format(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    return "设备信息" in first


def load_recording(path_or_dir: str, max_frames: Optional[int] = None) -> tuple[np.ndarray, dict]:
    p = Path(path_or_dir)
    if p.is_dir():
        csv_files = sorted(p.glob("device_*.csv"))
        if csv_files:
            return load_new_format(str(p), max_frames=max_frames)
        csv_files = sorted(p.glob("*.csv"))
        if csv_files:
            return load_old_format(str(csv_files[0]), max_frames=max_frames)
        raise FileNotFoundError(f"No CSV found in {path_or_dir}")

    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path_or_dir}")

    if is_old_format(str(p)):
        return load_old_format(str(p), max_frames=max_frames)
    else:
        parent_dir = p.parent
        session_dir = parent_dir.parent if parent_dir.name.startswith("2026") else parent_dir
        return load_new_format(str(session_dir), max_frames=max_frames)
