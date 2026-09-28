"""
诊断脚本：分析「零基线-反复增减同一负载」数据集
重点：对比 ~15s（首次加载，基线正确）与 ~245s（后续加载，基线丢失）的差异

输出到 temp/v4.1flash/plan/v3.3/results/
"""

import csv
import os
import sys
import json
import numpy as np
from pathlib import Path

REPO = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp")
DATA_DIR = REPO / "temp" / "算法数据&原始数据" / "working" / "零基线-反复增减同一负载" / "20260919_160854_single_device_7b3977"
OUT_DIR = REPO / "temp" / "v4.1flash" / "plan" / "v3.3" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(path, header_keyword="##Data"):
    """Load CSV with metadata header, return (header_cols, data_rows_as_float)"""
    rows = []
    header = None
    in_data = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == header_keyword:
                in_data = True
                continue
            if in_data:
                if header is None:
                    header = line.split(",")
                    continue
                parts = line.split(",")
                try:
                    vals = [float(x) for x in parts]
                    rows.append(vals)
                except ValueError:
                    continue
    return header, np.array(rows) if rows else np.empty((0,0))


def main():
    print("=== 加载数据 ===")
    
    # Load algorithm input (pre) and algorithm result (seg)
    hdr_pre, data_pre = load_csv(DATA_DIR / "device_001_pre_seg0.csv")
    hdr_seg, data_seg = load_csv(DATA_DIR / "device_001_seg000.csv")
    hdr_raw, data_raw = load_csv(DATA_DIR / "device_001_raw_seg000.csv")
    
    print(f"  pre (算法输入): {data_pre.shape}, header count: {len(hdr_pre) if hdr_pre else 0}")
    print(f"  seg (算法结果): {data_seg.shape}, header count: {len(hdr_seg) if hdr_seg else 0}")
    print(f"  raw (原始ADC):  {data_raw.shape}, header count: {len(hdr_raw) if hdr_raw else 0}")
    
    # pre文件表头只有3列名(timestamp,elapsed,frame_index)但数据有24列
    # 取通道数据: 列3~23 = ch0..ch20
    elapsed_pre = data_pre[:, 1]
    channels_pre = data_pre[:, 3:24]  # 21 channels, algorithm input
    total_pre = channels_pre.sum(axis=1)
    
    elapsed_seg = data_seg[:, 1]
    channels_seg = data_seg[:, 3:24]  # 21 channels, algorithm result
    total_seg = channels_seg.sum(axis=1)
    
    elapsed_raw = data_raw[:, 1]
    channels_raw = data_raw[:, 3:24]
    total_raw = channels_raw.sum(axis=1)
    
    print(f"\n  时间范围: {elapsed_pre[0]:.1f} ~ {elapsed_pre[-1]:.1f} s")
    print(f"  帧数: {len(elapsed_pre)}")
    print(f"  通道数: {channels_pre.shape[1]}")
    
    # === 1. 全局总量时序概览 ===
    print("\n=== 1. 全局总量时序概览 ===")
    
    # Find step edges by detecting large changes in total
    dt = np.diff(elapsed_pre)
    dtotal = np.diff(total_pre)
    
    # Compute smoothed total for step detection
    win = 10  # ~0.1s at 100Hz
    total_smooth = np.convolve(total_pre, np.ones(win)/win, mode='same')
    
    # Detect loading/unloading events: large changes
    threshold = 500  # ADC total change threshold
    event_times = []
    event_types = []
    
    # Use a sliding window approach
    for i in range(100, len(total_smooth) - 100):
        t = elapsed_pre[i]
        lvl_before = np.mean(total_pre[max(0,i-50):i-10])
        lvl_after = np.mean(total_pre[i+10:min(len(total_pre),i+50)])
        diff = lvl_after - lvl_before
        if abs(diff) > threshold:
            # Check it's not too close to a previous event
            if not event_times or (t - event_times[-1]) > 2.0:
                event_times.append(t)
                event_types.append("LOAD" if diff > 0 else "UNLOAD")
    
    print(f"  检测到 {len(event_times)} 个事件:")
    for t, etype in zip(event_times, event_types):
        idx = np.searchsorted(elapsed_pre, t)
        lvl = total_pre[idx]
        print(f"    t={t:7.1f}s  {etype:8s}  total={lvl:.0f}")
    
    # === 2. 分段分析：对比每次加载后的形状 ===
    print("\n=== 2. 逐事件分析：加载后形状对比 ===")
    
    load_events = [(t, i) for i, (t, tp) in enumerate(zip(event_times, event_types)) if tp == "LOAD"]
    
    results = []
    for evt_idx, (t_load, _) in enumerate(load_events):
        # Find the next event time (or end of data)
        next_evt_time = elapsed_pre[-1]
        for t2 in event_times:
            if t2 > t_load + 1.0:
                next_evt_time = t2
                break
        
        # Get data window: t_load-2s to next_event
        mask = (elapsed_pre >= t_load - 2) & (elapsed_pre < next_evt_time)
        t_win = elapsed_pre[mask] - t_load
        pre_win = total_pre[mask]
        seg_win = total_seg[mask]
        
        # Compute baseline (pre-event level)
        mask_base = (elapsed_pre >= t_load - 2) & (elapsed_pre < t_load - 0.5)
        if mask_base.sum() > 10:
            baseline_pre = np.median(total_pre[mask_base])
            baseline_seg = np.median(total_seg[mask_base])
        else:
            baseline_pre = 0
            baseline_seg = 0
        
        # Compute post-step plateau (5-10s after step)
        mask_plateau = (elapsed_pre >= t_load + 5) & (elapsed_pre < min(t_load + 15, next_evt_time - 1))
        if mask_plateau.sum() > 10:
            plateau_pre = np.median(total_pre[mask_plateau])
            plateau_seg = np.median(total_seg[mask_plateau])
        else:
            plateau_pre = np.nan
            plateau_seg = np.nan
        
        # Step amplitude
        step_pre = plateau_pre - baseline_pre if not np.isnan(plateau_pre) else np.nan
        step_seg = plateau_seg - baseline_seg if not np.isnan(plateau_seg) else np.nan
        
        # Post-step values at various timepoints
        timepoints = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]
        vals_at = {}
        for tp_s in timepoints:
            mask_tp = (elapsed_pre >= t_load + tp_s - 0.1) & (elapsed_pre < t_load + tp_s + 0.1)
            if mask_tp.sum() > 0:
                vals_at[tp_s] = {
                    'pre': np.median(total_pre[mask_tp]),
                    'seg': np.median(total_seg[mask_tp]),
                }
            else:
                vals_at[tp_s] = {'pre': np.nan, 'seg': np.nan}
        
        # 算法输出与baseline的偏差
        if not np.isnan(step_seg) and abs(step_seg) > 100:
            # Normalized deviations at each timepoint relative to step
            deviations = {}
            for tp_s, v in vals_at.items():
                if not np.isnan(v['seg']) and not np.isnan(plateau_seg):
                    dev = (v['seg'] - plateau_seg) / abs(step_seg) * 100  # percentage
                    deviations[tp_s] = dev
        else:
            deviations = {}
        
        result = {
            'event_idx': evt_idx,
            't_load': t_load,
            'baseline_pre': baseline_pre,
            'baseline_seg': baseline_seg,
            'plateau_pre': plateau_pre,
            'plateau_seg': plateau_seg,
            'step_pre': step_pre,
            'step_seg': step_seg,
            'vals_at': vals_at,
            'deviations': deviations,
        }
        results.append(result)
        
        print(f"\n  事件 #{evt_idx}: t_load = {t_load:.1f}s")
        print(f"    基线(pre): {baseline_pre:.0f}  基线(seg): {baseline_seg:.0f}")
        print(f"    台阶(pre): {step_pre:.0f}" if not np.isnan(step_pre) else "    台阶(pre): N/A")
        print(f"    台阶(seg): {step_seg:.0f}" if not np.isnan(step_seg) else "    台阶(seg): N/A")
        if deviations:
            print(f"    显示偏差(相对台阶%):")
            for tp_s, dev in sorted(deviations.items()):
                print(f"      t+{tp_s:.0f}s: {dev:+.1f}%")
    
    # === 3. 深度对比：15s 附近 vs 245s 附近 ===
    print("\n=== 3. 深度对比：首次加载 vs 245s附近加载 ===")
    
    # Find the first load event near 15s and the one near 245s
    first_load = None
    load_245 = None
    for r in results:
        if first_load is None and r['t_load'] < 30:
            first_load = r
        if r['t_load'] > 230 and r['t_load'] < 270:
            load_245 = r
    
    if first_load and load_245:
        print(f"  首次加载: t={first_load['t_load']:.1f}s")
        print(f"  245s加载: t={load_245['t_load']:.1f}s")
        print(f"")
        print(f"  {'指标':<25} {'首次加载':>12} {'245s加载':>12} {'差异':>12}")
        print(f"  {'-'*65}")
        print(f"  {'基线(pre)':<25} {first_load['baseline_pre']:>12.0f} {load_245['baseline_pre']:>12.0f} {load_245['baseline_pre']-first_load['baseline_pre']:>+12.0f}")
        print(f"  {'基线(seg)':<25} {first_load['baseline_seg']:>12.0f} {load_245['baseline_seg']:>12.0f} {load_245['baseline_seg']-first_load['baseline_seg']:>+12.0f}")
        print(f"  {'台阶(pre)':<25} {first_load['step_pre']:>12.0f} {load_245['step_pre']:>12.0f} {load_245['step_pre']-first_load['step_pre']:>+12.0f}")
        print(f"  {'台阶(seg)':<25} {first_load['step_seg']:>12.0f} {load_245['step_seg']:>12.0f} {load_245['step_seg']-first_load['step_seg']:>+12.0f}")
        
        # Baseline drift in seg relative to pre
        drift_first = first_load['baseline_seg'] - first_load['baseline_pre']
        drift_245 = load_245['baseline_seg'] - load_245['baseline_pre']
        print(f"  {'基线偏差(seg-pre)':<25} {drift_first:>12.0f} {drift_245:>12.0f} {drift_245-drift_first:>+12.0f}")
    else:
        print("  未找到匹配的事件对")
        if first_load:
            print(f"  首次加载: t={first_load['t_load']:.1f}s")
        else:
            print("  首次加载: 未找到")
        if load_245:
            print(f"  245s加载: t={load_245['t_load']:.1f}s")
        else:
            print("  245s加载: 未找到，搜索所有加载事件时间:")
            for r in results:
                print(f"    t={r['t_load']:.1f}s")
    
    # === 4. 卸载后的基线恢复分析 ===
    print("\n=== 4. 卸载后基线恢复分析 ===")
    
    unload_events = [(t, i) for i, (t, tp) in enumerate(zip(event_times, event_types)) if tp == "UNLOAD"]
    
    for evt_idx, (t_unload, orig_idx) in enumerate(unload_events):
        # Find the next load event
        next_load_time = elapsed_pre[-1]
        for t2 in event_times:
            if t2 > t_unload + 1.0:
                next_load_time = t2
                break
        
        # Pre-unload level (the plateau)
        mask_before = (elapsed_pre >= t_unload - 5) & (elapsed_pre < t_unload - 0.5)
        if mask_before.sum() > 10:
            level_before_pre = np.median(total_pre[mask_before])
            level_before_seg = np.median(total_seg[mask_before])
        else:
            continue
        
        # Post-unload "baseline" (should return to ~original zero baseline)
        mask_after = (elapsed_pre >= t_unload + 3) & (elapsed_pre < min(t_unload + 10, next_load_time - 1))
        if mask_after.sum() > 10:
            level_after_pre = np.median(total_pre[mask_after])
            level_after_seg = np.median(total_seg[mask_after])
        else:
            continue
        
        # The "original" zero baseline from t=0~5s
        mask_orig = (elapsed_pre >= 0) & (elapsed_pre < 5)
        orig_baseline = np.median(total_pre[mask_orig])
        
        residual_pre = level_after_pre - orig_baseline
        residual_seg = level_after_seg - orig_baseline
        
        print(f"  卸载 #{evt_idx}: t={t_unload:.1f}s")
        print(f"    卸载前(pre): {level_before_pre:.0f}  卸载后(pre): {level_after_pre:.0f}  残留(pre): {residual_pre:+.0f}")
        print(f"    卸载前(seg): {level_before_seg:.0f}  卸载后(seg): {level_after_seg:.0f}  残留(seg): {residual_seg:+.0f}")
        print(f"    原始基线: {orig_baseline:.0f}")
    
    # === 5. 逐帧连续性分析 (200-300s 区间) ===
    print("\n=== 5. 200-300s区间逐帧连续性 ===")
    
    mask_focus = (elapsed_pre >= 200) & (elapsed_pre < 300)
    t_focus = elapsed_pre[mask_focus]
    pre_focus = total_pre[mask_focus]
    seg_focus = total_seg[mask_focus]
    diff_focus = seg_focus - pre_focus
    
    print(f"  帧数: {mask_focus.sum()}")
    print(f"  算法偏差(seg-pre):")
    print(f"    均值: {np.mean(diff_focus):.1f}")
    print(f"    中位: {np.median(diff_focus):.1f}")
    print(f"    std:  {np.std(diff_focus):.1f}")
    print(f"    min:  {np.min(diff_focus):.1f}")
    print(f"    max:  {np.max(diff_focus):.1f}")
    
    # Find where the deviation is largest
    worst_idx = np.argmax(np.abs(diff_focus))
    worst_t = t_focus[worst_idx]
    print(f"  最大偏差时刻: t={worst_t:.1f}s, 偏差={diff_focus[worst_idx]:.0f}")
    
    # === 6. 全时间线的seg vs pre偏差 ===
    print("\n=== 6. 全时间线 seg-pre 偏差分段统计 ===")
    
    diff_all = total_seg - total_pre
    seg_boundaries = list(range(0, int(elapsed_pre[-1]) + 30, 30))
    
    for i in range(len(seg_boundaries) - 1):
        t0 = seg_boundaries[i]
        t1 = seg_boundaries[i+1]
        mask_s = (elapsed_pre >= t0) & (elapsed_pre < t1)
        if mask_s.sum() == 0:
            continue
        d = diff_all[mask_s]
        print(f"  [{t0:3d}, {t1:3d})s: mean={np.mean(d):+8.1f}  std={np.std(d):7.1f}  range=[{np.min(d):+8.0f}, {np.max(d):+8.0f}]")
    
    # === 7. Save detailed per-frame data around 245s for inspection ===
    print("\n=== 7. 保存245s附近详细数据 ===")
    
    mask_detail = (elapsed_pre >= 235) & (elapsed_pre < 275)
    detail_file = OUT_DIR / "detail_around_245s.csv"
    with open(detail_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed", "total_pre", "total_seg", "seg_minus_pre",
                         "total_raw"] + [f"ch{i}_pre" for i in range(21)] + [f"ch{i}_seg" for i in range(21)])
        idxs = np.where(mask_detail)[0]
        for idx in idxs:
            row = [
                f"{elapsed_pre[idx]:.3f}",
                f"{total_pre[idx]:.1f}",
                f"{total_seg[idx]:.1f}",
                f"{total_seg[idx] - total_pre[idx]:.1f}",
                f"{total_raw[idx]:.1f}" if idx < len(total_raw) else "",
            ]
            for ch in range(21):
                row.append(f"{channels_pre[idx, ch]:.1f}")
            for ch in range(21):
                row.append(f"{channels_seg[idx, ch]:.1f}")
            writer.writerow(row)
    print(f"  已保存: {detail_file}")
    
    # === 8. Save full timeline summary ===
    summary_file = OUT_DIR / "full_timeline_summary.csv"
    # Downsample to ~1 row per 0.1s
    step = max(1, len(elapsed_pre) // 3000)
    with open(summary_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed", "total_pre", "total_seg", "seg_minus_pre"])
        for idx in range(0, len(elapsed_pre), step):
            writer.writerow([
                f"{elapsed_pre[idx]:.3f}",
                f"{total_pre[idx]:.1f}",
                f"{total_seg[idx]:.1f}",
                f"{total_seg[idx] - total_pre[idx]:.1f}",
            ])
    print(f"  已保存: {summary_file}")
    
    # === 9. 关键发现总结 ===
    print("\n" + "="*70)
    print("=== 关键发现总结 ===")
    print("="*70)
    
    # Compute per-load-event the baseline offset of seg vs pre
    print("\n  逐事件基线偏差(seg vs pre):")
    for r in results:
        drift = r['baseline_seg'] - r['baseline_pre']
        print(f"    事件#{r['event_idx']} t={r['t_load']:.0f}s: 基线偏差={drift:+.0f} ADC")
    
    # Check if baseline is drifting cumulatively
    print("\n  逐卸载周期后的残留基线:")
    for evt_idx, (t_unload, _) in enumerate(unload_events):
        next_load_time = elapsed_pre[-1]
        for t2 in event_times:
            if t2 > t_unload + 1.0:
                next_load_time = t2
                break
        
        mask_after = (elapsed_pre >= t_unload + 3) & (elapsed_pre < min(t_unload + 10, next_load_time - 1))
        if mask_after.sum() > 10:
            level_after_seg = np.median(total_seg[mask_after])
            mask_orig = (elapsed_pre >= 0) & (elapsed_pre < 5)
            orig_baseline_seg = np.median(total_seg[mask_orig])
            residual = level_after_seg - orig_baseline_seg
            print(f"    卸载#{evt_idx} t={t_unload:.0f}s: 空载显示={level_after_seg:.0f}, 原始基线={orig_baseline_seg:.0f}, 残留={residual:+.0f}")

    # Dump results as JSON for later use
    json_file = OUT_DIR / "event_analysis.json"
    # Convert numpy to python types
    def to_json_safe(obj):
        if isinstance(obj, np.floating):
            return float(obj) if not np.isnan(obj) else None
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json_safe(v) for v in obj]
        return obj
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(results), f, indent=2, ensure_ascii=False)
    print(f"\n  已保存分析结果: {json_file}")


if __name__ == "__main__":
    main()
