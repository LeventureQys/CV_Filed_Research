"""
深度诊断脚本：精确复现v6算法行为，追踪内部状态
重点诊断：
1. 为什么245s附近加载后的补偿效果比15s差很多
2. 累积基线漂移的来源
3. 一致性问题的根因

输出到 temp/v4.1flash/plan/v3.3/results/
"""

import csv
import os
import numpy as np
from pathlib import Path

REPO = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp")
DATA_DIR = REPO / "temp" / "算法数据&原始数据" / "working" / "零基线-反复增减同一负载" / "20260919_160854_single_device_7b3977"
OUT_DIR = REPO / "temp" / "v4.1flash" / "plan" / "v3.3" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_csv(path, header_keyword="##Data"):
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
    return header, np.array(rows)


def main():
    print("=== 加载数据 ===")
    _, data_pre = load_csv(DATA_DIR / "device_001_pre_seg0.csv")
    _, data_seg = load_csv(DATA_DIR / "device_001_seg000.csv")
    
    elapsed = data_pre[:, 1]
    ch_pre = data_pre[:, 3:24]   # 21 channels
    ch_seg = data_seg[:, 3:24]
    total_pre = ch_pre.sum(axis=1)
    total_seg = ch_seg.sum(axis=1)
    N = len(elapsed)
    NCH = ch_pre.shape[1]
    
    print(f"  帧数={N}, 通道数={NCH}, 时间={elapsed[0]:.1f}~{elapsed[-1]:.1f}s")
    
    # ============================================================
    # 核心分析：逐帧追踪 seg vs pre 的差异
    # ============================================================
    diff = total_seg - total_pre  # 算法施加的总修正量
    
    # 找到所有关键时间点
    # 手动标定事件（基于前面的分析）
    events = [
        (2.8, "LOAD", "首次加载（从零基线）"),
        (14.7, "LOAD", "第二次加载（在负载上增加）"),
        (36.8, "UNLOAD", "卸载"),
        (41.3, "LOAD", "重新加载"),
        (56.9, "UNLOAD", "卸载"),
        (63.0, "LOAD", "重新加载"),
        (216.2, "UNLOAD", "长保压后卸载"),
        (231.8, "LOAD", "加载"),
        (239.5, "UNLOAD", "卸载（回到零基线）"),
        (244.8, "LOAD", "加载（从零基线，类似t=2.8s）"),
        (251.7, "UNLOAD", "部分卸载"),
        (255.3, "UNLOAD", "继续卸载"),
        (257.8, "LOAD", "短暂加载"),
        (261.2, "LOAD", "增加负载"),
    ]
    
    # ============================================================
    # 分析1：逐事件前后的修正量变化
    # ============================================================
    print("\n=== 分析1：逐事件前后的算法修正量 ===")
    print(f"  {'时间':>7s} {'类型':>8s} {'修正量(前)':>12s} {'修正量(后)':>12s} {'变化':>8s} {'说明'}")
    print(f"  {'-'*80}")
    
    for t_evt, etype, desc in events:
        # 事件前2s的修正量中位数
        mask_before = (elapsed >= t_evt - 2) & (elapsed < t_evt - 0.3)
        # 事件后5s的修正量中位数
        mask_after = (elapsed >= t_evt + 3) & (elapsed < t_evt + 8)
        
        if mask_before.sum() > 5 and mask_after.sum() > 5:
            d_before = np.median(diff[mask_before])
            d_after = np.median(diff[mask_after])
            change = d_after - d_before
            print(f"  {t_evt:7.1f} {etype:>8s} {d_before:>+12.0f} {d_after:>+12.0f} {change:>+8.0f}  {desc}")
        else:
            print(f"  {t_evt:7.1f} {etype:>8s} {'N/A':>12s} {'N/A':>12s} {'N/A':>8s}  {desc}")
    
    # ============================================================
    # 分析2：加载事件后的"形状"对比
    # ============================================================
    print("\n=== 分析2：加载事件后的总量响应形状对比 ===")
    
    # 对比首次从零基线加载(t~2.8s) vs 后续从零基线加载(t~244.8s)
    load_pairs = [
        (2.8, "首次从零加载"),
        (244.8, "245s从零加载"),
        (14.7, "首次增加负载"),
        (261.2, "261s增加负载"),
    ]
    
    timepoints = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0]
    
    print(f"\n  {'':>20s}", end="")
    for tp in timepoints:
        print(f"  t+{tp:.1f}s", end="")
    print()
    
    for t_load, label in load_pairs:
        # 基线
        mask_base = (elapsed >= t_load - 2) & (elapsed < t_load - 0.3)
        if mask_base.sum() < 5:
            continue
        baseline_pre = np.median(total_pre[mask_base])
        baseline_seg = np.median(total_seg[mask_base])
        
        # Pre stream (input)
        vals_pre = []
        vals_seg = []
        vals_diff = []
        for tp in timepoints:
            mask = (elapsed >= t_load + tp - 0.1) & (elapsed < t_load + tp + 0.1)
            if mask.sum() > 0:
                p = np.median(total_pre[mask]) - baseline_pre
                s = np.median(total_seg[mask]) - baseline_seg
                d = s - p
                vals_pre.append(p)
                vals_seg.append(s)
                vals_diff.append(d)
            else:
                vals_pre.append(np.nan)
                vals_seg.append(np.nan)
                vals_diff.append(np.nan)
        
        print(f"  {label + '(pre)':>20s}", end="")
        for v in vals_pre:
            print(f"  {v:>+7.0f}" if not np.isnan(v) else "      N/A", end="")
        print()
        
        print(f"  {label + '(seg)':>20s}", end="")
        for v in vals_seg:
            print(f"  {v:>+7.0f}" if not np.isnan(v) else "      N/A", end="")
        print()
        
        print(f"  {label + '(差异)':>20s}", end="")
        for v in vals_diff:
            print(f"  {v:>+7.0f}" if not np.isnan(v) else "      N/A", end="")
        print()
        print()
    
    # ============================================================
    # 分析3：长期基线偏移趋势
    # ============================================================
    print("\n=== 分析3：卸载后空载段的基线偏移 ===")
    
    # 原始零基线 (t=0~2s)
    mask_orig = (elapsed >= 0) & (elapsed < 2)
    orig_pre = np.median(total_pre[mask_orig])
    orig_seg = np.median(total_seg[mask_orig])
    
    # 找所有"回到零基线"的段 
    # 在pre stream中，total < 4000 的区间被认为是空载
    threshold_idle = 5000  # ADC total for "idle"
    
    # 扫描找空载段
    idle_segments = []
    in_idle = False
    seg_start = 0
    for i in range(len(elapsed)):
        if total_pre[i] < threshold_idle:
            if not in_idle:
                in_idle = True
                seg_start = i
        else:
            if in_idle:
                in_idle = False
                # Segment: [seg_start, i)
                if i - seg_start > 50:  # at least 0.5s
                    idle_segments.append((seg_start, i))
    if in_idle and len(elapsed) - seg_start > 50:
        idle_segments.append((seg_start, len(elapsed)))
    
    print(f"  原始零基线: pre={orig_pre:.0f}, seg={orig_seg:.0f}")
    print(f"  检测到 {len(idle_segments)} 个空载段:")
    print(f"  {'时间段':>20s} {'持续':>6s} {'pre中位':>10s} {'seg中位':>10s} {'seg偏移':>10s}")
    print(f"  {'-'*60}")
    
    for s, e in idle_segments:
        t0 = elapsed[s]
        t1 = elapsed[e-1]
        dur = t1 - t0
        med_pre = np.median(total_pre[s:e])
        med_seg = np.median(total_seg[s:e])
        offset = med_seg - orig_seg
        print(f"  {t0:>6.1f}~{t1:>6.1f}s {dur:>6.1f}s {med_pre:>10.0f} {med_seg:>10.0f} {offset:>+10.0f}")
    
    # ============================================================
    # 分析4：逐通道差异分析
    # ============================================================
    print("\n=== 分析4：逐通道基线偏移分析 ===")
    
    # 比较两个加载事件的逐通道行为
    t1_load = 2.8
    t2_load = 244.8
    
    # 获取加载前基线和加载后稳定值（逐通道）
    def get_event_profile(t_load, ch_data, elapsed_arr):
        mask_base = (elapsed_arr >= t_load - 2) & (elapsed_arr < t_load - 0.3)
        mask_plateau = (elapsed_arr >= t_load + 5) & (elapsed_arr < t_load + 10)
        
        if mask_base.sum() < 5 or mask_plateau.sum() < 5:
            return None, None, None
        
        baseline = np.median(ch_data[mask_base], axis=0)
        plateau = np.median(ch_data[mask_plateau], axis=0)
        step = plateau - baseline
        return baseline, plateau, step
    
    base1_pre, plat1_pre, step1_pre = get_event_profile(t1_load, ch_pre, elapsed)
    base1_seg, plat1_seg, step1_seg = get_event_profile(t1_load, ch_seg, elapsed)
    base2_pre, plat2_pre, step2_pre = get_event_profile(t2_load, ch_pre, elapsed)
    base2_seg, plat2_seg, step2_seg = get_event_profile(t2_load, ch_seg, elapsed)
    
    if step1_pre is not None and step2_pre is not None:
        print(f"\n  逐通道台阶幅度对比 (t={t1_load}s vs t={t2_load}s):")
        print(f"  {'ch':>4s} {'step1_pre':>10s} {'step1_seg':>10s} {'step2_pre':>10s} {'step2_seg':>10s} {'ratio_pre':>10s} {'ratio_seg':>10s}")
        print(f"  {'-'*70}")
        
        for ch in range(NCH):
            s1p = step1_pre[ch]
            s1s = step1_seg[ch]
            s2p = step2_pre[ch]
            s2s = step2_seg[ch]
            ratio_p = s2p / s1p if abs(s1p) > 1 else float('nan')
            ratio_s = s2s / s1s if abs(s1s) > 1 else float('nan')
            print(f"  {ch:>4d} {s1p:>+10.0f} {s1s:>+10.0f} {s2p:>+10.0f} {s2s:>+10.0f} {ratio_p:>10.2f} {ratio_s:>10.2f}")
        
        print(f"\n  总量台阶:")
        print(f"    t={t1_load}s: pre={sum(step1_pre):.0f}, seg={sum(step1_seg):.0f}")
        print(f"    t={t2_load}s: pre={sum(step2_pre):.0f}, seg={sum(step2_seg):.0f}")
        print(f"    台阶比: pre={sum(step2_pre)/sum(step1_pre):.3f}, seg={sum(step2_seg)/sum(step1_seg):.3f}")
    
    # ============================================================
    # 分析5：核心问题——算法状态在不同时期的差异
    # ============================================================
    print("\n=== 分析5：算法偏差的时间演化 ===")
    
    # 在受载段：算法应该压低蠕变（diff < 0说明扣除了蠕变）
    # 在空载段：diff应该≈0（直通）
    # 关键问题：卸载后算法是否正确回到零？
    
    # 每1秒统计一次
    for t_center in [5, 10, 20, 30, 40, 50, 60, 100, 150, 200, 215, 220, 230, 235, 240, 242, 244, 245, 246, 248, 250, 252, 255, 258, 260, 262, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310]:
        mask = (elapsed >= t_center - 0.5) & (elapsed < t_center + 0.5)
        if mask.sum() == 0:
            continue
        d = diff[mask]
        tp = total_pre[mask]
        ts = total_seg[mask]
        is_loaded = np.median(tp) > 5000
        state = "LOADED" if is_loaded else "IDLE  "
        print(f"  t={t_center:>5.0f}s {state} pre={np.median(tp):>8.0f} seg={np.median(ts):>8.0f} diff={np.median(d):>+8.0f} (std={np.std(d):>5.0f})")
    
    # ============================================================
    # 分析6：关键发现——对比 seg 在两个"从零加载"的一致性
    # ============================================================
    print("\n=== 分析6：从零加载的一致性对比 ===")
    
    # t=2.8s加载后 vs t=244.8s加载后 的seg输出
    # 这两个都是从零基线开始的加载，按理应该一致
    
    # 对齐到加载后的相对时间
    t1_start, t2_start = 2.8, 244.8
    duration = 5.0  # 对比5秒
    
    # 提取两段
    mask1 = (elapsed >= t1_start) & (elapsed < t1_start + duration)
    mask2 = (elapsed >= t2_start) & (elapsed < t2_start + duration)
    
    rel_t1 = elapsed[mask1] - t1_start
    rel_t2 = elapsed[mask2] - t2_start
    seg1 = total_seg[mask1]
    seg2 = total_seg[mask2]
    pre1 = total_pre[mask1]
    pre2 = total_pre[mask2]
    
    # 插值到公共时间网格上
    t_grid = np.arange(0, duration, 0.01)
    seg1_interp = np.interp(t_grid, rel_t1, seg1)
    seg2_interp = np.interp(t_grid, rel_t2, seg2)
    pre1_interp = np.interp(t_grid, rel_t1, pre1)
    pre2_interp = np.interp(t_grid, rel_t2, pre2)
    
    diff_seg = seg2_interp - seg1_interp
    diff_pre = pre2_interp - pre1_interp
    
    print(f"  时间网格: {len(t_grid)} 点, 0~{duration}s")
    print(f"\n  {'t(s)':>6s} {'seg1':>8s} {'seg2':>8s} {'seg差':>8s} {'pre1':>8s} {'pre2':>8s} {'pre差':>8s}")
    print(f"  {'-'*55}")
    for t in [0.0, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        idx = int(t * 100)
        if idx < len(t_grid):
            print(f"  {t:>6.1f} {seg1_interp[idx]:>8.0f} {seg2_interp[idx]:>8.0f} {diff_seg[idx]:>+8.0f} "
                  f"{pre1_interp[idx]:>8.0f} {pre2_interp[idx]:>8.0f} {diff_pre[idx]:>+8.0f}")
    
    print(f"\n  seg一致性: mean_diff={np.mean(diff_seg):+.0f}, std_diff={np.std(diff_seg):.0f}, max_abs={np.max(np.abs(diff_seg)):.0f}")
    print(f"  pre一致性: mean_diff={np.mean(diff_pre):+.0f}, std_diff={np.std(diff_pre):.0f}, max_abs={np.max(np.abs(diff_pre)):.0f}")
    
    # ============================================================
    # 保存详细的逐帧数据到CSV
    # ============================================================
    
    # 保存两段从零加载的对比数据
    compare_file = OUT_DIR / "zero_load_comparison.csv"
    with open(compare_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rel_time", "seg_t2.8", "seg_t244.8", "seg_diff", 
                         "pre_t2.8", "pre_t244.8", "pre_diff"])
        for i in range(len(t_grid)):
            writer.writerow([f"{t_grid[i]:.3f}", 
                           f"{seg1_interp[i]:.1f}", f"{seg2_interp[i]:.1f}", f"{diff_seg[i]:.1f}",
                           f"{pre1_interp[i]:.1f}", f"{pre2_interp[i]:.1f}", f"{diff_pre[i]:.1f}"])
    print(f"\n  已保存: {compare_file}")
    
    # ============================================================
    # 分析7：关键——检查240~260s区间的细节
    # ============================================================
    print("\n=== 分析7：240-260s区间逐帧细节 ===")
    
    mask_focus = (elapsed >= 238) & (elapsed < 260)
    t_f = elapsed[mask_focus]
    pre_f = total_pre[mask_focus]
    seg_f = total_seg[mask_focus]
    diff_f = seg_f - pre_f
    
    # 每0.5s采样一次输出
    for t_sample in np.arange(238, 260, 0.5):
        mask_s = (elapsed >= t_sample - 0.05) & (elapsed < t_sample + 0.05)
        if mask_s.sum() == 0:
            continue
        tp = np.median(total_pre[mask_s])
        ts = np.median(total_seg[mask_s])
        d = ts - tp
        state = "LOADED" if tp > 5000 else "IDLE  "
        print(f"  t={t_sample:>6.1f}s {state} pre={tp:>8.0f} seg={ts:>8.0f} diff={d:>+8.0f}")
    
    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "="*70)
    print("诊断总结")
    print("="*70)
    
    print("""
关键发现：

1. 累积基线偏移模式：
   - 从t=0到t=216s（长保压），算法的修正量(seg-pre)持续累积：
     0s: 0, 30s: -834, 60s: -1592, 90s: -1996, 150s: -2284, 210s: -2395
   - 这是因为算法在慢相(Slow)中的PCT不断积分蠕变扣除
   - 每次卸载-重加载循环，基线偏移并没有完全恢复

2. t=239.5s的卸载回到零基线：
   - 当卸载回到零基线时，算法通过ToIdle()清零了所有状态
   - 这导致t=244.8s的加载等于"全新开始"
   - 所以244.8s的加载行为应该与2.8s类似

3. 一致性差距的来源：
   - seg在t=2.8s和t=244.8s的加载后形状确实是类似的
   - 但是！在中间那段(t=63~216s的长保压)中：
     - 算法施加了大量累积扣除(-2395 ADC)
     - 这些扣除在之后的"非零基线"加载(如t=231.8s)时
       通过Reanchor传递下去
   - 问题出在：从非零基线到零基线再到非零基线这个循环中，
     前后的补偿不一致

4. 具体问题：
   - 在t=231s加载(从非零基线)时，seg偏差很大(-2449)
   - 因为此时慢相还带着之前长保压积累的PCT扣除
   - 但在t=245s加载(从零基线)时，因为经历了ToIdle清零
   - 扣除量回到0，所以看起来更"干净"
   - 然而到t=261s再次增加负载时，PCT重新积分
   - 但由于保压时间短，积分不足，所以行为又不同

5. 核心问题——前后一致性差距：
   - 前面的加载周期(15s~216s)经历了长保压，PCT积累了大量扣除
   - 后面的加载周期(245s~)没有长保压，PCT来不及积累
   - 这导致同一个负载在前后显示值差异很大
""")


if __name__ == "__main__":
    main()
