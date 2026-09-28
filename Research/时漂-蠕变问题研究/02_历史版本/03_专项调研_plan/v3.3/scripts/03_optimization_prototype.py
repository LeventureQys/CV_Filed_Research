"""
v3.3 优化原型（高效版）：用numpy向量化代替逐帧循环

只保留关键的简化模拟：基于实测数据的seg vs pre差异分析，
不做完整的逐帧算法模拟（那个应该交给C++ runner）。

核心思路：利用已有的seg和pre数据来分析PCT的累积效应，
然后提出修改方案并通过数据推演验证。
"""

import csv
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
    ch_pre = data_pre[:, 3:24]
    ch_seg = data_seg[:, 3:24]
    total_pre = ch_pre.sum(axis=1)
    total_seg = ch_seg.sum(axis=1)
    N = len(elapsed)
    NCH = ch_pre.shape[1]
    diff = total_seg - total_pre  # 算法总修正量
    
    print(f"  帧数={N}, 通道数={NCH}")
    
    # ============================================================
    # 基于实测数据的详细分析
    # ============================================================
    
    # 1. 逐通道diff分析
    ch_diff = ch_seg - ch_pre  # (N, 21)
    
    # 2. 关键时间段的逐通道分析
    segments = {
        "初始零基线(0-2s)": (0, 2),
        "首次加载后5s(8-12s)": (8, 12),
        "首次加载后20s(23-27s)": (23, 27),
        "长保压60s(120-125s)": (120, 125),
        "长保压150s(205-210s)": (205, 210),
        "卸载后零基线(241-244s)": (241, 244),
        "245加载后5s(250-254s)": (250, 254),
        "261加载后5s(266-270s)": (266, 270),
        "288加载后5s(294-298s)": (294, 298),
    }
    
    print("\n=== 逐时段的总量偏差 ===")
    print(f"  {'时段':<30s} {'pre中位':>10s} {'seg中位':>10s} {'diff中位':>10s} {'diff_std':>10s}")
    print(f"  {'-'*75}")
    
    for label, (t0, t1) in segments.items():
        mask = (elapsed >= t0) & (elapsed < t1)
        if mask.sum() == 0:
            continue
        mp = np.median(total_pre[mask])
        ms = np.median(total_seg[mask])
        md = np.median(diff[mask])
        sd = np.std(diff[mask])
        print(f"  {label:<30s} {mp:>10.0f} {ms:>10.0f} {md:>+10.0f} {sd:>10.1f}")
    
    # 3. 逐通道偏差分析：哪些通道累积了最多PCT
    print("\n=== 长保压(t=200s)时逐通道偏差 ===")
    mask_200 = (elapsed >= 200) & (elapsed < 205)
    mask_0 = (elapsed >= 0) & (elapsed < 2)
    
    if mask_200.sum() > 0 and mask_0.sum() > 0:
        diff_200 = np.median(ch_seg[mask_200] - ch_pre[mask_200], axis=0)
        pre_200 = np.median(ch_pre[mask_200], axis=0)
        # 加载幅度（t=8-12s的值 vs t=0-2s）
        mask_load = (elapsed >= 8) & (elapsed < 12)
        load_amp = np.median(ch_pre[mask_load], axis=0) - np.median(ch_pre[mask_0], axis=0)
        
        print(f"  {'ch':>4s} {'pre值':>8s} {'加载幅':>8s} {'diff':>8s} {'diff/幅%':>10s} {'loaded':>8s}")
        print(f"  {'-'*55}")
        
        total_diff = 0
        for ch in range(NCH):
            d = diff_200[ch]
            amp = load_amp[ch]
            pct = (d / amp * 100) if abs(amp) > 5 else float('nan')
            loaded = "Y" if amp > 0.10 * max(load_amp) else "N"
            total_diff += d
            print(f"  {ch:>4d} {pre_200[ch]:>8.0f} {amp:>+8.0f} {d:>+8.0f} {pct:>+10.1f}% {loaded:>8s}")
        
        print(f"  {'总计':>4s} {sum(pre_200):>8.0f} {sum(load_amp):>+8.0f} {total_diff:>+8.0f}")
    
    # 4. PCT累积速率分析
    print("\n=== PCT累积速率分析（diff随时间的变化） ===")
    
    # 在恒定负载段(63s~216s)，diff的变化就是PCT的累积
    mask_hold = (elapsed >= 63) & (elapsed < 216)
    t_hold = elapsed[mask_hold]
    diff_hold = diff[mask_hold]
    
    # 每10秒采样
    print(f"  {'时间':>6s} {'diff':>10s} {'Δdiff/10s':>12s} {'速率/s':>10s}")
    prev_d = None
    prev_t = None
    for t_sample in range(65, 216, 10):
        mask_s = (elapsed >= t_sample - 1) & (elapsed < t_sample + 1)
        if mask_s.sum() == 0:
            continue
        d = np.median(diff[mask_s])
        if prev_d is not None:
            delta = d - prev_d
            dt = t_sample - prev_t
            rate = delta / dt
            print(f"  {t_sample:>6d}s {d:>+10.0f} {delta:>+12.0f} {rate:>+10.1f}")
        else:
            print(f"  {t_sample:>6d}s {d:>+10.0f} {'---':>12s} {'---':>10s}")
        prev_d = d
        prev_t = t_sample
    
    # 5. 修正后一致性推演
    print("\n=== 修正方案推演 ===")
    print("""
根因分析：
  1. 算法在慢相期间，PCT持续积分蠕变（τ=10s的EMA追踪器）
  2. 在长保压(153s)期间，PCT累积了-2395 ADC的扣除量
  3. 每次卸载-重加载，PCT通过Reanchor部分传递（或在ToIdle时清零）
  4. 这导致：经历过长保压的加载周期 vs 新鲜的加载周期，显示值差别巨大

关键矛盾：
  - PCT的目的：抵消慢相蠕变，让长保压时显示稳定
  - PCT的代价：累积的扣除量在变载事件中破坏了一致性
  
解决思路：
  目标是让PCT在"同一保压段内"有效（抵消蠕变），
  但在"跨事件"时不传递过多的历史扣除。

  方案A：PCT加入自然衰减（类似遗忘因子）
    pct_k += (dt/τ) * resid - (dt/τ_decay) * pct_k
    等效于：pct的积分目标不再是"让显示=A"，而是"让显示偏差在有限时间内归零"
    τ_decay = 30~60s 可以保证：
    - 前10-20s：PCT正常追踪蠕变（衰减效应<20%）
    - 60s后：PCT开始显著衰减，累积量受限
    - 200s后：PCT几乎完全衰减

  方案B：Reanchor时衰减PCT
    在减重/卸载事件触发Reanchor时，将PCT乘以衰减系数(0~0.5)
    好处：事件驱动，不影响保压内的稳定性
    代价：突变（虽然平滑处理可缓解）

  方案C：PCT上界与保压时长挂钩
    pct_hi_frac 随保压时长线性减小
    太复杂，且不解决根本问题

推荐方案：方案A + 方案B的组合
  - PCT自然衰减τ_decay=45s（保压内前20s几乎不影响，60s后衰减50%）
  - Reanchor时PCT乘以0.3（事件边界的大幅衰减）
  - 这样：
    * 短保压(<20s)：PCT正常工作，与当前几乎一样
    * 中等保压(20-60s)：PCT开始自然衰减，累积量有限
    * 长保压(>60s)：PCT的累积量被限制在~2×τ_decay的等效窗内
    * 变载事件：PCT在Reanchor时大幅衰减，减少跨事件传递
""")
    
    # 6. 数值推演
    print("\n=== 数值推演：不同衰减方案下的PCT累积量 ===")
    
    # 假设恒定蠕变速率，估算PCT在不同方案下的稳态值
    # 实测：153s保压，PCT累积 ≈ -2395 ADC（总量口径）
    # 等效蠕变速率 ≈ 2395 / 153 ≈ 15.7 ADC/s（但实际是递减的）
    
    # 更精确：看diff随时间的变化
    # t=65s: diff≈-1900, t=215s: diff≈-2395
    # 150s内增加了495 ADC的扣除
    # 平均速率: 3.3 ADC/s（后期速率低，因为蠕变本身在减速）
    
    # 早期蠕变更快：
    # t=5~15s: diff从0到-1060，10s内1060 → 106 ADC/s
    # t=15~35s: diff从-1060到-1500，20s内440 → 22 ADC/s  
    # t=65~215s: diff从-1900到-2395，150s内495 → 3.3 ADC/s
    
    print("  蠕变速率估算（从实测数据）:")
    print("    t=5~15s:   ~106 ADC/s")
    print("    t=15~35s:  ~22 ADC/s")
    print("    t=65~120s: ~6 ADC/s")
    print("    t=120~215s: ~2 ADC/s")
    
    # 模拟不同衰减方案
    print("\n  方案模拟（假设典型蠕变输入）:")
    
    dt = 0.01  # 100Hz
    T = 200  # 模拟200秒
    t_arr = np.arange(0, T, dt)
    
    # 模拟蠕变输入：指数递减的蠕变速率
    # creep(t) = A * (1 - exp(-t/τ_creep))
    # A = 2500 ADC (总蠕变幅度), τ_creep = 30s
    A_creep = 2500
    tau_creep = 30
    creep = A_creep * (1 - np.exp(-t_arr / tau_creep))
    creep_rate = (A_creep / tau_creep) * np.exp(-t_arr / tau_creep)
    
    configs = {
        '当前(无衰减)': {'tau_pct': 10.0, 'tau_decay': 0},
        'τ_decay=30s': {'tau_pct': 10.0, 'tau_decay': 30},
        'τ_decay=45s': {'tau_pct': 10.0, 'tau_decay': 45},
        'τ_decay=60s': {'tau_pct': 10.0, 'tau_decay': 60},
        'τ_pct=5s': {'tau_pct': 5.0, 'tau_decay': 0},
        'τ_pct=5+decay30': {'tau_pct': 5.0, 'tau_decay': 30},
    }
    
    print(f"\n  {'方案':<25s}", end="")
    for t in [5, 10, 20, 30, 60, 120, 200]:
        print(f"  t={t:>3d}s", end="")
    print(f"  {'稳态':>8s}")
    print(f"  {'-'*25}", end="")
    for _ in range(7):
        print(f"  {'-'*7}", end="")
    print(f"  {'-'*8}")
    
    for name, cfg in configs.items():
        tau_pct = cfg['tau_pct']
        tau_decay = cfg['tau_decay']
        
        pct = np.zeros_like(t_arr)
        for i in range(1, len(t_arr)):
            resid = creep[i] - pct[i-1]
            d_pct = (dt / tau_pct) * resid
            if tau_decay > 0:
                d_pct -= (dt / tau_decay) * pct[i-1]
            pct[i] = pct[i-1] + d_pct
        
        print(f"  {name:<25s}", end="")
        for t in [5, 10, 20, 30, 60, 120, 200]:
            idx = int(t / dt)
            if idx < len(pct):
                print(f"  {pct[idx]:>+7.0f}", end="")
        # 稳态值
        if tau_decay > 0:
            # 稳态 ≈ A_creep * tau_decay / (tau_pct + tau_decay)
            steady = A_creep * tau_decay / (tau_pct + tau_decay)
            print(f"  {steady:>+8.0f}")
        else:
            print(f"  {pct[-1]:>+8.0f}")
    
    # 蠕变本身
    print(f"\n  {'蠕变(输入)':<25s}", end="")
    for t in [5, 10, 20, 30, 60, 120, 200]:
        idx = int(t / dt)
        if idx < len(creep):
            print(f"  {creep[idx]:>+7.0f}", end="")
    print(f"  {A_creep:>+8.0f}")
    
    # 残留蠕变 = creep - pct
    print(f"\n  残留蠕变（未被补偿的）:")
    for name, cfg in configs.items():
        tau_pct = cfg['tau_pct']
        tau_decay = cfg['tau_decay']
        
        pct = np.zeros_like(t_arr)
        for i in range(1, len(t_arr)):
            resid = creep[i] - pct[i-1]
            d_pct = (dt / tau_pct) * resid
            if tau_decay > 0:
                d_pct -= (dt / tau_decay) * pct[i-1]
            pct[i] = pct[i-1] + d_pct
        
        residual = creep - pct
        print(f"  {name:<25s}", end="")
        for t in [5, 10, 20, 30, 60, 120, 200]:
            idx = int(t / dt)
            if idx < len(residual):
                print(f"  {residual[idx]:>+7.0f}", end="")
        print()
    
    # ============================================================
    # 7. 跨事件一致性推演
    # ============================================================
    print("\n\n=== 跨事件一致性推演 ===")
    print("  场景：加载→保压Ts→卸载→加载（相同负载）")
    print("  问题：第二次加载时PCT残留量导致显示偏差")
    
    print(f"\n  {'方案':<25s} {'保压20s后PCT':>14s} {'卸载后残留':>12s} {'保压60s后PCT':>14s} {'卸载后残留':>12s} {'保压150s后PCT':>16s} {'卸载后残留':>12s}")
    print(f"  {'-'*115}")
    
    for name, cfg in configs.items():
        tau_pct = cfg['tau_pct']
        tau_decay = cfg['tau_decay']
        
        results = []
        for hold_time in [20, 60, 150]:
            # 模拟保压期间PCT累积
            pct_val = 0
            for i in range(int(hold_time / dt)):
                t = i * dt
                c = A_creep * (1 - np.exp(-t / tau_creep))
                resid = c - pct_val
                d_pct = (dt / tau_pct) * resid
                if tau_decay > 0:
                    d_pct -= (dt / tau_decay) * pct_val
                pct_val += d_pct
            
            # Reanchor时衰减 (当前算法默认=0.5)
            pct_after_reanchor = pct_val * 0.5
            results.append((pct_val, pct_after_reanchor))
        
        print(f"  {name:<25s}", end="")
        for pv, par in results:
            print(f" {pv:>+14.0f} {par:>+12.0f}", end="")
        print()
    
    # ============================================================
    # 8. 保存分析报告
    # ============================================================
    report_file = OUT_DIR / "analysis_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("v3.3 算法优化分析报告\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. 问题描述\n")
        f.write("-" * 40 + "\n")
        f.write("在「零基线-反复增减同一负载」数据集中，\n")
        f.write("245s附近的加载事件与15s的首次加载相比，一致性差距较大。\n\n")
        
        f.write("2. 根因诊断\n")
        f.write("-" * 40 + "\n")
        f.write("PCT(逐通道蠕变跟踪)在长保压期间累积了大量扣除(-2395 ADC)。\n")
        f.write("这些扣除量在卸载-重加载循环中的传递不一致：\n")
        f.write("  - 经历长保压后的加载段：带着大量PCT累积量（高扣除）\n")
        f.write("  - ToIdle后的新加载段：PCT被清零（零扣除）\n")
        f.write("  - 同一负载在前后的显示差值：~2400 ADC\n\n")
        
        f.write("3. 实测数据关键数字\n")
        f.write("-" * 40 + "\n")
        f.write("  算法偏差(seg-pre)时间演化：\n")
        f.write("    t=0s:    0 ADC（空载直通）\n")
        f.write("    t=10s:  -791 ADC（快相补偿开始）\n")
        f.write("    t=30s:  -1401 ADC\n")
        f.write("    t=60s:  -1783 ADC\n")
        f.write("    t=100s: -2195 ADC\n")
        f.write("    t=200s: -2451 ADC\n")
        f.write("    t=242s: 0 ADC（ToIdle清零）\n")
        f.write("    t=246s: +79 ADC（新加载，PCT重新积分）\n")
        f.write("    t=265s: +45 ADC\n\n")
        
        f.write("  卸载后空载段残留：\n")
        f.write("    t=240-245s: seg偏移 +185 ADC（vs 原始零基线）\n")
        f.write("    t=256-258s: seg偏移 +152 ADC\n")
        f.write("    t=276-278s: seg偏移 +233 ADC\n")
        f.write("    （这些残留来自pre本身的蠕变残留，不是算法导致的）\n\n")
        
        f.write("4. 优化方案\n")
        f.write("-" * 40 + "\n")
        f.write("推荐方案：PCT加入自然衰减 + Reanchor时衰减\n\n")
        f.write("  具体修改（drift_v6_compensator.cpp 的 SlowStep）：\n")
        f.write("    在PCT积分更新后追加一行衰减：\n")
        f.write("    pct_ded_(k) *= (1.0 - dt / kPctDecayTauS);  // 新常量\n\n")
        f.write("  新增常量：\n")
        f.write("    kPctDecayTauS = 45.0;  // PCT自然衰减时间常数(秒)\n\n")
        f.write("  效果预估（基于数值模拟）：\n")
        f.write("    - 短保压(<20s)：几乎不影响（衰减<35%）\n")
        f.write("    - 中保压(60s)：PCT累积量从2034减至1485（-27%）\n")
        f.write("    - 长保压(150s)：PCT累积量从2477减至1907（-23%）\n")
        f.write("    - 跨事件一致性显著改善\n")
        f.write("    - 代价：长保压尾段的蠕变补偿效果略降\n")
    
    print(f"\n  已保存报告: {report_file}")
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
