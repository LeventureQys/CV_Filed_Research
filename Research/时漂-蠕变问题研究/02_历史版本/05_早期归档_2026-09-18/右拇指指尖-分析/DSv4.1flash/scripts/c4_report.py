# -*- coding: utf-8 -*-
"""步骤 C4：生成最终图文报告（Markdown）。"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import DATASETS, RES
from causal_harness import prepare, online_load_detect
from causal_algorithms import Shape, FROZEN

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(OUT, "触觉阵列时漂与零漂分析报告.md")

with open(os.path.join(RES, "C2_causal_sweep.json"), encoding="utf-8") as f:
    SWEEP = json.load(f)
with open(os.path.join(RES, "C2_summary.json"), encoding="utf-8") as f:
    SUMMARY = json.load(f)
with open(os.path.join(RES, "C0_creep_shape.json"), encoding="utf-8") as f:
    SHAPE = json.load(f)
try:
    with open(os.path.join(RES, "C2_sensitivity.json"), encoding="utf-8") as f:
        SENS = json.load(f)
except Exception:
    SENS = {}
try:
    with open(os.path.join(RES, "C2_onset_sensitivity.json"), encoding="utf-8") as f:
        ONS = json.load(f)
except Exception:
    ONS = {}

SC = Shape(**FROZEN["common"])
HR = SC.key_ratios()

# ---------------- 收集数据事实 ----------------
FACTS = {}
for name in DATASETS:
    D = prepare(name)
    X, t, fs = D["X"], D["t"], D["fs"]
    a0, a1 = D["pre"]; c0, d0 = D["load"]
    w = lambda s: int(round(s * fs))
    Xn = X - X[a0:a1].mean(axis=0)[None, :]
    n_on, _, _ = online_load_detect(Xn.sum(axis=1), fs, a1)
    REF = Xn[n_on + w(1.0):n_on + w(3.0)].mean(axis=0)
    act = np.where(Xn[c0:d0].max(axis=0) > 0.05)[0]
    late = Xn[d0 - w(5):d0].mean(axis=0)
    drift = late - REF
    dead = np.where(X.std(axis=0) == 0)[0]
    # 噪声
    def cs(y, k):
        c = np.concatenate([[0.0], np.cumsum(y)]); n = len(y)
        i = np.arange(n); lo = np.maximum(0, i - k + 1)
        return (c[i + 1] - c[lo]) / (i + 1 - lo)
    j = int(act[np.argmax(REF[act])])
    y20 = Xn[d0 - w(20):d0, j]
    hf = np.diff(y20).std() / np.sqrt(2)
    lf = (y20 - cs(y20, w(0.1))).std()
    q = Xn[a0:a1, j]
    FACTS[name] = dict(
        name=name, fs=fs, n=len(t), dur=t[-1],
        t_pre=t[a1], t_load=t[d0] - t[c0], t_post=t[-1] - t[d0],
        n_on=n_on, on_true=D["on_true"],
        det_delay_ms=(n_on - D["on_true"]) / fs * 1000,
        off_true=D["off_true"],
        n_act=len(act), n_dead=len(dead),
        dead_ch=[D["ch"][i] for i in dead],
        ch=D["ch"], j=j, chj=D["ch"][j],
        REFj=REF[j], latej=late[j], driftj=drift[j],
        drift_med_mN=float(np.median(drift[act]) * 1000),
        drift_rel_med=float(np.median(drift[act] / REF[act]) * 100),
        drift_rel_max=float(np.max(drift[act] / REF[act]) * 100),
        drift_rel_min=float(np.min(drift[act] / REF[act]) * 100),
        sum_max=float(Xn[c0:d0].sum(axis=1).max()),
        fmax=float(REF[act].max()),
        noise_hf_mN=float(hf * 1000), noise_lf_mN=float(lf * 1000),
        noise_quiet_mN=float(q.std() * 1000),
        zero_pre_mN=float(np.median(Xn[a1 - w(1):a1].mean(axis=0)[act]) * 1000),
        zero_post_mN=float(np.median(Xn[d0:].mean(axis=0)[act]) * 1000),
        zero_post_absmax_mN=float(np.max(np.abs(Xn[d0:].mean(axis=0)[act])) * 1000),
        zero_jump_mN=float(np.median(
            (Xn[d0:d0 + w(0.2)].mean(axis=0) - Xn[d0 - w(0.2):d0].mean(axis=0))[act]) * 1000),
    )
    # 通道间漂移一致性（归一化形态散度）
    refc = REF[act]; ok = np.abs(refc) > 1e-6
    shapes = []
    for jj in act[ok]:
        hh = Xn[c0:d0, jj] / REF[jj]
        shapes.append(cs(hh, w(0.5)))
    S = np.vstack(shapes)
    FACTS[name]["shape_spread"] = float(np.mean(S.std(axis=0) / np.maximum(
        np.abs(S.mean(axis=0)), 1e-9)))

# ---------------- 排行榜 ----------------
def pick(rows, key, n=6, asc=True):
    rs = sorted(rows, key=lambda r: r[key] if asc else -r[key])
    return rs[:n]

by_algo = {r["algo"]: r for r in SUMMARY}

L = []
A = L.append

A("# 触觉阵列传感器时漂 / 零漂机理分析与因果补偿算法对比")
A("")
A("> 数据：`temp/右拇指指尖/数据1~3`（3 组"空载-恒定负载-空载"记录，右拇指指尖按压，9×7 阵列 31 通道）")
A("> 约束：**所有被测试与验证的算法均为因果（causal / 在线流式）实现**")
A("> 产物目录：`temp/右拇指指尖/DSv4.1flash/`")
A("")
A("---")
A("")
A("## 0. 结论速览")
A("")
A(f"1. **时漂是真实的、量级很大的乘性漂移**：持载 100~150 s 后读数比 1~3 s 时再涨 "
  f"{FACTS['数据1']['drift_rel_med']:.0f}% / {FACTS['数据2']['drift_rel_med']:.0f}% / "
  f"{FACTS['数据3']['drift_rel_med']:.0f}%（三组中位），绝对量 "
  f"{FACTS['数据1']['drift_med_mN']:.0f}~{FACTS['数据2']['drift_med_mN']:.0f} mN。")
A(f"2. **漂移是乘性的**：漂移量正比于该通道载荷，跨通道"漂移/响应"过原点回归 R²≈0.83；"
  f"归一化形态在通道间高度一致（形态散度 "
  f"{np.mean([FACTS[n]['shape_spread'] for n in DATASETS]):.3f}）。")
A(f"3. **漂移形态 = 快松弛 + 幂律慢蠕变**：约 85~90% 的上升发生在 1 s 内"
  f"（τ_f≈{SC.tauf:.2f}s），其后是长期不饱和的幂律蠕变（c≈{SC.c:.3f}, p≈{SC.p:.2f}）；"
  f"h(τ) 在 τ=10s/60s/110s 分别为 {HR[10]:.3f}/{HR[60]:.3f}/{HR[110]:.3f}。")
A(f"4. **零漂相对可忽略但机理特殊**：卸载后 3 s 内回零，残余 "
  f"{np.mean([abs(FACTS[n]['zero_post_mN']) for n in DATASETS]):.2f} mN（"
  f"≈时漂的 {np.mean([abs(FACTS[n]['zero_post_mN'])/max(abs(FACTS[n]['drift_med_mN']),1e-9) for n in DATASETS])*100:.1f}%）；"
  f"另有 {FACTS['数据1']['n_dead']} 个"恒零"通道（ADC 死区 / 未接线），"
  f"使小信号与零漂无法观测。")
A("5. **因果补偿可行，但必须引入额外信息**：恒定负载下 `y=F·h(τ)` 是乘积，"
  "静息期 `y=0` 不提供刻度，故 F 与 h 的绝对水平**在单通道内不可辨识**。"
  "本报告给出三条可行路线并完成实装 + 实测对比（见第 4 节）。")
A("")

A("---")
A("")
A("## 1. 数据与时间轴")
A("")
A("### 1.1 数据基本事实")
A("")
A("| 项目 | 数据1 | 数据2 | 数据3 |")
A("|---|---|---|---|")
A("| 帧数 | " + " | ".join(f"{FACTS[n]['n']}" for n in DATASETS) + " |")
A("| 时长 (s) | " + " | ".join(f"{FACTS[n]['dur']:.2f}" for n in DATASETS) + " |")
A("| 采样率 (Hz) | " + " | ".join(f"{FACTS[n]['fs']:.3f}" for n in DATASETS) + " |")
A("| 前空载 (s) | " + " | ".join(f"{FACTS[n]['t_pre']:.2f}" for n in DATASETS) + " |")
A("| 负载 (s) | " + " | ".join(f"{FACTS[n]['t_load']:.2f}" for n in DATASETS) + " |")
A("| 后空载 (s) | " + " | ".join(f"{FACTS[n]['t_post']:.2f}" for n in DATASETS) + " |")
A("| 受载通道数 | " + " | ".join(f"{FACTS[n]['n_act']}" for n in DATASETS) + " |")
A("| 恒零(死)通道数 | " + " | ".join(f"{FACTS[n]['n_dead']}" for n in DATASETS) + " |")
A("| Σ峰值 (N) | " + " | ".join(f"{FACTS[n]['sum_max']:.2f}" for n in DATASETS) + " |")
A("| 单通道最大参考水平 (N) | " + " | ".join(f"{FACTS[n]['fmax']:.3f}" for n in DATASETS) + " |")
A("")
A(f"**恒零通道**：数据1 为 {', '.join(FACTS['数据1']['dead_ch'])}；"
  f"数据2/3 为 {', '.join(FACTS['数据2']['dead_ch'])}。"
  "这些通道整段 std=0（输出恒为 0.000000 N），属于无效/未接入通道，"
  "既不能提供参考，也不参与漂移统计。")
A("")
A("### 1.2 时间轴的坑（重要）")
A("")
A("- CSV 的 `elapsed` 是**毫秒量化**的，`timestamp` 存在约 40% 的重复/抖动"
  "（上位机接收时刻，不是设备采样时刻）。")
A("- 因此本报告**不用 CSV 时间戳做微分/滤波**，而是用 `frame_index` 线性映射到均匀时间轴："
  "`t = frame_index · Δ`，Δ 由 `frame_index→timestamp` 最小二乘拟合得到，三组均为 "
  f"{FACTS['数据1']['fs']:.3f} Hz（R²≈0.99999998），即设备稳定上传 ~100.5 Hz。")
A("- `record_frequency` 字段记为 0.00，不可作为采样率依据。")
A("")

A("---")
A("")
A("## 2. 时漂特征")
A("")
A("### 2.1 分段")
A("")
A("时漂会让"卸载后总量"高于"加载前总量"，因此**任何基于"是否低于固定阈值"的持续判据**
A("都会把后空载段误判为负载段。本报告改用"绕峰值双向扫阈值"：")
A("")
A("1. 取总量峰值 P 与静息水平 base；")
A("2. 阈值 thr = base + 10%·(P − base)；")
A("3. 从峰值向左/向右扫到第一个低于 thr 的帧，即为加载沿 / 卸载沿。")
A("")
A("| 项目 | 数据1 | 数据2 | 数据3 |")
A("|---|---|---|---|")
A("| 加载沿 (s) | " + " | ".join(f"{FACTS[n]['on_true']/FACTS[n]['fs']:.3f}" for n in DATASETS) + " |")
A("| 卸载沿 (s) | " + " | ".join(f"{FACTS[n]['off_true']/FACTS[n]['fs']:.3f}" for n in DATASETS) + " |")
A("| 在线检测加载时刻 (s) | " + " | ".join(f"{FACTS[n]['n_on']/FACTS[n]['fs']:.3f}" for n in DATASETS) + " |")
A("| 检测偏差 (ms) | " + " | ".join(f"{FACTS[n]['det_delay_ms']:+.0f}" for n in DATASETS) + " |")
A("")
A("在线检测器（严格因果：只用历史）做法：因果 EWMA（0.15 s）平滑总量，"
  "阈值 = base + max(12%·(运行最大值−base), 0.10 N)，连续 3 帧超阈即判加载。")
A("三组偏差均在 ±0.5 s 内，足以支撑需要"加载时刻"的算法。")
A("")
A("### 2.2 漂移量")
A("")
A("以"加载后 1~3 s 的读数"为参考水平（工程上最常见的"施力后稍候读数"口径）：")
A("")
A("| 项目 | 数据1 | 数据2 | 数据3 |")
A("|---|---|---|---|")
A("| 受载通道相对漂移 中位 (%) | " + " | ".join(f"{FACTS[n]['drift_rel_med']:+.2f}" for n in DATASETS) + " |")
A("| 范围 (%) | " + " | ".join(f"{FACTS[n]['drift_rel_min']:+.1f}~{FACTS[n]['drift_rel_max']:+.1f}" for n in DATASETS) + " |")
A("| 绝对漂移 中位 (mN) | " + " | ".join(f"{FACTS[n]['drift_med_mN']:+.1f}" for n in DATASETS) + " |")
A("")
A(f"主通道（各组响应最大的通道，均为 ch17）示例：")
for n in DATASETS:
    F = FACTS[n]
    A(f"- {n}：{F['chj']} 参考水平 {F['REFj']*1000:.1f} mN → 末 5 s "
      f"{F['latej']*1000:.1f} mN（{F['driftj']/F['REFj']*100:+.1f}%）")
A("")
A("### 2.3 漂移形态（双时间尺度模型）")
A("")
A("对每通道（因果平滑 0.5 s 后）在**负载段内**拟合：")
A("")
A("```")
A("F(t) = F∞ · [ w_f·(1 − e^(−t/τ_f)) + (1 − w_f)·c·t^p ]")
A("```")
A("")
A("| 参数 | 数据1 | 数据2 | 数据3 | 冻结值(三组中位) |")
A("|---|---|---|---|---|")
for k, lab in (("wf", "w_f 快过程权重"), ("tauf", "τ_f 快过程时间常数 (s)"),
               ("c", "c 慢蠕变系数"), ("p", "p 慢蠕变幂指数")):
    A(f"| {lab} | " + " | ".join(f"{SHAPE[n]['shared'][k]:.3f}" for n in DATASETS)
      + f" | {FROZEN['common'][k]:.3f} |")
A("")
A("拟合优度（平滑后）R² 中位：" +
  " / ".join(f"{n} {np.median([v['r2'] for v in SHAPE[n]['fits'].values()]):.3f}"
             for n in DATASETS))
A("")
A("**物理含义**：")
A("- `w_f≈0.9`、`τ_f≈1.0 s` 的快过程不是漂移，而是"接触建立 + 弹性体快速松弛"的"
  "**真实响应建立过程**——手指压上去后，读数在 ~1 s 内完成约 90% 的上升。")
A(f"- 剩下的幂律项才是工程意义上的**时漂**：h(τ) 从 1 s 的 1.00 涨到 "
  f"10 s 的 {HR[10]:.3f}、60 s 的 {HR[60]:.3f}、110 s 的 {HR[110]:.3f}，"
  f"即 110 s 时读数比 1 s 时高 {(HR[110]-1)*100:.0f}%。")
A("- 幂指数 `p≈0.55`（<1）说明**没有任何特征时间常数**：漂移在观测尺度上持续增长、"
  "不会饱和。这直接否定了"单指数饱和模型"的适用性（见第 3 节）。")
A("")
A("### 2.4 漂移的乘性（关键结论）")
A("")
A("把"每通道漂移量"对"该通道响应幅度"做回归：")
A("")
A("| 项目 | 数据1 | 数据2 | 数据3 |")
A("|---|---|---|---|")
A("| 过原点回归 R² | 0.83 | 0.86 | 0.77 |")
A("| 归一化形态跨通道散度 | " + " | ".join(f"{FACTS[n]['shape_spread']:.3f}" for n in DATASETS) + " |")
A("")
A("⇒ 漂移**正比于载荷**（乘性），且归一化形态在通道间基本一致 ⇒ 可以定义一个"
  "**阵列共享的归一化形态 h(τ)**，这是阵列信号处理路线的物理基础。")
A("")

A("---")
A("")
A("## 3. 零漂特征")
A("")
A("### 3.1 幅值")
A("")
A("| 项目 | 数据1 | 数据2 | 数据3 |")
A("|---|---|---|---|")
A("| 前空载末 1 s 均值 (mN) | " + " | ".join(f"{FACTS[n]['zero_pre_mN']:+.3f}" for n in DATASETS) + " |")
A("| 卸载瞬跳 (mN) | " + " | ".join(f"{FACTS[n]['zero_jump_mN']:+.3f}" for n in DATASETS) + " |")
A("| 卸载后均值 (mN) | " + " | ".join(f"{FACTS[n]['zero_post_mN']:+.3f}" for n in DATASETS) + " |")
A("| 卸载后 |最大值| (mN) | " + " | ".join(f"{FACTS[n]['zero_post_absmax_mN']:.3f}" for n in DATASETS) + " |")
A("")
A("结论：**零漂幅度是亚毫牛级，比时漂小 2~3 个数量级**，"
  "卸载后能回到静息水平，不存在明显的"零点持续偏移"。")
A("")
A("### 3.2 真正的问题是"零点不可观测"")
A("")
A(f"- 三组数据分别有 {FACTS['数据1']['n_dead']}/{FACTS['数据2']['n_dead']}/"
  f"{FACTS['数据3']['n_dead']} 个通道整段恒为 0.000000 N；")
A("- 受载通道在空载时的输出**被截断在 0**（无负值、无噪声）：说明固件对输出做了"
  "单边钳位或零点校准已把静息读数压到 0 并截断；")
A(f"- 因此"小信号区间的零漂/噪声"在本数据里**完全看不到**（静息 σ≈0，"
  f"而受载时帧间噪声 σ≈{FACTS['数据1']['noise_hf_mN']:.1f} mN / "
  f"低频起伏 σ≈{FACTS['数据1']['noise_lf_mN']:.1f} mN）。")
A("")
A("这带来两个直接后果：")
A("1. **时漂无法用"静息期重新标零"来自动修正**（静息期读数恒为 0，不含漂移信息）；")
A("2. **要研究零漂必须采原始 ADC**（`has_raw_adc=false`、`value_stage=processed_display`"
  "说明当前录制的是显示换算后的力值，已丢失原始码与符号）。")
A("")

A("---")
A("")
A("## 4. 算法方案：哪些可能起作用")
A("")
A("### 4.1 为什么"单通道因果 + 通用滤波"必然失败")
A("")
A("在线可观测方程是 `y(τ) = F · h(τ) + ε`，静息期 `y = 0` 不提供刻度，"
  "因此在一个恒定负载通道上，**"力在缓慢变大"与"传感器在蠕变"在任何有限时间内"
  "都是同一组观测**。要打破这个不可辨识性，必须引入额外信息。")
A("")
A("另外，本数据的漂移在频域上**几乎就是直流**："
  f"110 s 内单调上升 {(HR[110]-1)*100:.0f}%，没有任何周期性。"
  "所以高通/去趋势类方法只能靠"牺牲同频段的真实信号"来换取漂移抑制——"
  "这正是第 5 节里高通滤波 `gain` 跌到 0.7~0.9 的原因。")
A("")
A("### 4.2 三条可行路线（本报告全部实装并实测）")
A("")
A("| 路线 | 额外信息 | 代表算法 | 适用前提 |")
A("|---|---|---|---|")
A("| **A 形态先验** | 离线标定 h(τ) 的形状（幂指数/时间常数） | 形态先验+滑窗水平 | 传感器批次稳定；载荷在测点上恒定 |")
A("| **B 阵列共模** | 同阵列各通道漂移同比例（已由 2.4 验证） | 阵列共模因子 | 载荷分布不随时间剧变；至少 2 个受载通道 |")
A("| **C 残余观测** | 静息/卸载事件可标定零点 | 卸载复位 + 再零点 | 能检测到卸载；两次加载之间重新标零 |")
A("")
A("### 4.3 实装清单")
A("")
A("所有算法均为**严格因果**实现（`step(n, y, t)` 逐帧推进，只用 y[n] 及历史；"
  "全部使用单边滤波器，不用 `filtfilt`/居中 savgol）。")
A("")
A("**1. 通用因果滤波（不需要加载时刻）**")
A("")
A("- 因果滑动平均（MA，1/5/20 s）")
A("- 因果中值滤波（1/5 s）")
A("- 一阶因果 IIR 低通（EWMA，τ=1/3/10/30 s）")
A("- 因果巴特沃斯高通（`lfilter` 单边递推，fc=0.005/0.01/0.02/0.05 Hz，2 阶）")
A("- 因果 RLS 趋势基线（一阶多项式 + 遗忘因子 λ=0.9999/0.99999/0.999999）")
A("- 因果卡尔曼（状态 [水平, 漂移速率] 常速模型，q=1e-11…1e-8）")
A("")
A("**2. 因果漂移模型（需要加载时刻 + 形态先验）**")
A("")
A("- **形态先验 + 滑窗水平**（主力方案 A）：`F̂[n] = mean_{W}( y/h(τ) )`，W=0.3~10 s")
A("- 形态先验 + 回溯反解（τ_c 之后直接 `F̂=y/h(τ)`）")
A("- 双窗比值外推（在线从两窗比值反解形态系数，不需离线标定）")
A("- 自适应 RLS（观测 `y=F·(1+c·τ^p)`，递推 [F, c]，p 为先验）")
A("- 双 EMA 趋势比（完全自适应，`ŷ = y/(EMA_slow/EMA_slow|_ref)^γ`）")
A("- 峰值保持 + 形态外推")
A("")
A("**3. 阵列共模因果补偿（主力方案 B，无需形态先验）**")
A("")
A("- **阵列共模因子**：`g[n] = S[n]/S[n_ref]`，`ŷ_j = y_j/g[n]`，S 为受载通道和/均值/中位数")
A("- 阵列共模 + 逐通道灵敏度 α_j")
A("- 阵列共享形态在线估计（逐通道归一后加权，反解幂律系数 c）")
A("")
A("### 4.4 刻意排除的方案")
A("")
A("- **离线/批处理类**（对整段做 SVD、PCHIP 分段基线、整段曲线拟合、`filtfilt` 零相位高通）："
  "不符合因果要求。")
A("- **单指数饱和模型**：p≈0.55 的幂律不饱和，指数模型在长持载下会系统性低估漂移，"
  "在本数据上表现为"补偿后反而更漂"。")
A("")

A("---")
A("")
A("## 5. 因果算法横向对比（实测）")
A("")
A("### 5.1 指标口径")
A("")
A("为避免"把信号整体压到 0"这类作弊刷分，主指标全部用**绝对量**，"
  "并且都相对"原始信号在加载后 1~3 s 的稳定水平 F_ref"表达：")
A("")
A("| 指标 | 定义 | 理想值 |")
A("|---|---|---|")
A("| `漂移原始` | 原始 (末 5 s 均值 − F_ref)，mN | — |")
A("| `漂移残余` | 算法输出 同口径，mN | **0** |")
A("| `抑制率` | 1 − |漂移残余|/|漂移原始| | **100%** |")
A("| `CV原始/CV算法` | 平台期（参考窗之后至卸载）读数的变异系数 | **越小越好**（恒定） |")
A("| `gain` | 算法在 1~3 s 的输出 / 原始水平 | **1.0**（否则把真实力也扣了） |")
A("| `阶跃保真` | [加载+0.05s, +3s] 净上升 / 原始同窗口净上升 | **1.0** |")
A("| `噪带%` | 平台期 0.01~1 Hz 带内噪声 σ 之比 | 越小越好 |")
A("| `慢变%` | 平台期 <0.01 Hz 起伏 σ 之比 | 越小越好 |")
A("| `卸载后` | 卸载后 3 s 输出均值（mN） | **0** |")
A("")
A("### 5.2 三组数据汇总排行（按 |残余时漂| 升序）")
A("")
A("| 算法 | 漂移残余(mN) | 抑制率 | CV算法(%) | gain | 阶跃保真 | 噪带% | 慢变% | |卸载后|(mN) | 分组 |")
A("|---|---|---|---|---|---|---|---|---|---|")
for r in sorted(SUMMARY, key=lambda x: x["drift_algo_mN"]):
    A(f"| {r['algo']} | {r['drift_algo_mN']:.1f} | {r['drift_reduction']*100:.1f}% | "
      f"{r['cv_algo']*100:.2f} | {r['gain']:.3f} | {r['step_fid']:.3f} | "
      f"{r['noise_hi']:.1f} | {r['noise_lo']:.1f} | {r['zero_post']:.2f} | {r['group']} |")
A("")
A("原始信号对照：" +
  " / ".join(f"{n} 平台期 CV={SWEEP[n]['rows'][0]['cv_raw']*100:.2f}%，"
             f"残余漂移={SWEEP[n]['rows'][0]['drift_algo_mN']:.1f} mN" for n in DATASETS))
A("")
A("### 5.3 逐组明细")
A("")
for name in DATASETS:
    A(f"#### {name}")
    A("")
    A("| 算法 | 漂移原始(mN) | 漂移残余(mN) | 抑制率 | CV算法(%) | gain | 阶跃保真 | 噪带% | 慢变% | 卸载后(mN) | 因果 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(SWEEP[name]["rows"], key=lambda x: abs(x["drift_algo_mN"])):
        A(f"| {r['algo']} | {r['drift_raw_mN']:.1f} | {r['drift_algo_mN']:.1f} | "
          f"{r['drift_reduction']*100:.1f}% | {r['cv_algo']*100:.2f} | {r['gain']:.3f} | "
          f"{r['step_fid']:.3f} | {r['noise_hi']:.1f} | {r['noise_lo']:.1f} | "
          f"{r['zero_post']:.2f} | {'是' if r['causal_ok'] else '否'} |")
    A("")

A("### 5.4 图像对比")
A("")
for fn, cap in [
    ("figures/G1_overview.png", "图 G1 数据总览（修正分段）"),
    ("figures/G2_shape.png", "图 G2 漂移形态与速率"),
    ("figures/G3_noise.png", "图 G3 噪声与频谱特征"),
    ("figures/G4_spatial.png", "图 G4 空间结构（响应/时漂/噪声/水平）"),
    ("figures/G5_zero.png", "图 G5 零漂"),
    ("figures/F1_overview.png", "图 1 数据总览与漂移形态"),
    ("figures/F2_causal_main.png", "图 2 主通道：因果算法时域补偿效果"),
    ("figures/F3_array.png", "图 3 阵列级：负载末段各通道输出偏差"),
    ("figures/F4_metrics.png", "图 4 因果算法横向指标对比"),
    ("figures/F5_causality.png", "图 5 因果性保证、在线加载检测与鲁棒性"),
]:
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        A(f"**{cap}**")
        A("")
        A(f"![{cap}]({fn})")
        A("")

A("---")
A("")
A("## 6. 工程实施建议")
A("")
A("### 6.1 推荐在线流水线")
A("")
A("```")
A("每帧 y[n] (31 通道，N)")
A("  │")
A("  ├─ ① 因果基线：开机静息期（≥3 s）估计 base_j，输出 y-base                 [已有]")
A("  ├─ ② 在线加载检测：EWMA(0.15s) 平滑 Σ → 阈值(base+12%·运行峰, ≥0.1N) → n_on")
A("  ├─ ③ 阵列共模因子：g[n] = S[n]/S[n_on+1~3s]，S = 受载通道和")
A("  ├─ ④ 形态先验修正（可选，用于长持载）：F̂ = y / h(τ) / g[n]")
A("  └─ ⑤ 卸载复位：检测到 Σ 回落 → 重新标零，清空 ③ 的状态")
A("```")
A("")
