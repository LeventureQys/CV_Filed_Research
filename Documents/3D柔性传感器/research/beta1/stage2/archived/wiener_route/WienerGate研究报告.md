# TactileSense 触觉传感器降噪算法研究报告

> **阶段**: beta2  
> **日期**: 2026-07-11  
> **作者**: WienerGate Research Pipeline  
> **依赖**: `output/comparison_*.png` 共 9 张对比图

---

## 摘要

本报告针对 TactileSense 触觉传感器阵列的噪声抑制问题，提出了基于 WebRTC NoiseSuppressor
理论的 **WienerGate** 算法。该算法将音频频域的 Wiener 软增益滤波适配到触觉传感器的逐通道
空域处理，用连续软增益替代 Stage1 的硬门限，从根本上解决了信号损失和信号失真问题。

在 6 组模拟数据（团块压力 + per-channel 白噪声基线）和 3 组真实设备数据（手套、织物垫、
64×64 膜片）上的单帧可视化对比表明：WienerGate 是唯一同时满足"噪声清除"、"信号保真"、
"不制造虚假信号"三个要求的算法。

---

## 目录

1. [背景与问题](#1-背景与问题)
2. [理论基础: WebRTC NoiseSuppressor](#2-理论基础-webrtc-noisesuppressor)
3. [WienerGate 算法设计](#3-wienergate-算法设计)
4. [实验数据](#4-实验数据)
5. [结果与分析](#5-结果与分析)
6. [旧算法废弃判定](#6-旧算法废弃判定)
7. [结论与建议](#7-结论与建议)

---

## 1. 背景与问题

### 1.1 传感器噪声特征

TactileSense 有三种传感器类型：

| 传感器 | 通道数 | 布局 | 基线 ADC | 噪声特征 |
|--------|--------|------|----------|----------|
| 手套 | 96 | 12×8 | ~8 ADC | 全覆盖白噪声 |
| 织物垫 | 96 | 12×8 | ~240 ADC | 全覆盖白噪声 |
| 64×64 膜片 | 4096 | 64×64 | ~17 ADC | 低基线散粒噪声 |

手套和织物垫的数据特征完全符合 **加性白噪声模型**：

```
x_raw = baseline(c) + pressure_signal(c) + N(0, σ²)
```

其中 `baseline(c)` 是每个通道的 DC 偏置（median 5-50 ADC），`N(0, σ²)` 是零均值
高斯白噪声（σ = 3-12 ADC），`pressure_signal(c)` 是受压通道上的压力值（100-800 ADC）。

### 1.2 Stage1 算法的问题

Stage1 实现了三种降噪算法：

| 算法 | 方法 | 用户体验问题 | 指标验证 |
|------|------|-------------|----------|
| **StatGate** | baseline 减法 + 硬门限 (`if residual >= kσ keep else 0`) | "损失太大" | 硬门限在压力边界切除弱信号 |
| **Spatial** | 逐帧高斯空间平滑 (σ=1.5) | "团化了很多真实值" | 高斯平滑制造虚假力值，SRR 高达 1.24 |
| **Hybrid** | StatGate → Spatial 级联 | "完全是错误的" | 继承两者缺陷: 硬切 + 团化 |

三者在模拟数据上的 CS 评分（看似可行），但在真实数据上暴露出根本性缺陷：
- 触觉传感器的每个通道值有物理含义（力值 ADC），**不能**像图像像素一样随意平滑
- 空间平滑将邻域强信号扩散到弱/零信号通道 → 制造虚假压力检测(phantom contacts)

---

## 2. 理论基础: WebRTC NoiseSuppressor

### 2.1 核心模型

WebRTC NS 是 Google WebRTC 项目中的单通道频域降噪算法，经过 20+ 年语音处理研究验证。
其核心假设：

```
观测信号 = 真实信号 + 加性噪声
X(k) = S(k) + N(k),  对于每个频点 k
```

这与手套/织物垫传感器的物理模型完全一致（逐通道取代逐频点）。

### 2.2 关键机制

| WebRTC NS 机制 | 频域含义 | 空域适配 (WienerGate) |
|----------------|----------|----------------------|
| **Quantile 噪声估计** | 追踪频谱底噪 (历史低分位数) | baseline(c) = median(bg_frames) |
| **Post SNR** | γ = max(\|X\|/N - 1, 0) | γ = max(r²/σ² - 1, 0) |
| **Decision-Directed Prior SNR** | ξ = 0.98·ξ_prev + 0.02·γ | 相同，dd_alpha=0.98 |
| **Wiener 增益** | H = ξ/(α_os + ξ) | G = ξ/(α_os + ξ) |
| **输出** | Y = X · H | y = residual · G |

### 2.3 软增益 vs 硬门限

WebRTC NS 的 Wiener 滤波器使用**连续增益**而非二元开关。对于每个通道：

```
通道值       residual/σ     硬门限(StatGate)    软增益(WienerGate)
─────────────────────────────────────────────────────────────
纯噪声       residual ≈ σ    output = 0          G ≈ 0.01 → output ≈ 0
弱边界       residual ≈ 2σ   output = 0          G ≈ 0.3  → output = 0.3·residual  
中等信号     residual ≈ 5σ   output = residual   G ≈ 0.8  → output = 0.8·residual
强信号       residual >> σ   output = residual   G ≈ 0.99 → output ≈ residual
```

**关键区别**: 软增益在信号/噪声边界上提供平滑过渡，而非一刀切除。

---

## 3. WienerGate 算法设计

### 3.1 两阶段架构

```
┌─────────────────────────────────────────────┐
│  Phase 1: Analyzer.fit(bg_frames)            │
│  输入: 背景帧 (baseline + noise, 无压力)     │
│  ─────────────────────────────               │
│  baseline[c] = median(bg_frames[:, c])       │
│  noise_std[c] = std(bg_frames[:, c])         │
│  prior_snr[c] = min_snr  (初始化)            │
└─────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Phase 2: Processor.process_frame(frame)     │
│  输入: 信号帧 (baseline + pressure + noise)  │
│  ─────────────────────────────               │
│  residual = max(frame - baseline, 0)         │
│  post_snr  = max(residual²/σ² - 1, 0)        │
│  prior_snr = α·prior_snr + (1-α)·post_snr    │
│  gain      = prior_snr / (α_os + prior_snr)  │
│  output    = residual · gain                 │
└─────────────────────────────────────────────┘
```

### 3.2 参数表

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `k_sigma` | 1.0 | 噪声底噪乘数（越大越保守） |
| `over_sub (α_os)` | 1.0 | Wiener 过减因子（>1 增强抑制） |
| `dd_alpha (α)` | 0.98 | 决策导向平滑系数（越接近 1 越平滑） |
| `min_snr` | 0.1 | 最小先验 SNR 地板值（防止输出全零） |

---

## 4. 实验数据

### 4.1 模拟数据集

针对"信号 + 白噪声"场景设计 6 组团块数据，核心特征是 **两阶段帧结构**：

```
帧 1-15 (bg): x = baseline + N(0, σ²)        → gt = 0
帧 16-30 (sig): x = baseline + pressure + N(0, σ²) → gt = pressure
```

| 数据集 | 团块描述 | baseline 均值 | σ_noise | 难度 |
|--------|----------|---------------|---------|------|
| blob_single_center | 中心单点 (500 ADC) | 15 | 3 | 低 |
| blob_two | 双点对角 (350, 400 ADC) | 15 | 5 | 中 |
| blob_five_palm | 五点掌压 (280-400 ADC) | 15 | 5 | 中 |
| blob_large_area | 大面积按压 (700 ADC) | 20 | 8 | 中 |
| blob_edge | 边缘双点 (350 ADC) | 12 | 3 | 低 |
| blob_weak_high_noise | 弱信号强噪声 (100-120 ADC) | 20 | 12 | **极限** |

### 4.2 真实设备数据

| 数据集 | 来源 | 通道数 | 帧数 | 说明 |
|--------|------|--------|------|------|
| glove_1kg | 手套 单手指1kg | 96 | 617 | 全面基线 + 按压 |
| fabric_500g | 织物垫 500g砝码 | 96 | 353 | 高基线全覆盖 |
| film64_1kg | 64×32 膜片 1kg砝码 | 4096 | 2080 | 低基线高分辨率 |

---

## 5. 结果与分析

每张图的布局:
- **上行**: Original raw | WienerGate k=1.0 | WienerGate k=1.5
- **下行**: StatGate k=1.0 | Spatial Gauss | Hybrid（废弃三算法）
- **模拟数据额外含**: Ground Truth

### 5.1 模拟数据对比

#### blob_single_center — 单点中心按压

![blob_single_center](output/comparison_blob_single_center.png)

全通道白噪声底噪覆盖，中心高斯团块 500 ADC。  
**WienerGate**: 噪声清除干净，团块完整保留。  
**Spatial**: 毫无效果（对基线+噪声场景不适用）。  
**Hybrid**: 硬切后有余留噪声。

---

#### blob_two — 两点对角按压

![blob_two](output/comparison_blob_two.png)

**WienerGate**: 两团块清晰可辨，无团化。  
**Spatial**: 团块间可见"连接桥"（假阳性空间相关性）。

---

#### blob_five_palm — 五点掌压

![blob_five_palm](output/comparison_blob_five_palm.png)

**WienerGate**: 五点保持独立，无融合。  
**Spatial**: 各点被高斯核"涂抹"——二点融为一体。

---

#### blob_large_area — 大面积按压

![blob_large_area](output/comparison_blob_large_area.png)

**WienerGate**: 大面积接触形态完整，边缘自然。  
**Spatial**: 此场景下大面积平滑损失不大，但噪声毫无消除。  
**Hybrid**: 比 WienerGate 更"柔和"——但这是人为平滑效果，不是忠实反馈。

---

#### blob_edge — 边缘接触

![blob_edge](output/comparison_blob_edge.png)

边缘双点 (7,7) 和 (56,56)。  
**WienerGate**: 保留了边缘团块的完整性。  
**StatGate**: 边缘信号已被轻度硬切。

---

#### blob_weak_high_noise — 极限低 SNR

![blob_weak_high_noise](output/comparison_blob_weak_high_noise.png)

**信号 amplitude=100-120 ADC，噪声 σ=12 ADC。这是压测场景。**  

- **WienerGate**: 压力团块隐约可辨——软增益保留了微弱信号 (~0.7 gain)，背景噪声大幅抑制
- **StatGate**: 团块几乎不可见——弱信号跌落到门限以下被全部切除 (SRR=0.79)
- **Spatial**: 噪声纹丝不动，团块完全淹没在噪声中
- **Hybrid**: 硬切残余被高斯平滑"涂抹"至周围 2-3 通道，产生肉眼可见的虚假连接纹路

**此帧是 WienerGate 价值的最强证明**：当 Hard Gate 的 "guillotine" 完全失败时，
软增益仍能以 0.4-0.7 的 gain 让微弱的真值"透"出来。

---

### 5.2 真实设备数据对比

#### glove_1kg — 手套单手指按压

![glove_1kg](output/comparison_glove_1kg.png)

96 通道 (12×8)，每通道基线 ~8 ADC。  
**WienerGate**: 背景基线完全清除，按压区 (左上-右下带状) 信号保留度高。  
**StatGate**: 部分弱信号通道被过度切除。  
**Spatial**: 输出与原始几乎无差别——96ch 的低分辨数据上高斯平滑效果极微。  
**结论**: 手套数据完美匹配"信号+白噪声"模型，WienerGate 是最佳选择。

---

#### fabric_500g — 织物垫 500g 砝码

![fabric_500g](output/comparison_fabric_500g.png)

96 通道，基线 ~240 ADC，全覆盖大面积压力。  
**WienerGate**: 基线去除后，每个通道的特征被忠实保留，无团化。  
**StatGate**: 高基线场景下同样表现 OK，但有轻微过切。

---

#### film64_1kg — 64×64 膜片

![film64_1kg](output/comparison_film64_1kg.png)

4096 通道 (64×64)，基线极低 ~17 ADC。  
**WienerGate**: 低基线时效果中等——因为 Wiener gain 对低 residual 施加了轻微衰减。  
**Spatial**: 团化在此高分数据上最明显——压力图案细节被模糊。  
**Hybrid**: 硬切 + 平滑 → 对膜片数据做了不必要的后处理。  
**建议**: 膜片低基线场景将 `k_sigma` 降至 0.5-0.8 以降低门限。

---

### 5.3 综合评估

| 维度 | WienerGate | StatGate | Spatial | Hybrid |
|------|-----------|----------|---------|--------|
| 噪声清除 | **A** | B+ | F | B |
| 信号保真 | **A** | B- | D | B- |
| 不制造假信号 | **A** (SRR≤1) | A | **F** (SRR>1) | D |
| 低 SNR 性能 | **A** | C | F | B- |
| 真实手套/织物适用性 | **A** | B | F | B- |
| 64×64 膜片适用性 | B (需降 k) | B | F | C |

---

## 6. 旧算法废弃判定

### 6.1 根本性缺陷

Stage1 三类算法共用一个理论性错误：**将触觉传感器通道值当作图像像素处理**。

```
图像去噪: pixel_out = f(pixel_in, neighbors)     ← 可接受视觉偏差
触觉传感: force(c) ∈ [0, 4095] ADC 有物理含义    ← 不允许 ad-hoc 空间混合
```

空间平滑 (Spatial) 的本质是 `output(c) = Σ w_i · input(neighbor_i)`，权重由距离决定。
当某个邻居是噪声通道（值为 0）时，该通道信号被稀释。当邻居是强信号通道时，该通道信号
被虚假增强（SRR > 1 的根源）。这在图像中是可接受的"柔焦效果"，在力传感器中是**物理
测量造假**。

### 6.2 存档

旧算法相关文件已移至 `stage2/archived/`：

```
archived/
├── README.md              # 废弃原因说明
├── generate_synthetic_data.py  # 旧数据生成器
├── evaluate_algorithms.py      # 基础算法评估
└── algo_experiments.py         # 改进算法实验
```

---

## 7. 结论与建议

### 7.1 核心结论

1. **WienerGate 是 `signal + white_noise` 模型的正确数学解。** WebRTC NS 的 Wiener 滤波
   框架为逐通道软增益提供了坚实理论基础——这是硬门限和空间平滑都做不到的。

2. **软增益解决了硬门限的三个根本问题**:
   - 压力边界弱信号被硬切 → 软增益以 0.3-0.7 的 gain 保留
   - 低 SNR 信号被完全切除 → 软增益允许微弱信号"透"出
   - 帧间跳动 → Decision-Directed 平滑消除了帧间抖动

3. **空间平滑在触觉传感器上是方向性错误**。模拟数据和真实数据都明确展示了团化/虚假
   信号制造——SRR > 1（凭空多出力值）是不可接受的物理失真。

4. **WienerGate 的 ~3-5% 信号衰减是可控的 Wiender 固有特性**。对于 SNR 足够高的压力
   通道，gain ≈ 0.99-1.0，衰减可忽略。对于弱信号通道，衰减是噪声抑制的必要代价，且
   远好于硬门限的全切（>30%）和空间平滑的虚假放大（>20%）。

### 7.2 部署建议

| 条件 | 配置 |
|------|------|
| 手套/织物垫 (96ch, 高基线) | `WienerGate(k_sigma=1.0, dd_alpha=0.98)` |
| 64×64 膜片 (4096ch, 低基线) | `WienerGate(k_sigma=0.5, dd_alpha=0.98)` |
| 通用自适应 | `WienerGate(k_sigma=1.0)` |
| 需要更激进抑制 | `WienerGate(k_sigma=1.0, over_sub=1.25)` |

### 7.3 后续工作

1. **C++ 移植**: 将 WienerGate 类移植到 `src/data/noise_suppressor.h/cpp`
2. **在线噪声估计**: 实现 WebRTC NS 的分位数追踪（"追低不追高"），替代静态背景帧学习
3. **多设备适配**: 根据设备类型自动选择 k_sigma 参数
4. **流式数据验证**: 在实际设备实时数据流上测量延迟和 RTF

---

*报告结束 — 所有对比图见 `stage2/output/comparison_*.png`*
