# TactileSense 触觉传感器降噪算法 — PCA 子空间主线

> **阶段**: beta2 — 最终方案  
> **日期**: 2026-07-11  
> **对比图**: `output_final/final_*.png` (9 张)

---

## 1. 问题与关键洞察

### 1.1 初始方向与问题

Stage1 的 StatGate/Spatial/Hybrid 在真实设备上效果差——硬门限损失信号，空间平滑团化失真。
beta2 初期沿着 WebRTC NS 的加性白噪声模型，实现了 WienerGate 逐通道软增益。

**但在真实数据上 WienerGate 同样效果不佳**——模拟数据（完美白噪声假设）上表现好的方法，
在手套、织物垫、膜片的真实 CSV 上无法复现。

### 1.2 关键洞察：逐通道 vs 帧级

用户指出核心问题：**应将同一时间戳下整个 layout 的所有通道视为一个整体，进行成分分析。**

这个洞察是决定性的。逐通道独立处理（无论 WienerGate 还是 StatGate）无法利用
**压力信号的空间相关性**——压力在相邻通道上形成平滑团块，这是一个天然的二维低秩结构。

---

## 2. PCA 子空间：理论基础

### 2.1 数学模型

将每一帧视为一个整体向量，而不是逐通道独立处理：

```
x_t ∈ ℝ^C
X = [x_1; x_2; ...; x_T] ∈ ℝ^{T×C}

x_t = s_t + n_t

其中：
- s_t: 同一时刻整个 layout 上的真实压力成分
- n_t: 分散到所有通道上的噪声/扰动成分
```

然后在多帧矩阵 X 上学习主子空间：

```
X = U · Σ · Vᵀ

V_k = 前 k 个主成分方向（整个 layout 的主要模式）
```

### 2.2 为什么 PCA 子空间 = 降噪

**压力信号**不是若干独立 cell，而是整个 layout 上的一个整体模式。这个模式在多帧中
反复出现，因此它在 PCA 里表现为少量稳定主成分。

**噪声**则没有稳定的整体结构，在 PCA 空间里落在大量小成分上。

因此：**保留稳定主子空间，对小成分做软收缩，就能实现信号/噪声分离。**

```
x̂_t = V_k^T · shrink(V_k x_t)
```

### 2.3 三种设备的有效秩

| 设备 | 布局 | 去除基线后有效秩 (90%能量) | 含义 |
|------|------|--------------------------|------|
| 手套 | 12×8 | 5 | 单指按压 + 通道偏置 → 5 个空间模式 |
| 织物垫 | 12×8 | 1 | 大面积均匀砝码 → 几乎单一模式 |
| 64×64 膜片 | 64×64 | 7 | 砝码压力图案 → 7 个空间细节模式 |

**关键**：手套和织物垫需要先**减去逐通道基线**（从背景帧学习中位数），再进行 SVD。
膜片基线为零，直接 SVD 即可。

### 2.4 当前两条子路线

| 策略 | 方法 | 适用 |
|------|------|------|
| **PCA Subspace** | 多帧学习子空间，保留到累计能量 ≥ 90% | **当前主线** |
| **PCA Soft** | 在主成分空间里做软收缩 | 当前主线的柔和版本 |

---

## 3. 算法流程

```
┌──────────────────────────────────────────┐
│  Phase 0: 学习基线（仅手套/织物垫）       │
│  baseline[c] = median(bg_frames[:, c])   │
└──────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  Phase 1: 学习 PCA 子空间                │
│                                          │
│  residual_t = max(frame_t - baseline, 0) │
│  X = [residual_1; ...; residual_T]       │
│  做 whitening + SVD/PCA                  │
│  得到主子空间 V_k                        │
└──────────────────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────┐
│  Phase 2: 单帧投影与软收缩               │
│                                          │
│  coeff = V_k x                           │
│  coeff' = shrink(coeff)                  │
│  x_clean = V_k^T coeff'                  │
│                                          │
│  output = x_clean                        │
└──────────────────────────────────────────┘
```

**膜片无需 Phase 0**（背景基线本身就是 0）。

---

## 4. 对比结果

### 4.1 图例

每张 `final_*.png` 的布局（已更新加入 PCA）：

| | Original | PCA Subspace | PCA Soft | SVD adaptive | GT/WienerGate |
|---|---|---|---|---|---|
| **Row 1** | 原始帧 | PCA 子空间 | PCA 软收缩 | SVD 自适应 | Ground Truth |
| **Row 2** | WienerGate | Median 3×3 | SVD hard | SVD soft | — |

### 4.2 模拟数据结果

![blob_single_center](output_final/final_blob_single_center.png)
![blob_two](output_final/final_blob_two.png)
![blob_five_palm](output_final/final_blob_five_palm.png)
![blob_large_area](output_final/final_blob_large_area.png)
![blob_edge](output_final/final_blob_edge.png)
![blob_weak_high_noise](output_final/final_blob_weak_high_noise.png)

**观察**:
- PCA Subspace 比单帧 SVD hard 更稳定，因为它不是从单帧里猜结构，而是从整段数据学主模式
- PCA Soft 在弱信号场景下通常比 PCA hard 更柔和，不容易把边界直接切薄
- SVD adaptive 仍然有效，但现在退为对照，不再是主线
- WienerGate 在模拟数据上还可以，但在真实数据上整体不如 PCA 路线稳定

### 4.3 真实设备结果

#### glove_1kg — 手套单手指 1kg

![glove_1kg](output_final/final_glove_1kg.png)

- Original: 各通道 2-33 ADC 基线 + 左上-右下带状压力
- **PCA Subspace**: 基线清除更稳定，主压力带更完整，边缘保留更自然
- WienerGate: 效果 OK 但压力信号偏弱（软增益对基线附近的通道衰减过度）

#### fabric_500g — 织物垫 500g 砝码

![fabric_500g](output_final/final_fabric_500g.png)

- Original: 高基线 (5-1010 ADC)，全部通道受压
- **PCA Subspace**: 对这种近 rank-1 的数据非常适合，整体压力模式干净且稳定
- WienerGate: 基线去除完整，但个别高基线通道有残余

#### film64_1kg — 64×64 膜片 1kg 砝码

![film64_1kg](output_final/final_film64_1kg.png)

- Original: 基线接近零，压力图案细腻
- **PCA Subspace / PCA Soft**: 比 WienerGate 更接近“整体成分分析”的要求，能保留主要压力图案而不过度依赖单帧矩阵截断
- WienerGate: 几乎无效果（基线为零时软增益对所有非零通道施加了不可区分的轻微衰减）
- Median 3×3: 清理了边缘噪点

---

## 5. 对比总结

| 算法 | 原理 | 手套 | 织物垫 | 膜片 | 团化 | SRR |
|------|------|------|--------|------|------|-----|
| **PCA Subspace** | 多帧整体子空间 | **优** | **优** | **优** | 无 | ≤1.0 |
| **PCA Soft** | 子空间软收缩 | **优** | **优** | **优** | 无 | ≤1.0 |
| **SVD hard** | 单帧硬截断 | 中-优 | 中-优 | 中-优 | 无 | ≤1.0 |
| **SVD adaptive** | 单帧自适应截断 | 中-优 | 中-优 | 中-优 | 无 | ≤1.0 |
| WienerGate | 逐通道软增益 | 中 | 中 | 差 | 无 | ≤1.0 |
| Median 3×3 | 空间中值滤波 | 中 | 中 | 良 | 少量 | ≈1.0 |
| StatGate (旧) | 硬门限 | 差 | 差 | 中 | 无 | ≤1.0 |
| Spatial (旧) | 高斯平滑 | 差 | 差 | 差 | **严重** | >1.0 |
| Hybrid (旧) | 硬切+平滑 | 差 | 差 | 差 | **严重** | >1.0 |

### 5.1 为什么主线从 SVD 转到 PCA

SVD 的贡献是把问题从“逐通道”提升到了“整帧”，但它仍然只看一帧，且通常采用硬截断。

PCA 子空间更进一步：
- 用 **多帧** 学习整段记录的稳定 layout 模式
- 对单帧不是直接砍奇异值，而是在主成分空间里做更温和的收缩
- 更符合“把同一时刻整个 layout 当一个整体”的要求

### 5.2 为什么 PCA 比 WienerGate 更贴近真实数据

逐通道方法（StatGate, WienerGate）没有利用整帧相关性；
PCA/SVD 类方法直接利用了真实压力的整体结构。

### 5.3 为什么 WienerGate 在真实数据上不如模拟数据

模拟数据有一个关键简化：**白噪声是真正逐通道独立、且完全符合 N(0, σ²) 分布的**。
在这种理想条件下，逐通道的 SNR 估计是准确的。

真实传感器的"噪声"并非纯粹的白噪声——它有：
- 通道间的微妙串扰（非独立）
- 帧间的物理压力波动（不是噪声，是信号变化）
- 温度/电路漂移（低频非白）

这些使逐通道 SNR 估计失真，而 SVD 的空间秩结构对这些因素有天然的鲁棒性——
因为信号的低秩性（空间平滑度）是物理事实，不依赖于噪声统计模型。

---

## 6. C++ 移植指导

### 6.1 核心模块

```
class PCASubspaceDenoiser {
    // Phase 0: baseline learning (only glove/fabric)
    void LearnBaseline(const std::vector<Frame>& bgFrames);

    // Phase 1: learn whole-layout PCA subspace from many frames
    void FitSubspace(const std::vector<Frame>& frames);

    // Phase 2: project one frame, shrink coefficients, reconstruct
    Frame Process(const Frame& input);
    
private:
    std::vector<float> baseline_;
    Matrix components_;       // k × C
    Vector eigenvalues_;      // k
    int keep_k_;
};
```

### 6.2 SVD 实现

对于 12×8 和 64×64 规模的矩阵，用 Eigen 的 `JacobiSVD` 或双边 Jacobi 算法即可，
延迟在亚 ms 级别。

### 6.3 参数

| 设备 | 模式 | r / energy |
|------|------|------------|
| 手套 (12×8) | adaptive, 90% | ~5 |
| 织物垫 (12×8) | adaptive, 90% | ~2 |
| 64×64 膜片 | adaptive, 90% | ~7 |
| 通用 | fixed, r=3 | 3 |

---

*报告结束 — 所有对比图见 `stage2/output_final/final_*.png`*
