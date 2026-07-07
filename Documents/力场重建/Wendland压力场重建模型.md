# v1.0.7 Wendland 压力场重建模型

## 概述

v1.0.7 将拟物 Layout 的 2D Smooth 渲染模式从旧"点高斯 Stamp 叠加"替换为"Wendland 紧支撑核中心点拟合 + Gaussian 后处理"。该模型将每个传感器中心点作为离散采样值，通过局部加权平均重建连续压力场。

## 核心思路

传感器读数是连续压力场在离散中心点位置上的采样值，不是力源。渲染不是模拟"力从传感器点向外扩散"，而是从离散采样重建连续场。

本模型让每个像素查询其周围一定半径内的传感器中心点，按紧支撑核加权平均，使同一值域的连续传感器区域呈现自然过渡，不产生点状局部峰。

## 数学公式

### Wendland C2 紧支撑核

```
phi(r) = (1 - r)^4 * (4r + 1),   r = clamp(dist / R, 0, 1)
```

- `r = 0`（像素在传感器中心）→ `phi(0) = 1.0`
- `r = 1`（像素在查询半径边界）→ `phi(1) = 0.0`
- `r > 1`（超出半径）→ `phi(r) = 0.0`
- 核函数在 `r=0` 和 `r=1` 处 C2 连续（函数值、一阶导数、二阶导数均连续），不会产生切面/片状伪影。

### 加权平均重建

```
value(px, py) = (sum(value_i * phi(r_i)) / sum(phi(r_i))) * min(1.0, sum(phi(r_i)))
```

其中 `r_i = dist((px,py), cell_i.centroid) / R`，`R = query_radius_mm`。

- `sum/denom` 为加权平均，保证颜色主要由压力值决定，不由 stamp 重叠数量决定。
- `min(1.0, denom)` 为 coverage 置信度衰减：边缘覆盖不足时自然渐隐，避免边缘无限外扩。

### Gaussian 后处理平滑

后处理采用可分离 1D 高斯卷积（先水平后垂直），kernel 不做归一化（中心权重保持 1.0），使恒定区域不受平滑改变：

```
kernel[i] = exp(-i^2 / (2 * sigma^2)),   sigma = post_gaussian_sigma (px)
```

## 算法流程

1. **坐标转换**：将 bounds（mm）映射到 grid 像素坐标
2. **空间分桶**：将每个 sensor cell 按质心像素坐标插入空间桶，桶大小为 `query_radius_mm * scale`
3. **逐像素计算**：对每个像素 (x, y)：
   - 确定该像素所在桶
   - 扫描该桶及 8 个相邻桶内的 sensor cell
   - 对每个 cell 计算距离 `d`，若 `d <= R` 则累加 `numer += value * phi(d/R)`, `denom += phi(d/R)`
   - 输出 `field = numer / denom * min(1, denom)`
4. **后处理平滑**（若 `post_gaussian_sigma >= 0.5`）：可分离 1D 高斯卷积
5. **Mask 裁剪**：将 mask 外像素置零

## 文件位置

`src/render/wendland_field.h`

## 配置结构

```cpp
struct WendlandFieldConfig {
    int grid_w = 420;              // 输出场图宽度 (px)
    int grid_h = 980;              // 输出场图高度 (px)
    double bounds_w_mm = 580.0;    // 布局宽度 (mm)
    double bounds_h_mm = 1350.0;   // 布局高度 (mm)
    double query_radius_mm = 60.0; // Wendland 核紧支撑半径 (mm)
    double post_gaussian_sigma = 1.5; // 后处理 Gaussian sigma (px)
};
```

## 默认参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `query_radius_mm` | `60.0` | 紧支撑范围，约 `2 × sensor_pitch` |
| `post_gaussian_sigma` | `1.5` | 后处理平滑，仅用于去噪 |

## 与旧点高斯模型对比

| 对比项 | 旧点高斯 Stamp | v1.0.7 Wendland |
|--------|--------------|------------------|
| 核函数 | `exp(-d²/2σ²)` 高斯 | `(1-r)⁴(4r+1)` Wendland C2 |
| 支撑范围 | 无边界（截断为 3σ） | 紧支撑（r>1 严格为零） |
| 累加方式 | 直接叠加 `field += value * weight` | 加权平均 `numer/denom` |
| 颜色受密度影响 | 是（重叠区域更亮） | 否（归一化消除密度伪影） |
| 局部峰数量 | ~100+ | ~4–5 |
| 片状伪影 | 无 | 无（C2 连续） |
| 边缘控制 | `coverage confidence` 无 | `min(1, denom)` 渐隐 |

## 性能

- 空间分桶避免全量距离计算：每个像素仅查询所在桶及相邻 8 桶
- 336 点坐垫数据在全分辨率下可实时渲染
- 后处理 Gaussian 采用可分离卷积，复杂度 O(w×h×kernel_size) 而非 O(w×h×kernel_size²)
