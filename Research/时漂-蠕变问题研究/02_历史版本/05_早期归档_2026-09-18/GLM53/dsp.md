# 柔性触觉压感阵列抗蠕变漂移与动态力保真 DSP 补偿方案指南

---

## 1. 核心问题与矛盾分析

在柔性压阻/电容触觉传感器阵列（如 $10 \times 10$ 面阵）中，施加恒定载荷（如局部 $5 \times 5$ 区域受力）后，读数随时间缓慢漂移。

### 1.1 核心痛点与限制条件
* **不可使用常规高通滤波/基线滑动消除：** 高通滤波或简单基线消除本质上是对直流信号（DC）进行衰减。一旦使用，恒定静压力（如持续施加 10N）会被逐渐抹杀归零，导致无法测量静态压力。
* **瞬态力跳变不可失真：** 突加负载（如 $10\text{N} \to 15\text{N}$，ADC 从 $1000 \to 1200$）的变化幅度不能被低通平滑钝化，更不能被当成蠕变扣除。
* **多像素空间应力扩散（Crosstalk）：** 在 $10 \times 10$ 阵列上按压 $5 \times 5$ 区域时，由于聚合物基底的弹性连续性，周边未受压像素也会受到牵拉形变，同时伴随时间维度的黏弹性蠕变。

### 1.2 物理本质
1. **黏弹性蠕变与应力松弛（Viscoelastic Creep & Stress Relaxation）：** 传感器基底（PDMS、聚氨酯 PU、硅胶等）具有高分子聚合物特性，应力应变关系具有时间依赖性。
2. **接触微结构渗流网络重组（Percolation Network Reorganization）：** 在微观导电颗粒接触面，压力使导电通路瞬时增加；在恒定压力下，微突起在微米尺度继续缓慢塑性流动，微接触面积逐渐扩大，导致阻抗继续缓慢降低（ADC 读数漂移）。

---

## 2. 动力学建模：将蠕变视为动态线性系统

解决该问题的关键是将传感器的响应拆分为：**静态力-电非线性映射** 与 **内部时间动力学环节**。

```
真实力 F(t) ───> [ 静态非线性映射 Φ(·) ] ───> 瞬时等效信号 u(t) ───> [ 线性动态蠕变环节 H(s) ] ───> 测量值 ADC(t)
```

### 2.1 广义 Kelvin-Voigt / Wiechert 黏弹性模型
对于一个瞬间阶跃力 $F(t) = F_0 \cdot \mathbf{1}(t)$，传感器的输出响应可以解耦为**瞬态弹性响应**与**多阶黏弹性延时响应**：

$$ADC(t) = ADC_{\text{elastic}} \left( 1 + \sum_{i=1}^{N} a_i \left(1 - e^{-t / \tau_i}\right) \right)$$

* $ADC_{\text{elastic}}$：弹性瞬间阶跃响应（瞬时弹性形变，无滞后，对应突变力响应）。
* $a_i$：第 $i$ 阶黏弹性蠕变模态的幅值权重。
* $\tau_i$：第 $i$ 阶松弛时间常数。通常聚合物表现为双时间常数分布：
  * **快蠕变分量：** $\tau_1 \approx 0.5 \sim 3\text{ s}$（接触面微结构快速适应）。
  * **慢蠕变分量：** $\tau_2 \approx 30 \sim 300\text{ s}$（高分子链段长程滑移）。

在拉普拉斯域（$s$ 域），正向动态传递函数为：

$$H(s) = \frac{ADC(s)}{ADC_{\text{elastic}}(s)} = 1 + \sum_{i=1}^{N} \frac{a_i}{1 + s \tau_i}$$

以二阶蠕变模型为例，化简为标准有理多项式：

$$H(s) = K \cdot \frac{(s + z_1)(s + z_2)}{(s + p_1)(s + p_2)}$$

其中极点为系统固有的松弛衰减速率：$p_1 = \frac{1}{\tau_1}$，$p_2 = \frac{1}{\tau_2}$。

---

## 3. DSP 核心方案：动态逆传递函数补偿器

### 3.1 逆系统设计（Inverse Filter）
为了消除长时间蠕变，同时保留瞬态跳变幅度和静态真实力，需构建正向系统的逆滤波器 $G_{\text{comp}}(s) = H^{-1}(s)$：

$$G_{\text{comp}}(s) = \frac{1}{H(s)} = \frac{1}{K} \cdot \frac{(s + p_1)(s + p_2)}{(s + z_1)(s + z_2)}$$

#### 核心约束（保真原则）：
1. **零极点对消：** 逆滤波器的零点完全对消原系统的极点，从而抵消指数慢爬升动态。
2. **严格保持直流增益归一化：** 必须满足：
   $$G_{\text{comp}}(0) = 1$$
   这保证了不论外力维持多长时间，滤波后的稳态直流幅度与力刚加上的瞬间初始弹性幅度 $ADC_{\text{elastic}}$ 严格相等。
3. **高频增益限制：** 逆系统在高频处应保持稳定有限增益，避免放大高频 ADC 噪声。

### 3.2 离散化（Tustin 双线性变换）
采用双线性变换法，令采样周期为 $T_s$：

$$s = \frac{2}{T_s} \frac{1 - z^{-1}}{1 + z^{-1}}$$

将 $s$ 带入 $G_{\text{comp}}(s)$，可得二阶离散 IIR 滤波器的标准脉冲传递函数：

$$G_{\text{comp}}(z) = \frac{Y(z)}{X(z)} = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$$

其中：
* 输入 $X[k]$ 为当前采样点原始数据 $ADC[k]$；
* 输出 $Y[k]$ 为消除蠕变漂移后的真实弹性响应等价值 $ADC_{\text{corrected}}[k]$。

### 3.3 对应差分方程（可直接在 MCU / DSP 运行）
$$Y[k] = b_0 X[k] + b_1 X[k-1] + b_2 X[k-2] - a_1 Y[k-1] - a_2 Y[k-2]$$

#### 行为表现：
* **工况 A（瞬间突加力 $10\text{N} \to 15\text{N}$）：**
  $X[k]$ 从 1000 瞬间跳至 1200。因差分方程高频响应等于高频增益，输出 $Y[k]$ 同步无延迟跳变至 1200，**动态阶跃完全保留**。
* **工况 B（15N 持续保压 10 分钟）：**
  原始数据 $X[k]$ 因蠕变从 1200 缓慢上升至 1350。逆滤波器的低通零极点动态衰减作用实时抵消上扬趋势，输出 $Y[k]$ 严格锁死并稳定在 1200，**静态力漂移被彻底抹平**。

---

## 4. 空间维度的解耦（解决 10x10 阵列中 5x5 受压扩散）

当压力集中在局部 $5 \times 5$ 时，聚合物的弹性耦合会导致受压边缘周围像素产生虚假的“牵引读数”。

完整的处理流水线应当是**时空解耦（Spatio-Temporal Decoupling）**：

```
原始 10x10 阵列 ADC 矩阵
          │
          ▼
┌──────────────────────────────────────────────┐
│  时间域 IIR 动态逆补偿 (各像素独立并行运行)   │  ──> 消除材料内部黏弹性时间漂移
└──────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────┐
│  静态 ADC-力值标定映射 (LUT / 多项式校准)      │  ──> 将校正后的 ADC 转换为等效应力矩阵
└──────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────┐
│  空间二维逆卷积 (PSF 逆滤波)                 │  ──> 解除 5x5 到周边基底的横向应力牵连
└──────────────────────────────────────────────┘
          │
          ▼
真实接触力分布矩阵 F_true (10x10)
```

### 空间 PSF 反卷积原理
将阵列响应在空间上建模为二维高斯点扩散函数（Point Spread Function, PSF）：

$$F_{\text{measured}}(x, y) = F_{\text{true}}(x, y) \ast \ast \, PSF(x, y)$$

通过微型 $3 \times 3$ 二维空间高通卷积核（拉普拉斯算子形）进行反卷积，即可压制受压区域边缘的边缘溢出形变。

---

## 5. 工程落地的快速实施步骤

### 步骤 1：阶跃响应辨识（离线参数标定）
1. 使用机械压力试验机（力传感器作为真值基准），对传感器施加一个**快速阶跃力**（如在 $50\text{ ms}$ 内加载至 $10\text{N}$）。
2. 保持压力严格恒定 5~10 分钟，以固定采样率（如 $100\text{ Hz}$）记录原始 $ADC(t)$ 曲线。
3. 对数据进行最小二乘非线性拟合（MATLAB / Python `scipy.optimize.curve_fit`）：
   $$ADC_{\text{norm}}(t) = 1 + a_1 (1 - e^{-t/\tau_1}) + a_2 (1 - e^{-t/\tau_2})$$
4. 获得物理参数：$a_1, \tau_1, a_2, \tau_2$。

### 步骤 2：生成离散滤波器系数
利用附录提供的 Python 脚本，输入标定得到的参数以及系统采样率 $f_s$（如 $50\text{ Hz}$），直接输出二阶 IIR 滤波器的分子分母系数 $[b_0, b_1, b_2]$ 和 $[a_1, a_2]$。

### 步骤 3：单片机 / 嵌入式平台实时运行
在 MCU 采集完一帧 $10 \times 10$ 数据后，遍历每一个像素单元执行差分递推。

---

## 6. 核心实现代码

### 6.1 Python 离线参数辨识与系数计算脚本

```python
import numpy as np
from scipy import signal

def calculate_creep_compensator_coefficients(a1, tau1, a2, tau2, fs):
    """
    计算二阶黏弹性逆补偿滤波器的 IIR 系数 (Tustin 双线性变换)
    
    参数:
        a1, a2: 蠕变模态幅值比
        tau1, tau2: 松弛时间常数 (秒)
        fs: 系统采样频率 (Hz)
    返回:
        b: 分子多项式系数 [b0, b1, b2]
        a: 分母多项式系数 [1.0, a1_coeff, a2_coeff]
    """
    # 连续域正向系统 H(s) = 1 + a1/(1 + s*tau1) + a2/(1 + s*tau2)
    # 通分展开为标准多项式: H(s) = (N2*s^2 + N1*s + N0) / (D2*s^2 + D1*s + D0)
    
    D2 = tau1 * tau2
    D1 = tau1 + tau2
    D0 = 1.0
    
    N2 = D2
    N1 = (1.0 + a1) * tau2 + (1.0 + a2) * tau1
    N0 = 1.0 + a1 + a2
    
    # 逆滤波器 G_comp(s) = 1 / H(s) = D(s) / N(s)
    # 并强制令 DC 增益归一化: G_comp(0) = 1
    gain_corr = N0 / D0
    
    num_s = [D2 * gain_corr, D1 * gain_corr, D0 * gain_corr]
    den_s = [N2, N1, N0]
    
    # 双线性变换 (Tustin)
    b_z, a_z = signal.bilinear(num_s, den_s, fs=fs)
    
    # 归一化 a0 = 1.0
    b_z = b_z / a_z[0]
    a_z = a_z / a_z[0]
    
    return b_z, a_z

# 示例标定参数
fs = 50.0  # 采样率 50 Hz
a1, tau1 = 0.15, 1.2    # 快速蠕变: 漂移 15%，时间常数 1.2 秒
a2, tau2 = 0.20, 45.0   # 慢速蠕变: 漂移 20%，时间常数 45.0 秒

b, a = calculate_creep_compensator_coefficients(a1, tau1, a2, tau2, fs)
print("IIR 补偿器系数:")
print(f"b = [{b[0]:.6f}, {b[1]:.6f}, {b[2]:.6f}]")
print(f"a = [{a[0]:.6f}, {a[1]:.6f}, {a[2]:.6f}]")
```

### 6.2 嵌入式 C 语言阵列实时补偿器实现

```c
#include <stdint.h>

#define ROWS 10
#define COLS 10

// 滤波器历史状态结构体 (Direct Form II Transposed，节省内存且抗溢出)
typedef struct {
    float d1;
    float d2;
} IIR_State;

typedef struct {
    float b0, b1, b2;
    float a1, a2;
} IIR_Coeffs;

// 全局系数与各像素状态矩阵
static IIR_Coeffs g_comp_coeffs = {
    .b0 = 0.724123f,
    .b1 = -1.412351f,
    .b2 = 0.690218f,
    .a1 = -1.421543f,
    .a2 = 0.423533f
};

static IIR_State g_pixel_states[ROWS][COLS] = {0};

/**
 * @brief 对单个像素更新 IIR 逆滤波
 */
static inline float update_pixel_filter(float raw_adc, IIR_State *state, const IIR_Coeffs *coeffs) {
    // 采用 Direct Form II Transposed 结构
    float out = coeffs->b0 * raw_adc + state->d1;
    state->d1 = coeffs->b1 * raw_adc - coeffs->a1 * out + state->d2;
    state->d2 = coeffs->b2 * raw_adc - coeffs->a2 * out;
    return out;
}

/**
 * @brief 整个 10x10 阵列的一帧实时处理函数
 * 
 * @param raw_matrix   原始采集到的 ADC 读数矩阵 (10x10)
 * @param corr_matrix  抗蠕变校正后的等效 ADC 矩阵 (10x10)
 */
void Process_Tactile_Array_Frame(const float raw_matrix[ROWS][COLS], float corr_matrix[ROWS][COLS]) {
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            corr_matrix[r][c] = update_pixel_filter(
                raw_matrix[r][c], 
                &g_pixel_states[r][c], 
                &g_comp_coeffs
            );
        }
    }
}
```

---

## 7. 领域经典参考文献与研究脉络

在检索与深入研究时，建议参考以下方向的高水平文献：

1. **黏弹性传感器逆动态补偿理论（基础必读）：**
   * *Hu et al., "A Viscoelastic Compensator for Force Sensors With Soft Materials", IEEE Transactions on Instrumentation and Measurement, 2023.*
   * **核心贡献：** 详细论证了在软材料力传感器中，如何通过设计连续域逆补偿传递函数，同时满足静态力不发散衰减与动态响应不滞后的数学充分条件。
2. **柔性压阻动态蠕变反演与历史信号序列利用：**
   * *Tian et al., "Compensation strategy of dynamic creep drift for flexible piezoresistive sensors with historical signals", Measurement, 2026.*
   * **核心贡献：** 针对变载荷保压下的复杂蠕变，利用历史递归状态估计当前材料松弛程度，有效解决大变载荷下非线性蠕变反算问题。
3. **迟滞与蠕变混合模型解耦（Rate-Dependent PI Models）：**
   * *Al Janaideh et al., "A Rate-Dependent Prandtl-Ishlinskii Model for Characterizing and Compensating Hysteresis", IEEE Transactions on Control Systems Technology.*
   * **适用场景：** 若传感器不仅在恒力下漂移（蠕变），在加载与卸载曲线之间还存在显著的闭环胖环（迟滞），必须在上述 IIR 前级串联一个 Rate-Dependent 逆 PI 模型。
4. **分数阶微积分在聚合物蠕变建模中的应用：**
   * 高分子材料长达半小时以上的长程漂移常遵循幂律衰减（Nutting 定律），采用分数阶导数弹簧-阻尼模型（Scott-Blair 模型）仅需极少参数即可拟合超长时间域蠕变。