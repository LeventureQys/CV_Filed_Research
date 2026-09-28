# v4.1flash —— dsp.md 方案评审与实测

> **[归档注记]** 本文件是 `temp/v4.1flash/` **重组织前**的原文，按原样保留；其中的 `Document/`、`figures/`、`results/`、`scripts/`、`paper*/` 等路径与图路径都是**当时的历史路径**，重组织后不再指向实际位置 —— 现有结构与入口见 `../../README.md`。

> 任务：评审 `temp/GLM53/dsp.md`（柔性触觉压感阵列抗蠕变漂移与动态力保真 DSP 补偿方案指南），
> 按其中的思路实现并用 `temp/` 下的多组数据实测。
> 日期：2026-09-16 · 执行：DSH Agent (v4.1flash)

## 必读

- **`Document/`**：**算法正式文档**（v5 现役 / v6 规格 / **v6.1 定点修复** / 验证结果 / C++ 落地骨架），
  版本关系与目录见 `Document/README.md`
- **`Document/08-v6.1算法说明.md`**：最新一轮——「切换负载类数据阶跃仍带尖峰、单一负载看不到」的
  定位与修复（形状库按最快实测形状上包络重标，只对 onset 生效；图 `figures/L1_v61_spike_fix.png`、
  `L2_v61_metrics.png`）
- **`dsp方案评审报告.md`**：dsp.md 的解析复核与逐条评审结论（§1~§8）
- **`快相与慢相分离分析.md`**：加载后「快相 0~4s / 慢相 4s~」两段结构的量化，
  以及「快相免责期」改造的实测（用户提问的回应）
- **`五算法实测对比.md`**：5 个最优算法 × 11 组实测数据的最终对比（图 `figures/G1~G4`，
  其中 `G2_varying.png` 为变化负载单独长图）
- **`免责期1s-3s-5s对比.md`**：把「无责」窗压到 1s 的实测对比与两因子拆解
  （图 `figures/H1_exempt_1s_3s_5s.png`、`H2_exempt_1s_ablation.png`；结论：1s 不可取）

---

## 结论速览

**dsp.md 的理论方向正确，但工程细节失守，实测劣于现有实现。**

| 项目 | 判定 |
|---|---|
| §2.1~§3.1 连续域建模与逆系统推导 | 正确（§6.1 系数经频域逐点验证误差 < 1e-6，零极点精确对消，直流增益恰为 1） |
| §2.1 两套公式的自洽性 | **矛盾**（直流增益相差 N0 = 1+a1+a2 倍） |
| §6.2「可直接上机」硬编码系数 | **与 §6.1 公式不是同一个滤波器**（分母差 0.559，不可复现） |
| §3.3 工况 A/B 的行为与数字 | **错误**（过冲 N0 倍而非零；工况 B 数字与自身参数差 2.8 倍） |
| §3.1 第 3 条「高频增益限制」 | **未实现**（幅频直流 1 → 奈奎斯特 N0，全带噪声放大 N0 倍） |
| §5 离线参数辨识 | **在 9 组实测数据上标定退化**，拿不到可用参数 |
| §4 空间 PSF 反卷积 | **错误**（拉普拉斯核直流增益 0，实测把总力打成负数） |
| 用它替换现有 GLM53 v3 | **不建议**（时漂残余 1.55% → 32.3%，噪声 0.72 → 2.44 倍） |

完整论证见 **`dsp方案评审报告.md`**。

> **与并行会话的关系**：本次评审进行期间，仓库内另有一个会话对同一份 dsp.md 做了独立复核，
> 产物在 `temp/v4pro/`。两份结论在全部缺陷定位上一致（详见评审报告 §7 逐项对照表），
> 仅在「§6.1 的归一化该不该保留」这一处解读口径不同——那是 §2.1 模型定义不清造成的，
> 两种解读各自自洽。v4pro 额外提供了 C++ harness（逐帧 Tustin 变 dt + 22/22 组数值一致性），
> 本目录额外提供了 §4 空间反卷积、§5 标定退化、数值条件 κ 三项补充证据。

---

## 目录结构

```
temp/v4.1flash/
├── README.md                    本文件
├── Document/                    ★ v4 算法正式文档（README + 算法说明 + 验证结果 + C++ 骨架）
├── dsp方案评审报告.md           对 dsp.md 替代路线的评审
├── 快相与慢相分离分析.md        快相/慢相结构量化与免责期验证
├── 五算法实测对比.md            5 算法 × 11 组数据最终对比
├── scripts/                     全部脚本（可独立复算）
├── cpp_snippet_check/           C++ 骨架片段的抽取-编译-自检（验证文档代码可编译）
├── results/                     全部中间与最终数据
│   └── superseded/              中途被推翻的错误结论（保留留痕）
└── figures/                     图表
```

---

## 快速复现

```powershell
cd temp/v4.1flash
$env:PYTHONIOENCODING='utf-8'      # 控制台中文/特殊符号需要

python scripts/v_dsp_review_probe.py   # §1~§3 解析性复核（秒级）
python scripts/v_final_check.py        # IIR 正确性裁决（秒级）
python scripts/y_spatial.py            # §4 空间反卷积实测（秒级）
python scripts/inventory.py            # 11 组数据盘点（约 1 分钟）
python scripts/e_dsp_fail.py           # §5 标定失败定位（约 2 分钟）
python scripts/w2_dsp_vs_v3.py         # 9 组恒载总对比（约 4 分钟）
python scripts/f_review_figs.py        # 生成评审报告图（约 1 分钟）
```

依赖：Python 3.14 + numpy 2.4.6 / scipy 1.17.1 / pandas 3.0.3 / matplotlib 3.10.9（均为本机已有版本）。

---

## 数据来源

| 位置 | 组数 | 通道/行列 | 时长 | 原始时漂（主通道） |
|---|---|---|---|---|
| `temp/右拇指指尖/数据1~3` | 3 | 31ch / 9×7 | 163~234 s | 23~33% |
| `temp/左拇指指尖/数据1~3` | 3 | 31ch / 9×7（mask 与右拇指镜像） | 192~199 s | ~5% |
| `temp/四指指尖/数据1~3` | 3 | 21ch / 8×5 | 191~200 s | 10~15% |
| `temp/变化负载/…`（2 组） | 2 | 21ch / 8×5 | 71 / 64 s | 11.4% / 21.2% |

盘点结果（含时间戳质量核实）在 `results/dataset_inventory.csv`。
两组变化负载为 ADC 域（0~3470）且帧到达呈成批突发（median dt ≈ 7 µs），
本次主对比只用了 9 组指尖恒载数据；变化负载用于分段电平核对。

**时间戳质量核实（推翻此前「约 3/4 的帧时间戳重复」的说法）**：
- `timestamp`：指尖数据重复率 0.01~0.04%，变化负载 5.1%/6.5% —— **可靠，时间轴必须用它**；
- `elapsed`：指尖数据 38~39%，变化负载 71~72% —— 量化重复，**不可用**。

---

## 核心实验一览

| 实验 | 脚本 | 产物 | 结论 |
|---|---|---|---|
| dsp.md 解析复核 | `v_dsp_review_probe.py` | `results/review_probe.txt` | §6.1 数学正确；§6.2 魔数不自洽；§2.1 两式差 N0 倍 |
| IIR 正确性裁决 | `v_final_check.py` `v_verdict.py` | `results/_final_check.txt` `_verdict.txt` | 频响与解析精确逆 6 位小数一致；对物理信号输出正确 |
| 数值条件 | `v_iir_diag2.py` `v_iir_acc.py` | `results/_iir_diag2.txt` `_iir_acc.txt` | κ = ‖b‖₁/\|B(1)\| 达 2e6~4.5e8，定点/单精度不可用 |
| 瞬态代价 | `v_transient_cost.py` | `results/_transient.txt` | 过冲 = N0−1，1% 建立时间 100 s 量级 |
| §4 空间反卷积 | `y_spatial.py` | `results/spatial_check.txt` `spatial_deconv.csv` | 总量比 −0.60~−0.85，52~68% 通道出负值 |
| §5 标定失败定位 | `e_dsp_fail.py` | `results/dsp_failure.txt` `model_shape_check.csv` | 9 组全部单调上升，无 dsp.md 预言的回落；标定退化 |
| 9 组恒载总对比 | `w2_dsp_vs_v3.py` | `results/dsp_vs_v3_summary.csv` `dsp_fit_params.csv` | v3 1.55% vs dsp.md 32.3% |
| 数据盘点 | `inventory.py` | `results/dataset_inventory.csv` | 时间戳质量核实、分段结构 |
| **快慢相结构** | `q_two_phase.py` `q2_slope_look.py` `w2_repeat.py` | `results/two_phase_structure.*` `fastphase_repeatability.txt` | 快相 0~1s 机械段 + 1~4s 尾巴；归一化形状 σ≈0.016 高度可复现 |
| **快相免责期改造** | `r_fastphase.py` `s_residual_nature.py` `t_trace_early.py` `u_varying_fast.py` `v2_track.py` | `results/fastphase_*.*` `residual_nature.csv` `varying_fastphase.csv` `track_accuracy.csv` | 慢相段时漂 1.50%→0.97%；A 参考窗必须落在快相之后 |
| **变载可辨识性** | `y3_varying_steps2.py` `z8_gap.py` `z4_v5_trace.py` `glm53_v7.py` `z7_v7_test.py` `z9_varying_fig.py` | `results/varying_identifiability.csv` `varying_steps*.csv` | v4 免责期在 8/8 变载事件上欠报不大于 v3（均值 9.6%→5.9%）；「只在首次 onset」无收益 |

---

## 图表

| 文件 | 内容 |
|---|---|
| `figures/f_review_1.png` | §6.1 阶跃响应（N0 倍过冲慢衰减）+ 幅频（直流 1 → 奈奎斯特 N0） |
| `figures/f_review_2.png` | 9 组实测形状检验（全部单调上升，无 dsp.md 预言的回落） |
| `figures/f_review_3.png` | §4 空间反卷积失败（总力变负；正确形式也让 RMSE 变大） |
| `figures/f_review_4.png` | 实测主通道时序：原始 / GLM53 v3 / dsp.md 三方对比 |
| `figures/f_fastphase_1.png` | 快相归一化轮廓（9 组重叠）+ 快慢相占比 + 免责期时序 |
| `figures/f_varying_change.png` | 变化负载中途变载处 v3 vs v4 逐帧对比 + 最坏欠报柱状 |
| **`figures/F1_overview.png`** | 4 算法 × 11 组时序总览（上一版，保留） |
| **`figures/G1_overview.png`** | **恒载 9 组总览（5 算法，最新）** |
| **`figures/G2_varying.png`** | **变化负载单独长图**：数据A/B 全长 + 数据B 中途变载 16~34s 局部拉长放大 |
| **`figures/G3_zoom.png`** | 负载段内放大（3 位置）+ 加载沿 0~9s（橙=免责3s、红=免责5s） |
| **`figures/G4_metrics.png`** | 恒载 9 组指标四联（5 算法） |
| **`figures/H1_exempt_1s_3s_5s.png`** | **免责期 1s/3s/5s 同图对比（3×3：时序 / 加载沿放大+首扣 / 首扣时延 / 恒载时漂 / 变载跟踪 / 扣除量 / A 窗 / 逐事件 / 全程偏差）** |
| **`figures/H2_exempt_1s_ablation.png`** | **1s 档两因子拆解（免责期长度 × 武装延时；含 pending/hold 误触发条带）** |
| `figures/f1~f3_*.png` | 9 组网格时序、负载段放大、阶跃沿放大（早期版本，w2 产出） |

---

## 复核留痕（重要）

本次复核过程中我**误判过两次**，结论已推翻，过程保留在 `results/superseded/`：

1. **直流增益不能用 `sum(b)/sum(a)` 算**。这类逆滤波器极点贴近 z=1、分子系数符号交替，
   `np.sum(b)` 会灾难性抵消（示例参数下 ≈ 1e-5），使 `sum(b)/sum(a)` 输出假的 1.000000。
   我据此一度判定「§6.1 的 gain_corr 破坏零极点对消」—— **错，已撤回**。
   正确写法是 $B(1)/A(1)$ 的多项式求值。
2. **自写的「并行等价实现」早期版本漏了归一化系数**，导致误判 IIR 有 16~44% 误差 ——
   频域逐点比对后确认 **IIR 是对的**（`results/_verdict.txt`）。

因此 **§6.1 的系数公式与实现本身没有问题**，本报告对 dsp.md 的批评集中在
§2.1 的自洽性、§6.2 的魔数、§3.3 的描述、§3.1 的高频承诺、§4 的空间反卷积、§5 的标定流程这六处。

---

## 与现有实现的关系

本次**未修改任何 `src/` 下的代码**。现有 `src/domain/drift/drift_compensator.{h,cpp}`
（GLM53 v3：卸载门控自动归零 + 混合蠕变补偿）在 9 组实测上时漂残余 1.55%、噪声比 0.72、
阶跃保真 1.00、事件跳变超额 0.03 ADC，全面优于 dsp.md 方案。

建议（详见评审报告 §5）：
1. **不要**引入 §6.1 的逆滤波器替换现有方案；
2. **可以吸收** §2.1 的显式双时间常数模型，作为「快速加载/短保压」工况的冷启动先验；
3. **可以吸收** §5 的离线标定思路，但必须补机械建立段（加载沿后 0.1~0.3 s）建模，
   并把标定窗移到机械建立段之后。
