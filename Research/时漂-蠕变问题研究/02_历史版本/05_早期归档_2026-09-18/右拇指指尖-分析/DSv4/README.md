# DSv4 输出目录说明

本目录保存“右拇指指尖 空载-恒定负载-空载”数据的时漂/零漂分析与因果算法对比。

## 必读

- `触觉阵列时漂与零漂分析报告_因果在线版.md`：完整分析报告
- `触觉阵列时漂与零漂分析报告_因果在线版.docx`：同内容 Word 版（已嵌入图件）

## 目录

| 路径 | 内容 |
|---|---|
| `figures/fig01~fig05` | 数据特征诊断图：总览、时漂形态、零漂、空间分布、共模性 |
| `figures/fig06~fig12` | 因果算法横向对比图：主通道时序、指标、热图、阵列残差、在线权衡、零漂、留一标定 |
| `scripts/01_characterize.py` | 数据特征分析 |
| `scripts/algorithms_causal.py` | 全部因果在线算法实现 |
| `scripts/04_causal_calibration.py` | 留一法超参数标定 |
| `scripts/02_causal_evaluate.py` | 因果算法评测与指标输出 |
| `scripts/03_causal_compare_plots.py` | 因果算法对比图件 |
| `scripts/05_causal_check.py` | 截断输入因果性数值校验 |
| `results/*.json/csv` | 指标、标定参数、因果性校验结果 |
| `legacy_non_causal/` | 早期非因果算法版本，仅归档对照 |

## 复现

```powershell
cd D:\workshop\Processing\multi-device-cascade-host-cpp\temp\右拇指指尖\DSv4
python scripts\01_characterize.py
python scripts\04_causal_calibration.py
python scripts\02_causal_evaluate.py
python scripts\03_causal_compare_plots.py
python scripts\05_causal_check.py
```

核心结论：门控基线 + 参数化对数漂移 Kalman + 3 点因果中值（留一标定 τ）在严格因果条件下，
把 ch17 负载段漂移从 51~65% 阶跃降至 1.0~1.5%，负载 RMSE 从 42~46% 降至 3.5~3.9%。
