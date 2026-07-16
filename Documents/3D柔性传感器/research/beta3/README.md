# v1.0.10 beta3 研究总览

beta3 在“膜片零压力为 0、外围微弱值属于串扰候选、织物类可能存在采集偏置”的新认知下，重新评估空间串扰补偿和长期稳定力值估计。当前内容为研究与仿真，不修改主项目代码，也不构成最终算法验收。

## 文档入口

| 文件 | 内容 |
|---|---|
| `研究约束.md` | 物理认知、产品场景、算法、数据和指标边界 |
| `工作计划与初步调研.md` | 三个工作包、算法方向和后续阶段计划 |
| `仿真研究报告.md` | 仿真模型、横向结果、结论和实物采集要求 |

## 目录

| 目录 | 内容 |
|---|---|
| `scripts/` | 仿真模型、候选滤波器和统一运行入口 |
| `tests/` | 模型与滤波器的轻量回归测试 |
| `data/` | 机器可读的膜片串扰和总力稳定指标 |
| `output/` | 膜片对比图、全部总力算法效果图和精度—稳定时间图 |

## 运行

依赖 Python、NumPy、SciPy 和 Matplotlib。在仓库根目录执行：

```powershell
python "Document\Update\v1.0.10 - research\beta3\tests\test_simulation_models.py"
python "Document\Update\v1.0.10 - research\beta3\tests\test_filters.py"
$env:MPLBACKEND='Agg'
python "Document\Update\v1.0.10 - research\beta3\scripts\run_simulation.py"
python "Document\Update\v1.0.10 - research\beta3\scripts\run_real_data.py"
```

仿真使用固定随机种子 `20260713`。运行脚本会覆盖 `data/` 和 `output/` 中对应结果。

`run_real_data.py` 对织物垫 500 g、手套单手指 1 kg、64×64 膜片 1 kg 的代表实测录制逐 Cell 扣除独立背景中位数，再把 beta3 因果时间算法放在同一张三行图中。它会生成：

- `output/real_data_all_algorithms_comparison.png`
- `data/real_data_filter_metrics.csv`

实测文件只有 ADC 和砝码标签，没有同步测力计真值及精确加载事件，因此该图只比较曲线平滑、滞后和趋势保持，不把滤波输出称为真实力，也不使用 RMSE 或稳定时间评价准确性。

## 当前结论

1. 膜片串扰在已知线性核的合成模型中可补偿，局部校正和 Richardson–Lucy 显著优于原始数据；模型失配会明显降低效果，因此必须先采集真实点载荷扫描。
2. 硬门控适合热图显示和接触掩膜，不适合宣称恢复真实力；3×3 空间中值在本轮模型中比原始数据更差。
3. EMA、IIR、Median-3 + EMA 和卡尔曼均可降低长期稳态跳动，但当前卡尔曼没有优于简单 EMA/IIR。
4. 推荐先以 EMA 0.8 s 或二阶 0.5 Hz IIR 做实物基线；存在尖峰时增加 Median-3 + EMA 对照，并重点检查弱 Cell 和总力 bias。
5. 平滑不等于校准。固定偏置、蠕变、迟滞和慢漂移必须独立评价。

下一阶段的具体实验设置、字段和最低重复次数见 `仿真研究报告.md` 的“实物数据的最低要求”。
