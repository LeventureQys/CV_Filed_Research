# 03_复算与验证 —— K9 的离线复算资产

> 这里的东西**可以原地跑**：Python 原型、C++↔Python 对拍、C++ 离线助手、出图脚本、K7/K9 的会话评估器。
> 图件在 `figure/`，数值结果在 `results/`。**改了 C++ 之后必须重跑这里并核对**，否则所有结论作废。

---

## 1. 目录

| 目录 | 内容 |
|---|---|
| `scripts/` | 该代的脚本（19 个源文件 + `build/`）：Python 原型核心、对拍、规律辨识、出图、C++ 离线助手 |
| `results/` | 该代数值结果（v3 评估、v2 评估、约定对比、规律辨识、`T_stable` 探针、全量对照） |
| `figure/` | 该代图件 39 张：`concept/` 12（机制示意，图 1–5、8–14）+ `derive/` 2（式形推导，图 6–7）+ `observer/` 23（20 个数据集各一张 + 总图）+ 桶根 2 张（约定对比、规律辨识） |
| `k7_scripts/` `k7_figures/` | **K7 的会话评估器**与 10 会话的指标/图（`k7_eval.py <session_dir> <out_dir>`） |
| `k9_scripts/` `k9_figures/` | **K9 的会话评估器**与图（`k9_eval.py <session_dir> <out_dir>`），逐行转写 C++ `Process()` |

---

## 2. 关键脚本

| 脚本 | 作用 |
|---|---|
| `scripts/v34_observer_core{,2,3}.py` | 观测器原型三代核心；**`core3` = 与当前 C++ 对拍的那一版** |
| `scripts/v34_parity_check.py` | **C++ ↔ Python 逐帧对拍**（调 `build/obs_runner.exe`） |
| `scripts/obs_runner.cpp` + `build_obs_runner.bat` | **链接产品源码的离线复算器**（stdin 喂帧，stdout 出显示值） |
| `scripts/v30_runner.cpp` + `build_v34_runner.bat` | 上一代（v3 口径）的离线助手 |
| `scripts/v30_lib.py` | 公共层：数据装载 / 事件检测 / 指标 |
| `scripts/v34_law_identify.py` | 粘弹性规律辨识（留一法）→ 精度上限 |
| `scripts/v34_observer_{eval,v2_eval,v3_eval}.py` | 三代全量评估 |
| `scripts/v34_probe_tstable.py` | `T_stable`（稳定时间）探针 |
| `scripts/v34_plot_conventions.py`、`v34_make_{concept,derive}_figs.py`、`v34_observer_plot_all.py` | 出图 |
| `scripts/v34_check_concept_fig_layout.py` | 图版式自检 |
| `scripts/x1_explore.py`、`x1_lab_lib.py` | x1 形状研究的共享层（在研课题 `x1形状与回落/` 会用到） |
| `k7_scripts/k7_eval.py` | K7 评估：`cycle_bias / growth（棘轮）/ idle_bias` + 平台偏差 |
| `k9_scripts/k9_eval.py` | K9 评估：在 K7 基础上加 `ramp_dwell` 与 `tc1` 混合 |

---

## 3. 怎么跑

```powershell
cd "01_当前工作_K9\03_复算与验证"

# ① Python 原型单跑
python scripts\v34_observer_core3.py

# ② C++ ↔ Python 对拍（先构建离线助手）
cmd /c scripts\build_obs_runner.bat
python scripts\v34_parity_check.py

# ③ K7 / K9 会话评估（任意新会话）
python k7_scripts\k7_eval.py "<会话目录>" "<输出目录>"
python k9_scripts\k9_eval.py "<会话目录>" "<输出目录>"

# ④ 规律辨识 / 稳定时间探针
python scripts\v34_law_identify.py
python scripts\v34_probe_tstable.py
```

依赖：Python 3 + numpy / scipy / pandas / matplotlib；C++ 部分需 MSVC 2022 + 仓库 `src` 与 `third_party/eigen-5.0.0`。

---

## 4. 可信度与已知限制

- **对拍结论**：`算法说明_v3.4观测器.md` 记录与 Python 原型 **PARITY OK，max|Δ| = 5e-5 ADC**。
- **K7 评估器的验证**：`k7_eval.py` 用 `86eca9` 双流验证 **max err = 1.8 ADC**。
  ⚠️ **C++ 改动后必须回改 `k7_eval.py` / `k9_eval.py` 并重跑全部会话**——它们是 C++ 的逐行转写，不同步就失真。
- **旧锚点注意**：`scripts/v30_lib.py` 的 `ROOT` 用 `dirname(__file__) + 5×".."`，
  再拼 `temp/算法数据&原始数据`——这是**归档前的仓库布局**，**在本次重构前就已经失效**（不是重构造成的）。
  要读数据请直接用仓库当前路径 `..\..\..\data\`。其余脚本用 `__file__` 相对锚点，随目录整块移动**仍然有效**。
- **未做真机 A/B 与界面手测**：这里的全部结论都来自离线回放与逐帧对拍。
- `scripts/__pycache__/` 与 `scripts/build/*.obj|*.exe` 是惰性产物，可删可重建。

---

## 5. K7 / K9 评估器的指标口径

| 指标 | 定义 |
|---|---|
| `cycle_bias` | 各加载段均值（pre − 输出）——补偿量是否对得上 |
| `growth` | \|末段\| − \|首段\|——**棘轮**（反复加卸载时补偿越扣越多） |
| `idle_bias` | 空载段均值差——空载是否塌陷 |
| 平台偏差 | 平台期 pre 与 out 的差（越接近真实蠕变越好） |
| 残余漂移率 | **同窗**拟合 `out − pre` 斜率（pre 定窗）——注意早期版本用了两个不同窗，口径有偏 |
