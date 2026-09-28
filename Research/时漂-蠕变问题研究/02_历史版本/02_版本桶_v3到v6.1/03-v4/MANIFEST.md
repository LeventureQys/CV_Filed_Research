# 03-v4 —— 快相免责期改造（本工作区的主线起点）

> 本文件由归档时生成，列出该版本桶的**全部文件**；版本之间的关系见 `../README.md`。

- 时间：2026-09-17 11:32 ~ 14:14
- 状态：**原型，未落地 C++**（其落地骨架 `04-C++实现骨架.md` 部分内容仍适用）
- 核心思想：加载瞬间的「快相」不要求算法处理，把补偿推后到快相结束，只对抗后续的慢相蠕变

本桶回答的问题是「**快相 0~4 s 的快速爬升能不能放它过去，只治之后的慢相**」。
先把两段结构量化（`快相与慢相分离分析.md`：归一化轮廓 σ≈0.016，高度可复现），
再把免责期从 3 s / 5 s / 8 s 扫一遍，最终定为 **3 s 与 5 s 两档**并完成 5 算法 × 11 组对比。

三份正式文档：
- `docs/02-v4算法说明.md` —— v4 的物理依据、算法定义、参数、代价与限制；
- `docs/03-验证与实测结果.md` —— 5 算法 × 11 组验证、图表索引、复现命令；
- `docs/04-C++实现骨架.md` —— v4 的 C++ 改动骨架（§3.0/3.1/3.2/3.4 要点仍适用，§3.3 关于 restep 起免责期的描述有误）。

本轮暴露的四个缺陷正是 v5 的修复对象；`五算法实测对比.md` 是这一轮的收官对比
（raw / v3 / v4-3s / v4-5s / dsp.md 逆滤波）。

**关键发现**：变载可辨识性 —— v4 免责期在 8/8 变载事件上欠报不大于 v3（均值 9.6% → 5.9%）；
「免责期只在首次 onset 生效」的变体（v7）在变化负载上与 v3 逐帧完全相同 ⇒ 无收益，见 `09-v7`。

## 文档（5）

| 文件 | 大小 |
|---|---|
| `docs/02-v4算法说明.md` | 12 KB |
| `docs/03-验证与实测结果.md` | 11 KB |
| `docs/04-C++实现骨架.md` | 10 KB |
| `docs/五算法实测对比.md` | 5 KB |
| `docs/快相与慢相分离分析.md` | 18 KB |

## 图件（15）

> 位置：`figures/`（桶根，文档按 `../figures/…` 引用）。

`F1_overview / F2_zoom / F3_metrics`（4 算法 × 11 组）、**`G1~G4`（5 算法 × 11 组最终对比，含变化负载长图）**、`f_fastphase_1`（快相归一化轮廓）、`f_varying_change`（中途变载逐帧）、`new_switch_load_*` / `rec_*`（实录验证）、`weight_scenario`（合成砝码 7 场景）。

- `figures/F1_overview.png`（375 KB）
- `figures/F2_zoom.png`（226 KB）
- `figures/F3_metrics.png`（119 KB）
- `figures/G1_overview.png`（252 KB）
- `figures/G2_varying.png`（311 KB）
- `figures/G3_zoom.png`（263 KB）
- `figures/G4_metrics.png`（123 KB）
- `figures/f_fastphase_1.png`（126 KB）
- `figures/f_varying_change.png`（108 KB）
- `figures/new_switch_load_metrics.png`（270 KB）
- `figures/new_switch_load_result.png`（557 KB）
- `figures/rec_13ffca_result.png`（367 KB）
- `figures/rec_1d9493_result.png`（282 KB）
- `figures/rec_two_metrics.png`（170 KB）
- `figures/weight_scenario.png`（333 KB）

## 脚本（50）

`ad_v4 / r_fastphase` 是 v4 算法本体（后者同时是原型入口）；`q_*` 快慢相量化；`s_* / t_* / u_* / v2_track / w2_repeat` 免责期专项；`y2/y3/z8/z9` 变载可辨识性；`aa/ab/ac_*` 三轮最终对比与出图；`ad_* / ae_* / af_*` 实录与砝码场景；`cpp_check/` 是「文档里的 C++ 骨架片段抽出来能否编译」的自检工程。

```text
_datab.py  _ema.py  _eq.py  _font.py  _gap.py  _png.py
_png2.py  _rob.py  _tmp_prof.py  _trace13.py  aa_final_compare.py  ab_final_figs.py
ac_final5.py  ad_check_v4_restep.py  ad_lib.py  ad_probe_new.py  ad_probe_profile.py  ad_single_fig.py
ad_single_run.py  ad_trace_step.py  ad_trace_tail.py  ad_trace_v3.py  ad_v4.py  ae_multi_step.py
ae_weight_fig.py  ae_weight_scenario.py  af_two_figs.py  af_two_recs.py  glm53_v3.py  glm53_v7.py
q2_slope_look.py  q_two_phase.py  r_fastphase.py  s_residual_nature.py  t_trace_early.py  u_varying_fast.py
v2_track.py  w2_repeat.py  x2_fastfig.py  y2_varying_steps.py  y3_varying_steps2.py  z2_v5.py
z8_gap.py  z9_varying_fig.py
（编译缓存：ad_lib.cpython-314.pyc, ad_v4.cpython-314.pyc, glm53_v3.cpython-314.pyc, glm53_v7.cpython-314.pyc, r_fastphase.cpython-314.pyc, z2_v5.cpython-314.pyc）
```

## 数据（27）

```text
_slope_look.txt  fastphase_metrics.csv  fastphase_repeatability.txt  fastphase_summary.csv
final5_drift_by_dataset.csv  final5_metrics.csv  final5_summary.csv  final_4algo_drift_by_dataset.csv
final_4algo_metrics.csv  final_4algo_summary.csv  new_switch_load.npz  new_switch_load_events.csv
new_switch_load_metrics.csv  rec_13ffca.npz  rec_13ffca_events.csv  rec_13ffca_metrics.csv
rec_1d9493.npz  rec_1d9493_events.csv  rec_1d9493_metrics.csv  rec_midload_events.csv
residual_nature.csv  track_accuracy.csv  two_phase_structure.csv  two_phase_structure.txt
varying_fastphase.csv  varying_identifiability.csv  varying_steps.csv
```

## `cpp_check/`（2）

```text
cpp_check/extract_check.py
cpp_check/v4_extracted.cpp
```

## 复现

```powershell
cd temp/v4.1flash/progress/03-v4
$env:PYTHONIOENCODING='utf-8'
python scripts/ac_final5.py       # 5 算法 × 11 组 + G1~G4（约 5 分钟）
python scripts/r_fastphase.py     # 免责 3/5/8 s 扫描（约 3 分钟）
python scripts/z8_gap.py          # 变载可辨识性（约 2 分钟）
python scripts/x2_fastfig.py      # 快相轮廓图（约 1 分钟）
```
