# 02-dsp-route —— dsp.md 逆滤波路线 —— 评审与实测否决

> 本文件由归档时生成，列出该版本桶的**全部文件**；版本之间的关系见 `../README.md`。

- 时间：2026-09-17 10:01 ~ 10:31（本工作区最早的一轮）
- 来源：评审 `temp/GLM53/dsp.md`《柔性触觉压感阵列抗蠕变漂移与动态力保真 DSP 补偿方案指南》
- 结论：**理论方向正确、工程细节失守，不建议采用**（时漂残余 1.55% → 15.2~32.3%）

这条路线与 v4/v5 是**并行技术路线**，不是版本演进的一环：它主张用二阶逆 IIR（§6.1）直接反演蠕变模型，
再用拉普拉斯高通核做空间反卷积（§4）。本桶是把它**照着实现并放到 9 组实测上跑**的全部证据链。

评审结论分六处（详见 `docs/dsp方案评审报告.md`）：
§2.1 两套公式不自洽（直流增益差 N₀ = 1+a₁+a₂ 倍）；§6.2 硬编码系数与 §6.1 公式不是同一个滤波器；
§3.3 工况 A/B 的行为描述错误；§3.1「高频增益限制」未实现（全带噪声放大 N₀ 倍）；
§5 离线标定在 9 组实测上退化；§4 空间反卷积把总力打成负数（52~68% 通道出负值）。
§6.1 的数学推导本身经频域逐点验证是正确的（误差 < 1e-6，零极点精确对消）。

**留痕**：`results/superseded/` 保留了本轮两次被推翻的误判（用 `sum(b)/sum(a)` 判直流增益、
早期自写并行实现漏归一化系数），README §复核留痕 有说明。

## 文档（1）

| 文件 | 大小 |
|---|---|
| `docs/dsp方案评审报告.md` | 23 KB |

## 图件（7）

> 位置：`docs/figures/`（与文档同目录，正文按 `figures/…` 引用）。

`f_review_1~4.png` 为评审报告配图（§6.1 阶跃响应/幅频、9 组形状检验、空间反卷积失败、主通道三方对比）；`f1_timeseries_grid / f2_load_zoom / f3_step_edge` 为 9 组恒载上 dsp vs v3 的对比。

- `docs/figures/f1_timeseries_grid.png`（262 KB）
- `docs/figures/f2_load_zoom.png`（208 KB）
- `docs/figures/f3_step_edge.png`（115 KB）
- `docs/figures/f_review_1.png`（89 KB）
- `docs/figures/f_review_2.png`（56 KB）
- `docs/figures/f_review_3.png`（88 KB）
- `docs/figures/f_review_4.png`（95 KB）

## 脚本（22）

`v_*` 为解析性探针与 IIR 正确性裁决；`x_model_id / y_spatial / z_shape_look` 为模型辨识、空间反卷积与形状诊断；`w_dsp_vs_v3 / w2_dsp_vs_v3 / e_dsp_fail` 为真实数据上的对比与失败点定位；`inventory.py` 产出 11 组数据盘点；`f_review_figs.py` 出图。

```text
e_dsp_fail.py  f_review_figs.py  glm53_v3.py  inventory.py  v_dsp_review_probe.py  v_final_check.py
v_iir_accuracy.py  v_iir_diag2.py  v_iir_equiv.py  v_iir_vs_parallel.py  v_probe_decisive.py  v_probe_p4_diag.py
v_probe_scale_diag.py  v_shape_diag.py  v_transient_cost.py  v_verdict.py  w2_dsp_vs_v3.py  w_dsp_vs_v3.py
x_model_id.py  y_spatial.py  z_shape_look.py
（编译缓存：glm53_v3.cpython-314.pyc）
```

## 数据（25）

```text
_final_check.txt  _iir_acc.txt  _iir_diag2.txt  _iir_equiv.txt
_transient.txt  _verdict.txt  dataset_inventory.csv  dsp_fail_metrics.csv
dsp_fail_summary.csv  dsp_failure.txt  dsp_fit_params.csv  dsp_vs_v3_metrics.csv
dsp_vs_v3_summary.csv  model_identification.csv  model_shape_check.csv  review_probe.txt
spatial_check.txt  spatial_deconv.csv  superseded/_decisive.txt  superseded/_iir_chk.txt
superseded/_model_id.txt  superseded/_p4_diag.txt  superseded/_probe1.txt  superseded/_scale_diag.txt
superseded/_shape_diag.txt
```

## 注意

`results/superseded/` 是被推翻结论的留痕，不要当作结论引用。

## 复现

```powershell
cd temp/v4.1flash/progress/02-dsp-route
$env:PYTHONIOENCODING='utf-8'
python scripts/v_dsp_review_probe.py   # §1~§3 解析性复核（秒级）
python scripts/v_final_check.py        # IIR 正确性裁决（秒级）
python scripts/y_spatial.py            # §4 空间反卷积实测（秒级）
python scripts/inventory.py            # 11 组数据盘点（约 1 分钟）
python scripts/e_dsp_fail.py           # §5 标定失败定位（约 2 分钟）
python scripts/w2_dsp_vs_v3.py         # 9 组恒载总对比（约 4 分钟）
python scripts/f_review_figs.py        # 生成评审报告图（约 1 分钟）
```
