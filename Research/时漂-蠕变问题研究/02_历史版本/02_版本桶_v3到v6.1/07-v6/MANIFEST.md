# 07-v6 —— 形状约束反演 + 滑行器（已落地 C++）

> 本文件由归档时生成，列出该版本桶的**全部文件**；版本之间的关系见 `../README.md`。

- 时间：2026-09-18 14:19 ~ 16:38（`glm53_v6.py` 16:33）
- 状态：**已落地 `src/domain/drift_v6/drift_v6_compensator.{h,cpp}`**（菜单「时漂/零漂补偿（v6 快速稳定）」），未做真机验证
- 目标：把「加载沿 → 显示稳定」从 3.82 s 压到 ≤ 1.0 s

v6 **只换掉「快相处理」这一段**：把 v5 的「等快相走完（免责期）」换成「用标定形状把快相算掉」；
慢相模块（median 共识 `g` + 逐通道 `γ` + 限幅 + 输出封顶）**逐行沿用 v5**，保证可比。
前端为：短滞后电平差 + 在线 σ 门限检测（回溯真沿）→ 四类工况分类 → 形状约束反演 `Â = Σy·g/Σg²`
（14 点形状 ROM）→ 速率受限 smoothstep 滑行器 → τ_ho=5 s 交接给慢相模块。

**收益**：T_stable（显示首次停下、此后 30 s 自身漂移 ≤5%×阶跃）恒载 9 组中位 **3.82 s → 0.55 s**；
恒载多数稳态指标反超 v5（全段时漂 1.46% vs 1.50/1.85）。
**代价**：阶跃保真 1.00 → 1.08、epoch 数 1.6~1.7 倍、**实录类全程最大偏差 1889 → 4571~4583 ADC**。
⇒ 单次加载/长保压类工况可替代现役实现，**多次变载的实录类工况还不能**。

本桶另含一条对旧文档的实证修正（`07-v6算法说明.md` §内）：既有「快相 0~4 s 爬升」轮廓是分析脚本
τ=2 s 平滑造成的伪影。`v6与免责1s3s对比.md` §9 是用户报告两个具体现象后的第二轮修复与逐帧根因。

## 文档（2）

| 文件 | 大小 |
|---|---|
| `docs/07-v6算法说明.md` | 64 KB |
| `docs/v6与免责1s3s对比.md` | 20 KB |

## 图件（5）

> 位置：`figures/`（桶根，文档按 `../figures/…` 引用）。

`I1`（13 份录制小倍数总览）、`I2`（4 面板指标）、`J1`（检测窗扫描）、`K1`（第二轮修复验证）。

- `figures/H6_1s_overview_all.png`（243 KB）
- `figures/I1_v6_overview_all.png`（280 KB）
- `figures/I2_v6_metrics_all.png`（173 KB）
- `figures/J1_v6_detwin_sweep.png`（127 KB）
- `figures/K1_v6_fix_verify.png`（217 KB）

## 脚本（30）

`cb~ch_*` 为形状/原始上升/尾部/估计器/泛化/自形状/单点诊断（v6 规格的实测依据）；`ci_v6_vs_1s_3s.py` 为 13×3 路对比（I1/I2）；`cj~cs_*` 为冒烟/追踪/保持/离群/用户报告复现/卸载/状态演化；`cp_v6_detwin_sweep.py` 检测窗扫描（J1）；`ct_v6_verify_figs.py` 修复验证（K1）。

```text
ad_lib.py  ad_v4.py  cb_v6_stepshape.py  cc_v6_rawrise.py  cd_v6_tail.py  ce_v6_estimator.py
cf_v6_generalize.py  cg_v6_selfshape.py  ch_v6_singlepoint.py  ci_v6_vs_1s_3s.py  cj_v6_smoke.py  ck_v6_trace.py
cl_v6_holddiag.py  cm_v6_outlier.py  cn_v6_userreport.py  cp_v6_detwin_sweep.py  cq_v6_userreport2.py  cr_v6_unload.py
cs_v6_stateevo.py  ct_v6_verify_figs.py  glm53_v3.py  glm53_v5.py  glm53_v51.py  glm53_v6.py
（编译缓存：ad_lib.cpython-314.pyc, ad_v4.cpython-314.pyc, glm53_v3.cpython-314.pyc, glm53_v5.cpython-314.pyc, glm53_v51.cpython-314.pyc, glm53_v6.cpython-314.pyc）
```

## 数据（34）

```text
_v6_detwin_sweep.log  _v6_estimator.log  _v6_generalize.log  _v6_holddiag.log
_v6_outlier.log  _v6_rawrise.log  _v6_selfshape.log  _v6_singlepoint.log
_v6_smoke.log  _v6_stateevo.log  _v6_stepshape.log  _v6_tail.log
_v6_trace.log  _v6_unload.log  _v6_userreport.log  _v6_userreport2.log
_v6_verifyfigs.log  _v6_vs_v5.log  v6_detect_latency.csv  v6_detwin_sweep.csv
v6_detwin_sweep_summary.csv  v6_onset_profile9.csv  v6_raw_vs_smoothed_profile.csv  v6_restep_delta.csv
v6_rise_times.csv  v6_selfshape.csv  v6_stepshape_curves.csv  v6_stepshape_events.csv
v6_tail_events.csv  v6_vs_v5.csv  v6_vs_v5_settle.csv  v6_vs_v5_v6measured.csv
v6_vs_v5_v6trim.csv  varying_steps_v6.csv
```

## 复现

```powershell
cd temp/v4.1flash/progress/07-v6
$env:PYTHONIOENCODING='utf-8'
python scripts/cb_v6_stepshape.py     # 加载形状实测
python scripts/ci_v6_vs_1s_3s.py      # v6 vs 免责 1s/3s × 13 份（约 5 分钟，I1/I2）
python scripts/cp_v6_detwin_sweep.py  # 检测窗扫描（J1）
python scripts/ct_v6_verify_figs.py   # 修复验证（K1）
```
