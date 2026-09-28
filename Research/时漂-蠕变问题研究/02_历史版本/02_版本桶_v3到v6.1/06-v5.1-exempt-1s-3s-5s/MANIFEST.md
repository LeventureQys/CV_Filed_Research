# 06-v5.1-exempt-1s-3s-5s —— 免责期 1 s / 3 s / 5 s 三档专项

> 本文件由归档时生成，列出该版本桶的**全部文件**；版本之间的关系见 `../README.md`。

- 时间：2026-09-18 00:13 ~ 01:18
- 提问：「把无责窗从 3 s 压到 **1 s** 会怎样？请与 3 s / 5 s 同图对比」
- 结论：**1 s 不可取**；3 s 仍是长保压默认折中，5 s 在变载跟踪上全面最好

这是对现役 v5.1 **运行期可切参数**（菜单三档）的一次专项扫描：同一算法本体
（`glm53_v51.py`，与 C++ 逐帧一致）只改免责期长度与随之派生的 A 窗、武装延时（三处联动）。

| 口径 | 1 s | 3 s | 5 s |
|---|---|---|---|
| 恒载 9 组 全段时漂 | 1.79% | 1.59% | 1.53% |
| 变载窗最大偏差（中位） | 3258 ADC | 1889 ADC | **512 ADC** |
| 台阶捕获比（中位） | 0.76~0.90 | 0.96 | **0.99** |
| pending 占比 / epoch 数 | 21.2% / 31 | 16.3% / 29 | 16.3% / — |

1 s 的问题在于**把真实加载当成蠕变扣掉**（台阶捕获比最小 0.76，全程最大偏差中位是 5 s 档的 3.7 倍），
并引入额外的假变载重锚。`H2_exempt_1s_ablation.png` 把 1 s 档拆成「免责期长度 × 武装延时」两因子，
证明欠报来自它们共同作用，而不是单一参数。

> **后续**：菜单最终按用户明确要求加入了「无责 1s」档，并在 tooltip 中标注欠报代价
> （见项目 `src/ui/menubar.{h,cpp}`）。

## 文档（1）

| 文件 | 大小 |
|---|---|
| `docs/免责期1s-3s-5s对比.md` | 19 KB |

## 图件（7）

> 位置：`figures/`（桶根，文档按 `../figures/…` 引用）。

`H1`（3×3 主对比：时序/加载沿放大+首扣/首扣时延/恒载时漂/变载跟踪/扣除量/A 窗/逐事件/全程偏差）、`H2`（1 s 两因子拆解）、`H3`（13ffca 加载沿）、`H4`（13ffca 时间分段）、`H5`（1868 s pending 冻结放大）、`H6/H7`（13 份录制 1 s 档总览与指标）。

- `figures/H1_exempt_1s_3s_5s.png`（381 KB）
- `figures/H2_exempt_1s_ablation.png`（112 KB）
- `figures/H3_13ffca_loadedge.png`（373 KB）
- `figures/H4_13ffca_time_segments.png`（343 KB）
- `figures/H5_zoom_1868_pending_freeze.png`（228 KB）
- `figures/H6_1s_overview_all.png`（243 KB）
- `figures/H7_1s_metrics_all.png`（112 KB）

## 脚本（20）

`bp_exempt_sweep.py` 是主扫描（37 KB，产出 H1/H2 与 `exempt_sweep_*` 全套）；`bq_probe_1s.py` 机制探针；`br/bz/by` 为 13ffca 与 1868 s 的定点放大；`bs/bt/bu/bx` 为加载沿时延、恒载时间线、pending vs 免责、无冻结 A/B；`ca_all_datasets_1s.py` 产出 13 份录制的 1 s 档总览。

```text
ad_lib.py  ad_v4.py  bp_exempt_sweep.py  bq_probe_1s.py  br_target_13ffca.py  bs_loadedge_latency.py
bt_constant_load_timeline.py  bu_pending_vs_exempt.py  bx_nofreeze_ab.py  by_timeline_segments_13ffca.py  bz_zoom_1868.py  ca_all_datasets_1s.py
glm53_v3.py  glm53_v5.py  glm53_v51.py
（编译缓存：ad_lib.cpython-314.pyc, ad_v4.cpython-314.pyc, glm53_v3.cpython-314.pyc, glm53_v5.cpython-314.pyc, glm53_v51.cpython-314.pyc）
```

## 数据（31）

```text
_all_datasets_1s.log  _constant_load_timeline.log  _exempt_1s_probe.log  _exempt_sweep.log
_loadedge_latency.log  _nofreeze_ab.log  _pending_vs_exempt.log  _target_13ffca.log
_timeline_segments.log  _zoom_1868.log  all_datasets_1s.csv  constant_load_timeline.csv
exempt_sweep.npz  exempt_sweep_ablation.csv  exempt_sweep_awin.csv  exempt_sweep_deddelay.csv
exempt_sweep_events.csv  exempt_sweep_mech.csv  exempt_sweep_recs.csv  exempt_sweep_static9.csv
exempt_sweep_static9_summary.csv  loadedge_latency.csv  nofreeze_ab.csv  pending_vs_exempt.csv
target_13ffca.npz  target_13ffca_deddelay.csv  target_13ffca_edges.csv  target_13ffca_events.csv
target_13ffca_metrics.csv  target_13ffca_phases.csv  timeline_segments_13ffca.csv
```

## 复现

```powershell
cd temp/v4.1flash/progress/06-v5.1-exempt-1s-3s-5s
$env:PYTHONIOENCODING='utf-8'
python scripts/bp_exempt_sweep.py   # 主扫描（约 10 分钟，产出 H1/H2 + exempt_sweep_*）
python scripts/ca_all_datasets_1s.py # 13 份录制 1 s 档总览（H6/H7）
python scripts/br_target_13ffca.py  # 13ffca 加载沿（H3）
```
