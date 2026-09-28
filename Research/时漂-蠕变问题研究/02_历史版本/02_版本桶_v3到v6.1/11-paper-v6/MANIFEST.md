# 11-paper-v6 —— 论文（二）：v6 抗蠕变补偿算法

> 本文件由归档时生成，列出该版本桶的**全部文件**；版本之间的关系见 `../README.md`。

- 时间：2026-09-18 16:57 ~ 18:04
- 交付：`docs/抗蠕变补偿算法_v6.md`（约 4.5 万字，8 章 + 3 附录 + 7 图）+ `docs/README.md`（工作区说明）
- 特点：把**为拟合实机数据所做的全部 trade-off** 完整写成台账

论文只节选 v6 **纯抗蠕变算法**部分（即「最终画图/最终交付」的那一版），
并写出为拟合实机数据做的全部取舍。

**一句话结论**：v6 只换掉「快相处理」这一段（把「等快相走完」换成「用标定形状把快相算掉」），
慢相蠕变模块逐行沿用 v5；收益是首次加载平稳时刻 **3.82 s → 0.55 s**，
代价是阶跃保真 1.00 → 1.08、epoch 数 1.6~1.7 倍、实录全程最大偏差 1889 → 4583 ADC。
**单次加载/长保压类工况可替代现役实现，多次变载的实录类工况还不能。**

**图件验收是两层口径**：出图脚本内置 `figcheck`（面板数、空白面板、文字越界与重叠）；
另有 `pv_vision_check.py` 把图与「期望描述」一起送 vision 模型逐项回答
（面板数/重叠/截断/空白/缺曲线/图例遮挡/乱码），最新一次 **7 张全部 PASS**，记录在 `results/vision_check.md`。
`pv_audit.py` 对论文里引用的数字做审计（期望 OK 74 / BAD 0）。

**未做/未验证**（论文 §末已声明）：未改 `src/`；未做真机与界面手测；新参数只有检测窗做过扫描；
形状库用测试集自身标定（**乐观上界**）；时间轴对 τ<0.2 s 有畸变，v6 数字可能低估其真实能力。

## 文档（2）

| 文件 | 大小 |
|---|---|
| `docs/README.md` | 4 KB |
| `docs/抗蠕变补偿算法_v6.md` | 57 KB |

## 图件（7）

> 位置：`docs/figures/`（与文档同目录，正文按 `figures/…` 引用）。

按**正文出现顺序**编号，脚本 `pfN_*.py` 与图号一一对应：`F1_two_phase`（§1 加载形状实测）、`F2_overview`（§3 算法总览）、`F3_invert_glide`（§5 形状反演与滑行）、`F4_slow_creep`（§6 慢相估计与扣除）、`F5_slow_tradeoff`（§7.4 慢相取舍）、`F6_overview13`（§8.1 13 份总览）、`F7_fast_tradeoff`（§8.3 快相取舍）。

- `docs/figures/F1_two_phase.png`（250 KB）
- `docs/figures/F2_overview.png`（311 KB）
- `docs/figures/F3_invert_glide.png`（199 KB）
- `docs/figures/F4_slow_creep.png`（203 KB）
- `docs/figures/F5_slow_tradeoff.png`（236 KB）
- `docs/figures/F6_overview13.png`（347 KB）
- `docs/figures/F7_fast_tradeoff.png`（228 KB）

## 脚本（31）

`pv_common.py` 是公共层（数据清单、指标口径、缓存），`pv_style.py` 绘图层；`pv_run.py` 主体复算（13×5 臂）；`pv_trim_ablation.py` trim 三档消融；`pf1~pf7` 逐图产出；`pv_vision_check.py` 图件视觉验收；`pv_audit.py` 数字审计。

```text
ad_lib.py  ad_v4.py  glm53_v3.py  glm53_v5.py  glm53_v51.py  glm53_v6.py
pf1_physical.py  pf2_overview.py  pf3_invert_glide.py  pf4_slow_creep.py  pf5_slow_tradeoff.py  pf6_overview13.py
pf7_fast_tradeoff.py  pv_audit.py  pv_common.py  pv_run.py  pv_style.py  pv_trim_ablation.py
pv_vision_check.py
（编译缓存：__pycache__/ad_lib.cpython-314.pyc, __pycache__/ad_v4.cpython-314.pyc, __pycache__/glm53_v3.cpython-314.pyc, __pycache__/pv_common.cpython-314.pyc, __pycache__/pv_style.cpython-314.pyc, ad_lib.cpython-314.pyc, ad_v4.cpython-314.pyc, glm53_v3.cpython-314.pyc, glm53_v5.cpython-314.pyc, glm53_v51.cpython-314.pyc, glm53_v6.cpython-314.pyc, pv_common.cpython-314.pyc）
```

## 数据（22）

```text
_pv_run.log  _pv_trim.log  cache/中途切换-13ffca.npz  cache/中途切换-1d9493.npz
cache/再切换负载.npz  cache/切换负载-快相无责.npz  cache/右拇指指尖_数据1.npz  cache/右拇指指尖_数据2.npz
cache/右拇指指尖_数据3.npz  cache/四指指尖_数据1.npz  cache/四指指尖_数据2.npz  cache/四指指尖_数据3.npz
cache/左拇指指尖_数据1.npz  cache/左拇指指尖_数据2.npz  cache/左拇指指尖_数据3.npz  epochs_all.csv
metrics_all.csv  metrics_settle.csv  paper_tables.txt  trim_ablation.csv
vision_check.json  vision_check.md
```

## 注意

`pf1_physical.py` / `pf3_invert_glide.py` 要读 v6 的既有分析产物，路径已指向 `progress/07-v6/results/`；`pv_vision_check.py` 从 `~/.dsh/.credentials.yaml` 读 `DEEPSEEK_API_KEY`，不硬编码。

## 复现

```powershell
cd temp/v4.1flash/progress/11-paper-v6
$env:PYTHONIOENCODING='utf-8'
python scripts/pv_run.py             # ① 主体复算 13×5 臂（≈5 分钟）→ metrics_*.csv + cache/
python scripts/pv_trim_ablation.py   # ② trim 三档消融（≈4 分钟）
python scripts/pf1_physical.py       # 以下逐图
python scripts/pv_vision_check.py    # ③ 图件视觉验收（需 DEEPSEEK_API_KEY）
python scripts/pv_audit.py           # ④ 数字审计（期望 OK 74 / BAD 0）
```
