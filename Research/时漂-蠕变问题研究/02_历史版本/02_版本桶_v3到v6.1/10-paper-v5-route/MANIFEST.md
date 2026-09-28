# 10-paper-v5-route —— 论文（一）：无责 3 s / 5 s 路线

> 本文件由归档时生成，列出该版本桶的**全部文件**；版本之间的关系见 `../README.md`。

- 时间：2026-09-18 10:54 ~ 13:52
- 交付：`docs/抗蠕变漂移补偿算法.md`（论文正文）+ 6 张配图 + 全部可复现脚本与数据
- 口径：与 `11-paper-v6` 并列、同时间轴同指标定义，互不覆盖

论文正文写的是**已落地的 v5.1 路线**：把「零漂」与「时漂（蠕变）」显式区分为两种不同性质的量，
只处理后者，并用三条语义把它锁住（非负载段输出等于输入 / 快相免责期 / 输出封顶）。

**摘要数字**：9 组恒载负载段内原始漂移占幅度 5.32%~33.75%（均值 15.43%）；
算法把**慢相段时漂残余从 11.79% 压到 1.41%**，变载台阶透传比中位 0.96，
并用独立参考实现逐帧对拍确认落地实现与原型相等。
代价明确写出：从真实加载沿到扣除量到位约 17.7 s（2.5 s 抗瞬态误判 + 3 s 隔离快相 + 其余过程平滑）。

**图件审查留痕**：本轮 6 张图在定稿前做过像素级复核（裁剪放大、逐面板核对文字越界与重叠），
过程产物在 `12-fig-audit`。

## 文档（1）

| 文件 | 大小 |
|---|---|
| `docs/抗蠕变漂移补偿算法.md` | 50 KB |

## 图件（6）

> 位置：`docs/figures/`（与文档同目录，正文按 `figures/…` 引用）。

`F1_two_phase`（快慢相两段结构）、`F2_overview`（算法总览）、`F3_step_detect`（阶跃判据）、`F4_latency`（时延分解）、`F5_scenarios`（场景）、`F6_static9`（恒载 9 组）。

- `docs/figures/F1_two_phase.png`（189 KB）
- `docs/figures/F2_overview.png`（326 KB）
- `docs/figures/F3_step_detect.png`（163 KB）
- `docs/figures/F4_latency.png`（289 KB）
- `docs/figures/F5_scenarios.png`（131 KB）
- `docs/figures/F6_static9.png`（264 KB）

## 脚本（20）

`pd.py` 是公共层（数据装载、口径、指标），`figstyle.py` 是绘图层；`pf1~pf6` 逐图产出。跨桶依赖 `ad_lib / glm53_v3 / ad_v4 / glm53_v5 / glm53_v51` 已随桶复制到本桶 `scripts/`。

```text
ad_lib.py  ad_v4.py  figstyle.py  glm53_v3.py  glm53_v5.py  glm53_v51.py
pd.py  pf1_physical.py  pf2_overview.py  pf3_step_mech.py  pf4_timeline.py  pf5_scenarios.py
pf6_static9.py
（编译缓存：ad_lib.cpython-314.pyc, ad_v4.cpython-314.pyc, figstyle.cpython-314.pyc, glm53_v3.cpython-314.pyc, glm53_v5.cpython-314.pyc, glm53_v51.cpython-314.pyc, pd.cpython-314.pyc）
```

## 数据（7）

```text
f1_creep_law.csv  f1_two_phase_profiles.csv  f3_detect_stats.csv  f4_latency.csv
f5_scenarios.csv  f6_static9.csv  f6_static9_fast5s.csv
```

## 复现

```powershell
cd temp/v4.1flash/progress/10-paper-v5-route
$env:PYTHONIOENCODING='utf-8'
python scripts/pf1_physical.py   # F1 加载形状的实测事实
python scripts/pf2_overview.py   # F2 算法总览
python scripts/pf3_step_mech.py  # F3 阶跃判据
python scripts/pf4_timeline.py   # F4 时延分解
python scripts/pf5_scenarios.py  # F5 场景
python scripts/pf6_static9.py    # F6 恒载 9 组
```
