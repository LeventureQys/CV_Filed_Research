# temp/v4.1flash/progress —— 抗蠕变漂移补偿算法 · 版本化归档

> **位置变更（2026-09-20）**：连同整个 `temp/`，本目录已整树平移到
> `Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/v4.1flash/progress/` —— 相对仓库根**下移 3 级**，内容未改。
> 本文标题与正文里的 `temp/...` 是**当时**的写法：把 `temp/` 换成 `Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/` 就是当前位置；
> 脚本锚点的 +3 级修正、入库范围（录制 CSV 与 npz/npy 不进 git）与新归档索引，见 `<v2.7 根>/README.md` §4/§5。

> 本目录把原先散落在 `temp/v4.1flash/` 各处的**文档、图片、脚本、数据**按**算法版本**重新归位，
> 一个版本一个桶，桶内自包含（脚本可直接跑、依赖同目录可导入）。
> 归档时间：2026-09-18 · **结构重组：2026-09-19**（见 §10）· **`currentworking/` 换代：2026-09-20**（见 §11）·
> **「当前工作算法在哪」的强制约定：见「★★ 规定」** · 归档方式见文末「搬迁台账」。
>
> **原目录的 `Document/` `figures/` `results/` `scripts/` `paper/` `paper_v6/` `_audit/` `_crop/` `_review_crops/` `_zoom/` `backup/` `cpp_snippet_check/` `cpp_v5_check/` 以及 6 个顶层 `*.md` 已全部清空并删除**，
> 内容都在本目录内。

## ★ 本目录现在的三层结构（2026-09-19 重组，**阅读与引用请先看这里**）

| 层 | 目录 | 内容 |
|---|---|---|
| **历史** | **`archived/`** | **全部 13 个版本桶**（`archived/01-v3-baseline` … `archived/13-v6-assessment`）——本文 §1~§9 里出现的桶名（`07-v6`、`13-v6-assessment` 等）**现在都在这一层下面**，读路径时要加 `archived/` 前缀 |
| **当前** | **`currentworking/`** | **★ 当前正在工作的算法（唯一落点，强制规定见下一节「★★ 规定」）**：2026-09-20 起为 **v3.4 在线双态蠕变观测器**（`observer-v3`）的文档 + 脚本 + 结果 + 图，上一代 creep-mem 已整包退到 `currentworking/archived/`，见 §2.1 |
| **原始入口** | `legacy/` | 重组织前的两个 README（原文保留）+ 搬迁工具脚本 |

> 因此本文中凡写作 `07-v6/docs/…`、`13-v6-assessment/…` 的**可操作路径**，实际都在 `archived/` 下；
> §3~§6 的路径已按新结构更新，**§7 与 §9 是历史留痕，路径保留当时写法**（见那两节开头的说明）。

---

## ★★ 规定：`currentworking/` 下的才是「当前正在工作的算法」（唯一落点 · 强制）

> 本节是**阅读与维护的强制约定**，优先于本文其它章节；与 §4「版本管理规则」冲突时以本节为准。
> 目的只有一个：**任何时候都能一眼判定「现在跑的算法是谁」，以及它的说明在哪。**

**R1 唯一性 —— 当前工作算法只允许放在 `currentworking/` 下。**
整个研究归档（本 `progress/`、其下 `archived/` 与 `legacy/`，以及同级的 `plan/`、`算法数据&原始数据/`、`原始数据only/`）中，
描述「当前正在工作的算法」的说明、脚本、结果与图**只能位于 `currentworking/` 内**。
其它位置出现的算法文档，无论名字里带不带"当前 / 最新 / 现役 / 交接"，**一律只能算历史或候选**，
不得当作当前工作算法来阅读、引用或据以改码。

**R2 判定判据 —— 只看位置，不看名字。**
判定某份材料"是不是当前工作算法"，**唯一依据是它在不在 `currentworking/` 下**；
桶号（`archived/07-v6` …）、文档标题、日期、版本号都不构成判据。
`currentworking/` 内（含其 `document/` 子目录）的说明文档 = 当前工作算法的权威说明。

**R3 一代一包 —— 当前一代只留一代。**
`currentworking/` 同时只承载**一代**算法：这一代的文档 + 它自己的 `scripts/` `results/` `figure/`（可原地重跑，见 R5）。
被取代的历代实现**整包退出**到 `currentworking/archived/<版本名>/`（现有 `archived/creep-mem/` 即此形态），
更早的版本进 `progress/archived/`。
**禁止把两代以上的算法文档并列堆在 `currentworking/` 根下或 `document/` 下。**

**R4 事实优先级 —— 代码 > 当前工作版文档 > 其它一切文档。**
三者冲突时按此顺序取值（沿用 `<v2.7 根>/README.md` §7）：
① `src/` 的实际实现（含录制 `session.json` 的 `algorithm.id` / `param_set`）；
② `currentworking/` 的说明；③ `archived/`、`plan/`、论文等。
即：文档写 A、代码做 B ⇒ **改文档**，不得改代码去迁就文档。

**R5 桶内自包含 —— 读它不需要先读历史。**
`currentworking/` 是**交接包**：正文自洽（不要求先读 `archived/` 才看得懂）；
脚本可原地重跑（依赖随包 `scripts/`，数据走 `原始数据only/`、`算法数据&原始数据/`）；
**文档里的图/附件相对引用必须逐条可达** —— 移动文档或图之后必须同步修正相对路径，并在 §2.1 更新位置。

**R6 同步义务 —— 动了当前算法就必须跟改这里。**
凡改动当前工作算法的行为、参数或落盘标识（`src/domain/drift_v6/**`、菜单开关、
`DataHandler::CurrentDisplayAlgorithm()` 的 `id`/`param_set`），**必须在同一次改动内**更新
`currentworking/` 内对应的说明文档，并按项目规则记入 `Document/ChangingLog/完整更新日志.md`。
只改代码不改 `currentworking/` = 违规。

**R7 过时即搬家 —— 不许含糊留旧。**
`currentworking/` 内不得留存"上一代的文档"：判为过时的文档要么移入 `currentworking/archived/<版本名>/`，
要么在文档顶部显式标注"已被 X 取代"并给出指针。
`archived/`、`legacy/` 是**只读历史**：只允许格式性修复（如标题前导空格），不得回改内容、不得追加新内容。

**R8 变更要记账 —— 换代必须留痕。**
每次换代（新算法取代旧算法）必须同时完成三件事：
① 更新本文 §2.1 的 `currentworking/` 内容表；② 在本文的结构变更章节（现为 §10 重组 / §11 换代）追加一条记录（做了什么 / 为什么 / 影响面）；
③ 记入 `Document/ChangingLog/完整更新日志.md`。三条缺一视为未完成。

**当前状态（2026-09-20）**：`currentworking/` 装的是 **v3.4 在线双态蠕变观测器**（`observer-v3`），
产品实现 `src/domain/drift_v6/creep_observer.{h,cpp}`，菜单「设备 → 时漂/零漂补偿（v3.4 观测器）」，
落盘 `algorithm.id = drift_v34_creep_observer` / `param_set = plan-v3.4 observer-v3`；
上一代事件式「蠕变记忆」（creep-mem）已整包退出到 `currentworking/archived/creep-mem/`；
**重组前放在本桶的 v6 交接包（`00_交接说明.md` / `01_v6算法说明_当前实现.md` / `02_算法更新日志.md` / `MANIFEST.md`）
已整包退出本目录、全树不再存在** —— 需要 v6 档（`plan-v3.1 PCT-fix`）口径时读
`archived/07-v6/`、`../plan/v3.0/`、`../plan/v3.1/`，它们按 R1 **属历史**。

---

## 1. 先看这张表：版本谱系

> **下表的桶名都是 `archived/` 下的目录名**（如 `archived/07-v6`）。

| 桶 | 版本 / 路线 | 时间 | C++ 落地状态 | 一句话定位 | 关键数字 |
|---|---|---|---|---|---|
| `01-v3-baseline` | **v3** 现役基线 | 09-03 落地 | 曾被实施，已被 v5.1 取代 | 卸载门控自动归零 + 负载比例蠕变场 | 全段时漂 1.58%、慢相段 1.49%、台阶捕获比 0.89 |
| `02-dsp-route` | **dsp.md 逆滤波**（并行路线） | 09-17 | **否决，未落地** | 二阶逆 IIR + 空间反卷积「直接反演」蠕变 | 时漂残余 1.55% → **15.2~32.3%**、噪声 ×2.44 |
| `03-v4` | **v4** 快相免责期 | 09-17 | 原型（骨架部分适用） | 加载快相「放它过去」，只治慢相蠕变 | 免责 3 s / 5 s 两档；变载欠报 9.6% → 5.9% |
| `04-v5` | **v5** 四项定点修复 | 09-17 | **已落地** `src/domain/drift/` | 修 P1 门限 / P2 基线 / P3 免责范围 / P4 插入点 | 台阶捕获比 0.89 → **0.98**、全程最大偏差 31.3% → **9.0%** |
| `05-v5.1` | **v5.1**（**当前现役**） | 09-17 | **现役** `src/domain/drift/` | 取消空载强制归零，零点=传感器当前读数 | 零点被钳 0 时长 0~6.64 s → **0~0.27 s** |
| `06-v5.1-exempt-1s-3s-5s` | 免责期 **1 s / 3 s / 5 s** 专项 | 09-18 | 运行期参数（菜单三档） | 把「无责」窗压到 1 s 会怎样 | 1 s 档把真实加载当蠕变扣掉 ⇒ **不可取**；5 s 变载跟踪最好 |
| `07-v6` | **v6** 形状约束反演 + 滑行器 | 09-18 | **已落地** `src/domain/drift_v6/` | 用标定形状把快相「算掉」，慢相沿用 v5 | T_stable **3.82 s → 0.55 s**；代价：实录最大偏差 1889 → 4571 ADC |
| `08-v6.1` | **v6.1** 阶跃尖峰定点修复 | 09-18 | 原型，未落地 | 形状 ROM 比现场加载慢 ⇒ Â 系统性高估 | 尖峰「峰后回落」+11.9% → **+0.4%** |
| `09-v7` | **v7** 已否决变体 | 09-17 | 未采用 | 免责期只在首次 onset 生效 | 在变化负载上**与 v3 逐帧完全相同** |
| `10-paper-v5-route` | **论文（一）** 无责 3 s / 5 s 路线 | 09-18 | — | 把 v5.1 写成论文 + 全部取舍 | 慢相段时漂残余 **11.79% → 1.41%**、台阶透传中位 0.96 |
| `11-paper-v6` | **论文（二）** v6 抗蠕变补偿算法 | 09-18 | — | v6 纯算法部分 + 为拟合实机做的全部 trade-off | 约 4.5 万字 + 7 张图（vision 验收 7/7 PASS；图号 F1~F7 按正文出现顺序，见 §9.1） |
| `12-fig-audit` | 图件审查留痕 | 09-18 | — | 交付图的像素级复核过程产物（179 个文件） | — |
| `13-v6-assessment` | **v6 评估与需求答复**（plan v1.0） | 09-18 | 只读分析，未改代码 | 答复 `plan/v1.0/需求文档.md` 的 6 组问题：三阶段时间特征、快相形状稳定性、onset/restep 两种形态、过充与滤波的作用边界、扰动下的基线识别、可重复性与卸载时漂 | 形状跨传感器差 15 个百分点；现役 ROM 偏慢 3~5 pt ⇒ Â 过充中位 +5.0%（最坏 +12.8%）、restep 欠充 −6.3%；卸载在 0.2 s 内完成 100% |

两条**并行路线**（不是版本演进）：`02-dsp-route`（逆滤波，已否决）；
两篇**并列论文**：`10`（无责 3 s/5 s 路线）与 `11`（v6 路线），口径一致、互不覆盖。

### 演进主线（一图流）

```
v3 现役基线 ──┬─► dsp.md 逆滤波路线 ──► 评审+实测否决（02）
              │
              └─► v4 快相免责期（03）──► v5 四项定点修复（04）──► v5.1 取消空载归零（05，现役）
                        │                                              │
                        │                                              ├─► 免责期 1s/3s/5s 三档（06）
                        │                                              └─► v6 形状约束反演（07，已落地）
                        │                                                       └─► v6.1 尖峰修复（08，原型）
                        └─► v7「只在首次 onset 生效」──► 无收益，存档（09）

论文线：无责 3s/5s 路线（10）  ‖  v6 路线（11）

当前工作线（不在这张表里 · 不在 archived/）：v3.4 在线双态蠕变观测器 —— 见 §2.1 与「★★ 规定」
```

> ⚠️ **§1 这张谱系表与上面的主线图只覆盖 `archived/` 下的 13 个历史桶**（v3 → v6.1 与两篇论文）。
> **当前正在工作的算法不在其中**：2026-09-20 起 `currentworking/` 装的是 **v3.4 在线双态蠕变观测器**（§2.1）。

---

## 2. 目录结构

```
temp/v4.1flash/                       ← 同级还有 plan/（第二轮调研，不属本归档）
└── progress/
    ├── README.md                    ← 本文件：版本谱系 + 阅读顺序 + 归档规则 + 搬迁台账 + 重组记录
    ├── archived/                    ← 【历史】全部 13 个版本桶（2026-09-19 移入本层）
    │   ├── 01-v3-baseline/
    │   │   ├── MANIFEST.md          ← 每个桶都有：版本定位 + 全部文件清单 + 复现命令
    │   │   ├── source/              ← v3 的 C++ 源码快照（drift_compensator.*.v3bak）
    │   │   └── scripts/
    │   ├── 02-dsp-route/{docs,scripts,results}/        ← 图在 docs/figures/
    │   ├── 03-v4/{docs,figures,scripts,results,cpp_check}/
    │   ├── 04-v5/{docs,figures,scripts,results,cpp_check}/
    │   ├── 05-v5.1/{docs,scripts,results}/
    │   ├── 06-v5.1-exempt-1s-3s-5s/{docs,figures,scripts,results}/
    │   ├── 07-v6/{docs,figures,scripts,results}/
    │   ├── 08-v6.1/                 ← ⚠️ 目前为空桶（0 文件、无 MANIFEST），见 §10.3
    │   ├── 09-v7/{scripts,results}/
    │   ├── 10-paper-v5-route/{docs,scripts,results}/   ← 图在 docs/figures/
    │   ├── 11-paper-v6/{docs,scripts,results}/         ← 图在 docs/figures/
    │   ├── 12-fig-audit/figures/{_audit,_crop,_review_crops,_zoom}/
    │   └── 13-v6-assessment/{docs,figures,results,scripts}/   ← plan v1.0 需求答复（只读分析）；⚠️ 无 MANIFEST
    ├── currentworking/              ← 【当前】★ 当前正在工作的算法（唯一落点，见「★★ 规定」+ §2.1）
    │   ├── 算法说明_v3.4观测器.md / 规律辨识报告.md / 观测器原型报告.md
    │   ├── document/                ← 当前一代的正式文档（主文与移植文档）
    │   │   ├── 在线双态蠕变观测器补偿算法.md      ← ★ 主文档
    │   │   └── 观测器嵌入式移植文档.md
    │   ├── scripts/                 ← 该代脚本（可原地重跑：原型核心 / 对拍 / 辨识 / 出图 / C++ 离线助手）
    │   ├── results/                 ← 该代数值结果
    │   ├── figure/                  ← 该代图（concept/ + derive/ + observer/）
    │   └── archived/creep-mem/      ← 上一代（事件式「蠕变记忆」）整包，只读
    └── legacy/
        ├── docs/                    ← 重组织前的两个 README（原始入口，按原文保留）
        └── relocation/              ← 首次归档用的分类/搬迁/校验脚本（可追溯）
```

桶内子目录，含义与原来一致：

| 子目录 | 内容 | 原来的位置 |
|---|---|---|
| `docs/` | 该版本的正式文档、报告、论文 | `Document/`、顶层 `*.md`、`paper*/` |
| `docs/figures/` | **文档内嵌的图**（02 / 10 / 11 三个桶），与正文同目录、正文直接写 `figures/…` | `paper*/figures/` |
| `figures/` | 该版本产出的图（其余桶），文档按 `../figures/…` 引用 | `figures/`、`_audit` 等 |
| `scripts/` | 该版本的复算脚本（**桶内自包含**） | `scripts/`、`paper*/scripts/` |
| `results/` | 该版本产出的 csv/npz/log | `results/`、`paper*/results/` |
| `source/`、`cpp_check/` | 源码快照、C++ 编译自检工程（仅 01/03/04 桶有） | `backup/`、`cpp_snippet_check/`、`cpp_v5_check/` |

> **图的两种位置约定**（都要求「文档里的引用能解析到图」，出图脚本的 `FIG` 与图的实际位置一致，重跑不会写错地方）：
> ① 论文/报告类桶（`02`、`10`、`11`）：图与文档同目录（`docs/figures/`），正文写 `figures/…`；
> ② 其余含图的桶（`03`、`04`、`06`、`07`、`08`）：图在桶根 `figures/`，文档写 `../figures/…`。
> 归档后已逐条核对：**全库文档里的图引用 37 处全部可达**（唯一例外是 `legacy/` 里那份重组织前的 README，其中的路径是历史路径，故意未改）。
> 这两条约定**与重组无关**（桶内相对路径不变，同层桶之间也没被拆开），故重组后仍然成立。

### 2.1 `currentworking/` —— 当前正在工作的算法（★ 唯一落点）

**这里装的才是"当前正在工作的算法"**（强制规定见上一节「★★ 规定」）。
一**代**算法 = 一个自包含包：说明文档 + 它自己的 `scripts/` `results/` `figure/`；
被取代的历代实现**整包退出**到 `currentworking/archived/<版本名>/`，更早的版本进 `archived/`。

**当前一代 = v3.4 在线双态蠕变观测器**（`observer-v3`，2026-09-19 落地 / 09-20 整理入桶）
—— 产品实现 `src/domain/drift_v6/creep_observer.{h,cpp}`，菜单「设备 → 时漂/零漂补偿（v3.4 观测器）」，
落盘 `algorithm.id = drift_v34_creep_observer` / `param_set = plan-v3.4 observer-v3`
（值域口径与另外三档的对照见 `<v2.7 根>/README.md` §1）。

| 路径 | 内容 |
|---|---|
| **`document/在线双态蠕变观测器补偿算法.md`** | **★ 主文档**（论文体，≈44 KB）：要解决的问题（v6 整片卸载丢基线 2516 ADC / creep-mem 误伤域外工况 ≈1200 ADC）、模型与推导、参数、全部实测（一致性 10/10、恒载漂移、首次加载稳定 2.0~2.6 s）与**如实列出的代价**、明确未做项 |
| `document/观测器嵌入式移植文档.md` | **移植包**：最小接口、11 个参数、逐帧执行顺序、伪码（与 `creep_observer.cpp` 的 `Process()` 逐行对应）、内存账（5 float/通道）、重置语义与移植陷阱 |
| `算法说明_v3.4观测器.md` | **实现细则**（源码为唯一事实来源）：逐帧主流程、零点判定与启动语义（带载启动 / 调零即重置）、参数表、边缘语义；与 Python 原型**逐帧对拍 PARITY OK（max\|Δ\|=5e-5 ADC）** |
| `规律辨识报告.md` | **前置定量边界**：9 参数线性粘弹性律 + 留一法；纯规律可预测精度 ≈1000 ADC ⇒ 零状态算法的一致性上限就在此 |
| `观测器原型报告.md` | 原型阶段（双时间常数版）实测与**未关闭项**（空载语义 / 超长保压 τ₂ 取舍 / 参数重标 / 落地形态）；含与 v6、creep-mem 的满载族极差对照（2516 / 1018 / ≈950） |
| `scripts/`（19 个源文件） | 原型核心 `v34_observer_core{,2,3}.py`、C++↔Python 对拍 `v34_parity_check.py`、规律辨识 `v34_law_identify.py`、出图 `v34_make_{concept,derive}_figs.py`、约定/稳定时间探针、公共层 `v30_lib.py`、C++ 离线助手 `v30_runner.cpp` / `obs_runner.cpp`（+ `build_*.bat`、`build/` 产物） |
| `results/`（7） | v3 评估、v2 评估、约定对比、规律辨识、`T_stable` 探针、全量对照等数值留痕 |
| `figure/`（39） | `concept/` 12（机制示意，图 1–5、8–14）、`derive/` 2（式形推导，图 6–7）、`observer/` 23（20 个数据集各一张 + 2 张总图）、桶根 2 张（约定对比、规律辨识） |
| `archived/creep-mem/` | **上一代**（事件式"蠕变记忆"）整包、只读：`问题清单.md` / `验收报告.md` / `algorithm/`（v6 源码快照）/ `scripts/` / `results/` / `figure/` |

> **桶内现状（2026-09-20 实测 127 个文件）**：本桶**当前没有** `MANIFEST.md`，也没有 `00_交接说明.md`。
> 重组前放在这里的 **v6 交接包**（`00_交接说明.md`、`01_v6算法说明_当前实现.md`、`02_算法更新日志.md`、`MANIFEST.md`）
> **已整包退出本目录**（全树检索已无这四份文件）。需要 v6 档（`plan-v3.1 PCT-fix`）口径时读
> `archived/07-v6/`、`../plan/v3.0/`、`../plan/v3.1/`、`../plan/v2.0/03_当前算法说明.md` —— 按 R1 它们**属历史**。
>
> **与 `archived/` 的关系**：`archived/07-v6/` 是 v6 的设计规格与原型（也是本代算法的动机出处），
> 维护当前算法读 `currentworking/`，追溯"为什么这么设计"才读 `archived/`。
> **自检**：`../plan/v3.0/scripts/v30_check_docsync.py` 检的是**旧 v6 交接包那 4 份文档**，
> 换代后已失效（`FILES` 列表与旧 `BASE` 都指向已退出的位置），下次改动时重写或改用本桶脚本自检。

---

## 3. 按目的选阅读路径

| 你想…… | 读这些 |
|---|---|
| **确认"哪个才是当前算法"** | 本文件「★★ 规定」+ §2.1（**唯一判据 = 是否在 `currentworking/` 下**）；代码真相 = `src/domain/drift_v6/creep_observer.{h,cpp}`，运行期真相 = 录制 `session.json` 的 `algorithm.id` / `param_set` |
| **快速知道现状**（现在跑的是什么、瓶颈在哪） | **`currentworking/document/在线双态蠕变观测器补偿算法.md`（★ 当前工作算法主文档）** → `currentworking/算法说明_v3.4观测器.md`（实现细则）→ `currentworking/观测器原型报告.md` §5（未关闭项）→ `archived/07-v6/docs/07-v6算法说明.md`（上一代 v6 规格，**历史**） |
| **理解算法是怎么一步步改出来的** | `archived/03-v4/docs/02~04` → `archived/04-v5/docs/05-v5算法说明.md` → `archived/05-v5.1/docs/06` → `archived/07-v6/docs/07` → `archived/08-v6.1/docs/08` |
| **看「加载后多久能稳定」这条线** | `archived/03-v4/docs/快相与慢相分离分析.md`（快相/慢相量化）→ `archived/06-v5.1-exempt-1s-3s-5s/docs/免责期1s-3s-5s对比.md` → `archived/07-v6/docs/v6与免责1s3s对比.md` |
| **评审算法／写报告** | 直接读两篇论文：`archived/10-paper-v5-route/docs/抗蠕变漂移补偿算法.md`、`archived/11-paper-v6/docs/抗蠕变补偿算法_v6.md`（含全部 trade-off 台账） |
| **搞清为什么没采用另一条路线** | `archived/02-dsp-route/docs/dsp方案评审报告.md`（含 `results/superseded/` 两次自我推翻的留痕） |
| **复现某个数字** | 对应桶的 `MANIFEST.md` → 里面的复现命令 → 该桶 `scripts/`（依赖已随桶复制，可直接 `python scripts/xxx.py`） |
| **核对某张图怎么来的** | 桶内 `MANIFEST.md` 的图件清单；`archived/10-paper-v5-route` 的图件复核过程在 `archived/12-fig-audit` |
| **评估当前算法 / 准备下一代方案** | **`currentworking/document/在线双态蠕变观测器补偿算法.md`（当前一代的模型、实测与代价）** → `currentworking/观测器原型报告.md` §5（未关闭项：空载语义 / 超长保压 τ₂ / 参数重标 / 落地形态）→ `currentworking/规律辨识报告.md`（可预测精度上限）→ `archived/13-v6-assessment/docs/v6评估与需求答复.md`（三阶段特征、快相形状稳定性、两种形态、过充与滤波边界、可重复性与卸载时漂），源需求在 `../plan/v1.0/需求文档.md` |
| **看 v6 的现场失效定因（基线飞掉）** | `../plan/v1.0/T9_现场回归_v6参数复核/分析报告_T9.md`（大偏置工况下空载判据结构性失效）→ 这一代怎么绕开它见 `currentworking/document/在线双态蠕变观测器补偿算法.md` §1.1~§1.3（丢基线 2516 ADC、事件式记忆被否决、规律辨识的定量边界）；v6 档当时的修复记录随 v6 交接包退出 `currentworking/`，读 `../plan/v3.1/` |

---

## 4. 版本管理规则（本归档怎么组织的）

### 4.1 分桶依据

分类**不是按名字猜的**，依据三条证据链：

1. **时间线**：脚本用两字母前缀（`aa_`→`di_`）编码了书写顺序，配合文件 mtime，
   可以切出 `dsp 评审 / v4 / v5 / v5.1 / 免责三档 / v6 / v6.1` 七个阶段；
2. **依赖关系**：解析每个 `.py` 的 `import`，得到模块依赖图，
   据此判定某个算法原型是「某版本独享」还是「多版本共用」；
3. **产出反查**：图件由源码里的 `savefig` 反查产出脚本；结果数据由「哪个脚本引用了这个文件名」反查；
   反查不出来的再用命名前缀规则，最后人工核对（当时 706 个文件全部落到桶里，没有一个下落不明）。

### 4.2 独享 ⇄ 共享的处理

- **某版本独享**的文件 → **剪切**到该桶（原件不再保留在原目录）；
- **多版本共用**的文件 → **复制**到每个用到的桶，再删除原件。
  共用清单（共 9 个模块 + 1 份文档，各自散落在多个桶）：

| 共用文件 | 出现在 | 为什么共用 |
|---|---|---|
| `glm53_v3.py` | 01,02,03,04,05,06,07,08,09,10,11 | v3 是所有版本的对拍基线 |
| `ad_v4.py` `ad_lib.py` | 03,04,05,06,07,08,10,11 | v4 算法本体 + 全工作区公共工具层（数据装载/事件检测/指标） |
| `glm53_v5.py` | 04,05,06,07,10,11 | v6 的慢相模块逐行沿用 v5 |
| `glm53_v51.py` | 05,06,07,10,11 | v5.1 是现役实现，v6 与论文都以它为对照 |
| `glm53_v6.py` | 07,08,11 | v6.1 是 v6 的定点修复 |
| `glm53_v7.py` | 03,09 | v4 的变载脚本（`z8_gap`）拿 v7 做对照 |
| `pv_common.py` | 08,11 | v6.1 复用论文（二）的公共层与指标口径 |
| `05-v5算法说明.md` | 04,05 | 该文 §1~§9 讲 v5，§10 起是 v5.1 修订 |

> 这样处理后每个桶**自包含**：同目录 `import` 全部可解析（已逐文件校验，0 处失败），
> 不再有「脚本在 A 桶、依赖在 B 桶」的隐性耦合。

### 4.3 脚本路径锚点的统一修正（重要）

脚本原来位于 `temp/v4.1flash/scripts/`，用这样的锚点推算路径：

```python
HERE = os.path.dirname(os.path.abspath(__file__))   # scripts/
OUT  = os.path.dirname(HERE)                        # temp/v4.1flash  （results/ figures/ 在这里）
TEMP = os.path.dirname(OUT)                         # temp/           （数据集在这里）
```

搬进 `progress/archived/<桶>/scripts/` 后 `OUT` 正好等于桶目录（`results/`、`figures/` 也在桶里，无需改），
但 `TEMP` / `ROOT` 少算了两级，跨桶引用（论文读 v6 产物、v6.1 引用论文公共层）也指错了地方。

因此归档时**统一修正了 95 个脚本、105 处路径锚点**：

| 原写法 | 修正后 | 处数 |
|---|---|---|
| `TEMP = os.path.dirname(OUT)` | 再上溯两级（→ `temp/`） | 62 |
| `sys.path.insert(0, os.path.join(OUT, "paper_v6", "scripts"))` | `sys.path.insert(0, HERE)`（依赖已随桶复制） | 13 |
| `ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))` | 再上溯两级（→ 仓库根） | 7 |
| `sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))` | 改为按 `__file__` 定位 | 6 |
| 论文侧 `FLASH = os.path.dirname(ROOT)` / `FLASH/scripts` | 重算 `FLASH`、改为同目录导入 | 7 |
| 论文（二）读 v6 产物 `C.FLASH/results/...` | 指向 `progress/07-v6/results/` | 2 |
| 其余个案（`v_shape_diag` 锚点块、`_diff` 的绝对路径与 exe 位置、数据集拼路径） | 见 `legacy/relocation/patch_paths.py` | 8 |

**修正后的校验结果**：199 个脚本的锚点全部解析正确（`OUT`=桶、`RES`=桶/results、`FIG`=桶/figures、`TEMP`=`temp/`）；
同目录 `import` 0 处失败；实跑抽检（`04-v5` 载入 16354 帧实采数据并跑 `GLM53v5`、
`05-v5.1` 载入 19973 帧并跑 `GLM53v51`、`08-v6.1` 的 `pv_common` 自检 `TEMP` 指向正确）全部通过。

### 4.4 其它归档约定

- `MANIFEST.md` 放在**桶根**，列出该桶全部文件与复现命令；本 README 只讲**版本之间**的关系。
- 脚本目录里形如 `glm53_v3.cpython-314.pyc` 的**编译缓存是惰性的**（Python 3 只读 `__pycache__/`），
  按「保留全部文件」原则原样留下，可安全忽略或删除。
- `12-fig-audit` 收的是**过程留痕**（裁剪/放大/像素扫描），审查对象是中间修订版的交付图，
  与最终图尺寸不一一对应属预期，不要当成交付物。

---

## 5. 复现方式

每个桶的 `scripts/` 都是自包含的，进入桶目录即可运行：

```powershell
cd temp/v4.1flash/progress/archived/<桶名>     # 桶现在都在 archived/ 下
$env:PYTHONIOENCODING='utf-8'          # 控制台中文/特殊符号需要
python scripts/<脚本>.py
```

依赖（本机已有）：**Python 3.14 + numpy 2.4.6 / scipy 1.17.1 / pandas 3.0.3 / matplotlib 3.10.9**。
数据集仍在原处（`temp/右拇指指尖/`、`temp/左拇指指尖/`、`temp/四指指尖/`、`temp/变化负载/`），
脚本通过修正后的 `TEMP` 锚点访问，**不需要改任何路径**。

各桶的典型入口（完整清单见各桶 `MANIFEST.md`；下表的桶号均为 `archived/` 下的目录）：

| 桶 | 入口脚本 | 说明 |
|---|---|---|
| `archived/02-dsp-route` | `v_dsp_review_probe.py`、`v_final_check.py`、`w2_dsp_vs_v3.py` | 秒级 → 约 4 分钟 |
| `archived/03-v4` | `ac_final5.py`、`r_fastphase.py`、`z8_gap.py` | 约 5 分钟 |
| `archived/04-v5` | `ba_v5_scenarios.py`、`bd_v5_static9.py`、`bg_v5_cpp_parity.py` | 对拍需先构建 `cpp_check` |
| `archived/05-v5.1` | `bo_v51_verify.py` | 零点 + 负载跟踪 + 恒载 9 组 |
| `archived/06-v5.1-exempt-1s-3s-5s` | `bp_exempt_sweep.py`、`ca_all_datasets_1s.py` | 约 10 分钟，产出 H1/H2/H6/H7 |
| `archived/07-v6` | `ci_v6_vs_1s_3s.py`、`cp_v6_detwin_sweep.py`、`ct_v6_verify_figs.py` | 产出 I1/I2、J1、K1 |
| `archived/08-v6.1` | ⚠️ 桶当前为空（见 §10.3） | 原 `cw_v61_ab.py`、`dc_v61_figs.py`、`dh_v61_13ffca.py` |
| `archived/10-paper-v5-route` | `pf1~pf6_*.py` | 逐图产出（论文 6 图） |
| `archived/11-paper-v6` | `pv_run.py` → `pv_trim_ablation.py` → `pf1~pf7` → `pv_vision_check.py` → `pv_audit.py` | 论文（二）全流程，约 10 分钟 |

> `archived/04-v5/cpp_check/build/` 是随桶搬来的构建产物，其中的绝对路径已失效；
> 需要 C++↔Python 对拍时请用 `cmake -S archived/04-v5/cpp_check -B <新目录>` 重新 configure
> （该 `CMakeLists.txt` 的 `ROOT` 锚点是 `../../..`，重组后已少算一级，见 §10.3）。
> `archived/11-paper-v6/scripts/pv_vision_check.py` 需要 `DEEPSEEK_API_KEY`（从 `~/.dsh/.credentials.yaml` 读取，不硬编码）。

---

## 6. 数据来源（各桶共用，未搬动）

| 位置 | 组数 | 通道/行列 | 时长 | 原始时漂（主通道） |
|---|---|---|---|---|
| `temp/右拇指指尖/数据1~3` | 3 | 31ch / 9×7 | 163~234 s | 23~33% |
| `temp/左拇指指尖/数据1~3` | 3 | 31ch / 9×7（mask 与右拇指镜像） | 192~199 s | ~5% |
| `temp/四指指尖/数据1~3` | 3 | 21ch / 8×5 | 191~200 s | 10~15% |
| `temp/变化负载/…`（4 份实录） | 4 | 21ch / 8×5 | 64~200 s | 11.4% / 21.2% |

- 时间轴必须用 `timestamp`（指尖数据重复率 0.01~0.04%），`elapsed` 量化重复 38~72% **不可用**；
- 恒载 9 组为 `processed_display` 域；变化负载为 ADC 域且帧到达呈成批突发；
- 盘点脚本与结果在 `archived/02-dsp-route/`（`inventory.py` → `results/dataset_inventory.csv`）。

---

## 7. 搬迁台账（可追溯，2026-09-18）

> ⚠️ **本节是历史留痕**：记的是**首次归档**（把散落在 `temp/v4.1flash/` 的文件按版本分桶）的过程与数字，
> 其中的桶路径写在 **2026-09-19 重组之前**（当时桶是 `progress/` 的直接子目录，现在都在 `progress/archived/` 下）。
> 本节数字**未随重组改动**（重组只移动目录层级、不增删内容文件，见 §10.2）。

| 项目 | 结果 |
|---|---|
| 源树文件总数 | **706**（`temp/v4.1flash/` 下，不含 `progress/`） |
| 唯一文件落桶 | 706 / 706，无遗漏（守恒校验逐文件核对） |
| 独享文件（剪切） | **689** 个 |
| 共享文件（复制到多桶后删原件） | **17** 个源文件 → 92 份副本（净增 75） |
| 实际动作 | `move` 706 次（689 独享 + 17 共享收尾）、`copy` 75 次 |
| 归档后内容文件 | **781** = 706 + 75 |
| 本次新增交付 | 12 个 `MANIFEST.md` + 本 `README.md` + `legacy/relocation/` 5 个工具脚本 = 18 |
| **归档时文件总数** | **799**（781 + 18），`temp/v4.1flash/` 下只剩 `progress/` 一个目录 |
| 脚本路径锚点修正 | 95 文件 / 105 处 |
| 校验 | 锚点 199/199 正确；同目录 import 0 失败；3 个桶实跑通过 |

> 归档之后又做了一轮**用图修复与依赖补齐**，文件数变为 **808**，详见 §9。

**过程中的两处问题与处理（留痕）**：

1. **分类器重复计数**：`paper/`、`paper_v6/` 下的脚本同时被「依赖图」和「目录树」两条路径收录，
   导致首次搬迁在 `10-paper-v5-route` 中断（已 move 的部分完好）。修正为去重后幂等续跑，0 错误完成。
2. **同名目标覆盖**：顶层 `README.md` 与 `Document/README.md` 的目标路径都是 `legacy/docs/README.md`，
   后者被覆盖。发现后按守恒核算精确定位（781 vs 780，差额恰为 1），
   已按原文恢复为 `legacy/docs/README-Document目录索引（原始，按原文恢复）.md`
   （字节数与原始一致：7373 B），顶层那份改名为 `README-根目录总入口（原始）.md`（11230 B）。

`legacy/relocation/` 下保留了可追溯的工具：
`classify.py`（分类规则与证据链）、`execute.py`（剪切/复制执行器）、
`patch_paths.py`（路径锚点修正表）、`verify.py`（锚点/import/引用校验）、
`make_manifests.py`（本套 MANIFEST 生成器）。

另有一份**回退点**：`temp/archived/_v41_organize/`（原 `temp/_v41_organize/`，2026-09-18 整理 `temp/` 时一并归档）
是本次归档的工作目录，其中
`scripts.backup/` 是**路径锚点修正前**的 151 个脚本原件（若想对照或回退修正，从这里取）；
`plan.json` 是搬迁时的作业台账。该目录不属于归档产物，可直接删除。

---

## 8. 边界与未验证项

- **未修改 `src/` 任何代码**：本次只做 `temp/` 下的归档与脚本路径修正；
  仓库源码、CMake、构建树、测试均未触碰。
- **界面手测未做**：v5.1 与 v6 虽已落地 `src/domain/`，但按项目规范，界面/真机验证由用户执行。
- **各版本的已知未验证项以各自文档为准**，几处共性的：
  跨载荷量级（5N/10N/20N）的快相形状一致性、加载保持 <5 s 的短保压工况、多档反复变载；
  论文（二）的形状库用测试集自身标定（**乐观上界**），时间轴对 τ<0.2 s 有畸变（v6 数字可能偏低）。
- **v6.1 未落地 C++**：文档给出了补丁与代价清单，但 `src/domain/drift_v6/` 目前是 v6 语义。

---

## 9. 归档后修订（用图修复与依赖补齐，2026-09-18）

> ⚠️ **本节同样是历史留痕**（与 §7 同理）：桶路径按当时写法（现在应加 `archived/` 前缀）。

用户提出「论文（二）的用图修一下」后做的第二轮处理，同时修掉了归档时遗留的三处问题。

### 9.1 论文（二）的图：位置、编号与引用

| 问题 | 处理 |
|---|---|
| 图被放在桶根 `figures/`，而正文按 `figures/…`（相对文档）引用 ⇒ 图不显示 | 7 张图移入 `docs/figures/`；`pv_common.py` 的 `FIG` 改为 `ROOT/docs/figures`，重跑出图写入同一目录 |
| 编号与出现顺序不一致（原为 F1→F2→F7→F5→F6→F3→F4） | 按**正文出现顺序**重排为 F1~F7；三张改名（`F7_slow_tradeoff`→`F5_`、`F5_overview13`→`F6_`、`F6_fast_tradeoff`→`F7_`），出图脚本同步改名为 `pf5_slow_tradeoff.py` / `pf6_overview13.py` / `pf7_fast_tradeoff.py`，使「`pfN_*.py` ↔ 图 FN」始终成立 |
| 图内标题写着旧图号 | 三张图**重新出图**（只改标题图号与输出名，绘图逻辑与数据未变），逐图 `figcheck` 通过；`results/vision_check.md` 追加了重编号对照说明（原记录保留不回写） |
| 图 F3（形状反演与滑行器）与图 F4（慢相估计与扣除）被埋在 §8.4，离讲述它们的 §5/§6 有 300~400 行 | 分别移入 §5、§6 末尾，并给 §8.4 的两段实测补上「（图 F3…）（图 F4…）」引用 |

### 9.2 其余桶的图引用与依赖

| 项目 | 说明 |
|---|---|
| 其余桶的图引用 | `03-v4`、`06`、`07`、`08` 共 6 份文档里的引用统一成 `../figures/…`（与图在桶根一致）；非 `legacy` 文档里的 **37 处图引用现已全部可达**（`legacy/` 里 14 处是历史路径，故意未改） |
| 跨桶图 | `07-v6/docs/v6与免责1s3s对比.md` 要用 `06` 桶的 `H6_1s_overview_all.png` ⇒ 按「共享即复制」规则复制一份到 `07-v6/figures/` |
| **动态 import 依赖**（归档时的漏检） | `z7_v7_test.py` / `z3_v6.py` 用 `importlib.import_module("r_fastphase"/"z2_v5")` 动态取模块，静态 import 闭包看不到 ⇒ `03-v4`、`04-v5`、`09-v7` 各补齐 `z2_v5.py` / `r_fastphase.py`（含 `.pyc`）。`verify.py` 已加入动态 import 检查 |
| `results/cache/` 层级 | `paper_v6/results/cache/*.npz` 在归档时被拍平到 `results/`，而 `pv_common.cache_path()` 按 `results/cache/` 读 ⇒ 13 个 npz 已归位（否则论文的复算与出图全部读不到缓存） |
| `09-v7` 桶 | 归档后被删除（非本会话动作），已按 `temp/archived/_v41_organize/scripts.backup/` 恢复 4 个脚本、并从 `03-v4`/`01-v3-baseline` 取回共享模块，`z7_v7_test.py` 实跑重建 `results/varying_steps_v7.csv`（3436 B，与归档前同尺寸，结论「v7 与 v3 逐帧相同」复现）。**若删除是有意为之，说一声即可再次移除** |

### 9.3 修订后的校验结果

- **锚点**：203 个脚本全部正确（`OUT`=桶、`RES`=桶/results、`TEMP`=`temp/`、`FIG`=该桶图的真实目录）；
- **同目录 import**（含动态 `import_module`）：**0 处失败**；
- **文档图引用**：37 处全部可达；
- **实跑验证**：`pf5/pf6/pf7` 重新出图 + `09-v7/z7_v7_test.py` 重建结果，均通过（顺带确认了缓存读得到、FIG 写对地方）；
- 计算文件数：**808** = 790 内容 + 12 `MANIFEST.md` + 本 `README.md` + `legacy/relocation/` 5 个工具
  （`legacy/docs/` 两份历史 README 计入内容）。
- 脚本运行产生的 `__pycache__` 已清理（可随时再生）；`scripts/` 下平铺的 `.pyc` 是归档原有文件，属惰性缓存，可忽略或删除。

### 9.4 同日的 temp/ 整理（本归档的分工边界）

同日按用户要求整理了 `temp/`：**只保留 `v4.1flash/`（本目录）与原始数据**（`右拇指指尖/数据1~3`、
`左拇指指尖/`、`四指指尖/`、`变化负载/`），其余 540 个文件全部移入 **`temp/archived/`**
（`GLM53/`、`glm53_cpp_harness/`、`Claude/`、`creep_check/`、`v41_review/`、`audit_crops/`、
`右拇指指尖/` 里拆出的分析部分、以及归档工作目录 `_v41_organize/`）。

- 与本目录的关系：本目录的脚本读数据集走 `TEMP`= `temp/`，**数据没动，因此照常可跑**（已复跑校验）；
- 连带更新：`src/domain/{drift,creep}/*.h` 的「另见」注释、`project_summary/modules/display.force-adc-pressure/`
  的指路与 `content_sha`（`check_stale` 为 VALID）、本目录 `02-dsp-route` / `03-v4` 两篇文档里的 `temp/GLM53/…` 路径；
- 归档内脚本的路径注意事项见 **`temp/archived/README.md`**（约 93 个历史脚本的数据集根锚点少算一级，属留存的历史原样）。

### 9.5 新增 `13-v6-assessment`：plan v1.0 的 v6 评估答复（2026-09-18）

用户把需求写在 `temp/v4.1flash/plan/v1.0/需求文档.md`（6 组问题），答复与全部复算产物按本归档约定新建
**`13-v6-assessment/`** 桶（docs / scripts / results / figures），并在 `plan/v1.0/答复-索引.md` 留了索引。
本桶**只做只读分析**：跑 `04-v5`/`05-v5.1`/`07-v6` 三个桶里的算法原型 + `temp/` 原始数据，
未改 `src/`、未构建、未跑测试。

四条主要结论（详细数字见答复正文）：
1. **1~2 s 稳定必须主动处理快相**：1 s 时原始/现役 v5.1 比真值低 6.7~8.1%，而快相 10%→90% 中位要 3.1 s（最慢 4.5 s），"等"不可能达标；
2. **快相形状"同传感器可复现、跨传感器不通用"**：组内 3 次重复差 4.5~6.3 个百分点，跨传感器差 15 个百分点；0.2 s 之前不可用（极差 0.70）；形状与幅度基本无关；
3. **现役 ROM 比实测中位慢 3~5 个百分点** ⇒ Â 过充中位 +5.0%（最坏 +12.8%）；同一 ROM 套 restep 则欠充 −6.3%（最坏 −93%）—— 用户说的"向上/向下过充"是同一根因的两个方向；
4. **滤波只能换时间、不能消除系统性过充**：输出端 EMA τ=0.3 s 是净收益（过充 9.1%→5.1%、T_stable 2.72→1.89 s），但要把过充压到 <3% 就得 τ≥1 s、T_stable 变 4.2 s（超预算）。

另外两条与验收口径有关的提醒：`T_stable` 对"以哪个幅度为参考"极敏感（同一份 v6 数据 1.41 s vs 2.72 s），
验收前必须先钉死口径；多次变载工况下全程 `T_stable` 不可测，应改为事件级指标。

---

## 10. 结构重组（2026-09-19）

用户把本目录重组成「历史归档 / 当前工作 / 原始入口」三层，本节记录重组内容、影响与遗留项。

### 10.1 做了什么

| 动作 | 内容 |
|---|---|
| **历史层** | 原 `progress/` 下的 **13 个版本桶**（`01-v3-baseline` … `13-v6-assessment`）整体移入新建的 **`archived/`**，桶号与桶内结构一字未动 |
| **当前层** | 新建 **`currentworking/`** 作为"当前正在维护的算法版本"桶（本轮首次填入 3 个文档，见 §2.1） |
| **原始入口** | `legacy/` 原地保留（两个历史 README + 首次归档的 5 个工具脚本） |
| 本 README | §1~§6 的可操作路径补上 `archived/` 前缀；新增本层说明与 §2.1；§7/§9 加历史留痕声明（路径不改） |

### 10.2 为什么影响面很小

- **桶内**的相对路径（`docs/`、`figures/`、`results/`、`scripts/`，以及文档里的 `figures/…`、`../figures/…`）
  **全部不变** —— 13 个桶是一起搬的，彼此仍是同层邻居 ⇒ 桶间相对引用（如 `07-v6` 引用 `06` 的图）仍然可达；
- 脚本的 `HERE / OUT / RES / FIG` 锚点（首次归档时统一修正过）**全部基于 `__file__` 的桶内层级，不受影响**；
- 数据集在 `temp/` 下没动，`TEMP` 锚点照常可解析。

### 10.3 ⚠️ 重组后的遗留项（**尚未处理**，需要时按此清单收尾）

| # | 项 | 现象 | 影响 |
|---|---|---|---|
| 1 | **`archived/08-v6.1/` 是空桶** | 0 文件、无 `MANIFEST.md`（重组前即为空） | v6.1 尖峰修复的文档/脚本/结果目前在本目录内**找不到**；`archived/09-v7/MANIFEST.md` 等处仍引用它 |
| 2 | **`archived/13-v6-assessment/` 无 `MANIFEST.md`** | 该桶 105 个文件，缺桶级清单（其余 12 个桶都有） | 复现入口需直接读桶内文件；不影响已有文档引用 |
| 3 | **`archived/04-v5/cpp_check/CMakeLists.txt` 的 `ROOT` 锚点少算一级** | `set(ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")`；重组后应为 `../../../..` | 只影响该 C++ 自检工程重新 configure（README §5 已提示改用 `-B <新目录>`） |
| 4 | **`archived/13-v6-assessment/scripts/check_refs.py` 的 `PLAN` 锚点少算一级** | `PLAN = join(dirname(dirname(BUCKET)), "plan", "v1.0")`；重组后需再多上溯一级 | 该引用复核脚本会报"缺失"；不影响归档内容 |
| 5 | **若干"活引用"仍指向重组前的桶路径** | 例：`project_summary/modules/display.force-adc-pressure/analysis.md`（已于本次修正）、`toolbox/数据解析工具/dptool/figure_export.py` 的注释 | 属**指路性文字**，不影响运行；建议后续顺手改成 `progress/archived/…` |

> 说明：`Document/ChangingLog/` 的历史条目、`legacy/` 里的历史 README **按项目既有约定保留当时的路径**，不追改。

### 10.4 重组后的文件数（本次实测）

| 层 | 文件数 |
|---|---|
| `archived/`（13 个桶） | **849** |
| `currentworking/` | **127**（2026-09-20 换代后实测：桶根报告 3 + `document/` 2 + `scripts/` 19 源文件（另 `build/` 6、`__pycache__/` 9）+ `results/` 7 + `figure/` 39 + `archived/creep-mem/` 42；2026-09-19 重组当时为 3、含 v6 交接包时为 4） |
| `legacy/` | **7** |
| `README.md`（本文件） | 1 |
| **合计** | **984**（重组当时 860；2026-09-19 交接包版 861） |

- 13 个桶中：**12 个有 `MANIFEST.md`**（缺 `13-v6-assessment`，见 §10.3）；
  单桶文件数 `01`=5、`02`=56、`03`=100、`04`=120、`05`=31、`06`=60、`07`=78、**`08`=0**、
  `09`=13、`10`=35、`11`=66、`12`=180、`13`=105。
- 与 §7/§9 的历史数字（799 / 808）不可直接相减：那两次统计的口径是"归档内容 + 本 README + 工具脚本"，
  且 §9 之后又新增了 `13-v6-assessment` 的收口产物等内容文件。

---

## 11. 换代记录：`currentworking/` 换成 v3.4 观测器（2026-09-20）

> 本节按「★★ 规定」R8 记账：换代做了什么 / 为什么 / 影响面。上一节（§10）记的是 09-19 的三层重组。

### 11.1 做了什么

| 项 | 内容 |
|---|---|
| **当前一代换人** | `currentworking/` 由 **v6 交接包**（`00_交接说明.md` / `01_v6算法说明_当前实现.md` / `02_算法更新日志.md` / `MANIFEST.md`，参数集 `plan-v3.1 PCT-fix`）换成 **v3.4 在线双态蠕变观测器**（`observer-v3`，`src/domain/drift_v6/creep_observer.{h,cpp}`，`algorithm.id = drift_v34_creep_observer`） |
| **新桶内容** | 桶根 3 份报告（`算法说明_v3.4观测器.md` / `规律辨识报告.md` / `观测器原型报告.md`）+ `document/` 2 份正式文档（`在线双态蠕变观测器补偿算法.md` 主文档、`观测器嵌入式移植文档.md`）+ `scripts/` `results/` `figure/`（该代自包含） |
| **上一代归档** | 事件式「蠕变记忆」（creep-mem）**整包**退到 `currentworking/archived/creep-mem/`（说明 + 问题清单 + 验收报告 + v6 源码快照 + 脚本 + 结果 + 图），只读 |
| **v6 交接包退出** | 上述 4 份 v6 文档**已不在本仓树内**（全树检索无此四份）；v6 档口径改由 `archived/07-v6/`、`../plan/{v2.0,v3.0,v3.1}` 承担，按 R1 属**历史** |
| **本次附带修正** | `document/在线双态蠕变观测器补偿算法.md` 里 **14 处图引用**由 `figure/…` 改为 `../figure/…`（文档移入 `document/` 后原写法解析到 `document/figure/`，不可达；出图脚本的 `OUT = scripts/../figure/…` 与正文出图说明里的 `../figure/…` 都证明桶根 `figure/` 才是对的目标），已逐条核对可达 |
| **规则化** | 本文新增「★★ 规定」（R1~R8），把"`currentworking/` 下的才是当前正在工作的算法"从描述升级为强制约定；§2.1、§3 阅读路径、§10.4 文件数同步更新 |

### 11.2 影响面

- **桶内其它层不变**：`archived/`（13 桶 / 849 文件）与 `legacy/`（7）一字未动；
  13 个历史桶的相对引用、脚本锚点、`TEMP` 数据集锚点均不受影响。
- **`src/` 未改**：本次只动 `progress/` 下的文档（README + 一篇主文档的图引用），
  未碰产品源码、未构建、未跑测试 ⇒ `project_summary/` 的 `content_sha` 无需重算。
- **失效的自检脚本**：`../plan/v3.0/scripts/v30_check_docsync.py` 仍指向退出本目录的 v6 交接包
  （`FILES` 4 份 + 旧 `BASE`），已失效，待下次改动时重写或改用本桶脚本。
- **外部的死指针（待收尾）**：`<v2.7 根>/README.md` §2 目录地图与 §3 阅读路径仍把
  `currentworking/00_交接说明.md` 当作第一入口，§6.3 仍记"该文档第 0 节自相矛盾" —— 该文件已不存在，
  这几处应改指本文件「★★ 规定」+ §2.1。本次未改那份 README。
