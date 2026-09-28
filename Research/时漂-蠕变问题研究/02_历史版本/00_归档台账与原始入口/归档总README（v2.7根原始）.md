# v2.7 - 抗蠕变补偿算法（研究归档）

> **归档时间**：2026-09-20 ｜ **来源**：仓库根 `temp/` **整树平移**（保持原目录结构、未改内容）｜ **本目录即 v2.7 版本目录**。
> 平移同时把软件版本推进为 `DEV v2.7.0-preview1`（`CMakeLists.txt` 的 `TACTILESENSE_DISPLAY_VERSION` + `CHANGELOG.md`）。
>
> 本目录是**研究 / 实测归档**，沿用 `temp/` 原有的研究结构，**不套用** `v2.0`~`v2.6` 那种「需求文档 / 设计文档 / 问题清单 / 验收文档」条目结构；
> 若日后正式启动 v2.7 开发，建议在本目录下另开 `v2.7.0`、`v2.7.1`… 条目目录（仿 `v2.6` 目录），本归档作为其算法与实测依据。

---

## 1. 这批研究是什么

显示层「时漂 / 零漂 / 蠕变」补偿算法的**完整研究线**：`GLM53` 混合方案 → v5.1（现役）→ `v6` 形状约束反演 + 滑行器 → `v3.4` 在线双态蠕变观测器，
外加 2026-09-19 的实机录制实测（算法前 / 算法结果 / 原始 ADC 三流）与两篇论文的图件审查留痕。

菜单「设备」下当前四项互斥算法开关与归档子树的对应关系：

| 菜单开关 | C++ 实现 | 归档入口 | `session.json` 的 `algorithm.id` / `param_set` |
|---|---|---|---|
| 时漂/零漂补偿（无责 1s / 3s / 5s） | `src/domain/drift/` | `archived/GLM53/`（前身分析）+ `v4.1flash/progress/archived/{04-v5,05-v5.1,06-v5.1-exempt-1s-3s-5s}` | `drift_fast_1s` / `3s` / `5s` |
| 变点感知对数蠕变补偿 | `src/domain/creep/` | `archived/Claude/`（v6c 设计与原型）+ `archived/creep_check/`（独立复核） | `creep_log_change_point` |
| 时漂/零漂补偿（v6 快速稳定） | `src/domain/drift_v6/drift_v6_compensator` | `v4.1flash/progress/archived/07-v6` + `v4.1flash/plan/{v1.0,v2.0,v3.0,v3.1}`（v3.4 的 creep-mem 也是本档的一个 `Params` 开关，见下条） | `drift_v6_fast_settle` / `plan-v3.1 PCT-fix` |
| 时漂/零漂补偿（v3.4 观测器） | `src/domain/drift_v6/creep_observer` | `v4.1flash/plan/v3.4` | `drift_v34_creep_observer` / `plan-v3.4 observer-v3` |

> 以上 id / param_set 由 `DataHandler::CurrentDisplayAlgorithm()` 写入录制 manifest。
> 注意 v6 档只有在 `Params::mem_tau_s > 0` 时才记 `plan-v3.4 creep-mem`，而菜单切档会把 `mem_tau_s` 置 0 ⇒ **正常运行时 v6 档落盘的是 `plan-v3.1 PCT-fix`**。

---

## 2. 目录地图

```
v2.7 - 抗蠕变补偿算法/                  ← 原 temp/（2885 文件 / 685 MB）
├── README.md                          ← 本文件：归档索引
├── 抗时漂 v3.4 算法代码说明与移植指南.md  ← ★ 移植主文档（2026-09-21 新增，非 temp 原件）
├── 抗时漂 v3.4 抗下漂变体设计（草案D1）.md ← 抗下漂候选改动（2026-09-21 新增，草案/未实装）
├── _const_load_sampling.csv           ← 恒载采样表（原 temp/ 顶层）
├── v4.1flash/                         ← 【算法】抗蠕变补偿算法的版本化研究归档（2278 文件 / 350 MB）
│   ├── progress/                      ← 版本桶归档，**入口：progress/README.md（版本谱系 + 阅读顺序）**
│   │   ├── README.md                  ← 13 个版本桶的谱系表、演进主线图、阅读路径、搬迁台账
│   │   ├── archived/01-v3-baseline … 13-v6-assessment   ← 全部历史版本桶
│   │   ├── currentworking/            ← 当前维护版本的实现细则；**入口：00_交接说明.md**
│   │   └── legacy/                    ← 重组前的 README 与搬迁脚本（历史留痕）
│   └── plan/                          ← 第二轮专项调研与候选实现（v1.0 … v3.6-v41flash）
│       ├── v1.0/                      ← T1~T9 任务制调研（稳定时间口径、三阶段特征、快相形态、v6 评估与裁决、现场回归）
│       ├── v2.0/                      ← C 单侧限幅 + F5 g 负向下界（已落地进 v6）
│       ├── v3.0/                      ← 慢相逐通道蠕变跟踪 PCT（定点修复「稳定工况残漂」）
│       ├── v3.1/                      ← PCT 扣除连续性修复（定点修复「卸载后显示飘高」）
│       ├── v3.2/ + v3.2-gpt5.6/       ← 交接塌陷与 gain 候选（另有独立交叉验证包）
│       ├── v3.3/                      ← 分析留痕
│       ├── v3.4/                      ← ★ 在线双态蠕变观测器（creep-mem）与嵌入式移植文档（observer-v3 口径，**不含后续 K6/K7/K9**）
│       └── v3.6-v41flash/             ← 会话蠕变比候选（v3.4 的并列机制，交叉验证用）
├── 算法数据&原始数据/                  ← 【实测】2026-09-19 实机录制（54 文件 / 198 MB）
│   ├── working/                       ← 当前工况组（零基线-反复增减同一负载 / 零基线-剧烈变化负载情况 / 零基线-同一荷载测试）
│   ├── archived/                      ← 9 类工况（持续恒定负载、各种奇怪工况、零负载-随机切换工况 …）
│   └── working/说明文档/               ← 《Venture's anti creep算法测试情况说明》docx（含原始备份）
├── 原始数据only/                       ← 【实测】纯原始录制（29 文件 / 77 MB）
│   └── 变化负载/ · 四指指尖/ · 左拇指指尖/ · 右拇指指尖/  各 `数据1`~`数据3`
└── archived/                          ← 【历史】2026-09-18 归档的早期研究（523 文件 / 60 MB）
    ├── README.md                      ← 该批归档的清单、路径注意与复原方法（**先读这个**）
    ├── GLM53/                         ← 混合蠕变补偿（v5 前身）的算法分析与三传感器实测 + `dsp.md`（被否决的替代路线）
    ├── glm53_cpp_harness/             ← GLM53 的 C++↔Python 逐帧对拍工程
    ├── Claude/                        ← 变点感知对数蠕变补偿（v6c）的设计、原型与交接说明
    ├── creep_check/                   ← v6c 估计器与状态轨迹的独立复核脚本
    ├── 右拇指指尖-分析/                ← 该数据集的非数据部分（DSv4 / DSv4.1flash 三批漂移分析）
    ├── v41_review/ · audit_crops/     ← 论文交付图与另一批图件的像素级复核裁剪
    ├── _v41_organize/                 ← 归档台账与工具（分类规则、搬迁执行器、锚点修正、校验器、MANIFEST 生成器、`plan.json`）
    └── _archive_log*.ps1              ← 「完整更新日志」100 KB 分档归档脚本（与 temp 分析无关，顺带归档）
```

**算法分流（2026-09-18 起）录制目录的结构**——`算法数据&原始数据` 下的会话目录：

| 文件 | 内容 |
|---|---|
| `device_001_seg000.csv` | 主数据文件 = **算法结果**（processed_display） |
| `device_001_pre_seg0.csv` | **算法前**读数（没开算法时本该显示的逐通道值，与主文件同列结构/同时间戳，可逐帧对比） |
| `device_001_raw_seg000.csv` | **原始 ADC** 流（未补偿） |
| `session.json` | 会话 manifest（含 `algorithm{id,label,params}`、显示模式、设备信息） |

> `原始数据only/` 里是**分流功能之前**的旧录制，通常只有 `device_001_seg000.csv` + `session.json`
> （`右拇指指尖` 三组另带一份 `device_001_seg000_kalman_compensated.csv`——当年外部 Kalman 对照，非本算法产物）。
> 算法分流的实现与口径见 `project_summary/modules/data.record-replay/analysis.md` 与 `CHANGELOG.md` 的 v2.6.7-preview3 段。

---

## 3. 按问题找入口

| 想了解 | 去哪 |
|---|---|
| 算法版本如何演进（v3→v5.1→v6→v6.1，各版本一句话定位与关键数字） | `v4.1flash/progress/README.md` §1 谱系表 + 演进主线图 |
| **当前 C++ 里跑的算法究竟是什么** | `v4.1flash/progress/currentworking/00_交接说明.md`（全局地图）→ `01_v6算法说明_当前实现.md`（实现细则）→ `02_算法更新日志.md`（改动史与现场失效定因） |
| 菜单四项算法分别怎么来的 | 本文件 §1；细节见 `project_summary/modules/display.force-adc-pressure/analysis.md` 第 6 章「相关文档索引」 |
| 「保压时显示还在慢慢爬」的定因 | `v4.1flash/plan/v3.0/`（`分析报告_稳定工况残漂.md` / `问题清单.md` / `验收报告.md`） |
| 「卸载后显示飘高」的定因 | `v4.1flash/plan/v3.1/`（两条定因、五处修复、19 条录制 A/B） |
| 「卸载-重载丢基线」与 creep-mem 机制 | `v4.1flash/plan/v3.4/`（`算法说明_v3.4观测器.md` / `在线双态蠕变观测器补偿算法.md` / `观测器嵌入式移植文档.md`） |
| **把 v3.4 观测器移植到别的平台（代码梗概 + 注意事项 + 验收）** | 本目录 `抗时漂 v3.4 算法代码说明与移植指南.md`（★ 以当前 C++ 源码为权威口径：27 参数 / 每通道 11 状态 / 含 K7 旁路与 K9）；嵌入式的逐帧伪码与定点建议见 `v4.1flash/plan/v4/k9/嵌入式迁移说明.md` |
| **快速回撤后显示基线下漂，想改算法** | 本目录 `抗时漂 v3.4 抗下漂变体设计（草案D1）.md`（三处正交改动 + 最小 diff + 验收集；**草案，未实装**） |
| 参数（κ / 交接时刻 / 撤销迟滞 / PCT / creep-mem）的裁决依据 | `v4.1flash/plan/v1.0/总报告.md`、`plan/v1.0/summary/修改建议总结.md` |
| 免责期该取 1 s / 3 s / 5 s | `v4.1flash/progress/archived/06-v5.1-exempt-1s-3s-5s/docs/免责期1s-3s-5s对比.md` |
| 某份录制的复算怎么做 | 先在 `算法数据&原始数据` / `原始数据only` 找到会话目录，再到对应 `plan/vX/scripts/`（多数脚本可一键复跑） |
| v6 规格与全部离线实测 | `v4.1flash/progress/archived/07-v6/docs/07-v6算法说明.md` + `docs/v6与免责1s3s对比.md` |
| v6c（对数蠕变）为什么这么设计 | `archived/Claude/01-研究过程记录.md` → `02-v6c算法说明.md` → `03-v6c实现交接说明-AgentA.md` |
| 被否决的路线有哪些、为什么否决 | `archived/GLM53/dsp.md`（逆滤波路线）+ `v4.1flash/progress/archived/02-dsp-route/docs/dsp方案评审报告.md` |
| 归档本身是怎么搬的、锚点怎么办 | `archived/README.md`（09-18 那批）+ `archived/_v41_organize/`（台账与工具） |

---

## 4. 入库范围（`.gitignore`）

本目录**体积 685 MB，其中约 597 MB 不入库**，只把**文档 / 脚本 / 图件**纳入版本管理：

| 类别 | 是否入库 | 规模 |
|---|---|---|
| `.md` / `.py` / `.json` / `.txt` / `.html` / `.docx` / `.cpp` / `.h` / `.ps1` / `.bat` | 入库 | 约 1702 个文件 / 87.5 MB（含 `.png` 61 MB） |
| 录制数据 `.csv` | **不入库** | 543 份 / 363 MB |
| 中间数值 `.npz` / `.npy` | **不入库** | 114 份 / 194 MB |
| 构建产物 `.obj` / `.exe` / `.pdb` / `.ilk` / `.log` / `__pycache__` | **不入库**（由仓库全局规则覆盖） | 约 411 份 / 40 MB |

> 未入库的文件**仍然躺在工作区磁盘上**、内容完整，只是不随仓库分发；需要入库时删掉 `.gitignore` 中对应的 `v2.7 - 抗蠕变补偿算法/**` 规则即可。
> 规则位置：`.gitignore` 末尾「v2.7 抗蠕变补偿算法研究归档」小节。

---

## 5. 位置变更对脚本的影响（重跑前必读）

整树相对仓库根**统一下移 3 级**：`temp/X` → `Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/X`。

- 归档内约 **92 个 `.py` 含字面 `temp/...` 路径**、约 **612 个 `.py` 含上溯锚点写法**（`os.path.dirname(...)` / `Path(__file__).parents[N]` / `../../`）。
  以脚本自身目录为基准找**同级文件**的锚点仍然有效（整树平移，相对关系不变）；**上溯到 `temp/` 当「数据集根」的那部分会少算 3 级**。
- 本次**只修「活引用」**：`src/`（7 个文件 16 处注释）、`tests/`（2 处）、`toolbox/数据解析工具/`（8 处）、`project_summary/`（模块文档与 4 个 SHA 刷新脚本）、`.gitignore`；顺带把其中若干**平移前就已指向错层级/已失效**的旧路径（`v4.1flash/Document/*`、`v4.1flash/scripts/*`、`progress/07-v6/*`、`算法数据&原始数据/working/…` 等）改指到真实位置。
  **归档内的历史脚本一律原样保留**（它们是研究留痕），要重跑请按 +3 级自行修正锚点。
- 批量修正可用既有工具：`archived/_v41_organize/patch_archived_paths.py`（先 dry-run 列出待修清单，再加 `--apply`）；
  相关注意事项见 `archived/README.md` §2。
- `Document/ChangingLog/` 里的历史条目（含 `project_summary/index.json` 的 `update_history[].reason`）**保留当时的 `temp/...` 写法**，属历史记录，未改写。

---

## 6. 已知缺口与不一致

1. **v6.1 的文档与图件未随归档保留**：`v4.1flash/progress/archived/08-v6.1/` 是空桶，
   原 `08-v6.1算法说明.md`、`L1_v61_spike_fix.png`、`L2_v61_metrics.png` 在归档中均不存在；
   幸存的只有原型与诊断脚本，位于 `archived/_v41_organize/scripts.backup/`（`glm53_v61.py` 与 `cu~di_v61_*.py` 一批）。
2. **一份录制目录不在归档内**：`20260919_141824_single_device_110871`（单次加载后恒定保压 393 s 的稳定工况录制）已不在工作区，
   只留下 `v4.1flash/plan/v3.0/results/v30_trace_20260919_141824_single_device_110871.{npy,cols}` 的复算轨迹。
3. **`currentworking/00_交接说明.md` 第 0 节内部不一致**：文档顶部「最后更新」行写「当前参数集 `plan-v3.4 creep-mem`」，§0「三十秒版」第 2 条又写「`plan-v3.1 PCT-fix`」。
   按 `src/data/data_handler.cpp::CurrentDisplayAlgorithm()` 的实际取值口径，菜单 v6 档（切档时把 `mem_tau_s` 置 0）落盘 `plan-v3.1 PCT-fix`，
   v3.4 观测器是**另一档**（`plan-v3.4 observer-v3`）⇒ **以 `src/` 与录制 `session.json` 的 `algorithm.params` 为准**。
4. **归档内脚本的路径锚点**：见 §5，未批量修正。

---

## 7. 维护约定

- 新增的本主题研究、原型与实测，**直接进本目录对应子树**（算法进 `v4.1flash/`，录制进 `算法数据&原始数据/` 或 `原始数据only/`），不要另起 `temp/` 之类的工作区。
- 往本目录**新增 `src/` 之外的文件不触发** `project_summary/` 缓存刷新；但若连带改了 `src/`、`CMakeLists.txt` 或 `CHANGELOG.md`，按 `AGENTS.md` 的规则刷新缓存并记入 `Document/ChangingLog/完整更新日志.md`。
- 本目录只做**归档**：算法结论以 `src/` 为准；缓存与文档冲突时改文档，不要改代码去迁就文档（沿用 `currentworking/00_交接说明.md` 的事实优先级）。
