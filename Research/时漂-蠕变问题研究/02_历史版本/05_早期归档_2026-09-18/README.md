# temp/archived —— 临时工作区归档（2026-09-18）

> **位置变更（2026-09-20）**：连同整个 `temp/`，本目录已整树平移到
> `Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/archived/` —— 相对仓库根**下移 3 级**，内容未改。
> 本文标题与正文里的 `temp/...` 是**当时**的写法：把 `temp/` 换成 `Document/Update/Dev-Version/v2.7 - 抗蠕变补偿算法/` 就是当前位置；
> 脚本锚点的 +3 级修正、入库范围（录制 CSV 与 npz/npy 不进 git）与新归档索引，见上一层 `../README.md` §4/§5。

> `temp/` 现在只保留两样东西：
> **`v4.1flash/`**（抗蠕变补偿算法的版本化归档，入口见 `../v4.1flash/progress/README.md`）
> 与 **原始数据**（`右拇指指尖/数据1~3`、`左拇指指尖/数据1~3`、`四指指尖/数据1~3`、`变化负载/`）。
> 其余内容全部移入本目录，**整目录平移、未改内容**，原来的相对关系保持不变。

当前规模：本目录 **547 个文件**；`temp/` 顶层只剩 `v4.1flash/`、4 个数据集目录和本目录。

---

## 1. 目录说明

| 目录 / 文件 | 内容 | 与现有工作的关系 |
|---|---|---|
| `GLM53/` | 混合蠕变补偿（现役算法的前身）的算法分析与实测：`时漂零漂算法分析报告.md`、`算法说明_混合蠕变补偿.md`、`变化负载分析与v2方案.md`、`新数据验证报告.md`、**`dsp.md`（被评审的替代方案指南）** + `scripts/ results/ figures/` | `src/domain/drift/` 的来源；`dsp.md` 正是 `v4.1flash/progress/02-dsp-route` 的评审对象 |
| `glm53_cpp_harness/` | GLM53 的 C++↔Python 对拍工程（`CMakeLists.txt`、`build/`、`compare*.py`、`cpp_*.csv`） | v5 落地前的逐帧对拍工具 |
| `Claude/` | 「变点感知对数蠕变补偿（v6c）」的设计与原型：`01-研究过程记录.md`、`02-v6c算法说明.md`、`03-v6c实现交接说明-AgentA.md`、`glm53_v6c*.py`、`_v6c_*.py` + `figures/ results/` | `src/domain/creep/` 的来源 |
| `creep_check/` | v6c 估计器与状态轨迹的独立复核脚本（`check_estimator.py`、`trace_state.py`） | 对应 `src/domain/creep/` |
| `v41_review/` | 论文（一）交付图的复核裁剪（25 张，按面板/边缘/文字区） | 与 `v4.1flash/progress/12-fig-audit` 同类，属另一轮 |
| `audit_crops/` | 另一批图件审查裁剪（25 张） | 同上 |
| `右拇指指尖-分析/` | 从 `temp/右拇指指尖/` 里拆出的**非数据部分**：`漂移分析与算法对比报告.md`、`scripts/`、`results/`、`figures/`、`DSv4/`（docx+html+md+figures）、`DSv4.1flash/`（figures+results+scripts） | 该数据集的三批漂移分析；**数据本身留在 `temp/右拇指指尖/数据1~3`** |
| `_v41_organize/` | v4.1flash 归档的台账与工具：分类规则 `classify.py`、搬迁执行器 `execute.py`、锚点修正表 `patch_paths.py`、校验器 `verify.py`、MANIFEST 生成器 `make_manifests.py`、本次搬迁脚本 `_archive_temp.py`，以及归档后的三个审计/修正脚本 `audit_archived_paths.py`（锚点是否可达）、`patch_archived_paths.py`（批量补一级，先 dry-run）、`audit_dataset_roots.py`（列出数据集根失准的脚本）；`scripts.backup/` 是**锚点修正前**的 151 个脚本原件（回退点）；`plan.json` 是搬迁作业台账 | 归档 v4.1flash 时使用；本次整理 `temp/` 也复用了这套工具 |
| `_archive_log*.ps1`（4 个） | 「完整更新日志」按 100 KB 分档归档的脚本（操作 `Document/ChangingLog/`） | 与 temp 分析无关，顺带归档；用绝对路径，移动后照常可用 |

---

## 2. 搬进 archived/ 后的路径注意

1. **约 93 个脚本的「数据集根」锚点会少算一级。**
   它们当年按 `temp/<目录>/` 的深度写死了到 `temp/` 的上溯层数（典型写法 `TEMP = os.path.dirname(HERE)`、
   `BASE = os.path.dirname(...×3...)`）。现在整体深了一层，这些锚点会解析成 `temp/archived/...`。
   **要重跑就把相应锚点再上溯一级，或直接改成指向 `temp/` 的绝对路径。**
   本次只对 **14 个文件**做了自动修正：`GLM53/scripts/*` 7 个 + `_v41_organize/*` 工具 7 个
   （工具必须能跑，故一并修好）；其余保持原样以留存历史。
   需要批量补一级时跑 `_v41_organize/patch_archived_paths.py`（先 dry-run 列出待修清单，再加 `--apply`）。
2. **`Claude/*.py` 里的 `SCRIPTS = HERE/../v4.1flash/scripts`**：v4.1flash 的脚本在上一轮已重组为
   `v4.1flash/progress/<版本桶>/scripts/`，这类引用要按目标算法改指到具体桶（例如 `.../progress/04-v5/scripts`）。
3. **活引用已同步更新**（这些已不再指向旧路径）：
   - `src/domain/drift/drift_compensator.h`、`src/domain/creep/change_point_creep_compensator.h` 的「另见」注释；
   - `project_summary/modules/display.force-adc-pressure/analysis.md`（同目录 `analysis.json` 的 `content_sha`
     已重算，`project_summary/tools/check_stale.ps1 -Module display.force-adc-pressure` 为 **VALID**）；
   - `v4.1flash/progress/02-dsp-route/docs/dsp方案评审报告.md`、`v4.1flash/progress/03-v4/docs/快相与慢相分离分析.md`。
   - 未改的是**历史记录**：`Document/ChangingLog/` 的条目与 `v4.1flash/progress/legacy/` 的历史 README
     保留当时的路径（那正是它们要记录的事实）。
4. **两处原本就悬空的指针顺带修正**：注释与缓存里提到的
   `Claude/Document/变点感知对数蠕变补偿算法详细设计.md` 与 `Claude/s4_final.py` 在档案里并不存在，
   已分别改指实际文档 `Claude/02-v6c算法说明.md` 与原型 `Claude/glm53_v6c.py`。
5. 本目录内 `.py` 的字节码缓存（`__pycache__`）随目录一起搬来，属可再生产物，可随时删除。

---

## 3. 复原

整体平移、未改内容，要把某项放回 `temp/` 直接移回同名位置即可；
搬迁清单与规则记录在 `_v41_organize/_archive_temp.py`（`--apply` 前会先 dry-run 列出 17 项 / 540 个文件）。
