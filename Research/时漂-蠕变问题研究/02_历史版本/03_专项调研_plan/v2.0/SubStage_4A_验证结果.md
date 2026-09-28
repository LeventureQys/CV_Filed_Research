# SubStage 4A 验证结果 · 协调器参数记忆（C-5）

> 本文件是 SubStage 4A 的实测证据留档。一次性工程 `temp/v4.1flash/plan/v2.0/__scratch_4a/`
> 已两次验证后均删除（其内容与命令完整记录在本文件中，可复现）。
>
> **阅读顺序**：§1–§5 是**修复前**的首轮验证（6 项 FAIL，据此定位出"第三条清空路径"= 注入点位于
> `Process` 首次 `ResetFor`/`Reset` 之前）；**§6「修复后复验」是最终结论（22/22 PASS，failures = 0）**。

## 1. 改动落地（首轮：严格照任务书 §3）

| 文件 | 位置 | 内容 |
|---|---|---|
| `src/domain/drift_v6/drift_v6_compensator.h` | L361–L370 | `DriftV6CompensationCoordinator` 新增 `SetParams` / `params()`（public）与 `DriftV6Compensator::Params params_;`（private） |
| `src/domain/drift_v6/drift_v6_compensator.cpp` | L824–L827 | `SetParams`：存入 `params_` 并遍历 `targets_` 对每个 `comp.SetParams(p)` |
| `src/domain/drift_v6/drift_v6_compensator.cpp` | L835–L848 | `Process`：`const bool is_new = (targets_.find(key) == targets_.end());` + `if (is_new \|\| entry.signature != signature) { Reset(); SetParams(params_); signature = signature; }` |
| `src/domain/drift_v6/drift_v6_compensator.cpp` | L850 | `ResetAll()` 保持 `targets_.clear();`（不清 `params_`） |
| `src/domain/drift_v6/drift_v6_compensator.cpp` | L829–L833 | `SetEnabled` 语义不变 |

首轮验证时 `DriftV6Compensator` 零改动（`Reset()` 里 `params_ = Params();` 原样保留）。验证期间临时插入的
`printf` 探针已全部移除，`git hash-object` 与探针插入前逐字节一致（`42f4868944bee01d3926a578e26730e3d0f7cd13`）。
> 该行描述的是**修复前**状态；修复后 `DriftV6Compensator::Reset()` 已由 MainAgent 修改（删去 `params_ = Params();`），见 §6.1。

## 2. 验收标准 A（编译）

```
cmake --build build-app --config Debug --target TactileSense -- /m:4
```
结果：**exit 0**（未重新 configure，复用既有 `build-app`）。新增无编译 warning；
输出中仅有既有的 `DeployTactileSense` 环境类 warning（`VCINSTALLDIR is not set`、`CRT DLL not found: vcruntime140d.dll` 等），与本次改动无关。
`drift_v6_compensator.cpp` 被重新编译并链接。

## 3. 验收标准 B/C/D（一次性小工程实测）

编译命令（`__scratch_4a/` 内，只编 `src/domain/drift_v6/drift_v6_compensator.cpp`，不参与主工程）：

```
cl /nologo /std:c++17 /EHsc /utf-8 /W3 /O2 /D_ALLOW_KEYWORD_MACROS /Fe:test_4a.exe /Fo:obj\ ^
   /I<repo>\src /I<repo>\thirdparty\eigen-5.0.0 main.cpp ^
   <repo>\src\domain\drift_v6\drift_v6_compensator.cpp
```

测试序列：4 通道 600 帧 @100 Hz（0–1.19 s 空载 → 1.2–3.59 s 阶跃爬升 → 3.6–5.99 s 回落）。
探针方式：`#define private public` 只读 `DriftV6Compensator::params_` 与 `DriftV6CompensationCoordinator::targets_`
（需 `_ALLOW_KEYWORD_MACROS`，MSVC STL 默认禁止宏化 `private`）。

实测输出：

```
[B-1] coordinator: SetParams(legacy=true, g_neg_floor=-0.5) BEFORE any frame
    coord.params(): legacy=1 g_neg_floor=-0.500
    inner comp after full sequence: legacy=0 g_neg_floor=-0.050
    [FAIL] B-1 legacy_fixes_enabled still in effect inside compensator
    [FAIL] B-1 g_neg_floor still -0.5 (not -0.05)

[B-1b] mechanism: which frame wipes the injected params?
    after frame#0:                    legacy=0 floor=-0.050
    after frame#0 + SetParams + #1:   legacy=1 floor=-0.500
    [PASS] B-1b injection AFTER the entry's first frame sticks (wipe is caused by frame#0 only)

[B-2] control: direct DriftV6Compensator SetParams BEFORE first frame
    after SetParams, before frames: legacy=1 g_neg_floor=-0.500
    after first frame:              legacy=0 g_neg_floor=-0.050
    [PASS] B-2 control: direct injection IS wiped back to defaults by first frame

[C] coordinator: params survive a signature change
    signature: 'sigA' -> 'sigB'; legacy=0 g_neg_floor=-0.050
    [PASS] C signature really changed (reset path taken)
    [FAIL] C legacy_fixes_enabled still true after signature change
    [FAIL] C g_neg_floor still -0.5 after signature change

[C-2] newly created second target
    target_count=2; t2 inner: legacy=0 g_neg_floor=-0.050
    [PASS] C-2 target_count == 2
    [FAIL] C-2 new target t2 got the remembered non-default params

[C-3] SetParams() propagates to an already existing target
    existing target after SetParams: clamp_alpha=0.770
    [PASS] C-3 SetParams() reaches already-existing targets_ entries

[D] SetEnabled(false) -> SetEnabled(true)
    after SetEnabled(false): targets=0 legacy=1 g_neg_floor=-0.500
    [PASS] D params() still injected after SetEnabled(false)
    [PASS] D params() still injected after SetEnabled(true)
    [PASS] D ResetAll cleared targets_ while keeping params_ 
    [FAIL] D re-created target after re-enable got injected params again

[E] regression: no injection -> coordinator keeps defaults
    defaults: legacy=0 g_neg_floor=-0.050
    [PASS] E legacy_fixes_enabled default false
    [PASS] E g_neg_floor default -0.05

==== failures = 6 ====
```

| 标准 | 结果 |
|---|---|
| A 编译 | **PASS**（exit 0） |
| B-1 协调器注入在首段序列中生效 | **FAIL** |
| B-2 直接注入被首帧清空（对照，既有契约） | **PASS**（符合预期，未改） |
| C signature 变化后参数保持 | **FAIL** |
| C-2 新建第二条目注入 | **FAIL** |
| D `params()` 记忆在 `SetEnabled` 前后保持 | **PASS** |
| D 重新启用后重建条目收到注入 | **FAIL** |
| E 未注入时保持默认 | **PASS** |

## 4. 根因（实测定位）

任务书 §1/§3 的模型假设协调器侧有**两条**清空路径（新建条目、signature 变化），
但实测证明还有**第三条，且在注入点之后**：

1. `DriftV6CompensationCoordinator::Process` 首次见到某 key：
   `is_new == true` → `entry.comp.Reset()`（`params_` 清默认）→ `entry.comp.SetParams(params_)`（注入**成功**）。
2. 紧接着的 `entry.comp.Process(...)` 内部第一句 `if (n_ != n) ResetFor(n);` 成立
   （`Reset()` 刚把 `n_` 置 0，而 `n` 为等价通道数），`ResetFor` → `Reset()` → `params_ = Params();`
   → **刚注入的参数被立刻清回默认值**。

实测探针（临时 `printf`，已移除）逐帧证据：

```
[PROBE] Process key=t1 is_new=1 sigdup=0 coord.legacy=1
[PROBE] DriftV6Compensator::Reset() this=... params_before.legacy=0     <- 协调器的 Reset()
[PROBE]   after inject: inner.legacy=1 inner.floor=-0.500               <- 注入确实成功
[PROBE]   pre-Process inner.legacy=1
[PROBE] DriftV6Compensator::Reset() this=... params_before.legacy=1     <- 首帧内部 ResetFor -> Reset()
[PROBE]   post-Process inner.legacy=0 inner.floor=-0.050                <- 被清回默认
```

即：在协调器侧"先 `Reset()`、再 `SetParams()`、再 `Process()`"的顺序下，
**同一个 `Process` 调用里**第 1 步的注入必然被第 3 步的首帧 `ResetFor` 抹掉；
而 `is_new` 只在首次为真、`signature` 之后也不再变化，所以**该条目此后永久停留在默认参数**。
`B-1b` 进一步证明抹除只发生在"该条目第一帧"，第二帧起注入即可保持
（这也解释了为什么 SubStage 2B 之前 A/B 脚本必须"先喂一帧再 SetParams"）。

结论：**按任务书 §3 逐字实现后，验收标准 B-1/C/C-2/D（重新启用后重建条目）仍不通过。**
协调器侧不动 `DriftV6Compensator`、且不感知补偿器内部 `n_` 的情况下，无法满足 B-1。

## 5. 未决问题（需 MainAgent 裁决）

### Q1（阻塞）B-1/C 在"协调器侧唯一改法"下不可能通过

- **冲突**：任务书 §3.2 要求注入必须在 `entry.comp.Process(...)` **之前**（参考实现 L95–L101）；
  而验收 B-1 要求注入在**首帧之后**仍然生效。`ResetFor` 由 `Process` 内部触发，协调器无法在其后插入注入点。
- **候选方案（均超出任务书 §3 的"逐字照此"范围，故未擅自采用）**：
  - **方案 1（推荐，仍不碰 `DriftV6Compensator`，仅协调器内加 1 个 `Entry` 字段）**：
    `Entry` 加 `bool pending_inject = false;`；`Process` 中若 `is_new || signature 变化`
    则先 `Reset()/SetParams()` 并置 `pending_inject = true`；每次 `Process` 之后
    若 `pending_inject` 为真则再 `SetParams(params_)` 并清标志。
    语义：参数在**该条目第二帧前**生效（第一帧仍按补偿器既有契约用默认值）。
    风险：`Process` 返回后再次 `SetParams` 等价于"重置后重新注入"，对算法无副作用（`SetParams` 只改 `params_`）。
  - **方案 2**：给 `DriftV6Compensator` 增加"是否已初始化"查询（如 `bool initialized() const`），
    协调器在 `Process` 后按需补注入。**触碰 `DriftV6Compensator`，与任务书 §6.1 冲突。**
  - **方案 3**：把"首帧 `ResetFor` 不清 `params_`"作为 §4.1 契约的修订（改 `Reset()`）。
    **直接违反任务书 §6.1 与设计文档 §4.1.1 C-5 的 A 侧契约，属重大路线变更。**
  - **方案 4**：接受现状，把 C-5 的"参数生效时机"明确为"该条目第二帧起"，
    并同步修订验收标准 B-1/C。**降低验收强度，需设计侧确认。**
- **本 SubAgent 处置**：未擅自扩大改动（任务书 §6.5「不确定就停下并列为未决问题」），
  代码保持 §3 逐字实现，构建 exit 0，B-2 对照与 D/E 通过项已实测。

### Q2（非阻塞）`Entry` 新增字段属于契约变更

方案 1 需要在 `Entry` 内加字段，属任务书 §3.1 未列出的 `.h` 改动；若采用需更新任务书与设计文档 §4.1.1。

## 6. 临时工程处置（首轮）

`temp/v4.1flash/plan/v2.0/__scratch_4a/`（`main.cpp` + `bak_compensator.cpp` + `obj/` + `test_4a.exe`）
**已删除**。理由：任务书 §5 明确要求"完成后必须删除"；该工程不参与主工程构建，
其源码与全部实测输出已完整留档于本文件，可按上文命令随时重建。

---

# 修复后复验（最终结论）

**修复内容（MainAgent 裁决采纳，SubAgent 未改 `src/` 任何文件）**：
`src/domain/drift_v6/drift_v6_compensator.cpp` 的 `Reset()`（L65–L105）中**删除** `params_ = Params();`，
并在原位置（现 L99–L105）补注释说明新语义：

> **`Reset()` 只重置补偿状态，不动运行期参数**；参数只由 `Params` 默认成员初始化器与显式 `SetParams()` 决定。

即采纳"方案 3 的修正版"：直接消除"静默抹参"缺陷本身，而非把参数生效时机推迟一帧；
不引入新状态字段，补偿状态重置语义（`n_`/`state_`/`A_`/`g_`/环缓等）完全不变。

## 6.1 最终代码状态

| 文件 | 位置 | 内容 |
|---|---|---|
| `src/domain/drift_v6/drift_v6_compensator.cpp` | L65–L105 | `Reset()` 中 `params_ = Params();` **已删除**（原 L99），替换为 L99–L105 的语义注释 |
| `src/domain/drift_v6/drift_v6_compensator.h` | L361–L370 | `DriftV6CompensationCoordinator::SetParams` / `params()`（public）+ `DriftV6Compensator::Params params_;`（private） |
| `src/domain/drift_v6/drift_v6_compensator.cpp` | L824–L827 | `SetParams`：存入 `params_` 并遍历 `targets_` 下发 |
| `src/domain/drift_v6/drift_v6_compensator.cpp` | L835–L848 | `Process`：`is_new \|\| signature 变化` → `Reset(); SetParams(params_);` |
| `src/domain/drift_v6/drift_v6_compensator.cpp` | L850 | `ResetAll()` 只 `targets_.clear();`（不清 `params_`） |

复验未修改 `src/` 下任何文件；临时探针（`#define private public` 只读）全部在一次性工程内，`src/` 无 `printf`/探针残留。

## 6.2 A（编译，复验）

```
cmake --build build-app --config Debug --target TactileSense -- /m:4
  drift_v6_compensator.cpp
  TactileSense.vcxproj -> D:\workshop\...\build-app\Debug\TactileSense.exe
exit=0
```
**PASS**：exit 0；`drift_v6_compensator.cpp` 重新编译；无 `error` / 无 `warning C`（仅既有 DeployTactileSense 环境类 warning）。未重新 configure，未新建/删除构建树。

## 6.3 复验命令

重建 `temp/v4.1flash/plan/v2.0/__scratch_4a/`（内容 = 下表 main.cpp），编译：

```
cl /nologo /std:c++17 /EHsc /utf-8 /W3 /O2 /D_ALLOW_KEYWORD_MACROS /Fe:test_4a.exe /Fo:obj\ ^
   /I<repo>\src /I<repo>\thirdparty\eigen-5.0.0 main.cpp ^
   <repo>\src\domain\drift_v6\drift_v6_compensator.cpp
```
（`obj\` 目录需先创建；`_ALLOW_KEYWORD_MACROS` 必加，MSVC STL 默认禁止宏化 `private`。）
序列与首轮一致：4 通道 600 帧 @100 Hz（空载 → 阶跃爬升 → 回落）。

## 6.4 复验实测输出（22/22 PASS）

```
[B-1] coordinator: SetParams(legacy=true, g_neg_floor=-0.5) BEFORE any frame
    coord.params(): legacy=1 g_neg_floor=-0.500
    inner comp after full sequence: legacy=1 g_neg_floor=-0.500
    [PASS] B-1 legacy_fixes_enabled still in effect inside compensator
    [PASS] B-1 g_neg_floor still -0.5 (not -0.05)

[B-2] control: direct DriftV6Compensator SetParams BEFORE first frame (NEW behaviour)
    after SetParams, before frames: legacy=1 g_neg_floor=-0.500
    after first frame:              legacy=1 g_neg_floor=-0.500
    [PASS] B-2 NEW: direct injection now SURVIVES the first frame (was wiped before the fix)
    after full sequence:            legacy=1 g_neg_floor=-0.500
    [PASS] B-2 NEW: direct injection survives the whole sequence

[B-3] new contract: Reset() no longer restores default params
    after explicit Reset(): legacy=1 floor=-0.500 alpha=0.770
    [PASS] B-3 Reset() keeps runtime params (compensation state still reset)
    [PASS] B-3 Reset() still clears compensation state (n_ == 0)

[C] coordinator: params survive a signature change
    signature: 'sigA' -> 'sigB'; legacy=1 g_neg_floor=-0.500
    [PASS] C signature really changed (reset path taken)
    [PASS] C legacy_fixes_enabled still true after signature change
    [PASS] C g_neg_floor still -0.5 after signature change

[C-2] a newly created second target
    target_count=2; t2 inner: legacy=1 g_neg_floor=-0.500
    [PASS] C-2 target_count == 2
    [PASS] C-2 new target t2 got the remembered non-default params

[C-3] SetParams() propagates to an already existing target
    existing target after SetParams: clamp_alpha=0.770 g_neg_floor=0.430
    [PASS] C-3 SetParams() reaches already-existing targets_ entries

[D] SetEnabled(false) -> SetEnabled(true)
    after SetEnabled(false): targets=0 legacy=1 g_neg_floor=-0.500
    [PASS] D params() still injected after SetEnabled(false)
    [PASS] D params() still injected after SetEnabled(true)
    [PASS] D ResetAll cleared targets_ while keeping params_
    re-created target inner: legacy=1 g_neg_floor=-0.500
    [PASS] D re-created target after re-enable got injected params again

[E] regression: no injection -> defaults preserved end-to-end
    coordinator defaults: legacy=0 floor=-0.050 alpha=0.005
    [PASS] E legacy_fixes_enabled default false
    [PASS] E g_neg_floor default -0.05
    [PASS] E clamp_alpha default 0.005
    standalone comp after frame#0: legacy=0 floor=-0.050 alpha=0.005
    [PASS] E standalone compensator without SetParams keeps defaults (constructor defaults)

[F-1] edge: no SetParams keeps defaults across Reset()
    after Process + Reset: legacy=0 floor=-0.050 alpha=0.005
    [PASS] F-1 no injection -> defaults still intact across Process + Reset

[F-2] edge: zero-length values_io returns early (n <= 0)
    after Process(empty): legacy=1 floor=-0.500 n_=0
    [PASS] F-2 empty frame keeps injected params
    [PASS] F-2 empty frame did not initialise state (early return)

[F-3] ResetAll() then a new Process still injects the remembered params
    after ResetAll + new Process: targets=1 legacy=1 floor=-0.500
    [PASS] F-3 ResetAll() keeps params_ and re-injects on the next new target

==== failures = 0 ====
exit=0
```

## 6.5 逐条对照（复验 vs 首轮）

| # | 检查项 | 首轮（修复前） | 复验（修复后） |
|---|---|---|---|
| A | 编译 exit 0 | PASS | **PASS** |
| B-1 | 协调器注入后直接喂整段序列，条目内保持注入值 | FAIL | **PASS** |
| B-2 | 直接注入 + 首帧 | FAIL（被清回默认，既有契约） | **PASS（行为已改变，见下）** |
| B-3 | 显式 `Reset()` 后参数保持 | 未测 | **PASS（新契约）** |
| C | signature 变化后保持 | FAIL | **PASS** |
| C-2 | 新建第二个条目注入 | FAIL | **PASS** |
| C-3 | `SetParams` 对已存在条目生效 | PASS | **PASS** |
| D | `SetEnabled` 前后 `params()` 记忆 + 重建条目注入 | 部分 FAIL | **PASS** |
| E | 未注入时默认值 `legacy=0 / floor=-0.05 / alpha=0.005` | PASS | **PASS** |
| F-1/2/3 | 默认值跨 `Reset()` 稳定 / 空帧早退 / `ResetAll` 后重建注入 | 未测 | **PASS** |
| — | 断言失败计数 | 6 | **0（22/22 PASS）** |

## 6.6 B-2 行为差异（如实报告，属契约变更）

- **修复前**：`DriftV6Compensator` 先 `SetParams` 再喂首帧 ⇒ 首帧 `ResetFor → Reset → params_ = Params()`，
  参数被静默清回默认（`legacy=0 / floor=-0.050`）。这是原 §4.1 契约，也是 C-5 缺陷本体。
- **修复后**：**同一个实验现在保持注入值**（首帧后仍 `legacy=1 / floor=-0.500`，整段 600 帧后亦然）。
- **差异定性**：这是**有意的契约变更**，影响所有直接使用 `DriftV6Compensator` 的调用方
  （此前"先喂一帧再 `SetParams`"的绕行写法不再必要，但依然有效）。需要同步修订设计文档 §4.1.1 C-5
  与任何依赖"`Reset()` 会还原参数默认值"的说明/脚本（如 C-5 条目里"A/B 脚本必须先喂一帧"的约束）。
- **新契约明确表述**：`Reset()` 只重置**补偿状态**，不再触碰**运行期参数**；参数只由
  `Params` 默认成员初始化器与显式 `SetParams()` 决定。补偿状态重置语义未变（`B-3` 实测 `n_ == 0`）。

## 6.7 复验的临时工程处置

`temp/v4.1flash/plan/v2.0/__scratch_4a/` **已再次删除**（`Test-Path` = False，
且 `temp/v4.1flash/plan/v2.0` 下已无 `test_4a*` / `obj` 残留）。
理由同首轮：任务书 §5 要求"完成后必须删除"；该工程不参与主工程，源码与全部输出已留档于本节，可按 §6.3 重建。

## 6.8 复验遗留

- 无阻塞项：任务书 §5 的 A/B/C/D 全部 PASS，新增 B-2 行为差异与 E/F 边界均已实测。
- 待 MainAgent 收口（不属 SubAgent 范围）：按 §6.6 修订设计文档 §4.1.1 C-5 表述；
  如需在 `data_handler.cpp` 两条链路调用 `coordinator.SetParams(...)`（本版未提供注入 UI/配置），
  该改动超出本 SubStage 边界（任务书 §2/§6.2 明确禁止）。

