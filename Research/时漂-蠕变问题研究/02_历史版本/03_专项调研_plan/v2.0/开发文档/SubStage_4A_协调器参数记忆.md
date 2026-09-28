# SubStage 4A · 协调器参数记忆（修 C-5：首帧静默清空 `params_`）

## SubStage: 4A
- **所属 Stage**：Stage 4（阶段三集成）
- **依赖前置**：SubStage 1A/1B/2A/2B（**需等待 2A 完成**，因为它也改同一个 `.h/.cpp`）
- **并行状态**：**需等待**（等 2A 交付后再开工；开工前先确认 `2A` 已落盘）
- **所属阶段**：阶段三 - 开发

> **你是 SubAgent**。开工前先完整阅读仓库根 `AGENTS.md`；不确定就停下报告 MainAgent。

---

## 1. 背景与目标（缺陷来源：SubStage 2B 实测；根因由本 SubStage 4A 定位）

`DriftV6Compensator::Process` 的第一句逻辑是 `if (n_ != n) ResetFor(n);`（首帧必然成立），
而 `ResetFor` → `Reset()` → 末尾原为 `params_ = Params();`。
⇒ **任何"首帧之前"调用的 `SetParams` 都会被静默还原为默认值**：
`legacy_fixes_enabled` 回到 `false`、`g_neg_floor` 回到 `-0.05`。

**本 SubStage 的第一版实现（"协调器叠加注入"）实测仍然失败**：协调器
`entry.comp.Reset(); entry.comp.SetParams(params_);` 之后，**同一个 `Process` 调用内**的
首帧 `ResetFor` 会再次 `Reset()` 并抹掉刚注入的参数；而该路径此后不再触发 ⇒
**条目永久停留在默认参数**。探针实证（见 `SubStage_4A_验证结果.md`）。

**最终处置（MainAgent 直接改一行）**：`Reset()` 中**删除** `params_ = Params();`，新语义为
**"`Reset()` 只重置补偿状态，不动运行期参数"**；参数只由 `Params` 的默认成员初始化器与
显式 `SetParams()` 决定。协调器的记忆式注入（本 SubStage 的改动）随之真正生效。

**本 SubStage 的目标**：协调器**记住**参数，并在"新建条目 / signature 变化重置"两条路径上重新注入。
**本 SubStage 不改任何补偿算法、不改参数默认值。**

---

## 2. 当前代码状态

| 位置 | 内容 |
|---|---|
| `src/domain/drift_v6/drift_v6_compensator.h` | `class DriftV6Compensator`（已含 `Params` / `params()` / `SetParams` / `ApplyOutputLimit` / `valley_*` / 只读访问器）+ 文件末尾 `class DriftV6CompensationCoordinator`（约 L350~L372）：`SetEnabled/Process/ResetAll/target_count` 与 `bool enabled_`、`std::map<std::string, Entry> targets_`、`struct Entry { DriftV6Compensator comp; std::string signature; }` |
| `src/domain/drift_v6/drift_v6_compensator.cpp` | 文件末尾为协调器实现：`SetEnabled`（`enabled_ = on; ResetAll();`）、`Process`（`auto& entry = targets_[key]; if (entry.signature != signature) { entry.comp.Reset(); entry.signature = signature; } entry.comp.Process(...)`）、`ResetAll`（`targets_.clear();`） |
| `src/data/data_handler.cpp` | v6 挂载：串口链路约 L749~L759、无线链路约 L1652~L1662，均调 `drift_v6_comp_.Process(key, sig, ts, data)`；**不要在 `data_handler.cpp` 里加参数注入调用**（本版不提供注入 UI/配置） |

---

## 3. 必须实现的接口契约

### 3.1 `.h`：`DriftV6CompensationCoordinator` 增加

```cpp
public:
    // 记忆式运行期参数: 记在协调器自身, 并在"新建条目 / signature 变化重置"后重新注入。
    // 背景见 temp/v4.1flash/plan/v2.0/设计文档.md §4.1.1 C-5:
    //   DriftV6Compensator 首帧会 ResetFor -> Reset -> params_ = Params(),
    //   因此"首帧之前"注入的参数会被静默清空。
    void SetParams(const DriftV6Compensator::Params& p);
    const DriftV6Compensator::Params& params() const { return params_; }

private:
    DriftV6Compensator::Params params_;
```

### 3.2 `.cpp`：协调器实现改动（**逐条照此**）

```cpp
void DriftV6CompensationCoordinator::SetParams(const DriftV6Compensator::Params& p) {
    params_ = p;
    for (auto& kv : targets_) kv.second.comp.SetParams(p);   // 已存在的条目立即生效
}

void DriftV6CompensationCoordinator::SetEnabled(bool on) {
    if (enabled_ == on) return;
    enabled_ = on;
    ResetAll();          // 保持原语义: 开关切换后从全新状态开始
}

void DriftV6CompensationCoordinator::Process(const std::string& key,
                                             const std::string& signature,
                                             double timestamp_s,
                                             Eigen::VectorXd& values_io) {
    if (!enabled_) return;
    Entry& entry = targets_[key];
    if (entry.signature != signature) {
        entry.comp.Reset();
        entry.comp.SetParams(params_);      // ← 新增: 重置后重新注入记住的参数
        entry.signature = signature;
    }
    entry.comp.Process(timestamp_s, values_io);
}

void DriftV6CompensationCoordinator::ResetAll() {
    targets_.clear();                       // params_ 不清: 它是"记忆", 与补偿状态无关
}
```

**关键点**：
- `SetEnabled`/`ResetAll` **不得**清 `params_`（它记在协调器上，不是补偿器状态）。
- **新建条目**（`targets_[key]` 首次插入）时，该 `DriftV6Compensator` 用的是它的**构造默认值**；
  若 `params_` 已被注入过非默认值，必须在**该条目的首帧之前**注入。实现方式：在 `Process` 里
  判断"本次是否新建了条目"，若是则 `entry.comp.SetParams(params_)`（**必须在 `entry.comp.Process(...)` 之前**）。
  参考实现（可自行等价改写，但语义必须一致）：
  ```cpp
  const bool is_new = (targets_.find(key) == targets_.end());
  Entry& entry = targets_[key];
  if (is_new || entry.signature != signature) {
      entry.comp.Reset();
      entry.comp.SetParams(params_);
      entry.signature = signature;
  }
  ```
- 不改 `DriftV6Compensator` 本身的任何一行。

---

## 4. 输入输出

- **输入**：`Params`（由调用方注入）。
- **输出**：无返回值（`SetParams`/`SetEnabled`/`ResetAll`）；`Process` 就地改写 `values_io`。
- **错误处理**：无异常；`Process` 在 `!enabled_` 时直接返回（原语义）。

---

## 5. 验收标准

| # | 标准 | 判定 |
|---|---|---|
| A | `cmake --build build-app --config Debug --target TactileSense -- /m:4` **exit 0**，无新增 warning | 命令输出（**不要**重新 configure） |
| B | **C-5 复现测试**：`coord.SetParams({legacy_fixes_enabled=true, g_neg_floor=-0.5})` 后**不喂帧**，直接走协调器喂一段"空载→阶跃→回落"序列 ⇒ 条目内 `comp.params()` **必须保持注入值** | 一次性临时工程 |
| C | **签名变化后参数保持**：注入非默认参数 → 喂若干帧 → 换一个 `signature` 再喂 ⇒ 参数仍为注入值；**新建第二个 key 条目**同样保持 | 同上 |
| D | `SetEnabled(false)` → `SetEnabled(true)`（会 `ResetAll` 重建条目）后，条目内参数仍为注入值 | 同上 |
| E | **新契约确认**：`Reset()` 之后 `params()` **不再**回到默认值（这是 §4.1.1 C-5 的新语义，必须在报告中明确写出）；未注入时默认值仍为 `legacy=false / g_neg_floor=-0.05 / clamp_alpha=0.005` | 同上 |

测试用一次性小工程（`temp/v4.1flash/plan/v2.0/__scratch_4a/`，不参与主工程；只编 `src/domain/drift_v6/drift_v6_compensator.cpp`，include `src` 与 `thirdparty/eigen-5.0.0`，`/EHsc /utf-8`）。
**完成后必须删除 `__scratch_4a/` 并在报告写明理由。**

---

## 6. 禁止事项

1. **禁止**改动 `DriftV6Compensator` 的**算法逻辑**（检测器/状态机/滑行器/慢相/限幅/F2/F3/F5/参数默认值一律不动）。
   > ⚠️ **例外（已由 MainAgent 直接落地，你不必也不得再改）**：`Reset()` 中原有的
   > `params_ = Params();` **已被删除**（见 §1 末段与设计文档 §4.1.1 C-5）——
   > 这是本 SubStage 定位到的根因的修复，已由 MainAgent 改完。你只需**验证**它的效果。
2. **禁止**改动 `src/domain/drift_v6/` 以外任何文件（**特别禁止**动 `data_handler.cpp`：本版不提供参数注入 UI/配置）。
3. **禁止**新建构建树、全量构建、`ALL_BUILD`、构建或运行任何 `test_*` 目标、全套件 ctest。
4. **禁止**并发构建：先 `job_list` 查重，有作业在跑就等待。
5. 不确定就停下并列为「未决问题」。

---

## 7. 构建要求

- 复用既有 `build-app`：`cmake --build build-app --config Debug --target TactileSense -- /m:4`（**禁止**重新 configure）。
- 长构建用 `run_in_background` + 60~120 s 轮询。

---

## 8. 完成报告要求

① 改动文件与行号；② A~D 的实测命令与输出；③ B 的对照结果（直接注入被清空 vs 协调器注入保持）；④ 临时工程是否已删除；⑤ 不确定项。
