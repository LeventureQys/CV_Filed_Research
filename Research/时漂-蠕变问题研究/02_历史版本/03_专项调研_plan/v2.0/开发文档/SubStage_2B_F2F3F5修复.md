# SubStage 2B · F2/F3/F5 修复（事件内卸载出口 + 重锚归零 + 慢相 g 保护）

## SubStage: 2B
- **所属 Stage**：Stage 2
- **依赖前置**：SubStage 1A（**需等待**：`valley_now_` / `valley_run_` / `params_` 已落地）
- **并行状态**：可与 SubStage 2A **并行**（2A 只改 `Process` 三个出口与新增 `ApplyOutputLimit`；本 SubStage 只改 `RunEvent`/`Reanchor`/`SlowStep`）
- **所属阶段**：阶段三 - 开发

> **你是 SubAgent**。开工前必须先完整阅读仓库根 `AGENTS.md`（含「Agent 命令执行预算」「验证与构建总则」「SubAgent 编译测试分工」），并遵守本文全部约束。

---

## 1. 背景与目标

实机录制实测三条失效（证据见 `temp/v4.1flash/plan/v2.0/results/design_failure_census.txt`、
`分析报告_A1.md`、`分析报告_C1.md`）：

1. **`idle_now` 在事件态内只真 1 帧、慢相态内真 0 帧**（空载态内真 3675/3693 帧）⇒
   `RunEvent` 唯一卸载出口 `if (tau > kUnloadFastS && idle_now)`（`.cpp` L493）在本工况**结构性不可达**，
   事件期内发生的卸载被漏掉（实测事件态内 `inc < 0.5·inc_max` 的帧有 **131 帧**）。
2. **`Reanchor` 的"保持扣除连续"**（`.cpp` L657-659 注释、L671-676 实现）会**原样继承错误扣除**，
   而已回到谷底时继承的前提（"实测电平里含真实受载"）不成立。
3. **慢相 `g` 长期为负**（slow 态 87.5% 帧 `g<0`、极值 −0.499）⇒ `ded = γ·A·g < 0` ⇒
   `out = v − ded = v + |ded|`，显示被**抬高**。

本 SubStage 落地用户裁决的 **F2 + F3 + F5** 三条修复。
**重要**：这三条在本录制 + `α=0.005` 限幅组合下**改进不可测**（`design_ab_verify.txt`：F2 触发 0~1 次、
F3 触发 3 次，指标与"只加 C"的臂逐位相同）⇒ 它们是**结构正确性修复**，
**验收只要求"行为正确 + 不劣化"，不要求指标改善**；命中次数必须写入诊断计数器。

---

## 2. 当前代码状态（行号）

| 位置 | 内容 |
|---|---|
| `.cpp` L441~L448 | `RunEvent` 开头：`tau = ts − ev_.t0; inc = total − ev_.base; inc_s = 0.10 s 窗均值 − base` |
| `.cpp` L455~L476 | C6 试探撤销（τ < `kRevokeS`=0.40，连续 `kRevokeHoldN`=3 帧） |
| `.cpp` L479~L490 | 减重对称撤销 |
| `.cpp` L492~L497 | **C4 完全卸载出口（唯一）**：`if (tau > kUnloadFastS && idle_now) { ToIdle(); *out = v; return true; }` |
| `.cpp` L653~L677 | `Reanchor`：L657-659 注释说明"保持扣除连续"的理由；L660 清 `hold_comp_`；L661-666 算 `A_new`；L671-676 反算 `g` |
| `.cpp` L679~L687 | `ToIdle()`：清 `ev_`、`state_=Idle`、`hold_comp_`、`loaded_`、`g_=0`，置 `idle_since_` 与 `ev_block_until_` |
| `.cpp` L704~L739 | `SlowStep`：L713 `g_ += (dt/kTauG)·(g_raw − g_)`；L715~L729 积分 `γ`；L730-738 逐通道输出 |

---

## 3. 接口契约（逐字实现，参数名不得改）

### 3.1 F2 · 事件期卸载出口（`.cpp` L492 的 C4 出口**之前**插入）

```cpp
    // ── F2: 事件期内的卸载出口(不依赖 idle_now) ──
    // 本录制的实测结论: idle_now 在事件态内只真 1 帧(见 design_failure_census.txt),
    // 因此上面那条 C4 出口在本工况结构性不可达; valley_now_ 只依赖近 40 s 的电平形状,
    // 与"空载绝对值"无关, 故在事件期内也能正确认出"已回到谷底"。
    // 默认关闭(params_.legacy_fixes_enabled = false): 14 份数据的回归显示本项收益未证实,
    // 且在「变化负载/切换负载-快相无责」上方向不一致(见设计文档 §3.4)。
    if (params_.legacy_fixes_enabled && tau > params_.event_valley_min_s && valley_now_) {
        ++params_.n_valley_exit;
        ToIdle();
        *out = v;
        return true;
    }
```

- **插入位置必须严格在 L492 之前**（即 `if (tau > kUnloadFastS && idle_now)` 那一行**上方**），
  这样两条出口取"或"关系，且**不改动**原出口。
- `tau` 已在 L444 算好，可直接用。
- **`params_.legacy_fixes_enabled` 默认 `false`**（见设计文档 §3.4）：它默认不生效，
  但**开关与实现都必须交付**。验收时通过 `SetParams` 打开它来验证功能。

### 3.2 F3 · `Reanchor` 回到谷底即归零（`.cpp` L653 `Reanchor` 函数体**第一句**）

```cpp
void DriftV6Compensator::Reanchor(double ts, const Eigen::VectorXd& v) {
    // ── F3: 已回到谷底 ⇒ 不做"保持扣除连续", 直接归零 ──
    // Reanchor 原有语义是"把当前扣除量代代继承"(L657-659 的理由), 前提是"实测电平里含真实受载";
    // 若已回到谷底, 该前提不成立, 继续继承会把错误扣除固化。
    // 默认关闭(与 F2 共用 params_.legacy_fixes_enabled)。
    if (params_.legacy_fixes_enabled
        && valley_now_ && valley_run_ >= params_.reanchor_valley_s) {
        ++params_.n_reanchor_idle;
        ToIdle();
        return;
    }
    hold_comp_.resize(0);
    ...（其余原样保留）
```

- 返回类型是 `void`，用**提前 return** 即可，不需要改签名。
- 注意：`ToIdle()` 内部使用 `last_ts_`，在 `Reanchor` 内调用是安全的（`last_ts_` 已在 `Process` 开头更新）。

### 3.3 F5 · 慢相 `g` 的负向下界 + 谷底归零（`.cpp` `SlowStep` 内，紧随 L713 的 `g_` 更新之后）

```cpp
    if (g_ < params_.g_neg_floor) {          // 负向下界: g<0 会让显示被抬高(out = v − γ·A·g)
        g_ = params_.g_neg_floor;
        ++params_.n_g_floor;
    }
    if (valley_now_ && valley_run_ >= params_.g_valley_reset_s && g_ < 0.0) {
        g_ = 0.0;                            // 已回到谷底 ⇒ 慢相残差不再有意义
        ++params_.n_g_valley_reset;
    }
```

**位置**：必须在 L713 `g_ += (dt / kTauG) * (g_raw - g_);` **之后**、L715 `if (g_ > kGEnable)` **之前**。
理由：下界必须在"本帧更新之后"施加才能约束本帧的输出；而 `g2_acc_/g_rel_acc_` 的积分只依赖 `g_ > kGEnable`，
放在其前不会改变积分条件（`g_` 只可能被抬高到 `kGNegFloor < 0 < kGEnable`）。

**不得**修改 `kTauG`、`kCreepLoFrac/kCreepHiFrac`、`kGammaMin/kGammaMax`、`Capped`。

---

## 4. 输入输出

- **F2**：输入 `tau` / `valley_now_`；输出"本帧直通 + 状态回 Idle"。
- **F3**：输入 `valley_now_` / `valley_run_`；输出"提前 `ToIdle()`"或"原行为"。
- **F5**：输入 `g_` / `valley_now_` / `valley_run_`；输出被约束的 `g_`。
- **错误处理**：全部为内部状态操作，无异常、无错误码。当 `valley_now_` 恒假（数据不足或工况不匹配）时，
  三条修复**自动退化为现役行为**（这是刻意的保守设计，见设计文档 §7 风险 R3）。

---

## 5. 验收标准

| # | 标准 | 判定 |
|---|---|---|
| A | `cmake --build build-app --config Debug --target TactileSense -- /m:4` **exit 0** | 命令输出 |
| B | **不劣化**：在三段夹具上，改动后的事件序列（`shape_hits()` 与直通帧数）与 base 臂一致或更优；不得出现"永不交接"（夹具跑完全段后 `in_event() == false`） | 单元测试 + 一次性脚本 |
| C | **F2 可观测（须先开开关）**：`SetParams` 设 `legacy_fixes_enabled = true` 后，在"空载→阶跃→回落保压"合成序列上 `n_valley_exit > 0`；默认（`false`）时 `n_valley_exit == 0` | 一次性脚本 |
| D | **F3 可观测（须先开开关）**：同上开关打开后，`seg_c_cycle` 上 `n_reanchor_idle ≥ 1`；默认时 `== 0` | 一次性脚本 |
| E | **F5 可观测**：`params().g_neg_floor = -0.05` 下，全程 `g_` 的最小值 ≥ `-0.05 - 1e-9`（用 `slow_residual_g()`；该 getter 由 SubStage 3A 添加，本 SubStage 可先用私有成员临时打印或用一次性脚本的等价实现） | 一次性脚本 |
| F | **零回归**：默认参数（`legacy_fixes_enabled=false`、`clamp_alpha=0.005`）下，既有 14 份数据的时漂残余与台阶捕获比与"只加 C + F5"一致 | 由 MainAgent 用 `design_full_regress.py` 复算；SubAgent 提供可复现命令 |

### 自测参考脚本（SubAgent 必做 B/C/D/E）

用 `tests/fixtures/drift_v6/` 的三段夹具，逐帧喂入并在结束时打印：
`shape_hits()`、`in_slow()`、四个计数器、`g` 的最小值。
（`n_valley_exit` 等计数器在 `Params` 里，`params()` 是 public 只读 ⇒ 直接读即可。）

---

## 6. 禁止事项

1. **禁止**改动 `Process`（L240~L404）、`RunEvent` 的 L441~L491 与 L498~L607、`Handoff`、`SlowStep` 除 §3.3 之外的部分。
2. **禁止**改动 `idle_now` 的两条判据（L317-319）、`min_ts_`/`level_ref_` 语义、`kKappaOnset`、`kHoMinS`、停滞判据、形状 ROM。
3. **禁止**改动限幅（C 路）相关代码：那是 SubStage 2A 的范围。
4. **禁止**修改 `src/` 下除 `drift_v6_compensator.{h,cpp}` 之外的任何文件。
5. **禁止**新建构建树 / 无 `--target` 构建 / `ALL_BUILD` / 全套件 ctest。
6. **禁止**同时构建主程序（与 MainAgent 或 2A 冲突）：先 `job_list` 检查。
7. 不确定就停并报告。

---

## 7. 构建要求

- 主程序：`cmake --build build-app --config Debug --target TactileSense -- /m:4`（**禁止**重新 configure）。
- 测试：`cmake --build build-tests-all --config Debug --target test_drift_v6_compensator -- /m:4`
  → `ctest --test-dir build-tests-all -C Debug -R "^test_drift_v6_compensator\." --output-on-failure`。
- 长构建用 `run_in_background` + 60~120 s 轮询。

---

## 8. 完成报告要求

① 三处改动的文件与行号；② A~E 的实测命令与输出；③ **四个计数器的实测数值**（F2/F3/F5 各触发多少次）；
④ 明确声明"本录制上改进不可测"是否被实测证实（给出对照数字）；
⑤ 是否有行为超出预期；⑥ 不确定项。**禁止**只写"编译通过"。
