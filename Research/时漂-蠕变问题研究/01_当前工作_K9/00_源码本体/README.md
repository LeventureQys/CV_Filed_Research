# 00_源码本体 —— 源码快照与路由口径

> **这是快照，不是源码本体。** 唯一权威始终是仓库：
> `D:\workshop\Processing\multi-device-cascade-host-cpp\src\...`
> 快照只用于离线阅读、对拍与交接；**与仓库冲突时以仓库为准**。

---

## 1. 快照清单与来源

| 文件 | 来源（仓库路径） | 快照时间 | 源文件 mtime |
|---|---|---|---|
| `creep_observer.h` | `src/domain/drift_v6/creep_observer.h` | 2026-09-27 | 2026-09-26 15:15:36 |
| `creep_observer.cpp` | `src/domain/drift_v6/creep_observer.cpp` | 2026-09-27 | 2026-09-26 15:02:08 |
| `creep_observer_params_dialog.h` | `src/ui/dialogs/creep_observer_params_dialog.h` | 2026-09-27 | 2026-09-26 15:15:28 |
| `creep_observer_params_dialog.cpp` | `src/ui/dialogs/creep_observer_params_dialog.cpp` | 2026-09-27 | 2026-09-26 15:48:01 |
| `参数与寄存器表.md` | —— 本目录编写（依据上面 4 份） | 2026-09-27 | —— |

### 相关但**未**纳入快照的关键文件（改代码时要一起看）

| 文件 | 作用 |
|---|---|
| `src/data/data_handler.cpp::CurrentDisplayAlgorithm()` | 菜单档 → `algorithm.id` / `param_set` 路由；27 个参数随录制落盘 |
| `src/ui/dialogs/compensation_algorithm_dialog.{h,cpp}` | 四项互斥算法开关与 v3.4 档入口 |
| `src/ui/menubar.cpp` | 菜单挂载点与 tooltip |

---

## 2. 快照的 git 状态（**重要**）

仓库当前在分支 **`LTS-Preview`**（最后提交 `fe4736ca`，2026-09-24 17:57）。
本快照对应的是**工作树（含未提交改动）**，不是某个提交：

```
$ git -C ... status --short -- src/domain/drift_v6 src/ui/dialogs
 M src/domain/drift_v6/creep_observer.h
 M src/ui/dialogs/compensation_algorithm_dialog.cpp
 M src/ui/dialogs/compensation_algorithm_dialog.h
 M src/ui/dialogs/creep_observer_params_dialog.cpp
 M src/ui/dialogs/creep_observer_params_dialog.h
```

**未提交的实质改动**：

1. `Params` 的 **8 个默认值**被改（见 `参数与寄存器表.md` §2 的粗体项）；
2. 参数对话框由 **7 项 » 8 项**（新增 `r_slow_max`），寄存器由 **200~207 » 200~208**；
3. 文件头注释更新（记录 2026-09-23 / 09-24 / 09-26 三轮现场调参结论）。

> 因此：**已提交的 HEAD 与现役 exe 的口径不一致**。要核对"设备里现在到底是多少"，
> 请读设备寄存器（或录制 `session.json` 的 `algorithm.params`），不要只读代码。

---

## 3. 类结构速览

```
namespace drift_v6 {

class CreepObserverCompensator {          // 单目标（单传感器）实例
  struct Params { ... };                  // 27 项
  void Process(double timestamp_s, Eigen::VectorXd& values_io);   // 就地：入 = 原始 ADC，出 = 显示值
  void Reset();                           // 全状态重置
  const Params& params() const;
  void SetParams(const Params& p);
 private:
  void ResetFor(int n);
  ... 11 个逐通道状态 + 4 个全局
};

class CreepObserverCoordinator {          // 多目标：按 key 隔离实例
  void SetEnabled(bool on);
  void Process(const std::string& key, const std::string& signature,
               double timestamp_s, Eigen::VectorXd& values_io);
  void ResetAll();
  void SetParams(const CreepObserverCompensator::Params& p);
 private:
  struct Entry { CreepObserverCompensator comp; std::string signature; };
  std::map<std::string, Entry> targets_;  // signature（显示值域签名）变化 ⇒ 自动重置该目标
};

}  // namespace drift_v6
```

**契约要点**：

- **输入输出同址**：`values_io` 传入为原始读数，返回即显示值（`v − applied`）；算法不改调用方的其它状态。
- **时间基**：`timestamp_s` 单调递增（秒）；帧内 `dt` 就地积分，**不要求固定采样率**，`dt > 0.1 s` 被钳到 0.1 s。
- **通道数变化 / signature 变化 ⇒ 全状态重置**（对应嵌入式"传感器配置变更时调 `Reset()`"）。
- 单线程逐帧调用，无内部锁；多设备各自独立实例，天然可分核。

---

## 4. 怎么重新生成快照（改了源码之后）

```powershell
$SRC = 'D:\workshop\Processing\multi-device-cascade-host-cpp\src'
$DST = 'D:\workshop\文档\v2.7 - 抗蠕变补偿算法\01_当前工作_K9\00_源码本体'
Copy-Item "$SRC\domain\drift_v6\creep_observer.h"   "$DST\" -Force
Copy-Item "$SRC\domain\drift_v6\creep_observer.cpp" "$DST\" -Force
Copy-Item "$SRC\ui\dialogs\creep_observer_params_dialog.h"   "$DST\" -Force
Copy-Item "$SRC\ui\dialogs\creep_observer_params_dialog.cpp" "$DST\" -Force
```

生成后**同步更新**本文件 §1 的 mtime 表、`参数与寄存器表.md` 的默认值列，以及
`../README.md` §3.2/§5 中引用的默认值。
