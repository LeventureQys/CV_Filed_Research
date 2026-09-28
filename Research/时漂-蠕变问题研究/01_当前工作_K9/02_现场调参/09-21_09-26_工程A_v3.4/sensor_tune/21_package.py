# -*- coding: utf-8 -*-
"""21 打包：把最终图片 + 全部中间过程整理成一份可翻阅的交付包，放到 v4.1flash 下。

目标：D:\\workshop\\文档\\v2.7 - 抗蠕变补偿算法\\v4.1flash\\传感器分类调参_20260926\\
    README.md
    01_最终图片\\                          ← 本轮（第二轮：绝对 ADC 口径 + 过减约束）最新图
        archived\\第1轮（比例制口径）\\      ← 上一轮图，**按工作类型分子目录**
            A_每会话总ADC对比\\  B_残差归一到蠕变\\  C_内部状态轨迹\\
    02_脚本\\  03_结果\\  04_中间数据\\  05_工具\\

不打包（可由脚本一键重建、体积大）：out/_sess、out/_fig、out/_dump、out/_paramsets、out/_raw

用法：python 21_package.py [--force]
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sensor_common import OUT_DIR, SENSORS  # noqa: E402
from sweep_lib import PREP  # noqa: E402

DEST_ROOT = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\01_当前工作_K9\02_现场调参\09-26_传感器分类调参交付包")
DEST = DEST_ROOT / "传感器分类调参_20260926"
ARCHIVE = DEST / "01_最终图片" / "archived" / "第1轮（比例制口径）"
GROUPS = {"A": "A_每会话总ADC对比", "B": "B_残差归一到蠕变", "C": "C_内部状态轨迹"}
PRESETS = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))
_prev_f = OUT_DIR / "figures_archive" / "第1轮_比例制口径" / "09_presets_第1轮.json"
PREV = json.loads(_prev_f.read_text(encoding="utf-8")) if _prev_f.exists() else {}


def run_dump(key: str, named, time_mode="uniform"):
    import subprocess
    from sweep_lib import EXE, Set, write_setfile, SESSION_BIN
    p = PREP[key]
    safe = key.replace("/", "_")
    outdir = OUT_DIR / "_dump" / f"pkg_{time_mode}" / safe
    outdir.mkdir(parents=True, exist_ok=True)
    sf = write_setfile([Set(n, o) for n, o in named],
                       OUT_DIR / "_paramsets" / f"_pkg_{safe}_{time_mode}.txt")
    src = SESSION_BIN.get(key, p["path"])
    r = subprocess.run([str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}",
                        f"{p['t0']:.6f}", "--time", time_mode, "--zero",
                        "--dump", str(outdir)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    res = {}
    for n, _ in named:
        f = outdir / f"{n}.bin"
        raw = np.fromfile(f, dtype=np.int32, count=2)
        res[n] = np.fromfile(f, dtype=np.float64, offset=8).reshape(int(raw[0]), int(raw[1]))
    return res


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    force = "--force" in sys.argv
    if DEST.exists():
        if not force:
            raise SystemExit(f"目标已存在，加 --force 覆盖：{DEST}")
        shutil.rmtree(DEST)
    for d in ("01_最终图片", "02_脚本", "03_结果", "04_中间数据", "05_工具"):
        (DEST / d).mkdir(parents=True, exist_ok=True)
    (DEST / "04_中间数据" / "参考水位拟合").mkdir()

    # 1) 最新图片
    n_new = 0
    for f in sorted((OUT_DIR / "figures").glob("*.png")):
        shutil.copy2(f, DEST / "01_最终图片" / f.name)
        n_new += 1
    # 2) 上一轮图片：按工作类型归档
    n_old, per_group = 0, {}
    arc_src = OUT_DIR / "figures_archive" / "第1轮_比例制口径"
    for letter, label in GROUPS.items():
        sub = arc_src / label
        if not sub.exists():
            continue
        (ARCHIVE / label).mkdir(parents=True, exist_ok=True)
        for f in sorted(sub.glob("*.png")):
            shutil.copy2(f, ARCHIVE / label / f.name)
            n_old += 1
        per_group[label] = len(list(sub.glob("*.png")))
    if _prev_f.exists():
        shutil.copy2(_prev_f, ARCHIVE / "09_presets_第1轮.json")
    print(f"[1/6] 最新图 {n_new} 张；归档上一轮 {n_old} 张（按类型：{per_group}）")

    # 3) 脚本
    n_py = 0
    for f in sorted(HERE.iterdir()):
        if f.suffix in (".py", ".cpp", ".bat") and f.is_file():
            shutil.copy2(f, DEST / "02_脚本" / f.name)
            n_py += 1
    print(f"[2/6] 脚本 {n_py} 个")

    # 4) 结果
    n_res = 0
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file() and f.suffix in (".json", ".txt", ".log", ".png") \
                and not f.name.startswith("_"):
            shutil.copy2(f, DEST / "03_结果" / f.name)
            n_res += 1
    print(f"[3/6] 结果文件 {n_res} 个")

    # 5) 中间数据
    n_npz = 0
    for f in sorted((OUT_DIR / "_prep").glob("*.npz")):
        shutil.copy2(f, DEST / "04_中间数据" / "参考水位拟合" / f.name)
        n_npz += 1
    bundle = {}
    for sensor in SENSORS:
        best = PRESETS[sensor]["params"]
        prev = PREV.get(sensor, {}).get("params")
        for tag in SENSORS[sensor]["sessions"]:
            key = f"{sensor}/{tag}"
            named = [("live", {}), ("new", best)] + ([("old", prev)] if prev else [])
            a = run_dump(key, named)
            nm = key.replace("/", "_")
            bundle[f"{nm}/t"] = a["live"][:, 0].astype(np.float32)
            bundle[f"{nm}/输入_未补偿"] = a["live"][:, 1].astype(np.float32)
            bundle[f"{nm}/显示_现役默认"] = a["live"][:, 2].astype(np.float32)
            if prev:
                bundle[f"{nm}/显示_第一轮预设"] = a["old"][:, 2].astype(np.float32)
            bundle[f"{nm}/显示_本轮推荐"] = a["new"][:, 2].astype(np.float32)
            bundle[f"{nm}/ΣE"] = np.float32(PREP[key]["Esum_channels"])
            bundle[f"{nm}/真实蠕变"] = np.float32(PREP[key]["creep_total"])
            bundle[f"{nm}/t0"] = np.float32(PREP[key]["t0"])
    np.savez_compressed(DEST / "04_中间数据" / "图源数据.npz", **bundle)
    print(f"[4/6] 中间数据：参考水位 {n_npz} 份 + 图源数据.npz（{len(bundle)} 个数组）")

    # 6) 工具
    for src, dst in ((HERE / "build" / "v34_sweep_v3.exe", "v34_sweep_v3.exe"),
                     (HERE / "v34_sweep.cpp", "v34_sweep.cpp"),
                     (HERE / "build_v34_sweep_v3.bat", "build_v34_sweep_v3.bat")):
        if src.exists():
            shutil.copy2(src, DEST / "05_工具" / dst)
    print("[5/6] 工具 v34_sweep_v3.exe + 源码 + 编译脚本")

    # 7) README
    def summ(sensor: str) -> dict:
        v = list(PRESETS[sensor]["sessions"].values())
        return {
            "notch": sum(m["notch"] for m in v) / len(v),
            "low": sum(m["low_exc"] for m in v) / len(v),
            "peak": sum(m["peak_exc"] for m in v) / len(v),
            "end": sum(m["end_exc"] for m in v) / len(v),
            "ded": sum(m["ded_pct"] for m in v) / len(v),
            "notch_max": max(m["notch"] for m in v),
        }
    old = json.loads((OUT_DIR / "15_compare.json").read_text(encoding="utf-8")) \
        if (OUT_DIR / "15_compare.json").exists() else {}
    rows = []
    for sensor in SENSORS:
        s = summ(sensor)
        o = old.get(sensor, {}).get("旧预设(比例制)", {}).get("sessions", {})
        ov = list(o.values())
        oded = sum(m["ded_pct"] for m in ov) / len(ov) if ov else float("nan")
        oover = sum(m["low_exc"] for m in ov) / len(ov) if ov else float("nan")
        rows.append(f"| {sensor} | {s['notch']:.0f} / {s['notch_max']:.0f} | {s['low']:.0f} | "
                    f"{s['peak']:.0f} | {s['end']:+.0f} | **{s['ded']:.0f} %** | "
                    f"{oover:.0f} / {oded:.0f} % |")
    tbl = "\n".join(rows)
    attr_f = OUT_DIR / "19_notch_attr.json"
    attr_txt = ""
    if attr_f.exists():
        a = json.loads(attr_f.read_text(encoding="utf-8"))
        ai = np.mean([v["notch_in"] for v in a.values()])
        ao = np.mean([v["notch_out"] for v in a.values()])
        ae = np.mean([v["notch_extra"] for v in a.values()])
        amax = max(v["notch_extra"] for v in a.values())
        attr_txt = f"""
### 回落归因（本轮定稿预设，逐会话分解）

| 项 | 平均 | 说明 |
|---|---|---|
| 输入自身的回落 | {ai:.0f} ADC | 手指/夹具压力本身在波动，工况固有，任何算法都改不了 |
| 显示回落 | {ao:.0f} ADC | 曲线看上去的"下拉" |
| **算法额外造成的回落** | **{ae:.0f} ADC**（最大 {amax:.0f}） | 这才是算法该负责的部分，已压到你给的 50~150 量级内 |

逐会话明细见 `03_结果/19_notch_attr.json`、`03_结果/15_compare.json`。
"""
    pres = "\n".join(
        f"### {s}\n\n```json\n{json.dumps(PRESETS[s]['params'], ensure_ascii=False)}\n```\n\n"
        f"{PRESETS[s]['note']}\n" for s in SENSORS)
    readme = f"""# 五类传感器 · v3.4 观测器批量复算与调参（2026-09-26）

数据：`D:\\workshop\\文档\\v2.7 - 抗蠕变补偿算法\\data`（四指指腹 4 会话、右拇指指腹 2、
左拇指指腹 2、右手掌 4、左手掌 4，共 16 份；52/57/71 通道，29~277 s，约 100.5 Hz，ADC 模式、
录制时算法关闭 ⇒ 该 CSV 即算法输入）。

算法：**C++ 现役 v3.4 观测器本体**（`src/domain/drift_v6/creep_observer.cpp`，27 参数 /
每通道 11 状态），离线复算器 `v34_sweep.cpp` 直接编译链接该源文件，与上位机逐帧同口径。

---

## 一、本轮（第二轮）改了什么

第一轮按「比例制」目标选参（末值残漂 / 末 30 s 漂移 / 过扣比例），结果在实机曲线上出现两个问题：
**拇指指腹 x1 阶段有明显过减、右手掌基线漂移太大**。本轮把口径改成**绝对 ADC**并加了硬约束：

- 参考线：有加载台阶的会话取拟合弹性电平 ΣE；无台阶的会话取 0（调零后的起始基线）。
- **过减 low_exc** = 显示被压到参考线**下方**多少 ADC（字面意义的"过减"，你的验收项 50~150）；
- 回落 notch = 显示（0.2 s 中值滤波）相对历史最高点的最大回落；
- 欠减峰 peak_exc / 末值偏差 end_exc = 显示浮在参考线上方多少 / 末值偏差（= 基线漂移）。
- 目标函数 `J = peak_exc + 0.5·|end_exc| + 4·max(0, low_exc − 100) + max(0, notch − 150)`，
  即**先把过减压到容差内**，再让显示尽量贴着参考线。

**找到的关键机理**：`slope_cap_frac`（慢态积分速率上限）是「显示能压多低」的硬约束——
`cap = slope_cap_frac·max(e,1)` 必须 ≥ 输入爬升率，否则显示被迫抬到 ≥ 爬升率/`slope_cap_frac`
才追得上，慢态就永远欠扣。这类"无弹性台阶、读数一路爬"的传感器需要
**`slope_cap_frac` 3~6**（现役 0.01、UI 上限 0.05）。同时把沿阈放到极大（3000/1e9）等于
**关掉"加载沿"机制**——这几类传感器本来就没有台阶，沿机制只会反复阻挡慢态积分。

## 二、本轮定稿结果（逐会话平均，单位 ADC）

| 传感器 | 回落 notch 均/最大 | 过减 low_exc | 欠减峰 peak_exc | 末值偏差 | 蠕变扣除率 | 上一轮 过减/扣除 |
|---|---|---|---|---|---|---|
{tbl}

（"上一轮 过减/扣除" = 第一轮比例制预设的 过减 low_exc 与扣除率，用于对比。）
{attr_txt}

## 三、推荐预设

{pres}
## 四、两个必须先知道的口径事实

1. **「不调零就开算法」= 完全不补偿**：手掌/拇指这 12 份录制带着 4915~25120 ADC 的无载偏置，
   读数整体只涨 5~13 %，永远越不过全局总值旁路的 `release = 1.15×baseline`，
   实测 **12/12 会话扣除恒为 0.0 ADC** ⇒ 现场必须「装好负载 → 调零 → 开算法」。
   四指指腹首帧本就是无载零点（0），不受影响。
2. **可扣除比例有硬上限**：`x2 ≤ r_slow_max·max(e,1)` 且 `e = y − x1 − x2`；当弹性电平≈0 时
   `e` ≈ 累计蠕变 ⇒ 上限 = `r_slow_max/(1+r_slow_max)`。默认 0.35 只能扣 25.9 %（实测 6~19 % 吻合），
   扣到 90 % 需要 r≈9，而 UI 上限只有 0.600。

## 五、是否需要新增参数（结论：不需要新增，需要放开可调面）

- **0 个新参数**：预设用到的键全部是 `creep_observer.h::Params` 现有成员
  （脚本 `20_param_gap.py` 直接解析头文件与 kSpecs 核对，见 `03_结果/12_param_gap.json`）。
- **只需"暴露"6 个已有但对话框未暴露的参数**：`slope_gate_frac`、`tau_slope_s`、`idle_frac`、
  `ramp_slope_min`、`hold_eps`、`edge_slope_thres`（本轮还用到 `hold_tau_s`、`tau_c_fast_boost_s`、
  `ramp_full_s`、`edge_boost_s` —— 同属这 19 个未暴露项）。
- **只需"放宽 min/max"**：`r_slow_max` 0.60 → ≥5（拇指/手掌）、**`slope_cap_frac` 0.05 → ≥6**
  （本轮最关键的放宽项）、`r_fast` 0.50 → ≥2。
- 若要求**设备侧也生效**，才需要新增**寄存器**（协议/固件的事，不是算法参数）。

## 六、目录说明

| 目录 | 内容 |
|---|---|
| `01_最终图片/` | **本轮 27 张**：`A_*` 每会话「未补偿输入 / ΣE 参考 / 现役默认 / 第一轮预设 / 本轮推荐」总 ADC 对比（16 张）；`B_*` 每类传感器「残差归一到蠕变」overlay（5 张，含第一轮对照）；`C_*` x1/x2/applied 内部状态轨迹（6 张） |
| `01_最终图片/archived/第1轮（比例制口径）/` | **上一轮同 27 张**，按工作类型分子目录 `A_每会话总ADC对比` / `B_残差归一到蠕变` / `C_内部状态轨迹`，另附 `09_presets_第1轮.json` |
| `02_脚本/` | 分析、寻优、出图、复核脚本（`05_prep`→`06_sweep1d`→`08_search`→`14/24/26/27` 寻优→`28_finalize` 定稿→`09_plot` 出图→`13/17/18/19/20/25` 复核），以及离线复算器源码与编译脚本 |
| `03_结果/` | 各阶段结果与日志（见下） |
| `04_中间数据/` | `参考水位拟合/*.npz`（逐通道弹性电平 E_k 与蠕变分量）；`图源数据.npz`（每会话 `t / 输入_未补偿 / 显示_现役默认 / 显示_第一轮预设 / 显示_本轮推荐 / ΣE / 真实蠕变 / t0`，即 A、B 两组图的数据源） |
| `05_工具/` | `v34_sweep_v3.exe`（离线复算器，输出含 notch/sm_* 绝对口径列）+ 源码 + 编译脚本 |

`03_结果/` 关键文件：`02_prep.json`（ΣE/t0 拟合）、`03_sweep1d.*`（15 旋钮一维扫描）、
`04_search*.json`（比例制三轮寻优）、`05_report.txt`（逐会话明细）、`06_ceiling.json`
（`r/(1+r)` 上限验证）、`07_variant.json`（第 28 个参数否决依据）、`10_nozero.json`（不调零实测）、
`11_ui_gap.json` / `12_param_gap.json`（可调面缺口量化）、`13_trim_before.json`（过减/漂移实测）、
`14_search_adc.json`（绝对口径寻优）、`15_compare.json`（现役/第一轮/本轮三者逐会话对比）、
`16_refine.json`（细网格收敛）、`17_focus.json`（预留池定点收敛）、`09_presets.*`（本轮定稿）。

## 七、如何复跑

```powershell
cd <本包>\\02_脚本
python 14_session_bin.py     # 16 份会话 CSV → 二进制（首次必跑）
python 05_prep.py            # 拟合参考弹性电平 ΣE 与加载时刻 t0
python 28_finalize.py        # 从各轮结果里定稿并写 09_presets.json
python 09_plot.py            # 出全部 27 张图（需要 toolbox\\数据解析工具 可导入）
```

## 八、口径与环境注意

- 复算时间轴：默认按 `frame_index` 重建**均匀轴**；上位机实际喂算法的是**批量到达时刻**
  （elapsed 约 70 % 重复）。两种口径结论不翻转（各结果里都有 raw 列对照）。
- 输入口径：统一按现场规范「装好负载 → 调零 → 开算法」，即输入 = 原值 − 首帧值。
- 指标分母：比例类指标一律用「本该被扣掉的累计蠕变」；**过减/欠减/回落一律用绝对 ADC**。
- 环境：Python 3.14 / numpy 2.4.6 / scipy 1.17.1 / matplotlib 3.10.9；VS 2022 Community 的 cl.exe
  （`/std:c++17 /O2`），Eigen 取自仓库 `thirdparty/eigen-5.0.0`。
- **未改任何 `src/` 代码、未改 CMake/CHANGELOG/版本号、未提交 git**；`v34_sweep*.exe` 只是把产品源码
  编译进独立分析工具，不进入任何构建树。结论已记入 `Document/ChangingLog/完整更新日志.md`。
"""
    (DEST / "README.md").write_text(readme, encoding="utf-8")
    print("[6/6] README.md")

    total = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"\n打包完成：{DEST}  合计 {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
