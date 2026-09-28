# -*- coding: utf-8 -*-
"""09 出图：用 toolbox\\数据解析工具（dptool）把最终结果全部画成 PNG。

图组
----
A. 每会话一张：原始输入 / 现役默认 / 该类最优参数 / 弹性电平 ΣE 参考线（总 ADC）
B. 每类传感器一张：把每会话的残差归一到「本该扣掉的蠕变」（%），
   实线 = 最优参数、虚线 = 现役默认、细灰线 = 未补偿输入
C. 两类机理图：x1 / x2 / applied 的逐帧轨迹（Python 版复算，已与 C++ 对拍）

产物：out/figures/*.png
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from creep_observer_k9 import CreepObserverK9  # noqa: E402
from sensor_common import DPTOOL_ROOT, LIVE, OUT_DIR, SENSORS, load  # noqa: E402
from sweep_lib import EXE, PREP, SESSION_BIN, Set, write_setfile  # noqa: E402

FIG = OUT_DIR / "figures"
TMP = OUT_DIR / "_fig"
DUMP = OUT_DIR / "_dump"
PRESETS = json.loads((OUT_DIR / "09_presets.json").read_text(encoding="utf-8"))
# 第一轮（比例制口径）预设，仅作对照；不存在就跳过该曲线
_prev_f = OUT_DIR / "figures_archive" / "第1轮_比例制口径" / "09_presets_第1轮.json"
PREV = json.loads(_prev_f.read_text(encoding="utf-8")) if _prev_f.exists() else {}


def run_dump(key: str, named_params: list[tuple[str, dict]], time_mode: str = "uniform") -> dict:
    """跑 C++ 复算器并把 (t, 输入总值, 显示总值) 落成 bin，返回 {名称: (3,n) ndarray}。"""
    p = PREP[key]
    safe = key.replace("/", "_")
    outdir = DUMP / safe / time_mode
    outdir.mkdir(parents=True, exist_ok=True)
    sf = write_setfile([Set(n, o) for n, o in named_params],
                       OUT_DIR / "_paramsets" / f"_fig_{safe}_{time_mode}.txt")
    src = SESSION_BIN.get(key, p["path"])
    r = subprocess.run([str(EXE), str(src), str(sf), f"{p['Esum_channels']:.6f}",
                        f"{p['t0']:.6f}", "--time", time_mode, "--zero",
                        "--dump", str(outdir)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    res = {}
    for n, _ in named_params:
        f = outdir / f"{n}.bin"
        raw = np.fromfile(f, dtype=np.int32, count=2)
        res[n] = np.fromfile(f, dtype=np.float64, offset=8).reshape(int(raw[0]), int(raw[1]))
    return res


def write_session_like(path: Path, t: np.ndarray, y: np.ndarray, title: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = t + 171642.0
    lines = ["##Session", "方案名, tactile_recording_session", "数据阶段,processed_display",
             "值阶段,processed_display", f"行数,{len(t)}", "列数,2", "数据点数,2",
             "显示模式,adc", "##Data", "timestamp,elapsed,frame_index,ch0"]
    for i in range(len(t)):
        lines.append(f"{ts[i]:.6f},{t[i]:.6f},{i},{y[i]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def label_of(params: dict) -> str:
    if not params:
        return "现役默认"
    order = ["r_fast", "r_slow_max", "slope_gate_frac", "slope_cap_frac", "tau_slope_s",
             "edge_slope_thres", "slow_confirm_s", "soft_unfreeze_s", "tau_c_fast_s",
             "tau_r_slow_idle_s", "tau_r_fast_s", "hold_eps", "idle_frac", "ramp_slope_min"]
    parts = [f"{k[:9]}={params[k]:g}" for k in order if k in params]
    return "最优(" + ",".join(parts) + ")"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    FIG.mkdir(parents=True, exist_ok=True)
    if TMP.exists():
        shutil.rmtree(TMP)
    sys.path.insert(0, str(DPTOOL_ROOT))
    from dptool import api  # noqa: E402

    only = sys.argv[1:] or list(SENSORS)
    for sensor in only:
        best = PRESETS[sensor]["params"]
        live_score = PRESETS[sensor]["live_score"]
        best_score = PRESETS[sensor]["preset_score"]
        # 第一轮（比例制口径）预设：只作为对照曲线画出来，看这一轮改了什么
        prev = PREV.get(sensor, {}).get("params")
        print(f"\n===== {sensor}（现役 score {live_score:.1f} → 推荐预设 {best_score:.1f}）=====")
        # ── 图组 B 的数据容器 ──
        b_dirs: list[Path] = []
        for tag in SENSORS[sensor]["sessions"]:
            key = f"{sensor}/{tag}"
            p = PREP[key]
            named = [("live", {}), ("new", best)] + ([("old", prev)] if prev else [])
            arr = run_dump(key, named)
            t = arr["live"][:, 0]
            tin, live, opt = arr["live"][:, 1], arr["live"][:, 2], arr["new"][:, 2]
            Esum, creep, t0 = p["Esum_channels"], max(p["creep_total"], 1.0), p["t0"]

            # ── 图组 A：每会话一张（总 ADC）──
            g = TMP / sensor / tag.replace("/", "_")
            write_session_like(g / "00_原始输入(未补偿)" / "device_001_seg000.csv", t, tin)
            write_session_like(g / "01_弹性电平ΣE参考" / "device_001_seg000.csv",
                               t, np.full_like(t, Esum))
            write_session_like(g / f"02_现役默认(score{live_score:.0f})" / "device_001_seg000.csv",
                               t, live)
            if prev:
                write_session_like(g / "03_第一轮预设(比例制)" / "device_001_seg000.csv",
                                   t, arr["old"][:, 2])
            write_session_like(g / f"04_本轮推荐(score{best_score:.1f})" / "device_001_seg000.csv",
                               t, opt)
            names = sorted(x.name for x in g.iterdir())
            png = FIG / f"A_{sensor}_{tag.replace('/', '_')}_总ADC对比.png"
            res = api.plot_to_file([str(g / n) for n in names], str(png), mode="overlay",
                                   series_by="dir", series_stream="out", signal="sum",
                                   smooth_s=0.0, stats=False, dpi=130, width_in=15, height_in=7,
                                   suptitle=f"{sensor} {tag}：现役默认 / 第一轮预设 / 本轮推荐"
                                            f"（总 ADC；参考线 ΣE={Esum:.0f}，真实蠕变={creep:.0f}"
                                            f"，占载荷 {creep / max(Esum, 1e-9) * 100:.1f}%）")
            print(f"  A {png.name}: {res.get('path') and 'OK' or res.get('error')}")

            # ── 图组 B：残差归一到蠕变 ──
            gb = TMP / f"B_{sensor}" / tag.replace("/", "_")
            write_session_like(gb / f"{tag.replace('/', '_')}·本轮推荐" / "device_001_seg000.csv",
                               t, (opt - Esum) / creep * 100.0)
            if prev:
                write_session_like(gb / f"{tag.replace('/', '_')}·第一轮预设"
                                   / "device_001_seg000.csv",
                                   t, (arr["old"][:, 2] - Esum) / creep * 100.0)
            write_session_like(gb / f"{tag.replace('/', '_')}·现役默认" / "device_001_seg000.csv",
                               t, (live - Esum) / creep * 100.0)
            write_session_like(gb / f"{tag.replace('/', '_')}·未补偿输入" / "device_001_seg000.csv",
                               t, (tin - Esum) / creep * 100.0)
            b_dirs.append(gb)
        # 一个 overlay：每会话 3 条（未补偿 / 现役 / 最优）
        dirs = [str(d / n) for d in b_dirs for n in sorted(x.name for x in d.iterdir())]
        png = FIG / f"B_{sensor}_残差归一到蠕变.png"
        res = api.plot_to_file(dirs, str(png), mode="overlay", series_by="dir",
                               series_stream="out", signal="sum", smooth_s=0.0, dpi=130,
                               width_in=15, height_in=7,
                               suptitle=f"{sensor}：显示相对弹性电平 ΣE 的残差（100 % = 完全没补偿；"
                                        f"0 % = 钉在弹性电平；< 0 = 过扣下漂）")
        print(f"  B {png.name}: {res.get('path') and 'OK' or res.get('error')}")

    # ── 图组 C：机理（x1/x2/applied 轨迹）──
    CASES = ["四指指腹/d1/6a679f", "左拇指指腹/d1", "右手掌/d1/A"]
    for key in CASES:
        if not any(key.startswith(s + "/") for s in only):
            continue
        sensor = key.split("/")[0]
        best = PRESETS[sensor]["params"]
        p = PREP[key]
        d = load(p["path"].split("data\\", 1)[-1].replace("\\", "/"))
        t = d["t"] - d["t"][0]
        V = d["V"] - d["V"][0]
        for nm, over in (("现役默认", {}), ("推荐预设", best)):
            c = CreepObserverK9(replace(LIVE, **over))
            n = len(t)
            for i in range(n):
                c.process(float(t[i]), V[i])
            tr = c.traces()
            g = TMP / f"C_{sensor}" / key.replace("/", "_") / nm
            write_session_like(g / "00_输入总值" / "device_001_seg000.csv", t, V.sum(axis=1))
            write_session_like(g / "01_显示总值" / "device_001_seg000.csv", t,
                               tr["total_out"].sum(axis=1))
            write_session_like(g / "02_快态x1" / "device_001_seg000.csv", t, tr["x_fast"].sum(axis=1))
            write_session_like(g / "03_慢态x2" / "device_001_seg000.csv", t, tr["x_slow"].sum(axis=1))
            write_session_like(g / "04_实际施加applied" / "device_001_seg000.csv", t,
                               tr["applied"].sum(axis=1))
            names = sorted(x.name for x in g.iterdir())
            png = FIG / f"C_{sensor}_{key.split('/')[-1]}_{nm}_内部状态.png"
            res = api.plot_to_file([str(g / x) for x in names], str(png), mode="overlay",
                                   series_by="dir", series_stream="out", signal="sum",
                                   smooth_s=0.0, dpi=130, width_in=15, height_in=7,
                                   suptitle=f"{key} · {nm}：输入/显示 与 观测器内部状态"
                                            f"（总 ADC；x1 快态、x2 慢态、applied 实际施加）")
            print(f"  C {png.name}: {res.get('path') and 'OK' or res.get('error')}")
    print(f"\n全部图片在 {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
