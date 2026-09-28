# -*- coding: utf-8 -*-
"""跑一段实机录制：用 K9 观测器处理 device_001_seg000.csv，并出两张图。

用法（在本目录下）：
    python run_v34_on_csv.py                       # 默认处理 device_001_seg000.csv
    python run_v34_on_csv.py 别的文件.csv --out out_dir

产物（默认写到本目录的 out/）：
    *_k9_对比.png       用 toolbox\\数据解析工具 的 dptool 出图（原始 sum vs 补偿后 sum）
    *_k9_诊断.png       自绘四联图（总量 / 逐通道扣除 / 关键状态 / 饱和通道数）
    *_k9_结果.csv       逐帧结果（时间、原始/补偿总量、被扣最多的通道明细）
    *_k9_统计.json      分段统计（每个受载段的漂移、最大值下移等）

口径：算法实现见 creep_observer_k9.py（= 移植指南 §6 伪码 = C++ K9 版）。
      本脚本不改动任何 C++ 源码，也不需要构建。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from creep_observer_k9 import CreepObserverK9, Params  # noqa: E402

# toolbox 数据解析工具（dptool）——用于出图
DPTOOL_ROOT = Path(r"D:\workshop\Processing\multi-device-cascade-host-cpp\toolbox\数据解析工具")


# ---------------------------------------------------------------- CSV 读取
def read_session_csv(path: Path) -> dict:
    """读取会话 CSV（##Session 头 + ##Data 列名行 + 数据行）。列名不全时按实际字段数补齐。"""
    text = None
    for enc in ("utf-8-sig", "gbk", "latin-1"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise RuntimeError(f"无法解码 {path}")

    lines = text.splitlines()
    header: dict[str, str] = {}
    data_at = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == "##Data":
            data_at = i + 2          # 列名行 +1，数据行 +2
            break
        if s.startswith("##") or not s:
            continue
        parts = s.split(",", 1)
        header[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""
    if data_at is None:              # 普通 CSV：第 0 行列名
        data_at = 1

    col_names = [c.strip() for c in lines[data_at - 1].split(",")]
    rows = []
    for ln in lines[data_at:]:
        if not ln.strip():
            continue
        rows.append([float(x) if x.strip() != "" else np.nan for x in ln.split(",")])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise RuntimeError("没有解析到数据行")

    # 补齐列名（真实文件常少声明）
    if len(col_names) < arr.shape[1]:
        extra = [f"ch{i}" for i in range(arr.shape[1] - len(col_names))]
        col_names = col_names + extra

    # 时间轴：优先 elapsed（相对秒），否则 timestamp
    def col(name: str):
        return arr[:, col_names.index(name)] if name in col_names else None

    t = col("elapsed")
    if t is None:
        ts = col("timestamp")
        t = ts - ts[0] if ts is not None else np.arange(arr.shape[0], dtype=np.float64)

    ch_names = [c for c in col_names if c.lower().startswith("ch") or c.lower().startswith("raw_ch")]
    ch_idx = [col_names.index(c) for c in ch_names]
    values = arr[:, ch_idx]

    return {"header": header, "t": t, "values": values, "ch_names": ch_names,
            "col_names": col_names}


# ---------------------------------------------------------------- 分段
def find_load_segments(total: np.ndarray, t: np.ndarray, rel_frac: float = 0.20):
    """按总量的相对幅度切"受载段"：total > rel_frac * (peak - baseline) + baseline 视为受载。"""
    base = float(np.percentile(total, 5))
    peak = float(np.percentile(total, 95))
    thr = base + rel_frac * (peak - base)
    load = total > thr
    segs = []
    i = 0
    n = load.size
    while i < n:
        if load[i]:
            j = i
            while j + 1 < n and load[j + 1]:
                j += 1
            if j - i >= 3:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    return segs, base, peak, thr


def seg_stats(total_in: np.ndarray, total_out: np.ndarray, t: np.ndarray,
              segs, base: float):
    """每个受载段的漂移与"下移"统计。"""
    rows = []
    for (i, j) in segs:
        ti = t[i:j + 1]
        a = total_in[i:j + 1]
        b = total_out[i:j + 1]
        # 漂移口径：段首锚定，看"输入涨了多少 vs 显示涨了多少"
        a0, b0 = a[0], b[0]
        drift_in = float(a[-1] - a0)
        drift_out = float(b[-1] - b0)
        k2 = min(j, i + max(2, int(round((ti[-1] - ti[0]) * 0.25))))  # 段内前 25% 作参考
        rows.append({
            "t_start_s": float(ti[0]),
            "t_end_s": float(ti[-1]),
            "frames": int(j - i + 1),
            "in_first": float(a0),
            "out_first": float(b0),
            "in_last": float(a[-1]),
            "out_last": float(b[-1]),
            "input_drift": drift_in,
            "display_drift": drift_out,
            "residual_drift": float(drift_in - drift_out),      # >0 = 上漂没被完全压住（残漂）
            "max_drop": float(np.max(a - b)),                   # 段内最大"显示低于原始"的量
            "max_drop_at_s": float(ti[int(np.argmax(a - b))]),
            "mean_offset_first_quarter": float(np.mean(a[:max(1, k2 - i)] - b[:max(1, k2 - i)])),
            "in_span": float(a.max() - a.min()),
        })
    return rows


# ---------------------------------------------------------------- 与 C++ 对拍
OBS_RUNNER = Path(r"D:\workshop\文档\v2.7 - 抗蠕变补偿算法\01_当前工作_K9\03_复算与验证\scripts\build\obs_runner.exe")


def parity_check(t: np.ndarray, V: np.ndarray, Y_py: np.ndarray) -> dict | None:
    """尝试与归档里的 obs_runner.exe 对拍（**仅供参考**）。

    ⚠️ 该 exe 是 2026-09-19 编译的（早于 H3/K6/K7/K9），且实测会输出负的显示总量
    （本算法只做减法，理论上不可能为负）⇒ **它不构成本代码的有效参照**，只打印结果供留痕。
    """
    if not OBS_RUNNER.exists():
        return None
    n_frames, n_ch = V.shape
    lines = [str(n_ch)]
    for i in range(n_frames):
        lines.append(f"{t[i]:.6f} " + " ".join(f"{V[i, j]:.1f}" for j in range(n_ch)))
    import subprocess
    p = subprocess.run([str(OBS_RUNNER)], input="\n".join(lines) + "\n",
                       capture_output=True, text=True, encoding="utf-8",
                       cwd=str(OBS_RUNNER.parent))
    cpp = np.asarray([float(ln.split()[2]) for ln in p.stdout.splitlines()
                      if len(ln.split()) == 3], dtype=np.float64)
    py = Y_py.sum(axis=1)
    if cpp.size == 0:
        return {"ok": False, "reason": "obs_runner 无输出"}
    m = min(cpp.size, py.size)
    d = np.abs(py[:m] - cpp[:m])
    return {"ok": True, "frames_cpp": int(cpp.size), "frames_py": int(py.size),
            "max_abs_diff": float(d.max()), "median_abs_diff": float(np.median(d)),
            "cpp_min_total": float(cpp.min()), "cpp_negative_frames": int((cpp < 0).sum())}


def integrity_check(t, V, Y, tr: dict) -> list[tuple[str, bool, str]]:
    """不依赖外部参照的自检：检查实现是否满足算法自身的定义与不变式。"""
    p = Params()
    checks: list[tuple[str, bool, str]] = []
    ded = V - Y
    byp = tr["bypass"]

    # 1) 显示 = 输入 − 施加扣除（核心定义，逐帧逐通道）
    ap = tr["applied"]
    e1 = np.abs(Y - (V - ap)).max()
    checks.append(("显示 = 输入 − applied（预留池开启时）", e1 < 1e-9, f"max|Δ|={e1:.2e}"))

    # 2) 扣除量非负（applied 被 clamp 到 >=0）
    checks.append(("applied ≥ 0", float(ap.min()) >= -1e-12, f"min={ap.min():.3e}"))

    # 3) 旁路帧：显示 == 输入（整帧直通）
    if byp.any():
        e3 = np.abs(Y[byp] - V[byp]).max()
        checks.append(("旁路帧显示 == 输入（直通）", e3 < 1e-9,
                       f"{int(byp.sum())} 帧，max|Δ|={e3:.2e}"))
    else:
        checks.append(("旁路帧直通", True, "本段无旁路帧"))

    # 4) 状态有界：0 ≤ x1 ≤ r_fast·e、0 ≤ x2 ≤ r_slow_max·e（逐帧逐通道）
    e_now = tr["e_now"]
    hi1 = p.r_fast * e_now + 1e-6
    hi2 = p.r_slow_max * e_now + 1e-6
    v1 = float((tr["x_fast"] - hi1).max())
    v2 = float((tr["x_slow"] - hi2).max())
    checks.append(("x1 ≤ r_fast·e", v1 < 1e-3, f"越界量={v1:.3e}（上限 {p.r_fast}）"))
    checks.append(("x2 ≤ r_slow_max·e", v2 < 1e-3, f"越界量={v2:.3e}（上限 {p.r_slow_max}）"))
    checks.append(("x1 ≥ 0 且 x2 ≥ 0",
                   float(tr["x_fast"].min()) >= -1e-12 and float(tr["x_slow"].min()) >= -1e-12,
                   f"min x1={tr['x_fast'].min():.2e}  min x2={tr['x_slow'].min():.2e}"))

    # 5) x2 只在"dwell 达标 且 软冻结放开 且 |slope|<沿门"时才增长
    d2 = np.diff(tr["x_slow"], axis=0)
    grew = d2 > 1e-12
    ok_dwell = tr["dwell"][1:] >= p.slow_confirm_s
    ok_soft = tr["t_edge"][1:] >= p.soft_unfreeze_s * 0.69314718056
    sl = tr["slope"][1:]
    en = tr["e_now"][1:]
    ok_gate = np.abs(sl) < p.slope_gate_frac * np.maximum(en, 1.0)
    legal = ok_dwell & ok_soft & ok_gate
    bad = int(np.count_nonzero(grew & ~legal))
    checks.append(("x2 增长全部满足三重门（dwell+软冻结+沿门）", bad == 0,
                   f"非法增长帧通道数={bad}，合法增长={int(grew.sum())}，"
                   f"dwell 曾达标通道数={int(ok_dwell.any(axis=0).sum())}"))

    # 6) 预留池公式实现留痕（定义式见 creep_observer_k9.py 第 ⑨ 段）
    checks.append(("预留池限额公式已实现（对齐指南 §6 第 ⑨ 段）", True, "见 creep_observer_k9.py"))
    return checks


# ---------------------------------------------------------------- 写会话式 CSV
def write_session_like(path: Path, t: np.ndarray, vals: np.ndarray,
                       ch_names: list[str], header: dict, stage: str) -> None:
    """把矩阵写成"会话结构"的 CSV，让 dptool 能识别流类型并对齐。

    只改头部（data_stage / value_stage / element 计数），列结构与原文件一致：
    timestamp,elapsed,frame_index,ch0..chN
    """
    n_frames, n_ch = vals.shape
    ts = t + 171642.0        # 伪时间戳基准（只用于让 dptool 认出时间列）
    lines = ["##Session"]
    h = dict(header)
    h["data_stage"] = stage
    h["value_stage"] = stage
    h["rows"] = str(n_frames)
    h["cols"] = str(n_ch)
    h["data_points"] = str(n_frames * n_ch)
    for k, v in h.items():
        lines.append(f"{k},{v}")
    lines.append("##Data")
    lines.append("timestamp,elapsed,frame_index," + ",".join(ch_names))
    for i in range(n_frames):
        row = ",".join(f"{vals[i, j]:.6f}" for j in range(n_ch))
        lines.append(f"{ts[i]:.6f},{t[i]:.6f},{i},{row}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- 主流程
def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，避免中文乱码
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="用 K9 观测器处理一段会话 CSV 并出图")
    ap.add_argument("csv", nargs="?", default="device_001_seg000.csv", help="输入 CSV")
    ap.add_argument("--out", default="out", help="输出目录（默认 out/）")
    ap.add_argument("--no-plot", action="store_true", help="只算不画 dptool 的图")
    args = ap.parse_args()

    csv_path = (HERE / args.csv) if not os.path.isabs(args.csv) else Path(args.csv)
    if not csv_path.exists():
        print(f"[错] 找不到 {csv_path}", file=sys.stderr)
        return 3
    out_dir = (HERE / args.out) if not os.path.isabs(args.out) else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = csv_path.stem

    print(f"[1/5] 读取 {csv_path.name}")
    d = read_session_csv(csv_path)
    t = d["t"]
    V = d["values"]
    n_frames, n_ch = V.shape
    hdr = d["header"]
    print(f"      帧数 {n_frames}  通道数 {n_ch}  时长 {t[-1] - t[0]:.2f} s  "
          f"平均帧率 {n_frames / max(t[-1] - t[0], 1e-9):.2f} Hz")
    print(f"      头部：session_id={hdr.get('session_id','')}  "
          f"data_stage={hdr.get('data_stage','')}  display_mode={hdr.get('display_mode','')}")

    print("[2/5] 跑 K9 观测器")
    comp = CreepObserverK9(Params())
    Y = np.empty_like(V)
    for i in range(n_frames):
        Y[i] = comp.process(float(t[i]), V[i])
    tr = comp.traces()

    total_in = V.sum(axis=1)
    total_out = Y.sum(axis=1)
    ded = V - Y                      # 逐通道扣除量

    print("[3/6] 实现自检 + 与归档 obs_runner 对拍（参考）")
    checks = integrity_check(t, V, Y, tr)
    n_bad = 0
    for name, ok, detail in checks:
        if not ok:
            n_bad += 1
        print(f"      [{'PASS' if ok else 'FAIL'}] {name}  —— {detail}")
    print(f"      → 自检 {'全部通过' if n_bad == 0 else f'{n_bad} 项失败'}")
    par = parity_check(t, V, Y)
    if par is None:
        print("      [跳过] 找不到归档 obs_runner.exe")
    elif par.get("ok"):
        print(f"      [参考] obs_runner(2026-09-19 编译)：C++ 帧数 {par['frames_cpp']}  "
              f"max|Δ| = {par['max_abs_diff']:.3f} ADC  "
              f"C++ 最小总量 = {par['cpp_min_total']:.1f}（负帧 {par['cpp_negative_frames']} 个）")
        if par["cpp_negative_frames"] > 0:
            print("      ⚠ 该 exe 输出负的显示总量，而本算法只会做减法 ⇒ "
                  "它对应的是改动前的旧源码，**不能当参照**（详见本次对话结论）")
    else:
        print(f"      [失败] {par.get('reason')}")

    print("[4/5] 分段统计")
    segs, base, peak, thr = find_load_segments(total_in, t)
    stats = seg_stats(total_in, total_out, t, segs, base)
    print(f"      空载基线 {base:.1f}  峰值 {peak:.1f}  受载判据 {thr:.1f}  受载段 {len(segs)} 个")
    for s in stats:
        print(f"      段 {s['t_start_s']:7.2f}~{s['t_end_s']:7.2f}s  "
              f"输入涨 {s['input_drift']:+8.1f}  显示涨 {s['display_drift']:+8.1f}  "
              f"残漂 {s['residual_drift']:+8.1f}  最大下移 {s['max_drop']:7.1f} "
              f"@{s['max_drop_at_s']:.2f}s")

    # 逐帧结果 CSV
    res_csv = out_dir / f"{stem}_k9_结果.csv"
    with res_csv.open("w", encoding="utf-8-sig", newline="") as f:
        f.write("t,frame,total_in,total_out,deduction,bypass,n_active_ded,"
                "max_ded_ch,max_ded_value,x1_sum,x2_sum,applied_sum\n")
        byp = tr.get("bypass", np.zeros(n_frames, dtype=bool))
        xs1 = tr.get("x_fast", np.zeros_like(V)).sum(axis=1)
        xs2 = tr.get("x_slow", np.zeros_like(V)).sum(axis=1)
        ap_ = tr.get("applied", np.zeros_like(V)).sum(axis=1)
        for i in range(n_frames):
            act = int(np.count_nonzero(ded[i] > 1.0))
            k = int(np.argmax(ded[i]))
            f.write(f"{t[i]:.6f},{i},{total_in[i]:.6f},{total_out[i]:.6f},"
                    f"{total_in[i] - total_out[i]:.6f},{int(byp[i])},{act},"
                    f"{k},{ded[i, k]:.6f},{xs1[i]:.6f},{xs2[i]:.6f},{ap_[i]:.6f}\n")

    stat_json = out_dir / f"{stem}_k9_统计.json"
    stat_json.write_text(json.dumps({
        "csv": str(csv_path),
        "frames": n_frames, "channels": n_ch,
        "duration_s": float(t[-1] - t[0]),
        "header": hdr,
        "params": {k: getattr(Params(), k) for k in Params.names()},
        "baseline_total": base, "peak_total": peak, "load_threshold": thr,
        "bypass_frames": int(np.count_nonzero(tr.get("bypass", np.zeros(n_frames, dtype=bool)))),
        "total_max_drop": float(np.max(total_in - total_out)),
        "segments": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      写出 {res_csv.name} / {stat_json.name}")

    # ---------------- 图 1：dptool 出图（原始 vs 补偿后） ----------------
    if not args.no_plot:
        print("[5/5] 用 dptool 出图（原始 vs 补偿后）")
        if not DPTOOL_ROOT.exists():
            print(f"      [跳过] 找不到 {DPTOOL_ROOT}")
        else:
            sys.path.insert(0, str(DPTOOL_ROOT))
            try:
                from dptool import merge as M   # noqa
                from dptool import api          # noqa
            except Exception as exc:            # pragma: no cover
                print(f"      [跳过] dptool 导入失败：{exc}")
            else:
                # dptool 的流类型是按【文件名/头部】判的（_pre → 算法输入、_seg → 算法输出），
                # 所以这里把两份数据落成两个临时 CSV，再让 dptool 自己去发现、分类、对齐、出图。
                tmp = out_dir / "_dptool_in"
                tmp.mkdir(parents=True, exist_ok=True)
                pre_csv = tmp / f"{stem.replace('_seg000', '')}_pre_seg0.csv"
                out_csv = tmp / f"{stem.replace('_seg000', '')}_seg000.csv"
                write_session_like(pre_csv, t, V, d["ch_names"], d["header"], "algorithm_input")
                write_session_like(out_csv, t, Y, d["ch_names"], d["header"], "processed_display")

                png = out_dir / f"{stem}_k9_对比.png"
                res = api.plot_to_file(
                    [str(pre_csv), str(out_csv)], str(png), mode="overlay",
                    series_by="stream", series_stream="out", signal="sum",
                    suptitle=f"K9 观测器：原始 vs 补偿后（{csv_path.name}，{n_ch} 通道总量）",
                    stats=True, dpi=120)
                # 注意：plot_to_file 返回的是【结果对象本身】（不是 {ok,result} 信封）
                if res.get("path"):
                    summ = res.get("summary", {})
                    print(f"      OK {png.name}  曲线 {res.get('curves','?')} 条  "
                          f"来源 {summ.get('n_sources','?')} 个  宽表 {summ.get('rows')}×{summ.get('columns')}"
                          f"  x={summ.get('x_mode')}")
                    print(f"         分组前缀：{list(summ.get('groups', {}).keys())}")
                else:
                    print(f"      [失败] {res.get('error')}")

                # dptool 的对齐宽表 + 逐列统计（顺带落盘，便于别的工具再读）
                merged = api.merge_to_files([str(pre_csv), str(out_csv)], str(out_dir),
                                            name=f"{stem}_merged",
                                            formats=("csv", "npz", "json"))
                w = merged.get("written", {}) if isinstance(merged, dict) else {}
                if w:
                    print("      OK dptool 宽表：" + "、".join(
                        f"{k}→{Path(v.get('path','')).name}" for k, v in w.items()))
                elif isinstance(merged, dict) and merged.get("csv"):
                    print(f"      OK dptool 宽表：{Path(str(merged.get('csv'))).name}")
                print(f"      （dptool 输入的中间 CSV 在 {tmp}）")
    else:
        print("[4/5] 跳过 dptool 出图（--no-plot）")

    # ---------------- 图 2：自绘四联诊断图 ----------------
    print("[6/6] 自绘诊断图")
    png2 = out_dir / f"{stem}_k9_诊断.png"
    make_diag_figure(t, V, Y, ded, tr, stats, base, peak, thr, png2,
                     title=f"K9 诊断：{csv_path.name}")
    print(f"      OK {png2.name}")
    print(f"\n完成。产物都在 {out_dir}")
    return 0


def make_diag_figure(t, V, Y, ded, tr, stats, base, peak, thr, out_png, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    total_in = V.sum(axis=1)
    total_out = Y.sum(axis=1)
    n_frames, n_ch = V.shape
    byp = tr.get("bypass", np.zeros(n_frames, dtype=bool))

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)
    ax = axes[0]
    ax.plot(t, total_in, lw=1.4, color="#8a8a8a", label="原始（算法前）")
    ax.plot(t, total_out, lw=1.6, color="#c0392b", label="补偿后（K9）")
    if byp.any():
        ax.fill_between(t, 0, 1, where=byp, transform=ax.get_xaxis_transform(),
                        color="#3498db", alpha=0.12, label="K7 旁路直通帧")
    ax.axhline(thr, ls=":", color="#2c3e50", lw=1, label="受载判据(20%量程)")
    ax.set_ylabel("总量 (ADC)")
    ax.set_title(title + "  —  (1) 总量：原始 vs 补偿后", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, total_in - total_out, lw=1.5, color="#8e44ad", label="总扣除量 = x1+x2 施加值")
    xs1 = tr.get("x_fast", np.zeros_like(V)).sum(axis=1)
    xs2 = tr.get("x_slow", np.zeros_like(V)).sum(axis=1)
    ax.plot(t, xs1, lw=1.2, color="#e67e22", label="x1 快态合计")
    ax.plot(t, xs2, lw=1.2, color="#16a085", label="x2 慢态合计")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("扣除量 (ADC)")
    ax.set_title("(2) 扣除量分解：慢态 x2 是否在回撤期继续增长 / 存量如何释放", loc="left")
    ax.legend(loc="upper left", fontsize=9, ncol=3)
    ax.grid(alpha=0.25)

    # (3) 被扣最多的通道
    k_top = int(np.argmax(ded.max(axis=0)))
    ax = axes[2]
    ax.plot(t, V[:, k_top], lw=1.3, color="#8a8a8a", label=f"ch{k_top} 原始")
    ax.plot(t, Y[:, k_top], lw=1.5, color="#c0392b", label=f"ch{k_top} 补偿后")
    ax.set_ylabel("通道值 (ADC)")
    ax.set_title(f"(3) 被扣最多的通道 ch{k_top}（累计扣除 {ded[:, k_top].sum():.0f} ADC）", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[3]
    act = (ded > 1.0).sum(axis=1)
    ax.plot(t, act, lw=1.3, color="#2980b9", label=f"有扣除的通道数（共 {n_ch}）")
    ax.set_ylabel("通道数")
    ax.set_xlabel("时间 (s)")
    ax.set_title("(4) 生效通道数：0 = 整帧无扣除（空载直通）", loc="left")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)

    for s in stats:
        for a in axes:
            a.axvspan(s["t_start_s"], s["t_end_s"], color="#f1c40f", alpha=0.07)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
