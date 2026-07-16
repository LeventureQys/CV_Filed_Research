"""
main.py — TactileSense v1.0.10 beta1 研究工具箱（中文 GUI 入口）

页面：
  1. 降噪动画回放：选择一个 CSV，用热力图动画对比原始数据和降噪后数据
  2. 算法横向 Benchmark：批量运行多种算法，比较抑制效果、信号保持率和单帧耗时
  3. 单文件处理与统计：对一个 CSV 做一次降噪并输出详细统计解释
"""

from __future__ import annotations

import os
import sys
import csv
import time
import threading
from collections import defaultdict
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from alg.data_loader import load_csv, parse_csv_header, guess_sampling_freq
from alg.noise_suppressor import Analyzer, Processor, ProcessorMode, AnalyzerResult, build_full_noise_suppressor
from gui_launch import launch_replay_animator

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import tkinter.font as tkfont
except ImportError:
    tk = None
    ttk = None
    messagebox = None


# =========================================================================
# Common helpers
# =========================================================================

DATA_SET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DataSet"))


def _find_all_csv() -> list[str]:
    result = []
    for root, dirs, files in os.walk(DATA_SET_DIR):
        for fn in files:
            if fn.endswith(".csv"):
                result.append(os.path.join(root, fn))
    result.sort()
    return result


def _short_label(path: str) -> str:
    rel = os.path.relpath(path, DATA_SET_DIR) if DATA_SET_DIR in path else path
    return rel.replace("\\", "/")


# =========================================================================
# Tab 1 — 降噪动画回放（启动独立 matplotlib 动画窗口）
# =========================================================================

class ReplayTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.csv_path: Optional[str] = None

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Button(top, text="选择录制 CSV", command=self._pick_csv).pack(side="left", padx=2)
        self.path_var = tk.StringVar(value="（尚未选择文件）")
        ttk.Label(top, textvariable=self.path_var, font=("Consolas", 9)).pack(side="left", padx=8, fill="x", expand=True)

        help_box = ttk.LabelFrame(self, text="这个页面是做什么的？", padding=6)
        help_box.pack(fill="x", padx=8, pady=4)
        ttk.Label(
            help_box,
            text=(
                "降噪动画回放用于直观看 processed 是否还有图像。选择一个录制 CSV 后，点击“启动动画窗口”，\n"
                "会打开一个独立动画窗口：左侧是原始热力图，右侧是降噪后的 residual 热力图。\n"
                "动画窗口与主界面分开运行，因此播放动画不会卡住主工具箱。"
            ),
            justify="left",
            wraplength=900,
        ).pack(anchor="w")

        mid = ttk.LabelFrame(self, text="动画与降噪参数", padding=6)
        mid.pack(fill="x", padx=8, pady=4)

        row0 = ttk.Frame(mid)
        row0.pack(fill="x")
        ttk.Label(row0, text="降噪模式").pack(side="left")
        self.mode_var = tk.StringVar(value="StatGate")
        mode_combo = ttk.Combobox(row0, textvariable=self.mode_var,
                                   values=["StatGate", "Spatial", "Hybrid", "EdgeGate", "TemporalGate", "SpatioTemporal"],
                                   state="readonly", width=10)
        mode_combo.pack(side="left", padx=4)
        ttk.Label(row0, text="k_sigma").pack(side="left", padx=(12, 0))
        self.k_sigma_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(row0, from_=0.5, to=6.0, increment=0.5, textvariable=self.k_sigma_var, width=6).pack(side="left", padx=4)
        ttk.Label(row0, text="decay").pack(side="left", padx=(12, 0))
        self.decay_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(row0, from_=0.0, to=5.0, increment=0.5, textvariable=self.decay_var, width=6).pack(side="left", padx=4)
        ttk.Label(row0, text="FPS").pack(side="left", padx=(12, 0))
        self.fps_var = tk.IntVar(value=10)
        ttk.Spinbox(row0, from_=1, to=60, textvariable=self.fps_var, width=5).pack(side="left", padx=4)

        row1 = ttk.Frame(mid)
        row1.pack(fill="x", pady=(4, 0))
        ttk.Label(row1, text="时域窗口").pack(side="left")
        self.tw_var = tk.IntVar(value=3)
        ttk.Spinbox(row1, from_=1, to=10, textvariable=self.tw_var, width=5).pack(side="left", padx=4)
        ttk.Label(row1, text="空间 sigma").pack(side="left", padx=(12, 0))
        self.ss_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(row1, from_=0.1, to=5.0, increment=0.1, textvariable=self.ss_var, width=5).pack(side="left", padx=4)

        ttk.Button(mid, text="启动动画窗口", command=self._launch).pack(pady=6)

    def _pick_csv(self):
        path = filedialog.askopenfilename(
            title="选择录制 CSV 文件",
            filetypes=[("CSV 文件", "*.csv")],
            initialdir=DATA_SET_DIR)
        if path:
            self.csv_path = path
            self.path_var.set(_short_label(path))

    def _launch(self):
        if not self.csv_path or not os.path.isfile(self.csv_path):
            messagebox.showerror("提示", "请先选择一个 CSV 文件。")
            return
        mode_map = {"StatGate": 0, "Spatial": 1, "Hybrid": 2, "EdgeGate": 3, "TemporalGate": 4, "SpatioTemporal": 5}
        try:
            launch_replay_animator(
                self.csv_path,
                k_sigma=self.k_sigma_var.get(),
                decay=self.decay_var.get(),
                fps=self.fps_var.get(),
                mode=mode_map[self.mode_var.get()],
                temporal_window=self.tw_var.get(),
                spatial_sigma=self.ss_var.get())
        except Exception as e:
            messagebox.showerror("启动动画失败", str(e))


# =========================================================================
# Tab 2 — 算法横向 Benchmark（批量对比算法效果和耗时）
# =========================================================================

class BenchmarkTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._running = False

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Button(top, text="运行全量 Benchmark", command=self._run_benchmark).pack(side="left", padx=2)
        ttk.Button(top, text="保存 Benchmark 结果", command=self._save_csv).pack(side="left", padx=2)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(top, textvariable=self.status_var, font=("Consolas", 9)).pack(side="left", padx=8)

        help_box = ttk.LabelFrame(self, text="这个页面是做什么的？", padding=6)
        help_box.pack(fill="x", padx=8, pady=4)
        ttk.Label(
            help_box,
            text=(
                "算法横向 Benchmark 会扫描 DataSet 下的所有录制 CSV，并对 raw、滑动平均、中值滤波、统计门限等算法逐一测试。\n"
                "表格展示每个文件、每个算法的背景残留、抑制量、信号保持率和单帧耗时，用于判断哪种算法更适合当前设备。\n"
                "运行可能需要几十秒；运行时主界面仍可响应。"
            ),
            justify="left",
            wraplength=900,
        ).pack(anchor="w")

        # Tree view for results
        cols = ("csv", *[f"{a}_{m}" for a in ("raw","moving_avg_3","median_3","stat_k3.0","stat_k2.0","stat_k3.0_d1") for m in ("bgTotal","supDB","tRet","msFr")])
        # simplified: show per-algo in columns
        self.tree = ttk.Treeview(self, columns=("algo", "bgTotal", "supDB", "tRet", "msFr"), show="tree headings", height=20)
        self.tree.heading("#0", text="录制文件")
        self.tree.heading("algo", text="算法")
        self.tree.heading("bgTotal", text="背景残留")
        self.tree.heading("supDB", text="抑制(dB)")
        self.tree.heading("tRet", text="信号保持")
        self.tree.heading("msFr", text="单帧ms")
        self.tree.column("#0", width=200)
        self.tree.column("algo", width=120)
        self.tree.column("bgTotal", width=80)
        self.tree.column("supDB", width=80)
        self.tree.column("tRet", width=80)
        self.tree.column("msFr", width=80)

        vbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vbar.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8,0), pady=4)
        vbar.pack(side="right", fill="y", pady=4)

        self._results: list[dict] = []
        self._labels: list[str] = []

    def _run_benchmark(self):
        if self._running:
            return
        self._running = True
        self.status_var.set("正在运行 Benchmark...")
        # clear
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._results.clear()
        self._labels.clear()

        def work():
            from tools.run_benchmark import benchmark_one, ALGORITHMS, print_summary_table
            all_csv = _find_all_csv()
            all_results = []
            csv_labels = []
            for path in all_csv:
                try:
                    meta = parse_csv_header(path)
                    label = f"{meta.get('设备名称','?')} {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
                    frames, meta2 = load_csv(path, max_frames=500)
                    freq = meta2.get("sample_freq_hz", 0.0)
                    if freq == 0:
                        freq = guess_sampling_freq(path)
                    res = benchmark_one(frames, freq)
                    res["csv_path"] = path
                    all_results.append(res)
                    csv_labels.append(label)
                except Exception as e:
                    pass
            self._results = all_results
            self._labels = csv_labels
            self._display_results()
            self.status_var.set(f"完成 — 已处理 {len(all_results)} 个文件")
            self._running = False

        threading.Thread(target=work, daemon=True).start()

    def _display_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        algo_names = ["raw", "moving_avg_3", "median_3", "stat_k3.0", "stat_k2.0", "stat_k3.0_d1"]
        for label, res in zip(self._labels, self._results):
            for algo in algo_names:
                bg = res.get(f"{algo}_bg_total_mean", 0)
                sup = res.get(f"{algo}_suppression_db", 0)
                tr = res.get(f"{algo}_total_retention", 0)
                ms = res.get(f"{algo}_ms_per_frame", 0)
                self.tree.insert("", "end", text=label[:60],
                                 values=(algo, f"{bg:.1f}", f"{sup:.1f}", f"{tr:.3f}", f"{ms:.4f}"))

    def _save_csv(self):
        if not self._results:
            messagebox.showinfo("提示", "还没有结果可保存，请先运行 Benchmark。")
            return
        path = filedialog.asksaveasfilename(
            title="保存 Benchmark 结果",
            defaultextension=".csv",
            filetypes=[("CSV 文件", "*.csv")])
        if not path:
            return
        try:
            from tools.run_benchmark import save_csv as _save
            _save(self._results, self._labels, path)
            messagebox.showinfo("已保存", f"结果已保存到：{path}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))


# =========================================================================
# Tab 3 — 单文件处理与统计（选择一个 CSV，跑一次降噪并解释指标）
# =========================================================================

class SingleProcessorTab(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.csv_path: Optional[str] = None
        self._frames: Optional[np.ndarray] = None
        self._meta: Optional[dict] = None

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Button(top, text="选择一个录制 CSV", command=self._pick_csv).pack(side="left", padx=2)
        self.path_var = tk.StringVar(value="（尚未选择文件）")
        ttk.Label(top, textvariable=self.path_var, font=("Consolas", 9)).pack(side="left", padx=8)

        help_box = ttk.LabelFrame(self, text="这个页面是做什么的？", padding=6)
        help_box.pack(fill="x", padx=8, pady=4)
        ttk.Label(
            help_box,
            text=(
                "本功能用于检查某一个 CSV 在当前降噪参数下的效果。\n"
                "它不会播放动画，也不会批量对比算法；它只做一件事：加载一个文件 → 建立噪声模型 → 处理全部帧 → 输出统计报告。\n"
                "适合用来快速回答：processed 是否还有有效信号、背景噪声压掉多少、单帧处理时长是多少。"
            ),
            justify="left",
            wraplength=900,
        ).pack(anchor="w")

        # Parameters
        mid = ttk.LabelFrame(self, text="参数", padding=6)
        mid.pack(fill="x", padx=8, pady=4)
        row0 = ttk.Frame(mid)
        row0.pack(fill="x")
        ttk.Label(row0, text="降噪模式").pack(side="left")
        self.mode_var = tk.StringVar(value="StatGate")
        mode_combo = ttk.Combobox(row0, textvariable=self.mode_var,
                                   values=["StatGate", "Spatial", "Hybrid", "EdgeGate", "TemporalGate", "SpatioTemporal"],
                                   state="readonly", width=10)
        mode_combo.pack(side="left", padx=4)
        ttk.Label(row0, text="k_sigma").pack(side="left", padx=(12, 0))
        self.k_sigma_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(row0, from_=0.5, to=6.0, increment=0.5, textvariable=self.k_sigma_var, width=6).pack(side="left", padx=4)
        ttk.Label(row0, text="decay").pack(side="left", padx=(12, 0))
        self.decay_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(row0, from_=0.0, to=5.0, increment=0.5, textvariable=self.decay_var, width=6).pack(side="left", padx=4)

        row1 = ttk.Frame(mid)
        row1.pack(fill="x", pady=(4, 0))
        ttk.Label(row1, text="时域窗口").pack(side="left")
        self.tw_var = tk.IntVar(value=3)
        ttk.Spinbox(row1, from_=1, to=10, textvariable=self.tw_var, width=5).pack(side="left", padx=4)
        ttk.Label(row1, text="空间 sigma").pack(side="left", padx=(12, 0))
        self.ss_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(row1, from_=0.1, to=5.0, increment=0.1, textvariable=self.ss_var, width=5).pack(side="left", padx=4)

        ttk.Button(mid, text="处理并生成统计报告", command=self._process).pack(pady=6)

        # Stats text
        self.stats_text = tk.Text(self, height=18, font=("Consolas", 10), wrap="none")
        self.stats_text.pack(fill="both", expand=True, padx=8, pady=4)
        vbar = ttk.Scrollbar(self, orient="vertical", command=self.stats_text.yview)
        vbar.pack(side="right", fill="y")
        self.stats_text.configure(yscrollcommand=vbar.set)

    def _pick_csv(self):
        path = filedialog.askopenfilename(
            title="选择一个录制 CSV 文件",
            filetypes=[("CSV 文件", "*.csv")],
            initialdir=DATA_SET_DIR)
        if path:
            self.csv_path = path
            self.path_var.set(_short_label(path))
            try:
                self._frames, self._meta = load_csv(path)
                self.stats_text.delete("1.0", "end")
                self.stats_text.insert("1.0",
                    f"已加载文件：{path}\n"
                    f"  帧数：{self._frames.shape[0]}\n"
                    f"  通道数：{self._frames.shape[1]}\n"
                    f"  设备行列：{self._meta.get('rows','?')} 行 × {self._meta.get('cols','?')} 列\n"
                    f"  估计采样率：{self._meta.get('sample_freq_hz',0):.1f} Hz\n"
                    f"  原始总和：{self._frames.sum():.1f}，原始均值：{self._frames.mean():.3f}\n\n"
                    f"点击“处理并生成统计报告”后，会用文件前 100 帧（或一半帧数）学习背景噪声，\n"
                    f"然后对整个文件执行降噪，并显示效果和耗时。\n")
            except Exception as e:
                messagebox.showerror("加载失败", str(e))

    def _process(self):
        if self._frames is None:
            messagebox.showerror("提示", "请先选择一个 CSV 文件。")
            return
        frames = self._frames
        bg = frames[:min(100, len(frames)//2)]
        freq = self._meta.get("sample_freq_hz", 0.0)
        rows = self._meta.get("rows", 0)
        cols = self._meta.get("cols", 0)

        mode_map = {"StatGate": 0, "Spatial": 1, "Hybrid": 2, "EdgeGate": 3, "TemporalGate": 4, "SpatioTemporal": 5}
        mode = mode_map[self.mode_var.get()]

        try:
            model, proc = build_full_noise_suppressor(
                bg, sample_freq_hz=freq,
                k_sigma=self.k_sigma_var.get(),
                noise_gate_decay=self.decay_var.get(),
                mode=mode, rows=rows, cols=cols,
                temporal_window=self.tw_var.get(),
                spatial_sigma=self.ss_var.get())
            out = proc.process_batch(frames)

            # Stats
            bg_orig = frames[:len(bg)]
            bg_proc = out[:len(bg)]
            stress_idx = len(bg)
            stress_orig = frames[stress_idx:] if stress_idx < len(frames) else frames[-1:]
            stress_proc = out[stress_idx:] if stress_idx < len(out) else out[-1:]

            def _safe(v): return 0.0 if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

            bg_orig_sum = _safe(bg_orig.sum(axis=1).mean())
            bg_proc_sum = _safe(bg_proc.sum(axis=1).mean())
            bg_sup = 0.0
            if bg_orig_sum > 0 and bg_proc_sum > 0:
                bg_sup = 20 * np.log10(bg_orig_sum / bg_proc_sum)
            bg_nz_orig = int(_safe((bg_orig > 0).sum(axis=1).mean()))
            bg_nz_proc = int(_safe((bg_proc > 0).sum(axis=1).mean()))

            text = (
                f"──── 功能说明 ────\n"
                f"单文件处理与统计：只针对当前 CSV 做一次完整降噪分析，用来检查参数是否合适。\n"
                f"它回答三个问题：背景噪声压掉了吗？processed 里面还有没有有效信号？单帧处理够不够快？\n\n"
                f"──── 背景段效果（前 {len(bg)} 帧）────\n"
                f"  原始背景总值均值：{bg_orig_sum:.2f}\n"
                f"  降噪后背景总值均值：{bg_proc_sum:.2f}\n"
                f"  背景抑制量：{bg_sup:.2f} dB\n"
                f"  原始非零通道数均值：{bg_nz_orig}\n"
                f"  降噪后非零通道数均值：{bg_nz_proc}\n\n"
                f"解释：背景总值越低，说明静默噪声被压得越干净；非零通道越少，说明误触发越少。\n\n"
                f"──── 噪声模型 ────\n"
                f"  baseline 均值={model.baseline.mean():.2f}，最小={model.baseline.min():.2f}，最大={model.baseline.max():.2f}\n"
                f"  noise_std 均值={model.noise_std.mean():.2f}\n"
                f"  活跃噪声通道：{model.channel_active.sum():.0f}/{model.n_channels}\n"
                f"  用于建模的帧数：{model.n_frames_analyzed}\n\n"
                f"解释：baseline 是每个通道的背景基线；noise_std 是背景波动；门限约等于 baseline + k_sigma × noise_std。\n\n"
                f"──── 性能 ────\n"
                f"  单帧处理时长：{proc.avg_process_time_ms:.4f} ms/frame\n"
                f"  RTF：{proc.rtf:.4f}\n\n"
                f"解释：RTF < 1 表示处理速度快于实时。比如手套约 20Hz，每帧间隔约 50ms；如果单帧处理 0.001ms，远小于 50ms。\n\n"
                f"──── 参数说明 ────\n"
                f"  k_sigma：噪声门限强度。越大越保守，噪声压得更干净，但也更容易把弱信号压没。\n"
                f"  decay：额外扣除量。对通过门限的 residual 再减一点，通常保持 0 即可。\n\n"
                f"──── 通道模型（前 48 个通道）────\n"
                f"  通道  baseline  noise_std  noise_floor  是否活跃噪声\n"
            )
            for c in range(min(48, model.n_channels)):
                text += f"  {c:4d}  {model.baseline[c]:8.1f}  {model.noise_std[c]:8.1f}  {model.noise_floor[c]:8.1f}  {'是' if model.channel_active[c] else '否'}\n"

            self.stats_text.delete("1.0", "end")
            self.stats_text.insert("1.0", text)
        except Exception as e:
            messagebox.showerror("处理失败", str(e))
            import traceback
            traceback.print_exc()


# =========================================================================
# Main Application
# =========================================================================

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TactileSense 降噪研究工具箱  v1.0.10 beta1")
        self.geometry("980x680")

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        nb.add(ReplayTab(nb), text="  降噪动画回放  ")
        nb.add(BenchmarkTab(nb), text="  算法横向 Benchmark  ")
        nb.add(SingleProcessorTab(nb), text="  单文件处理与统计  ")

        # Status bar
        self.status = ttk.Label(
            self,
            text="本工具用于 v1.0.10 beta1 研究版本：只做离线分析、动画展示和算法 Benchmark，不会修改主程序源码。",
            anchor="w",
            font=("Microsoft YaHei UI", 9),
        )
        self.status.pack(fill="x", padx=4, pady=2)


def main():
    if tk is None:
        print("ERROR: tkinter not available. Install python3-tk.")
        sys.exit(1)
    app = MainApp()
    app.mainloop()


if __name__ == "__main__":
    main()
