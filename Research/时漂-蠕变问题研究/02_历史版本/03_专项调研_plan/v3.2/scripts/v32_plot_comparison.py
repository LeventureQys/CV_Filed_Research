"""
v3.1 vs v3.2 总值波形对比图
输出：../figures/v32_total_comparison.png
"""
import subprocess, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAN_DIR = os.path.join(SCRIPT_DIR, "..")
V31_RUNNER = os.path.join(PLAN_DIR, "..", "v3.1", "scripts", "build", "v30_runner.exe")
V32_RUNNER = os.path.join(PLAN_DIR, "build", "v32_runner.exe")
DATA_DIR = os.path.join(PLAN_DIR, "..", "..", "..", "算法数据&原始数据", "working",
    "零基线-反复增减同一负载", "20260919_160854_single_device_7b3977")
PRE_CSV = os.path.join(DATA_DIR, "device_001_pre_seg0.csv")
FIG_DIR = os.path.join(PLAN_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def load_pre_csv(path):
    rows = []
    in_data = False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "##Data":
                in_data = True
                continue
            if not in_data:
                continue
            if line.startswith("timestamp"):
                continue
            parts = line.split(",")
            if len(parts) < 24:
                continue
            elapsed = float(parts[1])
            channels = [float(parts[i]) for i in range(3, 24)]
            rows.append((elapsed, channels))
    return rows

def run_algo(runner_exe, data, args=None):
    n_ch = 21
    stdin_lines = [str(n_ch)]
    for elapsed, channels in data:
        stdin_lines.append(f"{elapsed} " + " ".join(f"{c}" for c in channels))
    stdin_text = "\n".join(stdin_lines) + "\n"
    cmd = [runner_exe]
    if args:
        cmd.extend(args)
    proc = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, timeout=180)
    results = []
    for line in proc.stdout.strip().split("\n"):
        if line.startswith("OK ") or line.startswith("END ") or line.startswith("CH "):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            t = float(parts[0])
            sum_in = float(parts[1])
            sum_out = float(parts[2])
            results.append((t, sum_in, sum_out))
        except:
            continue
    return results

print("Loading data...")
data = load_pre_csv(PRE_CSV)
print(f"  {len(data)} frames")

print("Running v3.1...")
v31 = run_algo(V31_RUNNER, data)
print(f"  {len(v31)} frames")

print("Running v3.2 (gain=0.70)...")
v32 = run_algo(V32_RUNNER, data, ["--seed-gain", "0.70"])
print(f"  {len(v32)} frames")

# Downsample to every 5th frame for plotting (reduce density)
step = 5
t_arr = np.array([r[0] for r in v31[::step]])
inp_arr = np.array([r[1] for r in v31[::step]])
v31_arr = np.array([r[2] for r in v31[::step]])
v32_arr = np.array([r[2] for r in v32[::step]])

# ─── Plot ───
fig, axes = plt.subplots(3, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 3, 2]})

# Panel 1: Full timeline
ax = axes[0]
ax.plot(t_arr, inp_arr, color='#BBBBBB', linewidth=0.6, label='算法输入 (原始值)', zorder=1)
ax.plot(t_arr, v31_arr, color='#2176FF', linewidth=0.8, label='v3.1 (现役)', zorder=2)
ax.plot(t_arr, v32_arr, color='#E8383D', linewidth=0.8, label='v3.2 R1 (候选)', zorder=3)
ax.axvline(240, color='#666666', linestyle='--', linewidth=0.7, alpha=0.5)
ax.annotate('240s 完全卸载', xy=(240, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 18000),
            fontsize=8, color='#666666', ha='right')
ax.set_xlim(0, t_arr[-1])
ax.set_ylabel('总值 (ADC)')
ax.set_title('全程总值波形对比：v3.1 vs v3.2 R1', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Zoom into problem region 236-312s
ax2 = axes[1]
mask = (t_arr >= 236) & (t_arr <= 312)
ax2.plot(t_arr[mask], inp_arr[mask], color='#BBBBBB', linewidth=0.8, label='算法输入 (原始值)', zorder=1)
ax2.plot(t_arr[mask], v31_arr[mask], color='#2176FF', linewidth=1.2, label='v3.1 (现役)', zorder=2)
ax2.plot(t_arr[mask], v32_arr[mask], color='#E8383D', linewidth=1.2, label='v3.2 R1 (候选)', zorder=3)
ax2.set_xlim(236, 312)
ax2.set_ylabel('总值 (ADC)')
ax2.set_title('问题区段放大 (236-312s)：v3.1 贴着输入走，v3.2 把补偿接回来', fontsize=11)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Add annotations for key events
for t_ev, label in [(245.3, '245s\n重载'), (258.3, '258s'), (278.5, '278s'), (300.5, '300s')]:
    ax2.axvline(t_ev, color='#999999', linestyle=':', linewidth=0.5, alpha=0.7)

# Panel 3: Delta (output - input) comparison
ax3 = axes[2]
delta31 = v31_arr - inp_arr
delta32 = v32_arr - inp_arr
ax3.fill_between(t_arr, delta31, 0, alpha=0.15, color='#2176FF')
ax3.fill_between(t_arr, delta32, 0, alpha=0.15, color='#E8383D')
ax3.plot(t_arr, delta31, color='#2176FF', linewidth=0.8, label='v3.1 delta (output−input)')
ax3.plot(t_arr, delta32, color='#E8383D', linewidth=0.8, label='v3.2 delta (output−input)')
ax3.axhline(0, color='black', linewidth=0.5)
ax3.axvline(240, color='#666666', linestyle='--', linewidth=0.7, alpha=0.5)
ax3.set_xlim(0, t_arr[-1])
ax3.set_ylabel('Delta (ADC)')
ax3.set_xlabel('时间 (s)')
ax3.set_title('补偿量对比：delta < 0 = 算法在扣除蠕变', fontsize=11)
ax3.legend(loc='lower left', fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(FIG_DIR, "v32_total_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\n图已保存：{out_path}")
plt.close()
