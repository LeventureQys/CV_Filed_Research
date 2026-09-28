"""
v3.1 vs v3.2-R1 vs v3.2-R2 三臂总值波形对比图 + 数值报告
R1 = seed only (--mem 0)
R2 = seed + slow-phase memory (--mem 1, default)
输出：../figures/v32_r2_comparison.png, ../results/v32_r2_report.txt
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
RESULTS_DIR = os.path.join(PLAN_DIR, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_pre_csv(path):
    rows = []
    in_data = False
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "##Data":
                in_data = True; continue
            if not in_data: continue
            if line.startswith("timestamp"): continue
            parts = line.split(",")
            if len(parts) < 24: continue
            rows.append((float(parts[1]), [float(parts[i]) for i in range(3, 24)]))
    return rows

def run_algo(runner_exe, data, args=None):
    n_ch = 21
    stdin_lines = [str(n_ch)]
    for elapsed, channels in data:
        stdin_lines.append(f"{elapsed} " + " ".join(f"{c}" for c in channels))
    cmd = [runner_exe] + (args or [])
    proc = subprocess.run(cmd, input="\n".join(stdin_lines)+"\n",
                         capture_output=True, text=True, timeout=180)
    results = []
    for line in proc.stdout.strip().split("\n"):
        if line.startswith(("OK ","END ","CH ")): continue
        parts = line.split()
        if len(parts) < 5: continue
        try: results.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except: continue
    return results

def avg_ded(results, t0, t1, min_input=5000):
    vals = [si - so for t, si, so in results if t0 <= t <= t1 and si > min_input]
    return np.mean(vals) if vals else 0

def std_ded(results, t0, t1, min_input=5000):
    vals = [si - so for t, si, so in results if t0 <= t <= t1 and si > min_input]
    return np.std(vals) if vals else 0

print("Loading data...")
data = load_pre_csv(PRE_CSV)
print(f"  {len(data)} frames")

print("Running v3.1...")
v31 = run_algo(V31_RUNNER, data)
print(f"  {len(v31)} frames")

print("Running v3.2-R1 (seed only, no memory)...")
r1 = run_algo(V32_RUNNER, data, ["--seed-gain", "0.70", "--mem", "0"])
print(f"  {len(r1)} frames")

print("Running v3.2-R2 (seed + memory)...")
r2 = run_algo(V32_RUNNER, data, ["--seed-gain", "0.70", "--mem", "1"])
print(f"  {len(r2)} frames")

# Also run R2 without seed (memory only) to isolate the effect
print("Running v3.2-R2-only (memory only, no seed)...")
r2only = run_algo(V32_RUNNER, data, ["--seed-gain", "0.0", "--mem", "1"])
print(f"  {len(r2only)} frames")

# ─── Numerical report ───
report = []
report.append("=" * 90)
report.append("plan-v3.2 三臂对比：v3.1 / R1(seed) / R2(seed+memory) / R2-only(memory)")
report.append("数据：零基线-反复增减同一负载 / 20260919_160854_single_device_7b3977")
report.append("=" * 90)

segments = [
    ("前段 [10,30]s", 10, 30),
    ("前段 [100,200]s", 100, 200),
    ("后段 [245,252]s", 245, 252),
    ("后段 [258,262]s", 258, 262),
    ("后段 [262,276]s", 262, 276),
    ("后段 [278,282]s", 278, 282),
    ("后段 [290,296]s", 290, 296),
    ("后段 [300,308]s", 300, 308),
]

report.append(f"\n{'段':>22s} | {'v3.1':>10s} {'R1':>10s} {'R2':>10s} {'R2only':>10s} | {'v3.1 std':>10s} {'R1 std':>10s} {'R2 std':>10s}")
report.append("-" * 110)
for label, t0, t1 in segments:
    d31 = avg_ded(v31, t0, t1)
    dr1 = avg_ded(r1, t0, t1)
    dr2 = avg_ded(r2, t0, t1)
    dr2o = avg_ded(r2only, t0, t1)
    s31 = std_ded(v31, t0, t1)
    sr1 = std_ded(r1, t0, t1)
    sr2 = std_ded(r2, t0, t1)
    report.append(f"{label:>22s} | {d31:10.0f} {dr1:10.0f} {dr2:10.0f} {dr2o:10.0f} | {s31:10.0f} {sr1:10.0f} {sr2:10.0f}")

# Overall problem region stats
report.append(f"\n问题区段 [245,312]s 受载帧统计：")
for name, res in [("v3.1", v31), ("R1", r1), ("R2", r2), ("R2only", r2only)]:
    d = avg_ded(res, 245, 312)
    s = std_ded(res, 245, 312)
    report.append(f"  {name:>8s}: mean ded = {d:+8.0f}, std = {s:6.0f}")

# Zero regression check
report.append(f"\n首段 [10,230]s 零回归验证：")
for name, res in [("R1", r1), ("R2", r2), ("R2only", r2only)]:
    d = avg_ded(res, 10, 230)
    d31_ref = avg_ded(v31, 10, 230)
    report.append(f"  {name:>8s}: mean ded = {d:.1f} (v3.1 = {d31_ref:.1f}, diff = {d-d31_ref:+.1f})")

report_text = "\n".join(report)
report_path = os.path.join(RESULTS_DIR, "v32_r2_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"\n报告：{report_path}")
print(report_text)

# ─── Plot ───
step = 5
t_arr = np.array([r[0] for r in v31[::step]])
inp = np.array([r[1] for r in v31[::step]])
o31 = np.array([r[2] for r in v31[::step]])
or1 = np.array([r[2] for r in r1[::step]])
or2 = np.array([r[2] for r in r2[::step]])

fig, axes = plt.subplots(3, 1, figsize=(18, 15), gridspec_kw={'height_ratios': [3, 3, 2]})

# Panel 1: Full timeline
ax = axes[0]
ax.plot(t_arr, inp, color='#CCCCCC', linewidth=0.5, label='原始输入', zorder=1)
ax.plot(t_arr, o31, color='#2176FF', linewidth=0.7, label='v3.1 (现役)', zorder=2)
ax.plot(t_arr, or1, color='#FF8C00', linewidth=0.7, label='v3.2 R1 (seed)', zorder=3, alpha=0.8)
ax.plot(t_arr, or2, color='#E8383D', linewidth=0.9, label='v3.2 R2 (seed+memory)', zorder=4)
ax.axvline(240, color='#666', ls='--', lw=0.6, alpha=0.5)
ax.set_ylabel('总值 (ADC)')
ax.set_title('全程总值波形：v3.1 / R1 / R2', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.2)

# Panel 2: Zoom 236-312s
ax2 = axes[1]
m = (t_arr >= 236) & (t_arr <= 312)
ax2.plot(t_arr[m], inp[m], color='#CCCCCC', linewidth=0.6, label='原始输入', zorder=1)
ax2.plot(t_arr[m], o31[m], color='#2176FF', linewidth=1.0, label='v3.1', zorder=2)
ax2.plot(t_arr[m], or1[m], color='#FF8C00', linewidth=1.0, label='R1 (seed)', zorder=3, alpha=0.8)
ax2.plot(t_arr[m], or2[m], color='#E8383D', linewidth=1.2, label='R2 (seed+memory)', zorder=4)
ax2.set_xlim(236, 312)
ax2.set_ylabel('总值 (ADC)')
ax2.set_title('问题区段 236-312s 放大', fontsize=11)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.2)

# Panel 3: Delta
d31 = o31 - inp
dr1 = or1 - inp
dr2 = or2 - inp
ax3 = axes[2]
ax3.plot(t_arr, d31, color='#2176FF', linewidth=0.6, label='v3.1 delta', alpha=0.7)
ax3.plot(t_arr, dr1, color='#FF8C00', linewidth=0.6, label='R1 delta', alpha=0.7)
ax3.plot(t_arr, dr2, color='#E8383D', linewidth=0.8, label='R2 delta')
ax3.axhline(0, color='black', lw=0.4)
ax3.axvline(240, color='#666', ls='--', lw=0.6, alpha=0.5)
ax3.set_ylabel('Delta (ADC)')
ax3.set_xlabel('时间 (s)')
ax3.set_title('补偿量 delta = output − input', fontsize=11)
ax3.legend(loc='lower left', fontsize=9)
ax3.grid(True, alpha=0.2)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, "v32_r2_comparison.png")
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"图：{fig_path}")
plt.close()
