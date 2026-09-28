import subprocess, os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAN_DIR = os.path.join(SCRIPT_DIR, "..")
V31_RUNNER = os.path.join(PLAN_DIR, "..", "v3.1", "scripts", "build", "v30_runner.exe")
V32_RUNNER = os.path.join(PLAN_DIR, "build", "v32_runner.exe")
DATA_DIR = os.path.join(PLAN_DIR, "..", "..", "..", "算法数据&原始数据", "working",
    "零基线-反复增减同一负载", "20260919_160854_single_device_7b3977")
PRE_CSV = os.path.join(DATA_DIR, "device_001_pre_seg0.csv")

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
    proc = subprocess.run(cmd, input="\n".join(stdin_lines)+"\n", capture_output=True, text=True, timeout=180)
    results = []
    for line in proc.stdout.strip().split("\n"):
        if line.startswith(("OK ","END ","CH ")): continue
        parts = line.split()
        if len(parts) < 5: continue
        try: results.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except: continue
    return results

data = load_pre_csv(PRE_CSV)
v31 = run_algo(V31_RUNNER, data)
v32 = run_algo(V32_RUNNER, data, ["--seed-gain", "0.70"])

# Compare the "stability" of compensation in two regions:
# Region A: first hold 10-30s (stable, known good)
# Region B: after 245s, each hold segment

print("=== 前段 vs 后段的补偿稳定性对比 ===\n")

# For each "loaded" segment (input > 10000), compute:
# 1. Mean ded
# 2. Std of ded (stability)
# 3. Range of ded

def analyze_segment(results, t0, t1, min_input=10000):
    vals = [(t, si, so, si-so) for t, si, so in results if t0 <= t <= t1 and si > min_input]
    if not vals: return None
    deds = [d for _,_,_,d in vals]
    return {
        't0': t0, 't1': t1,
        'n': len(deds),
        'mean': np.mean(deds),
        'std': np.std(deds),
        'min': np.min(deds),
        'max': np.max(deds),
        'range': np.max(deds) - np.min(deds),
    }

segments = [
    ("前段 [10,30]", 10, 30),
    ("前段 [30,60]", 30, 60),
    ("前段 [100,200]", 100, 200),
    ("后段 [245,252]", 245, 252),
    ("后段 [258,262]", 258, 262),
    ("后段 [262,276]", 262, 276),
    ("后段 [278,282]", 278, 282),
    ("后段 [290,296]", 290, 296),
    ("后段 [300,304]", 300, 304),
]

print(f"{'段':>20s} | {'v3.1 mean':>10s} {'v3.1 std':>10s} {'v3.1 range':>10s} | {'v3.2 mean':>10s} {'v3.2 std':>10s} {'v3.2 range':>10s}")
print("-" * 100)
for label, t0, t1 in segments:
    s31 = analyze_segment(v31, t0, t1)
    s32 = analyze_segment(v32, t0, t1)
    if s31 and s32:
        print(f"{label:>20s} | {s31['mean']:10.0f} {s31['std']:10.0f} {s31['range']:10.0f} | {s32['mean']:10.0f} {s32['std']:10.0f} {s32['range']:10.0f}")

# Also look at the transitions - the actual jumps
print("\n\n=== 后段每次加载/卸载时的显示跳变 ===")
print("（v3.2 是否在加载/卸载瞬间有大跳变？）\n")

for idx in range(1, len(v32)):
    t = v32[idx][0]
    if t < 240 or t > 312: continue
    si_prev = v32[idx-1][1]
    si_now = v32[idx][1]
    so_prev = v32[idx-1][2]
    so_now = v32[idx][2]
    d_in = si_now - si_prev
    d_out = so_now - so_prev
    
    if abs(d_in) > 2000:  # significant input transition
        ded_before = si_prev - so_prev
        ded_after = si_now - so_now
        print(f"  t={t:7.2f}  input: {si_prev:8.0f} -> {si_now:8.0f} ({d_in:+7.0f})"
              f"  output: {so_prev:8.1f} -> {so_now:8.1f} ({d_out:+7.0f})"
              f"  ded: {ded_before:+7.0f} -> {ded_after:+7.0f}")
