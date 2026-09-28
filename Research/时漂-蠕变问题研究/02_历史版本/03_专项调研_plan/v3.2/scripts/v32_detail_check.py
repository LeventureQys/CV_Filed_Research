import subprocess, os, sys

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
    proc = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, timeout=120)
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
            results.append((t, sum_in, sum_out, sum_in - sum_out))
        except:
            continue
    return results

data = load_pre_csv(PRE_CSV)
v31 = run_algo(V31_RUNNER, data)
v32 = run_algo(V32_RUNNER, data, ["--seed-gain", "0.70"])

# Detailed view at every loading event in 240-312s
print("=== Detailed per-loading-event analysis (240-312s) ===")
print(f"{'event':>12s} {'t_load':>8s} {'t+3s':>8s} {'t+5s':>8s} {'t+10s':>8s} | {'v31[5,10]':>10s} {'v32[5,10]':>10s} {'improve':>8s}")
print("-" * 90)

# Find all loading transitions > 5000 ADC after 240s
load_events = []
for i in range(1, len(data)):
    t = data[i][0]
    total = sum(data[i][1])
    prev_total = sum(data[i-1][1])
    if t > 240 and total - prev_total > 5000:
        load_events.append(t)

# Deduplicate (same transition can appear in multiple frames due to dt=0)
deduped = []
for t in load_events:
    if not deduped or t - deduped[-1] > 1.0:
        deduped.append(t)

for t_load in deduped:
    def avg(results, t0, t1):
        vals = [d for t, si, so, d in results if t0 <= t <= t1 and si > 5000]
        return sum(vals)/len(vals) if vals else 0
    
    d31_3 = avg(v31, t_load+2.5, t_load+3.5)
    d31_5 = avg(v31, t_load+4.5, t_load+5.5)
    d31_10 = avg(v31, t_load+9.5, t_load+10.5)
    d31_avg = avg(v31, t_load+5, t_load+10)
    
    d32_3 = avg(v32, t_load+2.5, t_load+3.5)
    d32_5 = avg(v32, t_load+4.5, t_load+5.5)
    d32_10 = avg(v32, t_load+9.5, t_load+10.5)
    d32_avg = avg(v32, t_load+5, t_load+10)
    
    improve = d32_avg - d31_avg
    print(f"{t_load:12.2f} {d31_3:8.0f} {d31_5:8.0f} {d31_10:8.0f} | {d31_avg:10.0f} {d32_avg:10.0f} {improve:+8.0f}")

# Also show the full trajectory for 278.5s specifically
print("\n=== 278.5s loading: second-by-second ===")
print(f"{'t':>8s} {'input':>8s} {'v31_out':>10s} {'v31_ded':>10s} {'v32_out':>10s} {'v32_ded':>10s}")
for idx in range(len(v31)):
    t = v31[idx][0]
    if 276 <= t <= 297 and idx % 50 == 0:
        si1, so1, d1 = v31[idx][1], v31[idx][2], v31[idx][3]
        si2, so2, d2 = v32[idx][1], v32[idx][2], v32[idx][3]
        print(f"{t:8.2f} {si1:8.0f} {so1:10.1f} {d1:10.1f} {so2:10.1f} {d2:10.1f}")
