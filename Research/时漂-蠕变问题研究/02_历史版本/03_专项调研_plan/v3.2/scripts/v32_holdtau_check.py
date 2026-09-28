import subprocess, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAN_DIR = os.path.join(SCRIPT_DIR, "..")
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

def avg_ded(results, t0, t1):
    vals = [d for t, si, so, d in results if t0 <= t <= t1]
    return sum(vals)/len(vals) if vals else 0

data = load_pre_csv(PRE_CSV)

# Key loading events to check
events = [
    ("245.3s", 250, 260),
    ("258.3s", 263, 273),
    ("261.7s", 264, 274),
    ("278.5s", 283, 293),
    ("300.5s", 305, 310),
]

print("Checking idle gap durations...")
# Find the unload-to-load gaps
for i, (elapsed, channels) in enumerate(data):
    total = sum(channels)
    if i > 0:
        prev_total = sum(data[i-1][1])
        if total - prev_total > 8000 and elapsed > 240:  # big loading step
            # Find when it was last above 5000
            for j in range(i-1, max(0, i-500), -1):
                t_prev = sum(data[j][1])
                if t_prev > 5000:
                    gap = elapsed - data[j][0]
                    print(f"  Load at {elapsed:.1f}s, last loaded at {data[j][0]:.1f}s, gap = {gap:.1f}s")
                    break

print()
print(f"{'gain':>6s} {'hold_tau':>9s} | {'245.3s':>8s} {'258.3s':>8s} {'278.5s':>8s} {'300.5s':>8s}")
print("-" * 60)

# Note: the runner doesn't support --hold-tau directly, but let's check what we have
# For now just test the default gain=0.70 behavior
for gain in ["0.70"]:
    r = run_algo(V32_RUNNER, data, ["--seed-gain", gain])
    vals = [avg_ded(r, t0, t1) for _, t0, t1 in events]
    print(f"{gain:>6s} {'45(def)':>9s} | " + " | ".join(f"{v:8.0f}" for v in vals))

