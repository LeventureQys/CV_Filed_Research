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
    # Parse the END line to get diagnostic counts
    for line in proc.stdout.strip().split("\n"):
        if line.startswith("END "):
            print(f"  END line: {line}")
    results = []
    for line in proc.stdout.strip().split("\n"):
        if line.startswith(("OK ","END ","CH ")): continue
        parts = line.split()
        if len(parts) < 5: continue
        try: results.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except: continue
    return results

data = load_pre_csv(PRE_CSV)

# The END line should show n_mem_restore if we added it properly
# But the runner's END line format was not updated for n_mem_restore...
# Let me check what internal state g_ has at key transition points

# Run R2 and check g values at frame output
print("R2 (seed+mem):")
r2 = run_algo(V32_RUNNER, data, ["--seed-gain", "0.70", "--mem", "1"])

print("\nR1 (seed only):")
r1 = run_algo(V32_RUNNER, data, ["--seed-gain", "0.70", "--mem", "0"])

# The runner outputs g as column index 6 (0-based: t=0, sum_in=1, sum_out=2, state=3, ev_valid=4, ev_kind=5, g=6)
# Let me parse g values around 240-250s
print("\n=== g values around key transitions ===")
print(f"{'t':>8s} {'state_R1':>8s} {'g_R1':>10s} {'state_R2':>8s} {'g_R2':>10s}")
