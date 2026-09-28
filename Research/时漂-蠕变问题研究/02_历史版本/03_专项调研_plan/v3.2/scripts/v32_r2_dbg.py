import subprocess, os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAN_DIR = os.path.join(SCRIPT_DIR, "..")
V32_DBG = os.path.join(PLAN_DIR, "build", "v32_runner_dbg.exe")
DATA_DIR = os.path.join(PLAN_DIR, "..", "..", "..", "算法数据&原始数据", "working",
    "零基线-反复增减同一负载", "20260919_160854_single_device_7b3977")
PRE_CSV = os.path.join(DATA_DIR, "device_001_pre_seg0.csv")

rows = []
in_data = False
with open(PRE_CSV, "r") as f:
    for line in f:
        line = line.strip()
        if line == "##Data":
            in_data = True; continue
        if not in_data: continue
        if line.startswith("timestamp"): continue
        parts = line.split(",")
        if len(parts) < 24: continue
        rows.append((float(parts[1]), [float(parts[i]) for i in range(3, 24)]))

n_ch = 21
stdin_lines = [str(n_ch)]
for elapsed, channels in rows:
    stdin_lines.append(f"{elapsed} " + " ".join(f"{c}" for c in channels))

cmd = [V32_DBG, "--seed-gain", "0.70", "--mem", "1"]
proc = subprocess.run(cmd, input="\n".join(stdin_lines)+"\n",
                     capture_output=True, text=True, timeout=180)

# Print stderr (debug output) - filter for R2 lines
for line in proc.stderr.strip().split("\n"):
    if "R2-" in line or "NEWEV" in line:
        print(line)

print(f"\nTotal stderr lines: {len(proc.stderr.strip().split(chr(10)))}")
