"""
plan-v3.2 对比分析脚本：v3.1（现役）vs v3.2 R1（候选）
数据：零基线-反复增减同一负载 (20260919_160854_single_device_7b3977)

用法：python v32_compare.py
输出：../results/v32_compare_report.txt
"""
import subprocess, os, sys, csv, io, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAN_DIR = os.path.join(SCRIPT_DIR, "..")

V31_RUNNER = os.path.join(PLAN_DIR, "..", "v3.1", "scripts", "build", "v30_runner.exe")
V32_RUNNER = os.path.join(PLAN_DIR, "build", "v32_runner.exe")

DATA_DIR = os.path.join(
    PLAN_DIR, "..", "..", "..", "算法数据&原始数据", "working",
    "零基线-反复增减同一负载", "20260919_160854_single_device_7b3977"
)
PRE_CSV = os.path.join(DATA_DIR, "device_001_pre_seg0.csv")

RESULTS_DIR = os.path.join(PLAN_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_pre_csv(path):
    """Load algorithm-input CSV, return list of (elapsed, [ch0..ch20])."""
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
                continue  # header
            parts = line.split(",")
            if len(parts) < 24:
                continue
            elapsed = float(parts[1])
            channels = [float(parts[i]) for i in range(3, 24)]
            rows.append((elapsed, channels))
    return rows

def run_algo(runner_exe, data, args=None):
    """Run the runner on the data, return list of (t, sum_in, sum_out, ded)."""
    n_ch = 21
    stdin_lines = [str(n_ch)]
    for elapsed, channels in data:
        stdin_lines.append(f"{elapsed} " + " ".join(f"{c}" for c in channels))
    stdin_text = "\n".join(stdin_lines) + "\n"
    
    cmd = [runner_exe]
    if args:
        cmd.extend(args)
    
    proc = subprocess.run(cmd, input=stdin_text, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print(f"Runner failed: {proc.stderr[:500]}", file=sys.stderr)
        return []
    
    results = []
    lines = proc.stdout.strip().split("\n")
    for line in lines:
        if line.startswith("OK ") or line.startswith("END ") or line.startswith("CH "):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            t = float(parts[0])
            sum_in = float(parts[1])
            sum_out = float(parts[2])
            ded = sum_in - sum_out
            results.append((t, sum_in, sum_out, ded))
        except (ValueError, IndexError):
            continue
    return results

def find_steps(data, threshold=2000):
    """Find significant loading steps in the input data."""
    steps = []
    prev_total = None
    for i, (elapsed, channels) in enumerate(data):
        total = sum(channels)
        if prev_total is not None and abs(total - prev_total) > threshold:
            direction = "LOAD" if total > prev_total else "UNLOAD"
            steps.append((elapsed, direction, prev_total, total, total - prev_total))
        prev_total = total
    return steps

def avg_ded_window(results, t_start, t_end):
    """Average deduction in a time window."""
    vals = [ded for t, si, so, ded in results if t_start <= t <= t_end]
    return sum(vals) / len(vals) if vals else 0.0

def main():
    print(f"Loading data from {PRE_CSV}...")
    data = load_pre_csv(PRE_CSV)
    print(f"  {len(data)} frames loaded, {data[0][0]:.3f} - {data[-1][0]:.3f} s")
    
    # Find steps
    steps = find_steps(data)
    print(f"  {len(steps)} significant transitions found")
    
    # Run v3.1
    print(f"\nRunning v3.1 ({V31_RUNNER})...")
    v31 = run_algo(V31_RUNNER, data)
    print(f"  {len(v31)} output frames")
    
    # Run v3.2 (gain=0.70)
    print(f"\nRunning v3.2 gain=0.70 ({V32_RUNNER})...")
    v32_070 = run_algo(V32_RUNNER, data, ["--seed-gain", "0.70"])
    print(f"  {len(v32_070)} output frames")
    
    # Run v3.2 (gain=0, should equal v3.1)
    print(f"\nRunning v3.2 gain=0.00 (should = v3.1)...")
    v32_000 = run_algo(V32_RUNNER, data, ["--seed-gain", "0.00"])
    print(f"  {len(v32_000)} output frames")
    
    # Build report
    report = []
    report.append("=" * 80)
    report.append("plan-v3.2 对比分析：v3.1（现役）vs v3.2 R1（候选, gain=0.70）")
    report.append("数据：零基线-反复增减同一负载 / 20260919_160854_single_device_7b3977")
    report.append(f"帧数：{len(data)}  时长：{data[-1][0] - data[0][0]:.1f} s")
    report.append("=" * 80)
    
    # Section 1: Step-by-step comparison
    report.append("\n1. 逐加载沿补偿量对比（ded = input - output, 取沿后 [5,20] s 窗口均值）")
    report.append("-" * 80)
    report.append(f"{'加载沿 t':>10s} {'方向':>8s} {'台阶':>8s} {'v3.1 ded':>10s} {'v3.2 ded':>10s} {'改善':>8s}")
    report.append("-" * 80)
    
    load_times = []
    for t, direction, prev, cur, delta in steps:
        if direction == "LOAD" and delta > 2000:
            load_times.append((t, delta))
    
    for t_step, delta_step in load_times:
        ded31 = avg_ded_window(v31, t_step + 5, t_step + 20)
        ded32 = avg_ded_window(v32_070, t_step + 5, t_step + 20)
        improve = ded32 - ded31
        report.append(f"{t_step:10.2f} {'LOAD':>8s} {delta_step:8.0f} {ded31:10.1f} {ded32:10.1f} {improve:+8.1f}")
    
    # Section 2: Time-series comparison at key points
    report.append("\n\n2. 关键时段 delta（output - input）对比")
    report.append("-" * 80)
    
    windows = [
        ("首段保压 [10, 30]s", 10, 30),
        ("首段保压 [100, 200]s", 100, 200),
        ("首段保压尾段 [220, 240]s", 220, 240),
        ("问题区段 [245, 260]s", 245, 260),
        ("问题区段 [262, 275]s", 262, 275),
        ("问题区段 [278, 295]s", 278, 295),
        ("后段 [300, 310]s", 300, 310),
    ]
    
    report.append(f"{'时段':>30s} {'v3.1 delta':>12s} {'v3.2 delta':>12s} {'改善':>10s}")
    report.append("-" * 80)
    for label, t0, t1 in windows:
        d31 = [(so - si) for t, si, so, ded in v31 if t0 <= t <= t1]
        d32 = [(so - si) for t, si, so, ded in v32_070 if t0 <= t <= t1]
        avg31 = sum(d31) / len(d31) if d31 else 0
        avg32 = sum(d32) / len(d32) if d32 else 0
        improve = avg32 - avg31
        report.append(f"{label:>30s} {avg31:12.1f} {avg32:12.1f} {improve:+10.1f}")
    
    # Section 3: Verify gain=0 matches v3.1
    report.append("\n\n3. 零回归验证：v3.2(gain=0) vs v3.1 应逐位一致")
    report.append("-" * 80)
    max_diff = 0.0
    diff_frames = 0
    for (t1, si1, so1, d1), (t2, si2, so2, d2) in zip(v31, v32_000):
        diff = abs(so1 - so2)
        if diff > 0.01:
            diff_frames += 1
        max_diff = max(max_diff, diff)
    report.append(f"差异帧数：{diff_frames} / {len(v31)}")
    report.append(f"最大差异：{max_diff:.6f}")
    if diff_frames == 0:
        report.append("✓ 逐位一致（gain=0 完全回到 v3.1 行为）")
    else:
        report.append("✗ 有差异！")
    
    # Section 4: Step fidelity G
    report.append("\n\n4. 台阶保真 G（加载沿后 [3,5] s 的 Δ显示/Δ输入）")
    report.append("-" * 80)
    report.append(f"{'加载沿 t':>10s} {'台阶':>8s} {'G v3.1':>10s} {'G v3.2':>10s}")
    report.append("-" * 80)
    
    g_vals_31 = []
    g_vals_32 = []
    for t_step, delta_step in load_times:
        # delta display in [t+3, t+5]
        d_disp_31 = [(so - si_base) for t, si, so, ded in v31 
                     if t_step + 3 <= t <= t_step + 5
                     for si_base in [sum(ch for ch in data[0][1])]]  # approx
        d_disp_32 = [(so - si_base) for t, si, so, ded in v32_070
                     if t_step + 3 <= t <= t_step + 5
                     for si_base in [sum(ch for ch in data[0][1])]]
        
        # Simpler: just use ded ratio
        ded31_pre = avg_ded_window(v31, t_step - 1, t_step - 0.1)
        ded31_post = avg_ded_window(v31, t_step + 3, t_step + 5)
        ded32_pre = avg_ded_window(v32_070, t_step - 1, t_step - 0.1)
        ded32_post = avg_ded_window(v32_070, t_step + 3, t_step + 5)
        
        # G = delta_output / delta_input ≈ 1 - (ded_post - ded_pre)/delta_step
        g31 = 1.0 - (ded31_post - ded31_pre) / delta_step if abs(delta_step) > 100 else 1.0
        g32 = 1.0 - (ded32_post - ded32_pre) / delta_step if abs(delta_step) > 100 else 1.0
        g_vals_31.append(g31)
        g_vals_32.append(g32)
        report.append(f"{t_step:10.2f} {delta_step:8.0f} {g31:10.3f} {g32:10.3f}")
    
    if g_vals_31:
        g_vals_31.sort()
        g_vals_32.sort()
        report.append(f"{'中位':>10s} {'':>8s} {g_vals_31[len(g_vals_31)//2]:10.3f} {g_vals_32[len(g_vals_32)//2]:10.3f}")
    
    # Section 5: C-clamp invariant
    report.append("\n\n5. C 限幅硬不变量 (out ≤ raw + 0.5%·|raw|)")
    report.append("-" * 80)
    violations_31 = sum(1 for t, si, so, ded in v31 if so > si * 1.005 + 0.01)
    violations_32 = sum(1 for t, si, so, ded in v32_070 if so > si * 1.005 + 0.01)
    report.append(f"v3.1 越界帧：{violations_31}")
    report.append(f"v3.2 越界帧：{violations_32}")
    
    # Section 6: Summary
    report.append("\n\n6. 总结")
    report.append("=" * 80)
    
    # Compare problem region [245, 310]
    ded31_problem = [ded for t, si, so, ded in v31 if 245 <= t <= 310 and si > 5000]
    ded32_problem = [ded for t, si, so, ded in v32_070 if 245 <= t <= 310 and si > 5000]
    avg31_p = sum(ded31_problem) / len(ded31_problem) if ded31_problem else 0
    avg32_p = sum(ded32_problem) / len(ded32_problem) if ded32_problem else 0
    
    report.append(f"问题区段 [245,310]s 受载帧平均 ded：")
    report.append(f"  v3.1: {avg31_p:.1f} ADC")
    report.append(f"  v3.2: {avg32_p:.1f} ADC")
    report.append(f"  改善: {avg32_p - avg31_p:+.1f} ADC")
    report.append("")
    
    ded31_first = [ded for t, si, so, ded in v31 if 10 <= t <= 230 and si > 5000]
    ded32_first = [ded for t, si, so, ded in v32_070 if 10 <= t <= 230 and si > 5000]
    avg31_f = sum(ded31_first) / len(ded31_first) if ded31_first else 0
    avg32_f = sum(ded32_first) / len(ded32_first) if ded32_first else 0
    
    report.append(f"首段保压 [10,230]s 受载帧平均 ded：")
    report.append(f"  v3.1: {avg31_f:.1f} ADC")
    report.append(f"  v3.2: {avg32_f:.1f} ADC")
    report.append(f"  差异: {avg32_f - avg31_f:+.1f} ADC（应接近 0，首段不受影响）")
    
    report_text = "\n".join(report)
    out_path = os.path.join(RESULTS_DIR, "v32_compare_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\n报告已写入：{out_path}")
    print("\n" + report_text)

if __name__ == "__main__":
    main()
