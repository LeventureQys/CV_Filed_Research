# -*- coding: utf-8 -*-
"""T9-A 现场回归复核：恒定负载反复加减，v6 补偿后基线漂移。

数据：temp/算法数据&原始数据/恒定负载下反复加减同一个负载/20260919_092417_single_device_602c03
  device_001_pre_seg0.csv   算法前读数（后端施加与显示同一阈值；此处即"没开算法本该显示的 ADC"）
  device_001_seg000.csv     算法结果（v6，参数集 plan-v1.0 A1+A4+A5a）
  device_001_raw_seg000.csv 原始 ADC 逐帧流（未过阈值）
输出：
  results/t9a_series.csv    时间轴上的总量/偏移/逐通道（降采样 0.1 s 平均）
  results/t9a_plateaus.csv  每个平台的 pre/main 电平与偏移（按滞回自动切段）
  stdout                    概览
"""
import csv
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    *([".."] * 6)))
assert os.path.isfile(os.path.join(ROOT, "CMakeLists.txt")), ROOT
DATA = os.path.join(ROOT, "temp", "算法数据&原始数据", "恒定负载下反复加减同一个负载",
                    "20260919_092417_single_device_602c03")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

NCH = 21


def load(name):
    """返回 (ts[], elapsed[], frame[], values[nch][n])"""
    p = os.path.join(DATA, name)
    with open(p, encoding="utf-8-sig") as fh:
        rows = [r.rstrip("\n") for r in fh]
    di = rows.index("##Data")
    hdr = rows[di + 1].split(",")
    ch_cols = [i for i, h in enumerate(hdr) if h.startswith("ch") or h.startswith("raw_ch")]
    if not ch_cols:
        # 已知缺陷：单设备路径的 algorithm_input(pre) 文件表头只写 timestamp,elapsed,frame_index，
        # 不带通道名，但数据行有 3+21 列 ⇒ 位置回退（末 21 列即通道）。
        ncol = len(rows[di + 2].split(","))
        ch_cols = list(range(ncol - NCH, ncol))
    assert len(ch_cols) == NCH, (name, len(ch_cols))
    ts, el, fr = [], [], []
    vals = [[] for _ in range(NCH)]
    for r in rows[di + 2:]:
        if not r:
            continue
        f = r.split(",")
        ts.append(float(f[0]))
        el.append(float(f[1]))
        fr.append(int(f[2]))
        for k, ci in enumerate(ch_cols):
            vals[k].append(float(f[ci]))
    return ts, el, fr, vals


ts, el, fr, pre = load("device_001_pre_seg0.csv")
ts2, el2, fr2, main = load("device_001_seg000.csv")
ts3, el3, fr3, raw = load("device_001_raw_seg000.csv")
assert len(ts) == len(ts2) == len(ts3), (len(ts), len(ts2), len(ts3))
n = len(ts)

tot_pre = [sum(pre[k][i] for k in range(NCH)) for i in range(n)]
tot_main = [sum(main[k][i] for k in range(NCH)) for i in range(n)]
tot_raw = [sum(raw[k][i] for k in range(NCH)) for i in range(n)]

print(f"frames={n}  span={el[-1]-el[0]:.3f}s  dt_med={sorted(el[i+1]-el[i] for i in range(n-1))[n//2]:.4f}s")
print(f"tot_pre   min={min(tot_pre):.0f} max={max(tot_pre):.0f}")
print(f"tot_main  min={min(tot_main):.0f} max={max(tot_main):.0f}")
print(f"tot_raw   min={min(tot_raw):.0f} max={max(tot_raw):.0f}")

# ── 0.25 s 降采样概览（pre / main / 偏移）──
print("\n t(s)   pre_tot   main_tot   off(main-pre)   off(%) ")
step = 0.25
t0 = el[0]
ti = t0
i = 0
rows = []
while i < n:
    j = i
    while j < n and el[j] < ti + step:
        j += 1
    if j > i:
        a = sum(tot_pre[i:j]) / (j - i)
        b = sum(tot_main[i:j]) / (j - i)
        span = max(a, 1.0)
        rows.append((el[i], a, b, b - a, 100.0 * (b - a) / span))
    i = j
    ti += step
for r in rows:
    print(f"{r[0]:7.2f} {r[1]:9.0f} {r[2]:10.0f} {r[3]:12.0f} {r[4]:9.1f}")

# ── 平台自动切段（对 pre 用滞回：> 中位幅度 50% 视为受载）──
lo = min(tot_pre)
hi = max(tot_pre)
mid = 0.5 * (lo + hi)
hyst = 0.10 * (hi - lo)
state = None
segs = []
start = 0
for i in range(n):
    s = "loaded" if tot_pre[i] > mid + hyst else ("idle" if tot_pre[i] < mid - hyst else state)
    if s != state:
        if state is not None and i - 1 > start:
            segs.append((state, start, i - 1))
        state = s
        start = i
if state is not None:
    segs.append((state, start, n - 1))

with open(os.path.join(OUT, "t9a_plateaus.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["idx", "kind", "t_start_s", "t_end_s", "dur_s",
                "pre_tot_med", "main_tot_med", "off_med", "off_pct_of_load", "raw_tot_med"])
    k_idle = 0
    k_load = 0
    amp = None
    for kind, a, b in segs:
        if b - a < 20:      # <0.2 s 的过渡段丢弃
            continue
        idx = range(a, b + 1)
        tp = sorted(tot_pre[i] for i in idx)[len(idx) // 2]
        tm = sorted(tot_main[i] for i in idx)[len(idx) // 2]
        tr = sorted(tot_raw[i] for i in idx)[len(idx) // 2]
        if kind == "idle":
            k_idle += 1
        else:
            k_load += 1
        if amp is None and kind == "loaded":
            amp = tp - lo
        w.writerow([k_idle if kind == "idle" else f"L{k_load}", kind,
                    f"{el[a]:.3f}", f"{el[b]:.3f}", f"{el[b]-el[a]:.3f}",
                    f"{tp:.1f}", f"{tm:.1f}", f"{tm-tp:.1f}", "", f"{tr:.1f}"])

# 用第一个受载平台定义负载幅度参考
load_amp = None
for kind, a, b in segs:
    if kind == "loaded" and b - a > 50:
        idx = range(a, b + 1)
        load_amp = sorted(tot_pre[i] for i in idx)[len(idx) // 2] - lo
        break
print(f"\nidle_tot(全局最小)={lo:.1f}  load_amp(pre,首个受载平台)={load_amp:.1f}  mid={mid:.1f}")

# 空载平台偏移（以负载幅度归一）
print("\n── 空载平台上的基线偏移（main − pre）──")
print("  idx   t_start  t_end   dur   pre_tot   main_tot    off    off/load_amp")
k = 0
for kind, a, b in segs:
    if b - a < 20 or kind != "idle":
        continue
    k += 1
    idx = range(a, b + 1)
    tp = sorted(tot_pre[i] for i in idx)[len(idx) // 2]
    tm = sorted(tot_main[i] for i in idx)[len(idx) // 2]
    print(f"  #{k:<3d} {el[a]:8.2f} {el[b]:7.2f} {el[b]-el[a]:6.2f}  {tp:8.0f} {tm:9.0f} {tm-tp:7.0f}  "
          f"{100.0*(tm-tp)/load_amp:8.2f}%")

# 逐通道空载偏移（最后一段空载）
last_idle = None
for kind, a, b in segs:
    if kind == "idle" and b - a > 50:
        last_idle = (a, b)
print("\n── 最后一段空载的逐通道偏移（main − pre）──")
if last_idle:
    a, b = last_idle
    print(f"  段: t={el[a]:.2f}~{el[b]:.2f}s")
    for k in range(NCH):
        mp = sorted(pre[k][i] for i in range(a, b + 1))[(b - a) // 2]
        mm = sorted(main[k][i] for i in range(a, b + 1))[(b - a) // 2]
        print(f"   ch{k:<2d} pre={mp:8.1f} main={mm:8.1f} off={mm-mp:8.1f}")

# ── 降采样序列落盘 ──
with open(os.path.join(OUT, "t9a_series.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["t_s", "pre_tot", "main_tot", "raw_tot", "off_main_pre"])
    for r in rows:
        w.writerow([f"{r[0]:.3f}", f"{r[1]:.1f}", f"{r[2]:.1f}",
                    f"{sum(tot_raw[i] for i in range(max(0,int((r[0]-t0)/step)-1), min(n,int((r[0]-t0)/step)+2)))/3:.1f}",
                    f"{r[3]:.1f}"])
print("\nwrote results/t9a_series.csv, results/t9a_plateaus.csv")
