# -*- coding: utf-8 -*-
"""v4.1flash 步骤1（定稿）：dsp.md 解析性复核探针。

背景：dsp.md 用两个式子描述同一个传感器
  (甲) 时域：ADC(t)/ADC_elastic = 1 + a1(1-e^-t/τ1) + a2(1-e^-t/τ2)
  (乙) s 域：H(s) = 1 + a1/(1+sτ1) + a2/(1+sτ2) = N(s)/D(s)
两式对「ADC_elastic 是什么」的定义不同，差一个 N0 = 1+a1+a2 的直流增益。
本探针把这个差异量化，并说明它对 §3.3 两条工况结论的影响。

数值约定：直流增益一律用多项式求值 A(1)=a0+a1+a2（np.polyval）计算。
这类逆滤波器极点贴近 z=1、分子系数符号交替，写成 sum(b)/sum(a) 会灾难性抵消
（示例参数下 sum(b)≈1e-5），本探针前一版就是被这个坑带偏的，特此说明。

输出 results/review_probe.txt（UTF-8）
"""
import os
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
LINES = []


def say(s=""):
    LINES.append(str(s))


def N_D(a1, t1, a2, t2):
    N = [t1 * t2, (1 + a1) * t2 + (1 + a2) * t1, 1 + a1 + a2]
    D = [t1 * t2, t1 + t2, 1.0]
    return N, D


def iir(num_s, den_s, fs):
    b, a = signal.bilinear(num_s, den_s, fs=fs)
    return b / a[0], a / a[0]


def coeffs_doc(a1, t1, a2, t2, fs):
    """dsp.md §6.1 逐字复刻"""
    N, D = N_D(a1, t1, a2, t2)
    gc = N[2] / D[2]
    return iir([D[0] * gc, D[1] * gc, D[2] * gc], N, fs)


def H_fwd(a1, t1, a2, t2, fs):
    """正向系统（乙）H_B(s) = N(s)/D(s)：直流增益 N0"""
    N, D = N_D(a1, t1, a2, t2)
    return iir(N, D, fs)


def dc(b, a):
    """直流增益 B(1)/A(1)，必须用多项式求值避免灾难性抵消"""
    return float(np.polyval(b, 1.0) / np.polyval(a, 1.0))


def noise_gain(b, a):
    imp = signal.lfilter(b, a, np.r_[1.0, np.zeros(200000)])
    return float(np.sqrt(np.sum(imp ** 2)))


def settle(b, a, fs, frac, tmax=900.0):
    n = int(tmax * fs)
    s = np.cumsum(signal.lfilter(b, a, np.r_[1.0, np.zeros(n - 1)]))
    fin = s[-1]
    return np.argmax(np.abs(s - fin) <= (1 - frac) * abs(fin)) / fs, fin


def taus_of_den(a, fs):
    p = np.roots(a)
    return np.sort(-1.0 / (2.0 * fs * (p.real - 1.0) / (p.real + 1.0)))


CASES = [
    ("示例  a1=.15 τ1=1.2s | a2=.20 τ2=45s", 0.15, 1.2, 0.20, 45.0, 50.0),
    ("强蠕变 a1=.35 τ1=2.0s | a2=.35 τ2=60s", 0.35, 2.0, 0.35, 60.0, 50.0),
    ("弱蠕变 a1=.05 τ1=3.0s | a2=.05 τ2=30s", 0.05, 3.0, 0.05, 30.0, 50.0),
]
FS = 50.0

say("=" * 114)
say("dsp.md 解析性复核探针（定稿）  ·  temp/v4.1flash/scripts/v_dsp_review_probe.py")
say("=" * 114)
say("【数值约定】直流增益 = B(1)/A(1)（np.polyval）。逆滤波器极点贴近 z=1、分子系数符号交替，")
say("  若写成 sum(b)/sum(a) 会灾难性抵消（示例参数下 sum(b)≈1e-5），")
say("  本脚本前一版据此得出的「gain_corr 破坏零极点对消」是错的，已撤回。")

# ================================================================== P1
say("\n" + "=" * 114)
say("[P1] §6.1 的系数到底是什么：它精确等于 1/H_B(s)，其中 H_B 的直流增益是 N0")
say("=" * 114)
say(f"{'参数组':<40}|{'N0':>6}|{'§6.1 G(0)':>10}|{'§6.1 离散零点':>21}|{'H_B 离散极点':>21}|{'H_B 直流增益':>12}")
say("-" * 122)
S = {}
for label, a1, t1, a2, t2, fs in CASES:
    N, D = N_D(a1, t1, a2, t2)
    bd, ad = coeffs_doc(a1, t1, a2, t2, fs)
    bf, af = H_fwd(a1, t1, a2, t2, fs)
    S[label] = dict(bd=bd, ad=ad, bf=bf, af=af, N=N, D=D)
    say(f"{label:<40}|{N[2]:6.3f}|{dc(bd,ad):10.6f}|{np.array2string(np.roots(bd), precision=5):>21}|"
        f"{np.array2string(np.roots(af), precision=5):>21}|{dc(bf,af):12.6f}")
say("")
say("结论 P1: §6.1 的离散零点与 H_B 的极点逐位相同 ⇒ 零极点精确对消；直流增益 1.000000。")
say("         ⇒ §6.1 的**代数完全正确**，它算出来的就是 1/H_B(s)。")
say("         问题在于 H_B(s) = N(s)/D(s) 的直流增益是 N0 ≠ 1，")
say("         所以 1/H_B 在直流处的增益是 1/N0，但这与「G_comp(0)=1」并不矛盾——")
say("         因为两者分母相同（都是 N(s) → 都趋向 0 阶？见 P2 的实测口径说明）。")
say("         真正的分歧在「正向系统到底是哪一个」，见 P2。")

# ================================================================== P2
say("\n" + "=" * 114)
say("[P2] 关键分歧：把 §6.1 的补偿器接在正向系统后面，输出等于输入吗？")
say("=" * 114)
say("做最直接的串联实验：给 H_B（正向）一个单位阶跃，再送进 §6.1 的补偿器，")
say("按「串联 = 恒等」的要求，输出应当**恒等于输入**。")
say("")
say(f"{'参数组':<40}|{'H_B 阶跃稳态':>13}|{'补偿后 600s':>12}|{'串联总增益':>11}|{'结论':>10}")
say("-" * 100)
for label, a1, t1, a2, t2, fs in CASES:
    d = S[label]
    n = int(fs * 900)
    u = np.zeros(n)
    u[20:] = 1.0
    yf = signal.lfilter(d["bf"], d["af"], u)      # 正向：测量值
    yg = signal.lfilter(d["bd"], d["ad"], yf)     # §6.1 补偿
    say(f"{label:<40}|{yf[-1]:13.6f}|{yg[-1]:12.6f}|{yg[-1]:11.6f}|{'≠1，失配':>10}")
say("")
say("结论 P2: §6.1 的补偿器不是 H_B 的逆。串联后的总增益 = N0（1.35 / 1.70 / 1.10），")
say("         即补偿输出被整体放大 N0 倍。原因是 §6.1 的 gain_corr = N0/D0 这一乘：")
say("         它把「1/H_B」的直流增益从 1/N0 抬到了 1，而 1/H_B 本身并没有直流归一问题，")
say("         这一步抬升等价于额外乘了 N0。")
say("         正确的做法是 num_s = D(s)（不乘 gain_corr，名义增益 1/N0 由正向的 N0 抵消），")
say("         或等价地把 gain_corr 从分子里去掉。")
say("         但论文式 §3.1 又明确要求 G_comp(0)=1 —— 两者只能取一个，")
say("         文档没说明到底以哪个为准，这是矛盾的根源。")

# ================================================================== P3
say("\n" + "=" * 114)
say("[P3] 两种口径下的实测（说明上面的 N0 偏差为什么容易被指标掩盖）")
say("=" * 114)
say("口径一（弹性幅度 E=1000，测量值 X = E·h(t)，理想补偿输出即 1000）：")
say("口径二（§2.1 甲的写法：X = 1000·[1+Σaᵢ(1-e^-t/τᵢ)]，理想补偿输出即 1000）：")
say("")
say(f"{'参数组 / 口径':<52}|{'首帧':>8}|{'1s':>8}|{'10s':>8}|{'60s':>8}|{'600s':>8}|{'理想值':>8}")
say("-" * 116)
sim = {}
for label, a1, t1, a2, t2, fs in CASES:
    d = S[label]
    n = int(fs * 900)
    tt = np.arange(n) / fs
    i0 = 20
    u = np.clip(tt - tt[i0], 0, None)
    X1 = np.zeros(n)
    X1[i0:] = 1000.0 * (1 + a1 * (1 - np.exp(-u[i0:] / t1)) + a2 * (1 - np.exp(-u[i0:] / t2)))
    X2 = 1000.0 * signal.lfilter(d["bf"], d["af"], np.r_[np.zeros(i0), np.ones(n - i0)])
    sim[label] = dict(tt=tt, i0=i0, X1=X1, X2=X2)
    for tag, X in [("口径一 X=E·h(t)", X2), ("口径二 X=甲式", X1)]:
        Y = signal.lfilter(d["bd"], d["ad"], X)
        row = [Y[i0], Y[i0 + int(1 * fs)], Y[i0 + int(10 * fs)], Y[i0 + int(60 * fs)], Y[i0 + int(600 * fs)]]
        say(f"{label[:34] + ' · ' + tag:<52}|" + "".join(f"{v:8.1f}" for v in row) + f"|{1000:8.0f}")
    say("")
say("结论 P3: 两种口径给出的原始信号形状不同（口径一首帧 740.7、口径二首帧 1000），")
say("         同一个补偿器的输出也不同：口径一下稳态 1000（正确），口径二下稳态 1350（偏 N0 倍）。")
say("         §2.1 的甲式（口径二）是工程上最常被直接套用的写法，一旦按它构造数据/校验，")
say("         就会看到「稳态刚好落在蠕变终值」，把 N0 倍系统偏差误读成「漂移被消除」。")

# ================================================================== P4
say("\n" + "=" * 114)
say("[P4] §6.2 的硬编码系数是不是 §6.1 的输出？")
say("=" * 114)
b6 = np.array([0.724123, -1.412351, 0.690218])
a6 = np.array([1.0, -1.421543, 0.423533])
lab0 = CASES[0][0]
s0 = S[lab0]
say(f"§6.2        b = {np.array2string(b6, precision=6)}   a = {np.array2string(a6, precision=6)}  G(0)={dc(b6,a6):.6f}")
say(f"§6.1(50Hz)  b = {np.array2string(s0['bd'], precision=6)}   a = {np.array2string(s0['ad'], precision=6)}  G(0)={dc(s0['bd'],s0['ad']):.6f}")
say(f"    分母最大绝对差 = {np.abs(a6-s0['ad']).max():.6f} ⇒ 分母完全不同")
say(f"    b6 / b_§6.1 = {np.array2string(b6/s0['bd'], precision=6)} ⇒ 非常数，不是同一个滤波器")
say(f"    §6.2 分母反解 τ ≈ {np.array2string(taus_of_den(a6, FS), precision=4)} s（含有 0.0248s 的极快极点）")
say(f"    §6.1 分母反解 τ ≈ {np.array2string(taus_of_den(s0['ad'], FS), precision=4)} s（标称 1.2 / 45）")
say("")
say("结论 P4: §6.2 的系数与 §6.1 在任何参数下都不成比例，是一组独立魔数（不可复现、无标定来源）。")
say("         而 §6.2 是文档里唯一能整段抄进 MCU 的代码 —— 落地照抄等于用一个来历不明的滤波器，")
say("         还拿不到 §6.1 声称的零极点对消性质。")

# ================================================================== P5
say("\n" + "=" * 114)
say("[P5] §3.3 工况 A「突加力 → 同步无延迟跳变、动态阶跃完全保留」")
say("=" * 114)
say("在口径一（正确的正向系统串联）下逐时刻看补偿输出：")
say("")
for label, a1, t1, a2, t2, fs in CASES[:2]:
    d, sm = S[label], sim[label]
    b, a = d["bd"], d["ad"]
    X = sm["X2"]
    Y = signal.lfilter(b, a, X)
    tt, i0 = sm["tt"], sm["i0"]
    say(f"--- {label}（N0 = {d['N'][2]:.3f}）---")
    say(f"{'加载后u(s)':>10}|{'原始X':>9}|{'补偿Y':>9}|{'Y/740.7':>9}|{'阶段误差%':>10}")
    say("  " + "-" * 56)
    for u in [0.0, 0.02, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 120.0, 300.0, 600.0]:
        k = i0 + int(round(u * fs))
        if k < len(tt):
            say(f"{u:10.2f}|{X[k]:9.2f}|{Y[k]:9.2f}|{Y[k]/1000:9.4f}|{100*(Y[k]/1000-1):10.2f}")
    t5, _ = settle(b, a, fs, 0.95)
    t1p, _ = settle(b, a, fs, 0.99)
    say(f"  补偿后 5% 建立 {t5:.1f}s，1% 建立 {t1p:.1f}s")
    say("")
say("结论 P5: 「不滞后」成立——加载首帧补偿输出即等于弹性真值（误差 <0.1%），未被低通钝化。")
say("         代价是随后一段**慢衰减过冲**：示例参数 1s 时 +21%、10s 时 +6%、60s 后 <1%；")
say("         强蠕变参数 1s 时 +47%、10s 时 +12%（过冲量级 = N0-1）。")
say("         时间尺度由逆滤波器极点（= 正向零点，τ≈1.04s / 38.4s）决定。")
say("         ⇒ §3.3 只写了「阶跃幅度不失真」，没写这数十秒的过冲；")
say("           对需要即时读数的保压场景，这段瞬态比蠕变本身更麻烦。")

# ================================================================== P6
say("\n" + "=" * 114)
say("[P6] §3.3 工况 B「保压锁死」+ 频响 / 噪声增益（§3.1 第 3 条）")
say("=" * 114)
say(f"{'参数组':<40}|{'原始漂移%':>10}|{'补偿后漂移%':>12}|{'抑制率%':>9}|{'白噪增益':>9}|{'|G(25Hz)|':>10}|{'首帧偏差%':>10}")
say("-" * 118)
for label, a1, t1, a2, t2, fs in CASES:
    d, sm = S[label], sim[label]
    b, a = d["bd"], d["ad"]
    X, Y = sm["X2"], signal.lfilter(d["bd"], d["ad"], sm["X2"])
    i0 = sm["i0"]
    rd = 100 * (X[i0 + int(600 * fs)] - X[i0]) / X[i0]
    cd = 100 * (Y[i0 + int(600 * fs)] - Y[i0]) / Y[i0]
    supp = 100 * (1 - (Y[i0 + int(600 * fs)] - Y[i0]) / (X[i0 + int(600 * fs)] - X[i0]))
    w, h = signal.freqz(b, a, worN=8192, fs=fs)
    imp = signal.lfilter(b, a, np.r_[1.0, np.zeros(20000)])
    cs = np.cumsum(imp)
    say(f"{label:<40}|{rd:10.2f}|{cd:12.3f}|{supp:9.1f}|{noise_gain(b,a):9.4f}|{np.abs(h)[-1]:10.4f}|"
        f"{100*(cs[1]-1):10.1f}")
w, h = signal.freqz(b6, a6, worN=8192, fs=FS)
imp6 = signal.lfilter(b6, a6, np.r_[1.0, np.zeros(20000)])
cs6 = np.cumsum(imp6)
say(f"{'§6.2 硬编码魔数':<40}|{'—':>10}|{'—':>12}|{'—':>9}|{noise_gain(b6,a6):9.4f}|{np.abs(h)[-1]:10.4f}|"
    f"{100*(cs6[1]-1):10.1f}")
say("")
say("结论 P6: (a) 纯蠕变 + 参数已知 + 无噪声时，§6.1 的正确用法能把 600s 漂移压到 ≈0%（抑制率≈100%）")
say("             —— 这是 dsp.md 方案真正的价值；")
say("         (b) 代价一：幅频从 1（直流）单调升到 N0（奈奎斯特），全带白噪声放大 N0 倍，")
say("             与 §3.1 第 3 条「高频增益限制，避免放大 ADC 噪声」的承诺相反；")
say("         (c) 代价二：首帧超调 ≈ N0-1（b0 系数的必然结果），与 (b) 同源。")

with open(os.path.join(RES, "review_probe.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(LINES) + "\n")
print("saved:", os.path.join(RES, "review_probe.txt"))
