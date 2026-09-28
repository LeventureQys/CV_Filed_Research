# -*- coding: utf-8 -*-
"""src/domain/creep/change_point_creep_compensator.cpp 的逐行转写数值自检。

用途：在不构建任何 C++ 测试目标的前提下，验证新算法的「估计器数学」与「状态机行为」：
  S1 估计器：固定 tau 下 a 闭式解 + log tau 粗搜 + 三分细化，能否在合成对数蠕变数据上
             复原真值 (a, tau)，并与 scipy curve_fit(LM) 的结果对照；
  S2 端到端：合成「空载 → 加载(长保压) → 负载内阶跃 → 卸载 → 再加载」的 100Hz 序列，
             检查漂移抑制率、阶跃是否被当蠕变吃掉、卸载是否直通、Σ 补偿量一致性。
注意：这是对同一套数学的独立复算，不是项目单元测试（未构建/未运行任何 test_* 目标）。
用法: python temp/creep_check/check_estimator.py
"""
import sys
import numpy as np
from scipy.optimize import curve_fit

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── 与 C++ 头文件常量逐一对应 ──
SETTLE_S = 3.0
REFIT_S = 5.0
MIN_DATA_S = 10.0
STEP_MULT = 8.0
MIN_STEP_GAP_S = 1.5
TAU_INIT_S = 20.0
TAU_SMOOTH_S = 0.30
TAU_FAST_EMA_S = 0.10
FAST_EMA_ALPHA_MAX = 0.5
ENTER_RATIO = 0.30
EXIT_RATIO = 0.20
ABS_FLOOR_FRAC = 0.02
ENTER_SIGMA = 4.0
ENTER_PERSIST_S = 1.0
IDLE_REF_TAU_S = 2.0
PEAK_LEAK_RATE = 0.02
BASELINE_WIN_S = 2.0
NOISE_FLOOR_FRAC = 0.005
NOISE_CAP_FRAC = 0.03
TAU_MIN_S = 0.5
TAU_MAX_S = 5000.0
A_MAX_RATIO = 5.0
CREEP_LO_FRAC = -0.5
CREEP_HI_FRAC = 1.5
MIN_FIT_SAMPLES = 300
FIT_HIST_N = 1024


class Compensator(object):
    """ChangePointCreepCompensator 的 Python 转写（状态机与常数保持一致）。"""

    def __init__(self, n):
        self.n = n
        self.first = True
        self.last_ts = 0.0
        self.frames = 0
        self.sm = 0.0
        self.fast = 0.0
        self.idle_ref = 0.0
        self.peak_ref = 0.0
        self.idle_dev = 0.0
        self.in_load = False
        self.epoch_ts = 0.0
        self.enter_hold = 0.0
        self.anchor_ts = 0.0
        self.bl_ready = False
        self.established = False
        self.bl_mean = 0.0
        self.bl_m2 = 0.0
        self.bl_n = 0
        self.ch_acc = np.zeros(n)
        self.A = np.zeros(n)
        self.a_sum = 0.0
        self.y0 = 0.0
        self.noise = 0.0
        self.a_est = 0.0
        self.tau_est = TAU_INIT_S
        self.last_fit_ts = 0.0
        self.fit_t = np.zeros(FIT_HIST_N)
        self.fit_v = np.zeros(FIT_HIST_N)
        self.fit_n = 0
        self.fit_stride = 1
        self.last_push = 0
        self.carry = np.zeros(self.n)
        self.carry_total = 0.0
        self.last_ded = np.zeros(self.n)

    # ── 内部：与 C++ 同名逻辑 ──
    def _reset_model(self):
        self.bl_ready = False
        self.established = False
        self.bl_mean = 0.0
        self.bl_m2 = 0.0
        self.bl_n = 0
        self.y0 = 0.0
        self.noise = 0.0
        self.a_est = 0.0
        self.tau_est = TAU_INIT_S
        self.fit_n = 0
        self.fit_stride = 1
        self.last_push = self.frames
        self.a_sum = 0.0
        self.ch_acc[:] = 0.0
        self.A[:] = 0.0

    def _begin_epoch(self, ts, total, carry_over):
        self.carry[:] = 0.0
        self.carry_total = 0.0
        if carry_over:
            self.carry = self.last_ded.copy()
            self.carry_total = float(self.carry.sum())
        self.in_load = True
        self.epoch_ts = ts
        self.anchor_ts = ts + SETTLE_S
        self.fast = total
        self.last_fit_ts = ts
        self.enter_hold = ENTER_PERSIST_S
        self._reset_model()

    def _push(self, ts, total):
        if self.frames - self.last_push < self.fit_stride:
            return
        self.last_push = self.frames
        if self.fit_n >= FIT_HIST_N:
            half = FIT_HIST_N // 2
            self.fit_t[:half] = self.fit_t[:FIT_HIST_N:2][:half]
            self.fit_v[:half] = self.fit_v[:FIT_HIST_N:2][:half]
            self.fit_n = half
            self.fit_stride *= 2
        self.fit_t[self.fit_n] = ts - self.anchor_ts
        self.fit_v[self.fit_n] = total
        self.fit_n += 1

    def _sse(self, tau):
        u = np.log1p(self.fit_t[:self.fit_n] / tau)
        d = self.fit_v[:self.fit_n] - self.y0
        sdd = float(np.dot(d, d))
        sud = float(np.dot(u, d))
        suu = float(np.dot(u, u))
        if suu <= 0.0:
            return sdd
        a = float(np.clip(sud / suu, 0.0, A_MAX_RATIO * abs(self.y0)))
        return sdd - 2.0 * a * sud + a * a * suu

    def _fit(self):
        if self.fit_n < MIN_FIT_SAMPLES or not abs(self.y0) > 0.0:
            return False
        lo, hi = np.log(TAU_MIN_S), np.log(TAU_MAX_S)
        grid = 40
        dstep = (hi - lo) / grid
        best_lx, best_sse = lo, self._sse(np.exp(lo))
        for k in range(1, grid + 1):
            lx = lo + dstep * k
            sse = self._sse(np.exp(lx))
            if sse < best_sse:
                best_lx, best_sse = lx, sse
        a_l = max(best_lx - dstep, lo)
        b_l = min(best_lx + dstep, hi)
        it = 0
        while it < 20 and (b_l - a_l) > 1e-4:
            m1 = a_l + (b_l - a_l) / 3.0
            m2 = b_l - (b_l - a_l) / 3.0
            if self._sse(np.exp(m1)) <= self._sse(np.exp(m2)):
                b_l = m2
            else:
                a_l = m1
            it += 1
        tau = float(np.exp(0.5 * (a_l + b_l)))
        u = np.log1p(self.fit_t[:self.fit_n] / tau)
        d = self.fit_v[:self.fit_n] - self.y0
        sud, suu = float(np.dot(u, d)), float(np.dot(u, u))
        a = sud / suu if suu > 0.0 else 0.0
        a = float(np.clip(a, 0.0, A_MAX_RATIO * abs(self.y0)))
        self.tau_est = float(np.clip(tau, TAU_MIN_S, TAU_MAX_S))
        self.a_est = a if a > 0.0 else 0.0
        return self.a_est > 0.0

    # ── 主流程 ──
    def process(self, ts, v):
        v = v  # 就地修改
        n = v.size
        assert n == self.n
        total = float(v.sum())
        if not np.isfinite(total):
            return v
        self.frames += 1
        dt = 0.0
        if self.first:
            self.first = False
            self.last_ts = ts
            self.sm = total
            self.fast = total
            self.idle_ref = total
            self.peak_ref = total
        else:
            dt = ts - self.last_ts
            self.last_ts = ts
            if not dt > 0.0:
                dt = 0.0
            if dt > 0.1:
                dt = 0.1
            if dt > 0.0:
                self.sm += (dt / TAU_SMOOTH_S) * (total - self.sm)
                alpha = min(dt / TAU_FAST_EMA_S, FAST_EMA_ALPHA_MAX)
                self.fast += alpha * (total - self.fast)
            if not self.in_load:
                if dt > 0.0:
                    self.idle_ref += (dt / IDLE_REF_TAU_S) * (self.sm - self.idle_ref)
                    # 噪声尺度只在"确定空载"时估计(与 C++ 同门控)
                    if not self.enter_hold > 0.0:
                        self.idle_dev += (dt / IDLE_REF_TAU_S) * (abs(self.sm - self.idle_ref) - self.idle_dev)
                if self.sm > self.peak_ref:
                    self.peak_ref = self.sm
                elif dt > 0.0:
                    rng = max(self.peak_ref - self.idle_ref, 0.0)
                    if rng > 0.0:
                        self.peak_ref = max(self.peak_ref - PEAK_LEAK_RATE * dt * rng,
                                            self.idle_ref)
            elif self.sm > self.peak_ref:
                self.peak_ref = self.sm

        rng = max(self.peak_ref - self.idle_ref, 0.0)
        scale = max(abs(self.peak_ref), abs(self.idle_ref))
        floor_v = max(ABS_FLOOR_FRAC * scale, ENTER_SIGMA * self.idle_dev)
        thr_enter = self.idle_ref + max(ENTER_RATIO * rng, floor_v)
        thr_exit = self.idle_ref + max(EXIT_RATIO * rng, floor_v)
        gap_ok = (ts - self.epoch_ts) > MIN_STEP_GAP_S

        if not self.in_load:
            if self.sm > thr_enter:
                self.enter_hold += dt
                if self.enter_hold >= ENTER_PERSIST_S:
                    self._begin_epoch(ts, total, False)
            else:
                self.enter_hold = 0.0
        elif self.sm < thr_exit:
            self.in_load = False
            self.enter_hold = 0.0
            self.carry[:] = 0.0
            self.carry_total = 0.0
            self.last_ded[:] = 0.0
            self._reset_model()
        elif self.bl_ready and not self.established and gap_ok and self.sm > thr_enter:
            self._begin_epoch(ts, total, False)

        if not self.in_load:
            return v
        has_carry = self.carry_total != 0.0
        if ts < self.anchor_ts:
            if has_carry:
                v -= self.carry
            return v

        total_c = total - self.carry_total
        self._push(ts, total_c)

        if not self.bl_ready:
            d = total_c - self.bl_mean
            self.bl_n += 1
            self.bl_mean += d / self.bl_n
            self.bl_m2 += d * (total_c - self.bl_mean)
            self.ch_acc += v
            self.ch_acc -= self.carry
            if ts < self.anchor_ts + BASELINE_WIN_S:
                if has_carry:
                    v -= self.carry
                return v
            # CaptureBaseline
            self.y0 = self.bl_mean if self.bl_n > 0 else 0.0
            sd = np.sqrt(max(self.bl_m2 / (self.bl_n - 1), 0.0)) if self.bl_n > 1 else 0.0
            lo = NOISE_FLOOR_FRAC * abs(self.y0)
            hi = NOISE_CAP_FRAC * abs(self.y0)
            self.noise = min(max(sd, lo), hi)
            if not self.noise > 0.0:
                self.noise = NOISE_FLOOR_FRAC * max(abs(self.peak_ref), abs(self.idle_ref))
            self.A = (self.ch_acc / self.bl_n).clip(min=0.0) if self.bl_n > 0 else np.zeros(n)
            self.a_sum = float(self.A.sum())
            rng_c = max(self.peak_ref - self.idle_ref, 0.0)
            scale_c = max(abs(self.peak_ref), abs(self.idle_ref))
            thr_c = self.idle_ref + max(ENTER_RATIO * rng_c, ABS_FLOOR_FRAC * scale_c)
            self.established = (self.y0 > thr_c) and (self.a_sum > 0.0)
            self.bl_ready = True
            self.last_fit_ts = self.anchor_ts

        if abs(total - self.fast) > STEP_MULT * self.noise and gap_ok:
            carry_over = total > thr_exit
            self._begin_epoch(ts, total, carry_over)
            if carry_over:
                v -= self.carry
            else:
                self.last_ded[:] = 0.0
            return v

        t_since = ts - self.anchor_ts
        if self.established and ts - self.last_fit_ts >= REFIT_S and t_since >= MIN_DATA_S and self.fit_n >= MIN_FIT_SAMPLES:
            self._fit()
            self.last_fit_ts = ts

        drift_new = 0.0
        if self.established and self.a_est > 0.0 and t_since > 0.0:
            drift_new = self.a_est * np.log1p(t_since / self.tau_est)
        if not has_carry and not drift_new > 0.0:
            self.last_ded[:] = 0.0
            return v
        for i in range(n):
            ded = self.carry[i] if has_carry else 0.0
            if drift_new > 0.0 and self.a_sum > 0.0:
                ded += self.A[i] / self.a_sum * drift_new
            ded = float(np.clip(ded, CREEP_LO_FRAC * self.A[i], CREEP_HI_FRAC * self.A[i]))
            v[i] -= ded
            self.last_ded[i] = ded
        return v


# ══════════════════════════════════════════════════════════════════
# S1 估计器精度：与 scipy curve_fit(LM) 对照
# ══════════════════════════════════════════════════════════════════
def s1_estimator():
    rng = np.random.default_rng(20260917)
    y0_true = 100000.0
    a_true, tau_true = 9000.0, 20.0
    t = np.arange(0.0, 120.0, 0.01)
    clean = y0_true + a_true * np.log1p(t / tau_true)
    sig = 200.0
    y = clean + rng.normal(0.0, sig, t.size)

    c = Compensator(1)
    c.y0 = y0_true
    c.anchor_ts = 0.0
    for i in range(t.size):                # 走真实抽稀路径(1024 定长缓冲满载二取一)
        c.frames += 1
        c._push(t[i], y[i])
    c._fit()

    def model(tt, a, tau):
        return y0_true + a * np.log1p(tt / np.maximum(tau, 0.01))

    ts_ds = c.fit_t[:c.fit_n].copy()       # curve_fit 用同一份抽稀样本，保证可比
    ys_ds = c.fit_v[:c.fit_n].copy()
    popt, _ = curve_fit(model, ts_ds, ys_ds, p0=[1.0, 5.0], maxfev=3000,
                        bounds=([0, 0.5], [abs(y0_true) * 5, 5000]))
    print("[S1] 估计器(网格+三分+闭式 a)  a=%9.1f tau=%7.3f  |  真值 a=%9.1f tau=%7.3f"
          % (c.a_est, c.tau_est, a_true, tau_true))
    print("[S1] curve_fit(LM) 对照          a=%9.1f tau=%7.3f" % (popt[0], popt[1]))
    pred_ours = y0_true + c.a_est * np.log1p(ts_ds / c.tau_est)
    pred_lm = model(ts_ds, *popt)
    base = y0_true + a_true * np.log1p(ts_ds / tau_true)
    print("[S1] 拟合残差 RMSE: ours=%.2f  LM=%.2f  |  对真值 RMSE: ours=%.2f  LM=%.2f"
          % (np.sqrt(np.mean((pred_ours - ys_ds) ** 2)),
             np.sqrt(np.mean((pred_lm - ys_ds) ** 2)),
             np.sqrt(np.mean((pred_ours - base) ** 2)),
             np.sqrt(np.mean((pred_lm - base) ** 2))))
    return c


# ══════════════════════════════════════════════════════════════════
# S2 端到端：空载 → 加载 → 负载内阶跃 → 卸载 → 再加载
# ══════════════════════════════════════════════════════════════════
def s2_end_to_end(seed=7):
    fs = 100.0
    dt = 1.0 / fs
    duration = 330.0
    n_frames = int(duration * fs)
    w = np.array([0.5, 0.3, 0.2])          # 三通道负载占比(模拟逐通道基准 A_i)

    # 负载段定义: (起点s, 终点s, 电平)
    segments = [(15.0, 120.0, 100000.0), (120.0, 200.0, 160000.0),
                (230.0, 300.0, 100000.0)]
    a_true, tau_true = 9000.0, 20.0

    def true_level(t):
        for (t0, t1, lv) in segments:
            if t0 <= t < t1:
                return lv, (t - t0)
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    comp = Compensator(3)
    rows = []
    for k in range(n_frames):
        ts = k * dt
        lv, t_load = true_level(ts)
        creep = a_true * np.log1p(t_load / tau_true) if lv > 0 else 0.0
        ch_true = (lv + creep) * w
        raw = ch_true + rng.normal(0.0, 30.0, 3)
        v = raw.copy()
        comp.process(ts, v)
        rows.append((ts, raw.sum(), v.sum(), lv, ch_true.sum()))

    arr = np.array(rows)
    ts, raw_tot, comp_tot, level, truth = arr.T

    def window(t0, t1):
        m = (ts >= t0) & (ts < t1)
        return m

    print("\n[S2] 漂移抑制(保压末段误差, 相对该段真实稳定电平):")
    for (t0, t1, lv) in segments:
        m = window(t1 - 15.0, t1 - 1.0)
        raw_err = (raw_tot[m].mean() - lv) / lv * 100.0
        comp_err = (comp_tot[m].mean() - lv) / lv * 100.0
        print("  段 %6.1f~%6.1fs 电平 %8.0f | 原始偏差 %+7.2f%% → 补偿后 %+7.2f%%"
              % (t0, t1, lv, raw_err, comp_err))

    print("[S2] 负载内阶跃(120s 100k→160k)跟随:")
    for (t0, t1) in [(122.0, 126.0), (130.0, 140.0), (150.0, 195.0)]:
        m = window(t0, t1)
        print("  %6.1f~%6.1fs 显示/真实 = %6.3f (原始 %6.3f)"
              % (t0, t1, comp_tot[m].mean() / 160000.0, raw_tot[m].mean() / 160000.0))

    print("[S2] 卸载/再加载与直通性:")
    for (label, t0, t1) in [("卸载段 205~228s", 205.0, 228.0),
                            ("空载段 2~14s", 2.0, 14.0),
                            ("再加载前 3s 稳定期 230~233s", 230.0, 233.0)]:
        m = window(t0, t1)
        diff = np.abs(comp_tot[m] - raw_tot[m]).max()
        print("  %-28s 显示与原始最大差 = %.6f" % (label, diff))
    m = window(285.0, 299.0)
    print("  再加载保压末段(285~299s) 误差: 原始 %+6.2f%% → 补偿后 %+6.2f%%"
          % ((raw_tot[m].mean() - 100000.0) / 100000.0 * 100.0,
             (comp_tot[m].mean() - 100000.0) / 100000.0 * 100.0))

    # Σ 补偿量一致性 + 逐通道非负性抽检
    m = window(60.0, 200.0)
    print("[S2] 通道数变化/NaN 检查: comp 有限值=%s, 最小值=%.1f"
          % (bool(np.isfinite(comp_tot).all()), comp_tot.min()))

    # 与"原始"逐帧对比的纯算法跳变(应无剧烈跳变)
    d_comp = np.diff(comp_tot)
    d_raw = np.diff(raw_tot)
    excess = np.abs(d_comp - d_raw)
    print("[S2] 单帧 |Δ显示−Δ原始| 最大值 = %.1f (占电平 %.3f%%)"
          % (excess.max(), excess.max() / 100000.0 * 100.0))
    top = np.argsort(excess)[-5:][::-1]
    for idx in top:
        print("      t=%6.2f raw %9.0f→%9.0f  comp %9.0f→%9.0f  excess %8.0f"
              % (ts[idx], raw_tot[idx], raw_tot[idx + 1], comp_tot[idx], comp_tot[idx + 1],
                 excess[idx]))
    print("[S2] 直通帧占比(显示==原始) = %.1f%%"
          % (float(np.mean(np.abs(comp_tot - raw_tot) < 1e-9)) * 100.0))
    return arr


def s3_channel_consistency():
    """Σ 补偿 = drift_total（按通道基准占比分摊的总量一致性）与逐通道限幅。"""
    comp = Compensator(3)
    comp.y0 = 100000.0
    comp.noise = 500.0
    comp.A = np.array([50000.0, 30000.0, 20000.0])
    comp.a_sum = float(comp.A.sum())
    comp.a_est = 9000.0
    comp.tau_est = 20.0
    comp.anchor_ts = 0.0
    comp.in_load = True
    comp.bl_ready = True
    comp.established = True
    comp.last_ts = 0.0
    comp.first = False
    comp.last_push = -10
    comp.fast = float(comp.A.sum())          # 与当前电平一致 ⇒ 不触发阶跃分支
    v = comp.A.copy()
    ts = 60.0
    before = v.sum()
    comp.process(ts, v)
    drift = comp.a_est * np.log1p(ts / comp.tau_est)
    print("\n[S3] 分摊一致性: Σ扣除 = %.3f, drift_total = %.3f, 差 = %.3e"
          % (before - v.sum(), drift, abs((before - v.sum()) - drift)))
    print("[S3] 逐通道扣除占比 = %s (期望 %s)"
          % (np.round((comp.A - v) / comp.A, 4), np.round(np.full(3, drift / comp.a_sum), 4)))


def s4_continuous_creep(seed=11):
    """物理蠕变在变载时不重置(叠加)时的永久偏置探针：用于检验"新段基准是否吸收旧蠕变"。"""
    fs = 100.0
    dt = 1.0 / fs
    duration = 210.0
    n_frames = int(duration * fs)
    w = np.array([0.5, 0.3, 0.2])
    t_first_load = 15.0
    a_true, tau_true = 9000.0, 20.0
    step_t = 120.0

    def level(t):
        if t < t_first_load:
            return 0.0
        return 100000.0 if t < step_t else 160000.0

    rng = np.random.default_rng(seed)
    comp = Compensator(3)
    rows = []
    for k in range(n_frames):
        ts = k * dt
        lv = level(ts)
        # 蠕变按"载荷比例、状态连续"建模: creep(t) = (level/100k)·a·ln(1+(t−t_load)/τ)
        creep = (lv / 100000.0) * a_true * np.log1p(max(ts - t_first_load, 0.0) / tau_true) if lv > 0 else 0.0
        raw = (lv + creep) * w + rng.normal(0.0, 30.0, 3)
        v = raw.copy()
        comp.process(ts, v)
        rows.append((ts, raw.sum(), v.sum(), lv))
    arr = np.array(rows)
    ts, raw_tot, comp_tot, lv = arr.T
    print("\n[S4] 蠕变状态连续(叠加)工况：变载后是否留下永久正偏")
    print("  变载前 100~119s: 原始偏差 %+6.2f%% → 补偿后 %+6.2f%%"
          % ((raw_tot[(ts >= 100) & (ts < 119)].mean() - 100000.0) / 1000.0,
             (comp_tot[(ts >= 100) & (ts < 119)].mean() - 100000.0) / 1000.0))
    for t0, t1 in [(130.0, 145.0), (160.0, 180.0), (190.0, 205.0)]:
        m = (ts >= t0) & (ts < t1)
        print("  变载后 %5.0f~%5.0fs: 原始偏差 %+6.2f%% → 补偿后 %+6.2f%%"
              % (t0, t1, (raw_tot[m].mean() - 160000.0) / 1600.0,
                 (comp_tot[m].mean() - 160000.0) / 1600.0))
    return arr


if __name__ == "__main__":
    s1_estimator()
    s2_end_to_end()
    s4_continuous_creep()
    s3_channel_consistency()
