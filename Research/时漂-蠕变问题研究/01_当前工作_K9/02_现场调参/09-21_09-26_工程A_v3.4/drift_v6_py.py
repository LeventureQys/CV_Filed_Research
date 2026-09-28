# -*- coding: utf-8 -*-
"""DriftV6Compensator —— Python 移植（逐行对齐 src/domain/drift_v6/drift_v6_compensator.{h,cpp}）。

口径声明
--------
* 移植对象：C++ `drift_v6::DriftV6Compensator`（v6 快速稳定，plan-v3.1 PCT-fix）。
* **实跑口径**：菜单选 v6 档时 `DataHandler::SetDriftV6CompensationEnabled` 会把
  `mem_tau_s` 置 0（data_handler.cpp:103/110），即蠕变记忆（plan-v3.4）关闭。
  本移植默认参数 = `Params.as_run()`：除 `mem_tau_s=0` 外全部取 C++ 默认值；
  `mem_tau_s=120` 的记忆逻辑也已实现（`Params()` 默认），供对照。
* 唯一外部依赖：numpy。float64 与 C++ double 同口径。
* 已知边界（继承 C++ 头注释）：检测窗/迟滞按 ~100 Hz 标定，10 Hz 录制下
  0.20 s 近窗只有 1~2 帧，行为会退化（本移植不重新标定，保持与 C++ 同参数）。

与 C++ 的结构对应
----------------
  Reset/ResetFor/Push/FillMask/WinMean/MatMean/Backdate   ← 环形缓冲与窗口
  InvEst + ShapeAt                                        ← 形状 ROM 逆模型
  UpdateValley                                            ← 谷底判据
  Process                                                 ← 检测器 D + 状态机
  NewEvent/RunEvent                                       ← 事件（onset/restep/C5/decrease）
  ShareVector/Handoff/Reanchor/ToIdle/SlowStep/TrimA      ← 滑行器 G + 慢相 S（PCT）
  ApplyOutputLimit                                        ← C 限幅（plan-v2.0）
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── 形状 ROM: g(τ) = [Z(τ)−Z(0)] / [Z(5s)−Z(0)]，13 份录制 onset 合并标定 ──
_ROM_TAU = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.65,
            0.80, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00)
_ROM_G = (0.0, 0.680, 0.740, 0.790, 0.824, 0.854, 0.873,
          0.886, 0.904, 0.922, 0.941, 0.970, 0.989, 1.000)


def _shape_at(tau: float) -> float:
    if tau <= _ROM_TAU[0]:
        return 0.0
    if tau >= _ROM_TAU[-1]:
        return 1.0
    for i in range(1, len(_ROM_TAU)):
        if tau <= _ROM_TAU[i]:
            t0, t1 = _ROM_TAU[i - 1], _ROM_TAU[i]
            w = (tau - t0) / (t1 - t0)
            return _ROM_G[i - 1] + w * (_ROM_G[i] - _ROM_G[i - 1])
    return 1.0


def _capped(z: float, ded: float) -> float:
    return max(z, 0.0) if ded > z else ded


def _med(buf) -> float:
    """C++ MedianInPlace：nth_element 后取 size/2 位（偶数个时取上中位，非均值）。"""
    if len(buf) == 0:
        return 0.0
    s = sorted(buf)
    return s[len(s) // 2]


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class DriftV6Params:
    # 运行期可注入参数（默认值 = C++ static constexpr；语义见 drift_v6_compensator.h）
    clamp_alpha: float = 0.005
    valley_win_n: int = 4000
    valley_frac: float = 0.15
    valley_range_frac: float = 0.05
    event_valley_min_s: float = 0.60
    reanchor_valley_s: float = 0.20
    g_neg_floor: float = -0.05
    g_valley_reset_s: float = 0.30
    legacy_fixes_enabled: bool = False       # F2/F3 整组开关（默认关）
    kappa_onset: float = 1.05
    kappa_restep: float = 1.12
    ho_min_s: float = 3.5
    revoke_hold_n: int = 3
    pct_tau_s: float = 10.0                  # 0 = 关闭 PCT
    pct_elig_frac: float = 0.02
    pct_lo_frac: float = -0.5
    pct_hi_frac: float = 4.0
    pct_mono: bool = False
    freeze_slow_on_hit: bool = True
    mem_tau_s: float = 120.0                 # 0 = 关闭蠕变记忆
    mem_max_frac: float = 0.50
    mem_min_scale: float = 0.75

    @classmethod
    def as_run(cls) -> "DriftV6Params":
        """菜单 v6 档的实跑口径：mem_tau_s 被上位机置 0（plan-v3.1 PCT-fix）。"""
        p = cls()
        p.mem_tau_s = 0.0
        return p


# ── 常量（与 C++ 一一对应）──
TAU_TOTAL_S = 0.3
TAU_LEVEL_S = 10.0
DET_FAST_S, DET_GAP_S, DET_LAG_S = 0.20, 0.15, 0.30
DET_K, DET_REL, DET_ABS_FRAC, DET_IDLE_FRAC = 5.0, 0.05, 0.01, 0.10
DET_PERSIST = 3
IDLE_SETTLE_S, UNLOAD_BLOCK_S, BACKDATE_S = 0.50, 0.80, 0.60
TAIL_GATE_S, TAIL_GATE_FRAC = 3.0, 0.10
TAU_REF, A_WIN = 0.20, 0.60
GLIDE_MIN_S, GLIDE_MAX_S, RATE_MAX = 0.40, 0.80, 0.8
REVOKE_S, REVOKE_HOLD, REVOKE_COOLDOWN_S = 0.40, 3, 0.30
DECREASE_SETTLE_S, REANCHOR_SMOOTH_S = 0.30, 0.25
STALL_START_S, STALL_TAIL_FRAC, STALL_HOLD_S, STALL_MIN_FRAC = 0.60, 0.50, 0.45, 0.30
TRIM_RATE, TRIM_DEAD_FRAC = 0.0, 0.025
TAU_G, LOADED_FRAC = 3.0, 0.10
GAMMA_MIN, GAMMA_MAX = 0.3, 2.0
CREEP_LO_FRAC, CREEP_HI_FRAC, G_ENABLE = -0.5, 1.5, 0.02
IDLE_FRAC, UNLOAD_MIN_RATIO, UNLOAD_FAST_S = 0.10, 1.5, 0.30
CAP, DCAP = 4096, 1024

IDLE, EVENT, SLOW = "idle", "event", "slow"
ONSET, RESTEP, RESTEP_RELOAD, DECREASE = "onset", "restep", "restep_reload", "decrease"


class _Ev:
    """EventCtx（C++ POD 结构的等价物）。"""

    def __init__(self):
        self.valid = False
        self.kind = ONSET
        self.stalled = False
        self.prev_state = IDLE
        self.t0 = 0.0
        self.t_det = 0.0
        self.base = 0.0
        self.v0: np.ndarray | None = None
        self.y0: np.ndarray | None = None
        self.base_y = 0.0
        self.c0 = 0.0
        self.y_prev: np.ndarray | None = None
        self.a_hat = 0.0
        self.inc_max = 0.0
        self.dec_max = 0.0
        self.revoke_run = 0
        self.inc_ref_set = False
        self.inc_ref = 0.0
        self.stall_t = 0.0
        self.c_applied = 0.0
        self.tau_g0_set = False
        self.tau_g0 = 0.0
        self.tglide = GLIDE_MAX_S
        self.use_mem = False
        self.hist: list[tuple[float, float]] = []


class DriftV6:
    def __init__(self, params: DriftV6Params | None = None):
        self.p = params if params is not None else DriftV6Params.as_run()
        self.n_clamp = 0
        self.n_revoke = 0
        self.n_pct_hits = 0
        self.shape_hits = 0
        self._reset()
        self.n_ch = 0

    # ───────────────────────── 生命周期 ─────────────────────────
    def _reset(self) -> None:
        self.n = 0
        self.first_frame = True
        self.slow_freeze = False
        self.last_ts = 0.0
        self.ts_smooth = 0.0
        self.level_ref = 0.0
        self.min_ts = 0.0
        self.max_ts = 0.0
        self.max_tot = 0.0
        self.buf_t = [0.0] * CAP
        self.buf_v = [0.0] * CAP          # 3 帧中值总量
        self.buf_vx = np.zeros((0, CAP))
        self.buf_yx = np.zeros((0, CAP))
        self.buf_n = 0
        self.med_hist = [0.0, 0.0, 0.0]
        self.med_n = 0
        self.prev_med_total = 0.0
        self.det_d = [0.0] * DCAP
        self.det_n = 0
        self.sig_d = 0.0
        self.hit_run = 0
        self.quiet_run = 99
        self.armed = True
        self.state = IDLE
        self.ev = _Ev()
        self.ev_end = 0.0
        self.last_ev_ts = -1e9
        self.idle_since = -1e9
        self.ev_block_until = -1e9
        self.A = np.zeros(0)
        self.loaded: list[int] = []
        self.g = 0.0
        self.g2_acc = 0.0
        self.g_rel_acc = np.zeros(0)
        self.gamma = np.ones(0)
        self.hold_comp: np.ndarray | None = None   # None = 未捕获
        self.trim_target_sum = -1.0
        self.pct_ded = np.zeros(0)
        self.pct_elig: list[int] = []
        self.mem_ded = np.zeros(0)
        self.mem_ts = -1e9
        self.mem_lv = 0.0
        self.mem_valid = False
        self.prev_out = np.zeros(0)
        # 谷底判据
        self.valley_buf = [0.0] * self.p.valley_win_n
        self.valley_n = 0
        self.valley_now = False
        self.valley_run = 0.0

    def _reset_for(self, n: int) -> None:
        self._reset()
        self.n = n
        self.buf_vx = np.zeros((n, CAP))
        self.buf_yx = np.zeros((n, CAP))
        self.A = np.zeros(n)
        self.g_rel_acc = np.zeros(n)
        self.gamma = np.ones(n)
        self.loaded = [0] * n
        self.pct_ded = np.zeros(n)
        self.pct_elig = [0] * n

    def reset(self) -> None:
        n = self.n
        self._reset()
        if n:
            self._reset_for(n)

    # ───────────────────────── 缓冲 ─────────────────────────
    def _push(self, ts: float, v: np.ndarray, y: np.ndarray) -> None:
        i = self.buf_n % CAP
        self.buf_t[i] = ts
        self.buf_vx[:, i] = v
        self.buf_yx[:, i] = y
        # 写入的是上一帧算出的 3 帧中值（与原型 _push 一致）
        self.buf_v[i] = self.prev_med_total
        self.buf_n += 1

    def _fill_mask(self, t0: float, t1: float) -> list[int]:
        avail = min(self.buf_n, CAP)
        start = self.buf_n - avail
        out = []
        for k in range(start, self.buf_n):
            i = k % CAP
            t = self.buf_t[i]
            if t0 < t <= t1:
                out.append(i)
        return out

    def _win_mean(self, t0: float, t1: float):
        m = self._fill_mask(t0, t1)
        if not m:
            return False, 0.0
        return True, sum(self.buf_v[i] for i in m) / len(m)

    def _mat_mean(self, buf: np.ndarray, t0: float, t1: float):
        m = self._fill_mask(t0, t1)
        if not m or self.n <= 0:
            return False, None
        return True, buf[:, m].sum(axis=1) / len(m)

    def _backdate(self, ts_now: float):
        """回溯真实加载沿 → (t0, base, hist[(τ, inc)])。"""
        hist: list[tuple[float, float]] = []
        m = self._fill_mask(ts_now - BACKDATE_S, ts_now)
        if not m:
            last = (self.buf_n - 1) % CAP
            return ts_now, self.buf_v[last], hist
        tv = sorted((self.buf_t[i], self.buf_v[i]) for i in m)
        if len(tv) < 6:
            return tv[0][0], tv[0][1], hist
        q = max(3, len(tv) // 4)
        pre = _med([tv[k][1] for k in range(q)])
        jump = tv[-1][1] - pre
        if abs(jump) < 1e-12:
            return tv[0][0], pre, hist
        tgt = pre + 0.03 * jump
        idx = 0
        for k in range(len(tv)):
            if (tv[k][1] >= tgt) if jump > 0 else (tv[k][1] <= tgt):
                idx = k - 1 if k > 0 else 0
                break
        b0 = idx - 5 if idx > 5 else 0
        base_val = _med([tv[k][1] for k in range(b0, idx + 1)])
        t0_val = tv[idx][0]
        for k in range(idx, len(tv)):
            hist.append((tv[k][0] - t0_val, tv[k][1] - base_val))
        return t0_val, base_val, hist

    # ───────────────────────── 逆模型 ─────────────────────────
    def _inv_est(self, hist, tau: float, kappa: float):
        if len(hist) < 4 or tau < TAU_REF:
            return False, 0.0
        hi = min(tau, TAU_REF + A_WIN)
        num = den = 0.0
        cnt = 0
        for t, y in hist:
            if t < TAU_REF or t > hi:
                continue
            g = _shape_at(t)
            num += y * g
            den += g * g
            cnt += 1
        if cnt < 5 or den < 1e-12:
            return False, 0.0
        inc = hist[-1][1]
        if inc <= 0.0:
            return False, 0.0
        a = max(num / den, inc)
        self.shape_hits += 1
        return True, min(a, kappa * inc)

    # ───────────────────────── 谷底判据 ─────────────────────────
    def _update_valley(self, dt: float) -> None:
        w_n = self.p.valley_win_n
        self.valley_buf[self.valley_n % w_n] = self.ts_smooth
        self.valley_n += 1
        if self.valley_n < w_n:
            self.valley_now = False
            self.valley_run = 0.0
            return
        w_min = min(self.valley_buf)
        w_max = max(self.valley_buf)
        eps_v = 1e-6 * (1.0 + abs(w_max))
        self.valley_now = (self.ts_smooth < w_min + self.p.valley_frac * (w_max - w_min + eps_v)) and \
                          ((w_max - w_min) > self.p.valley_range_frac * max(abs(w_max), eps_v))
        if self.valley_now and dt > 0.0:
            self.valley_run += dt
        elif not self.valley_now:
            self.valley_run = 0.0

    # ───────────────────────── C 限幅 ─────────────────────────
    def _apply_output_limit(self, raw: np.ndarray, out: np.ndarray) -> None:
        if self.p.clamp_alpha <= 0.0 or raw.size != out.size:
            return
        lim = raw + self.p.clamp_alpha * np.abs(raw)
        mask = out > lim
        self.n_clamp += int(mask.sum())
        out[mask] = lim[mask]

    # ───────────────────────── 主流程 ─────────────────────────
    def process(self, ts: float, values_io: np.ndarray) -> np.ndarray:
        v = values_io.astype(np.float64).copy()
        n = v.size
        if n <= 0:
            return v
        if self.n != n:
            self._reset_for(n)

        total = float(v.sum())
        y = v if self.first_frame else self.prev_out

        if self.first_frame:
            self.first_frame = False
            self.last_ts = ts
            self.ts_smooth = self.level_ref = total
            self.min_ts = self.max_ts = total
            dt = 0.0
        else:
            dt = ts - self.last_ts
            self.last_ts = ts
            dt = min(dt, 0.1) if dt > 0.0 else 0.0
            if dt > 0.0:
                self.ts_smooth += (dt / TAU_TOTAL_S) * (total - self.ts_smooth)
                self.level_ref += (dt / TAU_LEVEL_S) * (self.ts_smooth - self.level_ref)
        self.min_ts = min(self.min_ts, self.ts_smooth)
        self.max_ts = max(self.max_ts, self.ts_smooth)
        self.max_tot = max(self.max_tot, total)
        eps = 1e-6 * (1.0 + abs(self.max_tot))

        self._update_valley(dt)

        # 本帧 3 帧中值总量
        if self.med_n < 3:
            self.med_hist[self.med_n] = total
            self.med_n += 1
        else:
            self.med_hist[0] = self.med_hist[1]
            self.med_hist[1] = self.med_hist[2]
            self.med_hist[2] = total
        self.prev_med_total = _med(self.med_hist[:self.med_n])

        # ── 检测器 D ──
        have_now, lv_now = self._win_mean(ts - DET_FAST_S, ts)
        have_ref, lv_ref = self._win_mean(ts - DET_FAST_S - DET_GAP_S - DET_LAG_S,
                                          ts - DET_FAST_S - DET_GAP_S)
        d = (lv_now - lv_ref) if (have_now and have_ref) else 0.0
        ref_mag = abs(lv_ref) if have_ref else 0.0

        self.det_d[self.det_n % DCAP] = d
        self.det_n += 1
        if self.det_n > 40:
            h = min(self.det_n, DCAP)
            tmp = self.det_d[:h]
            med = _med(tmp)
            sig = _med([abs(x - med) for x in tmp])
            self.sig_d = 1.4826 * sig
        thr_d = max(DET_K * self.sig_d, DET_REL * ref_mag, DET_ABS_FRAC * self.max_tot)

        tau_ep = (ts - self.ev.t0) if self.ev.valid else \
                 ((ts - self.ev_end) if self.state == SLOW else 1e9)
        tail_gate = (TAIL_GATE_FRAC * ref_mag) if (self.state != IDLE and tau_ep < TAIL_GATE_S) else 0.0
        raw_hit = abs(d) > max(thr_d, tail_gate)
        self.slow_freeze = self.p.freeze_slow_on_hit and raw_hit
        self.hit_run = self.hit_run + 1 if raw_hit else 0
        self.quiet_run = 0 if raw_hit else self.quiet_run + 1
        if self.quiet_run >= DET_PERSIST:
            self.armed = True
        hit = (self.hit_run >= DET_PERSIST) and self.armed

        lvl = max(self.level_ref, eps)
        idle_now = (self.ts_smooth < IDLE_FRAC * lvl) or \
                   (self.ts_smooth < UNLOAD_MIN_RATIO * self.min_ts + eps)

        if hit and d > 0 and self.state == IDLE:
            if (ts - self.idle_since) < IDLE_SETTLE_S:
                hit = False
            elif d <= max(thr_d, DET_IDLE_FRAC * self.max_tot):
                hit = False
        if hit and ts < self.ev_block_until:
            hit = False

        if hit and d > 0:
            build = False
            c5 = False
            if not self.ev.valid:
                if (ts - self.last_ev_ts) > REVOKE_COOLDOWN_S:
                    build = True
            elif tau_ep > REVOKE_S:
                build = True
                c5 = True
            if build:
                t0, base, hist = self._backdate(ts)
                ok_v0, v0 = self._mat_mean(self.buf_vx, t0 - 0.30, t0 - 0.05)
                ok_y0, y0 = self._mat_mean(self.buf_yx, t0 - 0.30, t0 - 0.05)
                if ok_v0 and ok_y0 and len(hist) >= 4:
                    idle_pre = (base < IDLE_FRAC * lvl) or (base < UNLOAD_MIN_RATIO * self.min_ts + eps)
                    kind = RESTEP
                    if c5:
                        kind = RESTEP_RELOAD
                    elif self.state == IDLE and idle_pre:
                        kind = ONSET
                    self._new_event(t0, base, v0, y0, kind)
                    self.armed = False
                self.last_ev_ts = ts
        elif hit and d < 0 and self.state != IDLE and not self.ev.valid:
            if (ts - self.last_ev_ts) > REVOKE_COOLDOWN_S:
                t0, base, hist = self._backdate(ts)
                ok_v0, v0 = self._mat_mean(self.buf_vx, t0 - 0.30, t0 - 0.05)
                ok_y0, y0 = self._mat_mean(self.buf_yx, t0 - 0.30, t0 - 0.05)
                if ok_v0 and ok_y0 and len(hist) >= 4:
                    # plan-v3.4：减重确认当帧快照总扣除与参考窗电平（记忆单调不缩水）
                    lv_cap = lv_ref if have_ref else total
                    cap = self._deduction_vector()
                    if self.mem_valid and self.mem_ded.size == n and self.p.mem_tau_s > 0.0:
                        decay = np.exp(-(ts - self.mem_ts) / self.p.mem_tau_s)
                        self.mem_ded = np.maximum(cap, self.mem_ded * decay)
                        self.mem_lv = max(self.mem_lv, lv_cap)
                    else:
                        self.mem_ded = cap
                        self.mem_lv = lv_cap
                    self.mem_ts = ts
                    self.mem_valid = (self.mem_ded.size == n) and \
                                     (float(np.abs(self.mem_ded).sum()) > eps)
                    self._new_event(t0, base, v0, y0, DECREASE)
                self.last_ev_ts = ts

        if self.ev.valid:
            out = self._run_event(ts, v, total, dt, eps, idle_now)
            if out is not None:
                self._apply_output_limit(v, out)
                self.prev_out = out
                self._push(ts, v, y)
                return out

        if self.state == SLOW:
            if idle_now and (ts - self.ev_end) > UNLOAD_FAST_S:
                self._to_idle()
                out = v.copy()
            else:
                out = self._slow_step(v, dt)
            self._apply_output_limit(v, out)
            self.prev_out = out
            self._push(ts, v, y)
            return out

        self.prev_out = v
        self._push(ts, v, y)
        return v

    # ───────────────────────── 事件 ─────────────────────────
    def _new_event(self, t0, base, v0, y0, kind) -> None:
        prev = self.state
        ev = _Ev()
        ev.valid = True
        ev.kind = kind
        ev.prev_state = prev
        ev.t0 = t0
        ev.t_det = self.last_ts
        ev.base = base
        ev.v0 = v0.copy()
        ev.y0 = y0.copy()
        ev.base_y = float(y0.sum())
        ev.c0 = ev.base_y - base
        ev.c_applied = ev.c0
        ev.tglide = GLIDE_MAX_S
        ev.y_prev = self.prev_out.copy() if self.prev_out.size else None
        ev.use_mem = (kind != DECREASE) and self.mem_valid and (self.p.mem_tau_s > 0.0)
        self.ev = ev
        self.state = EVENT
        self.hold_comp = None

    def _share_vector(self, ev: _Ev, v: np.ndarray) -> np.ndarray:
        w = np.maximum(v - ev.v0, 0.0)
        sw = float(w.sum())
        if sw > 1e-12:
            return w / sw
        return np.zeros_like(v)

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        tau = ts - ev.t0
        inc = total - ev.base
        inc_s = inc
        ok_wm, wm = self._win_mean(ts - 0.10, ts)
        if ok_wm:
            inc_s = wm - ev.base

        # C6 试探撤销（连续 kRevokeHoldN 帧）
        revoke_now = False
        if tau < REVOKE_S and ev.inc_max > eps and inc_s < 0.5 * ev.inc_max:
            ev.revoke_run += 1
            revoke_now = ev.revoke_run >= REVOKE_HOLD
        else:
            ev.revoke_run = 0
        ev.inc_max = max(ev.inc_max, inc_s)
        if revoke_now:
            self.n_revoke += 1
            prev = ev.prev_state
            y_prev = ev.y_prev
            self.ev = _Ev()
            self.state = prev if prev in (IDLE, SLOW) else IDLE
            self.last_ev_ts = ts
            if prev == IDLE and y_prev is not None and y_prev.size == self.n:
                return y_prev.copy()
            return v.copy()

        # 减重的对称撤销
        ev.dec_max = min(ev.dec_max, inc_s)
        if ev.kind == DECREASE and tau < REVOKE_S and ev.dec_max < -eps and inc_s > 0.5 * ev.dec_max:
            self.n_revoke += 1
            prev = ev.prev_state
            self.ev = _Ev()
            self.state = prev if prev in (IDLE, SLOW) else IDLE
            self.last_ev_ts = ts
            return v.copy()

        # F2（默认关）
        if self.p.legacy_fixes_enabled and tau > self.p.event_valley_min_s and self.valley_now:
            self._to_idle()
            return v.copy()

        # C4 事件期内落入空载带
        if tau > UNLOAD_FAST_S and idle_now:
            self._to_idle()
            return v.copy()

        if tau <= 1.0:
            ev.hist.append((tau, inc))

        if ev.kind == RESTEP and tau >= 0.30 and ev.base < 0.5 * total:
            ev.kind = ONSET

        if ev.kind == DECREASE:
            if (ts - ev.t_det) < DECREASE_SETTLE_S:
                if self.hold_comp is None or self.hold_comp.size != self.n:
                    self.hold_comp = self._deduction_vector()
                o = v.copy()
                for k in range(self.n):
                    if abs(self.hold_comp[k]) > 1e-12:
                        o[k] = v[k] - _capped(v[k], self.hold_comp[k])
                return o
            self.ev = _Ev()
            self.state = SLOW
            self.ev_end = ts
            self._reanchor(ts, v)
            self.ev_block_until = ts + UNLOAD_BLOCK_S
            return self._slow_step(v, dt)

        kappa = self.p.kappa_onset if ev.kind == ONSET else self.p.kappa_restep
        ok, a_hat = self._inv_est(ev.hist, tau, kappa)
        if ok and not ev.stalled:
            ev.a_hat = a_hat

        # 停滞检测
        if ok and not ev.stalled and tau > STALL_START_S:
            amp = max(abs(ev.a_hat), eps)
            if not ev.inc_ref_set and tau >= TAU_REF and len(ev.hist) >= 4:
                h = ev.hist
                for k in range(len(h) - 1):
                    if h[k][0] <= TAU_REF <= h[k + 1][0]:
                        span = h[k + 1][0] - h[k][0]
                        w = (TAU_REF - h[k][0]) / span if span > 1e-12 else 0.0
                        ev.inc_ref = h[k][1] + w * (h[k + 1][1] - h[k][1])
                        ev.inc_ref_set = True
                        break
            stalled_now = False
            if ev.inc_ref_set and ev.inc_ref > STALL_MIN_FRAC * amp:
                gr = _shape_at(TAU_REF)
                ratio_mod = (_shape_at(tau) - gr) / gr
                ratio_obs = (inc_s - ev.inc_ref) / ev.inc_ref
                if ratio_mod > 0.03 and ratio_obs < STALL_TAIL_FRAC * ratio_mod:
                    stalled_now = True
            ev.stall_t = ev.stall_t + dt if stalled_now else 0.0
            if ev.stall_t >= STALL_HOLD_S:
                ev.stalled = True
                ev.a_hat = max(inc_s, 0.0)
        if ev.stalled:
            ev.a_hat = max(inc_s, 0.0)

        mem_eff = (self._creep_memory(ts, ev.base, ev.a_hat, max(0.0, ev.base - ev.base_y))
                   if ev.use_mem else np.zeros(self.n))
        target = ev.base_y + ev.a_hat - float(mem_eff.sum())

        if not ev.tau_g0_set:
            if not ok or tau < TAU_REF:
                return v.copy()          # Â 尚不可用：直通
            ev.tau_g0_set = True
            ev.tau_g0 = tau
            amp = max(abs(ev.a_hat), eps)
            ev.tglide = _clamp(abs(target - total) / (RATE_MAX * amp), GLIDE_MIN_S, GLIDE_MAX_S)
        xx = _clamp((tau - ev.tau_g0) / ev.tglide, 0.0, 1.0)
        w = xx * xx * (3.0 - 2.0 * xx)   # smoothstep
        c_target = ev.c0 * (1.0 - w) + (target - total) * w
        amp = max(abs(ev.a_hat), eps)
        lim = RATE_MAX * amp * max(dt, 1e-4)
        if abs(c_target - ev.c_applied) > lim:
            c_target = ev.c_applied + (lim if c_target > ev.c_applied else -lim)
        ev.c_applied = c_target

        tau_ho = max(self.p.ho_min_s, ev.tau_g0 + ev.tglide)
        if tau >= tau_ho and ok:
            self._handoff(ts, v)
            return self._slow_step(v, dt)
        share = self._share_vector(ev, v)
        return v + share * c_target

    # ───────────────────────── 分配 / 交接 ─────────────────────────
    def _rank1_deduction_vector(self) -> np.ndarray:
        d = np.zeros(self.n)
        for k in range(self.n):
            if self.loaded[k] and self.A[k] > 1e-9:
                d[k] = _clamp(self.gamma[k] * self.A[k] * self.g,
                              CREEP_LO_FRAC * self.A[k], CREEP_HI_FRAC * self.A[k])
        return d

    def _deduction_vector(self) -> np.ndarray:
        d = self._rank1_deduction_vector()
        if self.pct_ded.size == self.n:
            for k in range(self.n):
                if k < len(self.pct_elig) and self.pct_elig[k] and self.A[k] > 1e-9:
                    d[k] += self.pct_ded[k]
        return d

    def _creep_memory(self, ts, base, a_hat, embedded_ded) -> np.ndarray:
        z = np.zeros(self.n)
        if not self.mem_valid or self.mem_ded.size != self.n or self.p.mem_tau_s <= 0.0:
            return z
        lv_now = base + a_hat
        if lv_now <= 1e-9 or self.mem_lv <= 1e-9:
            return z
        s = _clamp(lv_now / self.mem_lv, 0.0, 1.0)
        if s < self.p.mem_min_scale:
            return z
        decay = np.exp(-(ts - self.mem_ts) / self.p.mem_tau_s)
        if decay <= 1e-6:
            return z
        m = np.maximum(self.mem_ded, 0.0) * (s * decay)
        m_sum = float(m.sum())
        resid = m_sum - embedded_ded
        if m_sum <= 1e-12 or resid <= 1e-9:
            return z
        m *= resid / m_sum
        cap = self.p.mem_max_frac * max(a_hat, base)
        if float(m.sum()) > cap:
            m *= cap / float(m.sum())
        return m

    def _handoff(self, ts, v) -> None:
        ev = self.ev
        share = self._share_vector(ev, v)
        ded_old = -(share * ev.c_applied)
        A_new = np.maximum(ev.y0 + share * max(ev.a_hat, 0.0), 0.0)
        if ev.use_mem:
            mem_eff = self._creep_memory(self.last_ts, ev.base, ev.a_hat,
                                         max(0.0, ev.base - ev.base_y))
            A_new = np.maximum(A_new - mem_eff, 0.0)
        if A_new.max() <= 1e-9:
            A_new = np.maximum(ev.y0, 0.0)
        self.A = A_new
        amax = float(self.A.max())
        self.loaded = [1 if (amax > 1e-9 and self.A[k] > LOADED_FRAC * amax) else 0
                       for k in range(self.n)]
        self.pct_elig = [1 if (amax > 1e-9 and self.A[k] > self.p.pct_elig_frac * amax) else 0
                         for k in range(self.n)]
        self.pct_ded = np.zeros(self.n)

        vals = []
        for k in range(self.n):
            if self.loaded[k] and self.A[k] > 1e-9 and abs(self.gamma[k]) > 1e-12:
                vals.append(ded_old[k] / (self.gamma[k] * self.A[k]))
        self.g = _clamp(_med(vals), CREEP_LO_FRAC, CREEP_HI_FRAC) if vals else 0.0
        self.trim_target_sum = ev.base_y + (float(v.sum()) - ev.base)
        self.ev = _Ev()
        self.state = SLOW
        self.ev_end = self.last_ts

    def _reanchor(self, ts, v) -> None:
        if self.p.legacy_fixes_enabled and self.valley_now and \
                self.valley_run >= self.p.reanchor_valley_s:
            self._to_idle()
            return
        self.hold_comp = None
        ok, vs = self._mat_mean(self.buf_vx, ts - REANCHOR_SMOOTH_S, ts)
        if not ok:
            vs = v
        ded1_old = self._rank1_deduction_vector()
        ded_tot = self._deduction_vector()
        A_new = np.maximum(vs - ded_tot, 0.0)
        if A_new.max() <= 1e-9:
            A_new = np.maximum(vs, 0.0)
        self.A = A_new
        amax = float(self.A.max())
        self.loaded = [1 if (amax > 1e-9 and self.A[k] > LOADED_FRAC * amax) else 0
                       for k in range(self.n)]
        self.pct_elig = [1 if (amax > 1e-9 and self.A[k] > self.p.pct_elig_frac * amax) else 0
                         for k in range(self.n)]
        vals = []
        for k in range(self.n):
            if self.loaded[k] and self.A[k] > 1e-9 and abs(self.gamma[k]) > 1e-12:
                vals.append(ded1_old[k] / (self.gamma[k] * self.A[k]))
        self.g = _clamp(_med(vals), CREEP_LO_FRAC, CREEP_HI_FRAC) if vals else 0.0

    def _to_idle(self) -> None:
        self.ev = _Ev()
        self.state = IDLE
        self.hold_comp = None
        self.loaded = [0] * self.n
        self.pct_elig = [0] * self.n
        self.pct_ded = np.zeros(self.n)
        self.g = 0.0
        self.idle_since = self.last_ts
        self.ev_block_until = self.last_ts + UNLOAD_BLOCK_S

    def _trim_a(self, dt) -> None:
        if TRIM_RATE <= 0 or self.trim_target_sum < 0:
            return
        cur = float(self.A.sum())
        if cur <= 1e-9:
            return
        dev = self.trim_target_sum - cur
        dead = TRIM_DEAD_FRAC * abs(self.trim_target_sum)
        dev_eff = 0.0 if abs(dev) <= dead else (dev - (dead if dev > 0 else -dead))
        lim = TRIM_RATE * abs(self.trim_target_sum) * max(dt, 1e-6)
        dlt = _clamp(dev_eff, -lim, lim)
        if abs(dlt) > 1e-12:
            self.A *= 1.0 + dlt / cur

    def _slow_step(self, v, dt) -> np.ndarray:
        self._trim_a(dt)
        vals = []
        for k in range(self.n):
            if self.loaded[k] and self.A[k] > 1e-9:
                vals.append((v[k] - self.A[k]) / self.A[k])
        if vals and not self.slow_freeze:
            g_raw = _med(vals)
            self.g += (dt / TAU_G) * (g_raw - self.g)
        if self.g < self.p.g_neg_floor:
            self.g = self.p.g_neg_floor
        if self.valley_now and self.valley_run >= self.p.g_valley_reset_s and self.g < 0.0:
            self.g = 0.0
        if self.g > G_ENABLE:
            self.g2_acc += dt * self.g * self.g
            for k in range(self.n):
                if self.loaded[k] and self.A[k] > 1e-9:
                    self.g_rel_acc[k] += dt * self.g * ((v[k] - self.A[k]) / self.A[k])
            if self.g2_acc > 1e-8:
                for k in range(self.n):
                    self.gamma[k] = _clamp(self.g_rel_acc[k] / self.g2_acc, GAMMA_MIN, GAMMA_MAX) \
                        if self.loaded[k] else 1.0
        ded = np.zeros(self.n)
        for k in range(self.n):
            if self.loaded[k] and self.A[k] > 1e-9:
                ded[k] = _clamp(self.gamma[k] * self.A[k] * self.g,
                                CREEP_LO_FRAC * self.A[k], CREEP_HI_FRAC * self.A[k])
        # plan-v3.0：PCT
        if self.p.pct_tau_s > 0.0 and self.pct_ded.size == self.n and len(self.pct_elig) == self.n:
            if dt > 0.0 and not self.slow_freeze:
                for k in range(self.n):
                    if not self.pct_elig[k] or self.A[k] <= 1e-9:
                        continue
                    resid = (v[k] - self.A[k]) - ded[k] - self.pct_ded[k]
                    d = self.pct_ded[k] + (dt / self.p.pct_tau_s) * resid
                    if self.p.pct_mono and d < self.pct_ded[k]:
                        d = self.pct_ded[k]
                    d = max(d, self.p.pct_lo_frac * self.A[k])
                    d = min(d, self.p.pct_hi_frac * self.A[k])
                    self.pct_ded[k] = d
                    if abs(d) > 1e-12:
                        self.n_pct_hits += 1
            for k in range(self.n):
                if self.pct_elig[k] and self.A[k] > 1e-9:
                    ded[k] += self.pct_ded[k]
        out = v.copy()
        for k in range(self.n):
            if self.A[k] > 1e-9 and abs(ded[k]) > 1e-12:
                out[k] = v[k] - _capped(v[k], ded[k])
        return out
