# -*- coding: utf-8 -*-
# [T6] 由 temp\v4.1flash\progress\07-v6\scripts\glm53_v6.py 复制并改写 import（T6-00 设置脚本生成）。
# 改写: 0 处 import
# -*- coding: utf-8 -*-
"""GLM53 v6 原型（对应 `Document/07-v6算法说明.md` 的规格）。

与 v5 的关系：
  - **慢相模块（median 共识 g + 逐通道 γ + 限幅 + 输出封顶）逐行沿用 v5**，保证可比；
  - **前端整体替换**：短滞后电平差检测（σ 门限 + 回溯真沿、无 2.5 s 硬确认）
    → 工况分类 → 形状约束反演 Â → 速率受限滑行器（替代免责期）
    → τ_ho 交接给慢相模块（A=Â、g 按扣除连续性锚定）。
  - 空载不归零 / 输出封顶 / 未受载通道直通 三条安全语义与 v5 一致。

四处实现要点（都是实测踩出来的，不是理论推导）：
  1. **事件时间原点必须回溯到"真实加载沿"**：检测统计量要 0.1 s 近窗才敢动，
     直接把命中帧当 t0，形状里 τ=0 处就已有 10%~80% 的增量 ⇒ Â 被系统性低估。
     回溯方法 = 在命中前 ≤0.6 s 内找**单帧最大跳变**所在帧（真沿），base 取其前 0.12 s 均值。
  2. **逆模型用"电平域最小二乘"而不是"尾巴外推"**：
     Â = Σy·g / Σg²（窗口 [0.2, τ]）。尾巴外推式 Â = y(τ_ref)+Σ(y−y_ref)f/Σf² 在窗口很短时
     被 1/mean(f) 放大（短窗可达 ×25），实测把 14.2 的台阶算成 15.7；
     电平域式在形状不对口时也只按 g 的比值平移（实测跨工况只差 ±3%）。
  3. **A 是逐通道"无蠕变总电平"，不是增量**：A_i = y0_i + Â·share_i。
     写成增量会让 restep 的 g_raw=(Z−A)/A 爆炸（电平 20 / 增量 1 时 g_raw≈19）。
  4. **y0 必须取"事件前的显示值"**（输出环形缓冲的均值），不能用 (v0 − 当前输出) 反推 ——
     后者把加载跳变误算成"扣除"，实测让目标偏高 0.74（= 全部过冲）。
"""
import numpy as np

# ── 形状 ROM：g(τ) = [Z(τ)−Z(0)] / [Z(5s)−Z(0)]，13 份录制 onset 合并标定 ──
ROM_TAU = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80,
                    1.00, 1.50, 2.00, 3.00, 4.00, 5.00])
ROM_G = np.array([0.0, 0.680, 0.740, 0.790, 0.824, 0.854, 0.873, 0.886,
                  0.904, 0.922, 0.941, 0.970, 0.989, 1.000])


def g_shape(tau):
    return np.interp(np.asarray(tau, float), ROM_TAU, ROM_G, left=0.0, right=1.0)


def _cap(z, ded):
    """输出封顶：扣除量不得超过当前读数（只封顶输出值，不改内部状态）——同 v5.1。"""
    return max(z, 0.0) if ded > z else ded


class GLM53v6:
    # ── 预处理 ──
    TAU_TOTAL, TAU_LEVEL = 0.3, 10.0
    # ── 检测器 D ──
    # 窗长由 cp_v6_detwin_sweep.py 扫描确定（4 档：0.45/0.55/0.65/0.80/0.95 s 跨度）：
    #   0.45（原值）→ 恒载全段 1.46%、变载窗偏差 3099 ADC
    #   **0.65（现值）→ 恒载全段 1.11%、变载窗偏差 2220 ADC、捕获比中位 0.88→0.90**
    #   0.80/0.95   → 反而变差（1.22/1.82%）
    # 且 T_stable 在全部档位上都是 0.55 s —— **拉长检测窗不牺牲速度**（滑行 0.6~0.8 s 把
    # 检出延迟吸收了）。代价是检出晚 0.1~0.2 s（对 25% 台阶约 0.20 s）。
    DET_FAST, DET_LAG, DET_GAP = 0.20, 0.30, 0.15   # 参考窗 = [t−0.65, t−0.35]
    DET_K = 5.0
    DET_REL = 0.05          # 相对参考电平（= v5 的 LEV_REL）
    DET_ABS_FRAC = 0.01     # 相对历史最大总量（= v5 的 LEV_ABS_FRAC）
    DET_IDLE_FRAC = 0.10    # 空载态下额外要求
    DET_PERSIST = 3         # 连续帧数（≈30 ms）
    IDLE_SETTLE = 0.50      # 刚回到空载后的静默期：卸载沿后的回弹不再开新 epoch
    UNLOAD_BLOCK = 0.80     # 一次减重/卸载处理完后的事件静默期（收拢"卸载被拆成多个 epoch"）
    BACKDATE_S = 0.60
    # ── 逆模型 F ──
    TAU_REF, AWIN = 0.20, 0.60
    KAPPA_ONSET, KAPPA_RESTEP = 1.30, 1.12
    # ── 滑行器 G ──
    GLIDE_MIN, GLIDE_MAX, RATE_MAX = 0.40, 0.80, 0.8
    # 交接时刻 = 形状模型走完的时刻（τ=5 s），不是"越早越好"：
    # 慢相模块的 g 是 τ=3 s 低通，若在 τ≈1 s 交接，g 要从锚定值一路爬到真实蠕变水平，
    # 实测产生 3~9 s 的暂态（显示先冲高 ~5% 再回落），反而比 v5 差。
    # 在 τ=5 s 交接时 raw 已到 A、g_raw≈0、锚定 g≈0，无暂态。
    HO_MIN = 5.00
    REVOKE, UNLOAD_FAST = 0.40, 0.30
    REVOKE_COOLDOWN = 0.30
    DECREASE_SETTLE = 0.30
    REANCHOR_SMOOTH_S = 0.25   # 减重重锚用的平滑窗（不能取瞬时帧，见下）
    # ── 交接后 A 的慢修正（默认关；用于消掉形状先验带来的"平台静态偏置"）──
    #  背景：pin 模式下显示稳态值 ≡ A，而 A = 钉住值 = 形状反演结果，
    #  形状先验偏多少整个保压期就偏多少。实测 切换负载 @133.84 s 的 onset：
    #  钉住 28496 vs τ=5 s 实测 27167 ⇒ **偏高 +4.9%**，在 175~180 s 表现为 +875 的过充平台。
    #  开启后把 A 以 TRIM_RATE 限速朝「交接时刻实测电平」拉；限速使修正平缓、不外露成台阶，
    #  代价是显示在保压期内会缓慢移动（会体现在"慢相段时漂/平坦度"这两个口径上）。
    #  【2026-09-19 按用户要求回退】实测："trim 开"虽然把 切换负载 的平台过充从 +875 收进死区，
    #  但显示在保压期内会持续缓慢移动 —— 用户评价"还不如修复前稳定"，故默认设回 0（关闭）。
    #  保留代码路径：需要时把 TRIM_RATE 设为 0.002 即恢复（一行）。
    TRIM_RATE = 0.0            # 每秒朝目标修正的比例（0 = 关；0.002 = 0.2%/s）
    TRIM_DEAD_FRAC = 0.025     # 死区：偏差在 ±2.5%·A 以内**不修正**（保住"平"的记录不外露漂移）
    TAIL_GATE_S, TAIL_GATE_FRAC = 3.0, 0.10
    # ── 停滞检测（输入停住时不再"预判尾巴"）──
    #  判据由"斜率门"换成"**模型-实测电平落差**门"，因为它与加载方式无关：
    #  正常尾巴无论快慢，Â·g(τ) 与实测 inc(τ) 都同步（落差≈Â 自身误差，≤5%）；
    #  输入一旦停住，落差随 τ 单调涨到 Â − inc（本例 13%）。
    #  斜率门做不到这点：恒载族正常尾巴斜率 0.03~0.05·Â/s、实录族只有 0.011·Â/s，
    #  用同一个斜率阈值必然在一族上误触发（实测在 31/114/137 s 各误触发过一次）。
    #  判据最终定为「**实测尾巴增长比 vs 模型尾巴增长比**」，它同时避开了两个坑：
    #  ① 只用"模型-实测落差"会在形状先验偏 >6% 的**正常**记录上误判停滞（实测 T5% 中位 0.50→5.42 s）；
    #  ② 只用"斜率门"会被帧间波动淹没（实录族电平波动 ±2.5%，0.5 s 斜率噪声 ~±100/s，
    #     与正常尾巴斜率同量级）。
    #  增长比是无量纲的：正常尾巴（恒载族 0.11~0.13、实录族 0.19）都 ≥ 模型值；
    #  输入一旦停住则实测增长比 ≈ 0（噪声 <0.01），与模型值差一个数量级。
    STALL_START_S = 0.60      # τ 超过它才开始判
    STALL_TAIL_FRAC = 0.50    # 实测增长比 < 0.5×模型增长比 ⇒ 停滞
    STALL_HOLD_S = 0.45       # 持续多久算确认
    STALL_MIN_FRAC = 0.30     # 锚点增量至少要到 Â 的这个比例（排除瞬态）
    # ── 交接锚定模式 ──
    #  "pin"      : A = 钉住值（无蠕变电平）→ 与 g 锚定自洽 ⇒ 不产生暂态，代价是继承形状先验误差
    #  "measured" : A = 交接时刻实测电平 → 精度与 v5 同级，但显示必须先滑到实测电平（会有一次回落/抬升）
    ANCHOR_MODE = "pin"
    # ── 慢相模块（= v5）──
    TAU_G, LOADED_FRAC = 3.0, 0.10
    GAMMA_MIN, GAMMA_MAX = 0.3, 2.0
    CREEP_LO, CREEP_HI, G_ENABLE = -0.5, 1.5, 0.02
    IDLE_FRAC, UNLOAD_MIN_RATIO = 0.10, 1.5
    # ── 缓冲 ──
    CAP, DCAP = 4096, 1024

    def __init__(self, n):
        self.n = n
        self.first = True
        self.last_ts = self.t0 = 0.0
        self.ts_smooth = self.level_ref = 0.0
        self.min_ts = self.max_ts = 0.0
        self.max_tot = 0.0
        self._bt = np.zeros(self.CAP)
        self._bv = np.zeros(self.CAP)
        self._bvm = np.zeros(self.CAP)
        self._bvx = np.zeros((n, self.CAP))
        self._byx = np.zeros((n, self.CAP))
        self._bn = 0
        self._m3 = []
        self._d = np.zeros(self.DCAP)
        self._dn = 0
        self.sig_d = 0.0
        self._hit_run = 0
        self._quiet_run = 99
        self._armed = True
        self.state = "idle"
        self.ev = None
        self.ev_end = 0.0
        self.last_ev_ts = -1e9
        self.idle_since = -1e9
        self.ev_block_until = -1e9
        self.trim_target_sum = None
        self.A = np.zeros(n)
        self.loaded = np.zeros(n, bool)
        self.g = 0.0
        self.g2 = 0.0
        self.g_rel = np.zeros(n)
        self.gamma = np.ones(n)
        self.last_out = None
        self.hold_comp = None
        self.hold_until = 0.0
        self.epoch_t = []
        self.kind_log = []
        self.n_revoke = 0
        self.n_c5 = 0
        self.A_peak = 0.0
        self.debug = False
        self.dbg = []

    # ─────────────── 缓冲 ───────────────
    def _push(self, ts, total, v, y):
        i = self._bn % self.CAP
        self._bt[i] = ts
        self._bv[i] = total
        self._bvx[:, i] = v
        self._byx[:, i] = y
        # 检测/回溯用 3 帧中值：单帧掉点（实测在 右拇指/数据2 @147.31 出现过 −8% 的单帧跌落）
        # 会把 0.1 s 窗均值拉动 ~1 ADC 量级，足以伪造一次"减重"事件
        self._m3.append(total)
        if len(self._m3) > 3:
            self._m3.pop(0)
        self._bvm[i] = float(np.median(self._m3))
        self._bn += 1

    def _mask(self, t0, t1):
        n = min(self._bn, self.CAP)
        return (self._bt[:n] > t0) & (self._bt[:n] <= t1)

    def _win_mean(self, t0, t1):
        m = self._mask(t0, t1)
        n = min(self._bn, self.CAP)
        return float(self._bvm[:n][m].mean()) if m.any() else None

    def _mat_mean(self, buf, t0, t1):
        m = self._mask(t0, t1)
        n = min(self._bn, self.CAP)
        return buf[:, :n][:, m].mean(axis=1) if m.any() else None

    def _backdate(self, ts_now):
        """在命中前 ≤0.6 s 内定位真实加载沿：先取窗口最早的 1/4 作前置电平估计，
        再找「越过 前置+3%×总跳变」的第一帧，取其前一帧为 t0，base 取 t0 前 0.06 s 均值。"""
        m = self._mask(ts_now - self.BACKDATE_S, ts_now)
        n = min(self._bn, self.CAP)
        if not m.any():
            return ts_now, float(self._bv[(self._bn - 1) % self.CAP]), []
        tt = self._bt[:n][m]
        vv = self._bvm[:n][m]
        o = np.argsort(tt)
        tt, vv = tt[o], vv[o]
        if len(tt) < 6:
            return float(tt[0]), float(vv[0]), []
        q = max(3, len(vv) // 4)
        pre = float(np.median(vv[:q]))
        jump = float(vv[-1] - pre)
        if abs(jump) < 1e-12:
            return float(tt[0]), pre, []
        tgt = pre + 0.03 * jump
        if jump > 0:
            hit = np.where(vv >= tgt)[0]
        else:
            hit = np.where(vv <= tgt)[0]
        idx = int(hit[0]) - 1 if len(hit) else 0
        idx = max(idx, 0)
        mb = slice(max(0, idx - 5), idx + 1)
        base = float(np.median(vv[mb]))
        t0 = float(tt[idx])
        hist = [(float(tt[j] - t0), float(vv[j] - base)) for j in range(idx, len(tt))]
        return t0, base, hist

    # ─────────────── 逆模型（电平域最小二乘）───────────────
    def _inv_est(self, hist, tau, kappa):
        if len(hist) < 4 or tau < self.TAU_REF:
            return 0.0, False
        tt = np.array([h[0] for h in hist])
        yy = np.array([h[1] for h in hist])
        m = (tt >= self.TAU_REF) & (tt <= min(tau, self.TAU_REF + self.AWIN))
        if m.sum() < 5:
            return 0.0, False
        g = g_shape(tt[m])
        den = float((g * g).sum())
        if den < 1e-12:
            return 0.0, False
        inc = float(yy[-1])
        if inc <= 0.0:
            return 0.0, False
        A = float((yy[m] * g).sum()) / den
        A = max(A, inc)
        return float(min(A, kappa * inc)), True

    # ─────────────── 主流程 ───────────────
    def process(self, ts, v):
        v = np.asarray(v, float).copy()
        total = float(v.sum())
        if self.first:
            self.last_out = v.copy()
        self._push(ts, total, v, self.last_out[:self.n])

        dt = 0.0
        if self.first:
            self.first = False
            self.last_ts = self.t0 = ts
            self.ts_smooth = self.level_ref = total
            self.min_ts = self.max_ts = total
        else:
            dt = ts - self.last_ts
            self.last_ts = ts
            dt = 0.0 if not (dt > 0.0) else min(dt, 0.1)
            if dt > 0.0:
                self.ts_smooth += (dt / self.TAU_TOTAL) * (total - self.ts_smooth)
                self.level_ref += (dt / self.TAU_LEVEL) * (self.ts_smooth - self.level_ref)
        self.min_ts = min(self.min_ts, self.ts_smooth)
        self.max_ts = max(self.max_ts, self.ts_smooth)
        self.max_tot = max(self.max_tot, total)
        eps = 1e-6 * (1.0 + abs(self.max_tot))

        # ── 检测器 ──
        lv_now = self._win_mean(ts - self.DET_FAST, ts)
        lv_ref = self._win_mean(ts - self.DET_FAST - self.DET_GAP - self.DET_LAG,
                                ts - self.DET_FAST - self.DET_GAP)
        d = (lv_now - lv_ref) if (lv_now is not None and lv_ref is not None) else 0.0
        i = self._dn % self.DCAP
        self._d[i] = d
        self._dn += 1
        if self._dn > 40:
            h = self._d[:min(self._dn, self.DCAP)]
            self.sig_d = float(1.4826 * np.median(np.abs(h - np.median(h))))
        thr_d = max(self.DET_K * self.sig_d,
                    self.DET_REL * abs(lv_ref if lv_ref is not None else 0.0),
                    self.DET_ABS_FRAC * self.max_tot)
        # epoch 尾巴内（事件中 / 交接后 TAIL_GATE_S 内）：只认 ≥ TAIL_GATE_FRAC 电平的真实变载，
        # 避免把本 epoch 自己的快相尾巴误判成"负载内变载"（v5 用 6 s 抑制窗达到同一目的）
        tau_ep = (ts - self.ev["t0"]) if self.ev is not None else (
            (ts - self.ev_end) if self.state == "slow" else 1e9)
        tail_gate = (self.TAIL_GATE_FRAC * abs(lv_ref if lv_ref is not None else 0.0)
                     if (self.state != "idle" and tau_ep < self.TAIL_GATE_S) else 0.0)
        raw_hit = abs(d) > max(thr_d, tail_gate)
        self._hit_run = (self._hit_run + 1) if raw_hit else 0
        self._quiet_run = 0 if raw_hit else (self._quiet_run + 1)
        if self._quiet_run >= self.DET_PERSIST:
            self._armed = True                     # 本次抬升结束后才允许再建事件
        hit = (self._hit_run >= self.DET_PERSIST) and self._armed

        idle_now = (self.ts_smooth < self.IDLE_FRAC * max(self.level_ref, eps) or
                    self.ts_smooth < self.UNLOAD_MIN_RATIO * self.min_ts + eps)

        if hit and d > 0:
            if self.state == "idle":
                # 空载态：既要"台阶足够显著"，也要"已经从上一段卸载里静下来"
                # （否则卸载沿后的回弹会被当成新 onset —— 实测这是多余 epoch 的主要来源）
                if (ts - self.idle_since) < self.IDLE_SETTLE:
                    hit = False
                elif d <= max(thr_d, self.DET_IDLE_FRAC * self.max_tot):
                    hit = False
        if hit and ts < self.ev_block_until:      # 刚处理完一次减重/卸载：静默，不再拆事件
            hit = False
        if hit and d > 0:
            new_ev = None
            if self.ev is None:
                if (ts - self.last_ev_ts) > self.REVOKE_COOLDOWN:
                    new_ev = "onset_or_restep"
            elif tau_ep > self.REVOKE:
                new_ev = "restep_reload"            # C5：epoch 内再来一次真实变载
            if new_ev is not None:
                t0, base, hist = self._backdate(ts)
                v0 = self._mat_mean(self._bvx, t0 - 0.30, t0 - 0.05)
                y0 = self._mat_mean(self._byx, t0 - 0.30, t0 - 0.05)
                if v0 is not None and y0 is not None and len(hist) >= 4:
                    idle_pre = (base < self.IDLE_FRAC * max(self.level_ref, eps) or
                                base < self.UNLOAD_MIN_RATIO * self.min_ts + eps)
                    prev = self.state
                    if new_ev == "restep_reload":
                        kind = "restep_reload"
                    else:
                        kind = "onset" if (prev == "idle" and idle_pre) else "restep"
                    self._new_event(t0, base, v0, y0, kind, hist, prev)
                    self._armed = False
                self.last_ev_ts = ts
        elif hit and d < 0 and self.state != "idle" and self.ev is None:
            if (ts - self.last_ev_ts) > self.REVOKE_COOLDOWN:
                t0, base, hist = self._backdate(ts)
                v0 = self._mat_mean(self._bvx, t0 - 0.30, t0 - 0.05)
                y0 = self._mat_mean(self._byx, t0 - 0.30, t0 - 0.05)
                if v0 is not None and y0 is not None and len(hist) >= 4:
                    self._new_event(t0, base, v0, y0, "decrease", hist, self.state)
                self.last_ev_ts = ts

        if self.ev is not None:
            out = self._run_event(ts, v, total, dt, eps, idle_now)
            self.last_out = out
            return out

        if self.state == "slow":
            if idle_now and (ts - self.ev_end) > self.UNLOAD_FAST:
                self._to_idle()
                self.last_out = v.copy()
                return v
            if self.hold_comp is not None:
                if ts >= self.hold_until:
                    self._reanchor(ts, v)
                else:
                    m = self.loaded & (self.A > 1e-9)
                    out = v.copy()
                    if m.any():
                        out[m] = [v[k] - _cap(v[k], self.hold_comp[k])
                                  for k in range(self.n) if m[k]]
                    self.last_out = out
                    return out
            out = self._slow_step(ts, v, dt)
            self.last_out = out
            return out
        self.last_out = v.copy()
        return v

    # ─────────────── 事件 ───────────────
    def _new_event(self, t0, base, v0, y0, kind, hist, prev_state):
        self.epoch_t.append(float(t0))
        self.kind_log.append((float(t0), kind))
        if kind == "restep_reload":
            self.n_c5 += 1
        y0 = np.asarray(y0, float)
        self.ev = dict(t0=float(t0), t_det=float(self.last_ts), base=float(base), v0=np.asarray(v0, float).copy(),
                       y0=y0.copy(), base_y=float(y0.sum()),
                       c0=float(y0.sum() - base),           # 继承的当前修正（防"C5 时瞬间跳回原始"）
                       kind=kind, hist=list(hist), recent=[],
                       A_hat=0.0, inc_max=0.0, dec_max=0.0, inc_ref=None,
                       stalled=False, stall_t=0.0,
                       c_applied=float(y0.sum() - base), tau_g0=None,
                       Tglide=self.GLIDE_MAX, prev_state=prev_state,
                       y_prev=self.last_out.copy())
        self.state = "event"
        self.hold_comp = None
        if self.debug:
            self.dbg.append((float(t0), "NEW", kind, float(base),
                             float(np.asarray(v0, float).sum()), float(y0.sum())))

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        tau = ts - ev["t0"]
        inc = total - ev["base"]
        inc_s = self._win_mean(ts - 0.10, ts)
        inc_s = (inc_s - ev["base"]) if inc_s is not None else inc
        ev["inc_max"] = max(ev["inc_max"], inc_s)

        if tau < self.REVOKE and ev["inc_max"] > eps and inc_s < 0.5 * ev["inc_max"]:
            self.n_revoke += 1
            if self.debug:
                self.dbg.append((float(ts), "REVOKE", ev["kind"], float(tau),
                                 float(inc_s), float(ev["inc_max"])))
            prev_state = ev["prev_state"]
            y_prev = ev["y_prev"]
            self.ev = None
            self.state = prev_state if prev_state in ("idle", "slow") else "idle"
            self.last_ev_ts = ts
            if prev_state == "idle" and y_prev is not None:
                return np.asarray(y_prev, float)
            self.last_out = v.copy()
            return v

        # 减重的对称撤销：单帧掉点/手指抖动造成的"假减重"必须在 0.4 s 内被撤掉，
        # 否则会走 _reanchor 把 A 挪走（实测 右拇指/数据2 @146.6 s 就是这种假减重）
        ev["dec_max"] = min(ev["dec_max"], float(inc_s))
        if (ev["kind"] == "decrease" and tau < self.REVOKE and ev["dec_max"] < -eps
                and inc_s > 0.5 * ev["dec_max"]):
            self.n_revoke += 1
            if self.debug:
                self.dbg.append((float(ts), "REVOKE-", ev["kind"], float(tau),
                                 float(inc_s), float(ev["dec_max"])))
            prev_state = ev["prev_state"]
            self.ev = None
            self.state = prev_state if prev_state in ("idle", "slow") else "idle"
            self.last_ev_ts = ts
            self.last_out = v.copy()
            return v

        if tau > self.UNLOAD_FAST and idle_now:
            if self.debug:
                self.dbg.append((float(ts), "UNLOAD", ev["kind"], float(tau), 0.0, 0.0))
            self._to_idle()
            return v

        # 逆模型只需要 [0, 1.0] s 的样本（窗口 [τ_ref, τ_ref+AWIN] = [0.2, 0.8]）；
        # **不能按"最近 2 s"裁剪**，否则 τ>2.8 s 后窗口被裁掉 → 估计失败 → 永不交接。
        if tau <= 1.0:
            ev["hist"].append((float(tau), float(inc)))
        ev["recent"].append((float(tau), float(inc_s)))
        while ev["recent"] and ev["recent"][0][0] < tau - 0.8:
            ev["recent"].pop(0)

        if ev["kind"] == "restep" and tau >= 0.30 and ev["base"] < 0.5 * total:
            ev["kind"] = "onset"

        if ev["kind"] == "decrease":
            # 沉降窗从**检测时刻**起算：若检测本身晚于 t0+SETTLE（回溯到更早的沿），
            # 用 tau 会在建事件当帧立刻重锚，等于没有沉降窗
            if (ts - ev["t_det"]) < self.DECREASE_SETTLE:
                if self.hold_comp is None:
                    self.hold_comp = self._deduction_vector(v)
                m = self.loaded & (self.A > 1e-9)
                out = v.copy()
                if m.any():
                    out[m] = [v[k] - _cap(v[k], self.hold_comp[k])
                              for k in range(self.n) if m[k]]
                return out
            self.ev = None
            self.state = "slow"
            self.ev_end = ts
            self._reanchor(ts, v)
            self.ev_block_until = ts + self.UNLOAD_BLOCK
            return self._slow_step(ts, v, dt)

        kappa = self.KAPPA_ONSET if ev["kind"] == "onset" else self.KAPPA_RESTEP
        A_hat, ok = self._inv_est(ev["hist"], tau, kappa)
        if ok and not ev["stalled"]:            # 已判停滞：不再让形状估计覆盖（否则下一帧就被改回去）
            ev["A_hat"] = A_hat

        # ── 停滞检测：输入停住时不再"预判尾巴" ──
        # 形状模型假定"载荷会继续爬到 5 s 电平"；若输入中途停住（载荷分两级施加、中间保压），
        # 该假定被证伪，继续钉在 Â 就是过充（实测 切换负载 @11.4~13.6 s 过充 +14.6%）。
        if (ok and not ev["stalled"] and tau > self.STALL_START_S
                and ev["kind"] in ("onset", "restep", "restep_reload")):
            amp = max(abs(ev["A_hat"]), eps)
            if ev["inc_ref"] is None and tau >= self.TAU_REF and len(ev["hist"]) >= 4:
                tt = np.array([h[0] for h in ev["hist"]])
                yy = np.array([h[1] for h in ev["hist"]])
                ev["inc_ref"] = float(np.interp(self.TAU_REF, tt, yy))
            ref = ev["inc_ref"]
            stalled_now = False
            if ref is not None and ref > self.STALL_MIN_FRAC * amp:
                gr = float(g_shape(self.TAU_REF))
                ratio_mod = (float(g_shape(tau)) - gr) / gr          # 模型应增长的相对量
                ratio_obs = (inc_s - ref) / ref                      # 实测增长的相对量
                if ratio_mod > 0.03 and ratio_obs < self.STALL_TAIL_FRAC * ratio_mod:
                    stalled_now = True
            if stalled_now:
                ev["stall_t"] += dt
            else:
                ev["stall_t"] = 0.0
            if ev["stall_t"] >= self.STALL_HOLD_S:
                ev["stalled"] = True
                ev["A_hat"] = max(float(inc_s), 0.0)      # 停止预判：目标 = 当前实测增量
                if self.debug:
                    self.dbg.append((float(ts), "STALL", ev["kind"], float(tau),
                                     float(inc_s), float(ratio_obs if ref else 0.0)))
        if ev["stalled"]:
            # 停滞期内目标跟随实测电平（不再预判）；载荷恢复会触发 C5 新事件接管
            ev["A_hat"] = max(float(inc_s), 0.0)

        target = ev["base_y"] + ev["A_hat"]

        if ev["tau_g0"] is None:
            if not ok or tau < self.TAU_REF:
                return v
            ev["tau_g0"] = tau
            amp = max(abs(ev["A_hat"]), eps)
            ev["Tglide"] = float(np.clip(abs(target - total) / (self.RATE_MAX * amp),
                                         self.GLIDE_MIN, self.GLIDE_MAX))
        xx = float(np.clip((tau - ev["tau_g0"]) / ev["Tglide"], 0.0, 1.0))
        W = xx * xx * (3.0 - 2.0 * xx)
        # 修正量 = 继承值(1−W) + 新目标·W ⇒ W=0 时等于上一帧的修正，**不会瞬间跳回原始**
        c_target = ev["c0"] * (1.0 - W) + (target - total) * W
        amp = max(abs(ev["A_hat"]), eps)
        lim = self.RATE_MAX * amp * max(dt, 1e-4)
        if abs(c_target - ev["c_applied"]) > lim:
            c_target = ev["c_applied"] + np.sign(c_target - ev["c_applied"]) * lim
        ev["c_applied"] = c_target

        if self.debug and (not self.dbg or self.dbg[-1][1] != "TRK"
                           or ts - self.dbg[-1][0] > 0.1):
            self.dbg.append((float(ts), "TRK", ev["kind"], float(tau),
                             float(ev["A_hat"]), float(target - total)))
        tau_ho = max(self.HO_MIN, ev["tau_g0"] + ev["Tglide"])
        if tau >= tau_ho and ok:
            self._handoff(ts, v, ev)
            return self._slow_step(ts, v, dt)
        share = self._share_vector(ev, v)
        return v + share * c_target

    # ─────────────── 分配 / 交接 ───────────────
    def _share_vector(self, ev, v):
        inc = np.asarray(v, float) - ev["v0"]
        w = np.clip(inc, 0.0, None)
        sw = float(w.sum())
        return (w / sw) if sw > 1e-12 else np.zeros(self.n)

    def _deduction_vector(self, v):
        d = np.zeros(self.n)
        m = self.loaded & (self.A > 1e-9)
        if m.any():
            d[m] = np.clip(self.gamma[m] * self.A[m] * self.g,
                           self.CREEP_LO * self.A[m], self.CREEP_HI * self.A[m])
        return d

    def _handoff(self, ts, v, ev):
        share = self._share_vector(ev, v)
        ded_old = -(share * ev["c_applied"])            # 当前逐通道扣除（含负号=前置）
        if self.ANCHOR_MODE == "pin":
            # A = 钉住值（无蠕变电平）：与 g 锚定自洽 ⇒ 交接后不产生暂态（推荐）
            A_new = np.maximum(ev["y0"] + share * max(ev["A_hat"], 0.0), 0.0)
        else:
            # A = 交接时刻实测电平：稳态精度与 v5 同级，但显示必须先滑到实测电平
            A_new = np.maximum(np.asarray(v, float) - ded_old, 0.0)
        if float(A_new.max()) <= 1e-9:
            A_new = np.maximum(ev["y0"], 0.0)
        self.A = A_new
        self.A_peak = max(self.A_peak, float(A_new.max()))
        amax = float(self.A.max())
        self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
            else np.zeros(self.n, bool)
        m = self.loaded & (self.A > 1e-9)
        if m.any():
            self.g = float(np.clip(np.median(ded_old[m] / (self.gamma[m] * self.A[m])),
                                   self.CREEP_LO, self.CREEP_HI))
        else:
            self.g = 0.0
        if self.debug:
            self.dbg.append((float(ts), "HANDOFF", ev["kind"], float(ev["A_hat"]),
                             float(ev["c_applied"]), float(self.g)))
            self.dbg.append((float(ts), "HO2", float(amax), float(self.loaded.sum()),
                             float(self.A.sum()), float(np.sum(np.clip(
                                 self.gamma[self.loaded] * self.A[self.loaded] * self.g,
                                 self.CREEP_LO * self.A[self.loaded],
                                 self.CREEP_HI * self.A[self.loaded])))))
        # 交接时刻实测总电平（= v5 口径的幅度参考）→ 作为 A 慢修正的目标
        self.trim_target_sum = ev["base_y"] + (float(np.sum(v)) - ev["base"])
        self.ev = None
        self.state = "slow"
        self.ev_end = ts

    def _reanchor(self, ts, v):
        """减重后重锚。两处必须小心（都是实测踩出来的）：
           ① **不能取瞬时帧**：原始总量会出现 −8% 的单帧掉点（实测 右拇指/数据2 @147.31 s），
              用它算 A 会让 A 偏小 0.93（≈4%），随后 g 收敛到被高估的 g_raw，
              显示在 5 s 内悄悄下滑 0.67（≈4%）—— 就是用户看到的"奇怪下降"。
              改为取近 REANCHOR_SMOOTH_S 的**逐通道窗均值**。
           ② g 必须与当前扣除自洽（保持连续），不能归零。"""
        self.hold_comp = None
        vs = self._mat_mean(self._bvx, ts - self.REANCHOR_SMOOTH_S, ts)
        if vs is None or len(vs) != self.n:
            vs = np.asarray(v, float)
        ded_old = self._deduction_vector(np.asarray(v, float))   # 用旧 A/g 算出的当前扣除
        A_new = np.maximum(vs - ded_old, 0.0)
        if float(A_new.max()) <= 1e-9:
            A_new = np.maximum(vs, 0.0)
        self.A = A_new
        self.A_peak = max(self.A_peak, float(A_new.max()))
        amax = float(self.A.max())
        self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
            else np.zeros(self.n, bool)
        # g 取"与当前扣除自洽"的值 ⇒ display = v − γA g = v − ded_old，逐帧连续；
        # 且 g_raw = (v − A)/A = ded_old/A_new 与之一致 ⇒ 此后不会再漂
        m = self.loaded & (self.A > 1e-9)
        if m.any():
            self.g = float(np.clip(np.median(ded_old[m] / (self.gamma[m] * self.A[m])),
                                   self.CREEP_LO, self.CREEP_HI))
        else:
            self.g = 0.0

    def _to_idle(self):
        self.ev = None
        self.state = "idle"
        self.hold_comp = None
        self.loaded = np.zeros(self.n, bool)
        self.g = 0.0
        self.idle_since = self.last_ts
        self.ev_block_until = self.last_ts + self.UNLOAD_BLOCK

    # ─────────────── 慢相（= v5）───────────────
    def _slow_step(self, ts, v, dt):
        Z = v
        m = self.loaded & (self.A > 1e-9)
        # ── A 的慢修正（TRIM_RATE>0 时生效）────────────────────────────
        # 把 ΣA 以限速朝"交接时刻实测电平"拉：消掉形状先验造成的平台静态偏置。
        # 限速（默认 0.2%/s）保证它是缓变而不是台阶；目标在交接时一次性定下，不再随帧变，
        # 因此不会来回抖。
        if self.TRIM_RATE > 0.0 and self.trim_target_sum is not None:
            cur = float(self.A.sum())
            if cur > 1e-9:
                dev = self.trim_target_sum - cur
                dead = self.TRIM_DEAD_FRAC * abs(self.trim_target_sum)
                # 死区：|偏差|≤dead 视为"已经够准" ⇒ 不修正，显示保持绝对平
                dev_eff = 0.0 if abs(dev) <= dead else dev - np.sign(dev) * dead
                lim = self.TRIM_RATE * abs(self.trim_target_sum) * max(dt, 1e-6)
                dlt = float(np.clip(dev_eff, -lim, lim))
                if abs(dlt) > 1e-12:
                    self.A = self.A * (1.0 + dlt / cur)
                    if float(self.A.sum()) > 1e-9:
                        m = self.loaded & (self.A > 1e-9)
        if m.any():
            g_raw = float(np.median((Z[m] - self.A[m]) / self.A[m]))
            self.g += (dt / self.TAU_G) * (g_raw - self.g)
        if self.g > self.G_ENABLE:
            self.g2 += dt * self.g * self.g
            if m.any():
                self.g_rel[m] += dt * self.g * ((Z[m] - self.A[m]) / self.A[m])
            if self.g2 > 1e-8:
                self.gamma = np.where(self.loaded,
                                      np.clip(self.g_rel / self.g2,
                                              self.GAMMA_MIN, self.GAMMA_MAX), 1.0)
        out = Z.copy()
        if m.any():
            out[m] = [Z[k] - _cap(Z[k], float(np.clip(
                self.gamma[k] * self.A[k] * self.g,
                self.CREEP_LO * self.A[k], self.CREEP_HI * self.A[k])))
                for k in range(self.n) if m[k]]
        return out
