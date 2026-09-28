# -*- coding: utf-8 -*-
"""GLM53 v6c：v5.1 的「稳健慢相」+ 用形状 ROM 校正已观测幅度的「快速稳定前端」。

设计动机（对照 v5 与失败的 v6 原型）
--------------------------------------------------------------------
v5 / v5.1 的时间线（实测口径）：

    真实加载沿 ─[STEP_PERSIST=2.5s 确认]→ epoch 起点 ─[FAST_S=3s 免责期]→ 慢相
    0s                 2.5s                 2.5s           5.5s          5.5s+

免责期内 `g ≡ 0`、输出 = 读数 − carry（基本等于直通），所以真正开始扣蠕变是在真实加载后
5.5s，而且 `g` 还要从 0 以 τ=3s 向上爬 —— 用户看到的"总稳定 ~14s"里有 3s 是免责期空等。

v6 原型想用"形状 ROM 反演 + 滑行器"把这段压到 ~1s，代价是：检测器过度灵敏（epoch 44~57
vs v5 的 16）、输入中途停住时形状模型过充、交接锚定不自洽导致稳定后继续漂、单帧掉点触发
假事件，实录表现（gap 中位 4571 ADC）反而比 v5（1889）差。

v6c 的核心取舍：**形状 ROM 只用来"校正已经观测到的幅度"，绝不外推未来电平。**
物理依据很明确 —— ROM 的 g(τ) 是"τ 时刻读数占 5s 电平的比例"，快相的机械建立本身也服从
同一条曲线（1.8s 时 g≈0.93，3.5s 时 g≈0.98），所以

    A_corrected = A_observed / shape(τ_real)      （τ_real = 真实加载后经过的时间）

补回的是**快相机械尾巴**，得到"无蠕变载荷电平"的估计，而不是 5s 之后的电平。
快相窗 τ_real ∈ [2.5, 5.5]s 时 shape ∈ [0.94, 1.00]，校正量只有 1%~6%，
并被 SHAPE_MAX_CORRECT（默认 +15%）封顶 —— 比 v6 的 ×1.3 外推安全一个量级。

三条实现原则
--------------------------------------------------------------------
1. **保留 v5 的 2.5s 确认与电平判据**：epoch 控制逐行沿用 v5.1（不降敏也不增敏），
   所以 epoch 数与 v5 同级，不会出现 v6 的 44~57 个假事件；单帧掉点仍由 v5 的
   电平锁存 + pending 复位机制吸收。
2. **快相窗内把校正"限速铺开"，而不是交接瞬间跳**：校正量在快相窗内按 smoothstep
   从 0 爬到 1，起点由 RAMP_TAU 兜底，终点与交接状态**逐通道严格相等**，所以交接处
   既没有跳变、也没有"交接后继续漂"的暂态。这**不是** v6 的滑行器：滑行器服务的是
   "预测出来的未来电平"，这里的斜坡服务的是"已观测幅度的校正"，且被 RAMP_RATE_FRAC
   限速、被 SHAPE_MAX_CORRECT 封顶。
3. **交接自洽**：交接时 `g = ded_applied / (γ · A_corrected)`，其中 `ded_applied` 是
   快相窗**实际生效**的扣除量 ⇒ 交接当帧扣除量与快相末帧扣除量逐通道严格相等，
   display 无跳变、内部 (A, g) 与 display 自洽，慢相模块从第一帧起就以"正确的幅度 +
   正确的扣除量"运行，不需要再爬。

净效果：补偿从真实加载后 ≈3.3s 就开始（v5 是 5.5s 起再从 0 爬），且不引入 v6 的
过充 / 假事件 / 交接漂移。

对照开关（用于 A/B 归因，均为类属性，可在实例上覆盖）：
    SHAPE_CORRECT = False            → 关掉形状校正（A_corr ≡ A_obs）
    ACCEL_G_INIT  = False            → 关掉加速（斜坡终点 = carry，快相输出 = 读数 − carry）
    两者同时置 False                 → 与 `GLM53v51` 逐帧数值等价（回归基准）
"""
import os
import sys
import numpy as np

# ── 独立运行支持：算法本体在 temp/v4.1flash/scripts/ 下（glm53_v51 → v5 → v3）──
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.normpath(os.path.join(_HERE, os.pardir, "v4.1flash", "scripts"))
if os.path.isdir(_SCRIPTS) and _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from glm53_v51 import GLM53v51          # noqa: E402

__all__ = ["GLM53v6c", "g_shape", "ROM_TAU", "ROM_G", "run_glm53_v6c"]

# ── 形状 ROM：g(τ) = [Z(τ) − Z(0)] / [Z(5s) − Z(0)]，13 份录制 onset 合并标定 ──
#    用法**只有一个方向**：给定已观测的 τ，回推"当前读数是 5s 电平的百分之多少"；
#    绝不反过来预测 τ 之后还会涨多少（那正是 v6 过充的根源）。
ROM_TAU = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80,
                    1.00, 1.50, 2.00, 3.00, 4.00, 5.00])
ROM_G = np.array([0.0, 0.680, 0.740, 0.790, 0.824, 0.854, 0.873, 0.886,
                  0.904, 0.922, 0.941, 0.970, 0.989, 1.000])


def g_shape(tau):
    """形状 ROM 插值：τ<0 取 0，τ>5s 取 1。"""
    return np.interp(np.asarray(tau, float), ROM_TAU, ROM_G, left=0.0, right=1.0)


def _cap(z, ded):
    """输出封顶：扣除量不得超过当前读数（只封顶输出值，不改内部状态）—— 同 v5.1。"""
    return max(z, 0.0) if ded > z else ded


def _smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class GLM53v6c(GLM53v51):
    """v6c = v5.1（慢相逐行沿用）+ 形状校正快速前端（替换免责期的空等）。"""

    # ── 形状校正参数（v6c 新增）────────────────────────────────────────────
    SHAPE_CORRECT = True       # 关掉即精确退化为 v5.1（A/B 对照与回归定位用）
    SHAPE_MIN_FRAC = 0.70      # shape(τ_real) 低于此值不校正：太早，形状先验不可信
    SHAPE_MAX_CORRECT = 1.15   # 校正系数上限（最多把已观测幅度上调 15%）
    SHAPE_RAMP_CLAMP = True    # 斜坡扣除量只增不减（防 restep 时 carry 大于新载荷扣除量而回退）
    ACCEL_G_INIT = True        # 交接时用"已观测蠕变"初始化 g（否则退化为 v5 的锚定值）
    STEP_PERSIST_ESTIMATE = 2.5  # 真实加载沿 = 确认帧 − STEP_PERSIST（回溯失败时的兜底）
    BACKDATE_S = 3.0           # 回溯真实加载沿的时间窗

    # ── 快相窗内的限速斜坡（把校正铺开，避免交接瞬间跳）──────────────────
    RAMP_RATE_FRAC = 0.5       # 斜坡限速系数量纲：A/s
    RAMP_RATE_MIN = 5.0        # 限速绝对下限（ADC/s），防止 A→0 时把斜坡卡死

    def __init__(self, n):
        super().__init__(n)
        # ── epoch 溯源 ──
        self.epoch_t = []            # epoch 起点时间戳（= v5 口径的 onset_ts）
        self.real_step_t = []        # 每个 epoch 回溯出的**真实加载沿**
        self.n_epoch = 0
        self.epoch_start = 0.0
        self.epoch_kind = "none"     # idle->load / restep
        self.epoch_ramp = 0.0        # 本 epoch 斜坡进度（调试/机制分析用）
        self.real_step_ts = None
        # ── 形状校正状态 ──
        self.A_obs = np.zeros(n)     # 形状校正前的已观测幅度（exempt 窗均值 − carry）
        self.shape_ratio = 1.0       # 当前校正系数（A_corr = A_obs · shape_ratio）
        self.ramp_target_est = np.zeros(n)   # 当前斜坡终点（逐帧刷新，只用于限速估计）
        self._obs = np.zeros(n)      # 已观测蠕变（Z − A_corr），逐通道
        self._obs_n = 0
        self._ded_now = np.zeros(n)  # 快相窗内当前生效的扣除量
        self._ramp_t0 = 0.0
        self._ramp_dur = 0.0
        self._ramp_applied = None    # 逐通道"实际已铺开"的扣除量（限速后的真值）
        self._ramp_final = None      # 快相窗末次生效的扣除量（交接连续性的唯一依据）
        self._ramp_limited = False
        self.handoff_ts = 0.0
        self._dt = 1e-4
        # ── 调试 ──
        self.debug = False
        self.dbg = []
        self.ramp_a = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # 机制分析用：A_obs/A_corr/b0/b1/t0/dur

    # ══════════════════ ① epoch 溯源：找"真实加载沿" ══════════════════
    def _backdate(self, ts):
        """在确认帧前 BACKDATE_S 内的环形缓冲里回溯真实加载沿。

        与 v6 的 `_backdate` 同源（v6 实测踩出来的：直接把命中帧当 t0 会系统性低估 Â），
        但这里只要一个**时间原点**去查形状曲线，不需要 base / hist。
        判据 = 相邻帧差分的鲁棒中位 + 3×MAD 之上取最大跳变帧；没有明显单帧跳变
        （台阶被平滑过）时退回"确认帧 − STEP_PERSIST"。
        """
        m = min(self._bn, self.CAP)
        if m < 8:
            return None
        tt = self._buf[0, :m]
        vv = self._buf[1, :m]
        o = np.argsort(tt)
        tt, vv = tt[o], vv[o]
        sel = (tt >= ts - self.BACKDATE_S) & (tt <= ts)
        tt, vv = tt[sel], vv[sel]
        if tt.size < 8:
            return None
        d = np.abs(np.diff(vv))
        if d.size == 0:
            return None
        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)))
        j = int(np.argmax(d))
        if d[j] > med + 3.0 * max(mad, 1e-9):
            return float(tt[j])
        return float(ts - self.STEP_PERSIST_ESTIMATE)

    def _mark_epoch(self, ts, kind):
        """记录一次 epoch 起点。`_begin`/`_restep` 每次确认都会调用，按时间戳去重。

        去重按"与上一个 epoch 起点的差值 ≤ 该 epoch 内即可判定为同一次"来做，
        这样既容得下 `_begin` 后被外部再调用一次，也不会把真正相邻的两次变载
        （v5 有 STEP_SUPPRESS=6s 抑制窗，不可能更近）误并成一次。
        """
        if self.n_epoch > 0 and abs(float(ts) - self.epoch_start) <= 0.5:
            return
        self.n_epoch += 1
        self.epoch_start = float(ts)
        self.epoch_kind = kind
        self.epoch_t.append(float(ts))
        rs = self._backdate(ts)
        self.real_step_ts = float(rs if rs is not None else ts - self.STEP_PERSIST_ESTIMATE)
        self.real_step_t.append(self.real_step_ts)
        # 新 epoch：清空"已观测蠕变"证据（旧载荷的蠕变与新载荷无关）
        self._obs = np.zeros(self.n)
        self._obs_n = 0
        self.ramp_target_est = self.carry.copy()
        self._ramp_applied = None
        self._ramp_final = None
        self._ramp_t0 = 0.0
        self._ramp_dur = 0.0
        self.epoch_ramp = 0.0

    def _begin(self, ts):
        super()._begin(ts)
        self._mark_epoch(ts, "idle->load")

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self._mark_epoch(ts, "restep")

    # ══════════════════ ② 形状校正系数 ══════════════════
    def _shape_tau(self, ts):
        """真实加载后经过的时间（缺溯源信息时按确认帧 − 2.5s 估计）。"""
        if self.real_step_ts is not None:
            return max(0.0, float(ts) - float(self.real_step_ts))
        return max(0.0, float(ts) - self.epoch_start + self.STEP_PERSIST_ESTIMATE)

    def _shape_ratio_at(self, ts):
        """校正系数 = 1 / clip(shape(τ_real), SHAPE_MIN_FRAC, 1)，上限 SHAPE_MAX_CORRECT。"""
        if (not self.SHAPE_CORRECT) or self.real_step_ts is None:
            return 1.0
        s = float(np.clip(float(g_shape(self._shape_tau(ts))),
                          self.SHAPE_MIN_FRAC, 1.0))
        return float(min(1.0 / s, self.SHAPE_MAX_CORRECT))

    # ══════════════════ ③ 快相窗的限速斜坡 ══════════════════
    def _ramp_begin(self):
        """配置本 epoch 的校正斜坡**窗口**（每个 epoch 只配置一次：起点 = epoch 起点）。

        起点定在 epoch 起点而不是 RAMP_TAU，是为了让进度在窗口内单调：进度一旦因
        "重配窗口"而回退，扣除量就会被拉回 carry，反而把稳定时刻推迟（实测把 9.9s
        拖到 12.1s，几乎退化成 v5）。窗口长度按**预计最大变化量**估一个安全的
        smoothstep 时长；真实速率另有 RAMP_RATE_FRAC 兜底限速。
        """
        b0 = self.carry.copy()
        delta = float(np.max(np.abs(self.ramp_target_est - b0))) if self.n else 0.0
        amp = float(self.A_obs.max()) if self.A_obs.size else 0.0
        base = float(self.STEP_PERSIST_ESTIMATE)
        rate_dur = delta / max(self.RAMP_RATE_FRAC * max(amp, 1.0), 1e-9)
        dur = float(np.clip(max(base, rate_dur), base, self.FAST_S))
        self._ramp_t0 = self.epoch_start
        self._ramp_dur = dur
        self._ramp_applied = b0.copy()
        self.ramp_a = (float(self.A_obs.max()), float(self.A_obs.max() * self.shape_ratio),
                       float(b0.max()), float(self.ramp_target_est.max()),
                       float(self._ramp_t0), float(dur))

    def _ramp_progress(self, ts):
        if self._ramp_dur <= 0.0:
            return 0.0
        return _smoothstep((float(ts) - self._ramp_t0) / self._ramp_dur)

    def _ref_mask(self, A_corr):
        """受载通道掩码。优先用由 A_corr 现算的掩码（快相窗内 self.loaded 还是旧 epoch 的，
        直接用它会让斜坡目标恒等于 carry）；只有在 A_obs 还没成形时才退回 self.loaded。"""
        amax = float(A_corr.max()) if A_corr.size else 0.0
        if amax > 1e-9:
            return (A_corr > self.LOADED_FRAC * amax) & (A_corr > 1e-9)
        if self.loaded.size and bool(self.loaded.any()):
            return self.loaded & (A_corr > 1e-9)
        return np.zeros(self.n, bool)

    def _ramp_target(self, ratio):
        """斜坡终点＝形状校正后幅度 + 已观测蠕变对应的逐通道扣除量。

        注意 `_obs ≡ Z − A_corr` ⇒ g_obs = (Z − A_corr)/A_corr，于是
        ded = γ·A_corr·g_obs 正好复现"按校正后幅度归一化"的蠕变扣除。

        参考通道掩码：快相窗内 `self.loaded` 还是本 epoch 之前的值（慢相才更新），
        所以这里优先用**由 A_corr 现算**的掩码 —— 否则快相窗内目标恒等于 carry、
        斜坡整段空转，加速完全失效（这是实测踩到的坑）。

        `SHAPE_RAMP_CLAMP` 让终点不低于本 epoch 起点的 carry：restep 时旧载荷的 carry
        可能大于新载荷的蠕变扣除量。若不封顶，斜坡会先向下（显示被抬起）再被 rate
        limiter 拉住，交接时 `g` 与末帧扣除量对不上，慢相还要再漂一段。
        封顶后斜坡**只增不减**，方向单一、速率可界，交接自洽。
        """
        A_corr = np.maximum(self.A_obs * float(ratio), 0.0)
        ded = self.carry.copy()
        if (not self.ACCEL_G_INIT) or A_corr.size == 0 or self.n == 0:
            return ded
        amax = float(A_corr.max())
        if amax > 1e-9:
            m = self._ref_mask(A_corr)
            if m.any():
                g_obs = float(np.clip(np.median(self._obs[m] / A_corr[m]), 0.0, 1.0))
                ded[m] = np.clip(self.gamma[m] * A_corr[m] * g_obs,
                                 self.CREEP_LO * A_corr[m], self.CREEP_HI * A_corr[m])
        if self.SHAPE_RAMP_CLAMP:
            ded = np.maximum(ded, self.carry)
        return ded

    def _fast_frame(self, ts, v, Z, holding):
        """快相窗（0 ≤ ts−epoch_start < FAST_S）的逐帧处理与输出。

        与 v5.1 的差别只在"输出用什么扣除量"：v5.1 恒为 carry，v6c 为
        carry →（限速 smoothstep）→ 校正后目标。g=0、不采 A、hold 语义完全一致。

        `holding=True`（pending 冻结，变载尚未确认）时不更新斜坡，
        以免把"还没确认的台阶"当成新载荷去校正 —— 这正是 v6 假事件的一个来源。
        """
        if holding:
            if self.hold_comp is None:
                self.hold_comp = self.carry.copy()
            return np.array([Z[i] - _cap(Z[i], self.carry[i]) for i in range(self.n)])

        ratio = self._shape_ratio_at(ts)
        A_obs = (self.ex_a / self.ex_n) if self.ex_n > 0 else (Z - self.carry)
        A_obs = np.maximum(A_obs, 0.0)
        self.A_obs, self.shape_ratio = A_obs, ratio
        A_corr = np.maximum(A_obs * float(ratio), 0.0)

        # 已观测蠕变证据：**必须与 A_obs 同源**（同一 exempt 窗均值减同一 carry）。
        # 用"本帧瞬时读数 − A_corr"会踩坑：restep 时快相窗内读数还在机械爬升，
        # 瞬时值系统性高于 A_corr，被误记成蠕变 ⇒ 斜坡终点过高、过充
        # （实测 切换负载-快相无责 @76s 多扣 108 ADC）。窗均值则与 A_obs 严格同源，
        # 只反映"窗内已经发生的蠕变"，正好是设计要用的量。
        reff = self._ref_mask(A_corr)
        if self.ex_n > 0:
            self._obs = np.maximum(self.ex_a / self.ex_n - self.carry - A_corr, 0.0)
            self._obs_n += 1
        self.ramp_target_est = self._ramp_target(ratio)
        if self._ramp_dur <= 0.0:
            self._ramp_begin()

        b0 = self.carry.copy()
        b1 = self.ramp_target_est
        p = self._ramp_progress(ts)
        ded = b0 + (b1 - b0) * p
        amp = max(float(A_corr.max()), 1.0)
        lim = max(self.RAMP_RATE_MIN, self.RAMP_RATE_FRAC * amp) * max(self._dt, 1e-4)
        if self._ramp_applied is None:
            self._ramp_applied = b0.copy()
        step = ded - self._ramp_applied
        if np.max(np.abs(step)) > lim:
            ded = self._ramp_applied + np.sign(step) * lim
            self._ramp_limited = True
        self._ramp_applied = ded
        self._ramp_final = ded.copy()
        self._ded_now = ded
        self.epoch_ramp = float(p)
        if self.debug and (not self.dbg or ts - self.dbg[-1][0] > 0.1):
            self.dbg.append((float(ts), "FAST", float(p), float(ratio),
                             float(A_obs.max()), float(ded.max())))
        out = Z.copy()
        if reff.any():
            for i in np.where(reff)[0]:
                out[i] = Z[i] - _cap(Z[i], ded[i])
        return out

    # ══════════════════ ④ 交接：把校正后的初值交给慢相 ══════════════════
    def _finalize_fast(self, ts, v):
        """免责期结束当帧：A = A_corr（形状校正后幅度）、g = 实际生效扣除量 / (γ·A_corr)。

        关键不变式（保证交接无缝）：
            γ · A_corr · g_handoff ≡ 快相窗末帧实际生效的扣除量
        因此 display 既不跳变、也不是"换个基准继续漂"，内部 (A, g) 与 display 自洽 ——
        慢相第一帧起就以正确的扣除量运行，不会出现 v6 那种交接后继续漂的暂态。
        """
        self.fast_done = True
        ratio = self._shape_ratio_at(ts)
        A_obs = (self.ex_a / self.ex_n) if self.ex_n > 0 else (v - self.carry)
        A_obs = np.maximum(A_obs, 0.0)
        self.A_obs, self.shape_ratio = A_obs, ratio
        self.A = np.maximum(A_obs * ratio, 0.0)
        amax = float(self.A.max())
        self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
            else np.zeros(self.n, bool)
        self.a_captured = True
        m = self.loaded & (self.A > 1e-9)
        # 补上交接当帧的蠕变证据（与 _fast_frame 同源：exempt 窗均值 − carry − A_corr）
        if m.any() and self.ex_n > 0:
            obs = np.maximum(self.ex_a / self.ex_n - self.carry - self.A, 0.0)
            self._obs = obs
        # (a) 加速项：用"当前读数 vs 校正后幅度"直接算出已发生的蠕变量
        #     g_obs = median((Z − A_corr) / A_corr) for loaded channels
        #     这就是免责期结束时的真实蠕变比例。以此初始化 g 而不是从 0 开始，
        #     慢相模块就不需要 9s 从头追。
        #     对 onset（carry=0）：g_obs 就是"快相尾巴 + 已开始的蠕变"相对 A_corr 的比例
        #     对 restep（carry≠0）：同时考虑 carry 锚定值和观测值，取较大者保持连续
        if self.ACCEL_G_INIT and m.any():
            # Observation-based g: directly from current reading vs corrected amplitude
            g_obs = float(np.clip(
                np.median((v[m] - self.A[m]) / self.A[m]), 0.0, self.CREEP_HI))
            # Carry-based g: continuity anchor (v5 default)
            g_carry = 0.0
            if np.any(self.carry[m] > 1e-9):
                g_carry = float(np.clip(
                    np.median(self.carry[m] / (self.gamma[m] * self.A[m])), 0.0, 1.0))
            # Ramp-based g: from actual deduction applied during fast phase
            g_ramp = 0.0
            if self._ramp_final is not None and np.any(self._ramp_final[m] > 1e-9):
                g_ramp = float(np.clip(
                    np.median(self._ramp_final[m] / (self.gamma[m] * self.A[m])),
                    0.0, self.CREEP_HI))
            # Take the maximum: ensures we don't lose carry continuity, but also
            # accelerate when observation shows more creep than carry accounts for
            g_new = max(g_obs, g_carry, g_ramp)
        elif m.any():
            g_new = float(np.clip(
                np.median(self.carry[m] / (self.gamma[m] * self.A[m])), 0.0, 1.0))
        else:
            g_new = 0.0
        self.g = g_new
        self.carry = np.zeros(self.n)
        self.ex_a = np.zeros(self.n)
        self.ex_n = 0
        self._obs = np.zeros(self.n)
        self._obs_n = 0
        self._ramp_dur = 0.0
        self._ramp_applied = None
        self._ded_now = np.zeros(self.n)
        self.handoff_ts = float(ts)
        if self.debug:
            self.dbg.append((float(ts), "HANDOFF", float(ratio), float(amax),
                             float(g_new), float(self.epoch_ramp)))

    # ══════════════════ 主流程（v5.1 逐行沿用，仅替换免责期分支）══════════════════
    def process(self, ts, v):
        v = np.asarray(v, dtype=float).copy()
        total = float(v.sum())
        self._push(ts, total)
        dt = 0.0
        if self.first:
            self.first = False
            self.last_ts = self.t0 = ts
            self.ts_smooth = self.fast = self.slow = total
            self.min_ts = self.max_ts = self.level_ref = total
        else:
            dt = ts - self.last_ts
            self.last_ts = ts
            dt = 0.0 if not (dt > 0.0) else min(dt, 0.1)
            if dt > 0.0:
                self.ts_smooth += (dt / self.TAU_TOTAL) * (total - self.ts_smooth)
                self.fast += (dt / self.TAU_FAST) * (total - self.fast)
                self.slow += (dt / self.TAU_SLOW) * (total - self.slow)
                self.level_ref += (dt / self.TAU_LEVEL) * (self.ts_smooth - self.level_ref)
        self.min_ts = min(self.min_ts, self.ts_smooth)
        self.max_ts = max(self.max_ts, self.ts_smooth)
        self._dt = dt if dt > 0.0 else 1e-4
        eps = self._eps()
        div = abs(self.fast - self.slow)
        thr = (max(self.STEP_REL * max(self.slow, eps), self.STEP_ABS * self.max_ts)
               if self.in_load else
               max(self.ONSET_REL * max(self.slow, eps), self.STEP_ABS * self.max_ts))

        # ── ① 直接电平差判据（近期 0.3s 窗 vs 0.3~0.8s 前窗；与 div 判据取或）──
        lv_now = self._win_mean(ts - self.LEV_FAST_S, ts) if self._bn > 3 else None
        lev_ok = False
        if self.in_load and lv_now is not None:
            if self.lev_latch is not None:
                lev_ok = abs(lv_now - self.lev_latch) > self.PENDING_RESET * self.lev_thr
                if not lev_ok:
                    self.lev_latch = None            # 电平回来了 → 瞬态，取消
            elif (ts - self.onset_ts) > self.LEV_ARM_S:
                lv_ref = self._win_mean(ts - self.LEV_FAST_S - self.LEV_LAG_S,
                                        ts - self.LEV_FAST_S)
                if lv_ref is not None:
                    self.lev_thr = max(self.LEV_REL * max(lv_ref, eps),
                                       self.LEV_ABS_FRAC * self.max_ts)
                    if abs(lv_now - lv_ref) > self.lev_thr:
                        self.lev_latch = lv_ref
                        lev_ok = True

        step_now = (div > thr) if self.in_load else (div > thr and self.fast > self.slow)
        if self.in_load:
            step_now = step_now or lev_ok
        if step_now:
            if not self.pending:
                self.pending, self.pending_ts = True, ts
        elif self.pending and (not lev_ok) and div < self.PENDING_RESET * thr:
            self.pending = False
        step_conf = self.pending and (ts - self.pending_ts > self.STEP_PERSIST)
        u = ts - self.onset_ts if self.in_load else 0.0
        idle = (self.ts_smooth < self.IDLE_FRAC * max(self.level_ref, eps) or
                self.ts_smooth < self.UNLOAD_MIN_RATIO * self.min_ts + eps)

        if not self.in_load:
            if abs(ts - self.t0) <= 1e-9:
                if total > eps:
                    self._begin(ts)
                    self._align(self.ts_smooth)
            elif step_conf:
                self._begin(ts)
                self._align(self.fast)
            self.hold = False
        elif u > self.UNLOAD_FAST and idle:
            self.in_load = False
            self._align(self.ts_smooth)
            self.hold, self.hold_comp = False, None
            self.pending, self.pending_ts = False, 0.0
            self.fast_done = True
            self.lev_latch = None
        elif u > self.STEP_SUPPRESS and step_conf:
            if idle:
                self.in_load = False
                self._align(self.ts_smooth)
                self.hold, self.hold_comp = False, None
                self.pending, self.pending_ts = False, 0.0
                self.fast_done = True
                self.lev_latch = None
            else:
                self._restep(ts, v)                 # v5.1: 不再减基线
                self._align(self.fast)
        elif self.pending:
            self.hold = True
            self.a_new_acc = self.a_new_acc + v      # v5.1: 不再减基线
            self.a_new_frames += 1
        else:
            self.hold, self.hold_comp = False, None
            if self.a_new_frames > 0:
                self.a_new_acc = np.zeros(self.n)
                self.a_new_frames = 0

        # ═══════════ 快相（免责期）——v6c 唯一替换掉 v5.1 行为的地方 ═══════════
        if self.in_load and not self.fast_done:
            ux = ts - self.onset_ts
            if ux < self.FAST_S:
                self.a_captured = False
                self.a_acc = np.zeros(self.n)
                self.a_frames = 0
                self.g = 0.0
                if ux >= self.FAST_S - self.EXEMPT_AWIN:
                    self.ex_a = self.ex_a + (v - self.carry)
                    self.ex_n += 1
                return self._fast_frame(ts, v, v, bool(self.hold))
            # ── 免责期结束当帧：形状校正 + 加速初值 ──
            self._finalize_fast(ts, v)
            # 不 return：当帧继续走下面的 v5.1 慢相逻辑（与 v5.1 插入点一致）
        # ══════════════════════════════════════════════════════════════════════

        Z = v.copy()
        if not self.in_load:
            return Z
        if not self.a_captured:
            if self.A_W0 <= u <= self.A_W1:
                self.a_acc = self.a_acc + Z
                self.a_frames += 1
            if u > self.A_W1:
                self.A = self.a_acc / self.a_frames if self.a_frames > 0 else Z.copy()
                amax = self.A.max()
                self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                    else np.zeros(self.n, bool)
                self.a_captured = True
            else:
                return Z
        if self.hold:
            if self.hold_comp is None:
                self.hold_comp = self._deduction()
            m = self.loaded & (self.A > 1e-9)
            out = Z.copy()
            if m.any():
                out[m] = [Z[i] - _cap(Z[i], self.hold_comp[i])
                          for i in range(self.n) if m[i]]
            return out
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
                                              self.GAMMA_MIN, self.GAMMA_MAX),
                                      1.0)
        out = Z.copy()
        if m.any():
            out[m] = [Z[i] - _cap(
                Z[i],
                float(np.clip(self.gamma[i] * self.A[i] * self.g,
                              self.CREEP_LO * self.A[i], self.CREEP_HI * self.A[i])))
                for i in range(self.n) if m[i]]
        return out


def run_glm53_v6c(t, X, **kw):
    """便捷运行器：与 `run_glm53_v3` 同形，返回 (Y, compensator)。"""
    c = GLM53v6c(np.asarray(X).shape[1])
    for k, val in kw.items():
        setattr(c, k, val)
    Y = np.empty_like(np.asarray(X, dtype=float))
    for i in range(len(t)):
        Y[i] = c.process(t[i], X[i])
    return Y, c
