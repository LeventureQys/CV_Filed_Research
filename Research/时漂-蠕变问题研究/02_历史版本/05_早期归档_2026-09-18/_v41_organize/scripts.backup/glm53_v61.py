# -*- coding: utf-8 -*-
"""GLM53 v6.1 原型：修复「切换负载下阶跃仍带来尖峰」（对应 `Document/08-v6.1算法说明.md`）。

现象（实测，见 `Document/08-v6.1算法说明.md` §1）：
  同一条显示链路上，"阶跃"在两类数据上的表现不同 ——
    · **恒载 9 组（单一负载）**：阶跃后显示冲到真值之上 3.5%（中位），但原始随后继续爬升
      （载荷内 +10~45%），很快把这点前置量"吃掉"，所以看不出冲高回落；且该族电平只有 ~20 个
      显示单位，4% 的偏差绝对量只有 0.6，肉眼不可见。
    · **变载实录（切换负载）**：加载方式是"快速到位 + 保压"，原始不再爬升 ⇒ 前置量**没有东西
      来吃掉**。实测 `切换负载-快相无责` @185.98 s 的 onset：Â 高估 **+10.9%**，显示冲到
      真值之上 +3,696 ADC（0.5 s 内），随后被停滞检测一步拉回原始 ⇒ **尖峰**（峰后 2.2 s 掉回）。

根因（逐帧核对，同 §2）：
  1. 显示在快相内被钉在 `base_y + Â`，而逆模型的形状库是**13 份录制的中位形状**；
     当现场这一次加载比标定形状"更快到位"（本例 0.2 s 时已走完真值增量的 88%，ROM 只认 79%）
     时，`Â = Σyg/Σg²` 必然高估 —— 这是系统性偏差，不是噪声；
  2. 高估出来的前置量在"输入停住"时无处可去：停滞检测把 Â 一步改写为实测增量，
     显示以 0.8/s×幅度 的速率限幅回撤（本例 3,100 ADC 只用了 0.14 s）⇒ 尖峰的下降沿；
  3. 交接（τ=5 s）后 pin 锚定把 Â 的高估**永久保留**，于是同一份数据 @133.84 s 表现为
     +4.9% 的"过充平台"（即 §12.6 ② 那个 +875）。

v6.1 的修复（**主修复只有一处，另加一处安全网**；慢相模块与 v6 完全一致）：
  **F1 保守形状反演（主修复）**：形状库按"最快实测形状"上包络重标（`ROM_SCALE`，只对 onset 类生效），
     使 Â 在"输入比标定更快"时不再系统性高估。取值的依据见 §3.2（21 个 onset 的 Â/真值 分布）。
  **F2 前置量下行限速（安全网）**：目标下调时不再用上行速率（0.8/s）回撤，改用 `RATE_DOWN`（默认 0.25/s）。
     实测对聚合指标影响很小（恒载回落 max 0.54%→0.43%），作用是给"尚未见过的更快加载"留一条缓降路径。
  F3/F4（停滞确认提前、停滞尾巴保留）实测为中性，v6.1 **不改动 v6 行为**，仅保留可复现开关。

继承关系：本类只覆写 `_inv_est` 与 `_run_event`，其余（检测器、分类、滑行器、
慢相模块、卸载/减重/重锚路径）**逐行沿用 v6**，保证 A/B 是同一套机器。
"""
import numpy as np

from glm53_v6 import GLM53v6                                  # noqa: E402


def g61_shape(tau, scale):
    """v6.1 形状：v6 ROM 上包络重标 g61 = min(1, scale·g_v6)。"""
    from glm53_v6 import ROM_TAU, ROM_G
    g = np.interp(np.asarray(tau, float), ROM_TAU, ROM_G, left=0.0, right=1.0)
    return np.minimum(1.0, scale * g)


class GLM53v61(GLM53v6):
    # ── F1（主修复）：保守形状反演 ──
    #  ROM_SCALE：形状库按"最快实测形状"上包络重标。1.0 = v6 原样。
    #  取值依据（`db_v61_overest.py`，21 个 onset 的 Â/真值）：
    #    1.00 → 偏差 中位 +3.2%、max +10.9%（= 尖峰）；
    #    1.06 → 中位 −2.6%、max +4.6%（采用：中位两侧均衡，最坏超调砍掉 58%）
    #    1.08 → 中位 −4.4%、max +2.7%；1.12 → 中位 −7.9%、max −1.0%（永不超调，但欠报明显）
    ROM_SCALE = 1.06
    #  CONSERVATIVE_ONSET_ONLY：只对 onset 类生效。restep 的目标已被 κ=1.12 限幅，
    #  ROM 的误差到不了显示；对它一起收紧只会压低台阶捕获比（实测 0.91→0.87）。
    CONSERVATIVE_ONSET_ONLY = True
    # ── F2 ──
    RATE_DOWN = 0.25          # 下行（前置量回撤）速率上限，相对幅度 /s；0 = 用上行速率
    # ── F3（停滞确认时间）──
    #  曾把 0.45 s 收到 0.35 s；`df_v61_ablate.py` 的分项归因显示：在 F1 打开后，
    #  0.35 与 0.45 的超调/回落指标**逐项相同**（实采 4.65/165.4、恒载 0.45/0.54），
    #  即 F3 是中性改动 ⇒ 回到 v6 取值，v6.1 不再动它（保留属性便于复现扫描）。
    STALL_HOLD_S = 0.45
    # ── F4（停滞时的尾巴保留）──
    #  0 = 与 v6 相同（停滞即把目标改写为实测增量）；扫描显示 0/0.05/0.10 对指标无差别，
    #  故默认 0（不改 v6 行为），仅作为可复现开关保留。
    STALL_TAIL_KEEP = 0.0
    # ── 诊断 ──
    n_down_clip = 0

    def _rom_scale(self, kind):
        if self.CONSERVATIVE_ONSET_ONLY and kind != "onset":
            return 1.0
        return self.ROM_SCALE

    def _inv_est(self, hist, tau, kappa):
        """电平域最小二乘，但用保守形状 g61。"""
        if len(hist) < 4 or tau < self.TAU_REF:
            return 0.0, False
        tt = np.array([h[0] for h in hist])
        yy = np.array([h[1] for h in hist])
        m = (tt >= self.TAU_REF) & (tt <= min(tau, self.TAU_REF + self.AWIN))
        if m.sum() < 5:
            return 0.0, False
        scale = self._rom_scale(self.ev["kind"] if self.ev is not None else "onset")
        g = g61_shape(tt[m], scale)
        den = float((g * g).sum())
        if den < 1e-12:
            return 0.0, False
        inc = float(yy[-1])
        if inc <= 0.0:
            return 0.0, False
        A = float((yy[m] * g).sum()) / den
        A = max(A, inc)
        return float(min(A, kappa * inc)), True

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        """与 v6 逐行相同，只改滑行器的速率限幅（F2）、停滞确认与停滞目标（F3/F4）。"""
        ev = self.ev
        tau = ts - ev["t0"]
        inc = total - ev["base"]
        inc_s = self._win_mean(ts - 0.10, ts)
        inc_s = (inc_s - ev["base"]) if inc_s is not None else inc
        ev["inc_max"] = max(ev["inc_max"], inc_s)

        if tau < self.REVOKE and ev["inc_max"] > eps and inc_s < 0.5 * ev["inc_max"]:
            self.n_revoke += 1
            prev_state = ev["prev_state"]
            y_prev = ev["y_prev"]
            self.ev = None
            self.state = prev_state if prev_state in ("idle", "slow") else "idle"
            self.last_ev_ts = ts
            if prev_state == "idle" and y_prev is not None:
                return np.asarray(y_prev, float)
            self.last_out = v.copy()
            return v

        ev["dec_max"] = min(ev["dec_max"], float(inc_s))
        if (ev["kind"] == "decrease" and tau < self.REVOKE and ev["dec_max"] < -eps
                and inc_s > 0.5 * ev["dec_max"]):
            self.n_revoke += 1
            prev_state = ev["prev_state"]
            self.ev = None
            self.state = prev_state if prev_state in ("idle", "slow") else "idle"
            self.last_ev_ts = ts
            self.last_out = v.copy()
            return v

        if tau > self.UNLOAD_FAST and idle_now:
            self._to_idle()
            return v

        if tau <= 1.0:
            ev["hist"].append((float(tau), float(inc)))
        ev["recent"].append((float(tau), float(inc_s)))
        while ev["recent"] and ev["recent"][0][0] < tau - 0.8:
            ev["recent"].pop(0)

        if ev["kind"] == "restep" and tau >= 0.30 and ev["base"] < 0.5 * total:
            ev["kind"] = "onset"

        if ev["kind"] == "decrease":
            if (ts - ev["t_det"]) < self.DECREASE_SETTLE:
                if self.hold_comp is None:
                    self.hold_comp = self._deduction_vector(v)
                m = self.loaded & (self.A > 1e-9)
                out = v.copy()
                if m.any():
                    from glm53_v6 import _cap
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
        if ok and not ev["stalled"]:
            ev["A_hat"] = A_hat

        # ── 停滞检测（判据同 v6，只把确认时间提前 = F3）──
        if (ok and not ev["stalled"] and tau > self.STALL_START_S
                and ev["kind"] in ("onset", "restep", "restep_reload")):
            amp = max(abs(ev["A_hat"]), eps)
            if ev["inc_ref"] is None and tau >= self.TAU_REF and len(ev["hist"]) >= 4:
                tt = np.array([h[0] for h in ev["hist"]])
                yy = np.array([h[1] for h in ev["hist"]])
                ev["inc_ref"] = float(np.interp(self.TAU_REF, tt, yy))
            ref = ev["inc_ref"]
            stalled_now = False
            sc = self._rom_scale(ev["kind"])
            if ref is not None and ref > self.STALL_MIN_FRAC * amp:
                gr = float(g61_shape(self.TAU_REF, sc))
                ratio_mod = (float(g61_shape(tau, sc)) - gr) / gr
                ratio_obs = (inc_s - ref) / ref
                if ratio_mod > 0.03 and ratio_obs < self.STALL_TAIL_FRAC * ratio_mod:
                    stalled_now = True
            if stalled_now:
                ev["stall_t"] += dt
            else:
                ev["stall_t"] = 0.0
            if ev["stall_t"] >= self.STALL_HOLD_S:
                ev["stalled"] = True
                # F4：不再一步等于实测增量，保留一小段"可能还会补上"的尾巴
                ev["A_hat"] = max(float(inc_s) + self.STALL_TAIL_KEEP * max(ev["A_hat"], 0.0),
                                  0.0)
        if ev["stalled"]:
            ev["A_hat"] = max(float(inc_s) + self.STALL_TAIL_KEEP * max(ev["A_hat"], 0.0), 0.0)

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
        c_target = ev["c0"] * (1.0 - W) + (target - total) * W
        amp = max(abs(ev["A_hat"]), eps)
        # ── F2：上行走 RATE_MAX，下行走 RATE_DOWN ──
        rate = self.RATE_MAX
        if self.RATE_DOWN > 0.0 and c_target < ev["c_applied"]:
            rate = min(self.RATE_MAX, self.RATE_DOWN)
            self.n_down_clip += 1
        lim = rate * amp * max(dt, 1e-4)
        if abs(c_target - ev["c_applied"]) > lim:
            c_target = ev["c_applied"] + np.sign(c_target - ev["c_applied"]) * lim
        ev["c_applied"] = c_target

        tau_ho = max(self.HO_MIN, ev["tau_g0"] + ev["Tglide"])
        if tau >= tau_ho and ok:
            self._handoff(ts, v, ev)
            return self._slow_step(ts, v, dt)
        share = self._share_vector(ev, v)
        return v + share * c_target
