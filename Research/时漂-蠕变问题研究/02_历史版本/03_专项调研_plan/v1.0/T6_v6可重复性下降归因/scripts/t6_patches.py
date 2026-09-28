# -*- coding: utf-8 -*-
"""T6 消融/改进补丁类集合（全部派生自 `t6_a_common.make_traced(GLM53v6)` 的仪表化 v6）。

**原则**：每个补丁类只改**一处**；把该处设回原型默认值时，**必须与原型的逐帧输出完全一致**
（零差证明由 `t6c_attribution.py` 产出 `results/t6_patch_ab_zero.csv`）。
不做任何 `_run_event` 整体复制（除 Δ5 的一处前置 guard，且默认值下短路、行为不变）。

消融项（T6-Q4，v6 相对 v5.1 的 5 处改动逐条"退回去"）：
  D1_confirm250  DET_PERSIST 3 → 250 帧（≈2.5 s 硬确认，模拟 v5.1 的 STEP_PERSIST=2.5 s）
  D1_confirm50   DET_PERSIST 3 → 50 帧（≈0.5 s）—— 给一条梯度
  D2_noshape     KAPPA_ONSET/RESTEP 1.30/1.12 → 1.00（Â = 实测增量，取消形状前置量）
  D2_kappa110    κ 1.30/1.12 → 1.10/1.10（**C-4 占位**：T5-A 未交付，禁止作为最终裁决）
  D3_norate      RATE_MAX 0.8 → 50.0（取消滑行器速率限制）
  D3_rate04      RATE_MAX 0.8 → 0.4（速率更紧的对照）
  D4_ho1s        HO_MIN 5.0 → 1.0（交接提前）
  D4_ho10s       HO_MIN 5.0 → 10.0（交接推后）
  D5_delay5s     起扣推迟 5 s（对齐 v5 的 5 s 免责期；默认 0 = 与原型一致）

改进项（T6-Q5/Q6）：
  I1_ahgate      Â 的"形状拟合残差门控收缩"：λ = clip(1 − resid/R0, 0, 1)，Â←inc+λ(Â−inc)
  I2_cooldown    REVOKE_COOLDOWN 0.30 → 1.00 s（撤销后的冷却 = 迟滞，抑制反复建/撤）
  I3_tailgate    TAIL_GATE_FRAC 0.10 → 0.05（epoch 尾巴内的"变载"门限更严，少建边缘事件）
  I4_guard       **零成本口径类**：不改算法，只在验收层把 R5/Rstep 两口径与事件台账并列
                 （在报告里给，不占运行）
"""
import numpy as np

import t6_a_common as AC                                            # noqa: E402
from t6_glm53_v6 import GLM53v6, ROM_TAU, ROM_G                    # noqa: E402

BASE = AC.make_traced(GLM53v6)


class P1a(BASE):
    """Δ1：2.5 s 硬确认（v5.1 语义）。"""
    DET_PERSIST = 250


class P1b(BASE):
    """Δ1 梯度：0.5 s 确认。"""
    DET_PERSIST = 50


class P2a(BASE):
    """Δ2：取消形状前置量（Â 只能等于实测增量）。"""
    KAPPA_ONSET, KAPPA_RESTEP = 1.00, 1.00


class P2b(BASE):
    """Δ2 占位：第一轮用的 κ=1.10（**须由 T5-A 统一扫描裁决**）。"""
    KAPPA_ONSET, KAPPA_RESTEP = 1.10, 1.10


class P3a(BASE):
    """Δ3：取消滑行器速率限制（无限速率）。"""
    RATE_MAX = 50.0


class P3b(BASE):
    """Δ3 对照：速率更紧。"""
    RATE_MAX = 0.40


class P4a(BASE):
    """Δ4：交接提前到 τ=1 s。"""
    HO_MIN = 1.00


class P4b(BASE):
    """Δ4：交接推后到 τ=10 s。"""
    HO_MIN = 10.00


class P5(BASE):
    """Δ5：把"起扣"推迟 START_DELAY 秒（= 对齐 v5 的 5 s 免责期）。

    START_DELAY=0.0（默认）时该 guard 短路，与原型逐帧一致（见 patch_ab_zero）。
    保真度限制：免责窗内同时不做 revoke/unload 判定 —— 这与 v5.1 的免责期语义一致，
    但因此 Δ5 同时动了"起扣时刻"与"免责窗内的撤销/卸载处理"两件事，报告里已声明。
    """
    START_DELAY = 0.0

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        if ev is not None and self.START_DELAY > 0.0 and (ts - ev["t0"]) < self.START_DELAY:
            inc = total - ev["base"]
            if (ts - ev["t0"]) <= 1.0:
                ev["hist"].append((float(ts - ev["t0"]), float(inc)))
            return v
        return super()._run_event(ts, v, total, dt, eps, idle_now)


class I1(BASE):
    """I1：Â 的形状拟合残差门控收缩（默认 R0=0 → 不收缩，与原型一致）。"""
    AH_RESID_R0 = 0.0

    def _inv_est(self, hist, tau, kappa):
        A, ok = super()._inv_est(hist, tau, kappa)
        if not ok or self.AH_RESID_R0 <= 0.0:
            return A, ok
        tt = np.array([h[0] for h in hist])
        yy = np.array([h[1] for h in hist])
        m = (tt >= self.TAU_REF) & (tt <= min(tau, self.TAU_REF + self.AWIN))
        if m.sum() < 5:
            return A, ok
        g = np.interp(tt[m], ROM_TAU, ROM_G, left=0.0, right=1.0)
        resid = float(np.sqrt(np.mean((yy[m] - A * g) ** 2)) / max(np.sqrt(np.mean(yy[m] ** 2)), 1e-12))
        lam = float(np.clip(1.0 - resid / self.AH_RESID_R0, 0.0, 1.0))
        inc = float(yy[-1])
        return float(inc + lam * (A - inc)), True


class I1f(BASE):
    """I1f：Â 的**固定比例**收缩 Â←inc+λ(Â−inc)；λ=1.0（默认）= 原型不变。"""
    AH_LAMBDA = 1.0

    def _inv_est(self, hist, tau, kappa):
        A, ok = super()._inv_est(hist, tau, kappa)
        if not ok or self.AH_LAMBDA >= 1.0:
            return A, ok
        inc = float(hist[-1][1]) if hist else 0.0
        return float(inc + self.AH_LAMBDA * (A - inc)), True


class I6(BASE):
    """I6：Â 的估计量换成『窗内均值比』Â = mean(y)/mean(g)，替代最小二乘 Σyg/Σg²。

    动因（T6-Q2 的机制结论）：v6 的 Â 对**回溯真沿 t0** 极敏感（t0 是逐帧离散判据，
    噪声/时序抖动会让它偏 1~2 帧），而 t0 偏移会整体平移形状拟合窗 ⇒ Â 变化 100 ADC 量级、
    并被 pin 永久保留。窗内**均值比**对窗整体平移不敏感（窗长 0.6 s vs 偏移 0.02 s）。
    默认 AH_MODE="ls" ⇒ 与原型逐帧一致（零差对照）。
    """
    AH_MODE = "ls"

    def _inv_est(self, hist, tau, kappa):
        if self.AH_MODE == "ls":
            return super()._inv_est(hist, tau, kappa)
        if len(hist) < 4 or tau < self.TAU_REF:
            return 0.0, False
        tt = np.array([h[0] for h in hist])
        yy = np.array([h[1] for h in hist])
        m = (tt >= self.TAU_REF) & (tt <= min(tau, self.TAU_REF + self.AWIN))
        if m.sum() < 5:
            return 0.0, False
        g = np.interp(tt[m], ROM_TAU, ROM_G, left=0.0, right=1.0)
        gm = float(g.mean())
        inc = float(yy[-1])
        if gm < 1e-9 or inc <= 0.0:
            return 0.0, False
        A = float(yy[m].mean()) / gm
        A = max(A, inc)
        return float(min(A, kappa * inc)), True


class I7(BASE):
    """I7：**撤销判据加迟滞** —— 要求"增量掉到 inc_max 一半以下"连续 N 帧才允许撤销。

    动因（T6-Q3 的逐事件证据）：±1 包时序抖动下 v6 最常见的失败是 **onset 建事件的同一时刻
    立刻被 revoke**（序列里出现 `E:onset` 与 `R` 同帧），此后该次加载整段不再补偿、
    显示停在原始读数上约 10 s（实测 RMS 差 1615 ADC）。根因是撤销判据用的
    `inc_s = _win_mean(ts−0.10, ts)` 在"整包到达"的时序下会跨过抬升沿、瞬时读到 0。
    修法是纯粹的迟滞（不改阈值）：把判定持续 N 帧才算数。
    默认 REVOKE_HOLD_N=1 ⇒ 与原型逐帧完全一致（零差对照）。
    """
    REVOKE_HOLD_N = 1

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        if ev is None or self.REVOKE_HOLD_N <= 1:
            return super()._run_event(ts, v, total, dt, eps, idle_now)
        tau = ts - ev["t0"]
        inc_s = self._win_mean(ts - 0.10, ts)
        inc_s = (inc_s - ev["base"]) if inc_s is not None else (total - ev["base"])
        block = False
        if tau < self.REVOKE and ev["inc_max"] > eps and inc_s < 0.5 * ev["inc_max"]:
            ev["_rvn"] = ev.get("_rvn", 0) + 1
            if ev["_rvn"] < self.REVOKE_HOLD_N:
                ev["_saved_incmax"] = ev["inc_max"]
                ev["inc_max"] = 0.0           # 临时屏蔽父类撤销判据（该判据只用 inc_max）
                block = True
        else:
            ev["_rvn"] = 0
        out = super()._run_event(ts, v, total, dt, eps, idle_now)
        if block and self.ev is not None:
            self.ev["inc_max"] = self.ev.get("_saved_incmax", self.ev["inc_max"])
        return out


class I2(BASE):
    """I2：撤销后冷却（迟滞），默认 0.30 = 原型值。"""
    REVOKE_COOLDOWN = 0.30


class I3(BASE):
    """I3：epoch 尾巴内的变载门限，默认 0.10 = 原型值。"""
    TAIL_GATE_FRAC = 0.10


class I1I2(I1):
    """组合臂：残差门控收缩 + 撤销冷却（用于验证"两条正交改进能否叠加"）。"""
    AH_RESID_R0 = 0.10
    REVOKE_COOLDOWN = 0.30


# ── 零差证明清单： (类, 该类的旋钮, 恢复默认值的参数) ──
ZERO_AB = [
    ("D1_confirm250", P1a, dict(DET_PERSIST=3)),
    ("D1_confirm50", P1b, dict(DET_PERSIST=3)),
    ("D2_noshape", P2a, dict(KAPPA_ONSET=1.30, KAPPA_RESTEP=1.12)),
    ("D2_kappa110", P2b, dict(KAPPA_ONSET=1.30, KAPPA_RESTEP=1.12)),
    ("D3_norate", P3a, dict(RATE_MAX=0.8)),
    ("D3_rate04", P3b, dict(RATE_MAX=0.8)),
    ("D4_ho1s", P4a, dict(HO_MIN=5.00)),
    ("D4_ho10s", P4b, dict(HO_MIN=5.00)),
    ("D5_delay5s", P5, dict(START_DELAY=0.0)),
    ("I1_ahgate", I1, dict(AH_RESID_R0=0.0)),
    ("I1f_shrink", I1f, dict(AH_LAMBDA=1.0)),
    ("I2_cooldown", I2, dict(REVOKE_COOLDOWN=0.30)),
    ("I3_tailgate", I3, dict(TAIL_GATE_FRAC=0.10)),
    ("I7_revoke_hyst", I7, dict(REVOKE_HOLD_N=1)),
    ("I1I2_combo", I1I2, dict(AH_RESID_R0=0.0, REVOKE_COOLDOWN=0.30)),
]

# ── 消融臂（T6-Q4）──
ABLATION = [
    ("v6_baseline", BASE, {}),
    ("D1_confirm250", P1a, dict(DET_PERSIST=250)),
    ("D1_confirm50", P1b, dict(DET_PERSIST=50)),
    ("D2_noshape", P2a, dict(KAPPA_ONSET=1.00, KAPPA_RESTEP=1.00)),
    ("D2_kappa110", P2b, dict(KAPPA_ONSET=1.10, KAPPA_RESTEP=1.10)),
    ("D3_norate", P3a, dict(RATE_MAX=50.0)),
    ("D3_rate04", P3b, dict(RATE_MAX=0.40)),
    ("D4_ho1s", P4a, dict(HO_MIN=1.00)),
    ("D4_ho10s", P4b, dict(HO_MIN=10.00)),
    ("D5_delay5s", P5, dict(START_DELAY=5.00)),
]

# ── 改进臂（T6-Q5 / Q6）：(名称, 类, 生效参数, 该类旋钮的"原型默认值" = 零差对照) ──
#    说明：`HO_MIN` / `DET_*` 这类**纯数值旋钮**直接用 `BASE + kw` 表达（因为 run_traced 是
#    setattr 到实例上），比再写一个子类更能保证"只改一处"；其零差对照就是该旋钮的默认值。
IMPROVE = [
    ("v6_baseline", BASE, {}, {}),
    ("I4_ho10s", BASE, dict(HO_MIN=10.00), dict(HO_MIN=5.00)),
    ("I4_ho15s", BASE, dict(HO_MIN=15.00), dict(HO_MIN=5.00)),
    ("I6_rommean", I6, dict(AH_MODE="rom"), dict(AH_MODE="ls")),
    ("I6_rommean_I4ho10", I6, dict(AH_MODE="rom", HO_MIN=10.00),
     dict(AH_MODE="ls", HO_MIN=5.00)),
    ("I7_revoke_hyst3", I7, dict(REVOKE_HOLD_N=3), dict(REVOKE_HOLD_N=1)),
    ("I7_h3_I4ho10", I7, dict(REVOKE_HOLD_N=3, HO_MIN=10.00),
     dict(REVOKE_HOLD_N=1, HO_MIN=5.00)),
    ("I5_delay3s", P5, dict(START_DELAY=3.00), dict(START_DELAY=0.0)),
    ("I1f_shrink_050", I1f, dict(AH_LAMBDA=0.50), dict(AH_LAMBDA=1.0)),
    ("I2_cooldown_1s", I2, dict(REVOKE_COOLDOWN=1.00), dict(REVOKE_COOLDOWN=0.30)),
    ("I3_tail_005", I3, dict(TAIL_GATE_FRAC=0.05), dict(TAIL_GATE_FRAC=0.10)),
]
