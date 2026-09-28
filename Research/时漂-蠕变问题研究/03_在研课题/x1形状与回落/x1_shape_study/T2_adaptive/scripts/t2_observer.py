# -*- coding: utf-8 -*-
"""x1 形状自适应候选变体（T2 研究问题 2）。

全部变体共用同一套「零点跟踪 + x2 慢态」骨架（与 v34_observer_core3 逐帧同式），
只替换 x1 快态的**幅度/时间常数/形状**：

  base   基线：r1 固定 0.12，单指数 τc1=8（= v34_observer_core3.observe3）
  A_amp  幅度自适应：r1_eff 由在线估计的「蠕变比 α = x1/y」低通跟踪得到，
         并对剩余误差加上一个带界的加速学习项（保证首点收敛不会太慢）
  A_store 幅度自适应（简化版，无加速项）：纯低通 α，r1_eff = clamp(c·α)
  B_tau  时间常数自适应：τc1 = τ_base·(1+k·ρ)/(1+ρ)，ρ = |d ln e/dt| / k_ref
  B_bi   双速率并联：x1 = x1a + x1b，两个各自 (r,τ) 的并联，等效非指数形状
  B_bi2  双速率并联（快慢分离版：快路补瞬时、慢路补长尾）
  C_lp   x1 收敛目标改为「误差的慢分量」E_lp（τ_lp 低通）而不是当前 e

架构约束（全部满足）：因果、逐帧 O(1)、无事件检测、无状态清零、无跨事件账本。
"""
import numpy as np

# ---------------------------------------------------------------- 基线参数

P0 = dict(r1=0.12, tc1=8.0, tr1=6.0,
          r2max=0.35, tr2=150.0,
          tau_slope=3.0, slope_gate=0.02, slope_cap=0.01,
          tau_zero=8.0, idle_frac=0.05, y_max_tau=600.0)


def _base(v, zero, x1, x2, v_lp, y_max, dt, p):
    """共用骨架的一帧：返回 (显示, x1, x2, 诊断量)。x1 由调用方给定（已更新）。"""
    y = v - zero
    y_max = np.maximum(y_max * np.exp(-dt / p["y_max_tau"]),
                       np.maximum(y, 0.0))
    idle = y < p["idle_frac"] * np.maximum(y_max, 1.0)
    zero = np.where(idle, zero + (dt / p["tau_zero"]) * (v - zero), zero)
    return y, y_max, zero


def observe(ts, V, variant="base", p=None, x1fn=None):
    """回放一个会话。

    x1fn(ctx) -> (x1_new, dx1_rate, extra) 允许外部替换 x1 更新律；
    ctx 为 dict，含本帧所有中间量。
    返回 dict(D, X1, X2, E, R1EFF, ...)，均为 (n,) 的「通道和」时间序列，
    以及 (n, ch) 的逐通道 x1（供形状分析）。
    """
    pp = dict(P0)
    if p:
        pp.update(p)
    n, ch = V.shape
    x1 = np.zeros(ch)
    x2 = np.zeros(ch)
    zero = V[0].copy()
    y_max = np.maximum(V[0] - zero, 0.0)
    v_lp = V[0].copy()
    st = _init_state(variant, pp, ch)

    D = np.empty(n)
    X1 = np.empty(n)
    X2 = np.empty(n)
    E = np.empty(n)
    R = np.empty(n)
    TAU = np.empty(n)
    Y = np.empty(n)
    X1C = np.empty((n, ch))
    t_prev = ts[0]
    for i in range(n):
        dt = min(max(ts[i] - t_prev, 0.0), 0.1)
        t_prev = ts[i]
        v = V[i]
        if dt > 0.0:
            y, y_max, zero = _base(v, zero, x1, x2, v_lp, y_max, dt, pp)
            # 快态：基线更新律（变体在下面替换）
            e_now = np.maximum(y - x1 - x2, 0.0)
            dx1_rate = np.where(e_now > 0.0,
                                (pp["r1"] * e_now - x1) / pp["tc1"],
                                -x1 / pp["tr1"])
            # 慢态输入
            slope = (v - v_lp) / pp["tau_slope"]
            v_lp = v_lp + (dt / pp["tau_slope"]) * (v - v_lp)
            ctx = dict(y=y, e_now=e_now, x1=x1, x2=x2, v=v, dt=dt,
                       slope=slope, dx1_rate=dx1_rate, p=pp, st=st,
                       zero=zero, v_lp=v_lp)
            r_eff = np.full(ch, pp["r1"])
            tau_eff = np.full(ch, pp["tc1"])
            if variant == "base":
                pass
            else:
                x1, dx1_rate, r_eff, tau_eff = _step_x1(variant, ctx)
            if variant == "base":
                x1 = np.maximum(x1 + dt * dx1_rate, 0.0)
            # 慢态
            e = np.maximum(y - x1 - x2, 0.0)
            rate_cap = pp["slope_cap"] * np.maximum(e, 1.0)
            gate = pp["slope_gate"] * np.maximum(e, 1.0)
            dx2 = np.clip(slope - dx1_rate, -rate_cap, rate_cap)
            dx2 = np.where((e > 0.0) & (np.abs(slope) < gate), dx2, 0.0) * dt
            x2 = x2 + dx2
            x2 = np.where(e > 0.0,
                          np.clip(x2, 0.0, pp["r2max"] * np.maximum(e, 1.0)),
                          np.maximum(x2 - dt * x2 / pp["tr2"], 0.0))
        else:
            e = np.maximum(v - zero - x1 - x2, 0.0)
            r_eff = np.full(ch, pp["r1"])
            tau_eff = np.full(ch, pp["tc1"])
        D[i] = (v - x1 - x2).sum()
        X1[i] = x1.sum()
        X2[i] = x2.sum()
        E[i] = e.sum()
        R[i] = np.mean(r_eff)
        TAU[i] = np.mean(tau_eff)
        Y[i] = (v - zero).sum()
        X1C[i] = x1
    return dict(D=D, X1=X1, X2=X2, E=E, R1EFF=R, TAU=TAU, Y=Y, X1C=X1C)


# ---------------------------------------------------------------- 变体状态

def _init_state(variant, p, ch):
    st = dict(ch=ch)
    if variant in ("A_amp", "A_store"):
        st["alpha"] = np.zeros(ch)        # 蠕变比 α = x1/y 的低通
        st["res"] = np.zeros(ch)          # 残余误差低通（加速用）
        st["ready"] = np.zeros(ch)
    if variant == "C_lp":
        st["E_lp"] = np.zeros(ch)
    if variant in ("B_bi", "B_bi2"):
        st["x1b"] = np.zeros(ch)
    if variant == "B_bi2":
        st["e_lp"] = np.zeros(ch)
    return st


def _step_x1(variant, ctx):
    """返回 (x1_new, dx1_rate, r_eff, tau_eff)，x1_new 已含 dt 积分。"""
    y, x1, x2, dt = ctx["y"], ctx["x1"], ctx["x2"], ctx["dt"]
    e_now, st, p = ctx["e_now"], ctx["st"], ctx["p"]
    ch = st["ch"]
    loaded = e_now > 0.0
    tau0 = np.full(ch, p["tc1"])

    if variant in ("A_amp", "A_store"):
        ysafe = np.maximum(y, 1.0)
        # α 低通取「本帧观测比」；只在有负载、x1 已有可观幅度时有意义
        obs = x1 / ysafe
        obs = np.clip(obs, 0.0, 0.5)
        tau_a = p["tau_r1"]
        st["alpha"] = np.where(loaded,
                               st["alpha"] + (dt / tau_a) * (obs - st["alpha"]),
                               st["alpha"] * np.exp(-dt / tau_a))
        c = p["r1_c"]
        r_eff = np.clip(c * st["alpha"], p["r1_min"], p["r1_max"])
        if variant == "A_amp":
            # 残余误差加速学习：3s 内慢慢抬起 r1_eff（无事件检测，纯泄漏积分器）
            st["res"] = np.where(loaded,
                                 st["res"] + (dt / p["tau_res"]) * (e_now - st["res"]),
                                 st["res"] * np.exp(-dt / p["tau_res"]))
            rel = st["res"] / np.maximum(y - x2, 1.0)
            r_eff = np.clip(r_eff + p["k_acc"] * rel, p["r1_min"], p["r1_max"])
        rate = np.where(loaded,
                        (r_eff * e_now - x1) / p["tc1"],
                        -x1 / p["tr1"])
        return np.maximum(x1 + dt * rate, 0.0), rate, r_eff, tau0

    if variant == "B_tau":
        # ρ = |d ln e/dt| / k_ref ；τc1 = τ_base·(1+k·ρ)/(1+ρ)
        elog = np.log(np.maximum(e_now, 1.0))
        tau_base, k, kref = p["tc1"], p["k_tau"], p["k_ref"]
        de = st.get("_elog_prev")
        rho = np.zeros(ch)
        if de is not None:
            rate_ln = (elog - de) / dt
            rho = np.clip(np.abs(rate_ln) / kref, 0.0, 10.0) * loaded
        st["_elog_prev"] = elog
        tc1_eff = tau_base * (1.0 + k * rho) / (1.0 + rho)
        rate = np.where(loaded,
                        (p["r1"] * e_now - x1) / tc1_eff,
                        -x1 / p["tr1"])
        return (np.maximum(x1 + dt * rate, 0.0), rate, np.full(ch, p["r1"]),
                tc1_eff)

    if variant == "B_bi":
        x1b = st["x1b"]
        ra, rb = p["r1a"], p["r1b"]
        tca, tcb = p["tc1a"], p["tc1b"]
        rate_a = np.where(loaded, (ra * e_now - x1) / tca, -x1 / p["tr1"])
        rate_b = np.where(loaded, (rb * e_now - x1b) / tcb, -x1b / p["tr1"])
        x1n = np.maximum(x1 + dt * rate_a, 0.0)
        x1b = np.maximum(x1b + dt * rate_b, 0.0)
        st["x1b"] = x1b
        return x1n, rate_a + rate_b, np.full(ch, ra + rb), tca

    if variant == "B_bi2":
        # 快路看瞬时 e，慢路看误差的慢分量（补长尾）
        x1b = st["x1b"]
        tau_lp = p["tau_lp_x"]
        st["e_lp"] = st["e_lp"] + (dt / tau_lp) * (e_now - st["e_lp"])
        e_lp = st["e_lp"]
        rate_a = np.where(loaded, (p["r1a"] * e_now - x1) / p["tc1a"], -x1 / p["tr1"])
        rate_b = np.where(loaded, (p["r1b"] * e_lp - x1b) / p["tc1b"], -x1b / p["tr1"])
        x1n = np.maximum(x1 + dt * rate_a, 0.0)
        x1b = np.maximum(x1b + dt * rate_b, 0.0)
        st["x1b"] = x1b
        return x1n, rate_a + rate_b, np.full(ch, p["r1a"] + p["r1b"]), p["tc1a"]

    if variant == "C_lp":
        tau_lp = p["tau_lp"]
        st["E_lp"] = (st["E_lp"] + (dt / tau_lp) * (e_now - st["E_lp"]))
        E_lp = st["E_lp"]
        rate = np.where(loaded,
                        (p["r1"] * E_lp - x1) / p["tc1"],
                        -x1 / p["tr1"])
        return (np.maximum(x1 + dt * rate, 0.0), rate, np.full(ch, p["r1"]),
                np.full(ch, p["tc1"]))

    raise ValueError("unknown variant: " + variant)


# ---------------------------------------------------------------- 变体登记

# 目录名 -> (variant, 参数覆盖, 说明)
VARIANTS = {
    "base": ("base", {},
             "基线 v3.4：r1=0.12, τc1=8"),
    "A_store": ("A_store",
                dict(tau_r1=20.0, r1_c=0.18, r1_min=0.02, r1_max=0.30),
                "幅度自适应（纯低通 α=x1/y，τa=20s，r1=c·α）"),
    "A_amp": ("A_amp",
              dict(tau_r1=20.0, r1_c=0.18, r1_min=0.02, r1_max=0.30,
                   tau_res=3.0, k_acc=0.20),
              "幅度自适应 + 残余误差加速（τa=20s, k_acc=0.20）"),
    "B_tau_k2": ("B_tau", dict(k_tau=2.0, k_ref=0.05),
                 "τ 自适应 k=2, k_ref=0.05/s"),
    "B_tau_k4": ("B_tau", dict(k_tau=4.0, k_ref=0.05),
                 "τ 自适应 k=4, k_ref=0.05/s"),
    "B_tau_k4f": ("B_tau", dict(k_tau=4.0, k_ref=0.02),
                  "τ 自适应 k=4, k_ref=0.02/s（更易触发）"),
    "B_bi": ("B_bi", dict(r1a=0.06, tc1a=2.0, r1b=0.06, tc1b=16.0),
             "双速率并联 r=0.06/0.06, τ=2/16"),
    "B_bi_slow": ("B_bi", dict(r1a=0.04, tc1a=2.0, r1b=0.08, tc1b=20.0),
                  "双速率并联 r=0.04/0.08, τ=2/20（前段轻、后段重）"),
    "B_bi_light": ("B_bi", dict(r1a=0.03, tc1a=3.0, r1b=0.05, tc1b=25.0),
                   "双速率并联 r=0.03/0.05, τ=3/25（整体更轻更慢）"),
    "B_bi2": ("B_bi2", dict(r1a=0.06, tc1a=2.0, r1b=0.08, tc1b=12.0,
                            tau_lp_x=3.0),
              "双速率+慢分量 r=0.06/0.08, τ=2/12, τ_lp=3"),
    "C_lp_t3": ("C_lp", dict(tau_lp=3.0), "x1 目标 = e 的 3s 低通"),
    "C_lp_t6": ("C_lp", dict(tau_lp=6.0), "x1 目标 = e 的 6s 低通"),
    # 参数扫描（不调形状，只调 r1/τc1，用来标定"回落—蠕变跟随"的权衡曲线）
    "S_r1_0.06": ("base", dict(r1=0.06), "扫描：仅 r1=0.06"),
    "S_r1_0.08": ("base", dict(r1=0.08), "扫描：仅 r1=0.08"),
    "S_r1_0.16": ("base", dict(r1=0.16), "扫描：仅 r1=0.16"),
    "S_r1_0.24": ("base", dict(r1=0.24), "扫描：仅 r1=0.24"),
    "S_r1_0.30": ("base", dict(r1=0.30), "扫描：仅 r1=0.30"),
    "S_r1_0.34": ("base", dict(r1=0.34), "扫描：仅 r1=0.34"),
    "S_tc1_16": ("base", dict(tc1=16.0), "扫描：仅 τc1=16"),
    "S_tc1_32": ("base", dict(tc1=32.0), "扫描：仅 τc1=32"),
    "S_tc1_64": ("base", dict(tc1=64.0), "扫描：仅 τc1=64"),
    "S_r1_0.24_tc1_4": ("base", dict(r1=0.24, tc1=4.0),
                        "扫描：r1=0.24 + τc1=4（幅度大而上升快）"),
    "S_r1_0.30_tc1_64": ("base", dict(r1=0.30, tc1=64.0),
                         "扫描：r1=0.30 + τc1=64（幅度大、上升极慢）"),
    "S_r1_0.08_tc1_32": ("base", dict(r1=0.08, tc1=32.0),
                         "扫描：r1=0.08 + τc1=32（幅度小而上升慢）"),
}
