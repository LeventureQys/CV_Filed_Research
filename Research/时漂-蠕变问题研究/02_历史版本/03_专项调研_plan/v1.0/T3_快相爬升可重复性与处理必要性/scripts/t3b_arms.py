# -*- coding: utf-8 -*-
"""t3b_arms.py —— T3-B 的「臂」运行器（在环跑完整算法，输出臂 × 事件完整指标）。

设计要点（逐条对齐任务书硬性约束）：
  1. **在环跑完整算法**：每条臂都在 13 份录制上逐帧 `process`，不是只看 Â 的静态误差
     （范式来自 T4-B 的 `t4b_arm_metrics.csv`：臂 × 事件的完整指标表）；
  2. **每改一处只改一个参数**：`ARMS` 里除 `third` 组的候选路线外，每条 ROI 臂相对
     `BASE`（v6 默认）**只改一个**属性；
  3. **补丁零差证明**：`ArmComp`/`V51Traced` 都是「只记录、不改行为」的子类；
     `t3b_00_recon.py` 用逐帧输出最大绝对差 = 0 证明之（`t3b_patch_ab_zero.csv`）；
  4. **形状库/κ 可插拔接口复用 T4-B**：本任务直接使用 T4-B 的可插拔副本
     `t3b_glm53_v6.py`（= `T4_*/scripts/t4b_glm53_v6.py` 逐字节副本），
     形状库经 `G_TAU/G_G` 类属性注入，κ 经 `KAPPA_ONSET/KAPPA_RESTEP` 注入。
     差异：T4-B 用「事件真值分支」切库；T3-B 用「缩放后的单一形状库 + 滑行器速率 + κ 上限」
     做**连续参数扫描**（ROI 曲线），不引入分支。

指标口径：见 `t3b_settle.py`（= T1-A 冻结口径的代码实现）。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import t3b_ad_lib as L        # noqa: E402
import t3b_common as C        # noqa: E402
import t3b_glm53_v6 as G6     # noqa: E402
import t3b_glm53_v51 as G51   # noqa: E402
import t3b_settle as ST       # noqa: E402

RES = C.RES
CACHE = os.path.join(RES, "cache")
os.makedirs(CACHE, exist_ok=True)

O_TAU = np.asarray(G6.ROM_TAU, float).copy()
O_G = np.asarray(G6.ROM_G, float).copy()

# 主通道（逐字取自 T1-A `t1a_common.RECS`：与第一轮 r4_filter_tradeoff 的 CASES 相同）
MAIN_CH = {"右拇指指尖/数据1": 17, "右拇指指尖/数据2": 17, "右拇指指尖/数据3": 17,
           "左拇指指尖/数据1": 18, "左拇指指尖/数据2": 18, "左拇指指尖/数据3": 18,
           "四指指尖/数据1": 11, "四指指尖/数据2": 11, "四指指尖/数据3": 11,
           "切换负载-快相无责": 4, "再切换负载": 15,
           "中途切换-1d9493": 6, "中途切换-13ffca": 11}

SMOOTH_S = C.SMOOTH_S           # 0.5 s 中值（指标字典 Z̄），只用于指标评估


# ───────────────────────── 形状库工具 ─────────────────────────

def rom_scaled(alpha):
    """时间缩放形状库 g_α(τ) = g_onset(τ/α)（α<1 ⇒ 更慢）。与 T4-B 同名函数逐行相同。"""
    return O_TAU.copy(), np.interp(O_TAU / alpha, O_TAU, O_G, left=0.0, right=1.0)


def rom_scale_shape(scale):
    """v6.1-F1 式「上包络重标」：g = min(1, scale·g_v6)（scale>1 ⇒ 更早到位 ⇒ 更保守）。"""
    return O_TAU.copy(), np.minimum(1.0, scale * O_G)


# ───────────────────────── 仪表化子类（只记录，不改行为） ─────────────────────────

class V51Traced(G51.GLM53v51):
    """v5.1（免责期路线）的仪表化：记录 epoch（`_begin/_restep` 钩子，同 T1-A `make_traced_v5`）。"""

    def __init__(self, n):
        super().__init__(n)
        self.epoch_t = []

    def _begin(self, ts):
        super()._begin(ts)
        self.epoch_t.append(float(ts))

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self.epoch_t.append(float(ts))


class ArmComp(G6.GLM53v6):
    """v6 系臂：新增 ①计数器（epoch 由父类 `epoch_t` 提供、κ 上限触发次数）
    ②`N_GATE`（只对 onset 做形状反演）③输出端因果 EMA。

    `N_GATE=0 / EMA_TAU=0` 时**行为与父类逐帧相同**（见 t3b_00_recon.py 的 A/B 零差证明）。
    """
    N_GATE = 0            # 0 = 全部事件走形状反演；1 = 只有 onset 走，其余用实测增量
    EMA_TAU = 0.0         # >0 ⇒ 输出端因果 EMA（τ，s），模拟"形状反演 + 轻度滤波"

    def __init__(self, n):
        super().__init__(n)
        self.n_inv = 0        # 成功反演次数
        self.n_cap = 0        # 其中被 κ 上限截断的次数（C-4 的"触发率"分子）
        self.n_nogate = 0     # N_GATE=1 时被跳过反演的次数
        self.cap_ratio = []
        self._p_ts = None
        self._ema = None

    def _inv_est(self, hist, tau, kappa):
        """与父类逐行相同的电平域最小二乘，外加 (a) N_GATE (b) 截断计数。"""
        kind = self.ev["kind"] if self.ev is not None else "onset"
        if self.N_GATE == 1 and kind != "onset":
            if len(hist) < 4 or tau < self.TAU_REF:
                return 0.0, False
            inc = float(hist[-1][1])
            if inc <= 0.0:
                return 0.0, False
            self.n_nogate += 1
            return inc, True                    # 不做形状预测：目标 = 当前实测增量
        if len(hist) < 4 or tau < self.TAU_REF:
            return 0.0, False
        tt = np.array([h[0] for h in hist])
        yy = np.array([h[1] for h in hist])
        m = (tt >= self.TAU_REF) & (tt <= min(tau, self.TAU_REF + self.AWIN))
        if m.sum() < 5:
            return 0.0, False
        g = self.g_shape(tt[m])
        den = float((g * g).sum())
        if den < 1e-12:
            return 0.0, False
        inc = float(yy[-1])
        if inc <= 0.0:
            return 0.0, False
        A = float((yy[m] * g).sum()) / den
        A = max(A, inc)
        self.n_inv += 1
        self.cap_ratio.append(A / inc)
        if A > kappa * inc:
            self.n_cap += 1
        return float(min(A, kappa * inc)), True

    def process(self, ts, v):
        out = super().process(ts, v)
        if self.EMA_TAU > 0.0:
            dt = 0.01 if self._p_ts is None else min(max(ts - self._p_ts, 1e-4), 0.1)
            a = dt / self.EMA_TAU
            self._ema = out.copy() if self._ema is None else self._ema + a * (out - self._ema)
            out = self._ema
        self._p_ts = ts
        return out


# ───────────────────────── 臂定义 ─────────────────────────
# 除 group='third' 的候选路线外，每条臂相对 BASE 只改一个参数。
BASE = dict(G_TAU=O_TAU, G_G=O_G, KAPPA_ONSET=1.30, KAPPA_RESTEP=1.12, RATE_MAX=0.8,
            N_GATE=0, EMA_TAU=0.0, ANCHOR_MODE="pin", TRIM_RATE=0.0, GLIDE_MAX=0.80)
V51 = dict(FAST_S=3.0, EXEMPT_AWIN=1.0, LEV_ARM_S=3.0)


def _arm(name, group, kind, dim="", dim_value=np.nan, **kw):
    p = dict(BASE)
    p.update(kw)
    return dict(name=name, group=group, kind=kind, params=p, dim=dim, dim_value=dim_value)


ARMS = [
    # ── Q6/Q7：两条路线的现状臂 ──
    dict(name="raw", group="routes", kind="raw", params={}, dim="route", dim_value=np.nan),
    dict(name="v51_f1", group="routes", kind="v51", dim="exempt_s", dim_value=1.0,
         params=dict(V51, FAST_S=1.0, EXEMPT_AWIN=1.0 / 3.0, LEV_ARM_S=1.0)),
    dict(name="v51_f3", group="routes", kind="v51", dim="exempt_s", dim_value=3.0,
         params=dict(V51)),
    dict(name="v51_f5", group="routes", kind="v51", dim="exempt_s", dim_value=5.0,
         params=dict(V51, FAST_S=5.0, EXEMPT_AWIN=5.0 / 3.0, LEV_ARM_S=5.0)),
    _arm("v6", "routes", "v6", dim="route", dim_value=np.nan),
    _arm("v61_rs106", "routes", "v6", dim="rom_scale", dim_value=1.06,
         G_TAU=rom_scale_shape(1.06)[0], G_G=rom_scale_shape(1.06)[1]),
    # ── Q8 ROI：① 形状库快慢缩放（1.00 的点直接复用 `v6` 臂，参数完全相同）──
    _arm("rs104", "roi", "v6", dim="rom_scale", dim_value=1.04,
         G_TAU=rom_scale_shape(1.04)[0], G_G=rom_scale_shape(1.04)[1]),
    _arm("rs108", "roi", "v6", dim="rom_scale", dim_value=1.08,
         G_TAU=rom_scale_shape(1.08)[0], G_G=rom_scale_shape(1.08)[1]),
    _arm("rs112", "roi", "v6", dim="rom_scale", dim_value=1.12,
         G_TAU=rom_scale_shape(1.12)[0], G_G=rom_scale_shape(1.12)[1]),
    # ── Q8 ROI：② κ 上限（restep 按比例跟随，见报告口径说明） ──
    _arm("kap125", "roi", "v6", dim="kappa_onset", dim_value=1.25,
         KAPPA_ONSET=1.25, KAPPA_RESTEP=round(1.12 * 1.25 / 1.30, 4)),
    _arm("kap115", "roi", "v6", dim="kappa_onset", dim_value=1.15,
         KAPPA_ONSET=1.15, KAPPA_RESTEP=round(1.12 * 1.15 / 1.30, 4)),
    _arm("kap100", "roi", "v6", dim="kappa_onset", dim_value=1.00,
         KAPPA_ONSET=1.00, KAPPA_RESTEP=1.00),
    _arm("kap095", "roi", "v6", dim="kappa_onset", dim_value=0.95,
         KAPPA_ONSET=0.95, KAPPA_RESTEP=0.95),
    # ── Q8 ROI：③ 滑行器速率上限 ──
    _arm("gl030", "roi", "v6", dim="glide_rate", dim_value=0.30, RATE_MAX=0.30),
    _arm("gl050", "roi", "v6", dim="glide_rate", dim_value=0.50, RATE_MAX=0.50),
    _arm("gl160", "roi", "v6", dim="glide_rate", dim_value=1.60, RATE_MAX=1.60),
    _arm("gl300", "roi", "v6", dim="glide_rate", dim_value=3.00, RATE_MAX=3.00),
    # ── Q8 ROI：④ 只对 onset 生效（事件过滤模拟） ──
    _arm("onset_only", "roi", "v6", dim="onset_only", dim_value=1.0, N_GATE=1),
    # ── Q9 第三条路候选 ──
    _arm("rt_filter030", "third", "v6", dim="route3_filter", dim_value=0.30, EMA_TAU=0.30),
    _arm("rt_filter010", "third", "v6", dim="route3_filter", dim_value=0.10, EMA_TAU=0.10),
    _arm("rt_measured", "third", "v6", dim="route3_decouple", dim_value=1.0,
         ANCHOR_MODE="measured"),
    _arm("rt_meas_trim", "third", "v6", dim="route3_decouple", dim_value=2.0,
         ANCHOR_MODE="measured", TRIM_RATE=0.002),
    _arm("rt_onesided", "third", "v6", dim="route3_onesided", dim_value=1.0,
         G_TAU=rom_scale_shape(1.08)[0], G_G=rom_scale_shape(1.08)[1],
         KAPPA_ONSET=1.00, KAPPA_RESTEP=1.00),
]
ARM_BY_NAME = {a["name"]: a for a in ARMS}


# ───────────────────────── 数据与事件缓存 ─────────────────────────

def _npz(name):
    return os.path.join(CACHE, "t3b_%s.npz" % name.replace("/", "_"))


def load_all(verbose=True):
    """返回 (events DataFrame, cache dict)。cache[rec] = dict(tu, Xu, dt, tot, peak)。

    npz 缓存不存在时用 `t3b_common.build_events()`（读 13 份 CSV，约 1~2 min），
    并把网格与事件表落到 `results/cache/`，供后续脚本秒级加载。
    """
    ev_path = os.path.join(RES, "t3b_events.csv")
    have_cache = all(os.path.isfile(_npz(n)) for n, _, _ in C.RECS)
    if have_cache and os.path.isfile(ev_path):
        cache = {}
        for name, _p, dom in C.RECS:
            z = np.load(_npz(name))
            tu = z["tu"]
            Xu = z["Xu"]
            cache[name] = dict(tu=tu, Xu=Xu, dt=float(z["dt"]), tot=z["tot"],
                               peak=float(z["peak"]), dom=dom)
        ev = pd.read_csv(ev_path, encoding="utf-8-sig")
        if verbose:
            print("[cache] 命中 npz 网格 + t3b_events.csv（%d 事件）" % len(ev))
        if "clean_t4a" not in ev.columns:
            ev["clean_t4a"] = t4a_clean_flags(ev)
            ev.to_csv(ev_path, index=False, encoding="utf-8-sig")
        return ev, cache
    if verbose:
        print("[cache] 未命中：读 13 份 CSV 重建（build_events）…", flush=True)
    ev, cache = C.build_events(verbose=verbose)
    for name, cc in cache.items():
        np.savez_compressed(_npz(name), tu=cc["tu"], Xu=cc["Xu"], dt=cc["dt"],
                            tot=cc["tot"], peak=cc["peak"])
    ev["clean_t4a"] = t4a_clean_flags(ev)
    ev.to_csv(ev_path, index=False, encoding="utf-8-sig")
    if verbose:
        print("[cache] 已写 results/cache/t3b_*.npz 与 results/t3b_events.csv")
    return ev, cache


def t4a_clean_flags(ev):
    """读 T4-A 的 60 事件冻结表，给本任务事件打 `clean_t4a` 标记（±0.6 s 配对）。"""
    p = os.path.join(C.PLAN, "T4_两种快相形态与分支判据", "results", "t4a_morphology.csv")
    out = np.zeros(len(ev), bool)
    if not os.path.isfile(p):
        print("!! 缺 t4a_morphology.csv：clean_t4a 全部置 False")
        return out
    a = pd.read_csv(p, encoding="utf-8-sig")
    a = a[a.kind.isin(["onset", "restep"])]
    vals = []
    for _, r in ev.iterrows():
        c = a[(a.rec == r.rec) & (np.abs(a.t_on - r.t_on) <= 0.6)]
        vals.append(bool(c.iloc[(c.t_on - r.t_on).abs().argmin()].clean) if len(c) else False)
    return np.asarray(vals, bool)


# ───────────────────────── 逐臂运行 ─────────────────────────

def _make(arm, n):
    k = arm["kind"]
    if k == "v51":
        c = V51Traced(n)
    else:
        c = ArmComp(n)
    for kk, vv in arm["params"].items():
        setattr(c, kk, vv)
    return c


def run_arm(arm, ev, cache, progress=True):
    """跑一条臂，返回 (逐事件指标 DataFrame, 逐录制汇总 DataFrame)。"""
    rows, recs = [], []
    for name, path, dom in C.RECS:
        if name not in cache:
            continue
        cc = cache[name]
        tu, Xu, dt, tot, peak = cc["tu"], cc["Xu"], cc["dt"], cc["tot"], cc["peak"]
        n = len(tu)
        ch = MAIN_CH[name]
        comp = None
        if arm["kind"] == "raw":
            Y = Xu
        else:
            comp = _make(arm, Xu.shape[1])
            Y = np.empty_like(Xu)
            for i in range(n):
                Y[i] = comp.process(tu[i], Xu[i])
        k5 = max(1, int(round(SMOOTH_S / dt)))
        ytot = L.med_smooth(Y.sum(axis=1), k5)
        ych = L.med_smooth(Y[:, ch], k5)
        rtot = L.med_smooth(tot, k5)
        rch = L.med_smooth(Xu[:, ch], k5)
        loc = ev[ev.rec == name].sort_values("t_on")
        tlist = [(float(r.t_on), float(r.J_nc)) for _, r in loc.iterrows()
                 if np.isfinite(r.J_nc) and abs(r.J_nc) >= 0.02 * peak and r.kind in
                 ("onset", "restep")]
        n_inv = getattr(comp, "n_inv", 0) if comp is not None else 0
        n_cap = getattr(comp, "n_cap", 0) if comp is not None else 0
        n_nogate = getattr(comp, "n_nogate", 0) if comp is not None else 0
        n_epoch = len(getattr(comp, "epoch_t", [])) if comp is not None else 0
        span = float(tu[-1] - tu[0])
        cpr = (np.asarray(comp.cap_ratio, float) if comp is not None and n_inv else np.asarray([]))
        recs.append(dict(arm=arm["name"], group=arm["group"], rec=name, dom=dom, span_s=round(span, 2),
                         n_events=int(len(loc)), n_epoch=n_epoch,
                         epoch_per100s=(100.0 * n_epoch / span if span > 1 else np.nan),
                         n_inv=n_inv, n_cap=n_cap, n_nogate=n_nogate,
                         trigger_rate=(n_cap / n_inv if n_inv else np.nan),
                         cap_ratio_med=(float(np.median(cpr)) if cpr.size else np.nan),
                         cap_ratio_p90=(float(np.percentile(cpr, 90)) if cpr.size else np.nan),
                         dim=arm["dim"], dim_value=arm["dim_value"]))
        for j, (_, r) in enumerate(loc.iterrows()):
            k0 = int(np.searchsorted(tu, float(r.t_on)))
            if k0 >= n - 5:
                continue
            times = [(t, jj) for t, jj in tlist if not (abs(t - float(r.t_on)) < 1e-9)]
            cut_s = ST.next_event_cut(times, float(r.t_on), peak)
            cut_k = None if cut_s is None else int(round(cut_s / dt))
            win_avail = ((cut_k if cut_k is not None else n) * dt - float(r.t_on))
            J_ch, pre_ch, _ = ST.amp_5s(rch, k0)
            J_tot, pre_tot, _ = ST.amp_5s(ytot, k0)
            ju = ST.unload_index(rch, k0)
            J_creep = np.nan
            if ju is not None:
                a5 = max(k0, ju - int(5.0 / dt))
                if ju > a5:
                    J_creep = float(np.median(rch[a5:ju]) - pre_ch)
            Tch, cen_ch = ST.t_stable(tu, ych, k0, J_ch)
            Tto, cen_to = ST.t_stable(tu, ytot, k0, J_tot)
            Tev_ch, cev_ch, wev = ST.t_stable_ev(tu, ych, k0, J_ch, cut_k)
            Tev_to, cev_to, _ = ST.t_stable_ev(tu, ytot, k0, J_tot, cut_k)
            zc, zfrom = ST.z_final_of(ych, k0, cut_k)
            zt, _ = ST.z_final_of(ytot, k0, cut_k)
            ts5_ch, _ = ST.t_settle(tu, ych, k0, J_ch, 0.05, zc, cut_k)
            ts2_ch, _ = ST.t_settle(tu, ych, k0, J_ch, 0.02, zc, cut_k)
            ts5_to, _ = ST.t_settle(tu, ytot, k0, J_tot, 0.05, zt, cut_k)
            e1c = _err(ych, k0, pre_ch + J_ch, J_ch, 1.0, dt)
            e2c = _err(ych, k0, pre_ch + J_ch, J_ch, 2.0, dt)
            e1t = _err(ytot, k0, pre_tot + J_tot, J_tot, 1.0, dt)
            cap_raw = _cap_den(rch, k0, peak, 20.0, dt)
            rows.append(dict(
                arm=arm["name"], group=arm["group"], dim=arm["dim"], dim_value=arm["dim_value"],
                rec=name, dom=dom, kind=r.kind, t_on=round(float(r.t_on), 3),
                clean_t4a=bool(r.clean_t4a) if "clean_t4a" in r.index else False,
                J_nc=float(r.J_nc), pre_frac=float(r.pre_frac),
                J_ch=J_ch, J_tot=J_tot, J_creep=J_creep,
                T_stable_ch5=Tch, cens_ch5=cen_ch, T_stable_tot5=Tto, cens_tot5=cen_to,
                T_stable_ev_ch5=Tev_ch, cens_ev_ch5=cev_ch, win_ev_ch5=wev,
                T_stable_ev_tot5=Tev_to, cens_ev_tot5=cev_to,
                T_settle_ch5=ts5_ch, T_settle_ch2=ts2_ch, T_settle_tot5=ts5_to,
                zfinal_from_s=zfrom,
                OS_pct=ST.os_dir(ych, k0, pre_ch + J_ch, J_ch),
                US_pct=ST.us_dir(ych, k0, pre_ch + J_ch, J_ch),
                OS_tot_pct=ST.os_dir(ytot, k0, pre_tot + J_tot, J_tot),
                err_1s_pct=e1c, err_2s_pct=e2c, err_1s_tot_pct=e1t,
                MD_ch_adc=ST.md_abs(ych, rch, k0, cut_k),
                MD_tot_adc=ST.md_abs(ytot, rtot, k0, cut_k),
                G_20s=ST.capture_ratio(ytot, rtot, k0, 20.0, dt),
                G_den_adc=cap_raw,
                win_avail_s=win_avail, cut_s=(np.nan if cut_s is None else cut_s),
                rom_scale=_rom_dim(arm), kappa_onset=arm["params"].get("KAPPA_ONSET", np.nan),
                glide_rate=arm["params"].get("RATE_MAX", np.nan),
                onset_only=arm["params"].get("N_GATE", 0),
                ema_tau=arm["params"].get("EMA_TAU", 0.0),
                anchor_mode=arm["params"].get("ANCHOR_MODE", ""),
                trigger_rate_rec=(n_cap / n_inv if n_inv else np.nan),
            ))
        if progress:
            print("  %-14s %-20s n_ev=%2d epoch=%3d inv=%4d cap=%4d (%.2f)  %.0fs" % (
                arm["name"], name, len(loc), n_epoch, n_inv, n_cap,
                (n_cap / n_inv if n_inv else float("nan")), span), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(recs)


def _rom_dim(arm):
    """把形状库折算成一个可读的"快慢"标量：g(0.2 s) 相对 v6 ROM 的 0.790 的比值（≈scale）。

    非 v6 系臂（raw / v5.1）没有形状库 ⇒ 报 NaN，免得在 ROI 图里被当成 scale=1。
    """
    if arm["kind"] != "v6":
        return np.nan
    g = np.asarray(arm["params"].get("G_G", O_G), float)
    t = np.asarray(arm["params"].get("G_TAU", O_TAU), float)
    return float(np.interp(0.2, t, g)) / 0.790


def _err(Y, k0, target, ref, off, dt):
    i0 = k0 + int(round((off - 0.1) / dt))
    i1 = k0 + int(round((off + 0.1) / dt))
    if not np.isfinite(ref) or abs(ref) < 1e-12 or i1 >= len(Y):
        return np.nan
    return (float(np.median(Y[max(0, i0):i1])) - target) / ref * 100.0


def _cap_den(Yraw, k0, peak, lag, dt):
    i = k0 + int(round(lag / dt))
    if i >= len(Yraw):
        return np.nan
    return abs(float(Yraw[i]) - float(Yraw[k0]))


def append_metrics(df, path=None):
    """把逐事件指标追加到 results/t3b_arm_metrics.csv（幂等：先删同臂旧行）。"""
    path = path or os.path.join(RES, "t3b_arm_metrics.csv")
    if os.path.isfile(path) and len(df):
        old = pd.read_csv(path, encoding="utf-8-sig")
        old = old[~old.arm.isin(df.arm.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def append_recs(df, path=None):
    path = path or os.path.join(RES, "t3b_route_recording.csv")
    if os.path.isfile(path) and len(df):
        old = pd.read_csv(path, encoding="utf-8-sig")
        old = old[~old.arm.isin(df.arm.unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_group(group, ev, cache, skip_done=True, force=False):
    """跑一组臂；已存在于 t3b_arm_metrics.csv 的臂默认跳过（可中断续跑）。

    `force=True`（命令行 `--force`）⇒ 重跑并覆盖该组的旧行（换了口径后必须这样做）。
    """
    p = os.path.join(RES, "t3b_arm_metrics.csv")
    done = set()
    if skip_done and not force and os.path.isfile(p):
        done = set(pd.read_csv(p, encoding="utf-8-sig").arm.unique())
    for arm in ARMS:
        if arm["group"] != group:
            continue
        if arm["name"] in done:
            print("[skip] %s 已在 t3b_arm_metrics.csv" % arm["name"], flush=True)
            continue
        print("== arm %s (group=%s) ==" % (arm["name"], group), flush=True)
        df, recs = run_arm(arm, ev, cache)
        append_metrics(df)
        append_recs(recs)
        print("   -> %d 行事件指标（累计表 %s）" % (len(df), p), flush=True)
    return 0
