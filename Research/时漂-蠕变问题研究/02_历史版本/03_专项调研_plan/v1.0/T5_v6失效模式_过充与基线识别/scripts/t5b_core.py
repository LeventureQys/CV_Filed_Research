# -*- coding: utf-8 -*-
"""T5-B 核心：仪表化 v6（逐帧内部量）/ 评价窗 / 真实事件真值 / 失效判定。

设计纪律（对报告 §2 负责）
--------------------------
1. **不改变原型默认行为**：`TraceV6` 只做两件事——(a) 覆盖 `process` 在调用 `super()` 前清空
   本帧记录槽；(b) 覆盖 `_win_mean` 记录返回值。两者都不影响任何内部状态。
   `sig_d` 被改成"属性 + setter"，但默认 `CAPF=None` 时 setter 只做 `self._sig_d = v`，
   与原型 `self.sig_d = v` 等价 —— 零差证明见 `results/t5b_patch_ab_zero.csv`（max|ΔY| = 0）。
2. **逐帧落盘的量都是原型真实存在的内部量**；`thr_d / tail_gate / raw_hit` 由落盘量按原型公式
   **重算**（公式与 `t5b_glm53_v6.py::process` 逐行对应）。`armed_pre / hit_run_pre` 取
   "检测块进入前"的快照。
3. 逐帧记录存**列式数值数组**（不是 dict 列表），否则 ~1000 次 trial 的 DataFrame 构造会主导耗时。
4. 时间轴只用 `timestamp`（经 `t5b_ad_lib.load_rec` 自动定位 `##Data`），100 Hz 网格。
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
PLAN = os.path.dirname(TASK)
ROOT = os.path.abspath(os.path.join(PLAN, "..", "..", "..", ".."))
TEMP = os.path.join(ROOT, "temp")
RES = os.path.join(TASK, "results")
os.makedirs(RES, exist_ok=True)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from t5b_glm53_v6 import GLM53v6            # noqa: E402
from t5b_glm53_v51 import GLM53v51          # noqa: E402
from t5b_ad_lib import med_smooth           # noqa: E402
from t5b_a_common import (REC, load_uniform, make_perturb, add_tap, add_step)  # noqa: E402

DT = 0.01
STATE_CODE = {"idle": 0, "event": 1, "slow": 2}
KIND_CODE = {None: 0, "onset": 1, "restep": 2, "restep_reload": 3, "decrease": 4}
CODE_STATE = {v: k for k, v in STATE_CODE.items()}
CODE_KIND = {v: k for k, v in KIND_CODE.items()}

TR_COLS = ["ts", "total", "out", "lv_now", "lv_ref", "d", "sig_d", "thr_d", "tail_gate", "gate",
           "max_tot", "hit_run", "quiet_run", "armed_pre", "raw_hit", "ratio",
           "state_pre", "state_post", "ev_pre", "ev_now", "tau_ep", "A_hat", "c_applied",
           "inc_s", "sumA", "sumA_pre", "g", "g_pre", "n_loaded", "n_epoch",
           "last_ev_ts", "idle_since", "ev_block_until"]


def tr_frame(tr):
    """把列式 trace 转成可读 DataFrame（Q2 落盘用；kind/state 解码回字符串）。"""
    df = pd.DataFrame({k: np.asarray(tr[k]) for k in TR_COLS})
    df["state_pre"] = df["state_pre"].map(CODE_STATE)
    df["state_post"] = df["state_post"].map(CODE_STATE)
    df["ev_pre"] = df["ev_pre"].map(CODE_KIND)
    df["ev_now"] = df["ev_now"].map(CODE_KIND)
    return df


# ───────────────────────── 仪表化 v6 ─────────────────────────

class TraceV6(GLM53v6):
    """只记录、不改行为。`CAPF=None` 时 `sig_d` 属性与原型赋值完全等价。"""
    CAPF = None                     # 5σ_d 封顶系数（× 电平项）；None = 不封顶（默认）
    _last_wm = None
    _cur_lv_ref = None

    def __init__(self, n):
        self._sig_d = 0.0
        self._cur_lv_ref = None
        self._wm = 0
        self._mv = []
        self._snap = {}
        super().__init__(n)
        self.col = {k: [] for k in TR_COLS}
        self.tr_epoch, self.tr_revoke, self.tr_handoff = [], [], []
        self.tr_unload, self.tr_gevent, self.tr_deny = [], [], []

    # —— 记录钩子 ——
    def _win_mean(self, t0, t1):
        r = super()._win_mean(t0, t1)
        self._wm += 1
        self._mv.append(r)
        if self._wm == 2:                          # 本帧的 lv_ref 已算出，检测块尚未进入
            self._cur_lv_ref = r
            ev = self.ev
            self._snap = dict(state_pre=STATE_CODE[self.state],
                              ev_pre=KIND_CODE[None if ev is None else str(ev["kind"])],
                              ev_t0_pre=(np.nan if ev is None else float(ev["t0"])),
                              sumA_pre=float(np.sum(self.A)),
                              g_pre=float(self.g),
                              armed_pre=bool(self._armed),
                              hit_run_pre=int(self._hit_run),
                              quiet_run_pre=int(self._quiet_run),
                              last_ev_ts=float(self.last_ev_ts),
                              idle_since=float(self.idle_since),
                              ev_block_until=float(self.ev_block_until))
        return r

    # —— 默认等价的属性（CAPF=None 时与原型的普通赋值等价） ——
    @property
    def sig_d(self):
        return self._sig_d

    @sig_d.setter
    def sig_d(self, v):
        if self.CAPF is not None and self._cur_lv_ref is not None:
            lvl = max(self.DET_REL * abs(self._cur_lv_ref), self.DET_ABS_FRAC * self.max_tot)
            v = min(v, self.CAPF * lvl / self.DET_K)
        self._sig_d = float(v)

    # —— 事件记录钩子（与 a_common.make_traced 同构，只记录） ——
    def _new_event(self, t0, base, v0, y0, kind, hist, prev_state):
        super()._new_event(t0, base, v0, y0, kind, hist, prev_state)
        self.tr_epoch.append((float(t0), str(kind), float(self.last_ts),
                              float(base), float(np.sum(v0)), float(np.sum(y0))))

    def _handoff(self, ts, v, ev=None):
        ev = ev if ev is not None else self.ev
        rec = (float(ts), str(ev["kind"]), float(ev["A_hat"]), float(ev["c_applied"]))
        super()._handoff(ts, v, ev)
        self.tr_handoff.append(rec + (float(np.sum(self.A)), float(self.g)))

    def _to_idle(self):
        super()._to_idle()
        self.tr_unload.append(float(self.last_ts))

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        ev = self.ev
        kind = str(ev["kind"]) if ev is not None else None
        n_rev = self.n_revoke
        e0 = float(ev["t0"]) if (ev is not None and kind == "decrease") else None
        out = super()._run_event(ts, v, total, dt, eps, idle_now)
        if kind == "decrease" and self.n_revoke > n_rev:
            self.tr_gevent.append((float(ts), kind, e0))
        return out

    def process(self, ts, v):
        self._wm = 0
        self._mv = []
        self._snap = {}
        n_rev0 = self.n_revoke
        out = super().process(ts, v)
        if self.n_revoke > n_rev0:                 # 撤销（正/负事件都在 _run_event 里撤销）
            self.tr_revoke.append((float(ts), "revoke"))
        lv_now = self._mv[0] if len(self._mv) > 0 else None
        lv_ref = self._mv[1] if len(self._mv) > 1 else None
        d = (lv_now - lv_ref) if (lv_now is not None and lv_ref is not None) else 0.0
        thr_d = max(self.DET_K * self.sig_d,
                    self.DET_REL * abs(lv_ref if lv_ref is not None else 0.0),
                    self.DET_ABS_FRAC * self.max_tot)
        s = self._snap
        st_pre = s.get("state_pre", STATE_CODE[self.state])
        ev_t0 = s.get("ev_t0_pre", np.nan)
        if ev_t0 == ev_t0:
            tau_ep = ts - ev_t0
        elif st_pre == STATE_CODE["slow"]:
            tau_ep = ts - self.ev_end
        else:
            tau_ep = 1e9
        tail_gate = (self.TAIL_GATE_FRAC * abs(lv_ref if lv_ref is not None else 0.0)
                     if (st_pre != STATE_CODE["idle"] and tau_ep < self.TAIL_GATE_S) else 0.0)
        gate = max(thr_d, tail_gate)
        ev = self.ev
        c = self.col
        c["ts"].append(float(ts))
        c["total"].append(float(np.sum(v)))
        c["out"].append(float(np.sum(out)))
        c["lv_now"].append(np.nan if lv_now is None else float(lv_now))
        c["lv_ref"].append(np.nan if lv_ref is None else float(lv_ref))
        c["d"].append(float(d))
        c["sig_d"].append(float(self.sig_d))
        c["thr_d"].append(float(thr_d))
        c["tail_gate"].append(float(tail_gate))
        c["gate"].append(float(gate))
        c["max_tot"].append(float(self.max_tot))
        c["hit_run"].append(int(self._hit_run))
        c["quiet_run"].append(int(self._quiet_run))
        c["armed_pre"].append(bool(s.get("armed_pre", True)))
        c["raw_hit"].append(bool(abs(d) > gate))
        c["ratio"].append(float(abs(d) / gate if gate > 1e-12 else 0.0))
        c["state_pre"].append(int(st_pre))
        c["state_post"].append(STATE_CODE[self.state])
        c["ev_pre"].append(int(s.get("ev_pre", 0)))
        c["ev_now"].append(KIND_CODE[None if ev is None else str(ev["kind"])])
        c["tau_ep"].append(float(tau_ep))
        c["A_hat"].append(np.nan if ev is None else float(ev["A_hat"]))
        c["c_applied"].append(np.nan if ev is None else float(ev["c_applied"]))
        c["inc_s"].append(np.nan if ev is None else float(np.sum(v) - ev["base"]))
        c["sumA"].append(float(np.sum(self.A)))
        c["sumA_pre"].append(float(s.get("sumA_pre", np.nan)))
        c["g"].append(float(self.g))
        c["g_pre"].append(float(s.get("g_pre", np.nan)))
        c["n_loaded"].append(int(np.sum(self.loaded)))
        c["n_epoch"].append(int(len(self.tr_epoch)))
        c["last_ev_ts"].append(float(s.get("last_ev_ts", np.nan)))
        c["idle_since"].append(float(s.get("idle_since", np.nan)))
        c["ev_block_until"].append(float(s.get("ev_block_until", np.nan)))
        return out


def make_traced_v6(capf=None):
    """构造带 CAPF 的 traced 变体（capf=None = 与原型零差）。"""
    return type("T5BT", (TraceV6,), dict(CAPF=capf))


TRACED_V6 = TraceV6


# ───────────────────────── 运行 ─────────────────────────

def _setup(Cls, X, kw):
    c = Cls(X.shape[1])
    for k, val in kw.items():
        setattr(c, k, val)
    return c


def run_full(tu, X, cls=TraceV6, **kw):
    c = _setup(cls, X, kw)
    n = len(tu)
    Y = np.empty_like(X)
    for i in range(n):
        Y[i] = c.process(float(tu[i]), X[i])
    tr = {k: np.asarray(v) for k, v in c.col.items()}
    return dict(Y=Y, Z=X.sum(axis=1), Ysum=Y.sum(axis=1), c=c, tr=tr,
                epoch=c.tr_epoch, revoke=c.tr_revoke, handoff=c.tr_handoff,
                unload=c.tr_unload, gevent=c.tr_gevent)


def run_plain(tu, X, cls=GLM53v6, **kw):
    c = _setup(cls, X, kw)
    Y = np.empty_like(X)
    for i in range(len(tu)):
        Y[i] = c.process(float(tu[i]), X[i])
    return Y, c


# ───────────────────────── 评价窗与真值 ─────────────────────────

WIN = {
    # 选窗依据：`results/t5b_plateaus.csv` 的平台结构 + 每个真值事件后留 ≥8 s（M2 检查点可用）
    "W1_1w": dict(rec="切换负载-快相无责", t0=26.0, t1=92.0,
                  note="含平台≈12048 ADC（最接近用户所说 1w ADC）；真值沿 3 个"),
    "W2_3w": dict(rec="中途切换-最终测试目标", t0=80.0, t1=115.0,
                  note="平台≈29400~32100 ADC（高电平对照）；真值沿 4 个"),
    "W3_1p5w": dict(rec="零负载-切换负载-零负载-再切换负载", t0=5.0, t1=45.0,
                    note="平台 14491/15643/21772 ADC；真值沿 6 个（含 2 个卸载）"),
    "W4_1p9w": dict(rec="零负载-中途切换负载-零负载-切换负载", t0=5.0, t1=45.0,
                    note="平台 16469/18183/25619 ADC；真值沿 5 个（含卸载）"),
}

GT_REC_MAP = {"切换负载-快相无责": "SW1", "零负载-切换负载-零负载-再切换负载": "SW2",
              "零负载-中途切换负载-零负载-切换负载": "SW3",
              "中途切换-最终测试目标": "SW4"}
T4A_MORPH = os.path.join(PLAN, "T4_两种快相形态与分支判据", "results", "t4a_morphology.csv")


def load_gt():
    """真实事件真值 = T4-A 冻结的 60 事件表里 4 份变载实录的部分（只读引用）。"""
    df = pd.read_csv(T4A_MORPH, encoding="utf-8-sig")
    df = df[df["key"].isin(["SW1", "SW2", "SW3", "SW4"])].copy()
    df = df[np.isfinite(df["jump"]) & (df["jump"].abs() > 1e-9)]
    return df


def gt_in_window(gt, rec, t0, t1):
    g = gt[(gt["key"] == GT_REC_MAP[rec]) & (gt["t_on"] >= t0) & (gt["t_on"] <= t1)].copy()
    return g.sort_values("t_on")


def load_window(w):
    """返回 (tu, Xu, w)：`tu` 平移到窗起点为 0；`w["t0"]` = 窗在录制坐标下的起点。"""
    d = load_uniform(REC[w["rec"]], tmax=w["t1"])
    i0 = int(round(w["t0"] / DT))
    sl = slice(i0, len(d["tu"]))
    return d["tu"][sl] - w["t0"], d["Xu"][sl], w


# ───────────────────────── 扰动构造 ─────────────────────────

def perturb_window(tu, Xu, kind, amp, seed, **kw):
    """在评价窗上注入扰动。kind ∈ {white, band, common, tap, step}。

    `amp` 口径（`00_共享/指标字典与口径.md` §5）：white/band/common = 总量扰动 **RMS**(ADC)；
    tap/step = 总量扰动 **峰值/台阶**(ADC)。返回 (Xp, meta)。
    """
    rng = np.random.default_rng(seed)
    meta = dict(kind=kind, amp=float(amp), seed=int(seed))
    if kind in ("white", "band", "common"):
        E = make_perturb(Xu, amp, kind, rng,
                         f_lo=kw.get("f_lo", 0.3), f_hi=kw.get("f_hi", 40.0))
        return Xu + E, dict(meta, f_lo=kw.get("f_lo", 0.3), f_hi=kw.get("f_hi", 40.0))
    if kind == "tap":
        i0 = int(round(kw.get("t_inj", 20.0) / DT))
        rise = kw.get("rise_ms", 50.0)
        hold = kw.get("hold_ms", 100.0)
        Xp, _ = add_tap(Xu, i0, amp, fs=100.0, rise_ms=rise, hold_ms=hold,
                        fall_ms=kw.get("fall_ms", rise))
        return Xp, dict(meta, t_inj=kw.get("t_inj", 20.0), rise_ms=rise, hold_ms=hold,
                        dur_ms=2 * rise + hold)
    if kind == "step":
        i0 = int(round(kw.get("t_inj", 20.0) / DT))
        Xp = add_step(Xu, i0, amp, fs=100.0, rise_ms=kw.get("rise_ms", 50.0))
        return Xp, dict(meta, t_inj=kw.get("t_inj", 20.0), rise_ms=kw.get("rise_ms", 50.0))
    raise ValueError(kind)


# ───────────────────────── 失效判定 ─────────────────────────

FAIL_DEF = dict(
    thr_anchor=0.20, thr_disp=0.20, rel_gate=0.05, abs_gate=300.0,
    abs_anchor=300.0, match_lo=-1.20, match_hi=1.50, survive_s=0.50, ck_dt=20.0,
)


def eval_trial(tu, Xp, ref, gt_win, w_t0, fail_def=None, cls=TraceV6, **kw):
    """返回 dict(flags, ...)。`ref` = 干净运行的 run_full 结果；`gt_win["t_on"]` 用录制坐标。"""
    fd = dict(FAIL_DEF, **(fail_def or {}))
    r = run_full(tu, Xp, cls, **kw)
    tr, trr = r["tr"], ref["tr"]
    dev = r["Ysum"] - ref["Ysum"]
    Z = r["Z"]
    Zs = med_smooth(Z, 30)
    rows, flags = [], dict(M1_miss=0, M2_wrong_anchor=0, M3_false_capture=0, M4_disp=0)
    n_should = 0
    gt_w = np.array([float(x) - w_t0 for x in gt_win["t_on"]], float) if len(gt_win) else np.array([])
    for _, g in gt_win.iterrows():
        t_on_w = float(g["t_on"]) - w_t0
        i_on = int(np.argmin(np.abs(tu - t_on_w)))
        lvl = float(np.median(Zs[max(0, i_on - 50):max(1, i_on)]))
        should = (abs(float(g["jump"])) >= max(fd["rel_gate"] * abs(lvl), fd["abs_gate"]))
        n_should += int(should)
        m = [e for e in r["epoch"] if fd["match_lo"] <= (e[0] - t_on_w) <= fd["match_hi"]]
        miss = int(should and len(m) == 0)
        flags["M1_miss"] += miss
        i_ck = int(round((t_on_w + fd["ck_dt"]) / DT))
        if 0 <= i_ck < len(tr["sumA"]):
            dA = float(tr["sumA"][i_ck] - trr["sumA"][i_ck])
            lv_ck = float(np.median(Zs[max(0, i_ck - 100):i_ck + 1]))
            rel = abs(dA) / max(abs(lv_ck), 1.0)
            bad = int(rel > fd["thr_anchor"])
        else:
            dA, rel, bad = np.nan, np.nan, 0
        flags["M2_wrong_anchor"] += bad
        rows.append(dict(kind_gt=str(g["kind"]), t_on=round(float(g["t_on"]), 2),
                         t_on_w=round(t_on_w, 2), jump=round(float(g["jump"]), 1),
                         lvl=round(lvl, 1), should_detect=int(should), n_match=len(m),
                         match_kinds="|".join(sorted({e[1] for e in m})),
                         M1_miss=miss, dA_ck=(round(dA, 1) if dA == dA else np.nan),
                         dA_rel=(round(rel, 4) if rel == rel else np.nan), M2=bad))
    n_false = 0
    for e in r["epoch"]:
        t0e = e[0]
        if len(gt_w) and np.min(np.abs(gt_w - t0e)) <= 1.5:
            continue
        rev_later = [v[0] for v in r["revoke"] if v[0] > t0e]
        if (not rev_later) or (rev_later[0] - t0e) > fd["survive_s"]:
            n_false += 1
    flags["M3_false_capture"] = n_false
    lv_max = float(np.percentile(np.abs(Zs), 95))
    mdev = float(np.abs(dev).max())
    flags["M4_disp"] = int(mdev > fd["thr_disp"] * max(lv_max, 1.0))
    lc = (tr["sig_d"] * 5.0 > 0.05 * np.abs(tr["lv_ref"]))
    res = dict(flags=flags, n_should=n_should, n_epoch=len(r["epoch"]),
               n_revoke=len(r["revoke"]), n_handoff=len(r["handoff"]),
               maxdev=mdev, maxdev_rel=mdev / max(lv_max, 1.0), lv_max=lv_max,
               n_lowconf=int(lc.sum()), n_lowconf_frac=float(lc.mean()),
               dA_end=float(tr["sumA"][-1] - trr["sumA"][-1]),
               det_rows=rows, r=r, ref=ref)
    res["fail"] = int(any(flags[k] > 0 for k in
                          ("M1_miss", "M2_wrong_anchor", "M3_false_capture", "M4_disp")))
    return res
