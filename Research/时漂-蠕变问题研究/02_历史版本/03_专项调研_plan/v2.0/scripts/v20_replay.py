# -*- coding: utf-8 -*-
"""v2.0 A1 · v6 原型离线回放骨架 + 全内因追踪（可 import）。

目的
----
把归档原型 `progress/archived/07-v6/scripts/glm53_v6.py` 的 `GLM53v6` 包起来，
逐帧喂入录制数据集里 **算法输入流**（`device_001_pre_seg0.csv`）并记录
**每帧内部状态快照**与**全部状态迁移事件**，用于回答：

  Q1 原型是否复现现场输出（`device_001_seg000.csv`）？
  Q2 「44 s 附近基线错误估计」在算法内部究竟发生了什么？
  Q3 显示偏移 `main − pre` 的两条通路（事件期滑行器 / 慢相扣除）各占多少？
  Q4 `min_ts` / `level_ref` / `g` / `A.sum()` 的全程轨迹与漂移主导项。
  Q5 本录制是否仍落在 T9 的 never-idle 失效域内。

关键实现约定（与 C++ 实现 `src/domain/drift_v6/drift_v6_compensator.{h,cpp}` 的差异）
--------------------------------------------------------------------------------
原型 `GLM53v6` 的类常量 **不是** 当前落地的参数集：
  KAPPA_ONSET = 1.30 （C++ `kKappaOnset = 1.05`）
  HO_MIN      = 5.00 （C++ `kHoMinS = 3.5`）
  无 revoke 迟滞      （C++ `kRevokeHoldN = 3`）
本模块的 `V6Replay` 默认按当前参数集 `plan-v1.0 A1+A4+A5a` 覆盖这三项，
并**在子类中补上原型缺失的 revoke 迟滞**（`_run_event` 的撤销段，见 `_run_event`）。
`revoke_hold_n=1` 即原型的原始行为（可用 `run_variants()` 做参数消融对照）。

另有一处 **原型与 C++ 的缓冲写入口径差异**（由本模块显式测量，见报告 Q1）：
  原型 `_push()` 把 `median(m3)` 写入本帧槽位（自包含）；C++ `Push()` 把
  `prev_med_total_`（上一帧算出的中值）写入本帧槽位 ⇒ **C++ 的中值总量序列整体滞后 1 帧**。
  因此 `_win_mean` 在过渡段两端会与 C++ 略有出入（稳态下二者相同）。

禁止事项（与本文件无关但必须遵守）：不修改归档原型、不修改 `src/`、不构建、不跑测试。
"""
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import v20_lib as L  # noqa: E402

_PROTO_DIR = os.path.dirname(L.PROTO_V6)
if _PROTO_DIR not in sys.path:
    sys.path.insert(0, _PROTO_DIR)

import glm53_v6 as G  # noqa: E402

# 当前落地参数集 plan-v1.0 A1+A4+A5a
PARAMS_CURRENT = dict(kappa_onset=1.05, kappa_restep=1.12, ho_min_s=3.5, revoke_hold_n=3)
# 归档原型的裸常量（= 参数集之前的版本）
PARAMS_PROTO_RAW = dict(kappa_onset=1.30, kappa_restep=1.12, ho_min_s=5.0, revoke_hold_n=1)

# 事件 kind 名（原型 -> 展示名）
KIND_CN = {
    "onset": "Onset",
    "restep": "Restep",
    "restep_reload": "RestepReload",
    "decrease": "Decrease",
}

RESULT_DIR = os.path.abspath(os.path.join(_HERE, "..", "results"))
FRAMES_CSV = os.path.join(RESULT_DIR, "a1_replay_frames.csv")
EVENTS_CSV = os.path.join(RESULT_DIR, "a1_replay_events.csv")


def _f(x, nd=6):
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return ""


class V6Replay(G.GLM53v6):
    """`GLM53v6` 的可追踪子类：逐帧快照 + 事件/迁移日志 + 扣除通路拆解。

    构造参数（全部可覆盖）：
      kappa_onset / kappa_restep / ho_min_s / revoke_hold_n
      trace_events : 是否记录事件日志（默认 True）
      trace_ded    : 是否记录每条扣除通路（默认 True）
    只读结果：
      frames : list[dict]，每帧一条内部状态快照
      events : list[dict]，每次建事件/交接/重锚/回 idle/撤销 一条
    """

    def __init__(self, n, kappa_onset=PARAMS_CURRENT["kappa_onset"],
                 kappa_restep=PARAMS_CURRENT["kappa_restep"],
                 ho_min_s=PARAMS_CURRENT["ho_min_s"],
                 revoke_hold_n=PARAMS_CURRENT["revoke_hold_n"],
                 trace_events=True, trace_ded=True, med_lag=0):
        self.KAPPA_ONSET = float(kappa_onset)
        self.KAPPA_RESTEP = float(kappa_restep)
        self.HO_MIN = float(ho_min_s)
        self.REVOKE_HOLD_N = int(revoke_hold_n)
        # med_lag=1 复现 C++ 的中值缓冲写入口径：Push 写入 prev_med_total_（滞后 1 帧）。
        # 原型与 C++ 都仍把「本帧中值」用于 prev_med_total_（供下一帧 Push 使用），
        # 唯一差别是写入槽位的是本帧中值(原型) 还是上一帧中值(C++)。
        self.MED_LAG = int(med_lag)
        super().__init__(n)
        # C++ `prev_med_total_` 初值为 0（BufV 数组零初始化），首帧有效帧即写入 0
        self._tr_prev_med = 0.0

        self.trace_events = bool(trace_events)
        self.trace_ded = bool(trace_ded)

        self.frames = []
        self.events = []

        # 注意：追踪用属性一律加 `_tr_` 前缀，避免与 GLM53v6 内部字段撞名
        # （原型已占用 `_d`=检测统计量环缓, `_bv/_bt/_bvm/_bvx/_byx/_bn/_m3/_dn/
        #  `_hit_run/_quiet_run/_armed`）
        self._ts = 0.0
        self._v = None
        self._outv = None
        self._total = 0.0
        self._tr_d = 0.0
        self._thr_d = 0.0
        self._tail_gate = 0.0
        self._idle_now = False
        self._glide_ded = None
        self._slow_ded = None
        self._in_settle = False
        self._rev_run = 0
        self._pending_new = None

    # ───────────────── 工具 ─────────────────
    def _loaded_mask(self):
        amax = float(self.A.max()) if self.A.size else 0.0
        if amax <= 1e-9:
            return None, 0
        m = self.A > self.LOADED_FRAC * amax
        return m, int(m.sum())

    def _snapshot(self, ts):
        gate = self.gamma[self.loaded] if getattr(self, "loaded", None) is not None \
            and self.loaded.any() else self.gamma
        return dict(
            ts=float(ts),
            state=self.state,
            g=float(self.g),
            A_sum=float(self.A.sum()),
            n_loaded=int(self.loaded.sum()),
            gamma_med=float(np.median(gate)) if gate.size else 1.0,
            gamma_min=float(np.min(gate)) if gate.size else 1.0,
            gamma_max=float(np.max(gate)) if gate.size else 1.0,
        )

    def _log_event(self, **kw):
        if not self.trace_events:
            return
        self.events.append(kw)

    # ───────────────── 主流程（只做旁路记录，不改行为） ─────────────────
    def _push(self, ts, total, v, y):
        if self.MED_LAG == 0:
            return super()._push(ts, total, v, y)
        # med_lag=1：写入槽位的是**上一帧算出的 3 帧中值**（= C++ `prev_med_total_`），
        # 其余（_m3 维护、_bt/_bvx/_byx 写入）与父类逐字一致。
        i = self._bn % self.CAP
        self._bt[i] = ts
        self._bv[i] = total
        self._bvx[:, i] = v
        self._byx[:, i] = y
        self._bvm[i] = self._tr_prev_med
        self._m3.append(total)
        if len(self._m3) > 3:
            self._m3.pop(0)
        self._bn += 1
        # 本帧中值（= C++ 里 Push 之后算出的 prev_med_total_）留给下一帧 Push 用
        self._tr_prev_med = float(np.median(self._m3))

    def _win_mean(self, t0, t1):
        r = super()._win_mean(t0, t1)
        # 检测器 D 的两条窗口由 process() 以固定参数调用，借此缓存 d 的组成
        if (abs(t0 - (self._ts - self.DET_FAST)) < 1e-12
                and abs(t1 - self._ts) < 1e-12):
            self._lv_now = r
        elif (abs(t0 - (self._ts - self.DET_FAST - self.DET_GAP - self.DET_LAG)) < 1e-12
                and abs(t1 - (self._ts - self.DET_FAST - self.DET_GAP)) < 1e-12):
            self._lv_ref = r
        return r

    def process(self, ts, v):
        vv = np.asarray(v, float)
        self._ts = float(ts)
        self._v = vv.copy()
        self._total = float(vv.sum())
        self._lv_now = None
        self._lv_ref = None
        self._glide_ded = np.zeros(self.n) if self.trace_ded else None
        self._slow_ded = np.zeros(self.n) if self.trace_ded else None
        self._in_settle = False

        prev_row = self.frames[-1] if self.frames else None
        try:
            out = super().process(ts, vv)
        finally:
            # d / thr_d / tail_gate 与原型 process() 内逐字一致（原型未把它们存成属性）
            lvn, lvr = self._lv_now, self._lv_ref
            self._tr_d = (lvn - lvr) if (lvn is not None and lvr is not None) else 0.0
            ref_mag = abs(lvr if lvr is not None else 0.0)
            self._thr_d = max(self.DET_K * self.sig_d, self.DET_REL * ref_mag,
                              self.DET_ABS_FRAC * self.max_tot)
            if self.ev is not None:
                tau_ep = ts - self.ev["t0"]
            elif self.state == "slow":
                tau_ep = ts - self.ev_end
            else:
                tau_ep = 1e9
            self._tail_gate = (self.TAIL_GATE_FRAC * ref_mag
                               if (self.state != "idle" and tau_ep < self.TAIL_GATE_S) else 0.0)
            eps = 1e-6 * (1.0 + abs(self.max_tot))
            self._idle_now = bool(self.ts_smooth < self.IDLE_FRAC * max(self.level_ref, eps)
                                  or self.ts_smooth < self.UNLOAD_MIN_RATIO * self.min_ts + eps)
            self._outv = out
            self._dump_frame(prev_row)
        return out

    def _dump_frame(self, prev_row):
        ev = self.ev
        gd = self._glide_ded if self._glide_ded is not None else np.zeros(self.n)
        sd = self._slow_ded if self._slow_ded is not None else np.zeros(self.n)
        out = self._outv if self._outv is not None else self._v
        osc = 0
        if prev_row is not None and self.state == "slow":
            if abs(self.g - prev_row["g"]) > 1e-12 or \
               abs(self.A.sum() - prev_row["A_sum"]) > 1e-12:
                osc = 1
        row = dict(
            i=len(self.frames),
            t=self._ts,
            total=self._total,
            out_total=float(np.sum(out)),
            state=self.state,
            kind=(ev["kind"] if ev is not None else ""),
            phase=("event" if ev is not None else self.state),
            tau=((self._ts - ev["t0"]) if ev is not None else ""),
            A_hat=(ev["A_hat"] if ev is not None else ""),
            inc_max=(ev["inc_max"] if ev is not None else ""),
            c_applied=(ev["c_applied"] if ev is not None else ""),
            tau_g0=(ev["tau_g0"] if (ev is not None and ev["tau_g0"] is not None) else ""),
            Tglide=(ev["Tglide"] if ev is not None else ""),
            stalled=(int(ev["stalled"]) if ev is not None else 0),
            g=self.g,
            A_sum=float(self.A.sum()),
            min_ts=self.min_ts,
            max_ts=self.max_ts,
            level_ref=self.level_ref,
            ts_smooth=self.ts_smooth,
            idle_now=int(self._idle_now),
            d=self._tr_d,
            thr_d=self._thr_d,
            tail_gate=self._tail_gate,
            off=float(np.sum(self._v - out)),
            off_glide=float(np.sum(gd)),
            off_slow=float(np.sum(sd)),
            in_settle=int(self._in_settle),
            C6=int(self._rev_run),
            _sd=sd,
        )
        self.frames.append(row)

    # ───────────────── 事件钩子 ─────────────────
    def _new_event(self, t0, base, v0, y0, kind, hist, prev_state):
        h0 = self._snapshot(self._ts)
        super()._new_event(t0, base, v0, y0, kind, hist, prev_state)
        h1 = self._snapshot(self._ts)
        self._log_event(
            ev="NEW_EVENT", ts=self._ts, t0=float(t0), kind=kind,
            base=float(base), d=self._tr_d, thr_d=self._thr_d, tail_gate=self._tail_gate,
            A_sum_before=h0["A_sum"], A_sum_after=h1["A_sum"],
            g_before=h0["g"], g_after=h1["g"],
            state_before=h0["state"], state_after=h1["state"],
            c_before=h0["A_sum"] - float(base), c_after=float(self.ev["c0"]),
            c0_inherit=float(self.ev["c0"]), hist_len=int(len(hist)),
            n_loaded_before=h0["n_loaded"], n_loaded_after=h1["n_loaded"],
            C="", note=("c5" if kind == "restep_reload" else ""),
        )

    def _run_event(self, ts, v, total, dt, eps, idle_now):
        self._idle_now = bool(idle_now)
        ev = self.ev
        tau = ts - ev["t0"]
        # revoke 迟滞（原型只有单帧判据；C++ 为 kRevokeHoldN 帧）
        revoke_now = False
        if self.REVOKE_HOLD_N > 1 and tau < self.REVOKE and \
                ev["inc_max"] > eps and self._revoke_pred(ev, ts, v):
            self._rev_run += 1
            revoke_now = self._rev_run >= self.REVOKE_HOLD_N
        else:
            self._rev_run = 0
        if revoke_now:
            self._log_revoke(ts, ev, total, kind_flag="REVOKE_hyst")
            prev_state = ev["prev_state"]
            y_prev = ev["y_prev"]
            h0 = self._snapshot(ts)
            self.ev = None
            self.state = prev_state if prev_state in ("idle", "slow") else "idle"
            self.last_ev_ts = ts
            self._rev_run = 0
            out = self._revoke_output(prev_state, y_prev, v)
            self._log_event(
                ev="REVOKE", ts=ts, t0=float(ev["t0"]), kind=ev["kind"],
                A_sum_before=h0["A_sum"], A_sum_after=h0["A_sum"],
                g_before=h0["g"], g_after=h0["g"],
                state_before="event", state_after=self.state, C="hyst")
            self._outv = out
            self._dump_frame(self.frames[-1] if self.frames else None)
            return out
        out = super()._run_event(ts, v, total, dt, eps, idle_now)
        if self.ev is None:
            # 原型自带的两条 REVOKE 分支绕过钩子 ⇒ 用去重规则补记一条日志
            self._rev_run = 0
            if not any(e["ev"] == "REVOKE" and abs(float(e["ts"]) - ts) < 1e-9
                       for e in self.events[-6:]):
                self._log_revoke(ts, None, kind_flag="REVOKE_proto")
            elif not any(e["ev"] == "REANCHOR" and abs(float(e["ts"]) - ts) < 1e-9
                         for e in self.events[-4:]):
                self._log_revoke(ts, None, kind_flag="REVOKE_proto")
        return out

    def _revoke_pred(self, ev, ts, v):
        """复算原型的试探撤销判据（不调用父类私有逻辑，避免副作用）。"""
        wm = self._win_mean(ts - 0.10, ts)
        inc_s = (wm - ev["base"]) if wm is not None else (float(np.sum(v)) - ev["base"])
        return inc_s < 0.5 * ev["inc_max"]

    def _revoke_output(self, prev_state, y_prev, v):
        if prev_state == "idle" and y_prev is not None:
            return np.asarray(y_prev, float)
        return np.asarray(v, float).copy()

    def _log_revoke(self, ts, ev, kind_flag):
        h = self._snapshot(ts)
        self._log_event(
            ev="REVOKE", ts=ts, t0=(float(ev["t0"]) if ev is not None else ""),
            kind=(ev["kind"] if ev is not None else ""),
            A_sum_before=h["A_sum"], A_sum_after=h["A_sum"],
            g_before=h["g"], g_after=h["g"],
            state_before="event", state_after=self.state, C=kind_flag)

    def _handoff(self, ts, v, ev):
        h0 = self._snapshot(ts)
        c_applied = float(ev["c_applied"])
        a_hat = float(ev["A_hat"])
        share_n = 0
        A_new_sum = float(np.sum(np.maximum(ev["y0"] + self._share_vector(ev, v)
                                           * max(a_hat, 0.0), 0.0)))
        super()._handoff(ts, v, ev)
        h1 = self._snapshot(ts)
        self._log_event(
            ev="HANDOFF", ts=ts, t0=float(ev["t0"]), kind=ev["kind"],
            A_sum_before=h0["A_sum"], A_sum_after=h1["A_sum"],
            g_before=h0["g"], g_after=h1["g"],
            state_before="event", state_after="slow",
            c0_inherit=c_applied, A_hat=a_hat, A_new_sum=A_new_sum,
            n_loaded_before=h0["n_loaded"], n_loaded_after=h1["n_loaded"],
            C="", note=f"share_n={share_n}")

    def _reanchor(self, ts, v):
        h0 = self._snapshot(ts)
        super()._reanchor(ts, v)
        h1 = self._snapshot(ts)
        self._log_event(
            ev="REANCHOR", ts=ts, t0="", kind="",
            A_sum_before=h0["A_sum"], A_sum_after=h1["A_sum"],
            g_before=h0["g"], g_after=h1["g"],
            state_before=self.state, state_after=self.state,
            n_loaded_before=h0["n_loaded"], n_loaded_after=h1["n_loaded"], C="")

    def _to_idle(self):
        h0 = self._snapshot(self._ts)
        super()._to_idle()
        h1 = self._snapshot(self._ts)
        self._log_event(
            ev="TO_IDLE", ts=self._ts, t0="", kind="",
            A_sum_before=h0["A_sum"], A_sum_after=h1["A_sum"],
            g_before=h0["g"], g_after=h1["g"],
            state_before=h0["state"], state_after="idle",
            n_loaded_before=h0["n_loaded"], n_loaded_after=0, C="")

    # ───────────────── 扣除通路拆解 ─────────────────
    def _deduction_vector(self, v):
        d = super()._deduction_vector(v)
        if self._slow_ded is not None:
            # 同一帧内可能被调用多次（沉降窗 hold 分支），取绝对值最大者作为该帧上限
            take = np.abs(d) > np.abs(self._slow_ded)
            self._slow_ded = np.where(take, d, self._slow_ded)
        return d

    def _slow_step(self, ts, v, dt):
        Z = np.asarray(v, float)
        m = self.loaded & (self.A > 1e-9)
        if self.hold_comp is not None:
            self._in_settle = True
            if self._slow_ded is not None and m.any():
                hc = np.asarray(self.hold_comp, float)
                self._slow_ded = np.where(m, hc, self._slow_ded)
        out = super()._slow_step(ts, v, dt)
        if not self._in_settle and self._slow_ded is not None:
            ded = np.zeros(self.n)
            if m.any():
                idx = np.where(m)[0]
                ded[idx] = np.clip(self.gamma[idx] * self.A[idx] * self.g,
                                   self.CREEP_LO * self.A[idx], self.CREEP_HI * self.A[idx])
            self._slow_ded = ded
        return out

    def _share_vector(self, ev, v):
        s = super()._share_vector(ev, v)
        if self._glide_ded is not None:
            self._glide_ded = -(s * float(ev["c_applied"]))
        return s


# ───────────────────────── 回放 / 变体 ─────────────────────────

def replay_pre(ds=None, **kw):
    """按算法输入流回放一遍，返回 `V6Replay` 实例（frames/events 已填好）。"""
    if ds is None:
        ds = L.load_dataset(L.DS_ZERO)
    pre = ds["pre"]
    rp = V6Replay(L.NCH, **kw)
    for ts, v in zip(pre["ts"], pre["V"]):
        rp.process(float(ts), v)
    return rp, ds


def run_variants(param_sets):
    """param_sets: dict[name -> kwargs]；返回 dict[name -> V6Replay]。"""
    ds = L.load_dataset(L.DS_ZERO)
    return {name: replay_pre(ds, **kw)[0] for name, kw in param_sets.items()}, ds


# ───────────────────────── 平台切片 ─────────────────────────

def platforms(pre, hyst_frac=0.25, min_dur=0.5):
    """基于 `pre` 总量（算法输入 = 真值参考）切 idle/loaded 平台段。"""
    tot = pre["V"].sum(1)
    return L.plateau_segments(tot, pre["el"], hyst_frac=hyst_frac, min_dur=min_dur)


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for r in rows:
            w.writerow(r)
    return path


FRAME_FIELDS = ["i", "t", "total", "out_total", "state", "kind", "phase", "tau",
                "A_hat", "inc_max", "c_applied", "tau_g0", "Tglide", "stalled",
                "g", "A_sum", "min_ts", "max_ts", "level_ref", "ts_smooth",
                "idle_now", "d", "thr_d", "tail_gate",
                "off", "off_glide", "off_slow", "in_settle", "C6"]

EVENT_FIELDS = ["ev", "ts", "t0", "kind", "base", "d", "thr_d", "tail_gate",
                "A_sum_before", "A_sum_after", "g_before", "g_after",
                "state_before", "state_after", "c_before", "c_after",
                "c0_inherit", "A_hat", "A_new_sum", "hist_len",
                "n_loaded_before", "n_loaded_after", "C", "note"]


def dump_frames(rp, path=FRAMES_CSV):
    rows = []
    for r in rp.frames:
        rows.append([r["i"], _f(r["t"], 4), _f(r["total"], 2), _f(r["out_total"], 2),
                     r["state"], r["kind"], r["phase"],
                     _f(r["tau"], 4) if r["tau"] != "" else "",
                     _f(r["A_hat"], 3) if r["A_hat"] != "" else "",
                     _f(r["inc_max"], 3) if r["inc_max"] != "" else "",
                     _f(r["c_applied"], 3) if r["c_applied"] != "" else "",
                     _f(r["tau_g0"], 4) if r["tau_g0"] != "" else "",
                     _f(r["Tglide"], 4) if r["Tglide"] != "" else "",
                     r["stalled"],
                     _f(r["g"], 6), _f(r["A_sum"], 2), _f(r["min_ts"], 2),
                     _f(r["max_ts"], 2), _f(r["level_ref"], 2), _f(r["ts_smooth"], 2),
                     r["idle_now"], _f(r["d"], 3), _f(r["thr_d"], 3), _f(r["tail_gate"], 3),
                     _f(r["off"], 2), _f(r["off_glide"], 2), _f(r["off_slow"], 2),
                     r["in_settle"], r["C6"]])
    return write_csv(path, rows, FRAME_FIELDS)


def dump_events(rp, path=EVENTS_CSV):
    rows = []
    for e in rp.events:
        rows.append([
            e.get("ev", ""), _f(e.get("ts"), 4),
            (_f(e.get("t0"), 4) if e.get("t0", "") != "" else ""),
            e.get("kind", ""),
            _f(e.get("base"), 3) if e.get("base", "") != "" else "",
            _f(e.get("d"), 3), _f(e.get("thr_d"), 3), _f(e.get("tail_gate"), 3),
            _f(e.get("A_sum_before"), 2), _f(e.get("A_sum_after"), 2),
            _f(e.get("g_before"), 6), _f(e.get("g_after"), 6),
            e.get("state_before", ""), e.get("state_after", ""),
            _f(e.get("c_before"), 2) if e.get("c_before", "") != "" else "",
            _f(e.get("c_after"), 2) if e.get("c_after", "") != "" else "",
            _f(e.get("c0_inherit"), 2) if e.get("c0_inherit", "") != "" else "",
            _f(e.get("A_hat"), 3) if e.get("A_hat", "") != "" else "",
            _f(e.get("A_new_sum"), 2) if e.get("A_new_sum", "") != "" else "",
            e.get("hist_len", ""),
            e.get("n_loaded_before", ""), e.get("n_loaded_after", ""),
            e.get("C", ""), e.get("note", ""),
        ])
    return write_csv(path, rows, EVENT_FIELDS)
