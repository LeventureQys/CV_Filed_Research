# -*- coding: utf-8 -*-
"""T1-B / 07：**代价-收益矩阵**（T1-Q10）。

产物：results/t1b_cost_benefit.csv、results/_t1b_07_costbenefit.log

口径（逐字按 T1-Q10）：
  * **收益**：抖动（白噪 r=2% 电平，10 个种子）与拍击（30%/50% 电平，50/100/50 ms）下的
    误触发/撤销/最大显示偏差；另加 ADC 域强噪声（white 2000 ADC）下的**漏检数**（救援改造的对象）。
  * **代价**：`ΔT_stable` = 同一批 clean onset 上该改造相对 `base(=原型)` 的 T_stable 变化
    （中位 / p90），以及**建事件延迟** `ep_latency`（t_det − t_on）。
  * 第一轮结论「+0.15~0.25 s 驻留最划算」「CAPF 封顶被证伪」在本矩阵里是**两个点**，
    其余旋钮（DET_REL / DET_K / REVOKE / IDLE_FRAC / RESCUE）是本次新增的系统扫描。
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(HERE)
RES = os.path.join(TASK, "results")
sys.path.insert(0, HERE)

SEEDS = list(range(10))
EV_KEYS = [("RT2", 12.32), ("LT1", 11.69), ("F42", 11.03), ("RT3", 10.59)]
TAP_FRACS = [0.10, 0.30, 0.50]
NOISE_R = 0.02
NOISE_R_DET = 0.05           # 确定性单实现档（σ_d 开始绑定，见 07b 旋钮自检）
RESCUE_KNOBS = ["base", "dwell0.15", "dwell0.25", "capf1.0", "rescue0.40"]
ADC_AMP = 2000.0
CROP_S = 40.0

_CLEAN = {}


def _knob(name, L):
    from t1b_variants import KNOBS
    for n, cls, kw in KNOBS:
        if n == name:
            return cls, kw
    raise KeyError(name)


_TRACED = {}


def run_knob(name, tu, Xu):
    """跑一个旋钮设置（用仪表化子类，记录 epoch 的 t0 与 t_det）。"""
    L = __import__("t1b_lib")
    from t1_common import make_traced, run_traced
    cls, kw = _knob(name, L)
    if name not in _TRACED:
        _TRACED[name] = make_traced(cls)
    d = run_traced(_TRACED[name], tu, Xu, **kw)
    return dict(Z=d["Y"].sum(axis=1), epoch=d["epoch"], revoke=d["revoke"],
                handoff=d["handoff"], comp=d["comp"])


def worker(spec):
    L = __import__("t1b_lib")
    from t1_common import add_tap, make_perturb
    key, t_on = spec["key"], spec["t_on"]
    d = L.get_grid(key)
    tu, Xu = L.window_of(d, dict(kind="hold", t_on=t_on, span=CROP_S))
    scen, knob = spec["scenario"], spec["knob"]
    X = Xu
    t_tap = t_on + 20.0
    i_tap = int(round(t_tap / 0.01))
    if scen.startswith("tap"):
        frac = float(scen[3:]) / 100.0
        amp = frac * float(spec["L_hold"])
        X, _ = add_tap(Xu, i_tap, amp, rise_ms=50, hold_ms=100, fall_ms=50)
    elif scen == "noise":
        rng = np.random.default_rng(30000 + spec["seed"])
        X = Xu + make_perturb(Xu, NOISE_R * float(spec["L_hold"]), "white", rng)
    elif scen == "noise5":
        rng = np.random.default_rng(30000 + 7)
        X = Xu + make_perturb(Xu, NOISE_R_DET * float(spec["L_hold"]), "white", rng)
    r = run_knob(knob, tu, Xu if scen == "clean" else X)
    Z = r["Z"]
    m = L.eval_event(tu, Z, t_on, float(spec["J"]), float(spec["t_next"]))
    ck = (key, round(t_on, 2))
    if ck not in _CLEAN:
        _CLEAN[ck] = run_knob("base", tu, Xu)["Z"]
    md, sd = L.maxdev(tu, _CLEAN[ck], Z, t_on)
    ep = np.array([e[0] for e in r["epoch"]], float)
    ep_pre = np.array([e[2] for e in r["epoch"]], float)
    lat = np.nan
    if len(ep):
        after = ep[ep >= t_on - 0.5]
        if len(after):
            j = int(np.argmin(np.abs(ep - after[0])))
            lat = float(ep_pre[j] - t_on)
    ho, rv = r["comp"], r["comp"]  # noqa: F841  (保留以便后续扩展读取内部量)
    _ = (ho, rv)
    return [dict(scope="hold", key=key, ev=f"{key}@{t_on:.2f}", t_on=t_on, knob=knob,
                 scenario=scen, seed=spec["seed"], J=float(spec["J"]),
                 L_hold=float(spec["L_hold"]),
                 t_stable_v1=m["t_stable_v1"], t_stable_v2=m["t_stable_v2"],
                 t_stable_ok=m["t_stable_ok"], os_pct=m["os_pct"], z_at_1=m["z_at_1"],
                 maxdev=md, slow_dev=sd, n_epoch=len(r["epoch"]), ep_latency=lat,
                 n_epoch_near_tap=int(np.sum(np.abs(ep - t_tap) <= 1.0)) if len(ep) else 0,
                 n_rescue=int(getattr(r["comp"], "n_rescue", 0)))]


def worker_adc(spec):
    """ADC 域：强噪声下的漏检（救援改造的主要收益面）。"""
    L = __import__("t1b_lib")
    from t1_common import make_perturb
    key = spec["key"]
    d = L.get_grid(key)
    tu, Xu = d["tu"], d["Xu"]
    truth = L.truth_events_all()[key]
    rng = np.random.default_rng(40000 + spec["seed"])
    X = Xu + make_perturb(Xu, ADC_AMP, "white", rng)
    r = run_knob(spec["knob"], tu, X)
    mt = L.match_truth(truth, r["epoch"])
    return [dict(scope="adc", key=key, ev=f"{key}(全长)", t_on=0.0, knob=spec["knob"],
                 scenario=f"adc_noise{int(ADC_AMP)}", seed=spec["seed"], J=np.nan,
                 L_hold=np.nan, t_stable_v1=np.nan, t_stable_v2=np.nan, t_stable_ok=0,
                 os_pct=np.nan, z_at_1=np.nan, maxdev=np.nan, slow_dev=np.nan,
                 n_epoch=len(r["epoch"]), ep_latency=np.nan, n_epoch_near_tap=-1,
                 n_miss=len(mt[1]), n_extra=len(mt[2]),
                 n_rescue=int(getattr(r["comp"], "n_rescue", 0)))]


def build_tasks():
    import t1b_lib as L
    from t1b_variants import KNOBS
    ev = L.event_table().set_index("ev")
    tasks = []
    for name, _, _ in KNOBS:
        for key, t_on in EV_KEYS:
            row = ev.loc[f"{key}@{t_on:.2f}"]
            base = dict(key=key, t_on=t_on, J=float(row["J"]), L_hold=float(row["L_hold"]),
                        t_next=float(row["t_next"]), knob=name)
            tasks.append(dict(**base, scenario="clean", seed=0))
            for f in TAP_FRACS:
                tasks.append(dict(**base, scenario=f"tap{int(f*100)}", seed=0))
            for s in SEEDS:
                tasks.append(dict(**base, scenario="noise", seed=s))
            tasks.append(dict(**base, scenario="noise5", seed=0))
    for name in RESCUE_KNOBS:
        for s in SEEDS:
            tasks.append(dict(key="SW4", knob=name, seed=s))
    return tasks


def log_write(path, s):
    with open(path, "a", encoding="utf-8") as f:
        f.write(s + "\n")
    print(s, flush=True)


def main():
    t0 = time.time()
    out = os.path.join(RES, "t1b_cost_benefit.csv")
    logp = os.path.join(RES, "_t1b_07_costbenefit.log")
    log_write(logp, "=== T1-B / 07 代价-收益矩阵 ===")
    log_write(logp, "cmd: python scripts/t1b_07_cost_benefit.py")
    tasks = build_tasks()
    n_adc = sum(1 for t in tasks if "scenario" not in t)
    nw = max(1, min(3, (os.cpu_count() or 4) - 6))
    log_write(logp, f"任务数 {len(tasks)}（其中 ADC 强噪声 {n_adc}）；workers={nw}")
    done, rows = 0, []
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = []
        for t in tasks:
            futs.append(ex.submit(worker_adc, t) if "scenario" not in t else ex.submit(worker, t))
        for fut in as_completed(futs):
            rows.extend(fut.result())
            done += 1
            if done % 30 == 0 or done == len(tasks):
                el = time.time() - t0
                log_write(logp, f"  {done}/{len(tasks)} 任务，{len(rows)} 行，{el:.0f}s"
                                f"（剩约 {(len(tasks)-done)*el/max(done,1)/60:.1f} min）")
                pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    log_write(logp, f"完成 {len(df)} 行，{(time.time()-t0)/60:.1f} min")

    h = df[df.scope == "hold"]
    base_clean = h[(h.knob == "base") & (h.scenario == "clean")]
    b_med = base_clean.t_stable_v1.median()
    b_tap = h[(h.knob == "base") & (h.scenario == "tap30")].n_epoch_near_tap.median()
    b_dev = h[(h.knob == "base") & (h.scenario == "noise")].maxdev.median()
    log_write(logp, f"base: clean T_stable 中位 {b_med:.2f}s；tap30 拍击窗 epoch 中位 {b_tap:.1f}；"
                    f"白噪 r=2% maxdev 中位 {b_dev:.2f}")
    log_write(logp, f"{'knob':12s} {'cleanT':>7s} {'ΔT':>7s} {'noiseT':>7s} {'tap30ep':>8s} "
                    f"{'tap50ep':>8s} {'revoke':>7s} {'maxdev':>8s} {'lat':>7s}")
    for name, g in h.groupby("knob"):
        cl = g[g.scenario == "clean"]
        nz = g[g.scenario == "noise"]
        t30 = g[g.scenario == "tap30"]
        t50 = g[g.scenario == "tap50"]
        log_write(logp, f"{name:12s} {cl.t_stable_v1.median():7.2f} "
                        f"{cl.t_stable_v1.median()-b_med:+7.2f} "
                        f"{nz[nz.t_stable_ok==1].t_stable_v1.median():7.2f} "
                        f"{t30.n_epoch_near_tap.median():8.1f} {t50.n_epoch_near_tap.median():8.1f} "
                        f"{t30.n_epoch_near_tap.mean():7.2f} {nz.maxdev.median():8.3f} "
                        f"{cl.ep_latency.median():7.3f}")
    a = df[df.scope == "adc"]
    log_write(logp, "--- ADC 域 white 2000 ADC：漏检（SW4 全长，n=10 种子）---")
    for name, g in a.groupby("knob"):
        log_write(logp, f"  {name:12s} 漏={g.n_miss.mean():5.2f} 误={g.n_extra.mean():5.2f} "
                        f"epoch={g.n_epoch.mean():5.2f} rescue={g.n_rescue.mean():6.1f}")
    log_write(logp, "ALL DONE")
    with open(os.path.join(RES, "_t1b_07_done.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rows=len(df), minutes=round((time.time() - t0) / 60, 1)), f)


if __name__ == "__main__":
    main()
