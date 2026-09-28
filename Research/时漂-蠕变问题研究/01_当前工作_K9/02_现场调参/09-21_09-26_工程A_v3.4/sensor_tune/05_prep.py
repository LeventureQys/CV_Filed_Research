# -*- coding: utf-8 -*-
"""05 预处理：为每份会话算出「参考弹性电平 ΣE」与「加载时刻 t0」。

口径（重要）
-----------
1. **调零口径**：`CreepObserverCompensator` 内部一律用 y = v − zero（zero = 首帧读数）。
   上位机的现场规范是「装好负载 → 调零 → 开算法」，调零把显示值抬到 ≈0。
   本批数据里手掌/拇指会话**录制时就带着无载偏置**（4900~25000 ADC）；若直接喂原值，
   全局总值旁路（release = 1.15×baseline）会整场判空载、逐帧直通、完全不补偿。
   故正式评估统一按调零口径：输入 = 原值 − 首帧值（四指指腹 d1 首帧本就是 0）。
2. **t0**：升到总升幅 50% 用时 < 5 s ⇒ 存在真实加载台阶，t0 = 台阶前最后时刻；
   否则视为「录制开始时已受载」，t0 = 0（该会话的整个上升都算蠕变）。
3. **参考电平 ΣE**：模型 = 弹性电平 E + 快蠕变 c1(1−e^{−t/τ1}) + 慢蠕变 c2(1−e^{−t/τ2})。
   E 被约束在 [0, v(t0+0.6s)]：对「录制时已受载」的会话即「加载后 0.6 s 的读数」，
   对四指指腹即台阶平台电平。ΣE_k（逐通道和）= 显示被完全补偿后应钉住的水位；
   c1+c2 = 应被扣掉的真实累计蠕变。

产物：out/02_prep.json + out/_prep/<key>.npz（逐通道输入与 E_k，备用）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sensor_common import OUT_DIR, SENSORS, load  # noqa: E402

RISE_S = 0.6


def creep_model(t, E, c1, tau1, c2, tau2, t0, rise):
    x = np.maximum(t - t0, 0.0)
    ramp = np.minimum(x / rise, 1.0)
    rel = np.maximum(x - rise, 0.0)
    return E + (c1 * (1.0 - np.exp(-rel / tau1)) + c2 * (1.0 - np.exp(-rel / tau2))) * ramp


def detect_t0(t: np.ndarray, tin: np.ndarray) -> tuple[float, bool]:
    v_start = float(np.median(tin[t <= min(0.2, t[-1])]))
    peak = float(tin.max())
    rise = peak - v_start
    if rise <= 1e-9:
        return 0.0, False
    i50 = int(np.argmax(tin >= v_start + 0.5 * rise))
    if t[i50] <= 5.0:
        thr = v_start + 0.05 * rise
        i = int(np.argmax(tin >= thr))
        return max(float(t[max(i - 1, 0)]), 0.0), True
    return 0.0, False


def _e_hi(t, tin, t0):
    i = min(int(np.searchsorted(t, t0 + RISE_S)), len(t) - 1)
    return max(float(tin[i]), 1.0)


def fit_total(t, tin, t0):
    m = t >= t0 + RISE_S
    peak = float(tin.max())
    span = max(peak - float(tin[0]), 1.0)
    e_hi = _e_hi(t, tin, t0)
    p0 = [min(max(float(tin[m][0]), 0.0), e_hi), span * 0.20, 5.0, span * 0.35, 40.0]
    lo = [0.0, 0.0, 0.3, 0.0, 1.0]
    hi = [e_hi, span * 2.0, 60.0, span * 3.0, 900.0]
    p0 = [min(max(p0[i], lo[i]), hi[i]) for i in range(5)]
    popt, _ = curve_fit(
        lambda tt, E, c1, tau1, c2, tau2: creep_model(tt, E, c1, tau1, c2, tau2, t0, RISE_S),
        t[m], tin[m], p0=p0, bounds=(lo, hi), maxfev=80000)
    return popt


def fit_channels(t, V, tau1, tau2, t0):
    m = t >= t0 + RISE_S
    span = max(float(V.sum(axis=1).max()) - float(V[0].sum()), 1.0)
    e_hi = _e_hi(t, V.sum(axis=1), t0)
    per_hi = max(e_hi / V.shape[1] * 3.0, 1.0)
    Es = np.zeros(V.shape[1])
    cs = np.zeros((V.shape[1], 2))
    for k in range(V.shape[1]):
        lo = [0.0, 0.0, 0.0]
        hi = [per_hi, span, span]
        p0 = [min(max(float(V[m][:, k][0]), 0.0), per_hi), span * 0.005, span * 0.01]
        p0 = [min(max(p0[i], lo[i]), hi[i]) for i in range(3)]
        try:
            popt, _ = curve_fit(
                lambda tt, E, c1, c2: creep_model(tt, E, c1, c2, tau1, tau2, t0, RISE_S),
                t[m], V[m, k], p0=p0, bounds=(lo, hi), maxfev=60000)
            Es[k], cs[k, 0], cs[k, 1] = popt
        except Exception:
            Es[k] = max(float(V[m][:, k].mean()), 0.0)
    return Es, cs


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "_prep").mkdir(parents=True, exist_ok=True)
    out = {}
    print(f"{'会话':<24s} {'t0':>6s} {'台阶':>4s} {'ΣE':>9s} {'蠕变':>8s} {'蠕变%':>7s} "
          f"{'快c1':>7s} {'τ1':>6s} {'慢c2':>7s} {'τ2':>6s} {'末值':>8s} {'首帧原值':>9s}")
    for sensor, info in SENSORS.items():
        for tag, rel in info["sessions"].items():
            d = load(rel)
            t = d["t"] - d["t"][0]
            Vr = d["V"]
            V = Vr - Vr[0]
            tin = V.sum(axis=1)
            t0, has_step = detect_t0(t, tin)
            E, c1, tau1, c2, tau2 = fit_total(t, tin, t0)
            Es, cs = fit_channels(t, V, tau1, tau2, t0)
            Esum = float(Es.sum())
            creep = float(tin[-1] - Esum)
            key = f"{sensor}/{tag}"
            out[key] = {
                "sensor": sensor, "tag": tag, "path": str(d["path"]),
                "frames": int(len(t)), "channels": int(V.shape[1]),
                "duration_s": float(t[-1]), "v0_total_raw": float(Vr[0].sum()),
                "t0": float(t0), "has_step": bool(has_step),
                "E_total_fit": float(E), "Esum_channels": Esum, "creep_total": creep,
                "creep_pct": float(creep / max(Esum, 1.0) * 100.0),
                "c1": float(c1), "tau1": float(tau1), "c2": float(c2), "tau2": float(tau2),
                "in_end": float(tin[-1]),
                "v_at_t0_plus": float(tin[min(int(np.searchsorted(t, t0 + RISE_S)), len(t) - 1)]),
            }
            np.savez(OUT_DIR / "_prep" / (key.replace("/", "_") + ".npz"),
                     t=t, tin=tin, V0=Vr[0], Es=Es, cs=cs, t0=t0)
            print(f"{key:<24s} {t0:6.2f} {str(has_step):>4s} {Esum:9.1f} {creep:8.1f} "
                  f"{creep / max(Esum, 1.0) * 100:7.2f} {c1:7.1f} {tau1:6.1f} {c2:7.1f} "
                  f"{tau2:6.1f} {tin[-1]:8.0f} {Vr[0].sum():9.0f}", flush=True)
    (OUT_DIR / "02_prep.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n写出 {OUT_DIR / '02_prep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
