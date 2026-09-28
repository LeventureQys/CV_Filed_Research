# -*- coding: utf-8 -*-
"""r1：三阶段分解（阶跃 / 快相爬升 / 慢相爬升），产出 results/phases.csv。

口径（先定死，避免和慢相蠕变混淆）：
  以受载段内「最陡上升帧」为加载沿 t_e（用 0.3 s 平滑后的主通道）：
    A_step = y(t_e+0.20 s) − pre      —— 机械阶跃（0.20 s 内基本到位的那一段）
    A_5s   = y(t_e+5.0  s) − pre      —— 阶跃 + 快相（v5 让快相跑 5 s 的口径）
    A_end  = y(t_e+末端 1 s 均值) − pre —— 含慢相蠕变的总涨幅
  三阶段：
    阶跃段：0 → t_e+0.2 s（幅度 A_step，时长 = 从其 10% 到 90% 的时间）
    快相段：t_e+0.2 s → t_e+5 s（幅度 A_5s−A_step；给出 10%→90% 时间与形态）
    慢相段：t_e+5 s → 段末（幅度 A_end−A_5s，占 A_5s 的百分比；单指数拟合 τ）
"""
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
FLASH = os.path.dirname(os.path.dirname(OUT))
TEMP = os.path.dirname(FLASH)
RES = os.path.join(OUT, "results")
os.makedirs(RES, exist_ok=True)
sys.path.insert(0, os.path.join(FLASH, "progress", "04-v5", "scripts"))
import ad_lib as L  # noqa: E402

B = os.path.join(TEMP, "变化负载")
ALL = [(f"{loc}/数据{i}", os.path.join(TEMP, loc, f"数据{i}", "device_001_seg000.csv"))
       for loc in ("右拇指指尖", "左拇指指尖", "四指指尖") for i in (1, 2, 3)] + [
    ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                       "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
    ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
    ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
    ("中途切换-13ffca", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                     "最终测试目标", "device_001_seg000.csv")),
]
MAIN_CH = {"右拇指指尖": 17, "左拇指指尖": 18, "四指指尖": 11}


def find_segments(tot, dt, frac=0.20, min_s=4.0):
    thr = frac * np.percentile(tot, 99.5)
    ld = tot > thr
    d = np.diff(ld.astype(int))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld) - 1]
    return [(a, min(b, len(ld) - 1)) for a, b in zip(s, e) if (b - a) * dt >= min_s]


def analyze_segment(tu, y, a, b, dt, k=0.3):
    """返回该受载段的三阶段量化。"""
    w = max(1, int(k / dt))
    ys = L.med_smooth(y, w)
    seg = ys[a:b]
    if len(seg) < int(6 / dt):
        return None
    # 加载沿 = 段内最陡（单帧最大正跳变，用 0.2 s 窗差分抗噪）
    dw = max(1, int(0.2 / dt))
    dif = seg[dw:] - seg[:-dw]
    i_e = int(np.argmax(dif)) + dw
    t_e = a + i_e
    pre = float(np.median(ys[max(0, t_e - int(2.0 / dt)):t_e - int(0.3 / dt) + 1])) \
        if t_e > int(2.3 / dt) else float(ys[max(0, t_e - w)])
    yy = lambda s: float(np.median(ys[t_e + int(s / dt) - max(1, w // 2):
                                      t_e + int(s / dt) + max(2, w // 2)]))
    A_step = yy(0.20) - pre
    A_5s = yy(5.0) - pre
    A_end = float(np.median(ys[max(t_e, b - int(1.0 / dt)):b])) - pre
    # 阶跃段 10%→90%（在 pre → A_step 之间）
    out = dict(t_e=round(float(tu[t_e]), 2), pre=round(pre, 2),
               A_step=round(A_step, 2), A_5s=round(A_5s, 2), A_end=round(A_end, 2),
               fast_amp=round(A_5s - A_step, 2),
               fast_frac=round(100 * (A_5s - A_step) / A_5s, 1) if A_5s else np.nan,
               slow_amp=round(A_end - A_5s, 2),
               slow_frac=round(100 * (A_end - A_5s) / A_5s, 1) if A_5s else np.nan,
               step_frac=round(100 * A_step / A_5s, 1) if A_5s else np.nan)
    for tag, f in (("t50", 0.5), ("t90", 0.9)):
        tgt = pre + f * A_step
        kk = np.where(ys[t_e:t_e + int(0.3 / dt)] >= tgt)[0]
        out["step_" + tag] = round(float(kk[0] * dt), 3) if len(kk) else np.nan
    for tag, f in (("t50", 0.5), ("t90", 0.9)):
        tgt = pre + A_step + f * (A_5s - A_step)
        kk = np.where(ys[t_e:t_e + int(5.0 / dt)] >= tgt)[0]
        out["fast_" + tag] = round(float(kk[0] * dt), 3) if len(kk) else np.nan
    # 快相形态：t_e+0.2s 起、归一到 5 s 的 17 点轮廓（供稳定性评估）
    grid = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                     0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00])
    y0 = yy(0.20)
    den = (yy(5.0) - y0)
    out["prof"] = [round((yy(float(g)) - y0) / den, 4) if abs(den) > 1e-9 else np.nan
                   for g in grid] if A_5s else [np.nan] * len(grid)
    # 慢相单指数拟合
    tt = tu[t_e + int(5.0 / dt):b] - tu[t_e + int(5.0 / dt)]
    vv = ys[t_e + int(5.0 / dt):b] - pre
    if len(tt) > 30 and A_5s > 0:
        try:
            from scipy.optimize import curve_fit
            f = lambda t, c, tau: A_5s + c * (1 - np.exp(-t / tau))
            p, _ = curve_fit(f, tt, vv, p0=[max(A_end - A_5s, 1e-3), 100.0],
                             maxfev=20000, bounds=([-1e6, 1.0], [1e6, 1e5]))
            out["slow_amp_fit"] = round(float(p[0]), 2)
            out["slow_tau"] = round(float(p[1]), 1)
        except Exception:                                                     # noqa: BLE001
            out["slow_amp_fit"] = out["slow_tau"] = np.nan
    out["_idx"] = t_e
    return out


def main():
    rows, profs = [], []
    for name, path in ALL:
        if not os.path.isfile(path):
            print("!! 缺:", path)
            continue
        d = L.prep(path)
        tu, Xu, dt = d["tu"], d["Xu"], d["dtm"]
        tot = Xu.sum(axis=1)
        loc = name.split("/")[0]
        ch = MAIN_CH.get(loc)
        if ch is None or ch >= Xu.shape[1]:
            ch = int(np.argmax(Xu[int(len(tu) * 0.2):int(len(tu) * 0.8)].std(axis=0)))
        y = Xu[:, ch] / max(dt, 1e-9)
        for k, (a, b) in enumerate(find_segments(tot, dt)):
            r = analyze_segment(tu, y, a, b, dt)
            if r is None:
                continue
            r.update(rec=name, seg=k, ch=ch, t_end=round(float(tu[b]), 2))
            rows.append(r)
            profs.append(dict(rec=name, seg=k, stage="onset" if r["pre"] < 0.15 * max(r["A_5s"], 1) else "restep",
                              A_5s=r["A_5s"], **{("f%02d" % i): v for i, v in enumerate(r.pop("prof"))}))
    df = pd.DataFrame(rows)
    for c in ("rec", "seg", "ch", "t_e", "t_end", "pre", "A_step", "A_5s", "A_end",
              "fast_amp", "fast_frac", "slow_amp", "slow_frac", "step_frac",
              "step_t50", "step_t90", "fast_t50", "fast_t90", "slow_tau", "slow_amp_fit"):
        if c in df:
            df = df[[c] + [x for x in df.columns if x != c]]
    df.to_csv(os.path.join(RES, "phases.csv"), index=False, encoding="utf-8-sig")
    pf = pd.DataFrame(profs)
    pf.to_csv(os.path.join(RES, "fastphase_profiles.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 240)
    print(df.drop(columns=["_idx"], errors="ignore").to_string(index=False))
    print("\n-> results/phases.csv（%d 段）、results/fastphase_profiles.csv（%d 条）" % (len(df), len(pf)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
