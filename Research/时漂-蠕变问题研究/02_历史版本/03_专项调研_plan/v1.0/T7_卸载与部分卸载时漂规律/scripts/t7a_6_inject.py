# -*- coding: utf-8 -*-
"""T7-A / step6：**注入补充实验**（标注：注入，非实测）——卸载幅度与卸载沿速度对补偿器的影响。

真实数据只有 14 处卸载沿、且幅度/沿速度都由人手工决定，无法扫参数。
本脚本在**真实录制**上把加载段的读数按 α 比例拉低到空载电平（α=卸载幅度占当前电平的比例），
沿时长 T 用余弦半波（与 00 号文档 §5 的口径一致：`rise_ms` 默认 50 ms，此处扫 20/300 ms），
再逐帧跑 v5.1(e3s) 与 v6，量三件事：
  ① 额外下冲 dip_extra = 原始下冲 − 该臂下冲（占本次注入幅度）
  ② 额外贴零时长 extra_pin = 该臂「读数 ≤ max(3σ, 0.5%·幅度)」的时长 − 原始的同口径时长
  ③ 输出相对"注入后电平"的稳态偏差（冻结偏差 Δ_frozen）
**只用于回答"补偿器行为随幅度/沿速度怎么变"，不用于任何物理规律结论；所有行标 `injected=True`。**

产出：results/t7_inject_amp.csv、results/_t7a_6_inject.log
用法：python scripts/t7a_6_inject.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t7a_common as C                                          # noqa: E402
from t7a_glm53_v51 import GLM53v51                              # noqa: E402
from t7a_glm53_v6 import GLM53v6                                # noqa: E402

LOG = C.Log("6_inject")
CASES = [("右拇指指尖/数据1", 115.37), ("中途切换-13ffca", 68.90), ("左拇指指尖/数据1", 147.38)]
ALPHAS = (0.3, 0.6, 1.0)
EDGES = (0.02, 0.30)


def inject(Xu, tu, t0, edge_s, alpha, lift, share, t_revert):
    """把「卸载沿」换成受控的 cos 半波：总读数从 plat 在 edge_s 内降到
    `plat − α·(plat − z0)`；实现为把 t0 之后的读数**加性抬高** lift·share（逐通道，
    share = 受载通道的相对占比，Σ share = 1），使"残留的载荷"落在原来受载的通道上。
    （必须逐通道分配：Z 是 31/21 通道之和，直接给每个通道加 lift 会把总量放大 N 倍。）
    在下一个真实变载前 0.3 s 内把注入量平滑回 0，保证后面的重载不被改写。
    α=1 时 lift=0 ⇒ 该行等于**原始信号**（自洽性核对）。
    """
    n = len(tu)
    dtm = float(tu[1] - tu[0])
    i0 = int(round(t0 / dtm))
    ne = max(1, int(round(edge_s / dtm)))
    w = np.zeros(n)
    w[i0:] = 1.0
    for k in range(i0, min(n, i0 + ne)):
        w[k] = 0.5 * (1.0 - np.cos(np.pi * (k - i0) / ne))
    if np.isfinite(t_revert):
        ie = min(n, int(round(t_revert / dtm)))
        ir = max(i0, ie - int(round(0.3 / dtm)))
        for k in range(ir, ie):
            w[k] *= 0.5 * (1.0 + np.cos(np.pi * (k - ir) / max(1, ie - ir)))
        w[ie:] = 0.0
    add = np.outer(w, lift * share)
    return Xu + add


def run(arm, tu, Xu):
    c = GLM53v51(Xu.shape[1]) if arm == "e3s" else GLM53v6(Xu.shape[1])
    if arm == "e3s":
        c.FAST_S, c.EXEMPT_AWIN, c.LEV_ARM_S = 3.0, 1.0, 3.0
    Y = np.empty(len(tu))
    for i in range(len(tu)):
        Y[i] = float(np.sum(c.process(float(tu[i]), Xu[i])))
    return Y


def main():
    C.log_reconfigure()
    LOG("=" * 122)
    LOG("T7-A step6：**注入补充实验（injected=True，非实测）**——卸载幅度 α 与卸载沿时长 T 的扫描")
    LOG("注入方式：真实录制上把加载段按 cos 半波在 T 秒内拉到 (1−α)·电平；" 
        "同一条注入轨迹分别喂 raw / v5.1(e3s) / v6，比较三者响应。")
    LOG("=" * 122)
    rows = []
    for tag, t0 in CASES:
        d = C.prep(tag)
        tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
        ds = C.zbar(d["tot"], dtm)
        i0 = int(round(t0 / dtm))
        plat = float(np.median(ds[i0 - int(2.0 / dtm):i0]))
        gaps, eps, floor, peak = C.parse_plateaus(ds, tu, dtm)
        gp = [g for g in gaps if g["i1"] <= i0]
        zero = C.zero_level(ds, gp[-1], dtm, 0.5, 3.0)[0] if gp else float(np.percentile(ds, 5))
        # 逐通道占比：受载通道 = (加载段电平 − 空载段电平) 的正部
        if gp:
            g = gp[-1]
            Xz = np.median(Xu[g["i0"]:g["i1"]], axis=0)
        else:
            Xz = np.median(Xu[:int(3 / dtm)], axis=0)
        Xp = np.median(Xu[i0 - int(2.0 / dtm):i0], axis=0)
        wch = np.clip(Xp - Xz, 0.0, None)
        share = wch / wch.sum() if wch.sum() > 1e-12 else np.ones(Xu.shape[1]) / Xu.shape[1]
        noise = float(1.4826 * np.median(np.abs(ds[:int(3 / dtm)] - np.median(ds[:int(3 / dtm)]))))
        edges, _ = C.detect_edges(tu, d["tot"], dtm, ds=ds, min_frac=0.10)
        nxt = [x["t_cand"] for x in edges if x["t_cand"] > t0 + 2.0]
        t_revert = min(nxt) if nxt else float("nan")
        for T in EDGES:
            for al in ALPHAS:
                Xj = inject(Xu, tu, t0, T, al, (1.0 - al) * (plat - zero), share, t_revert)
                amp = al * (plat - zero)
                if amp <= 1e-9:
                    continue
                tot_j = Xj.sum(axis=1)
                dsj = C.zbar(tot_j, dtm)
                tol = max(3.0 * noise, 0.005 * abs(amp))
                j8 = min(len(tu), i0 + int(8.0 / dtm))
                min_raw = float(np.min(dsj[i0:j8]))
                pin_raw = float(np.sum(dsj[i0:j8] <= tol) * dtm)
                # 注入后的"新电平"（理论值）与稳态实测（窗口须落在下一次真实变载之前）
                la, lb = i0 + int(6.0 / dtm), i0 + int(14.0 / dtm)
                if np.isfinite(t_revert):
                    lb = min(lb, int(round((t_revert - 1.0) / dtm)))
                late_a, late_b = la, min(len(tu), lb)
                if late_b - late_a < int(1.0 / dtm):
                    late_a, late_b = i0 + int(3.0 / dtm), max(i0 + int(4.0 / dtm), late_b)
                st_raw = float(np.median(dsj[late_a:late_b])) if late_b > late_a else np.nan
                newlvl = plat - amp
                rec = dict(rec=tag, t0=t0, edge_s=T, alpha=al, amp=amp, plat=plat, zero=zero,
                           new_level=newlvl, tol=tol, injected=True,
                           raw_min=min_raw, raw_pin=pin_raw,
                           raw_dip_post=newlvl - min_raw,
                           raw_resid_late=(st_raw - newlvl) if np.isfinite(st_raw) else np.nan)
                for arm in ("e3s", "v6"):
                    Y = run(arm, tu, Xj)
                    Yb = C.med_smooth(Y, int(round(0.5 / dtm)))
                    min_a = float(np.min(Yb[i0:j8]))
                    pin_a = float(np.sum(Yb[i0:j8] <= tol) * dtm)
                    st_a = float(np.median(Yb[late_a:late_b])) if late_b > late_a else np.nan
                    rec.update({f"{arm}_min": min_a, f"{arm}_pin": pin_a,
                                f"{arm}_dip_post": newlvl - min_a,
                                f"{arm}_dip_extra": float(np.min(dsj[i0:j8])) - min_a,
                                f"{arm}_dip_extra_pct": (float(np.min(dsj[i0:j8])) - min_a) / abs(amp),
                                f"{arm}_extra_pin": pin_a - pin_raw,
                                f"{arm}_resid_late": (st_a - newlvl) if np.isfinite(st_a) else np.nan,
                                f"{arm}_resid_late_pct": ((st_a - newlvl) / abs(amp)
                                                          if np.isfinite(st_a) else np.nan)})
                rows.append(rec)
                LOG(f"  {tag:<16} T={T*1000:>4.0f}ms α={al:.1f}  额外下冲 e3s "
                    f"{rec['e3s_dip_extra_pct']*100:>+6.2f}% / v6 {rec['v6_dip_extra_pct']*100:>+6.2f}%"
                    f"   额外贴零 e3s {rec['e3s_extra_pin']:>+5.2f}s / v6 {rec['v6_extra_pin']:>+5.2f}s"
                    f"   稳态偏差 e3s {rec['e3s_resid_late']:>+9.3f} / v6 {rec['v6_resid_late']:>+9.3f}")
    df = pd.DataFrame(rows)
    C.save(df, "t7_inject_amp.csv")
    LOG("")
    LOG("自洽性核对：α=1.0 的行注入量恒 0（等于原信号），应与 step3 的真实事件结论一致。")
    LOG("按卸载幅度 α 聚合（同 T 内取中位；n=2 录制 × 2 沿时长）：")
    for T in EDGES:
        for al in ALPHAS:
            s = df[(df["edge_s"] == T) & (df["alpha"] == al)]
            LOG(f"  T={T*1000:>4.0f}ms α={al:.1f}  n={len(s)}  额外下冲中位 e3s "
                f"{np.median(s['e3s_dip_extra_pct'])*100:>+6.2f}% / v6 "
                f"{np.median(s['v6_dip_extra_pct'])*100:>+6.2f}%   额外贴零中位 e3s "
                f"{np.median(s['e3s_extra_pin']):>+5.2f}s / v6 {np.median(s['v6_extra_pin']):>+5.2f}s")
    LOG("")
    LOG("沿速度影响（同 α 内 T=20ms vs 300ms 的中位差）：")
    for al in ALPHAS:
        a = df[(df["edge_s"] == 0.02) & (df["alpha"] == al)]
        b = df[(df["edge_s"] == 0.30) & (df["alpha"] == al)]
        LOG(f"  α={al:.1f}  e3s 额外下冲 {np.median(a['e3s_dip_extra_pct'])*100:>+6.2f}% → "
            f"{np.median(b['e3s_dip_extra_pct'])*100:>+6.2f}% ；额外贴零 "
            f"{np.median(a['e3s_extra_pin']):>+5.2f}s → {np.median(b['e3s_extra_pin']):>+5.2f}s")
    LOG.close("python scripts/t7a_6_inject.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
