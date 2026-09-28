# -*- coding: utf-8 -*-
"""「真实加载沿」上到底包含哪几段时延 —— 逐帧实测拆解。

对每个**真实加载沿**（上升沿）测量以下时刻（相对加载沿，s）：
    t_detect   首次 pending=1 的时刻      → 检测窗/判据延迟（电平判据需 0.3s 近窗 + 0.5s 滞后窗）
    t_epoch    epoch 起点（BeginLoad/Restep）→ 确认延迟 = t_epoch − t_detect（应 ≈ kStepPersistS=2.5s）
    t_ded_theo = t_epoch + FAST_S         → 免责期（含 A 采集窗，窗末即首扣）
    t_ded_meas 首个 |扣除| > 0.5%×本沿台阶的时刻 → 再叠加 g 从 0 起 τ=3s 的"爬升到可见"延迟

输出三档（1s/3s/5s）+ v3（现役参考）的对照表，以及 4 份录制上的均值。
产出：results/loadedge_latency.csv、results/_loadedge_latency.log（调用方 tee）
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
TEMP = os.path.dirname(os.path.dirname(os.path.dirname(OUT)))
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                     # noqa: E402
from glm53_v3 import GLM53v3                           # noqa: E402
from glm53_v51 import GLM53v51                         # noqa: E402

KS = ["1s", "3s", "5s"]
FASTOF = {"1s": 1.0, "3s": 3.0, "5s": 5.0}
CONFIRM = 2.5
TAU_G = 3.0
CFG = [(k, GLM53v51, dict(FAST_S=FASTOF[k], EXEMPT_AWIN=FASTOF[k] / 3.0, LEV_ARM_S=FASTOF[k]))
       for k in KS] + [("v3", GLM53v3, {})]

B = os.path.join(TEMP, "变化负载")
RECS = [("★最终测试目标(13ffca)", os.path.join(B, "零负载-中途切换负载-零负载-切换负载",
                                               "最终测试目标", "device_001_seg000.csv")),
        ("切换负载-快相无责", os.path.join(B, "切换负载-快相无责的测试",
                                           "20260917_133923_single_device_ee20bc", "device_001_seg000.csv")),
        ("再切换负载", os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("中途切换-1d9493", os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv"))]


def make_traced(cls):
    class _T(cls):
        def __init__(self, n):
            super().__init__(n)
            self.epoch_t = []
            self.st = []

        def process(self, ts, v):
            y = super().process(ts, v)
            self.st.append((float(ts), int(self.pending)))
            return y

        def _begin(self, ts):
            super()._begin(ts)
            self.epoch_t.append(float(ts))

        def _restep(self, ts, z_now):
            super()._restep(ts, z_now)
            self.epoch_t.append(float(ts))

    return _T


rows = []
for tag, path in RECS:
    if not os.path.exists(path):
        print(f"[skip] {path}")
        continue
    d = L.prep(path)
    tu, Xu, dtm = d["tu"], d["Xu"], d["dtm"]
    tot_s = L.med_smooth(d["tot"], 0.5 / dtm)
    ev = L.event_table(d, [e for e, _ in L.detect_events(d["tot"], dtm)], {"raw": Xu}, [], algos=[])
    edges = ev[ev.jump > 2000].copy()
    # fresh = 空载→负载（起扣前扣除本就为 0，才可能测"首个可见扣除"）
    edges["kind"] = np.where(edges["pre"] < 0.05 * float(d["tot"].max()), "空载→负载", "负载内变载")
    print("=" * 128)
    print(f"[{tag}] {d['span']:.1f}s / {len(tu)} 帧 / {Xu.shape[1]}ch / 真实加载沿 {len(edges)} 个"
          f"（空载→负载 {int((edges.kind == '空载→负载').sum())} 个）")
    print(f"{'加载沿':>8} {'台阶':>8} {'类型':>10} {'档位':>4} | {'t_detect':>8} {'确认(实测)':>10} "
          f"{'t_epoch':>8} {'免责':>5} {'首扣理论':>8} {'首扣实测':>8} {'爬升到可见':>10} "
          f"{'沿后6s捕获':>10}")
    print("-" * 128)
    traces = {}
    for k, cls, kw in CFG:
        c = make_traced(cls)(Xu.shape[1])
        for kk, vv in kw.items():
            setattr(c, kk, vv)
        Y = np.empty_like(Xu)
        for i in range(len(tu)):
            Y[i] = c.process(tu[i], Xu[i])
        traces[k] = (Y, c)
    for _, e in edges.iterrows():
        i0 = int(np.searchsorted(tu, e.t))
        base = float(np.median(tot_s[max(0, i0 - int(2 / dtm)):i0]))
        amp = float(abs(tot_s[min(len(tu) - 1, i0 + int(8 / dtm))] - base))
        j6 = min(len(tu) - 1, i0 + int(6 / dtm))
        fresh = e.kind == "空载→负载"
        for k, cls, kw in CFG:
            Y, c = traces[k]
            st = np.array(c.st)
            m = st[:, 0] >= e.t - 0.25
            det = st[m & (st[:, 1] > 0), 0]
            t_det = (det[0] - e.t) if len(det) else np.nan
            ep = [x for x in c.epoch_t if x >= e.t - 0.25]
            t_ep = (ep[0] - e.t) if ep else np.nan
            ded = (Xu - Y).sum(axis=1)
            seg = ded[max(0, i0):i0 + int(12 / dtm)]
            hit = np.where(seg > 0.005 * amp)[0]
            t_dm = float(hit[0] * dtm) if (len(hit) and fresh) else np.nan
            fast = FASTOF.get(k, np.nan)
            t_th = (t_ep + fast) if (ep and k in KS) else np.nan
            y = L.med_smooth(Y.sum(axis=1), 0.5 / dtm)
            cap6 = (y[j6] - y[i0]) / (tot_s[j6] - tot_s[i0]) if abs(tot_s[j6] - tot_s[i0]) > 1e-9 else np.nan
            rows.append(dict(rec=tag, t_edge=float(e.t), jump=float(e.jump), kind=e.kind,
                             algo=k, amp=amp, fresh=fresh,
                             t_detect=t_det, t_epoch=t_ep,
                             confirm=(t_ep - t_det) if (ep and len(det)) else np.nan,
                             exempt=fast, t_ded_theo=t_th, t_ded_meas=t_dm,
                             visible_extra=(t_dm - t_th) if (np.isfinite(t_dm) and np.isfinite(t_th)) else np.nan,
                             cap6=cap6))
    sub = pd.DataFrame(rows)
    sub = sub[sub.rec == tag]
    for _, e in edges.iterrows():
        for k, _, _ in CFG:
            r = sub[(sub.t_edge == e.t) & (sub.algo == k)].iloc[0]
            print(f"{e.t:8.2f} {e.jump:8.0f} {e.kind:>10} {k:>4} | {r.t_detect:8.2f} {r.confirm:10.2f} "
                  f"{r.t_epoch:8.2f} {r.exempt:5.1f} {r.t_ded_theo:8.2f} {r.t_ded_meas:8.2f} "
                  f"{r.visible_extra:10.2f} {r.cap6:10.2f}")
    print("-" * 128)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "loadedge_latency.csv"), index=False, encoding="utf-8-sig")

print("\n" + "=" * 122)
print("时延拆解汇总（4 份录制 · 21 个真实加载沿）")
print("=" * 122)
fs = df[df.fresh]
print(f"[A] 空载→负载沿（{len(fs) // 4} 个）：起扣前扣除为 0，四项时延都可测")
print(fs.groupby("algo").agg(
    t_detect=("t_detect", "mean"), confirm=("confirm", "mean"), exempt=("exempt", "mean"),
    t_epoch=("t_epoch", "mean"), t_ded_theo=("t_ded_theo", "mean"),
    t_ded_meas=("t_ded_meas", "mean"), visible_extra=("visible_extra", "mean"),
    cap6=("cap6", "mean"), n=("t_edge", "count")).reindex(KS + ["v3"]).round(2).to_string())
ms = df[~df.fresh]
print(f"\n[B] 负载内变载沿（{len(ms) // 4} 个）：扣除本来就非零，「首个可见扣除」口径无意义，只列检测/确认")
print(ms.groupby("algo").agg(
    t_detect=("t_detect", "mean"), confirm=("confirm", "mean"),
    t_epoch=("t_epoch", "mean"), t_ded_theo=("t_ded_theo", "mean"),
    cap6=("cap6", "mean"), n=("t_edge", "count")).reindex(KS + ["v3"]).round(2).to_string())
print(f"\n检测延迟的离散度（全部沿）：t_detect 中位 "
      f"{df.t_detect.median():.2f}s，最小 {df.t_detect.min():.2f}s，最大 {df.t_detect.max():.2f}s；"
      f"确认延迟实测 {df.confirm.min():.2f}~{df.confirm.max():.2f}s（常量 kStepPersistS=2.5s）")
print("\n分档（空载→负载口径）")
for k in KS:
    r = fs[fs.algo == k]
    print(f"  {k:>3}: 检测 {r.t_detect.mean():.2f} + 确认 {r.confirm.mean():.2f} + 免责 {r.exempt.mean():.1f}"
          f" = 首扣理论 {r.t_ded_theo.mean():.2f}s ；实测可见首扣 {r.t_ded_meas.mean():.2f}s"
          f"（再 +{r.visible_extra.mean():.2f}s）；扣到位95% ≈ {r.t_ded_theo.mean() + 3 * TAU_G:.1f}s")
print(f"\n（v3 参考：检测 {fs[fs.algo == 'v3'].t_detect.mean():.2f}s、确认 "
      f"{fs[fs.algo == 'v3'].confirm.mean():.2f}s、无免责期，实测可见首扣 "
      f"{fs[fs.algo == 'v3'].t_ded_meas.mean():.2f}s）")
