# -*- coding: utf-8 -*-
"""步骤1（重跑）：分段修正后的数据特征刻画。"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tac_common import load_dataset, detect_segments, dump_json, DATASETS, RES

np.set_printoptions(precision=4, suppress=True, linewidth=140)
lines = []
out = {}


def P(s=""):
    print(s)
    lines.append(s)


for name in DATASETS:
    D = load_dataset(name)
    fi = D["frame_index"]
    k, b = np.polyfit(fi, D["t"], 1)
    t = fi * k; fs = 1.0 / k
    X = D["X"]
    total = X.sum(axis=1)
    seg = detect_segments(total, t, fs=fs)
    a, bb = seg["pre"]; c0, d0 = seg["load"]
    rl = seg["raw_load"]
    base = X[a:bb].mean(axis=0)
    Xn = X - base
    w = lambda s_: int(round(s_ * fs))
    resp = Xn[c0:d0].max(axis=0)
    act = np.where(resp > 0.05)[0]
    qui = np.where(np.abs(Xn[c0:d0]).max(axis=0) <= 0.05)[0]

    P("=" * 96)
    P(f"### {name}  fs={fs:.4f}Hz  N={D['n']}  总时长={t[-1]:.2f}s")
    P(f"    沿检测: 上升沿@{t[rl[0]]:.2f}s  下降沿@{t[rl[1]]:.2f}s  "
      f"(Σ: 基线={seg['base']:.1f}mN 峰值={seg['base']+seg['amp']:.1f}mN)")
    P(f"    分段(含 {0.5}s 保护带): 前空载 0~{t[bb]:.2f}s | "
      f"负载 {t[c0]:.2f}~{t[d0]:.2f}s ({t[d0]-t[c0]:.2f}s) | 后空载 {t[d0]:.2f}~{t[-1]:.2f}s")
    P(f"    受载通道 {len(act)} 个, 静默通道 {len(qui)} 个 "
      f"(死通道 {int((X.std(axis=0)==0).sum())} 个: "
      f"{[D['ch_cols'][i] for i in np.where(X.std(axis=0)==0)[0]]})")

    # 分段正确性验证：后空载段应回到 ~0
    post_mean = Xn[d0:].mean(axis=0)[act]
    pre_mean = Xn[a:bb].mean(axis=0)[act]
    P(f"    分段校验: 后空载均值(受载通道) 中位={np.median(post_mean)*1000:+.3f}mN "
      f"max|.|={np.abs(post_mean).max()*1000:.3f}mN  "
      f"(前空载应≈0: max|.|={np.abs(pre_mean).max()*1000:.3f}mN)")

    # 真值锚点
    Fref = Xn[c0 + w(1.0):c0 + w(3.0)].mean(axis=0)
    Fref_early = Xn[c0 + w(0.5):c0 + w(1.0)].mean(axis=0)
    P(f"    F_ref(1~3s) 受载通道: 中位={np.median(Fref[act])*1000:.1f}mN "
      f"max={Fref[act].max()*1000:.1f}mN")
    for j in act[np.argsort(-Fref[act])[:3]]:
        late = Xn[d0 - w(5):d0, j].mean()
        drift = (late - Fref[j]) / Fref[j] * 100
        mon = 0.0
        sref = Xn[c0 + w(1.0):c0 + w(3.0), j].mean()
        P(f"      {D['ch_cols'][j]:>5s}: F_ref={Fref[j]*1000:7.2f}mN  "
          f"末5s={late*1000:7.2f}mN  相对漂移={drift:+6.2f}%")
    drift_all = np.array([(Xn[d0 - w(5):d0, j].mean() - Fref[j]) / Fref[j] * 100
                          for j in act])
    P(f"    全受载通道相对漂移: 中位={np.median(drift_all):+.2f}%  "
      f"范围={drift_all.min():+.2f}%~{drift_all.max():+.2f}%")

    # 每个通道相对漂移 vs 灵敏度
    cc = np.corrcoef(Fref[act], drift_all)[0, 1]
    P(f"    相对漂移 与 F_ref 的空间相关 r={cc:+.3f}")

    # 卸载后残余
    z_end = Xn[d0 - w(5):d0].mean(axis=0)[act]
    z_post = Xn[-w(3):].mean(axis=0)[act]
    z_pre = Xn[a:bb].mean(axis=0)[act]
    P(f"    零漂: 前空载={np.median(z_pre)*1000:+.3f}mN  "
      f"卸载瞬间={np.median(z_end)*1000:+.3f}mN  "
      f"末尾={np.median(z_post)*1000:+.3f}mN")
    P(f"          残余/时漂 = {np.median(np.abs(z_post)/np.maximum(np.abs(z_end),1e-9))*100:.1f}%")

    out[name] = dict(fs=float(fs), dur=float(t[-1]), n=int(D["n"]),
                     seg={kk: [int(v[0]), int(v[1])] for kk, v in seg.items()
                          if kk in ("pre", "load", "post")},
                     raw_load=[int(rl[0]), int(rl[1])],
                     t_pre=float(t[bb]), t_load0=float(t[c0]), t_load1=float(t[d0]),
                     act=[int(i) for i in act], qui=[int(i) for i in qui],
                     Fref=Fref, drift_rel=drift_all,
                     zero_pre=z_pre, zero_post=z_post)

dump_json({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv)
               for kk, vv in v.items()} for k, v in out.items()},
          "A_characterization_fixed.json")
with open(os.path.join(RES, "A_characterization_fixed.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("\n[ok] 步骤1 重跑完成")
