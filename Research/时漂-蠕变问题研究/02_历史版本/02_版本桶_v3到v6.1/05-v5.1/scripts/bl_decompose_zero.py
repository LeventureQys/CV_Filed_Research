# -*- coding: utf-8 -*-
"""分解「零点（零负载）显示的抑制量」来源：空载基线 b_ vs 蠕变扣除(carry/hold/γAg)。

对每份含零负载段的录制，逐帧记录：
  raw 总量、显示总量、b_ 总量、carry 总量、hold_comp 总量、γAg 总扣除、路径标记；
再按零负载段汇总：显示被压到 ≤0 的时长、各来源贡献、卸载沿后恢复（显示>0）耗时。
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import ad_lib as L                                                    # noqa: E402
from glm53_v3 import GLM53v3                                          # noqa: E402
from glm53_v5 import GLM53v5                                          # noqa: E402

B = r"D:\workshop\Processing\multi-device-cascade-host-cpp\temp\变化负载"
RECS = [("零负载-切换负载-零负载-再切换负载",
         os.path.join(B, "零负载-切换负载-零负载-再切换负载", "device_001_seg000.csv")),
        ("零负载-中途切换-零负载-切换负载",
         os.path.join(B, "零负载-中途切换负载-零负载-切换负载", "device_001_seg000.csv")),
        ("切换负载-快相无责",
         os.path.join(B, "切换负载-快相无责的测试", "20260917_133923_single_device_ee20bc",
                      "device_001_seg000.csv"))]


def trace(cls, tu, Xu):
    c = cls(Xu.shape[1])
    n = Xu.shape[1]
    rec = {k: np.zeros(len(tu)) for k in
           ("disp", "b", "carry", "holdc", "gag", "A", "g")}
    rec["in_load"] = np.zeros(len(tu), bool)
    rec["hold"] = np.zeros(len(tu), bool)
    rec["exempt"] = np.zeros(len(tu), bool)
    for i in range(len(tu)):
        Y = c.process(tu[i], Xu[i])
        rec["disp"][i] = Y.sum()
        rec["b"][i] = c.b.sum()
        rec["carry"][i] = getattr(c, "carry", np.zeros(n)).sum()
        rec["holdc"][i] = 0.0 if getattr(c, "hold_comp", None) is None else float(np.sum(c.hold_comp))
        m = c.loaded & (c.A > 1e-9)
        rec["gag"][i] = float(np.sum(np.clip(c.gamma[m] * c.A[m] * c.g,
                                             c.CREEP_LO * c.A[m], c.CREEP_HI * c.A[m]))) if m.any() else 0.0
        rec["A"][i] = float(c.A.max())
        rec["g"][i] = float(c.g)
        rec["in_load"][i] = c.in_load
        rec["hold"][i] = c.hold
        rec["exempt"][i] = c.in_load and (not getattr(c, "fast_done", True))
    return rec


for tag, path in RECS:
    d = L.prep(path)
    tu, Xu, dtm, tot = d["tu"], d["Xu"], d["dtm"], d["tot"]
    tot_s = L.med_smooth(tot, 0.5 / dtm)
    peak = float(tot_s.max())
    zero = tot_s < 0.05 * peak
    segs, i = [], 0
    while i < len(zero):
        if zero[i]:
            j = i
            while j + 1 < len(zero) and zero[j + 1]:
                j += 1
            if (j - i) * dtm >= 2.0:
                segs.append((i, j))
            i = j + 1
        else:
            i += 1
    print("=" * 112)
    print(f"[{tag}] 峰值={peak:.0f} dtm={dtm*1000:.2f}ms  零负载段={[(round(float(tu[a]),1),round(float(tu[b]),1)) for a,b in segs]}")
    R = {name: trace(cls, tu, Xu) for name, cls in (("v3", GLM53v3), ("v5", GLM53v5))}
    for a, b in segs:
        raw = tot[a:b + 1]
        print(f"  ── 零负载段 {tu[a]:.1f}~{tu[b]:.1f}s（原始均值 {raw.mean():.0f} ADC）")
        for name in ("v3", "v5"):
            r = R[name]
            disp = r["disp"][a:b + 1]
            ded = raw - disp                      # 总抑制量
            clamp = disp <= 1.0                   # 显示层(阈值 0)会把 ≤0 钳成 0
            print(f"     {name}: 显示均值={disp.mean():8.0f} 总抑制均值={ded.mean():8.0f} | "
                  f"b_均值={r['b'][a:b+1].mean():7.0f} carry均值={r['carry'][a:b+1].mean():7.0f} "
                  f"hold均值={r['holdc'][a:b+1].mean():7.0f} γAg均值={r['gag'][a:b+1].mean():7.0f} | "
                  f"被钳 0 时长={clamp.sum()*dtm:5.2f}s/{((b+1-a)*dtm):.2f}s ({100*clamp.mean():.0f}%) | "
                  f"in_load 帧占比={100*r['in_load'][a:b+1].mean():.0f}% "
                  f"免责期占比={100*r['exempt'][a:b+1].mean():.0f}% hold 占比={100*r['hold'][a:b+1].mean():.0f}%")
    # 卸载沿细看：沿后 4s 内逐 0.5s 显示均值
    load = ~zero
    trans = np.where((load[:-1]) & (zero[1:]))[0] + 1
    print("  卸载沿后 4s 的显示总量（0.5s 平均，raw 对照；负值在显示层会被钳成 0）：")
    for e in trans:
        row = [f"    t={tu[e]:6.1f}s"]
        for k in range(8):
            i0 = e + int(k * 0.5 / dtm)
            i1 = e + int((k + 1) * 0.5 / dtm)
            row.append(f"[{k*0.5:.1f}s] raw={tot[i0:i1].mean():6.0f} "
                       f"v3={R['v3']['disp'][i0:i1].mean():7.0f} v5={R['v5']['disp'][i0:i1].mean():7.0f}")
        print("\n".join(row[:1]) + " | " + " ".join(row[1:4]))
        print(" " * 12 + " | " + " ".join(row[4:]))
