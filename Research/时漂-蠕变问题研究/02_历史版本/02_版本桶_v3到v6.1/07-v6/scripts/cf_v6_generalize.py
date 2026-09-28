# -*- coding: utf-8 -*-
"""v6 泛化性实验（只读）：
  F1 形状库跨数据集泛化：用 9 组恒载 onset 的形状库反演 4 份实录的 onset；
  F2 跨位置泛化：留一位置（右/左/四指）out；
  F3 分类是否必要：用 onset 形状库反演 restep 的增量，与 restep 自己的形状库对比。
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
RES = os.path.join(os.path.dirname(HERE), "results")
TAU_REF = 0.20
TMAXFIT = 1.00

CL = [("右拇指_1", "右拇指", r"temp\右拇指指尖\数据1\device_001_seg000.csv"),
      ("右拇指_2", "右拇指", r"temp\右拇指指尖\数据2\device_001_seg000.csv"),
      ("右拇指_3", "右拇指", r"temp\右拇指指尖\数据3\device_001_seg000.csv"),
      ("左拇指_1", "左拇指", r"temp\左拇指指尖\数据1\device_001_seg000.csv"),
      ("左拇指_2", "左拇指", r"temp\左拇指指尖\数据2\device_001_seg000.csv"),
      ("左拇指_3", "左拇指", r"temp\左拇指指尖\数据3\device_001_seg000.csv"),
      ("四指_1", "四指", r"temp\四指指尖\数据1\device_001_seg000.csv"),
      ("四指_2", "四指", r"temp\四指指尖\数据2\device_001_seg000.csv"),
      ("四指_3", "四指", r"temp\四指指尖\数据3\device_001_seg000.csv")]
VL = [("切换负载", "变化", r"temp\变化负载\切换负载-快相无责的测试\20260917_133923_single_device_ee20bc\device_001_seg000.csv"),
      ("再切换", "变化", r"temp\变化负载\零负载-切换负载-零负载-再切换负载\device_001_seg000.csv"),
      ("中途1d9493", "变化", r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\device_001_seg000.csv"),
      ("中途13ffca", "变化", r"temp\变化负载\零负载-中途切换负载-零负载-切换负载\最终测试目标\device_001_seg000.csv")]


def load_frames(path):
    df = pd.read_csv(path, skiprows=24)
    ch = [c for c in df.columns if c.startswith("ch")]
    t = df["timestamp"].to_numpy(float)
    X = df[ch].to_numpy(float)
    t = t - t[0]
    grp = np.concatenate([[0], np.cumsum(np.diff(t) >= 1e-3)])
    n = grp[-1] + 1
    cnt = np.bincount(grp, minlength=n)
    Xp = np.column_stack([np.bincount(grp, weights=X[:, c], minlength=n) / cnt
                          for c in range(X.shape[1])])
    tp = np.bincount(grp, weights=t, minlength=n) / cnt
    return tp, Xp, float(np.median(np.diff(tp)))


def events(tot, dt):
    sm = pd.Series(tot).rolling(3, center=True, min_periods=1).median().to_numpy()
    w = max(2, int(round(0.08 / dt)))
    slope = np.zeros_like(sm)
    slope[w:] = (sm[w:] - sm[:-w]) / (w * dt)
    med = np.median(slope)
    sig = 1.4826 * np.median(np.abs(slope - med)) + 1e-12
    cand = np.where(np.abs(slope - med) > 10 * sig)[0]
    if not len(cand):
        return []
    groups, cur = [], [cand[0]]
    for c in cand[1:]:
        if (c - cur[-1]) * dt <= 2.0:
            cur.append(c)
        else:
            groups.append(cur)
            cur = [c]
    groups.append(cur)
    out = []
    for g in groups:
        pk = g[int(np.argmax(np.abs(slope[g])))]
        if pk < int(1.5 / dt) or pk > len(tot) - int(7.0 / dt):
            continue
        pre = float(np.median(sm[pk - int(1.5 / dt):pk - int(0.3 / dt)]))
        post = float(np.median(sm[pk + int(4.5 / dt):pk + int(5.5 / dt)]))
        jump = post - pre
        if abs(jump) < 1e-9:
            continue
        t0 = max(pk - int(0.4 / dt), 1)
        for i in range(pk, max(pk - int(0.4 / dt), 1), -1):
            if abs(sm[i] - pre) > 0.04 * abs(jump):
                t0 = i - 1
            else:
                break
        out.append((max(t0, 1), jump, pre, post))
    return out


def build(recs):
    evs = []
    for name, pos, rel in recs:
        tp, Xp, dt = load_frames(os.path.join(ROOT, rel))
        tot = Xp.sum(axis=1)
        peak = float(np.percentile(tot, 99.5))
        ts = [tp[e[0]] for e in events(tot, dt)]
        for i0, jump, pre, post in events(tot, dt):
            if jump <= 0.02 * peak:
                continue
            if any(0 < abs(x - tp[i0]) <= (3.0 if x < tp[i0] else 8.0) for x in ts if x != tp[i0]):
                continue
            kind = "onset" if pre < 0.15 * peak else "restep"
            grid = np.arange(0.0, 5.0 + 1e-9, dt)
            z = np.array([float(tot[min(len(tot) - 1, i0 + int(round(g / dt)))]) for g in grid])
            evs.append(dict(ds=name, pos=pos, kind=kind, dt=dt, grid=grid,
                            z=z, pre=pre, post=post, jump=jump, t=float(tp[i0])))
    return evs


CG = np.arange(0.0, 5.0 + 1e-9, 1.0 / 60.0)


def tail_shape(e, ref=TAU_REF):
    """事件 e 的尾巴形状（公共网格 CG，相对 Z(ref)，末端=1）。"""
    g, z = e["grid"], e["z"]
    zr = float(np.interp(ref, g, z))
    z5 = float(np.interp(5.0, g, z))
    if abs(z5 - zr) < 1e-12:
        return None
    return (np.interp(CG, g, z) - zr) / (z5 - zr)


def shape_lib(evs, ref=TAU_REF):
    fs = [s for s in (tail_shape(e, ref) for e in evs) if s is not None]
    if not fs:
        return CG, np.ones_like(CG)
    return CG, np.median(np.vstack(fs), axis=0)


def inv_err(evs, lib, taus, ref=TAU_REF, win=0.6):
    """用形状库 lib（公共网格 CG 上的尾巴形状，末端=1）反演各事件 5s 电平。"""
    out = {}
    for td in taus:
        errs = []
        for e in evs:
            g, z = e["grid"], e["z"]
            zr = float(np.interp(ref, g, z))
            z5 = float(np.interp(5.0, g, z))
            mg = (g >= td) & (g <= td + win)
            if not mg.any():
                continue
            f = np.interp(g[mg], lib[0], lib[1])
            y = z[mg] - zr
            if (f * f).sum() <= 0:
                continue
            Ah = zr + float((y * f).sum() / (f * f).sum())
            errs.append(100 * (Ah / z5 - 1))
        out[td] = np.array(errs) if errs else np.array([np.nan])
    return out


def show(tag, res):
    print(f"{tag:<34}" + "".join(
        f"{np.median(np.abs(res[t])):>7.1f}/{np.max(np.abs(res[t])):>6.1f}" for t in res))


TAUS = [0.30, 0.50, 0.75, 1.00, 1.50]
cl = build(CL)
vl = build(VL)
on_cl = [e for e in cl if e["kind"] == "onset"]
on_vl = [e for e in vl if e["kind"] == "onset"]
re_vl = [e for e in vl if e["kind"] == "restep"]
re_cl = [e for e in cl if e["kind"] == "restep"]

print("=" * 108)
print("样本：恒载 onset %d，实录 onset %d，恒载 restep %d，实录 restep %d"
      % (len(on_cl), len(on_vl), len(re_cl), len(re_vl)))
print("口径：Â = Z(0.2s) + Σ_{τ∈[τ_d,τ_d+0.6]}(Z(τ)−Z(0.2s))·f(τ)/Σf(τ)²，误差 = Â/Z(5s)−1 (%)")
print("格式：中位|误差| / 最大|误差|")
print("=" * 108)
print(f"{'实验':<34}" + "".join(f"{t:>14.2f}" for t in TAUS))

# F1 交叉数据集
lib_cl = shape_lib(on_cl)
lib_vl = shape_lib(on_vl)
show("F1a 恒载库→恒载 onset(留一)", inv_err(on_cl, shape_lib(on_cl), TAUS))
show("F1b 恒载库→实录 onset(跨域)", inv_err(on_vl, lib_cl, TAUS))
show("F1c 实录库→实录 onset(自域)", inv_err(on_vl, lib_vl, TAUS))

# F2 跨位置
print()
for pos in ("右拇指", "左拇指", "四指"):
    tr = [e for e in on_cl if e["pos"] != pos]
    te = [e for e in on_cl if e["pos"] == pos]
    show(f"F2 留出{pos}（训练其余 {len(tr)} 组→测试 {len(te)} 组）",
         inv_err(te, shape_lib(tr), TAUS))

# F3 分类必要性
print()
lib_on_all = shape_lib(on_cl + on_vl)
show("F3a onset 库→onset（同域基准）", inv_err(on_cl + on_vl, lib_on_all, TAUS))
show("F3b onset 库→restep（跨工况）", inv_err(re_cl + re_vl, lib_on_all, TAUS))
show("F3c restep 库→restep（同工况）", inv_err(re_cl + re_vl, shape_lib(re_cl + re_vl), TAUS))

print()
print("形状库（尾巴，相对 Z(0.2s)，末端=1）：")
print("τ(s)     :" + "".join(f"{g:>8.2f}" for g in lib_cl[0][::10]))
print("onset(恒):" + "".join(f"{v:>8.3f}" for v in lib_cl[1][::10]))
print("onset(实):" + "".join(f"{v:>8.3f}" for v in lib_vl[1][::10]))
if len(re_cl) + len(re_vl) > 1:
    lr = shape_lib(re_cl + re_vl)[1]
    print("restep   :" + "".join(f"{v:>8.3f}" for v in lr[::10]))
