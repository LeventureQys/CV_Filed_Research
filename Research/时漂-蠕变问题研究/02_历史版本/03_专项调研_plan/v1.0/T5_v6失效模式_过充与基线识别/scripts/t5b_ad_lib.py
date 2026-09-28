# -*- coding: utf-8 -*-
"""T5-B 数据层（最小自包含）。

来源与血缘
----------
由 `temp/v4.1flash/progress/07-v6/scripts/ad_lib.py` **复制并改写**（复制件，原件未改、不被 import）：
  * `load_rec` 改为**自动定位 `##Data` 标记**（原实现写死 `skiprows=24`，
    违反 `00-共享/数据与脚本复用清单.md` §2「新脚本请用自动定位版本」）；
  * **删除**模块级 `from glm53_v3 import ... / from ad_v4 import ...` 与 `ALGOS/LBL/ORDER`
    —— 那些依赖 v3/v4 原型，本任务不需要，且会让 `import t5b_ad_lib` 在只放了必要原型的
    `scripts/` 目录里失败（复制整条依赖链的成本大于收益）；
  * 保留 `med_smooth`（与 `t1_common.py` 逐字一致：`pandas.rolling(center=True).median()`）；
  * 保留 `find_periods`、`detect_events`、`mid_load_events` 原样（口径未动）。

新增（T5-B 专用）
    `find_transients`  —— 找「快速偏移后 ≤`rec_s` 秒回到原电平」的实录瞬态（真实拍击类候选）
    `channel_activity` —— 空间模式特征（活跃通道数 / 集中度）
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def load_rec(p):
    """自动定位 `##Data` 行，返回 (t, X)；t 取 `timestamp` 列并平移到 0 起（禁用 `elapsed`）。"""
    hdr = None
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.startswith("##Data"):
                hdr = i + 1
                break
    if hdr is None:
        raise RuntimeError("no ##Data marker in " + p)
    df = pd.read_csv(p, skiprows=hdr)
    ch = [c for c in df.columns if c.startswith("ch")]
    tr = df["timestamp"].to_numpy(float)
    return tr - tr[0], df[ch].to_numpy(float)


def med_smooth(x, k):
    return pd.Series(x).rolling(max(1, int(k)), center=True, min_periods=1).median().to_numpy()


def find_periods(tot, dt, frac=0.05, min_s=3.0):
    thr = frac * tot.max()
    ld = tot > thr
    d = np.diff(ld.astype(int))
    s = list(np.where(d == 1)[0] + 1)
    e = list(np.where(d == -1)[0] + 1)
    if ld[0]:
        s = [0] + s
    if ld[-1]:
        e = e + [len(ld)]
    return sorted([(a, b) for a, b in zip(s, e) if (b - a) * dt >= min_s])


def detect_events(tot, dt, rel=0.15, absfrac=0.08):
    n = len(tot)
    pn = int(2.0 / dt)
    cand = []
    for i in range(pn, n - pn, max(1, int(0.1 / dt))):
        pre = np.median(tot[i - pn:i])
        post = np.median(tot[i:i + pn])
        if abs(post - pre) > max(rel * abs(pre), absfrac * tot.max()):
            cand.append((i, post - pre))
    ev = []
    for i, dl in cand:
        if ev and i - ev[-1][0] <= int(1.5 / dt):
            if abs(dl) > abs(ev[-1][1]):
                ev[-1] = (i, dl)
        else:
            ev.append((i, dl))
    ref, seen = [], set()
    for i, dl in ev:
        a, b = max(0, i - int(2 / dt)), min(n - 1, i + int(2 / dt))
        sm = med_smooth(tot[a:b], max(3, int(0.15 / dt)))
        k = int(np.argmax(np.abs(np.diff(sm))))
        if a + k in seen:
            continue
        seen.add(a + k)
        ref.append((a + k, dl))
    return ref


def mid_load_events(ev):
    """只保留「负载内变载」：变载前后都在受载电平上（排除空载→负载与卸载）。"""
    if not len(ev):
        return ev
    thr = 0.30 * ev.pre.max()
    return ev[(ev.pre > thr) & (ev.post > thr)].copy()


# ───────────────────────── T5-B 新增 ─────────────────────────

def find_transients(tot, dt, up_frac=0.03, rec_s=0.60, pre_s=0.30, tail_s=0.60,
                    min_peak=None, dedup_s=1.0):
    """实录「瞬态」候选：在 `pre_s` 内快速偏移、并在 `rec_s` 内回到原电平的段。

    判据（写死，报告 §2 引用）：
        peak = max_t |Z̄(t) − Z̄(t−pre_s)|              （0.3 s 中值平滑后）
        back = |Z̄(t+rec_s) − Z̄(t−pre_s)|              （回流残差）
        命中条件： peak > max(up_frac·Z̄_pre, min_peak) 且 back ≤ 0.35·peak
    这只是「**瞬态候选**」，不等于已确认的人手拍击（见报告未验证项）。
    返回 [(i_peak, peak_signed, back, t_pre_level), ...]
    """
    Z = med_smooth(np.asarray(tot, float), max(3, int(round(0.30 / dt))))
    n = len(Z)
    np_pre = max(1, int(round(pre_s / dt)))
    np_rec = max(1, int(round(rec_s / dt)))
    np_tail = max(1, int(round(tail_s / dt)))
    base = Z - np.concatenate([np.full(np_pre, Z[0]), Z[:-np_pre]])
    cand = []
    for i in range(np_pre, n - np_rec - np_tail):
        pre = Z[i - np_pre]
        p = abs(Z[i] - pre)
        thr = max(up_frac * abs(pre), 0.0 if min_peak is None else min_peak)
        if p <= thr:
            continue
        back = abs(Z[i + np_rec] - pre)
        if back <= 0.35 * p:
            cand.append((i, float(Z[i] - pre), float(back), float(pre)))
    out = []
    for c in sorted(cand, key=lambda r: -abs(r[1])):
        if all(abs(c[0] - o[0]) * dt >= dedup_s for o in out):
            out.append(c)
    return sorted(out)


def channel_activity(dv, frac=0.10, floor=1.0):
    """由「逐通道变化向量」算空间模式特征。

    dv        : (n_ch,) 逐通道电平变化（可为负）
    frac      : 判定"活跃通道"的相对门限（相对 max|dv|）
    floor     : 绝对下限（ADC），低于它不算活跃（避免纯噪声被算成活跃）

    返回 dict(n_ch, n_active, active_frac, hhi, participation, coup, jsd_like)
        n_active     : |dv_c| > max(frac·max|dv|, floor) 的通道数
        active_frac  : n_active / n_ch
        hhi          : Σ p_c² （p 为 |dv| 归一化份额）；1 = 单通道独占，1/n_ch = 完全均匀
        participation: (Σp)²/Σp² = 1/hhi
        coup         : Σ|dv| / max|dv|  （等效"活跃通道数"的 L1 版本，对少量大通道敏感）
    """
    dv = np.asarray(dv, float)
    ad = np.abs(dv)
    mx = float(ad.max()) if len(ad) else 0.0
    if mx <= 1e-12:
        return dict(n_ch=len(dv), n_active=0, active_frac=0.0, hhi=float("nan"),
                    participation=float("nan"), coup=0.0)
    thr = max(frac * mx, floor)
    act = ad > thr
    p = ad / ad.sum()
    hhi = float((p ** 2).sum())
    return dict(n_ch=len(dv), n_active=int(act.sum()),
                active_frac=float(act.sum()) / len(dv), hhi=hhi,
                participation=1.0 / hhi if hhi > 1e-12 else float("nan"),
                coup=float(ad.sum() / mx))
