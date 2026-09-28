# -*- coding: utf-8 -*-
"""v2.0 验收补充（N3）· T9 案例对照：在 T9 的**现场失效通路**上复算本版修复。

背景：T9 那份录制在**新鲜回放**下几乎不产生补偿（偏移 ≈ 0），其现场失效来自
「录制之前补偿器已带状态进场」（`min_ts_` 被锁在低值 ⇒ `idle_now` 恒假 ⇒ never-idle）。
`t9g_seed.py` 用「首帧之后把 `min_ts` 预置为低值」复现该通路。

本脚本在 T9 数据集上跑四个臂（含 seeded 通路）：
  base            现役（plan-v1.0 A1+A4+A5a）
  base_seeded     现役 + seeded min_ts（= 现场失效通路）
  default_seed    本版默认（C 限幅 + F5）+ seeded
  legacy_seed     本版全开（另加 F2/F3）+ seeded
口径：`off = 算法输出总量 − 算法输入总量`（逐帧相减后再聚合）；空载段 = `pre < p10`。
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v20_lib as L  # noqa: E402

SPEC = importlib.util.spec_from_file_location("glm53_v6", L.PROTO_V6)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)

FAV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_ab_verify.py")
_fs = importlib.util.spec_from_file_location("fav", FAV_PATH)
FAV = importlib.util.module_from_spec(_fs)
_fs.loader.exec_module(FAV)


class SeededBase(P.GLM53v6):
    """现役参数 + seeded min_ts（复现 T9 的 never-idle 现场通路）。"""

    def __init__(self, n, seed):
        super().__init__(n)
        self.KAPPA_ONSET, self.KAPPA_RESTEP, self.HO_MIN = 1.05, 1.12, 3.5
        self.seed = seed

    def process(self, ts, v):
        out = super().process(ts, v)
        if self.first is False and self._seeded_done is False:
            self.min_ts = float(self.seed)
            self._seeded_done = True
        return out

    _seeded_done = False


def run_seeded(el, V, seed, clamp=None, legacy=False):
    """seeded 通路按 t9g_seed.py 的做法：首帧之后把 min_ts 预置为**带外低值**。

    注意：seed 必须**远低于**本机空载电平（T9 实测空载 ≈13 385），否则 `1.5·min_ts`
    仍高于空载电平、`idle_now` 照常成立，就复现不出 never-idle 通路
    （实测 seed=13385 → 2 个 epoch、偏移 0；seed=1900 → 19 个 epoch、偏移 ±470）。
    """
    ch = V.shape[1]
    if legacy or clamp is not None:
        c = FAV.Final(ch, clamp=clamp, legacy=legacy, g_protect=True)
    else:
        c = SeededBase(ch, seed)
    out = np.empty((len(el), ch))
    for i in range(len(el)):
        out[i] = c.process(float(el[i]), V[i])
        if i == 0:
            c.min_ts = float(seed)
    return out.sum(1)


def main():
    pre_s = L.load_stream(L.DS_T9, "device_001_pre_seg0.csv")
    el, V = pre_s["el"], pre_s["V"]
    pre = V.sum(1)
    main_s = L.load_stream(L.DS_T9, "device_001_seg000.csv")
    m_field = main_s["V"].sum(1)
    idle = pre < np.percentile(pre, 10)

    print("== T9 数据集(%d 帧 / %.1f s) ==" % (len(el), el[-1] - el[0]))
    print("现场现役 main：空载段 off 中位=%+.0f 最小=%+.0f 最大=%+.0f；末段 off 中位=%+.0f"
          % (np.median((m_field - pre)[idle]), (m_field - pre)[idle].min(),
             (m_field - pre)[idle].max(), np.median((m_field - pre)[-500:])))
    print()

    import design_full_regress as DFR  # 已由 final_ab_verify 载入同一实现
    fresh_base = FAV.DFR.run(el, V)

    seed = 1900.0   # 带外低值，复现 never-idle（见 t9g_seed.py）
    seeded_base = run_seeded(el, V, seed, clamp=None, legacy=False)
    seeded_def = run_seeded(el, V, seed, clamp=0.005, legacy=False)
    seeded_all = run_seeded(el, V, seed, clamp=0.005, legacy=True)

    rows = [("base 新鲜(不自带状态)", fresh_base),
            ("base seeded(现场通路)", seeded_base),
            ("default(C+F5) seeded", seeded_def),
            ("legacy_on 全开 seeded", seeded_all)]
    print("seed = %.0f（带外低值，复现『算法先开、后装夹/加载』的 never-idle 通路）" % seed)
    print("%-24s %14s %14s %12s %14s"
          % ("臂", "空载|off|中位", "空载|off|最大", "末段 off", "受载|off|中位"))
    for tag, o in rows:
        off = o - pre
        loaded = pre > np.percentile(pre, 60)
        print("%-24s %14.0f %14.0f %12.0f %14.0f"
              % (tag, np.median(np.abs(off[idle])), np.max(np.abs(off[idle])),
                 np.median(off[-500:]), np.median(off[loaded])))
    print()
    print("T9 报告登记的现场实测：空载偏差 −810 → +3070 ADC（≈ 1 个完整负载）")
    print("判定：本版在 seeded 通路上的『空载|off|最大』必须 ≤ 现役该值（不更差）。")


if __name__ == "__main__":
    main()
