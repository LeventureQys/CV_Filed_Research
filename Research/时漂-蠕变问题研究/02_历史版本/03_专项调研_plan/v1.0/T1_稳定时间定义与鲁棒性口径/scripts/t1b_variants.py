# -*- coding: utf-8 -*-
"""T1-B：v6 的**可开关改造变体**（只用于离线消融，不改任何原型文件）。

变体都在 `t1b_glm53_v6.py`（原型的逐行复制件）上子类化，**只覆写 `process` 的两个门限旋钮**：

  DWELL_S    驻留确认：把原型"连续 3 帧(≈30 ms)"换成"判据连续保持 DWELL_S 秒"
             （直接换算成 `DET_PERSIST = round(DWELL_S/dt)`，因为 dt 恒为 10 ms）
  DET_REL / DET_K / REVOKE / DET_IDLE_FRAC  直接作为实例属性下传（原型本来就支持）
  CAPF       `min(5σ_d, CAPF·DET_REL·|lv_ref|)`：把 σ 项用"相对电平门限的 CAPF 倍"封顶
             （第一轮 `a6_fix2.py` 的 CAPF=1.0 就是本族的一个点，已被判为有害）
  RESCUE_S   原型**没有**恢复机制：一旦 `5σ_d` 反超电平门限，真实沿被永久漏掉。
             本变体实现"救援"：当 `|d|` 已过电平门限但被 σ 项挡住、且**同向持续**
             `RESCUE_S` 秒时，临时放开 σ 项（等价于按电平门限判定）。原型建议的
             "持续 ≥0.4 s 且不反号"（`a_快速扰动下基线识别分析.md` §4 建议 4）即此形态；
             **本实现是原型级近似，未做 C++ 落地**。
"""
import numpy as np

from t1b_glm53_v6 import GLM53v6


class V6Knob(GLM53v6):
    DWELL_S = 0.0            # 0 = 用原型 DET_PERSIST
    CAPF = 0.0               # 0 = 关
    RESCUE_S = 0.0           # 0 = 关
    _BASE_K = 5.0            # = GLM53v6.DET_K（原型值）

    def __init__(self, n):
        super().__init__(n)
        self.DET_PERSIST = self._dwell_frames()
        self._rescue_t = 0.0
        self._rescue_sign = 0
        self.n_rescue = 0

    def _dwell_frames(self):
        """DWELL_S>0 时把"驻留秒数"换算成原型的连续帧数；否则用原型值。"""
        return (max(1, int(round(self.DWELL_S / 0.01))) if self.DWELL_S > 0
                else type(self).DET_PERSIST)

    def process(self, ts, v):
        # DWELL_S 可能在 __init__ 之后才被 setattr 进来（run_traced/run_plain 的做法），
        # 所以每帧重新换算一次；同时把 DET_K 复位（CAPF/RESCUE 会改写它）。
        if self.DWELL_S > 0:
            self.DET_PERSIST = self._dwell_frames()
        self.DET_K = self._BASE_K
        if self.RESCUE_S > 0.0:
            lv_ref = self._win_mean(ts - self.DET_FAST - self.DET_GAP - self.DET_LAG,
                                    ts - self.DET_FAST - self.DET_GAP)
            lv_now = self._win_mean(ts - self.DET_FAST, ts)
            if lv_ref is not None and lv_now is not None:
                d = lv_now - lv_ref
                sgn = float(np.sign(d))
                thr_level = max(self.DET_REL * abs(lv_ref), self.DET_ABS_FRAC * self.max_tot)
                # "电平门限够、但被 σ 项挡住" = 原型漏掉真实沿的充要条件
                blocked = (abs(d) > thr_level) and (abs(d) <= self._BASE_K * self.sig_d)
                if blocked and sgn == self._rescue_sign and abs(d) > 1e-9:
                    self._rescue_t += 0.01            # 同向持续计时
                else:
                    self._rescue_t = 0.0
                self._rescue_sign = sgn
            if self._rescue_t >= self.RESCUE_S:
                self.DET_K = 0.0                      # 放开 σ 项 ⇒ 退化为电平门限判定
                self._rescue_t = 0.0
                self.n_rescue += 1
        elif self.CAPF > 0.0:
            lv_ref = self._win_mean(ts - self.DET_FAST - self.DET_GAP - self.DET_LAG,
                                    ts - self.DET_FAST - self.DET_GAP)
            if lv_ref is not None and self.sig_d > 1e-12:
                cap = self.CAPF * self.DET_REL * abs(lv_ref) / self.sig_d
                self.DET_K = float(min(self._BASE_K, cap))
        return super().process(ts, v)


# 代价-收益矩阵的旋钮表：name -> (类, 额外 kwargs)
KNOBS = [
    ("base", V6Knob, {}),
    ("dwell0.10", V6Knob, dict(DWELL_S=0.10)),
    ("dwell0.15", V6Knob, dict(DWELL_S=0.15)),
    ("dwell0.25", V6Knob, dict(DWELL_S=0.25)),
    ("dwell0.40", V6Knob, dict(DWELL_S=0.40)),
    ("rel0.08", V6Knob, dict(DET_REL=0.08)),
    ("rel0.12", V6Knob, dict(DET_REL=0.12)),
    ("k7", V6Knob, dict(DET_K=7.0)),
    ("revoke0.80", V6Knob, dict(REVOKE=0.80)),
    ("idle0.25", V6Knob, dict(DET_IDLE_FRAC=0.25)),
    ("capf1.0", V6Knob, dict(CAPF=1.0)),
    ("capf2.0", V6Knob, dict(CAPF=2.0)),
    ("rescue0.40", V6Knob, dict(RESCUE_S=0.40)),
]
