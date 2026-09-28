# -*- coding: utf-8 -*-
"""GLM53 v7：快相免责期的正确实现（继承 v3，改动最小）。

前几版的坑（记录在案，避免重犯）：
  · v4/v5 让父类的 A 捕获窗（名义 onset+1~3s）留在免责期内，
    窗口被冻结状态整段跳过 ⇒ 免责期结束后 `A` 永远捕获不到，
    结果 `A=0 → loaded 全 False → 补偿恒为 0`，算法悄悄退化成直通。
    （v5 在变化负载上"增益 1.000"就是这么来的假象。）
  · 因此 v7 把 A 窗**显式改到免责期末段**，并在免责期结束时立刻完成捕获。

设计：
  1. 免责期只在**首次 onset（空载→负载）**启用；负载内变载交给 v3 的 restep，不冻结；
  2. 免责期内冻结 g/γ/creep（输出只扣零漂基线 b），父类状态机照常运行；
  3. A 窗 = [FAST_S − A_WIN, FAST_S]；窗口结束即捕获 A、清零 g/γ 累加器，
     随后完全交给 v3 的原有蠕变逻辑。
"""
from glm53_v3 import GLM53v3
import numpy as np


class GLM53v7(GLM53v3):
    FAST_S = 5.0        # 免责期长度
    A_WIN = 1.0         # 幅度采集窗长度（紧贴免责期末端）

    def _begin(self, ts):
        super()._begin(ts)
        first = not getattr(self, "ever_loaded_", False)
        self.ever_loaded_ = True
        self.fast_on_ = first            # 仅首次 onset 冻结
        self.fast_done_ = not first

    def process(self, ts, v):
        if not self.in_load:
            return super().process(ts, v)
        u = ts - self.onset_ts
        if self.fast_on_ and not self.fast_done_:
            if u < self.FAST_S:
                Z = v - self.b
                # 冻结：清空一切蠕变状态，保证免责期输出 = Z
                self.a_captured = False
                self.g = 0.0
                self.g2 = 0.0
                self.g_rel = np.zeros(self.n)
                self.gamma = np.ones(self.n)
                # 幅度采集窗紧贴免责期末端
                if u >= self.FAST_S - self.A_WIN:
                    self.a_acc = self.a_acc + Z
                    self.a_frames += 1
                return Z
            # ---- 免责期结束：立刻捕获 A 并交回父类 ----
            self.fast_done_ = True
            if self.a_frames > 0:
                A = self.a_acc / self.a_frames
            else:
                A = v - self.b
            self.A = A.copy()
            amax = A.max()
            self.loaded = (A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                else np.zeros(self.n, bool)
            self.a_captured = True
            self.g = 0.0
            self.g2 = 0.0
            self.g_rel = np.zeros(self.n)
            self.gamma = np.ones(self.n)
        return super().process(ts, v)
