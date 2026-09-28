# -*- coding: utf-8 -*-
"""v4 快相免责期算法的独立模块（不执行 r_fastphase.py 的脚本主体）。

GLM53v4 与 temp/v4.1flash/scripts/r_fastphase.py::GLM53v4 逐行一致（原样搬运）。
GLM53v4r 是「免责期同时覆盖负载内变载(restep)」的变体，对应
  Document/02-v4算法说明.md §3.3 与 Document/04-C++实现骨架.md §3.3 的写法；
  原型 GLM53v4 的 _restep 不经过 _begin，故其免责期实际只在空载→负载时启动。
"""
import numpy as np

from glm53_v3 import GLM53v3


class GLM53v4(GLM53v3):
    """在 v3 上加「快相免责期」：onset 后 FAST_S 秒内冻结补偿与状态估计。"""

    FAST_S = 5.0          # 免责期长度
    A_W0_V4 = 3.5         # 幅度采集窗起点（落在免责期末段）
    A_W1_V4 = 5.0         # 幅度采集窗终点 = 免责期终点

    def _begin(self, ts):
        super()._begin(ts)
        self.fast_done_ = False
        self.creep_started_ = False

    def process(self, ts, v):
        # 免责期内：完全冻结（不积分 g/γ、不更新 A、不累积），显示直通
        state = getattr(self, "fast_done_", True)
        u = (ts - self.onset_ts) if self.in_load else 0.0
        if self.in_load and not state and u < self.FAST_S:
            # 仍需要跑 v3 的状态机（事件检测/卸载判定），但把补偿冻结掉：
            # 用一个很短的 dt 让 EMA 正常演化，同时把 a_captured_ 置 False 并清空累加
            self.a_captured = False
            self.a_acc = np.zeros(self.n)
            self.a_frames = 0
            self.g = 0.0
            self.g2 = 0.0
            self.g_rel = np.zeros(self.n)
            self.gamma = np.ones(self.n)
            # 在免责期末段采集幅度
            if self.A_W0_V4 <= u <= self.A_W1_V4:
                self.a_acc = self.a_acc + (v - self.b)
                self.a_frames += 1
            return v - self.b          # 直通（仅扣零漂基线）
        if self.in_load and not state and u >= self.FAST_S:
            self.fast_done_ = True
            self.creep_started_ = True
            self.a_captured = True
            if self.a_frames > 0:
                self.A = self.a_acc / self.a_frames
                amax = self.A.max()
                self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                    else np.zeros(self.n, bool)
            else:
                self.A = v - self.b
                amax = self.A.max()
                self.loaded = (self.A > self.LOADED_FRAC * amax) if amax > 1e-9 \
                    else np.zeros(self.n, bool)
        return super().process(ts, v)


class GLM53v4r(GLM53v4):
    """变体：负载内变载（restep）同样重新起免责期（doc 02 §3.3 / doc 04 §3.3 的写法）。"""

    def _restep(self, ts, z_now):
        super()._restep(ts, z_now)
        self.fast_done_ = False
        self.a_captured = False
        self.a_acc = np.zeros(self.n)
        self.a_frames = 0
