import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bg_v5_cpp_parity as P   # 复用其 cpp_run/py_run（会先跑一遍 main，无副作用）
