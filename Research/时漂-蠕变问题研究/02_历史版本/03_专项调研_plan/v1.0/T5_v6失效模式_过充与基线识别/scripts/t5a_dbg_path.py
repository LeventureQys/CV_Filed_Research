# -*- coding: utf-8 -*-
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t5a_common as C

print("TEMP =", repr(C.TEMP))
print("REC_PATH[RT1] =", repr(C.REC_PATH["RT1"]))
print("exists:", os.path.exists(C.REC_PATH["RT1"]))

import t5a_ad_lib as AD
_orig = AD.load_rec


def spy(p):
    print("  -> load_rec got:", repr(p))
    return _orig(p)


# 替换 t5a_a_common 名字空间里绑定的 load_rec
import t5a_a_common as AC
AC.load_rec = spy
try:
    d = C.recordings()
    print("OK recordings:", {k: len(v["tu"]) for k, v in d.items()})
except Exception as e:
    import traceback
    traceback.print_exc()
