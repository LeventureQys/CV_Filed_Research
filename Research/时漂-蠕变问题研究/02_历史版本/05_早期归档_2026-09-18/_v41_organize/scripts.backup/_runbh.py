import os, sys, runpy
os.environ["FIG_CHECK"] = "1"
sys.argv = ["bh_v5_why_fig.py"]
# 给 bh 脚本注入 audit：直接检查渲染后的注释是否越界
runpy.run_path("scripts/bh_v5_why_fig.py", run_name="__main__")
