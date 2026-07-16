import os
import sys
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from gui_launch import launch_replay_animator


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "DataSet"))
    csv_path = os.path.join(data_dir, "手套", "单手指1kg", "录制数据_20260710141208_part0.csv")
    t0 = time.perf_counter()
    proc = launch_replay_animator(csv_path, k_sigma=1.0, decay=0.0, fps=10)
    elapsed = time.perf_counter() - t0
    print(f"launch_return_s={elapsed:.3f} pid={proc.pid}")
    if elapsed > 0.5:
        proc.terminate()
        raise AssertionError("launch_replay_animator blocked too long")
    time.sleep(1.0)
    running = proc.poll() is None
    print(f"child_running={running}")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    if not running:
        raise AssertionError("child animator process exited immediately")
    print("nonblocking launch acceptance passed")
