import os
import subprocess
import sys


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def build_replay_animator_command(csv_path: str, k_sigma: float, decay: float, fps: int,
                                  mode: int = 0, temporal_window: int = 3,
                                  spatial_sigma: float = 1.5) -> list[str]:
    script_path = os.path.join(BASE_DIR, "tools", "replay_animator.py")
    cmd = [
        sys.executable,
        script_path,
        csv_path,
        "--k-sigma", str(k_sigma),
        "--decay", str(decay),
        "--fps", str(fps),
        "--mode", str(mode),
        "--temporal-window", str(temporal_window),
        "--spatial-sigma", str(spatial_sigma),
    ]
    return cmd


def launch_replay_animator(csv_path: str, k_sigma: float, decay: float, fps: int,
                           mode: int = 0, temporal_window: int = 3,
                           spatial_sigma: float = 1.5):
    command = build_replay_animator_command(csv_path, k_sigma, decay, fps,
                                            mode, temporal_window, spatial_sigma)
    return subprocess.Popen(command, cwd=BASE_DIR)
