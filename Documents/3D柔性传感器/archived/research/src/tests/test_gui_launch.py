import os
import sys
from unittest.mock import patch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from gui_launch import build_replay_animator_command, launch_replay_animator


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_build_replay_animator_command_preserves_spaced_paths_and_parameters():
    csv_path = r"D:\tmp\v1.0.10 - research\file.csv"
    command = build_replay_animator_command(csv_path, k_sigma=1.5, decay=0.5, fps=12)

    assert_true(command[0] == sys.executable, f"expected current Python executable, got {command[0]}")
    assert_true(command[1].endswith(os.path.join("tools", "replay_animator.py")), command[1])
    assert_true(command[2] == csv_path, f"CSV path changed: {command[2]}")
    assert_true("--k-sigma" in command and "1.5" in command, command)
    assert_true("--decay" in command and "0.5" in command, command)
    assert_true("--fps" in command and "12" in command, command)


def test_launch_replay_animator_returns_immediately_after_starting_child_process():
    with patch("gui_launch.subprocess.Popen") as popen:
        proc = launch_replay_animator("dummy.csv", k_sigma=1.0, decay=0.0, fps=10)

    assert_true(proc is popen.return_value, "launch did not return Popen handle")
    assert_true(popen.call_count == 1, "expected exactly one child process")


if __name__ == "__main__":
    test_build_replay_animator_command_preserves_spaced_paths_and_parameters()
    test_launch_replay_animator_returns_immediately_after_starting_child_process()
    print("gui launch tests passed")
