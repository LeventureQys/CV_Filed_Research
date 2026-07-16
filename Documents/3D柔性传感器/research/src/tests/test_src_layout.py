import os


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BETA1_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def test_generated_python_code_lives_under_src():
    for name in ["alg", "tools", "tests", "stats"]:
        assert_true(os.path.isdir(os.path.join(SRC_DIR, name)), f"missing src/{name}")
        assert_true(not os.path.exists(os.path.join(BETA1_DIR, name)), f"unexpected beta1/{name}")
    assert_true(os.path.isfile(os.path.join(SRC_DIR, "main.py")), "missing src/main.py")


if __name__ == "__main__":
    test_generated_python_code_lives_under_src()
    print("src layout tests passed")
