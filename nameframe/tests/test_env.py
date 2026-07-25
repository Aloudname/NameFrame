from nameframe.utils.env import capture_env


class TestCaptureEnv:
    def test_returns_dict_with_required_keys(self):
        env = capture_env()
        for key in ["python", "torch", "cuda", "cudnn", "platform"]:
            assert key in env

    def test_values_are_strings(self):
        env = capture_env()
        for v in env.values():
            assert isinstance(v, str)

    def test_python_is_version_like(self):
        env = capture_env()
        parts = env["python"].split(".")
        assert len(parts) >= 2
