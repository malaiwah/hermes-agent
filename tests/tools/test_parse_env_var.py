"""Tests for _parse_env_var and _get_env_config env-var validation."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import sys
import tools.terminal_tool  # noqa: F401 -- ensure module is loaded
_tt_mod = sys.modules["tools.terminal_tool"]
from tools.terminal_tool import _parse_env_var


class TestParseEnvVar:
    """Unit tests for _parse_env_var."""

    # -- valid values work normally --

    def test_valid_int(self):
        with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "300"}):
            assert _parse_env_var("TERMINAL_TIMEOUT", "180") == 300

    def test_valid_float(self):
        with patch.dict("os.environ", {"TERMINAL_CONTAINER_CPU": "2.5"}):
            assert _parse_env_var("TERMINAL_CONTAINER_CPU", "1", float, "number") == 2.5

    def test_valid_json(self):
        volumes = '["/host:/container"]'
        with patch.dict("os.environ", {"TERMINAL_DOCKER_VOLUMES": volumes}):
            result = _parse_env_var("TERMINAL_DOCKER_VOLUMES", "[]", json.loads, "valid JSON")
            assert result == ["/host:/container"]

    def test_get_env_config_parses_docker_forward_env_json(self):
        with patch.dict("os.environ", {
            "TERMINAL_ENV": "docker",
            "TERMINAL_DOCKER_FORWARD_ENV": '["GITHUB_TOKEN", "NPM_TOKEN"]',
        }, clear=False):
            config = _tt_mod._get_env_config()
            assert config["docker_forward_env"] == ["GITHUB_TOKEN", "NPM_TOKEN"]

    def test_create_environment_passes_docker_forward_env(self):
        fake_env = object()
        with patch.object(_tt_mod, "_DockerEnvironment", return_value=fake_env) as mock_docker:
            result = _tt_mod._create_environment(
                "docker",
                image="python:3.11",
                cwd="/root",
                timeout=180,
                container_config={"docker_forward_env": ["GITHUB_TOKEN"]},
            )

        assert result is fake_env
        assert mock_docker.call_args.kwargs["forward_env"] == ["GITHUB_TOKEN"]

    def test_terminal_tool_applies_task_specific_docker_workspace_overrides(self, monkeypatch, tmp_path):
        captured = {}
        fake_env = SimpleNamespace(execute=lambda command, **kwargs: {"output": "ok", "returncode": 0})

        monkeypatch.setattr(_tt_mod, "_active_environments", {})
        monkeypatch.setattr(_tt_mod, "_last_activity", {})
        monkeypatch.setattr(_tt_mod, "_start_cleanup_thread", lambda: None)
        monkeypatch.setattr(
            _tt_mod,
            "_get_env_config",
            lambda: {
                "env_type": "docker",
                "docker_image": "python:3.11",
                "docker_volumes": [],
                "docker_mount_cwd_to_workspace": False,
                "docker_forward_env": [],
                "docker_network": None,
                "cwd": "/root",
                "host_cwd": None,
                "timeout": 180,
                "container_cpu": 1,
                "container_memory": 5120,
                "container_disk": 51200,
                "container_persistent": True,
                "modal_mode": "auto",
            },
        )

        def _fake_create_environment(**kwargs):
            captured.update(kwargs)
            return fake_env

        monkeypatch.setattr(_tt_mod, "_create_environment", _fake_create_environment)

        _tt_mod.register_task_env_overrides(
            "workspace-override-test",
            {
                "cwd": "/workspace",
                "host_cwd": str(tmp_path),
                "docker_mount_cwd_to_workspace": False,
                "docker_volumes": [f"{tmp_path}:/workspace:ro"],
            },
        )

        try:
            result = json.loads(
                _tt_mod.terminal_tool("echo ok", task_id="workspace-override-test", force=True)
            )
        finally:
            _tt_mod.clear_task_env_overrides("workspace-override-test")

        assert result["exit_code"] == 0
        assert captured["cwd"] == "/workspace"
        assert captured["host_cwd"] == str(tmp_path)
        assert captured["container_config"]["docker_volumes"] == [f"{tmp_path}:/workspace:ro"]

    def test_falls_back_to_default(self):
        with patch.dict("os.environ", {}, clear=False):
            # Remove the var if it exists, rely on default
            import os
            env = os.environ.copy()
            env.pop("TERMINAL_TIMEOUT", None)
            with patch.dict("os.environ", env, clear=True):
                assert _parse_env_var("TERMINAL_TIMEOUT", "180") == 180

    # -- invalid int raises ValueError with env var name --

    def test_invalid_int_raises_with_var_name(self):
        with patch.dict("os.environ", {"TERMINAL_TIMEOUT": "5m"}):
            with pytest.raises(ValueError, match="TERMINAL_TIMEOUT"):
                _parse_env_var("TERMINAL_TIMEOUT", "180")

    def test_invalid_int_includes_bad_value(self):
        with patch.dict("os.environ", {"TERMINAL_SSH_PORT": "ssh"}):
            with pytest.raises(ValueError, match="ssh"):
                _parse_env_var("TERMINAL_SSH_PORT", "22")

    # -- invalid JSON raises ValueError with env var name --

    def test_invalid_json_raises_with_var_name(self):
        with patch.dict("os.environ", {"TERMINAL_DOCKER_VOLUMES": "/host:/container"}):
            with pytest.raises(ValueError, match="TERMINAL_DOCKER_VOLUMES"):
                _parse_env_var("TERMINAL_DOCKER_VOLUMES", "[]", json.loads, "valid JSON")

    def test_invalid_json_includes_type_label(self):
        with patch.dict("os.environ", {"TERMINAL_DOCKER_VOLUMES": "not json"}):
            with pytest.raises(ValueError, match="valid JSON"):
                _parse_env_var("TERMINAL_DOCKER_VOLUMES", "[]", json.loads, "valid JSON")
