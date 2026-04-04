"""Tests for _parse_env_var and _get_env_config env-var validation."""

import json
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

    def test_terminal_tool_passes_docker_forward_env_into_container_config(self):
        fake_env = type(
            "FakeEnv",
            (),
            {"execute": lambda self, *args, **kwargs: {"output": "ok", "exit_code": 0, "error": ""}},
        )()
        config = {
            "env_type": "docker",
            "docker_image": "python:3.11",
            "cwd": "/root",
            "timeout": 180,
            "container_cpu": 1,
            "container_memory": 5120,
            "container_disk": 51200,
            "container_persistent": True,
            "modal_mode": "auto",
            "docker_volumes": [],
            "docker_mount_cwd_to_workspace": False,
            "docker_forward_env": ["GITHUB_TOKEN", "NPM_TOKEN"],
            "host_cwd": None,
        }
        task_id = "docker-forward-env-test"
        _tt_mod._active_environments.clear()
        _tt_mod._last_activity.clear()
        _tt_mod._task_env_overrides.clear()

        try:
            with patch.object(_tt_mod, "_get_env_config", return_value=config), \
                 patch.object(_tt_mod, "_create_environment", return_value=fake_env) as mock_create, \
                 patch.object(_tt_mod, "_start_cleanup_thread"), \
                 patch.object(_tt_mod, "_check_all_guards", return_value={"approved": True}):
                result = json.loads(_tt_mod.terminal_tool("echo ok", task_id=task_id))

            assert result["exit_code"] == 0
            assert mock_create.call_args.kwargs["container_config"]["docker_forward_env"] == [
                "GITHUB_TOKEN",
                "NPM_TOKEN",
            ]
        finally:
            _tt_mod._active_environments.clear()
            _tt_mod._last_activity.clear()

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
