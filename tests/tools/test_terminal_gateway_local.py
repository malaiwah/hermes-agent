"""Tests for the privileged gateway-local terminal escape hatch."""

import json
from unittest.mock import patch

import tools.terminal_tool as terminal_tool_module


class _FakeEnv:
    def __init__(self):
        self.calls = []

    def execute(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        return {"output": "ok", "returncode": 0}


def _config(**overrides):
    config = {
        "env_type": "docker",
        "cwd": "/workspace",
        "timeout": 60,
        "enable_gateway_local": True,
        "docker_image": "img",
        "singularity_image": "img",
        "modal_image": "img",
        "daytona_image": "img",
        "container_cpu": 1,
        "container_memory": 512,
        "container_disk": 1024,
        "container_persistent": True,
        "modal_mode": "auto",
        "docker_volumes": [],
        "docker_mount_cwd_to_workspace": False,
        "docker_forward_env": [],
        "docker_network": None,
        "local_persistent": False,
        "host_cwd": None,
    }
    config.update(overrides)
    return config


def test_gateway_local_rejected_when_disabled(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: _config(enable_gateway_local=False))

    result = json.loads(terminal_tool_module.terminal_tool("pwd", gateway_local=True))

    assert "disabled" in result["error"].lower()


def test_gateway_local_rejected_for_background(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: _config())

    result = json.loads(
        terminal_tool_module.terminal_tool("sleep 10", gateway_local=True, background=True)
    )

    assert "does not support background" in result["error"].lower()


def test_gateway_local_uses_one_shot_local_env(monkeypatch):
    fake_env = _FakeEnv()
    created = {}
    approvals = {}

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("MESSAGING_CWD", "/srv/gateway-work")
    monkeypatch.setenv("TERMINAL_CWD", "/host/project")
    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: _config())
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool_module, "_active_environments", {})
    def _create_environment(**kwargs):
        created["kwargs"] = kwargs
        return fake_env

    monkeypatch.setattr(terminal_tool_module, "_create_environment", _create_environment)

    def _approve(command, env_type, **kwargs):
        approvals["command"] = command
        approvals["env_type"] = env_type
        approvals["kwargs"] = kwargs
        return {"approved": True, "message": None}

    monkeypatch.setattr(terminal_tool_module, "_check_all_guards", _approve)

    result = json.loads(
        terminal_tool_module.terminal_tool(
            "pwd",
            gateway_local=True,
            task_id="task-1",
            workdir="/workspace/app",
        )
    )

    assert created["kwargs"]["env_type"] == "local"
    assert created["kwargs"]["cwd"] == "/host/project"
    assert created["kwargs"]["local_config"] == {"persistent": False}
    assert approvals["env_type"] == "local"
    assert approvals["kwargs"]["disable_smart_approval"] is True
    assert approvals["kwargs"]["extra_warnings"][0]["pattern_key"] == "gateway_local_execution"
    assert result["execution_scope"] == "gateway_local"
    assert "gateway container" in result["output"].lower()
    assert fake_env.calls[0][1]["cwd"] == "/workspace/app"
    assert terminal_tool_module._active_environments == {}


def test_gateway_local_relative_cwd_resolves_against_messaging_cwd(monkeypatch):
    monkeypatch.setenv("MESSAGING_CWD", "/srv/chat-root")
    monkeypatch.setenv("TERMINAL_CWD", "./project")

    assert terminal_tool_module._resolve_gateway_local_cwd() == "/srv/chat-root/project"


def test_terminal_requirements_allow_gateway_local_fallback(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setattr(
        terminal_tool_module,
        "_get_env_config",
        lambda: _config(env_type="docker", enable_gateway_local=True),
    )
    monkeypatch.setattr(terminal_tool_module, "can_offer_gateway_local", lambda config=None: True)

    assert terminal_tool_module.check_terminal_requirements() is True


def test_gateway_local_real_guard_wrapper_accepts_extra_warnings(monkeypatch):
    fake_env = _FakeEnv()

    monkeypatch.setenv("HERMES_GATEWAY_SESSION", "1")
    monkeypatch.setenv("MESSAGING_CWD", "/srv/gateway-work")
    monkeypatch.setattr(terminal_tool_module, "_get_env_config", lambda: _config())
    monkeypatch.setattr(terminal_tool_module, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool_module, "_active_environments", {})
    monkeypatch.setattr(
        terminal_tool_module,
        "_create_environment",
        lambda **kwargs: fake_env,
    )

    with patch("tools.approval._get_approval_mode", return_value="off"), \
         patch("tools.tirith_security.check_command_security", return_value={"action": "allow", "findings": [], "summary": ""}):
        result = json.loads(
            terminal_tool_module.terminal_tool(
                "pwd",
                gateway_local=True,
                task_id="task-2",
            )
        )

    assert result["error"] is None
    assert result["execution_scope"] == "gateway_local"
