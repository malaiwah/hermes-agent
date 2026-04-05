import json


def test_terminal_tool_surfaces_allowlist_persistence_warning(monkeypatch):
    import tools.terminal_tool as mod

    class FakeEnv:
        def execute(self, command, **kwargs):
            return {"output": "command output", "returncode": 0}

    monkeypatch.setattr(
        mod,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "cwd": ".",
            "timeout": 30,
            "docker_image": "",
            "singularity_image": "",
            "modal_image": "",
            "daytona_image": "",
        },
    )
    monkeypatch.setattr(mod, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(
        mod,
        "_check_all_guards",
        lambda command, env_type: {
            "approved": True,
            "message": "Permanent approval was applied for this session only.",
        },
    )
    monkeypatch.setattr(mod, "_active_environments", {"default": FakeEnv()})
    monkeypatch.setattr(mod, "_last_activity", {})

    result = json.loads(mod.terminal_tool("echo hello"))

    assert result["exit_code"] == 0
    assert "Permanent approval was applied for this session only." in result["output"]
    assert "command output" in result["output"]
