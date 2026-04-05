from argparse import Namespace
from pathlib import Path

from hermes_cli.config import ConfigWriteError


def _config_write_error(tmp_path: Path) -> ConfigWriteError:
    return ConfigWriteError(
        path=tmp_path / "config.yaml",
        action="save configuration",
        error=OSError(16, "Device or resource busy"),
        blocked=True,
        diff="--- before\n+++ after",
    )


def test_setup_wizard_catches_config_write_error(tmp_path, monkeypatch, capsys):
    from hermes_cli import setup as setup_mod

    monkeypatch.setattr("hermes_cli.config.is_managed", lambda: False)
    monkeypatch.setattr(setup_mod, "ensure_hermes_home", lambda: None)
    monkeypatch.setattr(setup_mod, "load_config", lambda: {})
    monkeypatch.setattr(setup_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(setup_mod, "is_interactive_stdin", lambda: True)
    monkeypatch.setattr("hermes_cli.auth.get_active_provider", lambda: None)
    monkeypatch.setattr(setup_mod, "SETUP_SECTIONS", [("model", "Model", lambda config: None)])
    monkeypatch.setattr(setup_mod, "save_config", lambda config: (_ for _ in ()).throw(_config_write_error(tmp_path)))

    args = Namespace(section="model", non_interactive=False, reset=False)
    setup_mod.run_setup_wizard(args)

    out = capsys.readouterr().out
    assert "read-only or otherwise locked" in out
    assert "Apply this patch manually" in out


def test_tools_command_catches_config_write_error(tmp_path, monkeypatch, capsys):
    from hermes_cli import tools_config as mod

    monkeypatch.setattr(mod, "_get_enabled_platforms", lambda: ["cli"])
    monkeypatch.setattr(mod, "_get_platform_tools", lambda *args, **kwargs: set())
    monkeypatch.setattr(mod, "_prompt_toolset_checklist", lambda *args, **kwargs: {"web"})
    monkeypatch.setattr(mod, "_configure_toolset", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "apply_nous_managed_defaults", lambda *args, **kwargs: set())
    monkeypatch.setattr(mod, "_save_platform_tools", lambda *args, **kwargs: (_ for _ in ()).throw(_config_write_error(tmp_path)))

    mod.tools_command(first_install=True, config={})

    out = capsys.readouterr().out
    assert "read-only or otherwise locked" in out


def test_mcp_command_catches_config_write_error(tmp_path, monkeypatch, capsys):
    from hermes_cli import mcp_config as mod

    monkeypatch.setattr(mod, "cmd_mcp_configure", lambda args: (_ for _ in ()).throw(_config_write_error(tmp_path)))
    mod.mcp_command(Namespace(mcp_action="configure", name="demo"))

    out = capsys.readouterr().out
    assert "read-only or otherwise locked" in out


def test_skills_command_catches_config_write_error(tmp_path, monkeypatch, capsys):
    from hermes_cli import skills_config as mod

    monkeypatch.setattr(mod, "_list_all_skills", lambda: [{
        "name": "demo",
        "category": "general",
        "description": "demo skill",
    }])
    monkeypatch.setattr(mod, "_select_platform", lambda: None)
    monkeypatch.setattr("builtins.input", lambda *_args, **_kwargs: "1")
    monkeypatch.setattr("hermes_cli.curses_ui.curses_checklist", lambda *args, **kwargs: set())
    monkeypatch.setattr(mod, "save_disabled_skills", lambda *args, **kwargs: (_ for _ in ()).throw(_config_write_error(tmp_path)))

    mod.skills_command()

    out = capsys.readouterr().out
    assert "read-only or otherwise locked" in out


def test_auth_command_catches_config_write_error(tmp_path, monkeypatch, capsys):
    from hermes_cli import auth_commands as mod

    monkeypatch.setattr(mod, "auth_add_command", lambda args: (_ for _ in ()).throw(_config_write_error(tmp_path)))
    mod.auth_command(Namespace(auth_action="add"))

    out = capsys.readouterr().out
    assert "read-only or otherwise locked" in out


def test_plugins_command_catches_config_write_error(tmp_path, monkeypatch, capsys):
    from hermes_cli import plugins_cmd as mod

    monkeypatch.setattr(mod, "cmd_enable", lambda name: (_ for _ in ()).throw(_config_write_error(tmp_path)))
    mod.plugins_command(Namespace(plugins_action="enable", name="demo"))

    out = capsys.readouterr().out
    assert "read-only or otherwise locked" in out
