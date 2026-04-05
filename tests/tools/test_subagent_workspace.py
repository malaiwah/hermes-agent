import os

import pytest

from tools.subagent_workspace import build_workspace_overrides, resolve_parent_workspace_root


def test_full_ro_mounts_parent_workspace_read_only(tmp_path):
    plan = build_workspace_overrides(
        visibility="full_ro",
        mappings=None,
        workspace_root=tmp_path,
        child_token="child-1",
        backend="docker",
    )

    assert plan["task_env_overrides"]["cwd"] == "/workspace"
    assert plan["task_env_overrides"]["docker_volumes"] == [f"{tmp_path.resolve()}:/workspace:ro"]
    assert "read-only" in plan["prompt_note"]



def test_mapped_rejects_workspace_escape(tmp_path):
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="escapes the parent workspace"):
        build_workspace_overrides(
            visibility="mapped",
            mappings=[{"source": str(outside), "target": "shared"}],
            workspace_root=tmp_path,
            child_token="child-3",
            backend="docker",
        )


def test_mapped_supports_multiple_targets_and_read_only(tmp_path):
    (tmp_path / "pkg-a").mkdir()
    (tmp_path / "pkg-b").mkdir()

    plan = build_workspace_overrides(
        visibility="mapped",
        mappings=[
            {"source": "pkg-a", "target": "work/a"},
            {"source": "pkg-b", "target": "/workspace/work/b", "read_only": True},
        ],
        workspace_root=tmp_path,
        child_token="child-4",
        backend="docker",
    )

    assert plan["task_env_overrides"]["cwd"] == "/workspace"
    assert plan["task_env_overrides"]["docker_volumes"] == [
        f"{(tmp_path / 'pkg-a').resolve()}:/workspace/work/a",
        f"{(tmp_path / 'pkg-b').resolve()}:/workspace/work/b:ro",
    ]
    assert "/workspace/work/a" in plan["prompt_note"]
    assert "/workspace/work/b" in plan["prompt_note"]


def test_mapped_rejects_container_targets_outside_workspace(tmp_path):
    (tmp_path / "pkg-a").mkdir()

    with pytest.raises(ValueError, match="within /workspace"):
        build_workspace_overrides(
            visibility="mapped",
            mappings=[{"source": "pkg-a", "target": "/workspace2"}],
            workspace_root=tmp_path,
            child_token="child-6",
            backend="docker",
        )


def test_resolve_parent_workspace_root_falls_back_to_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    assert resolve_parent_workspace_root() == tmp_path.resolve()


def test_resolve_parent_workspace_root_prefers_terminal_cwd_when_valid(tmp_path, monkeypatch):
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("TERMINAL_CWD", str(cwd_dir))
    monkeypatch.setenv("HERMES_HOME", str(home_dir))
    assert resolve_parent_workspace_root() == cwd_dir.resolve()


def test_resolve_parent_workspace_root_skips_nonexistent_terminal_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", "/nonexistent/workspace")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert resolve_parent_workspace_root() == tmp_path.resolve()


def test_restricted_modes_require_docker_backend(tmp_path):
    with pytest.raises(ValueError, match="requires the docker terminal backend"):
        build_workspace_overrides(
            visibility="full_ro",
            mappings=None,
            workspace_root=tmp_path,
            child_token="child-5",
            backend="local",
        )
