from pathlib import Path

import pytest

from tools.subagent_workspace import build_workspace_overrides


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


def test_temp_rw_creates_isolated_subworkspace(tmp_path):
    plan = build_workspace_overrides(
        visibility="temp_rw",
        mappings=None,
        workspace_root=tmp_path,
        child_token="child-2",
        backend="docker",
    )

    mount = plan["task_env_overrides"]["docker_volumes"][0]
    host_path = Path(mount.split(":", 1)[0])
    assert host_path.is_dir()
    assert host_path.parent == tmp_path / ".hermes-subagents"
    assert mount.endswith(":/workspace")
    assert "isolated writable subworkspace" in plan["prompt_note"]


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


def test_restricted_modes_require_docker_backend(tmp_path):
    with pytest.raises(ValueError, match="requires the docker terminal backend"):
        build_workspace_overrides(
            visibility="full_ro",
            mappings=None,
            workspace_root=tmp_path,
            child_token="child-5",
            backend="local",
        )
