import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent import async_delegate_tasks as adt


class FakeChild:
    def __init__(self):
        self.session_id = "child-session-1"
        self.tool_progress_callback = None
        self._delegate_terminal_overrides = {"env_type": "docker"}
        self.interrupted = False

    def interrupt(self):
        self.interrupted = True


@pytest.fixture()
def manager(monkeypatch, tmp_path):
    mgr = adt.get_async_delegate_manager()
    mgr.reset_for_tests()
    monkeypatch.setattr(
        adt,
        "_load_async_delegate_config",
        lambda: {
            "enabled": True,
            "max_per_session": 2,
            "max_global": 4,
            "idle_timeout_seconds": 10,
            "max_duration_seconds": 20,
            "output_dir": ".hermes-async-delegates",
        },
    )
    monkeypatch.setattr(adt, "_resolve_workspace_root", lambda parent: tmp_path)
    yield mgr
    mgr.reset_for_tests()


def test_async_delegate_writes_workspace_file_and_nudges_completion(manager, monkeypatch, tmp_path):
    child = FakeChild()
    run_gate = threading.Event()
    parent_tool_names = ["terminal", "file", "web"]
    child_tool_names = ["delegate_task", "terminal"]

    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {"max_iterations": 50})
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_delegation_profile",
        lambda cfg, profile: {"name": profile or "", "toolsets": ["terminal", "file"], "memory": "none", "provider_tools": False, "terminal": {"backend": "docker"}},
    )
    monkeypatch.setattr(adt, "_resolve_workspace_root", lambda parent: tmp_path / "workspace")

    def _fake_build_child_agent(**kwargs):
        import model_tools

        model_tools._last_resolved_tool_names = list(child_tool_names)
        return child

    monkeypatch.setattr("tools.delegate_tool._build_child_agent", _fake_build_child_agent)

    def _fake_run_single_child(**kwargs):
        run_gate.wait(timeout=2)
        return {
            "task_index": 0,
            "status": "completed",
            "summary": "Likely root cause is stale cache state.",
            "api_calls": 3,
            "duration_seconds": 1.0,
        }

    monkeypatch.setattr("tools.delegate_tool._run_single_child", _fake_run_single_child)

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    parent = SimpleNamespace(session_id="session-1", _delegate_depth=0, _current_workspace=str(workspace_root))

    import model_tools

    original_tool_names = list(getattr(model_tools, "_last_resolved_tool_names", []))
    model_tools._last_resolved_tool_names = list(parent_tool_names)
    spawned = manager.spawn(
        owner_session_id="session-1",
        parent_agent=parent,
        goal="Investigate flaky tests",
        context="Check cache state first",
        toolsets=["terminal", "file"],
        profile="friendly",
        max_iterations=30,
        creds={"model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None},
    )

    assert spawned["success"] is True
    output_file = Path(spawned["output_file"])
    assert output_file.exists()
    assert str(workspace_root / ".hermes-async-delegates") in spawned["output_file"]
    assert child._delegate_terminal_overrides["docker_mount_cwd_to_workspace"] is True
    assert child._delegate_terminal_overrides["cwd"] == "/workspace"
    assert child._delegate_saved_tool_names == parent_tool_names
    assert model_tools._last_resolved_tool_names == parent_tool_names

    child.tool_progress_callback("terminal", "pytest -q")
    run_gate.set()

    deadline = time.time() + 2
    while time.time() < deadline:
        record = next(iter(manager._records.values()))
        if record.status == "completed":
            break
        time.sleep(0.05)

    record = next(iter(manager._records.values()))
    assert record.status == "completed"
    contents = output_file.read_text(encoding="utf-8")
    assert "pytest -q" in contents
    assert "Likely root cause is stale cache state." in contents

    context = manager.render_turn_context("session-1")
    assert "completed" in context
    assert record.id in context
    assert manager.render_turn_context("session-1") == ""
    model_tools._last_resolved_tool_names = original_tool_names


def test_resolve_workspace_root_prefers_parent_workspace_over_process_cwd(tmp_path, monkeypatch):
    process_cwd = tmp_path / "process-cwd"
    process_cwd.mkdir()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    monkeypatch.chdir(process_cwd)

    parent = SimpleNamespace(_current_workspace=str(workspace_root))
    assert adt._resolve_workspace_root(parent) == workspace_root.resolve()


def test_async_delegate_renders_idle_without_nudge(manager, monkeypatch):
    child = FakeChild()
    run_gate = threading.Event()

    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {"max_iterations": 50})
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_delegation_profile",
        lambda cfg, profile: {"name": profile or "", "toolsets": ["terminal"], "memory": "none", "provider_tools": False, "terminal": {"backend": "docker"}},
    )
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **kwargs: child)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda **kwargs: (run_gate.wait(timeout=2), {"task_index": 0, "status": "completed", "summary": "done", "api_calls": 1, "duration_seconds": 1.0})[1],
    )

    parent = SimpleNamespace(session_id="session-1", _delegate_depth=0)
    spawned = manager.spawn(
        owner_session_id="session-1",
        parent_agent=parent,
        goal="Long task",
        context="",
        toolsets=["terminal"],
        profile=None,
        max_iterations=20,
        creds={},
    )
    record = manager._records[spawned["id"]]
    record.last_activity_at = adt._now() - 30

    context = manager.render_turn_context("session-1")
    assert "[idle]" in context
    assert "Async delegated subagent updates" not in context

    run_gate.set()


def test_async_delegate_max_duration_interrupts_and_nudges(manager, monkeypatch):
    child = FakeChild()
    run_gate = threading.Event()

    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {"max_iterations": 50})
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_delegation_profile",
        lambda cfg, profile: {"name": profile or "", "toolsets": ["terminal"], "memory": "none", "provider_tools": False, "terminal": {"backend": "docker"}},
    )
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", lambda **kwargs: child)
    monkeypatch.setattr(
        "tools.delegate_tool._run_single_child",
        lambda **kwargs: (run_gate.wait(timeout=2), {"task_index": 0, "status": "completed", "summary": "partial", "api_calls": 1, "duration_seconds": 1.0})[1],
    )

    parent = SimpleNamespace(session_id="session-1", _delegate_depth=0)
    spawned = manager.spawn(
        owner_session_id="session-1",
        parent_agent=parent,
        goal="Time bounded task",
        context="",
        toolsets=["terminal"],
        profile=None,
        max_iterations=20,
        creds={},
    )
    record = manager._records[spawned["id"]]
    record.max_duration_at = adt._now() - 1

    manager._sweep_once()
    assert child.interrupted is True
    assert record.status == "timed_out"
    context = manager.render_turn_context("session-1")
    assert "maximum runtime" in context

    run_gate.set()
