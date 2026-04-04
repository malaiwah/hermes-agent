import json

import pytest

from agent import background_subagents as bg


class FakeExecSession:
    def __init__(self):
        self.stdout_handler = None
        self.stderr_handler = None
        self.exit_handler = None
        self.alive = True
        self.client_responses = []
        self.pending_prompts = []
        self.writes = []
        self.pid = 4321

    def read_loop(self, *, stdout_handler=None, stderr_handler=None, exit_handler=None):
        self.stdout_handler = stdout_handler
        self.stderr_handler = stderr_handler
        self.exit_handler = exit_handler

    def write_line(self, payload: str) -> None:
        msg = json.loads(payload)
        self.writes.append(msg)
        method = msg.get("method")
        if method == "initialize":
            self._emit({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
        elif method == "session/new":
            self._emit({"jsonrpc": "2.0", "id": msg["id"], "result": {"sessionId": "acp-session-1"}})
        elif method == "session/prompt":
            self.pending_prompts.append(msg)
            self._emit(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {
                        "update": {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"text": f"working:{msg['params']['prompt'][0]['text']}"},
                        }
                    },
                }
            )
        elif method == "session/cancel":
            self._emit({"jsonrpc": "2.0", "id": msg["id"], "result": {}})
        elif "result" in msg or "error" in msg:
            self.client_responses.append(msg)

    def complete_prompt(self, *, stop_reason: str = "end_turn") -> None:
        prompt = self.pending_prompts.pop(0)
        self._emit(
            {
                "jsonrpc": "2.0",
                "id": prompt["id"],
                "result": {"stopReason": stop_reason},
            }
        )

    def emit_stderr(self, text: str) -> None:
        if self.stderr_handler:
            self.stderr_handler(text)

    def emit_raw_stdout(self, line: str) -> None:
        if self.stdout_handler:
            self.stdout_handler(line)

    def is_session_alive(self) -> bool:
        return self.alive

    def terminate_session(self) -> None:
        if not self.alive:
            return
        self.alive = False
        if self.exit_handler:
            self.exit_handler(0)

    def _emit(self, payload: dict) -> None:
        if self.stdout_handler:
            self.stdout_handler(json.dumps(payload))


class FakeEnvironment:
    def __init__(self, exec_session: FakeExecSession):
        self.exec_session = exec_session
        self.cleanup_called = False
        self.start_args = None
        self._container_id = "container-123"

    def start_persistent_exec(self, *, cwd="", command=None, env=None):
        self.start_args = {
            "cwd": cwd,
            "command": list(command or []),
            "env": dict(env or {}),
        }
        return self.exec_session

    def cleanup(self):
        self.cleanup_called = True


@pytest.fixture()
def manager(monkeypatch):
    mgr = bg.get_background_subagent_manager()
    mgr.reset_for_tests()
    monkeypatch.setattr(
        bg,
        "_load_background_subagent_config",
        lambda: {
            "enabled": True,
            "max_per_session": 2,
            "max_global": 4,
            "idle_timeout_seconds": 900,
            "max_lifetime_seconds": 7200,
            "default_agent_kind": "opencode",
            "agents": {
                "opencode": {
                    "command": "opencode",
                    "args": ["acp"],
                    "cwd_mode": "session",
                }
            },
        },
    )
    yield mgr
    mgr.reset_for_tests()


def test_background_subagent_lifecycle(manager, monkeypatch):
    exec_session = FakeExecSession()
    environment = FakeEnvironment(exec_session)
    monkeypatch.setattr(manager, "_create_environment", lambda **kwargs: environment)

    spawned = manager.spawn_subagent(
        owner_session_id="session-1",
        purpose="Track a long-running code investigation",
        initial_task="Inspect failing tests",
        cwd="/workspace",
    )

    assert spawned["success"] is True
    subagent_id = spawned["id"]
    assert environment.start_args["command"] == ["opencode", "acp"]

    listed = manager.list_subagents(owner_session_id="session-1")
    assert [item["id"] for item in listed["subagents"]] == [subagent_id]
    assert listed["subagents"][0]["status"] == "running"

    polled = manager.poll_subagent(owner_session_id="session-1", subagent_id=subagent_id)
    kinds = [event["kind"] for event in polled["events"]]
    assert "task_dispatched" in kinds
    assert "agent_message_chunk" in kinds

    exec_session.complete_prompt()
    status = manager.get_status(owner_session_id="session-1", subagent_id=subagent_id)
    assert status["status"] == "idle"

    context = manager.render_turn_context("session-1")
    assert "finished its latest task" in context
    assert subagent_id in context
    assert "Track a long-running code investigation" in context
    assert "Inspect failing tests" not in manager.render_turn_context("session-1")

    send = manager.send_message(
        owner_session_id="session-1",
        subagent_id=subagent_id,
        message="Inspect the Docker environment next",
    )
    assert send["success"] is True
    assert send["queued"] is False

    queued = manager.send_message(
        owner_session_id="session-1",
        subagent_id=subagent_id,
        message="Then summarize the likely root cause",
    )
    assert queued["success"] is True
    assert queued["queued"] is True

    exec_session.complete_prompt()
    mid_status = manager.get_status(owner_session_id="session-1", subagent_id=subagent_id)
    assert mid_status["status"] == "running"
    exec_session.complete_prompt()
    final_status = manager.get_status(owner_session_id="session-1", subagent_id=subagent_id)
    assert final_status["status"] == "idle"

    stopped = manager.stop_subagent(
        owner_session_id="session-1",
        subagent_id=subagent_id,
        reason="done",
    )
    assert stopped["success"] is True
    assert stopped["status"] == "stopped"
    assert environment.cleanup_called is True


def test_background_subagent_limits(manager, monkeypatch):
    exec_one = FakeExecSession()
    env_one = FakeEnvironment(exec_one)
    monkeypatch.setattr(manager, "_create_environment", lambda **kwargs: env_one)

    manager.spawn_subagent(
        owner_session_id="session-1",
        purpose="First",
        initial_task="One",
        cwd="/workspace",
    )

    monkeypatch.setattr(
        bg,
        "_load_background_subagent_config",
        lambda: {
            "enabled": True,
            "max_per_session": 1,
            "max_global": 1,
            "idle_timeout_seconds": 900,
            "max_lifetime_seconds": 7200,
            "default_agent_kind": "opencode",
            "agents": {"opencode": {"command": "opencode", "args": ["acp"], "cwd_mode": "session"}},
        },
    )

    limited = manager.spawn_subagent(
        owner_session_id="session-1",
        purpose="Second",
        initial_task="Two",
        cwd="/workspace",
    )
    assert limited["success"] is False
    assert "limit reached" in limited["error"].lower()
    assert len(limited["stoppable_subagents"]) == 1


def test_background_subagent_idle_timeout(manager, monkeypatch):
    exec_session = FakeExecSession()
    environment = FakeEnvironment(exec_session)
    monkeypatch.setattr(manager, "_create_environment", lambda **kwargs: environment)

    spawned = manager.spawn_subagent(
        owner_session_id="session-1",
        purpose="Timeout me",
        initial_task="Start work",
        cwd="/workspace",
    )
    exec_session.complete_prompt()
    record = manager._get_owned_record("session-1", spawned["id"])
    assert record is not None
    record.idle_timeout_at = bg._now() - 1

    manager._sweep_once()

    status = manager.get_status(owner_session_id="session-1", subagent_id=spawned["id"])
    assert status["status"] == "timed_out"
    assert environment.cleanup_called is True


def test_background_subagent_protocol_error_stops_session(manager, monkeypatch):
    exec_session = FakeExecSession()
    environment = FakeEnvironment(exec_session)
    monkeypatch.setattr(manager, "_create_environment", lambda **kwargs: environment)

    spawned = manager.spawn_subagent(
        owner_session_id="session-1",
        purpose="Break protocol",
        initial_task="Start work",
        cwd="/workspace",
    )
    exec_session.emit_raw_stdout("not-json")

    status = manager.get_status(owner_session_id="session-1", subagent_id=spawned["id"])
    assert status["status"] == "error"
    context = manager.render_turn_context("session-1")
    assert "protocol_error" in context
