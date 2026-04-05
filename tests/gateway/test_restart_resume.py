import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _source(chat_id="123456", chat_type="dm"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type=chat_type,
    )


def test_gateway_config_round_trips_resume_inflight_flag():
    cfg = GatewayConfig.from_dict({"resume_inflight_sessions_on_restart": True})
    assert cfg.resume_inflight_sessions_on_restart is True
    assert cfg.to_dict()["resume_inflight_sessions_on_restart"] is True


@pytest.mark.asyncio
async def test_gateway_stop_persists_inflight_resume_ledger(tmp_path, monkeypatch):
    import gateway.run as run_mod

    ledger_path = tmp_path / "inflight_resume.json"
    monkeypatch.setattr(run_mod, "_INFLIGHT_RESUME_LEDGER_PATH", ledger_path)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        resume_inflight_sessions_on_restart=True,
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._running_agents_ts = {}
    runner._session_model_overrides = {}
    runner._inflight_turns = {}

    source = _source()
    session_key = build_session_key(source)
    running_agent = MagicMock()
    adapter = StubAdapter()
    adapter._pending_messages[session_key] = MessageEvent(
        text="queued follow-up",
        source=source,
        message_id="msg-2",
    )

    runner._running_agents = {session_key: running_agent}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._pending_approvals = {session_key: {"command": "rm -rf /tmp/x"}}
    runner._session_model_overrides = {
        session_key: {"model": "openrouter/test-model", "provider": "openrouter"}
    }
    runner._inflight_turns = {
        session_key: {
            "session_key": session_key,
            "session_id": "sess_001",
            "source": source.to_dict(),
            "event": GatewayRunner._serialize_resume_event(
                MessageEvent(text="initial work", source=source, message_id="msg-1")
            ),
            "started_at": "2026-04-05T12:00:00",
        }
    }

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    running_agent.interrupt.assert_called_once_with("Gateway shutting down")
    assert ledger_path.exists()
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    entry = payload[0]
    assert entry["session_key"] == session_key
    assert entry["session_id"] == "sess_001"
    assert entry["pending_event"]["text"] == "queued follow-up"
    assert entry["approval_blocked"] is True
    assert entry["session_model_override"]["model"] == "openrouter/test-model"
    assert runner._inflight_turns == {}


@pytest.mark.asyncio
async def test_gateway_stop_skips_resume_ledger_for_turns_that_finish_during_shutdown(tmp_path, monkeypatch):
    import gateway.run as run_mod

    ledger_path = tmp_path / "inflight_resume.json"
    monkeypatch.setattr(run_mod, "_INFLIGHT_RESUME_LEDGER_PATH", ledger_path)

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        resume_inflight_sessions_on_restart=True,
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._running_agents_ts = {}
    runner._session_model_overrides = {}

    source = _source()
    session_key = build_session_key(source)
    runner._inflight_turns = {
        session_key: {
            "session_key": session_key,
            "session_id": "sess_001",
            "source": source.to_dict(),
            "event": GatewayRunner._serialize_resume_event(
                MessageEvent(text="initial work", source=source, message_id="msg-1")
            ),
            "started_at": "2026-04-05T12:00:00",
        }
    }

    running_agent = MagicMock()

    def _interrupt(_reason):
        runner._inflight_turns.pop(session_key, None)

    running_agent.interrupt.side_effect = _interrupt
    runner._running_agents = {session_key: running_agent}
    runner.adapters = {Platform.TELEGRAM: StubAdapter()}

    with patch("gateway.status.remove_pid_file"), patch("gateway.status.write_runtime_status"):
        await runner.stop()

    assert not ledger_path.exists()


@pytest.mark.asyncio
async def test_resume_interrupted_session_uses_hidden_prefix_for_pending_followup(tmp_path, monkeypatch):
    import gateway.run as run_mod

    ledger_path = tmp_path / "inflight_resume.json"
    monkeypatch.setattr(run_mod, "_INFLIGHT_RESUME_LEDGER_PATH", ledger_path)

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_model_overrides = {}
    runner._inflight_turns = {}
    runner._handle_message_with_agent = AsyncMock(return_value=None)

    source = _source()
    session_key = build_session_key(source)
    entry = {
        "session_key": session_key,
        "session_id": "sess_002",
        "source": source.to_dict(),
        "pending_event": {
            "text": "Please use the latest requirements file.",
            "message_type": "text",
            "message_id": "msg-3",
            "media_urls": [],
            "media_types": [],
            "reply_to_message_id": None,
            "reply_to_text": None,
            "auto_skill": None,
            "persist_user_message": None,
            "timestamp": None,
        },
        "approval_blocked": True,
        "session_model_override": {"model": "openrouter/override"},
    }
    ledger_path.write_text(json.dumps([entry]), encoding="utf-8")

    await runner._resume_interrupted_session(entry)

    runner._handle_message_with_agent.assert_awaited_once()
    event, call_source, call_session_key = runner._handle_message_with_agent.call_args.args
    assert call_source.chat_id == source.chat_id
    assert call_session_key == session_key
    assert "gateway restarted" in event.text.lower()
    assert "queued user follow-up" in event.text.lower()
    assert event.persist_user_message == "Please use the latest requirements file."
    assert runner._session_model_overrides[session_key]["model"] == "openrouter/override"
    assert session_key not in runner._running_agents
    assert not ledger_path.exists()


@pytest.mark.asyncio
async def test_resume_interrupted_session_keeps_ledger_entry_on_failure(tmp_path, monkeypatch):
    import gateway.run as run_mod

    ledger_path = tmp_path / "inflight_resume.json"
    monkeypatch.setattr(run_mod, "_INFLIGHT_RESUME_LEDGER_PATH", ledger_path)

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._session_model_overrides = {}
    runner._inflight_turns = {}

    async def _boom(*args, **kwargs):
        raise RuntimeError("resume failed")

    runner._handle_message_with_agent = AsyncMock(side_effect=_boom)

    source = _source()
    session_key = build_session_key(source)
    entry = {
        "session_key": session_key,
        "session_id": "sess_003",
        "source": source.to_dict(),
    }
    ledger_path.write_text(json.dumps([entry]), encoding="utf-8")

    with pytest.raises(RuntimeError, match="resume failed"):
        await runner._resume_interrupted_session(entry)

    remaining = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(remaining) == 1
    assert remaining[0]["session_key"] == session_key


@pytest.mark.asyncio
async def test_recover_inflight_sessions_only_schedules_connected_platforms(tmp_path, monkeypatch):
    import gateway.run as run_mod

    ledger_path = tmp_path / "inflight_resume.json"
    monkeypatch.setattr(run_mod, "_INFLIGHT_RESUME_LEDGER_PATH", ledger_path)

    source = _source()
    other_source = SessionSource(platform=Platform.SLACK, chat_id="C123", chat_type="channel")
    entries = [
        {
            "session_key": build_session_key(source),
            "session_id": "sess_tg",
            "source": source.to_dict(),
        },
        {
            "session_key": build_session_key(other_source),
            "session_id": "sess_slack",
            "source": other_source.to_dict(),
        },
    ]
    ledger_path.write_text(json.dumps(entries), encoding="utf-8")

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        resume_inflight_sessions_on_restart=True,
    )
    runner.adapters = {Platform.TELEGRAM: StubAdapter()}
    runner._background_tasks = set()

    async def _resume(entry):
        entries = json.loads(ledger_path.read_text(encoding="utf-8"))
        entries = [item for item in entries if item.get("session_key") != entry.get("session_key")]
        ledger_path.write_text(json.dumps(entries), encoding="utf-8")

    runner._resume_interrupted_session = AsyncMock(side_effect=_resume)
    runner._resuming_inflight_sessions = set()

    await runner._recover_inflight_sessions()
    await asyncio.sleep(0)

    runner._resume_interrupted_session.assert_awaited_once()
    remaining = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert len(remaining) == 1
    assert remaining[0]["session_id"] == "sess_slack"
