"""Tests for gateway /status behavior and token persistence."""

import threading
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source(platform: Platform = Platform.TELEGRAM) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str, platform: Platform = Platform.TELEGRAM) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(platform),
        message_id="m1",
    )


def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._session_db.get_session_token_totals.return_value = None
    runner._session_db.get_session.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._session_model_overrides = {}
    runner._pending_hidden_turns = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_status_command_reports_running_agent_without_interrupt(monkeypatch):
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=321,
    )
    runner = _make_runner(session_entry)
    running_agent = MagicMock()
    running_agent.model = "openai/test-model"
    running_agent.provider = "openai"
    running_agent.api_mode = "chat_completions"
    running_agent.session_input_tokens = 111
    running_agent.session_output_tokens = 210
    running_agent.session_total_tokens = 321
    running_agent.session_cache_read_tokens = 0
    running_agent.session_cache_write_tokens = 0
    running_agent.session_reasoning_tokens = 0
    running_agent.session_estimated_cost_usd = 0.0
    running_agent.session_cost_status = "estimated"
    runner._running_agents[build_session_key(_make_source())] = running_agent

    result = await runner._handle_message(_make_event("/status"))

    assert "Hermes Agent v" in result
    assert "**Model:** `openai/test-model`" in result
    assert "**Usage:** 111 in · 210 out · 321 total" in result
    assert "**Session:** `agent:main:telegram:dm:c1`" in result
    assert "**ID:** `sess-1`" in result
    assert "**Queue:** depth 0 · **State:** running" in result
    assert "**Chats:** telegram" in result
    assert "**Title:**" not in result
    running_agent.interrupt.assert_not_called()
    assert runner._pending_messages == {}


@pytest.mark.asyncio
async def test_status_command_includes_session_title_when_present():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=321,
    )
    runner = _make_runner(session_entry)
    runner._session_db.get_session_title.return_value = "My titled session"

    result = await runner._handle_message(_make_event("/status"))

    assert "**Title:** My titled session" in result


@pytest.mark.asyncio
async def test_status_command_prefers_sessiondb_token_totals():
    """When SessionDB has token totals, /status uses them as the
    authoritative source, not the (now-vestigial) SessionStore field."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=321,  # stale fallback value
    )
    runner = _make_runner(session_entry)
    runner._session_db.get_session_token_totals.return_value = {
        "input_tokens": 100,
        "output_tokens": 200,
        "cache_read_tokens": 10,
        "cache_write_tokens": 5,
        "reasoning_tokens": 6,
        "total_tokens": 3210,
    }
    runner._session_db.get_session.return_value = {
        "model": "openai/test-model",
        "billing_provider": "openai",
        "estimated_cost_usd": 1.2345,
        "cost_status": "estimated",
    }

    result = await runner._handle_message(_make_event("/status"))

    assert "**Usage:** 100 in · 200 out · 3,210 total · **Cost:** $1.2345 est." in result
    assert "**Cache:** 10 read · 5 write · 9% hit · 6 reasoning" in result


@pytest.mark.asyncio
async def test_status_command_falls_back_when_sessiondb_row_missing():
    """When SessionDB has no row for this session (fresh install, DB
    unavailable, or pre-SessionDB session), fall back to the persisted
    SessionStore total."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=321,
    )
    runner = _make_runner(session_entry)
    runner._session_db.get_session_token_totals.return_value = None

    result = await runner._handle_message(_make_event("/status"))

    assert "**Usage:** 0 in · 0 out · 321 total" in result


@pytest.mark.asyncio
async def test_status_command_shows_live_context_metrics():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    running_agent = MagicMock()
    running_agent.model = "openai/test-model"
    running_agent.provider = "openai"
    running_agent.api_mode = "chat_completions"
    running_agent.session_input_tokens = 120
    running_agent.session_output_tokens = 45
    running_agent.session_total_tokens = 165
    running_agent.session_cache_read_tokens = 25
    running_agent.session_cache_write_tokens = 5
    running_agent.session_reasoning_tokens = 9
    running_agent.session_estimated_cost_usd = 0.4321
    running_agent.session_cost_status = "estimated"
    running_agent.context_compressor = SimpleNamespace(
        last_prompt_tokens=28000,
        context_length=500000,
        compression_count=2,
    )
    runner._running_agents[session_entry.session_key] = running_agent

    result = await runner._handle_message(_make_event("/status"))

    assert "**Cache:** 25 read · 5 write · 17% hit · 9 reasoning" in result
    assert "**Context:** 28,000 / 500,000 (6%) · **Compactions:** 2" in result


@pytest.mark.asyncio
async def test_status_command_shows_idle_context_from_last_prompt_tokens(monkeypatch):
    """When no live agent exists but session has last_prompt_tokens and
    compression_count, show estimated context and persisted compaction
    count (idle fallback, inspired by PR #4678 and issue #7317)."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        last_prompt_tokens=45000,
        compression_count=3,
    )
    runner = _make_runner(session_entry)
    runner._session_db.get_session.return_value = {
        "model": "openai/gpt-4o",
        "billing_provider": "openai",
        "estimated_cost_usd": 0.0,
        "cost_status": "estimated",
    }
    import agent.model_metadata as _mm
    monkeypatch.setattr(_mm, "get_model_context_length", lambda model, **kw: 200_000)

    result = await runner._handle_message(_make_event("/status"))

    assert "**Context:** 45,000 / 200,000 (22%) · **Compactions:** 3" in result


@pytest.mark.asyncio
async def test_status_command_omits_context_when_lookup_fails(monkeypatch):
    """When model metadata lookup fails, context section is omitted."""
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner._session_db.get_session.return_value = {
        "model": "openai/test-model",
        "billing_provider": "openai",
        "estimated_cost_usd": 0.0,
        "cost_status": "estimated",
    }

    def _raise(*args, **kwargs):
        raise Exception("not found")

    import agent.model_metadata as _mm
    monkeypatch.setattr(_mm, "get_model_context_length", _raise)

    result = await runner._handle_message(_make_event("/status"))

    assert "**Context:**" not in result


@pytest.mark.asyncio
async def test_status_command_shows_model_override_and_reasoning():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner._session_model_overrides[session_entry.session_key] = {
        "model": "anthropic/claude-test",
        "provider": "anthropic",
        "api_mode": "anthropic_messages",
    }
    runner._reasoning_config = {"enabled": True, "effort": "high"}

    result = await runner._handle_message(_make_event("/status"))

    assert "**Model:** `anthropic/claude-test` · **Provider:** anthropic" in result
    assert "**Runtime:** Anthropic Messages · Reasoning high" in result


@pytest.mark.asyncio
async def test_status_command_splits_chat_platforms_and_services():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.adapters[Platform.API_SERVER] = MagicMock()
    runner.adapters[Platform.WEBHOOK] = MagicMock()

    result = await runner._handle_message(_make_event("/status"))

    assert "**Chats:** telegram" in result
    assert "**Services:** api_server, webhook" in result


@pytest.mark.asyncio
async def test_status_command_reports_non_zero_queue_depth():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner._pending_messages[session_entry.session_key] = "follow-up"

    result = await runner._handle_message(_make_event("/status"))

    assert "**Queue:** depth 1 · **State:** idle" in result


@pytest.mark.asyncio
async def test_status_command_counts_pending_hidden_turns_in_queue_depth():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner._pending_hidden_turns[session_entry.session_key] = {
        "session_key": session_entry.session_key,
        "note": "nudge",
    }

    result = await runner._handle_message(_make_event("/status"))

    assert "**Queue:** depth 1 · **State:** idle" in result


@pytest.mark.asyncio
async def test_status_command_supports_discord_channel():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source(Platform.DISCORD)),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.DISCORD,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.adapters = {Platform.DISCORD: MagicMock()}
    runner._session_db.get_session.return_value = {
        "model": "openai/test-model",
        "billing_provider": "openai",
        "estimated_cost_usd": 0.25,
        "cost_status": "estimated",
    }

    result = await runner._handle_message(_make_event("/status", Platform.DISCORD))

    assert "**Chats:** discord" in result
    assert "Transport" not in result


@pytest.mark.asyncio
async def test_status_command_supports_api_server_channel():
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source(Platform.API_SERVER)),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.API_SERVER,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.adapters = {Platform.API_SERVER: MagicMock()}
    runner._session_db.get_session.return_value = {
        "model": "openai/test-model",
        "billing_provider": "openai",
    }

    result = await runner._handle_message(_make_event("/status", Platform.API_SERVER))

    assert "**Services:** api_server" in result
    assert "**Chats:**" not in result


@pytest.mark.asyncio
async def test_handle_message_persists_agent_token_counts(monkeypatch):
    import gateway.run as gateway_run

    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner = _make_runner(session_entry)
    runner.session_store.load_transcript.return_value = [{"role": "user", "content": "earlier"}]
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "ok",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "last_prompt_tokens": 80,
            "input_tokens": 120,
            "output_tokens": 45,
            "model": "openai/test-model",
        }
    )

    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100000,
    )

    result = await runner._handle_message(_make_event("hello"))

    assert result == "ok"
    runner.session_store.update_session.assert_called_once_with(
        session_entry.session_key,
        last_prompt_tokens=80,
        compression_count=0,
    )



@pytest.mark.asyncio
async def test_status_command_bypasses_active_session_guard():
    """When an agent is running, /status must be dispatched immediately via
    base.handle_message — not queued or treated as an interrupt (#5046)."""
    import asyncio
    from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
    from gateway.session import build_session_key
    from gateway.config import Platform, PlatformConfig, GatewayConfig

    source = _make_source()
    session_key = build_session_key(source)

    handler_called_with = []

    async def fake_handler(event):
        handler_called_with.append(event)
        return "📊 **Hermes Gateway Status**\n**Agent Running:** Yes ⚡"

    # Concrete subclass to avoid abstract method errors
    class _ConcreteAdapter(BasePlatformAdapter):
        platform = Platform.TELEGRAM

        async def connect(self): pass
        async def disconnect(self): pass
        async def send(self, chat_id, content, **kwargs): pass
        async def get_chat_info(self, chat_id): return {}

    platform_config = PlatformConfig(enabled=True, token="***")
    adapter = _ConcreteAdapter(platform_config, Platform.TELEGRAM)
    adapter.set_message_handler(fake_handler)

    sent = []

    async def fake_send_with_retry(chat_id, content, reply_to=None, metadata=None):
        sent.append(content)

    adapter._send_with_retry = fake_send_with_retry

    # Simulate an active session
    interrupt_event = asyncio.Event()
    adapter._active_sessions[session_key] = interrupt_event

    event = MessageEvent(
        text="/status",
        source=source,
        message_id="m1",
        message_type=MessageType.COMMAND,
    )
    await adapter.handle_message(event)

    assert handler_called_with, "/status handler was never called (event was queued or dropped)"
    assert sent, "/status response was never sent"
    assert "Agent Running" in sent[0]
    assert not interrupt_event.is_set(), "/status incorrectly triggered an agent interrupt"
    assert session_key not in adapter._pending_messages, "/status was incorrectly queued"
