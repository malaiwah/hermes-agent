import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.config import Platform
from tests.gateway.test_restart_resume import StubAdapter


def _source(chat_id="123456"):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
    )


@pytest.mark.asyncio
async def test_arm_self_nudge_replaces_existing_timer():
    runner = object.__new__(GatewayRunner)
    runner._self_nudge_tasks = {}
    runner._self_nudge_entries = {}
    runner._pending_hidden_turns = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}

    source = _source()
    session_key = build_session_key(source)

    first = await runner._arm_self_nudge(session_key, source, 300, "First note")
    second = await runner._arm_self_nudge(session_key, source, 120, "Second note")

    assert first["armed"] is True
    assert second["armed"] is True
    assert second["replaced_existing"] is True
    assert runner._self_nudge_entries[session_key]["note"] == "Second note"

    await runner._cancel_self_nudge(session_key, reason="test_cleanup")


@pytest.mark.asyncio
async def test_fire_self_nudge_queues_hidden_turn_when_session_busy(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._self_nudge_tasks = {}
    runner._self_nudge_entries = {}
    runner._pending_hidden_turns = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}

    source = _source()
    session_key = build_session_key(source)
    entry = {
        "session_key": session_key,
        "source": source.to_dict(),
        "delay_seconds": 5,
        "note": "Check the deploy result.",
    }
    runner._self_nudge_entries[session_key] = entry
    runner._running_agents[session_key] = object()

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    await runner._fire_self_nudge(entry)

    assert session_key not in runner._self_nudge_entries
    assert runner._pending_hidden_turns[session_key]["note"] == "Check the deploy result."


@pytest.mark.asyncio
async def test_run_self_nudge_entry_injects_hidden_turn():
    runner = object.__new__(GatewayRunner)
    runner._self_nudge_tasks = {}
    runner._self_nudge_entries = {}
    runner._pending_hidden_turns = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._inflight_turns = {}
    runner._handle_message_with_agent = AsyncMock(return_value=None)

    source = _source()
    session_key = build_session_key(source)
    entry = {
        "session_key": session_key,
        "source": source.to_dict(),
        "delay_seconds": 300,
        "note": "Check whether the background task finished.",
    }

    await runner._run_self_nudge_entry(entry)

    runner._handle_message_with_agent.assert_awaited_once()
    event, call_source, call_session_key = runner._handle_message_with_agent.call_args.args
    assert call_source.chat_id == source.chat_id
    assert call_session_key == session_key
    assert "self-nudge timer" in event.text.lower()
    assert "background task finished" in event.text.lower()
    assert event.persist_user_message == "[System note: a self-nudge timer fired.]"
    assert session_key not in runner._running_agents


@pytest.mark.asyncio
async def test_cancel_self_nudge_clears_pending_hidden_turn():
    runner = object.__new__(GatewayRunner)
    runner._self_nudge_tasks = {}
    runner._self_nudge_entries = {}
    runner._pending_hidden_turns = {}

    source = _source()
    session_key = build_session_key(source)
    task = asyncio.create_task(asyncio.sleep(60))
    runner._self_nudge_tasks[session_key] = task
    runner._self_nudge_entries[session_key] = {"session_key": session_key, "source": source.to_dict()}
    runner._pending_hidden_turns[session_key] = {"session_key": session_key}

    cancelled = await runner._cancel_self_nudge(session_key, reason="user_message")

    assert cancelled is True
    assert session_key not in runner._self_nudge_tasks
    assert session_key not in runner._self_nudge_entries
    assert session_key not in runner._pending_hidden_turns
    task.cancel()


@pytest.mark.asyncio
async def test_queued_user_followup_cancels_pending_hidden_turn(monkeypatch):
    import gateway.run as run_mod
    import hermes_cli.tools_config as tools_cfg
    import run_agent as run_agent_mod

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: StubAdapter()}
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._ephemeral_system_prompt = None
    runner._prefill_messages = None
    runner._show_reasoning = False
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner._resolve_turn_agent_config = lambda message, model, runtime: {
        "model": model,
        "runtime": runtime,
    }
    runner._load_reasoning_config = lambda: None
    runner._evict_cached_agent = lambda *_args, **_kwargs: None
    runner._effective_model = None
    runner._effective_provider = None
    runner._pending_hidden_turns = {}
    runner._cancel_self_nudge = AsyncMock(return_value=True)

    source = _source()
    session_key = build_session_key(source)
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter._pending_messages[session_key] = MessageEvent(
        text="queued follow-up",
        source=source,
        message_id="msg-2",
    )
    runner._pending_hidden_turns[session_key] = {
        "session_key": session_key,
        "source": source.to_dict(),
        "delay_seconds": 60,
        "note": "nudge",
    }

    class FakeAgent:
        def __init__(self, **kwargs):
            self.model = kwargs.get("model", "test-model")
            self.context_compressor = SimpleNamespace(last_prompt_tokens=0)
            self.session_prompt_tokens = 0
            self.session_completion_tokens = 0
            self.tools = []
            self.message_callback = None
            self.self_nudge_callback = None
            self.status_callback = None
            self.reasoning_config = None
            self.step_callback = None
            self.stream_delta_callback = None
            self.background_review_callback = None

        def run_conversation(self, *args, **kwargs):
            return {
                "final_response": "Initial response",
                "messages": [
                    {"role": "user", "content": "initial"},
                    {"role": "assistant", "content": "Initial response"},
                ],
                "api_calls": 1,
            }

    monkeypatch.setattr(run_mod, "_load_gateway_config", lambda: {})
    monkeypatch.setattr(run_mod, "_resolve_gateway_model", lambda config=None: "test-model")
    monkeypatch.setattr(
        run_mod,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "api_key": "sk-test",
            "base_url": "",
            "provider": "",
            "api_mode": None,
            "command": None,
            "args": [],
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(run_mod, "_platform_config_key", lambda _platform: "telegram")
    monkeypatch.setattr(tools_cfg, "_get_platform_tools", lambda cfg, key: ["user_updates"])
    monkeypatch.setattr(run_agent_mod, "AIAgent", FakeAgent)

    recursive = AsyncMock(
        return_value={"final_response": "follow-up", "messages": [], "history_offset": 0}
    )
    runner._run_agent = recursive

    result = await GatewayRunner._run_agent(
        runner,
        message="initial",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key=session_key,
    )

    runner._cancel_self_nudge.assert_awaited_once_with(
        session_key,
        reason="queued_user_followup",
    )
    recursive.assert_awaited_once()
    assert recursive.await_args.kwargs["message"] == "queued follow-up"
    assert result["final_response"] == "follow-up"
