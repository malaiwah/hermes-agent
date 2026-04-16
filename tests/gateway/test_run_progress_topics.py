"""Tests for topic-aware gateway progress updates."""

import asyncio
import importlib
import sys
import time
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class ProgressCaptureAdapter(BasePlatformAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self.sent = []
        self.edits = []
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="progress-1")

    async def edit_message(self, chat_id, message_id, content) -> SendResult:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class DiscordVoiceProgressAdapter(ProgressCaptureAdapter):
    def __init__(self):
        super().__init__(platform=Platform.DISCORD)
        self._voice_text_channels = {111: "voice-chat"}
        self._voice_last_language = {"voice-chat": "English"}
        self._last_tts_voice = {"voice-chat": "vc_test_voice"}
        self._last_tts_instruct = {"voice-chat": "warm and steady"}
        self.play_calls = []
        self.tts_play_calls = []

    def is_in_voice_channel(self, guild_id: int) -> bool:
        return guild_id == 111

    async def play_pcm_stream_in_voice_channel(self, guild_id: int, source) -> bool:
        self.play_calls.append(
            {
                "guild_id": guild_id,
                "trace_id": getattr(source, "trace_id", ""),
                "trace_meta": dict(getattr(source, "_trace_meta", {}) or {}),
            }
        )
        callback = getattr(source, "_trace_meta", {}).get("on_playback_begin")
        if callable(callback):
            callback()
        return True

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        self.tts_play_calls.append(
            {
                "chat_id": chat_id,
                "audio_path": audio_path,
                "kwargs": kwargs,
            }
        )
        return SendResult(success=True, message_id="tts-play-1")


class FakeAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback("tool.started", "terminal", "pwd", {})
        time.sleep(0.35)
        self.tool_progress_callback("tool.started", "browser_navigate", "https://example.com", {})
        time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class LongPreviewAgent:
    """Agent that emits a tool call with a very long preview string."""
    LONG_CMD = "cd /home/teknium/.hermes/hermes-agent/.worktrees/hermes-d8860339 && source .venv/bin/activate && python -m pytest tests/gateway/test_run_progress_topics.py -n0 -q"

    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        self.tool_progress_callback("tool.started", "terminal", self.LONG_CMD, {})
        time.sleep(0.35)
        return {
            "final_response": "done",
            "messages": [],
            "api_calls": 1,
        }


class SilentFinalAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        return {
            "final_response": "[SILENT]",
            "messages": [{"role": "assistant", "content": "[SILENT]"}],
            "api_calls": 1,
        }


class VoiceAckAgent:
    def __init__(self, **kwargs):
        self.tool_progress_callback = kwargs.get("tool_progress_callback")
        self.message_callback = kwargs.get("message_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.message_callback:
            self.message_callback("Still checking that now.")
        return {
            "final_response": "[SILENT]",
            "messages": [{"role": "assistant", "content": "[SILENT]"}],
            "api_calls": 1,
        }


class VoiceMediaSidebandAgent:
    def __init__(self, **kwargs):
        self.media_message_callback = kwargs.get("media_message_callback")
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None):
        if self.media_message_callback:
            self.media_message_callback([("/tmp/sideband.ogg", True)])
        return {
            "final_response": "[SILENT]",
            "messages": [{"role": "assistant", "content": "[SILENT]"}],
            "api_calls": 1,
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    GatewayRunner = gateway_run.GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    runner._running_agents = {}
    runner._pending_hidden_turns = {}
    runner._interactive_timing_state = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    return runner


def test_gateway_silent_marker_normalization():
    gateway_run = importlib.import_module("gateway.run")

    assert gateway_run._is_gateway_silent_response("[SILENT]") is True
    assert gateway_run._is_gateway_silent_response("  `[SILENT]`  ") is True
    assert gateway_run._is_gateway_silent_response("**[SILENT]**") is True
    assert gateway_run._is_gateway_silent_response("[SILENT] thanks") is False


@pytest.mark.asyncio
async def test_run_agent_progress_stays_in_originating_topic(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        thread_id="17585",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key="agent:main:telegram:group:-1001:17585",
    )

    assert result["final_response"] == "done"
    assert adapter.sent == [
        {
            "chat_id": "-1001",
            "content": '💻 terminal: "pwd"',
            "reply_to": None,
            "metadata": {"thread_id": "17585"},
        }
    ]
    assert adapter.edits
    assert all(call["metadata"] == {"thread_id": "17585"} for call in adapter.typing)


@pytest.mark.asyncio
async def test_run_agent_progress_does_not_use_event_message_id_for_telegram_dm(monkeypatch, tmp_path):
    """Telegram DM progress must not reuse event message id as thread metadata."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-2",
        session_key="agent:main:telegram:dm:12345",
        event_message_id="777",
    )

    assert result["final_response"] == "done"
    assert adapter.sent
    assert adapter.sent[0]["metadata"] is None
    assert all(call["metadata"] is None for call in adapter.typing)


def test_build_interactive_timing_guidance_mentions_recent_slow_tools():
    runner = _make_runner(ProgressCaptureAdapter())

    turn = runner._start_interactive_turn(
        "sess-key",
        platform_name="telegram",
        chat_id="123",
        message_type="text",
        is_voice=False,
    )
    runner._mark_turn_first_visible_output(turn, "sideband_text")
    runner._mark_turn_response_ready(turn, 4.2)
    runner._record_turn_tool_timing(turn, "web_search", 2.3)
    runner._record_turn_tool_timing(turn, "send_user_message", 0.1)
    runner._finalize_interactive_turn(
        "sess-key",
        turn,
        delivery_attempted=True,
        delivery_succeeded=True,
    )

    guidance = runner._build_interactive_timing_guidance(
        "sess-key",
        platform_name="telegram",
        is_voice=False,
    )

    assert guidance.startswith("[Timing: ")
    assert "goal=reactive" in guidance
    assert "prev_visible=" in guidance
    assert "slow=web_search:2.3s" in guidance
    assert "if_slow=send_user_message_once" in guidance


def test_build_interactive_timing_guidance_mentions_voice_metrics():
    runner = _make_runner(ProgressCaptureAdapter(platform=Platform.DISCORD))

    turn = runner._start_interactive_turn(
        "voice-sess",
        platform_name="discord",
        chat_id="123",
        message_type="voice",
        is_voice=True,
        asr_seconds=0.4,
    )
    runner._mark_turn_first_audible_output(turn, "final_voice")
    runner._mark_turn_response_ready(turn, 6.8)
    runner._finalize_interactive_turn(
        "voice-sess",
        turn,
        delivery_attempted=False,
        delivery_succeeded=False,
    )

    guidance = runner._build_interactive_timing_guidance(
        "voice-sess",
        platform_name="discord",
        is_voice=True,
    )

    assert "voice_prev_asr=0.4s" in guidance
    assert "voice_prev_audible=" in guidance
    assert "live_voice_ack=spoken" in guidance


def test_build_interactive_timing_guidance_hydrates_from_persisted_history():
    runner = _make_runner(ProgressCaptureAdapter(platform=Platform.DISCORD))
    history = [
        {
            "role": "assistant",
            "content": "Done",
            "timing_metadata": {
                "type": "interactive_turn_timing",
                "version": 1,
                "turn_id": "turn_hist_1",
                "platform": "discord",
                "chat_id": "123",
                "message_type": "voice",
                "is_voice": True,
                "asr_seconds": 0.31,
                "since_last_turn_seconds": 5.2,
                "first_visible_output_seconds": 1.1,
                "first_visible_output_kind": "sideband_text",
                "first_audible_output_seconds": 2.4,
                "first_audible_output_kind": "ack_voice",
                "response_ready_seconds": 6.0,
                "final_turn_seconds": 7.3,
                "sideband_updates": 1,
                "tool_timings": [{"name": "browser_navigate", "seconds": 2.2}],
                "slow_tools": [{"name": "browser_navigate", "seconds": 2.2}],
                "delivery_attempted": True,
                "delivery_succeeded": True,
            },
        }
    ]

    guidance = runner._build_interactive_timing_guidance(
        "voice-sess-history",
        platform_name="discord",
        is_voice=True,
        history=history,
    )

    assert "prev_final=7.3s" in guidance
    assert "voice_prev_asr=0.3s" in guidance
    assert "slow=browser_navigate:2.2s" in guidance


@pytest.mark.asyncio
async def test_run_agent_progress_uses_event_message_id_for_slack_dm(monkeypatch, tmp_path):
    """Slack DM progress should keep event ts fallback threading."""
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.SLACK)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="D123",
        chat_type="dm",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-3",
        session_key="agent:main:slack:dm:D123",
        event_message_id="1234567890.000001",
    )

    assert result["final_response"] == "done"
    assert adapter.sent
    assert adapter.sent[0]["metadata"] == {"thread_id": "1234567890.000001"}
    assert all(call["metadata"] == {"thread_id": "1234567890.000001"} for call in adapter.typing)


@pytest.mark.asyncio
async def test_run_agent_progress_uses_discrete_messages_for_discord(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="discord-chan",
        chat_type="group",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-discord-progress",
        session_key="agent:main:discord:group:discord-chan",
    )

    assert result["final_response"] == "done"
    assert [call["content"] for call in adapter.sent] == [
        '💻 terminal: "pwd"',
        '⚙️ browser_navigate: "https://example.com"',
    ]
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_run_agent_silent_marker_returns_empty_final_response(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = SilentFinalAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = ProgressCaptureAdapter(platform=Platform.DISCORD)
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="1493945387470422257",
        chat_type="group",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="test silent",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-silent",
        session_key="agent:main:discord:group:1493945387470422257",
    )

    assert result["final_response"] == ""
    assert result["messages"] == []


@pytest.mark.asyncio
async def test_run_agent_send_user_message_plays_spoken_ack_in_live_discord_vc(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = VoiceAckAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = DiscordVoiceProgressAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    fake_tts_tool = types.ModuleType("tools.tts_tool")
    fake_tts_tool._load_tts_config = lambda: {"tts": {"provider": "qwen3"}}
    fake_tts_tool._get_provider = lambda cfg: "qwen3"
    fake_tts_tool._preprocess_tts_text = lambda text, language=None: text.strip()
    fake_tts_tool._resolve_qwen3_config = lambda cfg, lang: {
        "base_url": "http://tts.local",
        "voice": "cfg_voice",
        "instruct": "cfg_instruct",
    }
    monkeypatch.setitem(sys.modules, "tools.tts_tool", fake_tts_tool)

    pcm_requests = []

    class FakeStreamingPCMAudioSource:
        def __init__(self, trace_id="", trace_meta=None):
            self.trace_id = trace_id
            self._trace_meta = trace_meta or {}
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

    def fake_feed_pcm_stream_sync(url, source, timeout):
        pcm_requests.append(
            {
                "url": url,
                "trace_id": source.trace_id,
                "trace_meta": dict(source._trace_meta),
                "timeout": timeout,
            }
        )

    fake_discord_platform = types.ModuleType("gateway.platforms.discord")
    fake_discord_platform.StreamingPCMAudioSource = FakeStreamingPCMAudioSource
    fake_discord_platform.feed_pcm_stream_sync = fake_feed_pcm_stream_sync
    monkeypatch.setitem(sys.modules, "gateway.platforms.discord", fake_discord_platform)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="voice-chat",
        chat_type="group",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-voice-ack",
        session_key="agent:main:discord:group:voice-chat",
    )
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert result["final_response"] == ""
    assert adapter.sent
    assert adapter.sent[0]["content"] == "💬 Still checking that now."
    assert len(adapter.play_calls) == 1
    assert adapter.play_calls[0]["guild_id"] == 111
    assert adapter.play_calls[0]["trace_id"].endswith("_ack") or adapter.play_calls[0]["trace_id"].startswith("ack_")
    assert adapter.play_calls[0]["trace_meta"]["turn_kind"] == "ack"
    assert adapter.play_calls[0]["trace_meta"]["chat_id"] == "voice-chat"
    assert len(pcm_requests) == 1
    assert "text=Still+checking+that+now." in pcm_requests[0]["url"]
    assert "voice=vc_test_voice" in pcm_requests[0]["url"]
    assert "instruct=warm+and+steady" in pcm_requests[0]["url"]
    assert pcm_requests[0]["trace_meta"]["turn_kind"] == "ack"


@pytest.mark.asyncio
async def test_run_agent_media_sideband_prefers_play_tts_in_live_discord_vc(monkeypatch, tmp_path):
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = VoiceMediaSidebandAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    adapter = DiscordVoiceProgressAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="voice-chat",
        chat_type="group",
        thread_id=None,
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-voice-media-sideband",
        session_key="agent:main:discord:group:voice-chat",
    )

    await asyncio.sleep(0)

    assert result["final_response"] == ""
    assert adapter.tts_play_calls
    assert adapter.tts_play_calls[0]["chat_id"] == "voice-chat"
    assert adapter.tts_play_calls[0]["audio_path"] == "/tmp/sideband.ogg"


@pytest.mark.asyncio
async def test_deliver_media_from_response_prefers_play_tts_for_audio(monkeypatch, tmp_path):
    adapter = DiscordVoiceProgressAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="voice-chat",
        chat_type="group",
        thread_id=None,
    )
    event = SimpleNamespace(source=source)

    await runner._deliver_media_from_response(
        "[[audio_as_voice]]\nMEDIA:/tmp/final-reply.ogg",
        event,
        adapter,
    )

    assert adapter.tts_play_calls
    assert adapter.tts_play_calls[0]["chat_id"] == "voice-chat"
    assert adapter.tts_play_calls[0]["audio_path"] == "/tmp/final-reply.ogg"


# ---------------------------------------------------------------------------
# Preview truncation tests (all/new mode respects tool_preview_length)
# ---------------------------------------------------------------------------


def _run_long_preview_helper(monkeypatch, tmp_path, preview_length=0):
    """Shared setup for long-preview truncation tests.

    Returns (adapter, result) after running the agent with LongPreviewAgent.
    ``preview_length`` controls display.tool_preview_length in the config file
    that _run_agent reads — so the gateway picks it up the same way production does.
    """
    import asyncio
    import yaml

    monkeypatch.setenv("HERMES_TOOL_PROGRESS_MODE", "all")

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = LongPreviewAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    # Write config.yaml so _run_agent picks up tool_preview_length
    config = {"display": {"tool_preview_length": preview_length}}
    (tmp_path / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")

    adapter = ProgressCaptureAdapter()
    runner = _make_runner(adapter)
    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="12345",
        chat_type="dm",
        thread_id=None,
    )

    result = asyncio.get_event_loop().run_until_complete(
        runner._run_agent(
            message="hello",
            context_prompt="",
            history=[],
            source=source,
            session_id="sess-trunc",
            session_key="agent:main:telegram:dm:12345",
        )
    )
    return adapter, result


def test_all_mode_default_truncation_40_chars(monkeypatch, tmp_path):
    """When tool_preview_length is 0 (default), all/new mode truncates to 40 chars."""
    adapter, result = _run_long_preview_helper(monkeypatch, tmp_path, preview_length=0)
    assert result["final_response"] == "done"
    assert adapter.sent
    content = adapter.sent[0]["content"]
    # The long command should be truncated — total preview <= 40 chars
    assert "..." in content
    # Extract the preview part between quotes
    import re
    match = re.search(r'"(.+)"', content)
    assert match, f"No quoted preview found in: {content}"
    preview_text = match.group(1)
    assert len(preview_text) <= 40, f"Preview too long ({len(preview_text)}): {preview_text}"


def test_all_mode_respects_custom_preview_length(monkeypatch, tmp_path):
    """When tool_preview_length is explicitly set (e.g. 120), all/new mode uses that."""
    adapter, result = _run_long_preview_helper(monkeypatch, tmp_path, preview_length=120)
    assert result["final_response"] == "done"
    assert adapter.sent
    content = adapter.sent[0]["content"]
    # With 120-char cap, the command (165 chars) should still be truncated but longer
    import re
    match = re.search(r'"(.+)"', content)
    assert match, f"No quoted preview found in: {content}"
    preview_text = match.group(1)
    # Should be longer than the 40-char default
    assert len(preview_text) > 40, f"Preview suspiciously short ({len(preview_text)}): {preview_text}"
    # But still capped at 120
    assert len(preview_text) <= 120, f"Preview too long ({len(preview_text)}): {preview_text}"


def test_all_mode_no_truncation_when_preview_fits(monkeypatch, tmp_path):
    """Short previews (under the cap) are not truncated."""
    # Set a generous cap — the LongPreviewAgent's command is ~165 chars
    adapter, result = _run_long_preview_helper(monkeypatch, tmp_path, preview_length=200)
    assert result["final_response"] == "done"
    assert adapter.sent
    content = adapter.sent[0]["content"]
    # With a 200-char cap, the 165-char command should NOT be truncated
    assert "..." not in content, f"Preview was truncated when it shouldn't be: {content}"
