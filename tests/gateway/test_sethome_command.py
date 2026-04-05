"""Tests for gateway /sethome config persistence fallback."""

import errno
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/sethome", platform=Platform.TELEGRAM, user_id="12345", chat_id="67890"):
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
        chat_name="Ops",
    )
    return MessageEvent(text=text, source=source)


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


@pytest.mark.asyncio
async def test_sethome_reports_session_only_when_config_is_locked(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    def locked_write(*args, **kwargs):
        raise OSError(errno.EBUSY, "Device or resource busy")

    monkeypatch.setattr("hermes_cli.config.save_env_value", locked_write)

    runner = _make_runner()
    result = await runner._handle_set_home_command(_make_event())

    assert "for this running gateway only" in result
    assert "could not persist to .env" in result
