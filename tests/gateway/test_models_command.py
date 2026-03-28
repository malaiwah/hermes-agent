"""Tests for the /models slash command."""

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import yaml

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/models"):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="testuser",
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
    return runner


def _write_config(path, provider="openai", model="gpt-4o"):
    path.write_text(
        yaml.dump({"model": {"provider": provider, "model": model}}),
        encoding="utf-8",
    )


class TestModelsCommand:

    def test_lists_models_for_current_provider(self, tmp_path, monkeypatch):
        _write_config(tmp_path / "config.yaml", provider="openai", model="gpt-4o")
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

        with patch("hermes_cli.models.provider_model_ids", return_value=["gpt-4o", "gpt-4-turbo"]):
            result = asyncio.run(_make_runner()._handle_models_command(_make_event("/models")))

        assert "gpt-4o" in result
        assert "gpt-4-turbo" in result

    def test_marks_active_model(self, tmp_path, monkeypatch):
        _write_config(tmp_path / "config.yaml", provider="openai", model="gpt-4o")
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

        with patch("hermes_cli.models.provider_model_ids", return_value=["gpt-4o", "gpt-4-turbo"]):
            result = asyncio.run(_make_runner()._handle_models_command(_make_event("/models")))

        assert "← active" in result
        assert result.count("← active") == 1
        assert "`gpt-4o` ← active" in result

    def test_explicit_provider_arg(self, tmp_path, monkeypatch):
        _write_config(tmp_path / "config.yaml", provider="openai", model="gpt-4o")
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

        with patch("hermes_cli.models.provider_model_ids", return_value=["claude-opus-4-5", "claude-sonnet-4-5"]):
            result = asyncio.run(_make_runner()._handle_models_command(_make_event("/models anthropic")))

        assert "claude-opus-4-5" in result
        assert "claude-sonnet-4-5" in result
        # Active model belongs to openai, not anthropic — no marker expected
        assert "← active" not in result

    def test_truncates_long_list(self, tmp_path, monkeypatch):
        _write_config(tmp_path / "config.yaml", provider="openrouter", model="openai/gpt-4o")
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

        models = [f"model-{i}" for i in range(80)]
        with patch("hermes_cli.models.provider_model_ids", return_value=models):
            result = asyncio.run(_make_runner()._handle_models_command(_make_event("/models openrouter")))

        assert "model-0" in result
        assert "model-49" in result
        assert "model-50" not in result
        assert "30 more" in result

    def test_no_truncation_when_under_limit(self, tmp_path, monkeypatch):
        _write_config(tmp_path / "config.yaml", provider="anthropic", model="claude-sonnet-4-5")
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

        models = [f"claude-model-{i}" for i in range(5)]
        with patch("hermes_cli.models.provider_model_ids", return_value=models):
            result = asyncio.run(_make_runner()._handle_models_command(_make_event("/models anthropic")))

        assert "more" not in result
        for m in models:
            assert m in result

    def test_unknown_provider_returns_error(self, tmp_path, monkeypatch):
        _write_config(tmp_path / "config.yaml")
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

        with patch("hermes_cli.models.provider_model_ids", return_value=[]), \
             patch("hermes_cli.models.curated_models_for_provider", return_value=[]):
            result = asyncio.run(_make_runner()._handle_models_command(_make_event("/models nonexistentprovider")))

        assert "No model list available" in result

    def test_custom_named_provider(self, tmp_path, monkeypatch):
        _write_config(tmp_path / "config.yaml")
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

        custom_cfg = {
            "model": {"provider": "openai", "model": "gpt-4o"},
            "custom_providers": [
                {"name": "lmstudio", "base_url": "http://localhost:1234/v1", "api_key": ""}
            ],
        }
        with patch("hermes_cli.config.load_config", return_value=custom_cfg), \
             patch("hermes_cli.models.fetch_api_models", return_value=["qwen2.5-7b", "llama-3.2-3b"]):
            result = asyncio.run(_make_runner()._handle_models_command(_make_event("/models custom:lmstudio")))

        assert "qwen2.5-7b" in result
        assert "llama-3.2-3b" in result
        assert "lmstudio" in result

    def test_custom_named_provider_not_found(self, tmp_path, monkeypatch):
        _write_config(tmp_path / "config.yaml")
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

        custom_cfg = {"model": {}, "custom_providers": []}
        with patch("hermes_cli.config.load_config", return_value=custom_cfg):
            result = asyncio.run(_make_runner()._handle_models_command(_make_event("/models custom:unknown")))

        assert "No custom provider named" in result
        assert "unknown" in result

    def test_custom_provider_endpoint_unreachable(self, tmp_path, monkeypatch):
        _write_config(tmp_path / "config.yaml")
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)

        custom_cfg = {
            "model": {},
            "custom_providers": [
                {"name": "lmstudio", "base_url": "http://localhost:1234/v1", "api_key": ""}
            ],
        }
        with patch("hermes_cli.config.load_config", return_value=custom_cfg), \
             patch("hermes_cli.models.fetch_api_models", return_value=None):
            result = asyncio.run(_make_runner()._handle_models_command(_make_event("/models custom:lmstudio")))

        assert "Could not fetch" in result

    def test_models_is_dispatched_in_handle_message(self):
        import inspect
        source = inspect.getsource(gateway_run.GatewayRunner._handle_message)
        assert '"models"' in source
