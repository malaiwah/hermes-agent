from unittest.mock import patch

import pytest

from gateway.builtin_hooks import boot_md


def test_run_boot_agent_uses_gateway_runtime():
    captured = {}
    pool = object()

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run_conversation(self, prompt):
            captured["prompt"] = prompt
            return {"final_response": "[SILENT]"}

    runtime_kwargs = {
        "api_key": "sk-test",
        "base_url": "http://example.test/v1",
        "provider": "openai",
        "api_mode": "chat_completions",
        "command": "copilot-acp",
        "args": ["serve", "--stdio"],
        "credential_pool": pool,
    }

    with patch("gateway.run._resolve_gateway_model", return_value="gpt-test"), patch(
        "gateway.run._resolve_runtime_agent_kwargs", return_value=runtime_kwargs
    ), patch("run_agent.AIAgent", FakeAgent):
        boot_md._run_boot_agent("Check the startup state.")

    assert captured["kwargs"]["model"] == "gpt-test"
    assert captured["kwargs"]["platform"] == "gateway"
    assert captured["kwargs"]["api_key"] == "sk-test"
    assert captured["kwargs"]["base_url"] == "http://example.test/v1"
    assert captured["kwargs"]["provider"] == "openai"
    assert captured["kwargs"]["api_mode"] == "chat_completions"
    assert captured["kwargs"]["command"] == "copilot-acp"
    assert captured["kwargs"]["args"] == ["serve", "--stdio"]
    assert captured["kwargs"]["credential_pool"] is pool
    assert "BOOT.md" in captured["prompt"]
    assert "Check the startup state." in captured["prompt"]


@pytest.mark.asyncio
async def test_handle_skips_missing_boot_file(tmp_path):
    missing = tmp_path / "BOOT.md"
    with patch("gateway.builtin_hooks.boot_md.BOOT_FILE", missing):
        assert await boot_md.handle("gateway:startup", {}) is None
