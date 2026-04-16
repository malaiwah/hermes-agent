from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            model="qwen35-397b-nothink",
            provider="custom",
            base_url="http://hermes-litellm:4000/v1",
            api_key="litellm-secret",
            api_mode="chat_completions",
            extra_headers={"X-Test": "1"},
            platform="discord",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


def test_background_review_inherits_explicit_runtime_credentials():
    agent = _make_agent()
    captured = {}

    class FakeReviewAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self._session_messages = []
            self.client = None

        def run_conversation(self, user_message, conversation_history):
            captured["user_message"] = user_message
            captured["conversation_history"] = conversation_history

    class ImmediateThread:
        def __init__(self, target=None, **_kwargs):
            self._target = target

        def start(self):
            self._target()

    with patch("run_agent.AIAgent", FakeReviewAgent), \
         patch("run_agent.threading.Thread", ImmediateThread):
        agent._spawn_background_review(
            messages_snapshot=[{"role": "user", "content": "hello"}],
            review_memory=True,
            review_skills=False,
        )

    assert captured["model"] == "qwen35-397b-nothink"
    assert captured["provider"] == "custom"
    assert captured["base_url"] == "http://hermes-litellm:4000/v1"
    assert captured["api_key"] == "litellm-secret"
    assert captured["api_mode"] == "chat_completions"
    assert captured["extra_headers"] == {"X-Test": "1"}
    assert captured["platform"] == "discord"
    assert captured["quiet_mode"] is True
    assert captured["conversation_history"] == [{"role": "user", "content": "hello"}]
