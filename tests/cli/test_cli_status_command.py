from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-sonnet-4-20250514"
    cli_obj.provider = "anthropic"
    cli_obj.api_mode = "chat_completions"
    cli_obj.reasoning_config = {"enabled": True, "effort": "high"}
    cli_obj.session_id = "20260412_104559_e1760b66"
    cli_obj.session_start = datetime.now() - timedelta(minutes=5)
    cli_obj.agent = None
    cli_obj._agent_running = False
    cli_obj._pending_input = MagicMock()
    cli_obj._pending_input.qsize.return_value = 0
    cli_obj._session_db = MagicMock()
    cli_obj._session_db.get_session_title.return_value = None
    cli_obj._session_db.get_session_token_totals.return_value = None
    cli_obj._session_db.get_session.return_value = None
    cli_obj._session_db.get_session_last_active.return_value = None
    cli_obj._pending_title = None
    cli_obj.console = MagicMock()
    cli_obj.conversation_history = []
    return cli_obj


def test_process_command_dispatches_status():
    cli_obj = _make_cli()
    cli_obj._show_session_status = MagicMock()

    result = cli_obj.process_command("/status")

    assert result is True
    cli_obj._show_session_status.assert_called_once_with()


def test_process_command_status_handles_lightweight_cli_without_session_db(capsys):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-sonnet-4-20250514"
    cli_obj.provider = "auto"
    cli_obj.api_mode = "chat_completions"
    cli_obj.reasoning_config = None
    cli_obj.agent = None
    cli_obj._agent_running = False
    cli_obj.session_start = datetime.now() - timedelta(minutes=1)
    cli_obj._pending_title = None
    cli_obj.console = MagicMock()

    class _Queue:
        def qsize(self):
            return 0

    cli_obj._pending_input = _Queue()

    result = cli_obj.process_command("/status")
    output = capsys.readouterr().out

    assert result is True
    assert "Hermes Agent v" in output


def test_show_session_status_uses_live_agent(capsys):
    cli_obj = _make_cli()
    cli_obj._agent_running = True
    cli_obj._pending_input.qsize.return_value = 2
    cli_obj.agent = SimpleNamespace(
        model="openai/gpt-5.4",
        provider="openai",
        api_mode="codex_responses",
        base_url="https://api.openai.com/v1",
        session_input_tokens=25_000,
        session_output_tokens=610,
        session_cache_read_tokens=25_000,
        session_cache_write_tokens=0,
        session_reasoning_tokens=120,
        session_total_tokens=50_730,
        session_estimated_cost_usd=0.0,
        session_cost_status="estimated",
        context_compressor=SimpleNamespace(
            last_prompt_tokens=28_000,
            context_length=500_000,
            compression_count=0,
        ),
    )

    cli_obj._show_session_status()
    output = capsys.readouterr().out

    assert "Hermes Agent v" in output
    assert "Model: openai/gpt-5.4 · Provider: openai" in output
    assert "Usage: 25,000 in · 610 out · 50,730 total · Cost: $0.0000 est." in output
    assert "Cache: 25,000 read · 0 write · 50% hit · 120 reasoning" in output
    assert "Context: 28,000 / 500,000 (6%) · Compactions: 0" in output
    assert "Runtime: Responses · Reasoning high · CLI interactive" in output
    assert "Queue: depth 2 · State: running" in output


def test_show_session_status_falls_back_to_persisted_session(capsys):
    cli_obj = _make_cli()
    cli_obj._session_db.get_session_title.return_value = "Saved Session"
    cli_obj._session_db.get_session_token_totals.return_value = {
        "input_tokens": 1000,
        "output_tokens": 250,
        "cache_read_tokens": 500,
        "cache_write_tokens": 50,
        "reasoning_tokens": 20,
        "total_tokens": 1820,
    }
    cli_obj._session_db.get_session.return_value = {
        "model": "anthropic/claude-opus-4.6",
        "billing_provider": "anthropic",
        "billing_mode": "anthropic_messages",
        "estimated_cost_usd": 1.25,
        "cost_status": "estimated",
    }
    cli_obj._session_db.get_session_last_active.return_value = (
        datetime.now() - timedelta(minutes=7)
    ).timestamp()

    cli_obj._show_session_status()
    output = capsys.readouterr().out

    assert "Model: anthropic/claude-opus-4.6 · Provider: anthropic" in output
    assert "Usage: 1,000 in · 250 out · 1,820 total · Cost: $1.2500 est." in output
    assert "Cache: 500 read · 50 write · 33% hit · 20 reasoning" in output
    assert "Context:" not in output
    assert "Title: Saved Session" in output
    assert "Runtime: Anthropic Messages · Reasoning high · CLI interactive" in output
    assert "Queue: depth 0 · State: idle" in output


def test_show_session_status_uses_pending_title_before_first_persist(capsys):
    cli_obj = _make_cli()
    cli_obj._pending_title = "Queued Session Title"

    cli_obj._show_session_status()
    output = capsys.readouterr().out

    assert "Title: Queued Session Title" in output
