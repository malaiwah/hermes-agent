"""Tests for hermes_cli.status_format shared helpers."""

from datetime import datetime, timedelta

from hermes_cli.status_format import (
    format_api_mode_label,
    format_reasoning_effort_label,
    format_status_cost,
    format_status_relative_time,
    safe_status_float,
    safe_status_int,
)


class TestFormatStatusRelativeTime:
    def test_none_returns_unknown(self):
        assert format_status_relative_time(None) == "unknown"

    def test_just_now(self):
        assert format_status_relative_time(datetime.now()) == "just now"

    def test_minutes_ago(self):
        ts = datetime.now() - timedelta(minutes=5)
        assert format_status_relative_time(ts) == "5m ago"

    def test_hours_ago(self):
        ts = datetime.now() - timedelta(hours=3)
        assert format_status_relative_time(ts) == "3h ago"

    def test_days_ago(self):
        ts = datetime.now() - timedelta(days=2)
        assert format_status_relative_time(ts) == "2d ago"


class TestFormatStatusCost:
    def test_included(self):
        assert format_status_cost(None, "included") == "included"

    def test_none_amount_unknown_status(self):
        assert format_status_cost(None, "") is None
        assert format_status_cost(None, "unknown") is None

    def test_actual_amount(self):
        assert format_status_cost(1.5, "actual") == "$1.5000"

    def test_estimated_amount(self):
        assert format_status_cost(0.25, "estimated") == "$0.2500 est."

    def test_custom_status(self):
        assert format_status_cost(0.1, "cached") == "$0.1000 cached"

    def test_none_amount_custom_status(self):
        assert format_status_cost(None, "pending") == "pending"


class TestFormatReasoningEffortLabel:
    def test_none_config(self):
        assert format_reasoning_effort_label(None) == "medium"

    def test_disabled(self):
        assert format_reasoning_effort_label({"enabled": False}) == "none"

    def test_high_effort(self):
        assert format_reasoning_effort_label({"enabled": True, "effort": "high"}) == "high"

    def test_missing_effort_key(self):
        assert format_reasoning_effort_label({"enabled": True}) == "medium"


class TestFormatApiModeLabel:
    def test_none(self):
        assert format_api_mode_label(None) is None

    def test_empty(self):
        assert format_api_mode_label("") is None

    def test_known_modes(self):
        assert format_api_mode_label("chat_completions") == "Chat Completions"
        assert format_api_mode_label("codex_responses") == "Responses"
        assert format_api_mode_label("anthropic_messages") == "Anthropic Messages"

    def test_unknown_mode(self):
        assert format_api_mode_label("custom_mode") == "Custom Mode"


class TestSafeStatusInt:
    def test_none_returns_default(self):
        assert safe_status_int(None) == 0
        assert safe_status_int(None, default=5) == 5

    def test_valid_int(self):
        assert safe_status_int(42) == 42

    def test_string_number(self):
        assert safe_status_int("123") == 123

    def test_invalid_returns_default(self):
        assert safe_status_int("abc") == 0


class TestSafeStatusFloat:
    def test_none_returns_none(self):
        assert safe_status_float(None) is None

    def test_valid_float(self):
        assert safe_status_float(1.5) == 1.5

    def test_string_number(self):
        assert safe_status_float("2.5") == 2.5

    def test_invalid_returns_none(self):
        assert safe_status_float("abc") is None
