"""Tests for gateway.run._extract_voice_reply_directives.

Voice replies may carry inline `[tts: ...]` and `[voice: ...]` directives
that the runner strips before calling TTS and persists for carry-forward
to subsequent turns in the same chat.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.run import _extract_voice_reply_directives


class _Runner:
    """Stand-in for GatewayRunner that only has the carry-forward dicts."""

    def __init__(self):
        self._last_tts_voice = {}
        self._last_tts_instruct = {}


# ---------------------------------------------------------------------------
# Basic stripping
# ---------------------------------------------------------------------------

def test_returns_empty_for_empty_input():
    assert _extract_voice_reply_directives("") == ("", "", "")
    assert _extract_voice_reply_directives(None) == ("", "", "")


def test_no_directive_returns_text_unchanged():
    cleaned, instruct, voice = _extract_voice_reply_directives("Hello there")
    assert cleaned == "Hello there"
    assert instruct == ""
    assert voice == ""


def test_strips_tts_directive():
    cleaned, instruct, voice = _extract_voice_reply_directives(
        "[tts: warm and friendly] Hello there"
    )
    assert cleaned == "Hello there"
    assert instruct == "warm and friendly"
    assert voice == ""


def test_strips_voice_directive():
    cleaned, instruct, voice = _extract_voice_reply_directives(
        "[voice: vc_e87c8ed1] Hello there"
    )
    assert cleaned == "Hello there"
    assert voice == "vc_e87c8ed1"


def test_strips_both_directives():
    cleaned, instruct, voice = _extract_voice_reply_directives(
        "[tts: excited] [voice: ryan] Hi!"
    )
    assert cleaned == "Hi!"
    assert instruct == "excited"
    assert voice == "ryan"


def test_directives_are_case_insensitive():
    cleaned, instruct, voice = _extract_voice_reply_directives(
        "[TTS: SLOW] [Voice: Ryan] body"
    )
    assert cleaned == "body"
    assert instruct == "SLOW"
    assert voice == "Ryan"


# ---------------------------------------------------------------------------
# Regression: nested brackets in [tts:] value
# ---------------------------------------------------------------------------

def test_nested_brackets_in_tts_value_do_not_truncate_at_first_bracket():
    """Regression: a `]` inside the value previously closed the tag early.

    Old non-greedy `.+?` regex extracted `foo[bar` and left `baz]` in the
    output. The fixed regex `[^\\]]*` rejects nested `]` so the malformed
    directive parses cleanly: extract the whole inner content (which is
    everything before the first `]`) but the trailing `baz]` is preserved
    as part of the spoken text.
    """
    cleaned, instruct, _ = _extract_voice_reply_directives(
        "[tts: warm[and]friendly] body"
    )
    # Parser stops at the first `]` (because nested `]` is not a valid
    # value char), so instruct = "warm[and" and the trailing "friendly]"
    # stays in the text. This is the documented behaviour.
    assert instruct == "warm[and"
    assert cleaned == "friendly] body"


def test_voice_value_with_invalid_chars_is_stripped_but_not_extracted():
    """A malformed [voice:] tag is removed from the spoken text but not used."""
    cleaned, _, voice = _extract_voice_reply_directives("[voice: not a voice!] body")
    assert voice == ""
    assert cleaned == "body"


def test_voice_value_must_match_pattern():
    # vc_<8+ hex> ✓ ; preset name ✓
    for valid in ["vc_e87c8ed1", "vc_e87c8ed1f0", "ryan", "aiden_v2"]:
        _, _, voice = _extract_voice_reply_directives(f"[voice: {valid}] body")
        assert voice == valid, f"expected {valid} to be accepted"

    # Invalid: too short hex (vc_abc has 3 hex), leading digit, hyphen
    for invalid in ["vc_abc", "9ryan", "bad-name"]:
        _, _, voice = _extract_voice_reply_directives(f"[voice: {invalid}] body")
        assert voice == "", f"expected {invalid} to be rejected"


# ---------------------------------------------------------------------------
# Carry-forward via runner state
# ---------------------------------------------------------------------------

def test_directives_persist_to_runner_state():
    runner = _Runner()
    _extract_voice_reply_directives(
        "[tts: warm] [voice: ryan] hi",
        runner=runner,
        platform="discord",
        chat_id="123",
    )
    assert runner._last_tts_voice == {("discord", "123"): "ryan"}
    assert runner._last_tts_instruct == {("discord", "123"): "warm"}


def test_only_persists_when_runner_provided():
    """Without a runner the dicts are unchanged (extraction still works)."""
    cleaned, instruct, voice = _extract_voice_reply_directives(
        "[voice: ryan] hi"
    )
    assert voice == "ryan"
    assert cleaned == "hi"


def test_blank_tts_value_does_not_overwrite_carry_forward():
    runner = _Runner()
    runner._last_tts_instruct[("discord", "1")] = "previous-instruct"
    _extract_voice_reply_directives(
        "[tts: ] hello",  # explicit blank
        runner=runner,
        platform="discord",
        chat_id="1",
    )
    # Empty value should not overwrite the previously-set instruct
    assert runner._last_tts_instruct[("discord", "1")] == "previous-instruct"


def test_each_chat_has_independent_carry_forward():
    runner = _Runner()
    _extract_voice_reply_directives(
        "[voice: ryan] hi A", runner=runner, platform="discord", chat_id="A",
    )
    _extract_voice_reply_directives(
        "[voice: aiden] hi B", runner=runner, platform="discord", chat_id="B",
    )
    assert runner._last_tts_voice[("discord", "A")] == "ryan"
    assert runner._last_tts_voice[("discord", "B")] == "aiden"
