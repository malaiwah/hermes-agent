import logging
from array import array
from unittest.mock import MagicMock, patch

import pytest

from gateway.platforms.discord import VoiceReceiver
from gateway.platforms.discord_vad import DiscordVCVADConfig, DiscordVCVADState


def _pcm_frame(value: int = 1000) -> bytes:
    # 20 ms of 48 kHz stereo int16 PCM = 960 stereo samples = 1920 int16 values
    return array("h", [value] * 1920).tobytes()


class _FakeVADStream:
    def __init__(self, probs):
        self._original = list(probs)
        self._probs = list(probs)

    def reset(self):
        self._probs = list(self._original)

    def process_chunk(self, _chunk):
        return self._probs.pop(0) if self._probs else 0.0


class _FakeVADModel:
    CHUNK_SAMPLES = 512

    def __init__(self, probs):
        import numpy as np

        self._np = np
        self._probs = list(probs)

    def new_stream(self):
        return _FakeVADStream(self._probs)


def _make_receiver(voice_config=None):
    mock_vc = MagicMock()
    mock_vc._connection.secret_key = [0] * 32
    mock_vc._connection.dave_session = None
    mock_vc._connection.ssrc = 9999
    mock_vc._connection.add_socket_listener = MagicMock()
    mock_vc._connection.remove_socket_listener = MagicMock()
    mock_vc._connection.hook = None
    mock_vc.guild.id = 111
    return VoiceReceiver(mock_vc, voice_config=voice_config)


def test_vad_speech_start_after_consecutive_positive_chunks():
    cfg = DiscordVCVADConfig(
        vad_min_speech_ms=64,
        vad_min_silence_ms=550,
        vad_start_prob=0.55,
        vad_end_prob=0.35,
        rms_fallback_threshold=0,
    )
    state = DiscordVCVADState(_FakeVADModel([0.8, 0.9]), cfg)
    now = 100.0

    events = []
    for idx in range(4):
        events.extend(state.push_pcm_frame(_pcm_frame(), now + (idx * 0.02), rms=400))

    assert state.speech_active is True
    assert any(event["type"] == "speech_start" for event in events)


def test_vad_brief_pause_does_not_finalize():
    cfg = DiscordVCVADConfig(
        vad_min_speech_ms=64,
        vad_min_silence_ms=550,
        vad_start_prob=0.55,
        vad_end_prob=0.35,
        rms_fallback_threshold=0,
    )
    state = DiscordVCVADState(_FakeVADModel([0.8, 0.9, 0.1]), cfg)
    now = 100.0

    for idx in range(6):
        state.push_pcm_frame(_pcm_frame(), now + (idx * 0.02), rms=400)

    assert state.speech_active is True
    assert state.silence_started_at is not None
    assert state.should_finalize(now + 0.20) is None


def test_vad_finalize_after_silence_window():
    cfg = DiscordVCVADConfig(
        vad_min_speech_ms=64,
        vad_min_silence_ms=550,
        vad_start_prob=0.55,
        vad_end_prob=0.35,
        rms_fallback_threshold=0,
    )
    state = DiscordVCVADState(_FakeVADModel([0.8, 0.9, 0.1]), cfg)
    now = 100.0

    for idx in range(6):
        state.push_pcm_frame(_pcm_frame(), now + (idx * 0.02), rms=400)

    assert state.should_finalize(now + 0.70) == "vad_silence"


def test_vad_finalize_on_max_utterance():
    cfg = DiscordVCVADConfig(
        vad_min_speech_ms=64,
        vad_min_silence_ms=550,
        vad_max_utterance_s=1.0,
        vad_start_prob=0.55,
        vad_end_prob=0.35,
        rms_fallback_threshold=0,
    )
    state = DiscordVCVADState(_FakeVADModel([0.8, 0.9, 0.9, 0.9]), cfg)
    now = 100.0

    for idx in range(8):
        state.push_pcm_frame(_pcm_frame(), now + (idx * 0.02), rms=400)

    assert state.should_finalize(now + 1.10) == "max_utterance"


def test_receiver_falls_back_to_rms_when_vad_init_fails(caplog):
    receiver = _make_receiver({"discord_vc": {"vad_enabled": True}})

    with patch("gateway.platforms.discord.SileroOnnxVAD", side_effect=RuntimeError("boom")):
        with caplog.at_level(logging.WARNING):
            receiver.start()

    assert receiver._vad_enabled is False
    assert "vc_vad_fallback" in caplog.text
