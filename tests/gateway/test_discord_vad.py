"""Tests for gateway.platforms.discord_vad — Silero VAD state machine.

Uses a fake VAD model that returns scripted speech probabilities so the
state-machine logic can be exercised without onnxruntime. Real numpy is
used because the [vad] extra always installs it. The lazy-download path is
exercised against a local file (no network)."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

import numpy as np
import pytest

from gateway.platforms import discord_vad
from gateway.platforms.discord_vad import (
    DiscordVCVADConfig,
    DiscordVCVADState,
)


# ---------------------------------------------------------------------------
# Fake VAD model — onnxruntime stand-in
# ---------------------------------------------------------------------------

class _FakeStream:
    def __init__(self, probs):
        self._probs = list(probs)

    def reset(self):
        pass

    def process_chunk(self, _chunk):
        if not self._probs:
            return 0.0
        return self._probs.pop(0)


class _FakeVADModel:
    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512
    CONTEXT_SAMPLES = 64

    def __init__(self, scripted_probs):
        self._np = np
        self._scripted = list(scripted_probs)

    def new_stream(self):
        return _FakeStream(self._scripted)


# A 32 ms frame at 48kHz stereo int16 is enough to produce ~512 mono samples
# at 16kHz, i.e. exactly one VAD chunk per push_pcm_frame call.
def _frame_bytes_32ms() -> bytes:
    # 48000 Hz × 0.032 s = 1536 stereo samples × 2 channels × 2 bytes = 6144
    return b"\x00" * 6144


# ---------------------------------------------------------------------------
# DiscordVCVADConfig — config parsing
# ---------------------------------------------------------------------------

class TestDiscordVCVADConfig:
    def test_defaults_disabled(self):
        cfg = DiscordVCVADConfig()
        assert cfg.vad_enabled is False
        assert cfg.vad_mode == "silero_hybrid"

    def test_from_voice_config_none(self):
        cfg = DiscordVCVADConfig.from_voice_config(None)
        assert cfg.vad_enabled is False

    def test_from_voice_config_enabled_and_overrides(self):
        cfg = DiscordVCVADConfig.from_voice_config({
            "discord_vc": {
                "vad_enabled": True,
                "vad_min_speech_ms": 400,
                "vad_min_silence_ms": 800,
                "vad_start_prob": 0.7,
            }
        })
        assert cfg.vad_enabled is True
        assert cfg.vad_min_speech_ms == 400
        assert cfg.vad_min_silence_ms == 800
        assert cfg.vad_start_prob == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# DiscordVCVADState — speech_start / speech_end / finalize
# ---------------------------------------------------------------------------

class TestDiscordVCVADState:
    def _make_state(self, probs, **cfg_overrides):
        cfg = DiscordVCVADConfig(
            vad_enabled=True,
            vad_min_speech_ms=cfg_overrides.get("vad_min_speech_ms", 50),
            vad_min_silence_ms=cfg_overrides.get("vad_min_silence_ms", 100),
            vad_start_prob=0.55,
            vad_end_prob=0.35,
            vad_max_utterance_s=cfg_overrides.get("vad_max_utterance_s", 5.0),
            rms_fallback_threshold=0,  # always feed VAD
        )
        model = _FakeVADModel(probs)
        return DiscordVCVADState(model=model, config=cfg)

    def test_speech_start_emitted_after_min_speech_window(self):
        # Each push_pcm_frame consumes 1 chunk == 32ms of audio.
        # vad_min_speech_ms=50 → need 2 consecutive high-prob chunks.
        state = self._make_state([0.9, 0.9, 0.9])
        events = []
        for i in range(3):
            events.extend(state.push_pcm_frame(_frame_bytes_32ms(), frame_time=float(i), rms=500))
        starts = [e for e in events if e["type"] == "speech_start"]
        assert len(starts) == 1
        assert state.speech_active is True

    def test_no_speech_start_below_threshold(self):
        state = self._make_state([0.2, 0.2, 0.2])
        events = []
        for i in range(3):
            events.extend(state.push_pcm_frame(_frame_bytes_32ms(), frame_time=float(i), rms=500))
        assert all(e["type"] != "speech_start" for e in events)
        assert state.speech_active is False

    def test_speech_end_candidate_after_low_probs(self):
        state = self._make_state([0.9, 0.9, 0.2, 0.2])
        for i in range(4):
            state.push_pcm_frame(_frame_bytes_32ms(), frame_time=float(i), rms=500)
        assert state.speech_active is True
        assert state.silence_started_at is not None

    def test_should_finalize_on_silence_window(self):
        cfg = DiscordVCVADConfig(
            vad_enabled=True,
            vad_min_speech_ms=50,
            vad_min_silence_ms=100,
            vad_start_prob=0.55,
            vad_end_prob=0.35,
            rms_fallback_threshold=0,
        )
        model = _FakeVADModel([0.9, 0.9, 0.2, 0.2])
        state = DiscordVCVADState(model=model, config=cfg)
        for i in range(4):
            state.push_pcm_frame(_frame_bytes_32ms(), frame_time=float(i), rms=500)
        # silence_started_at was set at frame_time≈3.0 (or earlier);
        # advance "now" past min_silence_ms to trigger finalize.
        assert state.should_finalize(now=state.silence_started_at + 1.0) == "vad_silence"

    def test_should_finalize_on_max_utterance(self):
        state = self._make_state([0.9, 0.9, 0.9, 0.9, 0.9, 0.9], vad_max_utterance_s=0.05)
        for i in range(6):
            state.push_pcm_frame(_frame_bytes_32ms(), frame_time=i * 0.01, rms=500)
        assert state.should_finalize(now=10.0) == "max_utterance"

    def test_should_finalize_returns_none_when_idle(self):
        state = self._make_state([])
        assert state.should_finalize(now=1.0) is None

    def test_reset_clears_state(self):
        state = self._make_state([0.9, 0.9])
        for i in range(2):
            state.push_pcm_frame(_frame_bytes_32ms(), frame_time=float(i), rms=500)
        state.reset()
        assert state.speech_active is False
        assert state.silence_started_at is None
        assert state.last_prob == 0.0


# ---------------------------------------------------------------------------
# Lazy ONNX download — verify cache, checksum behaviour
# ---------------------------------------------------------------------------

class TestEnsureSileroModel:
    def test_returns_cached_when_checksum_matches(self, tmp_path, monkeypatch):
        cache_dir = tmp_path
        cached = cache_dir / "hermes-agent" / "silero_vad.onnx"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"hello world")
        monkeypatch.setattr(discord_vad, "SILERO_VAD_SHA256",
                            hashlib.sha256(b"hello world").hexdigest())
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))

        result = discord_vad.ensure_silero_model()
        assert result == cached

    def test_redownloads_when_cache_corrupt(self, tmp_path, monkeypatch):
        cache_dir = tmp_path
        cached = cache_dir / "hermes-agent" / "silero_vad.onnx"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b"corrupt-bytes")
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))
        monkeypatch.setattr(discord_vad, "SILERO_VAD_SHA256",
                            hashlib.sha256(b"good-bytes").hexdigest())

        # Stub urlopen to deliver the "good" payload.
        class _FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __init__(self): self._payload = b"good-bytes"
            def read(self, _n):
                p, self._payload = self._payload, b""
                return p

        with patch("urllib.request.urlopen", return_value=_FakeResp()):
            result = discord_vad.ensure_silero_model()

        assert result.read_bytes() == b"good-bytes"

    def test_raises_on_checksum_mismatch_after_download(self, tmp_path, monkeypatch):
        cache_dir = tmp_path
        monkeypatch.setenv("XDG_CACHE_HOME", str(cache_dir))
        monkeypatch.setattr(discord_vad, "SILERO_VAD_SHA256",
                            hashlib.sha256(b"expected").hexdigest())

        class _FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def __init__(self): self._payload = b"WRONG"
            def read(self, _n):
                p, self._payload = self._payload, b""
                return p

        with patch("urllib.request.urlopen", return_value=_FakeResp()):
            with pytest.raises(RuntimeError, match="checksum mismatch"):
                discord_vad.ensure_silero_model()
