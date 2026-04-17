"""Silero VAD support for Discord voice channels.

This module is opt-in. When ``voice.discord_vc.vad_enabled`` is true and
``onnxruntime`` + ``numpy`` are installed, ``DiscordVCVADState`` performs
streaming voice-activity detection on per-SSRC PCM buffers, allowing the
voice receiver to finalise utterances as soon as the speaker stops talking
rather than waiting for a fixed silence timer.

The Silero VAD ONNX model (~2.3 MB) is downloaded on first use to the user's
cache directory rather than bundled in the repository. Downloads are pinned
to a specific upstream commit and SHA-256 verified.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Pinned to snakers4/silero-vad commit bfdc019 (v6.2 model, 2025-11-06).
# Verified SHA-256 over the 2,327,524-byte ONNX file.
SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/bfdc019/"
    "src/silero_vad/data/silero_vad.onnx"
)
SILERO_VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SILERO_VAD_SIZE = 2_327_524


def _vad_cache_dir() -> Path:
    """Return the directory used to cache the Silero VAD ONNX model.

    Honours ``XDG_CACHE_HOME``; otherwise uses ``~/.cache``. The directory
    is created on demand.
    """
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    cache = Path(base) / "hermes-agent"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest() == expected


def ensure_silero_model() -> Path:
    """Return a path to a verified Silero VAD ONNX model, downloading if needed.

    Raises ``RuntimeError`` if the download fails or the SHA-256 does not match.
    """
    cache = _vad_cache_dir() / "silero_vad.onnx"
    if cache.exists() and _verify_sha256(cache, SILERO_VAD_SHA256):
        return cache

    if cache.exists():
        logger.warning("Cached silero_vad.onnx failed checksum, re-downloading")
        cache.unlink()

    logger.info("Downloading Silero VAD model (%d bytes) to %s", SILERO_VAD_SIZE, cache)
    from urllib.request import urlopen

    tmp_fd, tmp_path = tempfile.mkstemp(prefix="silero_vad_", suffix=".onnx", dir=str(cache.parent))
    tmp = Path(tmp_path)
    try:
        with os.fdopen(tmp_fd, "wb") as out, urlopen(SILERO_VAD_URL, timeout=60) as resp:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
        if not _verify_sha256(tmp, SILERO_VAD_SHA256):
            raise RuntimeError(
                f"Silero VAD checksum mismatch (downloaded from {SILERO_VAD_URL})"
            )
        tmp.replace(cache)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return cache


@dataclass
class DiscordVCVADConfig:
    """Tunable parameters for the Discord voice-channel VAD pipeline."""

    vad_enabled: bool = False
    vad_mode: str = "silero_hybrid"
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 550
    vad_speech_pad_ms: int = 150
    vad_start_prob: float = 0.55
    vad_end_prob: float = 0.35
    vad_max_utterance_s: float = 20.0
    rms_fallback_threshold: int = 200
    min_utterance_rms: int = 300

    @classmethod
    def from_voice_config(cls, voice_cfg: Optional[Dict[str, Any]]) -> "DiscordVCVADConfig":
        voice_cfg = voice_cfg or {}
        vc_cfg = voice_cfg.get("discord_vc") or {}
        return cls(
            vad_enabled=bool(vc_cfg.get("vad_enabled", False)),
            vad_mode=str(vc_cfg.get("vad_mode", "silero_hybrid") or "silero_hybrid"),
            vad_min_speech_ms=int(vc_cfg.get("vad_min_speech_ms", 250)),
            vad_min_silence_ms=int(vc_cfg.get("vad_min_silence_ms", 550)),
            vad_speech_pad_ms=int(vc_cfg.get("vad_speech_pad_ms", 150)),
            vad_start_prob=float(vc_cfg.get("vad_start_prob", 0.55)),
            vad_end_prob=float(vc_cfg.get("vad_end_prob", 0.35)),
            vad_max_utterance_s=float(vc_cfg.get("vad_max_utterance_s", 20.0)),
            rms_fallback_threshold=int(vc_cfg.get("rms_fallback_threshold", 200)),
            min_utterance_rms=int(vc_cfg.get("min_utterance_rms", 300)),
        )


class SileroOnnxVAD:
    """Thin ONNX wrapper around the official Silero VAD model."""

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = 512
    CONTEXT_SAMPLES = 64

    def __init__(self, model_path: Optional[Path] = None):
        import numpy as np
        import onnxruntime

        self._np = np
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        providers = (
            ["CPUExecutionProvider"]
            if "CPUExecutionProvider" in onnxruntime.get_available_providers()
            else None
        )
        path = str(model_path or ensure_silero_model())
        self._session = onnxruntime.InferenceSession(path, providers=providers, sess_options=opts)

    def new_stream(self) -> "SileroOnnxVADStream":
        return SileroOnnxVADStream(self)


class SileroOnnxVADStream:
    """Per-speaker streaming context for SileroOnnxVAD."""

    def __init__(self, model: SileroOnnxVAD):
        self._model = model
        self._np = model._np
        self.reset()

    def reset(self) -> None:
        self._state = self._np.zeros((2, 1, 128), dtype=self._np.float32)
        self._context = self._np.zeros((1, self._model.CONTEXT_SAMPLES), dtype=self._np.float32)

    def process_chunk(self, chunk: "Any") -> float:
        if chunk.shape[0] != self._model.CHUNK_SAMPLES:
            raise ValueError(
                f"Silero VAD expects {self._model.CHUNK_SAMPLES} samples, got {chunk.shape[0]}"
            )
        x = chunk.astype(self._np.float32)[None, :] / 32768.0
        x = self._np.concatenate([self._context, x], axis=1)
        inputs = {
            "input": x,
            "state": self._state,
            "sr": self._np.array(self._model.SAMPLE_RATE, dtype=self._np.int64),
        }
        out, state = self._model._session.run(None, inputs)
        self._state = state
        self._context = x[:, -self._model.CONTEXT_SAMPLES:]
        return float(out.reshape(-1)[0])


@dataclass
class DiscordVCVADState:
    """Per-SSRC streaming VAD state machine.

    Consumes 48kHz stereo int16 PCM frames (Discord native), downsamples to
    16kHz mono for Silero, and emits speech_start / speech_end_candidate
    events. ``should_finalize`` reports when the caller should treat the
    current utterance as complete (silence threshold reached or maximum
    utterance length exceeded).
    """

    model: SileroOnnxVAD
    config: DiscordVCVADConfig
    mono_buffer_16k: bytearray = field(default_factory=bytearray)
    pre_roll_pcm: bytearray = field(default_factory=bytearray)
    speech_active: bool = False
    speech_started_at: float = 0.0
    silence_started_at: Optional[float] = None
    last_speech_time: float = 0.0
    last_prob: float = 0.0
    speech_run_ms: float = 0.0

    PCM_BYTES_PER_SECOND = 48000 * 2 * 2
    VAD_FRAME_MS = 32.0  # 512 samples at 16kHz

    def __post_init__(self) -> None:
        self.stream = self.model.new_stream()

    def _pcm48k_stereo_to_mono16_bytes(self, pcm: bytes) -> bytes:
        np = self.model._np
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return b""
        mono48 = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
        # 48kHz → 16kHz: take mean of every 3 samples
        usable = (mono48.size // 3) * 3
        if usable == 0:
            return b""
        mono16 = mono48[:usable].reshape(-1, 3).mean(axis=1).astype(np.int16)
        return mono16.tobytes()

    def _trim_pre_roll(self) -> None:
        max_bytes = int((self.config.vad_speech_pad_ms / 1000.0) * self.PCM_BYTES_PER_SECOND)
        if max_bytes <= 0:
            self.pre_roll_pcm.clear()
            return
        overflow = len(self.pre_roll_pcm) - max_bytes
        if overflow > 0:
            del self.pre_roll_pcm[:overflow]

    def push_pcm_frame(self, pcm: bytes, frame_time: float, rms: int) -> List[Dict[str, Any]]:
        """Feed one PCM frame into the VAD; return any state-transition events.

        Each event is ``{"type": "speech_start"|"speech_end_candidate", "prob": float}``.
        Callers should treat ``speech_end_candidate`` as advisory; a confirmed
        end is reported by ``should_finalize`` once the silence threshold has
        been held continuously.
        """
        events: List[Dict[str, Any]] = []
        if not self.speech_active:
            self.pre_roll_pcm.extend(pcm)
            self._trim_pre_roll()

        # Skip the VAD model entirely on near-silent frames once we've seen
        # speech end — saves CPU on idle channels.
        should_feed_vad = self.speech_active or rms >= self.config.rms_fallback_threshold
        if not should_feed_vad:
            return events

        mono16 = self._pcm48k_stereo_to_mono16_bytes(pcm)
        if not mono16:
            return events

        self.mono_buffer_16k.extend(mono16)
        chunk_bytes = self.model.CHUNK_SAMPLES * 2
        while len(self.mono_buffer_16k) >= chunk_bytes:
            chunk = self.model._np.frombuffer(
                bytes(self.mono_buffer_16k[:chunk_bytes]), dtype=self.model._np.int16
            )
            del self.mono_buffer_16k[:chunk_bytes]
            prob = self.stream.process_chunk(chunk)
            self.last_prob = prob

            if not self.speech_active:
                if prob >= self.config.vad_start_prob:
                    self.speech_run_ms += self.VAD_FRAME_MS
                else:
                    self.speech_run_ms = 0.0
                if self.speech_run_ms >= self.config.vad_min_speech_ms:
                    self.speech_active = True
                    self.speech_started_at = frame_time
                    self.last_speech_time = frame_time
                    self.silence_started_at = None
                    events.append({"type": "speech_start", "prob": prob})
            else:
                if prob >= self.config.vad_end_prob:
                    self.last_speech_time = frame_time
                    self.silence_started_at = None
                elif self.silence_started_at is None:
                    self.silence_started_at = frame_time
                    events.append({"type": "speech_end_candidate", "prob": prob})
        return events

    def silence_duration(self, now: float) -> float:
        if self.silence_started_at is None:
            return 0.0
        return max(0.0, now - self.silence_started_at)

    def utterance_duration(self, now: float) -> float:
        if not self.speech_active or self.speech_started_at <= 0:
            return 0.0
        return max(0.0, now - self.speech_started_at)

    def should_finalize(self, now: float) -> Optional[str]:
        """Return the reason to finalise the current utterance, or ``None``."""
        if not self.speech_active:
            return None
        if self.utterance_duration(now) >= self.config.vad_max_utterance_s:
            return "max_utterance"
        if (
            self.silence_started_at is not None
            and self.silence_duration(now) >= (self.config.vad_min_silence_ms / 1000.0)
        ):
            return "vad_silence"
        return None

    def reset(self) -> None:
        self.stream.reset()
        self.mono_buffer_16k.clear()
        self.pre_roll_pcm.clear()
        self.speech_active = False
        self.speech_started_at = 0.0
        self.silence_started_at = None
        self.last_speech_time = 0.0
        self.last_prob = 0.0
        self.speech_run_ms = 0.0
