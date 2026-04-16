from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class DiscordVCVADConfig:
    vad_enabled: bool = True
    vad_mode: str = "silero_hybrid"
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 550
    vad_speech_pad_ms: int = 150
    vad_start_prob: float = 0.55
    vad_end_prob: float = 0.35
    vad_max_utterance_s: float = 20.0
    rms_fallback_threshold: int = 200
    min_utterance_rms: int = 300
    barge_in_guard: float = 0.5
    barge_in_rms: int = 600

    @classmethod
    def from_voice_config(cls, voice_cfg: Optional[Dict[str, Any]]) -> "DiscordVCVADConfig":
        voice_cfg = voice_cfg or {}
        vc_cfg = voice_cfg.get("discord_vc") or {}
        return cls(
            vad_enabled=bool(vc_cfg.get("vad_enabled", True)),
            vad_mode=str(vc_cfg.get("vad_mode", "silero_hybrid") or "silero_hybrid"),
            vad_min_speech_ms=int(vc_cfg.get("vad_min_speech_ms", 250)),
            vad_min_silence_ms=int(vc_cfg.get("vad_min_silence_ms", 550)),
            vad_speech_pad_ms=int(vc_cfg.get("vad_speech_pad_ms", 150)),
            vad_start_prob=float(vc_cfg.get("vad_start_prob", 0.55)),
            vad_end_prob=float(vc_cfg.get("vad_end_prob", 0.35)),
            vad_max_utterance_s=float(vc_cfg.get("vad_max_utterance_s", 20.0)),
            rms_fallback_threshold=int(vc_cfg.get("rms_fallback_threshold", 200)),
            min_utterance_rms=int(vc_cfg.get("min_utterance_rms", 300)),
            barge_in_guard=float(vc_cfg.get("barge_in_guard", voice_cfg.get("barge_in_guard", 0.5))),
            barge_in_rms=int(vc_cfg.get("barge_in_rms", voice_cfg.get("barge_in_rms", 600))),
        )


def get_vad_model_path() -> Path:
    return Path(__file__).resolve().parents[1] / "assets" / "silero_vad.onnx"


class SileroOnnxVAD:
    """Lightweight ONNX wrapper for the official Silero VAD model."""

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
        providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in onnxruntime.get_available_providers() else None
        path = str(model_path or get_vad_model_path())
        self._session = onnxruntime.InferenceSession(path, providers=providers, sess_options=opts)

    def new_stream(self) -> "SileroOnnxVADStream":
        return SileroOnnxVADStream(self)


class SileroOnnxVADStream:
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
    utterance_trace_id: str = ""

    PCM_BYTES_PER_SECOND = 48000 * 2 * 2
    VAD_FRAME_MS = 32.0

    def __post_init__(self) -> None:
        self.stream = self.model.new_stream()

    def _pcm48k_stereo_to_mono16_bytes(self, pcm: bytes) -> bytes:
        np = self.model._np
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return b""
        mono48 = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
        mono16 = mono48.reshape(-1, 3).mean(axis=1).astype(np.int16)
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
        events: List[Dict[str, Any]] = []
        if not self.speech_active:
            self.pre_roll_pcm.extend(pcm)
            self._trim_pre_roll()

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
        if not self.speech_active:
            return None
        if self.utterance_duration(now) >= self.config.vad_max_utterance_s:
            return "max_utterance"
        if self.silence_started_at is not None and self.silence_duration(now) >= (self.config.vad_min_silence_ms / 1000.0):
            return "vad_silence"
        return None

    def pre_roll_duration(self) -> float:
        return len(self.pre_roll_pcm) / self.PCM_BYTES_PER_SECOND

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
        self.utterance_trace_id = ""
