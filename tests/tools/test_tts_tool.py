"""Tests for tools.tts_tool — Qwen3-TTS provider and speak_text dispatch.

Covers the new Qwen3-TTS provider that uses a query-parameter API
instead of the OpenAI JSON body format.
"""

import io
import os
import tempfile
from unittest.mock import MagicMock, patch, call

import pytest


# ============================================================================
# _generate_qwen3_tts — Qwen3-TTS provider
# ============================================================================

class TestGenerateQwen3TTS:
    """Unit tests for _generate_qwen3_tts."""

    def _make_mock_response(self, audio_bytes: bytes):
        """Create a mock urllib response context manager."""
        resp = MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        # read() returns chunks then empty bytes to signal EOF
        resp.read.side_effect = [audio_bytes, b""]
        return resp

    def test_calls_query_param_endpoint(self, tmp_path):
        """Qwen3-TTS request must use query params, not a JSON body."""
        from tools.tts_tool import _generate_qwen3_tts

        output = str(tmp_path / "out.mp3")
        tts_config = {"qwen3": {"base_url": "http://10.15.0.157:8001", "voice": "ryan"}}

        with patch("urllib.request.urlopen") as mock_open, \
             patch("urllib.request.Request") as mock_request_cls:
            mock_open.return_value = self._make_mock_response(b"fake-audio")
            _generate_qwen3_tts("Hello world", output, tts_config)

        url_arg = mock_request_cls.call_args[0][0]
        assert "text=Hello+world" in url_arg or "text=Hello%20world" in url_arg
        assert "voice=ryan" in url_arg
        assert "/v1/audio/speech" in url_arg
        assert "10.15.0.157:8001" in url_arg

    def test_uses_default_base_url_and_voice(self, tmp_path):
        """Defaults to localhost:8001 and voice='ryan' when not configured."""
        from tools.tts_tool import _generate_qwen3_tts

        output = str(tmp_path / "out.mp3")

        with patch("urllib.request.urlopen") as mock_open, \
             patch("urllib.request.Request") as mock_request_cls:
            mock_open.return_value = self._make_mock_response(b"audio")
            _generate_qwen3_tts("test", output, {})

        url_arg = mock_request_cls.call_args[0][0]
        assert "localhost:8001" in url_arg
        assert "voice=ryan" in url_arg

    def test_response_format_mp3_for_mp3_extension(self, tmp_path):
        from tools.tts_tool import _generate_qwen3_tts

        output = str(tmp_path / "speech.mp3")
        with patch("urllib.request.urlopen") as mock_open, \
             patch("urllib.request.Request") as mock_request_cls:
            mock_open.return_value = self._make_mock_response(b"audio")
            _generate_qwen3_tts("hi", output, {})

        url_arg = mock_request_cls.call_args[0][0]
        assert "response_format=mp3" in url_arg

    def test_response_format_opus_for_ogg_extension(self, tmp_path):
        from tools.tts_tool import _generate_qwen3_tts

        output = str(tmp_path / "speech.ogg")
        with patch("urllib.request.urlopen") as mock_open, \
             patch("urllib.request.Request") as mock_request_cls:
            mock_open.return_value = self._make_mock_response(b"audio")
            _generate_qwen3_tts("hi", output, {})

        url_arg = mock_request_cls.call_args[0][0]
        assert "response_format=opus" in url_arg

    def test_writes_audio_to_output_file(self, tmp_path):
        """The response body should be written to the output path."""
        from tools.tts_tool import _generate_qwen3_tts

        output = str(tmp_path / "out.mp3")
        audio_data = b"FAKE-MP3-AUDIO-DATA"

        with patch("urllib.request.urlopen") as mock_open, \
             patch("urllib.request.Request"):
            mock_open.return_value = self._make_mock_response(audio_data)
            result = _generate_qwen3_tts("hello", output, {})

        assert result == output
        with open(output, "rb") as f:
            assert f.read() == audio_data

    def test_uses_post_method(self, tmp_path):
        """Qwen3-TTS must use POST (not GET)."""
        from tools.tts_tool import _generate_qwen3_tts

        output = str(tmp_path / "out.mp3")
        with patch("urllib.request.urlopen") as mock_open, \
             patch("urllib.request.Request") as mock_request_cls:
            mock_open.return_value = self._make_mock_response(b"audio")
            _generate_qwen3_tts("test", output, {})

        assert mock_request_cls.call_args[1]["method"] == "POST"

    def test_uses_default_timeout_of_120(self, tmp_path):
        """Default timeout must be 120s (not the old 30s) to handle long responses."""
        from tools.tts_tool import _generate_qwen3_tts

        output = str(tmp_path / "out.mp3")
        with patch("urllib.request.urlopen") as mock_open, \
             patch("urllib.request.Request"):
            mock_open.return_value = self._make_mock_response(b"audio")
            _generate_qwen3_tts("test", output, {})

        _, kwargs = mock_open.call_args
        assert kwargs.get("timeout") == 120

    def test_timeout_is_configurable(self, tmp_path):
        """tts.qwen3.timeout should override the default."""
        from tools.tts_tool import _generate_qwen3_tts

        output = str(tmp_path / "out.mp3")
        tts_config = {"qwen3": {"base_url": "http://localhost:8001", "timeout": 60}}
        with patch("urllib.request.urlopen") as mock_open, \
             patch("urllib.request.Request"):
            mock_open.return_value = self._make_mock_response(b"audio")
            _generate_qwen3_tts("test", output, tts_config)

        _, kwargs = mock_open.call_args
        assert kwargs.get("timeout") == 60

    def test_get_provider_returns_qwen3(self):
        """_get_provider should return 'qwen3' when tts.provider is set to 'qwen3'."""
        from tools.tts_tool import _get_provider
        config = {"provider": "qwen3"}
        assert _get_provider(config) == "qwen3"
