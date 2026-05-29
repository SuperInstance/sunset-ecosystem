"""Tests for Soniqo Bridge voice integration.

Covers mock voice engines, tile submission, and room interaction.
"""

import pytest

from voice.soniqo_bridge import SoniqoBridge, VoiceTile, _MockASR, _MockTTS, _MockVAD


class TestMockEngines:
    def test_mock_asr(self):
        asr = _MockASR()
        result = asr.transcribe(b"fake audio")
        assert "mock transcription" in result

    def test_mock_tts(self):
        tts = _MockTTS()
        audio = tts.synthesize("hello")
        assert len(audio) > 0

    def test_mock_vad(self):
        vad = _MockVAD()
        assert vad.is_speech() is True


class TestSoniqoBridge:
    def test_init(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        assert bridge.room_id == "harbor"
        assert bridge.node_id == "alpha"
        assert bridge._connected is False

    def test_connect(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        assert bridge.connect() is True
        assert bridge._connected is True
        assert bridge._asr_engine is not None
        assert bridge._tts_engine is not None
        assert bridge._vad_engine is not None

    def test_disconnect(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        bridge.connect()
        bridge.disconnect()
        assert bridge._connected is False

    def test_listen_and_transcribe(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        bridge.connect()
        result = bridge.listen_and_transcribe(timeout=0.5)
        assert result is not None
        assert "mock" in result.lower()

    def test_speak(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        bridge.connect()
        assert bridge.speak("hello world") is True

    def test_speak_not_connected(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        assert bridge.speak("hello") is False

    def test_submit_voice_tile(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        bridge.connect()
        tile = bridge.submit_voice_tile("fleet status is green", "operator")
        assert tile.room_id == "harbor"
        assert tile.speaker == "operator"
        assert tile.transcript == "fleet status is green"
        assert tile.confidence > 0.9

    def test_query_room(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        bridge.connect()
        response = bridge.query_room("What is the fleet status?")
        assert "harbor" in response
        assert "fleet status" in response.lower()

    def test_voice_history(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        bridge.connect()
        bridge.submit_voice_tile("hello", "a")
        bridge.submit_voice_tile("world", "b")
        history = bridge.get_voice_history()
        assert len(history) == 2
        assert history[0].speaker == "a"
        assert history[1].speaker == "b"

    def test_get_status(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        bridge.connect()
        status = bridge.get_status()
        assert status["room_id"] == "harbor"
        assert status["node_id"] == "alpha"
        assert status["connected"] is True
        assert status["engines"]["asr"] is True
        assert status["engines"]["tts"] is True
        assert status["engines"]["vad"] is True

    def test_voice_tile_metadata(self):
        bridge = SoniqoBridge(room_id="harbor", node_id="alpha")
        bridge.connect()
        tile = bridge.submit_voice_tile("test", "op")
        assert tile.metadata["node_id"] == "alpha"
        assert "engine" in tile.metadata


class TestVoiceTile:
    def test_defaults(self):
        tile = VoiceTile(
            tile_id="1",
            room_id="r",
            speaker="s",
            transcript="hello",
            audio_hash="h",
            duration_ms=1000.0,
            confidence=0.95
        )
        assert tile.timestamp > 0
        assert tile.metadata == {}

    def test_tile_fields(self):
        tile = VoiceTile(
            tile_id="voice:123",
            room_id="forge",
            speaker="ccc",
            transcript="the fleet is ready",
            audio_hash="abc123",
            duration_ms=2500.0,
            confidence=0.98,
            metadata={"source": "microphone"}
        )
        assert tile.tile_id == "voice:123"
        assert tile.room_id == "forge"
        assert tile.transcript == "the fleet is ready"
        assert tile.metadata["source"] == "microphone"
