"""Unit tests for RmsVAD core logic."""

import struct
import time

import pytest

from rms_vad import RmsVAD, VADConfig, VADEvent, VADEventType


def _make_silence(chunk_size=1024, channels=1):
    """Generate a silent audio chunk (all zeros)."""
    return b"\x00" * (chunk_size * channels * 2)


def _make_loud(chunk_size=1024, channels=1, amplitude=20000):
    """Generate a loud audio chunk (sine-like constant amplitude)."""
    samples = [amplitude] * (chunk_size * channels)
    return struct.pack("<{}h".format(len(samples)), *samples)


def _make_tone(chunk_size=1024, channels=1, amplitude=10000):
    """Generate audio with a specific amplitude."""
    import math

    samples = []
    for i in range(chunk_size * channels):
        val = int(amplitude * math.sin(2 * math.pi * 440 * i / 16000))
        samples.append(max(-32768, min(32767, val)))
    return struct.pack("<{}h".format(len(samples)), *samples)


class TestVADConfig:
    def test_defaults(self):
        config = VADConfig()
        assert config.sample_rate == 16000
        assert config.sample_width == 2
        assert config.channels == 1
        assert config.threshold == 0.5
        assert config.attack == 0.2
        assert config.release == 1.5

    def test_custom(self):
        config = VADConfig(threshold=0.3, attack=0.1, release=2.0)
        assert config.threshold == 0.3
        assert config.attack == 0.1
        assert config.release == 2.0

    def test_repr(self):
        config = VADConfig()
        r = repr(config)
        assert "VADConfig(" in r
        assert "threshold=0.5" in r


class TestSilenceDetection:
    def test_silence_stays_silent(self):
        vad = RmsVAD(VADConfig(threshold=0.5))
        silence = _make_silence()
        events = vad.feed(silence, timestamp=0.0)
        # No speech start in silence
        assert not any(e.type == VADEventType.SPEECH_START for e in events)
        assert not vad.is_speaking

    def test_silence_no_speech_end_without_speech(self):
        vad = RmsVAD(VADConfig(threshold=0.5))
        silence = _make_silence()
        for i in range(100):
            events = vad.feed(silence, timestamp=float(i) * 0.064)
        assert not any(e.type == VADEventType.SPEECH_END for e in events)


class TestSpeechOnset:
    def test_loud_triggers_speech_start(self):
        config = VADConfig(threshold=0.01, attack=0.0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        # First feed initializes timing
        vad.feed(loud, timestamp=0.0)
        events = vad.feed(loud, timestamp=0.1)
        types = [e.type for e in events]
        assert VADEventType.SPEECH_START in types

    def test_attack_debounce(self):
        config = VADConfig(threshold=0.01, attack=0.5)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        # Within attack period - should not trigger
        vad.feed(loud, timestamp=0.0)
        events = vad.feed(loud, timestamp=0.3)
        types = [e.type for e in events]
        assert VADEventType.SPEECH_START not in types
        # After attack period - should trigger
        events = vad.feed(loud, timestamp=0.6)
        types = [e.type for e in events]
        assert VADEventType.SPEECH_START in types

    def test_pre_buffer_delivered(self):
        config = VADConfig(threshold=0.01, attack=0.0, pre_buffer_size=5)
        vad = RmsVAD(config)
        silence = _make_silence()
        # Fill pre-buffer with silence frames
        for i in range(5):
            vad.feed(silence, timestamp=float(i) * 0.064)
        # Trigger speech
        loud = _make_loud(amplitude=20000)
        events = vad.feed(loud, timestamp=0.5)
        start_events = [e for e in events if e.type == VADEventType.SPEECH_START]
        assert len(start_events) == 1
        assert start_events[0].pre_buffer is not None
        assert len(start_events[0].pre_buffer) == 5


class TestSpeechEnd:
    def test_silence_after_speech_triggers_end(self):
        config = VADConfig(threshold=0.01, attack=0.0, release=0.5)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()
        # Start speech
        vad.feed(loud, timestamp=0.0)
        vad.feed(loud, timestamp=0.1)
        assert vad.is_speaking
        # Silence, but within release
        vad.feed(silence, timestamp=0.3)
        assert vad.is_speaking
        # Silence, beyond release
        events = vad.feed(silence, timestamp=0.8)
        types = [e.type for e in events]
        assert VADEventType.SPEECH_END in types
        assert not vad.is_speaking

    def test_brief_silence_no_end(self):
        config = VADConfig(threshold=0.01, attack=0.0, release=1.5)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()
        # Start speech
        vad.feed(loud, timestamp=0.0)
        vad.feed(loud, timestamp=0.1)
        # Brief silence (< release)
        events = vad.feed(silence, timestamp=0.5)
        assert vad.is_speaking
        assert not any(e.type == VADEventType.SPEECH_END for e in events)


class TestDynamicThreshold:
    def test_threshold_adapts_down(self):
        config = VADConfig(threshold=0.5, adapt_down_rate=1.0)
        vad = RmsVAD(config)
        silence = _make_silence()
        initial = vad.threshold
        for i in range(50):
            vad.feed(silence, timestamp=float(i) * 0.064)
        assert vad.threshold < initial

    def test_threshold_adapts_up_slowly(self):
        config = VADConfig(threshold=0.01, adapt_up_rate=0.0025)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=15000)
        initial = vad.threshold
        for i in range(50):
            vad.feed(loud, timestamp=float(i) * 0.064)
        # Should increase but slowly
        assert vad.threshold > initial
        # Should not jump to full level
        assert vad.threshold < 0.5

    def test_manual_threshold_set(self):
        vad = RmsVAD()
        vad.threshold = 0.8
        assert vad.threshold == 0.8


class TestCallbacks:
    def test_on_speech_start_callback(self):
        called = []
        config = VADConfig(threshold=0.01, attack=0.0)
        vad = RmsVAD(config)
        vad.on_speech_start = lambda pre_buf: called.append(("start", pre_buf))
        loud = _make_loud(amplitude=20000)
        vad.feed(loud, timestamp=0.0)
        vad.feed(loud, timestamp=0.1)
        assert len(called) == 1
        assert called[0][0] == "start"

    def test_on_audio_callback(self):
        chunks = []
        config = VADConfig(threshold=0.01, attack=0.0)
        vad = RmsVAD(config)
        vad.on_audio = lambda chunk: chunks.append(chunk)
        loud = _make_loud(amplitude=20000)
        vad.feed(loud, timestamp=0.0)
        vad.feed(loud, timestamp=0.1)
        assert len(chunks) >= 1

    def test_on_speech_end_callback(self):
        called = []
        config = VADConfig(threshold=0.01, attack=0.0, release=0.3)
        vad = RmsVAD(config)
        vad.on_speech_end = lambda: called.append("end")
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()
        vad.feed(loud, timestamp=0.0)
        vad.feed(loud, timestamp=0.1)
        vad.feed(silence, timestamp=0.5)
        vad.feed(silence, timestamp=0.9)
        assert "end" in called


class TestIterEvents:
    def test_iter_events_basic(self):
        config = VADConfig(threshold=0.01, attack=0.0, release=0.3)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()
        stream = [loud, loud, silence, silence]
        timestamps = [0.0, 0.1, 0.5, 0.9]
        events = list(vad.iter_events(stream, timestamps=timestamps))
        types = [e.type for e in events]
        assert VADEventType.SPEECH_START in types
        assert VADEventType.AUDIO in types
        assert VADEventType.SPEECH_END in types


class TestMultiChannel:
    def test_stereo_input(self):
        config = VADConfig(threshold=0.01, attack=0.0, channels=2)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000, channels=2)
        vad.feed(loud, timestamp=0.0)
        events = vad.feed(loud, timestamp=0.1)
        types = [e.type for e in events]
        assert VADEventType.SPEECH_START in types


class TestReset:
    def test_reset_clears_state(self):
        config = VADConfig(threshold=0.01, attack=0.0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        vad.feed(loud, timestamp=0.0)
        vad.feed(loud, timestamp=0.1)
        assert vad.is_speaking
        vad.reset()
        assert not vad.is_speaking
        assert vad.threshold == config.threshold


class TestEventRepr:
    def test_speech_start_repr(self):
        evt = VADEvent(VADEventType.SPEECH_START, pre_buffer=[b"a", b"b"])
        assert "SPEECH_START" in repr(evt)
        assert "2" in repr(evt)

    def test_audio_repr(self):
        evt = VADEvent(VADEventType.AUDIO, chunk=b"\x00" * 100)
        assert "AUDIO" in repr(evt)
        assert "100" in repr(evt)

    def test_speech_end_repr(self):
        evt = VADEvent(VADEventType.SPEECH_END)
        assert "SPEECH_END" in repr(evt)
