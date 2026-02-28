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
        assert config.max_speech_duration == 0
        assert config.min_speech_duration == 0

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

    def test_new_params_in_repr(self):
        config = VADConfig(max_speech_duration=30, min_speech_duration=0.5)
        r = repr(config)
        assert "max_speech_duration=30" in r
        assert "min_speech_duration=0.5" in r


class TestVADConfigValidation:
    def test_valid_config(self):
        # Should not raise
        VADConfig()

    def test_invalid_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            VADConfig(sample_rate=0)

    def test_invalid_sample_width(self):
        with pytest.raises(ValueError, match="sample_width"):
            VADConfig(sample_width=3)

    def test_invalid_channels(self):
        with pytest.raises(ValueError, match="channels"):
            VADConfig(channels=0)

    def test_invalid_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_size"):
            VADConfig(chunk_size=-1)

    def test_invalid_max_level(self):
        with pytest.raises(ValueError, match="max_level"):
            VADConfig(max_level=0)

    def test_invalid_threshold_low(self):
        with pytest.raises(ValueError, match="threshold"):
            VADConfig(threshold=-0.1)

    def test_invalid_threshold_high(self):
        with pytest.raises(ValueError, match="threshold"):
            VADConfig(threshold=1.5)

    def test_invalid_attack(self):
        with pytest.raises(ValueError, match="attack"):
            VADConfig(attack=-1)

    def test_invalid_release(self):
        with pytest.raises(ValueError, match="release"):
            VADConfig(release=-0.5)

    def test_invalid_history_size(self):
        with pytest.raises(ValueError, match="history_size"):
            VADConfig(history_size=0)

    def test_invalid_avg_window(self):
        with pytest.raises(ValueError, match="avg_window"):
            VADConfig(avg_window=0)

    def test_invalid_pre_buffer_size(self):
        with pytest.raises(ValueError, match="pre_buffer_size"):
            VADConfig(pre_buffer_size=-1)

    def test_invalid_adapt_up_rate(self):
        with pytest.raises(ValueError, match="adapt_up_rate"):
            VADConfig(adapt_up_rate=-0.01)

    def test_invalid_adapt_down_rate(self):
        with pytest.raises(ValueError, match="adapt_down_rate"):
            VADConfig(adapt_down_rate=-1)

    def test_invalid_max_speech_duration(self):
        with pytest.raises(ValueError, match="max_speech_duration"):
            VADConfig(max_speech_duration=-5)

    def test_invalid_min_speech_duration(self):
        with pytest.raises(ValueError, match="min_speech_duration"):
            VADConfig(min_speech_duration=-1)

    def test_edge_valid_values(self):
        # Boundary values that should be valid
        VADConfig(threshold=0.0)
        VADConfig(threshold=1.0)
        VADConfig(attack=0)
        VADConfig(release=0)
        VADConfig(pre_buffer_size=0)
        VADConfig(max_speech_duration=0)
        VADConfig(min_speech_duration=0)


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

    def test_reset_clears_stats(self):
        config = VADConfig(threshold=0.01, attack=0.0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        vad.feed(loud, timestamp=0.0)
        vad.feed(loud, timestamp=0.1)
        assert vad.get_stats()["total_chunks"] == 2
        vad.reset()
        stats = vad.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["speech_chunks"] == 0
        assert stats["speech_segments"] == 0


class TestEventRepr:
    def test_speech_start_repr(self):
        evt = VADEvent(VADEventType.SPEECH_START, pre_buffer=[b"a", b"b"], timestamp=1.0)
        r = repr(evt)
        assert "SPEECH_START" in r
        assert "2" in r

    def test_audio_repr(self):
        evt = VADEvent(VADEventType.AUDIO, chunk=b"\x00" * 100, rms_level=0.5)
        r = repr(evt)
        assert "AUDIO" in r
        assert "100" in r

    def test_speech_end_repr(self):
        evt = VADEvent(VADEventType.SPEECH_END, duration=2.5)
        r = repr(evt)
        assert "SPEECH_END" in r
        assert "2.5" in r

    def test_speech_timeout_repr(self):
        evt = VADEvent(VADEventType.SPEECH_TIMEOUT, duration=30.0)
        r = repr(evt)
        assert "SPEECH_TIMEOUT" in r
        assert "30" in r


class TestEventFields:
    """Test new event fields: timestamp, rms_level, duration."""

    def test_speech_start_has_timestamp_and_rms(self):
        config = VADConfig(threshold=0.01, attack=0.0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        vad.feed(loud, timestamp=0.0)
        events = vad.feed(loud, timestamp=0.1)
        start_events = [e for e in events if e.type == VADEventType.SPEECH_START]
        assert len(start_events) == 1
        assert start_events[0].timestamp == 0.1
        assert start_events[0].rms_level is not None
        assert start_events[0].rms_level > 0

    def test_audio_has_timestamp_and_rms(self):
        config = VADConfig(threshold=0.01, attack=0.0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        vad.feed(loud, timestamp=0.0)
        events = vad.feed(loud, timestamp=0.1)
        audio_events = [e for e in events if e.type == VADEventType.AUDIO]
        assert len(audio_events) >= 1
        assert audio_events[0].timestamp == 0.1
        assert audio_events[0].rms_level is not None

    def test_speech_end_has_duration(self):
        config = VADConfig(threshold=0.01, attack=0.0, release=0.3)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()
        vad.feed(loud, timestamp=0.0)
        vad.feed(loud, timestamp=0.1)
        all_events = []
        # Feed enough silence to exceed release period
        for i in range(15):
            events = vad.feed(silence, timestamp=0.2 + float(i) * 0.1)
            all_events.extend(events)
        end_events = [e for e in all_events if e.type == VADEventType.SPEECH_END]
        assert len(end_events) == 1
        assert end_events[0].duration is not None
        assert end_events[0].duration > 0


class TestMaxSpeechDuration:
    def test_speech_timeout_event(self):
        config = VADConfig(threshold=0.01, attack=0.0, max_speech_duration=1.0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        all_events = []
        # Start speech
        for i in range(20):
            events = vad.feed(loud, timestamp=float(i) * 0.064)
            all_events.extend(events)
        types = [e.type for e in all_events]
        assert VADEventType.SPEECH_START in types
        assert VADEventType.SPEECH_TIMEOUT in types

    def test_speech_timeout_has_duration(self):
        config = VADConfig(threshold=0.01, attack=0.0, max_speech_duration=0.5)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        all_events = []
        for i in range(20):
            events = vad.feed(loud, timestamp=float(i) * 0.064)
            all_events.extend(events)
        timeout_events = [e for e in all_events if e.type == VADEventType.SPEECH_TIMEOUT]
        assert len(timeout_events) >= 1
        assert timeout_events[0].duration is not None
        assert timeout_events[0].duration >= 0.5

    def test_no_timeout_when_disabled(self):
        config = VADConfig(threshold=0.01, attack=0.0, max_speech_duration=0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        all_events = []
        for i in range(50):
            events = vad.feed(loud, timestamp=float(i) * 0.064)
            all_events.extend(events)
        types = [e.type for e in all_events]
        assert VADEventType.SPEECH_TIMEOUT not in types

    def test_timeout_callback(self):
        called = []
        config = VADConfig(threshold=0.01, attack=0.0, max_speech_duration=0.5)
        vad = RmsVAD(config)
        vad.on_speech_end = lambda: called.append("end")
        loud = _make_loud(amplitude=20000)
        for i in range(20):
            vad.feed(loud, timestamp=float(i) * 0.064)
        assert "end" in called


class TestMinSpeechDuration:
    def test_short_speech_discarded(self):
        config = VADConfig(threshold=0.01, attack=0.0, release=0.2, min_speech_duration=1.0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()
        all_events = []
        # Brief speech (0.2s), then silence
        vad.feed(loud, timestamp=0.0)
        all_events.extend(vad.feed(loud, timestamp=0.1))
        all_events.extend(vad.feed(loud, timestamp=0.2))
        # Silence to trigger end
        for i in range(10):
            all_events.extend(vad.feed(silence, timestamp=0.3 + i * 0.1))
        types = [e.type for e in all_events]
        # SPEECH_START fires normally, but SPEECH_END should be suppressed
        assert VADEventType.SPEECH_END not in types

    def test_long_speech_emitted(self):
        config = VADConfig(threshold=0.01, attack=0.0, release=0.2, min_speech_duration=0.1)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()
        all_events = []
        # Long speech
        for i in range(20):
            all_events.extend(vad.feed(loud, timestamp=float(i) * 0.064))
        # Silence
        for i in range(20):
            all_events.extend(vad.feed(silence, timestamp=1.3 + float(i) * 0.064))
        types = [e.type for e in all_events]
        assert VADEventType.SPEECH_END in types


class TestCurrentLevel:
    def test_level_is_zero_for_silence(self):
        vad = RmsVAD(VADConfig(threshold=0.5))
        silence = _make_silence()
        vad.feed(silence, timestamp=0.0)
        assert vad.current_level == 0.0

    def test_level_positive_for_loud(self):
        vad = RmsVAD(VADConfig(threshold=0.5))
        loud = _make_loud(amplitude=20000)
        vad.feed(loud, timestamp=0.0)
        assert vad.current_level > 0

    def test_level_initial_zero(self):
        vad = RmsVAD()
        assert vad.current_level == 0.0


class TestSpeechSilenceDuration:
    def test_speech_duration_while_speaking(self):
        config = VADConfig(threshold=0.01, attack=0.0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        vad.feed(loud, timestamp=0.0)
        vad.feed(loud, timestamp=0.1)
        vad.feed(loud, timestamp=0.5)
        assert vad.is_speaking
        assert vad.speech_duration > 0

    def test_speech_duration_zero_when_silent(self):
        vad = RmsVAD(VADConfig(threshold=0.5))
        silence = _make_silence()
        vad.feed(silence, timestamp=0.0)
        assert vad.speech_duration == 0.0

    def test_silence_duration_when_silent(self):
        vad = RmsVAD(VADConfig(threshold=0.5))
        silence = _make_silence()
        vad.feed(silence, timestamp=0.0)
        vad.feed(silence, timestamp=1.0)
        assert not vad.is_speaking
        # silence_duration depends on internal timing
        assert vad.silence_duration >= 0

    def test_silence_duration_zero_when_speaking(self):
        config = VADConfig(threshold=0.01, attack=0.0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        vad.feed(loud, timestamp=0.0)
        vad.feed(loud, timestamp=0.1)
        assert vad.is_speaking
        assert vad.silence_duration == 0.0


class TestGetStats:
    def test_initial_stats(self):
        vad = RmsVAD()
        stats = vad.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["speech_chunks"] == 0
        assert stats["silence_chunks"] == 0
        assert stats["speech_ratio"] == 0.0
        assert stats["speech_segments"] == 0

    def test_stats_after_speech_and_silence(self):
        config = VADConfig(threshold=0.01, attack=0.0, release=0.3)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()
        # Some speech
        for i in range(5):
            vad.feed(loud, timestamp=float(i) * 0.064)
        # Some silence
        for i in range(10):
            vad.feed(silence, timestamp=0.5 + float(i) * 0.064)
        stats = vad.get_stats()
        assert stats["total_chunks"] == 15
        assert stats["speech_chunks"] > 0
        assert stats["silence_chunks"] > 0
        assert stats["speech_segments"] >= 1
        assert stats["avg_rms"] > 0
        assert stats["max_rms"] > 0

    def test_stats_speech_ratio(self):
        config = VADConfig(threshold=0.01, attack=0.0)
        vad = RmsVAD(config)
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()
        # Feed 5 loud + 5 silence
        for i in range(5):
            vad.feed(loud, timestamp=float(i) * 0.064)
        for i in range(5):
            vad.feed(silence, timestamp=0.5 + float(i) * 0.064)
        stats = vad.get_stats()
        assert 0 < stats["speech_ratio"] < 1.0


class TestContextManager:
    def test_context_manager_basic(self):
        with RmsVAD(VADConfig(threshold=0.01, attack=0.0)) as vad:
            loud = _make_loud(amplitude=20000)
            vad.feed(loud, timestamp=0.0)
            vad.feed(loud, timestamp=0.1)
            assert vad.is_speaking
        # After exit, state should be reset
        assert not vad.is_speaking

    def test_context_manager_exception(self):
        try:
            with RmsVAD() as vad:
                raise ValueError("test")
        except ValueError:
            pass
        assert not vad.is_speaking
