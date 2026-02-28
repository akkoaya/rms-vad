"""Tests for audio_utils module."""

import math
import struct
import os
import tempfile

import pytest

from rms_vad import (
    RmsVAD,
    VADConfig,
    VADEventType,
    AudioCollector,
    SpeechSegment,
    pcm_to_wav,
    wav_to_pcm,
    save_wav,
    compute_energy_db,
    compute_snr,
)


def _make_silence(chunk_size=1024):
    return b"\x00" * (chunk_size * 2)


def _make_loud(chunk_size=1024, amplitude=20000):
    samples = [amplitude] * chunk_size
    return struct.pack("<{}h".format(len(samples)), *samples)


def _make_tone(chunk_size=1024, amplitude=10000, freq=440, sample_rate=16000):
    samples = []
    for i in range(chunk_size):
        val = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        samples.append(max(-32768, min(32767, val)))
    return struct.pack("<{}h".format(len(samples)), *samples)


class TestPcmToWav:
    def test_roundtrip(self):
        pcm = _make_tone(1024)
        wav_data = pcm_to_wav(pcm, sample_rate=16000, sample_width=2, channels=1)
        # WAV should start with RIFF header
        assert wav_data[:4] == b"RIFF"
        # Roundtrip
        pcm2, sr, sw, ch = wav_to_pcm(wav_data)
        assert pcm2 == pcm
        assert sr == 16000
        assert sw == 2
        assert ch == 1

    def test_stereo(self):
        pcm = _make_loud(512, amplitude=5000)
        # Double the data for stereo (interleaved)
        stereo_pcm = pcm + pcm  # Not proper interleaving but tests WAV structure
        wav_data = pcm_to_wav(stereo_pcm, sample_rate=44100, sample_width=2, channels=2)
        pcm2, sr, sw, ch = wav_to_pcm(wav_data)
        assert sr == 44100
        assert ch == 2

    def test_silence(self):
        pcm = _make_silence(256)
        wav_data = pcm_to_wav(pcm)
        pcm2, sr, sw, ch = wav_to_pcm(wav_data)
        assert pcm2 == pcm


class TestSaveWav:
    def test_save_and_read(self):
        pcm = _make_tone(1024)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            save_wav(path, pcm, sample_rate=16000, sample_width=2, channels=1)
            # Read back using wav_to_pcm with file path
            with open(path, "rb") as f:
                wav_data = f.read()
            pcm2, sr, sw, ch = wav_to_pcm(wav_data)
            assert pcm2 == pcm
            assert sr == 16000
        finally:
            os.unlink(path)


class TestComputeEnergyDb:
    def test_silence_is_neg_inf(self):
        silence = _make_silence(1024)
        db = compute_energy_db(silence)
        assert db == float("-inf")

    def test_loud_is_positive(self):
        loud = _make_loud(1024, amplitude=20000)
        db = compute_energy_db(loud)
        assert db > 0

    def test_louder_is_higher(self):
        quiet = _make_loud(1024, amplitude=1000)
        loud = _make_loud(1024, amplitude=20000)
        db_quiet = compute_energy_db(quiet)
        db_loud = compute_energy_db(loud)
        assert db_loud > db_quiet

    def test_empty_is_neg_inf(self):
        db = compute_energy_db(b"")
        assert db == float("-inf")


class TestComputeSnr:
    def test_silence_noise_is_inf(self):
        speech = _make_loud(1024, amplitude=10000)
        noise = _make_silence(1024)
        snr = compute_snr(speech, noise)
        assert snr == float("inf")

    def test_positive_snr(self):
        speech = _make_loud(1024, amplitude=20000)
        noise = _make_loud(1024, amplitude=100)
        snr = compute_snr(speech, noise)
        assert snr > 0

    def test_zero_snr_for_same_signal(self):
        signal = _make_loud(1024, amplitude=10000)
        snr = compute_snr(signal, signal)
        assert abs(snr) < 0.01  # Should be ~0 dB


class TestAudioCollector:
    def _run_vad_with_collector(self, config=None, collector=None):
        """Helper: run VAD over speech + silence and collect segments."""
        if config is None:
            config = VADConfig(threshold=0.01, attack=0.0, release=0.3)
        vad = RmsVAD(config)
        if collector is None:
            collector = AudioCollector()
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()

        segments = []
        t = 0.0
        # Speech
        for _ in range(15):
            for event in vad.feed(loud, timestamp=t):
                seg = collector.feed(event)
                if seg is not None:
                    segments.append(seg)
            t += 0.064
        # Silence to end speech
        for _ in range(15):
            for event in vad.feed(silence, timestamp=t):
                seg = collector.feed(event)
                if seg is not None:
                    segments.append(seg)
            t += 0.064
        return segments, collector

    def test_collects_one_segment(self):
        segments, collector = self._run_vad_with_collector()
        assert len(segments) == 1
        assert segments[0].num_bytes > 0
        assert segments[0].duration is not None
        assert segments[0].duration > 0
        assert segments[0].segment_index == 0
        assert not segments[0].timed_out

    def test_segment_count(self):
        segments, collector = self._run_vad_with_collector()
        assert collector.segment_count == 1

    def test_collector_callback(self):
        received = []
        collector = AudioCollector(on_segment=lambda seg: received.append(seg))
        self._run_vad_with_collector(collector=collector)
        assert len(received) == 1
        assert isinstance(received[0], SpeechSegment)

    def test_collector_with_pre_buffer(self):
        config = VADConfig(threshold=0.01, attack=0.0, release=0.3, pre_buffer_size=5)
        collector = AudioCollector(include_pre_buffer=True)
        segments, _ = self._run_vad_with_collector(config=config, collector=collector)
        assert len(segments) == 1
        # With pre_buffer included, audio should be larger
        assert segments[0].num_bytes > 0

    def test_collector_without_pre_buffer(self):
        config = VADConfig(threshold=0.01, attack=0.0, release=0.3, pre_buffer_size=5)
        collector = AudioCollector(include_pre_buffer=False)
        segments, _ = self._run_vad_with_collector(config=config, collector=collector)
        assert len(segments) == 1

    def test_collector_reset(self):
        collector = AudioCollector()
        self._run_vad_with_collector(collector=collector)
        assert collector.segment_count == 1
        collector.reset()
        assert not collector.is_collecting

    def test_collector_multiple_segments(self):
        config = VADConfig(threshold=0.01, attack=0.0, release=0.3)
        vad = RmsVAD(config)
        collector = AudioCollector()
        loud = _make_loud(amplitude=20000)
        silence = _make_silence()
        segments = []
        t = 0.0

        # Segment 1
        for _ in range(10):
            for event in vad.feed(loud, timestamp=t):
                seg = collector.feed(event)
                if seg is not None:
                    segments.append(seg)
            t += 0.064
        for _ in range(15):
            for event in vad.feed(silence, timestamp=t):
                seg = collector.feed(event)
                if seg is not None:
                    segments.append(seg)
            t += 0.064

        # Segment 2
        for _ in range(10):
            for event in vad.feed(loud, timestamp=t):
                seg = collector.feed(event)
                if seg is not None:
                    segments.append(seg)
            t += 0.064
        for _ in range(15):
            for event in vad.feed(silence, timestamp=t):
                seg = collector.feed(event)
                if seg is not None:
                    segments.append(seg)
            t += 0.064

        assert len(segments) == 2
        assert segments[0].segment_index == 0
        assert segments[1].segment_index == 1
        assert collector.segment_count == 2

    def test_timeout_segment(self):
        config = VADConfig(threshold=0.01, attack=0.0, max_speech_duration=0.5)
        vad = RmsVAD(config)
        collector = AudioCollector()
        loud = _make_loud(amplitude=20000)
        segments = []
        t = 0.0
        for _ in range(20):
            for event in vad.feed(loud, timestamp=t):
                seg = collector.feed(event)
                if seg is not None:
                    segments.append(seg)
            t += 0.064
        assert len(segments) >= 1
        assert segments[0].timed_out


class TestSpeechSegment:
    def test_repr(self):
        seg = SpeechSegment(audio=b"\x00" * 200, duration=1.5, segment_index=0)
        r = repr(seg)
        assert "200" in r
        assert "1.5" in r

    def test_repr_timed_out(self):
        seg = SpeechSegment(audio=b"\x00" * 100, duration=30.0, segment_index=0, timed_out=True)
        r = repr(seg)
        assert "timed_out" in r

    def test_num_samples(self):
        seg = SpeechSegment(audio=b"\x00" * 200)
        assert seg.num_samples == 100  # 200 bytes / 2 bytes per sample

    def test_to_wav(self):
        pcm = _make_tone(512)
        seg = SpeechSegment(audio=pcm, duration=0.032)
        wav_data = seg.to_wav(sample_rate=16000)
        assert wav_data[:4] == b"RIFF"
        # Roundtrip
        pcm2, sr, sw, ch = wav_to_pcm(wav_data)
        assert pcm2 == pcm

    def test_save_wav(self):
        pcm = _make_tone(512)
        seg = SpeechSegment(audio=pcm, duration=0.032)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            seg.save_wav(path, sample_rate=16000)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
