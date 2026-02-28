"""Audio utility functions for rms-vad.

Provides helpers for WAV I/O, audio collection, SNR estimation,
and energy computation -- all with zero required dependencies.
"""

import io
import math
import struct
import wave

from .events import VADEventType


def pcm_to_wav(pcm_data, sample_rate=16000, sample_width=2, channels=1):
    """Convert raw PCM bytes to WAV format bytes.

    Args:
        pcm_data: Raw PCM audio bytes.
        sample_rate: Sample rate in Hz.
        sample_width: Bytes per sample (1, 2, or 4).
        channels: Number of audio channels.

    Returns:
        bytes: Complete WAV file content.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def wav_to_pcm(wav_data):
    """Extract raw PCM bytes and parameters from WAV data.

    Args:
        wav_data: WAV file content as bytes, or a file path string.

    Returns:
        tuple: (pcm_bytes, sample_rate, sample_width, channels)
    """
    if isinstance(wav_data, str):
        with open(wav_data, "rb") as f:
            wav_data = f.read()
    buf = io.BytesIO(wav_data)
    with wave.open(buf, "rb") as wf:
        pcm = wf.readframes(wf.getnframes())
        return pcm, wf.getframerate(), wf.getsampwidth(), wf.getnchannels()


def save_wav(file_path, pcm_data, sample_rate=16000, sample_width=2, channels=1):
    """Save raw PCM data to a WAV file.

    Args:
        file_path: Output file path.
        pcm_data: Raw PCM audio bytes.
        sample_rate: Sample rate in Hz.
        sample_width: Bytes per sample.
        channels: Number of channels.
    """
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)


def compute_energy_db(pcm_data, sample_width=2):
    """Compute energy in decibels for PCM audio.

    Args:
        pcm_data: Raw PCM audio bytes.
        sample_width: Bytes per sample (only 2 supported).

    Returns:
        float: Energy in dB. Returns -inf for silence.
    """
    if len(pcm_data) == 0:
        return float("-inf")
    try:
        import audioop
    except ModuleNotFoundError:
        import audioop_lts as audioop
    rms = audioop.rms(pcm_data, sample_width)
    if rms == 0:
        return float("-inf")
    return 20.0 * math.log10(rms)


def compute_snr(speech_pcm, noise_pcm, sample_width=2):
    """Estimate Signal-to-Noise Ratio in decibels.

    Args:
        speech_pcm: PCM bytes of the speech segment.
        noise_pcm: PCM bytes of the noise/silence segment.
        sample_width: Bytes per sample.

    Returns:
        float: SNR in dB. Returns inf if noise is zero.
    """
    speech_db = compute_energy_db(speech_pcm, sample_width)
    noise_db = compute_energy_db(noise_pcm, sample_width)
    if noise_db == float("-inf"):
        return float("inf")
    return speech_db - noise_db


class AudioCollector:
    """Collects audio data from VAD events into complete speech segments.

    Usage::

        collector = AudioCollector()
        for event in vad.iter_events(stream):
            segment = collector.feed(event)
            if segment is not None:
                # segment is a SpeechSegment with .audio, .duration, etc.
                process(segment)

    Or use the callback style::

        collector = AudioCollector(on_segment=lambda seg: process(seg))
        for event in vad.iter_events(stream):
            collector.feed(event)
    """

    def __init__(self, on_segment=None, include_pre_buffer=True):
        """Initialize the collector.

        Args:
            on_segment: Optional callback called with a SpeechSegment
                when a complete segment is collected.
            include_pre_buffer: Whether to include pre-buffer frames in the
                collected audio. Default True.
        """
        self._on_segment = on_segment
        self._include_pre_buffer = include_pre_buffer
        self._chunks = []
        self._collecting = False
        self._start_timestamp = None
        self._segment_count = 0

    @property
    def is_collecting(self):
        """Whether the collector is currently collecting audio."""
        return self._collecting

    @property
    def segment_count(self):
        """Number of completed speech segments so far."""
        return self._segment_count

    def feed(self, event):
        """Process a VAD event and return a SpeechSegment if complete.

        Args:
            event: A VADEvent from RmsVAD.

        Returns:
            SpeechSegment if a speech segment just completed, else None.
        """
        if event.type == VADEventType.SPEECH_START:
            self._collecting = True
            self._chunks = []
            self._start_timestamp = event.timestamp
            if self._include_pre_buffer and event.pre_buffer:
                self._chunks.extend(event.pre_buffer)

        elif event.type == VADEventType.AUDIO:
            if self._collecting and event.chunk:
                self._chunks.append(event.chunk)

        elif event.type in (VADEventType.SPEECH_END, VADEventType.SPEECH_TIMEOUT):
            if self._collecting:
                self._collecting = False
                self._segment_count += 1
                audio = b"".join(self._chunks)
                segment = SpeechSegment(
                    audio=audio,
                    duration=event.duration,
                    timestamp=self._start_timestamp,
                    segment_index=self._segment_count - 1,
                    timed_out=event.type == VADEventType.SPEECH_TIMEOUT,
                )
                self._chunks = []
                if self._on_segment is not None:
                    self._on_segment(segment)
                return segment

        return None

    def reset(self):
        """Reset collector state."""
        self._chunks = []
        self._collecting = False
        self._start_timestamp = None


class SpeechSegment:
    """A complete speech segment collected from VAD events.

    Attributes:
        audio: Complete PCM audio bytes for the segment.
        duration: Duration of the speech segment in seconds.
        timestamp: Timestamp when speech started.
        segment_index: Zero-based index of this segment.
        timed_out: True if the segment ended due to max_speech_duration.
    """

    __slots__ = ("audio", "duration", "timestamp", "segment_index", "timed_out")

    def __init__(self, audio, duration=None, timestamp=None,
                 segment_index=0, timed_out=False):
        self.audio = audio
        self.duration = duration
        self.timestamp = timestamp
        self.segment_index = segment_index
        self.timed_out = timed_out

    @property
    def num_bytes(self):
        """Number of bytes in the audio data."""
        return len(self.audio) if self.audio else 0

    @property
    def num_samples(self):
        """Number of 16-bit samples in the audio (assumes sample_width=2)."""
        return len(self.audio) // 2 if self.audio else 0

    def to_wav(self, sample_rate=16000, sample_width=2, channels=1):
        """Convert the segment audio to WAV bytes.

        Args:
            sample_rate: Sample rate in Hz.
            sample_width: Bytes per sample.
            channels: Number of channels.

        Returns:
            bytes: WAV file content.
        """
        return pcm_to_wav(self.audio, sample_rate, sample_width, channels)

    def save_wav(self, file_path, sample_rate=16000, sample_width=2, channels=1):
        """Save the segment audio to a WAV file.

        Args:
            file_path: Output file path.
            sample_rate: Sample rate in Hz.
            sample_width: Bytes per sample.
            channels: Number of channels.
        """
        save_wav(file_path, self.audio, sample_rate, sample_width, channels)

    def __repr__(self):
        return "SpeechSegment(bytes={}, duration={}, index={}{})".format(
            self.num_bytes,
            "{:.3f}s".format(self.duration) if self.duration else "None",
            self.segment_index,
            ", timed_out" if self.timed_out else "",
        )
