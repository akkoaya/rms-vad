"""VAD event types and data."""

import enum


class VADEventType(enum.Enum):
    """Types of events emitted by the VAD."""

    SPEECH_START = "speech_start"
    """Speech onset detected. Carries pre-buffered audio frames."""

    AUDIO = "audio"
    """Audio chunk during active speech."""

    SPEECH_END = "speech_end"
    """Speech ended after silence exceeded release time."""

    SPEECH_TIMEOUT = "speech_timeout"
    """Speech auto-ended because max_speech_duration was exceeded."""


class VADEvent:
    """An event emitted by the VAD state machine.

    Attributes:
        type: The event type.
        chunk: Audio data (mono, int16 bytes). Present for AUDIO events.
        pre_buffer: List of pre-buffered audio frames. Present for SPEECH_START events.
        timestamp: Timestamp in seconds when the event was generated.
        rms_level: Normalized RMS level (0.0~1.0+) at event time.
        duration: Speech segment duration in seconds. Present for SPEECH_END
            and SPEECH_TIMEOUT events.
    """

    __slots__ = ("type", "chunk", "pre_buffer", "timestamp", "rms_level", "duration")

    def __init__(self, event_type, chunk=None, pre_buffer=None,
                 timestamp=None, rms_level=None, duration=None):
        self.type = event_type
        self.chunk = chunk
        self.pre_buffer = pre_buffer
        self.timestamp = timestamp
        self.rms_level = rms_level
        self.duration = duration

    def __repr__(self):
        if self.type == VADEventType.SPEECH_START:
            return "VADEvent(SPEECH_START, pre_buffer_frames={}, ts={})".format(
                len(self.pre_buffer) if self.pre_buffer else 0,
                _fmt_ts(self.timestamp),
            )
        if self.type == VADEventType.AUDIO:
            return "VADEvent(AUDIO, bytes={}, rms={})".format(
                len(self.chunk) if self.chunk else 0,
                _fmt_float(self.rms_level),
            )
        if self.type == VADEventType.SPEECH_END:
            return "VADEvent(SPEECH_END, duration={})".format(
                _fmt_float(self.duration),
            )
        if self.type == VADEventType.SPEECH_TIMEOUT:
            return "VADEvent(SPEECH_TIMEOUT, duration={})".format(
                _fmt_float(self.duration),
            )
        return "VADEvent({})".format(self.type)


def _fmt_ts(val):
    if val is None:
        return "None"
    return "{:.3f}".format(val)


def _fmt_float(val):
    if val is None:
        return "None"
    return "{:.4f}".format(val)
