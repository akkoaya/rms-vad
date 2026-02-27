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


class VADEvent:
    """An event emitted by the VAD state machine.

    Attributes:
        type: The event type.
        chunk: Audio data (mono, int16 bytes). Present for AUDIO events.
        pre_buffer: List of pre-buffered audio frames. Present for SPEECH_START events.
    """

    __slots__ = ("type", "chunk", "pre_buffer")

    def __init__(self, event_type, chunk=None, pre_buffer=None):
        self.type = event_type
        self.chunk = chunk
        self.pre_buffer = pre_buffer

    def __repr__(self):
        if self.type == VADEventType.SPEECH_START:
            return "VADEvent(SPEECH_START, pre_buffer_frames={})".format(
                len(self.pre_buffer) if self.pre_buffer else 0
            )
        if self.type == VADEventType.AUDIO:
            return "VADEvent(AUDIO, bytes={})".format(
                len(self.chunk) if self.chunk else 0
            )
        return "VADEvent(SPEECH_END)"
