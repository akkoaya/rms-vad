"""RMS-based Voice Activity Detection core."""

try:
    import audioop
except ModuleNotFoundError:
    import audioop_lts as audioop
import time
from collections import deque

from .config import VADConfig
from .events import VADEvent, VADEventType

try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


def _to_mono_numpy(buffer, channels):
    """Convert multi-channel PCM16 to mono using numpy."""
    data = np.frombuffer(buffer, dtype=np.int16)
    data = data.reshape(-1, channels)
    return np.mean(data, axis=1).astype(np.int16).tobytes()


def _to_mono_pure(buffer, channels):
    """Convert multi-channel PCM16 to mono using pure Python."""
    import struct

    sample_count = len(buffer) // (2 * channels)
    mono_samples = []
    for i in range(sample_count):
        total = 0
        for ch in range(channels):
            offset = (i * channels + ch) * 2
            (sample,) = struct.unpack_from("<h", buffer, offset)
            total += sample
        mono_samples.append(int(total / channels))
    return struct.pack("<{}h".format(len(mono_samples)), *mono_samples)


def _to_mono(buffer, channels):
    """Convert multi-channel PCM16 to mono."""
    if channels == 1:
        return buffer
    if _HAS_NUMPY:
        return _to_mono_numpy(buffer, channels)
    return _to_mono_pure(buffer, channels)


class RmsVAD:
    """RMS-based Voice Activity Detector.

    Processes audio chunks and emits VADEvent objects indicating speech
    start, active audio, and speech end.

    Usage::

        vad = RmsVAD()
        for chunk in audio_stream:
            for event in vad.feed(chunk):
                if event.type == VADEventType.SPEECH_START:
                    asr.start()
                    for frame in event.pre_buffer:
                        asr.send(frame)
                elif event.type == VADEventType.AUDIO:
                    asr.send(event.chunk)
                elif event.type == VADEventType.SPEECH_END:
                    asr.end()

    Callback style::

        vad = RmsVAD()
        vad.on_speech_start = lambda pre_buffer: ...
        vad.on_audio = lambda chunk: ...
        vad.on_speech_end = lambda: ...
        vad.feed(chunk)
    """

    def __init__(self, config=None):
        """Initialize the VAD.

        Args:
            config: A VADConfig instance. Uses defaults if None.
        """
        if config is None:
            config = VADConfig()
        self._config = config

        # State
        self._is_speaking = False
        self._last_mute_time = 0.0
        self._last_speaking_time = 0.0
        self._dynamic_threshold = config.threshold
        self._started = False

        # Buffers
        self._history_level = deque(maxlen=config.history_size)
        self._pre_buffer = deque(maxlen=config.pre_buffer_size)

        # Callbacks (optional)
        self.on_speech_start = None  # callable(pre_buffer: list[bytes])
        self.on_audio = None  # callable(chunk: bytes)
        self.on_speech_end = None  # callable()

    @property
    def is_speaking(self):
        """Whether the VAD currently detects active speech."""
        return self._is_speaking

    @property
    def threshold(self):
        """Current dynamic threshold value."""
        return self._dynamic_threshold

    @threshold.setter
    def threshold(self, value):
        """Manually set the dynamic threshold."""
        self._dynamic_threshold = value

    def reset(self):
        """Reset VAD state. Clears all buffers and resets to silence."""
        self._is_speaking = False
        self._last_mute_time = 0.0
        self._last_speaking_time = 0.0
        self._dynamic_threshold = self._config.threshold
        self._started = False
        self._history_level.clear()
        self._pre_buffer.clear()

    def feed(self, buffer, timestamp=None):
        """Process an audio chunk and return a list of VAD events.

        Args:
            buffer: Raw PCM audio bytes (int16).
            timestamp: Optional timestamp in seconds. If None, uses time.time().

        Returns:
            List of VADEvent objects (may be empty).
        """
        if timestamp is None:
            timestamp = time.time()

        if not self._started:
            self._last_mute_time = timestamp
            self._last_speaking_time = timestamp
            self._started = True

        events = []

        # RMS and threshold update
        mono_data = _to_mono(buffer, self._config.channels)
        level = audioop.rms(mono_data, self._config.sample_width)
        self._history_level.append(level)
        percentage = level / self._config.max_level
        self._update_threshold()

        if percentage > self._dynamic_threshold:
            # Above threshold
            self._last_speaking_time = timestamp
            if not self._is_speaking:
                if timestamp - self._last_mute_time > self._config.attack:
                    # Transition: Silence -> Speaking
                    self._is_speaking = True
                    pre_buf = list(self._pre_buffer)
                    self._pre_buffer.clear()

                    evt = VADEvent(
                        VADEventType.SPEECH_START, pre_buffer=pre_buf
                    )
                    events.append(evt)
                    if self.on_speech_start is not None:
                        self.on_speech_start(pre_buf)
        else:
            # Below threshold
            self._last_mute_time = timestamp
            if self._is_speaking:
                if timestamp - self._last_speaking_time > self._config.release:
                    # Transition: Speaking -> Silence
                    self._is_speaking = False
                    evt = VADEvent(VADEventType.SPEECH_END)
                    events.append(evt)
                    if self.on_speech_end is not None:
                        self.on_speech_end()

        if self._is_speaking:
            evt = VADEvent(VADEventType.AUDIO, chunk=mono_data)
            events.append(evt)
            if self.on_audio is not None:
                self.on_audio(mono_data)
        else:
            # Buffer audio for pre-activation
            self._pre_buffer.append(mono_data)

        return events

    def iter_events(self, audio_stream, timestamps=None):
        """Iterate over VAD events from an audio stream.

        Args:
            audio_stream: Iterable yielding raw PCM audio bytes.
            timestamps: Optional iterable of timestamps (seconds).
                        If None, uses time.time() for each chunk.

        Yields:
            VADEvent objects.
        """
        if timestamps is None:
            for chunk in audio_stream:
                for event in self.feed(chunk):
                    yield event
        else:
            for chunk, ts in zip(audio_stream, timestamps):
                for event in self.feed(chunk, timestamp=ts):
                    yield event

    def _update_threshold(self):
        """Adapt the dynamic threshold based on recent RMS history."""
        if len(self._history_level) == 0:
            return
        history_pct = self._get_history_percentage()
        if self._dynamic_threshold <= 0:
            self._dynamic_threshold = history_pct
            return
        if history_pct > self._dynamic_threshold:
            # Slow rise - avoids false positives from transient noise
            self._dynamic_threshold += (
                (history_pct - self._dynamic_threshold) * self._config.adapt_up_rate
            )
        elif history_pct < self._dynamic_threshold:
            # Fast drop - quickly adapts to quieter environments
            self._dynamic_threshold += (
                (history_pct - self._dynamic_threshold) * self._config.adapt_down_rate
            )

    def _get_history_percentage(self):
        """Calculate smoothed history percentage with hysteresis."""
        avg = self._get_history_average()
        return (
            (avg / self._config.max_level) * self._config.hysteresis_multiply
            + self._config.hysteresis_offset
        )

    def _get_history_average(self):
        """Calculate average of recent RMS levels."""
        window = self._config.avg_window
        total = 0
        count = 0
        for i in range(len(self._history_level) - 1, -1, -1):
            total += self._history_level[i]
            count += 1
            if count >= window:
                break
        if count == 0:
            return 0
        return total / count
