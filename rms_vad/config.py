"""VAD configuration."""


class VADConfig:
    """Configuration for RMS-based Voice Activity Detection.

    Args:
        sample_rate: Audio sample rate in Hz. Default 16000.
        sample_width: Bytes per sample (2 = 16-bit PCM). Default 2.
        channels: Number of audio channels. Default 1.
        chunk_size: Samples per audio chunk. Default 1024.
        max_level: Normalization baseline for RMS. Default 25000.
        threshold: Initial dynamic threshold (0.0~1.0). Default 0.5.
        attack: Seconds of sustained volume before speech onset. Default 0.2.
        release: Seconds of silence before speech end. Default 1.5.
        history_size: Max RMS history frames for moving average. Default 500.
        avg_window: Number of recent frames for moving average. Default 30.
        pre_buffer_size: Pre-activation audio frame count. Default 10.
        adapt_up_rate: Threshold upward adaptation rate. Default 0.0025.
        adapt_down_rate: Threshold downward adaptation rate. Default 1.0.
        hysteresis_multiply: Multiplier for history percentage. Default 1.05.
        hysteresis_offset: Offset added to history percentage. Default 0.02.
    """

    __slots__ = (
        "sample_rate",
        "sample_width",
        "channels",
        "chunk_size",
        "max_level",
        "threshold",
        "attack",
        "release",
        "history_size",
        "avg_window",
        "pre_buffer_size",
        "adapt_up_rate",
        "adapt_down_rate",
        "hysteresis_multiply",
        "hysteresis_offset",
    )

    def __init__(
        self,
        sample_rate=16000,
        sample_width=2,
        channels=1,
        chunk_size=1024,
        max_level=25000,
        threshold=0.5,
        attack=0.2,
        release=1.5,
        history_size=500,
        avg_window=30,
        pre_buffer_size=10,
        adapt_up_rate=0.0025,
        adapt_down_rate=1.0,
        hysteresis_multiply=1.05,
        hysteresis_offset=0.02,
    ):
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels
        self.chunk_size = chunk_size
        self.max_level = max_level
        self.threshold = threshold
        self.attack = attack
        self.release = release
        self.history_size = history_size
        self.avg_window = avg_window
        self.pre_buffer_size = pre_buffer_size
        self.adapt_up_rate = adapt_up_rate
        self.adapt_down_rate = adapt_down_rate
        self.hysteresis_multiply = hysteresis_multiply
        self.hysteresis_offset = hysteresis_offset

    def __repr__(self):
        fields = ", ".join(
            "{}={!r}".format(name, getattr(self, name)) for name in self.__slots__
        )
        return "VADConfig({})".format(fields)
