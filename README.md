# rms-vad

[![PyPI version](https://img.shields.io/pypi/v/rms-vad.svg)](https://pypi.org/project/rms-vad/)
[![Python](https://img.shields.io/pypi/pyversions/rms-vad.svg)](https://pypi.org/project/rms-vad/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight RMS-based Voice Activity Detection for ASR pipelines.

Zero-dependency core (only `audioop` from stdlib). Optional `numpy` for faster multi-channel conversion.

[English](README.md) | [中文](README_CN.md)

## Features

- **Zero dependencies** - only uses `audioop` from Python stdlib
- **Adaptive threshold** - automatically adjusts to ambient noise levels
- **Pre-buffering** - captures audio before speech onset to avoid clipping leading phonemes
- **Multiple API styles** - event-driven, callback, and iterator interfaces
- **Multi-channel support** - auto-converts to mono (optional `numpy` acceleration)
- **Lightweight & fast** - pure Python, minimal CPU overhead

## Installation

```bash
pip install rms-vad

# With numpy (recommended for multi-channel audio)
pip install rms-vad[numpy]
```

## Quick Start

### Event-driven style

```python
from rms_vad import RmsVAD, VADConfig, VADEventType

vad = RmsVAD(VADConfig(threshold=0.5, attack=0.2, release=1.5))

for chunk in mic_stream:
    for event in vad.feed(chunk):
        if event.type == VADEventType.SPEECH_START:
            asr.start()
            for frame in event.pre_buffer:
                asr.send(frame)
        elif event.type == VADEventType.AUDIO:
            asr.send(event.chunk)
        elif event.type == VADEventType.SPEECH_END:
            asr.end()
```

### Callback style

```python
vad = RmsVAD(VADConfig(threshold=0.3))

vad.on_speech_start = lambda pre_buf: asr.start()
vad.on_audio = lambda chunk: asr.send(chunk)
vad.on_speech_end = lambda: asr.end()

for chunk in mic_stream:
    vad.feed(chunk)
```

### Iterator style

```python
for event in vad.iter_events(mic_stream):
    ...
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | Audio sample rate (Hz) |
| `sample_width` | 2 | Bytes per sample (2 = 16-bit PCM) |
| `channels` | 1 | Audio channels (auto-converts to mono) |
| `chunk_size` | 1024 | Samples per chunk |
| `max_level` | 25000 | RMS normalization baseline |
| `threshold` | 0.5 | Initial VAD threshold (0.0~1.0) |
| `attack` | 0.2 | Seconds before speech onset confirmed |
| `release` | 1.5 | Seconds of silence before speech end |
| `history_size` | 500 | RMS history buffer size |
| `avg_window` | 30 | Frames for moving average |
| `pre_buffer_size` | 10 | Pre-activation frame count |
| `adapt_up_rate` | 0.0025 | Threshold slow-rise rate |
| `adapt_down_rate` | 1.0 | Threshold fast-drop rate |
| `hysteresis_multiply` | 1.05 | History percentage multiplier |
| `hysteresis_offset` | 0.02 | History percentage offset |

## Algorithm

1. **RMS calculation** - `audioop.rms(buffer, 2)` on each chunk, normalized to `[0, 1]`
2. **Moving average** - 30-frame window with hysteresis (`*1.05 + 0.02`)
3. **Adaptive threshold** - asymmetric: slow rise (0.25%/frame), fast drop (100%/frame)
4. **State machine** - `SILENCE <-> SPEAKING` with attack/release debouncing
5. **Pre-buffer** - captures audio before speech onset to avoid clipping leading phonemes

## Requirements

- Python >= 3.8
- No required dependencies (uses `audioop` from stdlib)
- Optional: `numpy` for efficient multi-channel to mono conversion

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/akkoaya/rms-vad.git
cd rms-vad
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
