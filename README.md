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
- **Config validation** - catches misconfiguration early with clear error messages
- **Max / Min speech duration** - auto-end long segments, discard too-short ones
- **Rich event metadata** - every event carries timestamp, RMS level and speech duration
- **Runtime statistics** - speech ratio, segment count, min/max/avg RMS via `get_stats()`
- **Audio utilities** - WAV I/O, energy dB, SNR estimation, `AudioCollector` for easy segment collection
- **Context manager** - `with RmsVAD() as vad:` for automatic cleanup

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
            print("Speech duration:", event.duration, "s")
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

### AudioCollector - collect complete speech segments

```python
from rms_vad import RmsVAD, AudioCollector

vad = RmsVAD()
collector = AudioCollector()

for chunk in mic_stream:
    for event in vad.feed(chunk):
        segment = collector.feed(event)
        if segment is not None:
            # segment.audio  - complete PCM bytes
            # segment.duration - speech length in seconds
            segment.save_wav("speech_{}.wav".format(segment.segment_index))
```

### Context manager

```python
with RmsVAD(VADConfig(max_speech_duration=30)) as vad:
    for chunk in mic_stream:
        for event in vad.feed(chunk):
            ...
# state auto-reset on exit
```

### Runtime monitoring

```python
vad = RmsVAD()
for chunk in mic_stream:
    vad.feed(chunk)
    print("Level:", vad.current_level, "Threshold:", vad.threshold)

stats = vad.get_stats()
print("Speech ratio:", stats["speech_ratio"])
print("Segments:", stats["speech_segments"])
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
| `max_speech_duration` | 0 | Auto-end speech after N seconds (0 = disabled) |
| `min_speech_duration` | 0 | Discard segments shorter than N seconds (0 = disabled) |

All parameters are validated on construction. Invalid values raise `ValueError` with a clear message.

## Event Types

| Event | Description | Key Fields |
|-------|-------------|------------|
| `SPEECH_START` | Speech onset detected | `pre_buffer`, `timestamp`, `rms_level` |
| `AUDIO` | Audio chunk during active speech | `chunk`, `timestamp`, `rms_level` |
| `SPEECH_END` | Speech ended after silence | `timestamp`, `rms_level`, `duration` |
| `SPEECH_TIMEOUT` | Speech auto-ended (max duration) | `timestamp`, `rms_level`, `duration` |

## Audio Utilities

```python
from rms_vad import pcm_to_wav, wav_to_pcm, save_wav, compute_energy_db, compute_snr

# PCM <-> WAV conversion
wav_bytes = pcm_to_wav(pcm_data, sample_rate=16000)
pcm_data, sr, sw, ch = wav_to_pcm(wav_bytes)
save_wav("output.wav", pcm_data)

# Audio analysis
db = compute_energy_db(pcm_data)           # energy in dB
snr = compute_snr(speech_pcm, noise_pcm)   # SNR in dB
```

## Algorithm

1. **RMS calculation** - `audioop.rms(buffer, 2)` on each chunk, normalized to `[0, 1]`
2. **Moving average** - 30-frame window with hysteresis (`*1.05 + 0.02`)
3. **Adaptive threshold** - asymmetric: slow rise (0.25%/frame), fast drop (100%/frame)
4. **State machine** - `SILENCE <-> SPEAKING` with attack/release debouncing
5. **Pre-buffer** - captures audio before speech onset to avoid clipping leading phonemes
6. **Duration guard** - optional max/min speech duration enforcement

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
