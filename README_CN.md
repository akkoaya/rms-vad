# rms-vad

[![PyPI version](https://img.shields.io/pypi/v/rms-vad.svg)](https://pypi.org/project/rms-vad/)
[![Python](https://img.shields.io/pypi/pyversions/rms-vad.svg)](https://pypi.org/project/rms-vad/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

轻量级基于 RMS 的语音活动检测（VAD），专为 ASR 流水线设计。

核心零依赖（仅使用标准库 `audioop`）。可选 `numpy` 加速多声道转换。

[English](README.md) | [中文](README_CN.md)

## 特性

- **零依赖** - 仅使用 Python 标准库中的 `audioop`
- **自适应阈值** - 自动适应环境噪声水平
- **预缓冲** - 捕获语音起始前的音频，避免截断起始音素
- **多种 API 风格** - 事件驱动、回调和迭代器三种接口
- **多声道支持** - 自动转换为单声道（可选 `numpy` 加速）
- **轻量高效** - 纯 Python 实现，CPU 开销极低
- **参数校验** - 构造时自动校验所有配置参数，提前捕获错误配置
- **语音时长控制** - 超时自动结束长语音段，过短语音段自动丢弃
- **丰富的事件元数据** - 每个事件携带时间戳、RMS 电平和语音时长
- **运行时统计** - 通过 `get_stats()` 获取语音占比、段数、RMS 最小/最大/平均值
- **音频工具集** - WAV 读写、能量 dB 计算、信噪比估算、`AudioCollector` 语音段收集器
- **上下文管理器** - `with RmsVAD() as vad:` 自动清理状态

## 安装

```bash
pip install rms-vad

# 安装 numpy 支持（推荐用于多声道音频）
pip install rms-vad[numpy]
```

## 快速开始

### 事件驱动风格

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
            print("语音时长:", event.duration, "秒")
            asr.end()
```

### 回调风格

```python
vad = RmsVAD(VADConfig(threshold=0.3))

vad.on_speech_start = lambda pre_buf: asr.start()
vad.on_audio = lambda chunk: asr.send(chunk)
vad.on_speech_end = lambda: asr.end()

for chunk in mic_stream:
    vad.feed(chunk)
```

### 迭代器风格

```python
for event in vad.iter_events(mic_stream):
    ...
```

### AudioCollector - 收集完整语音段

```python
from rms_vad import RmsVAD, AudioCollector

vad = RmsVAD()
collector = AudioCollector()

for chunk in mic_stream:
    for event in vad.feed(chunk):
        segment = collector.feed(event)
        if segment is not None:
            # segment.audio    - 完整 PCM 字节数据
            # segment.duration - 语音时长（秒）
            segment.save_wav("speech_{}.wav".format(segment.segment_index))
```

### 上下文管理器

```python
with RmsVAD(VADConfig(max_speech_duration=30)) as vad:
    for chunk in mic_stream:
        for event in vad.feed(chunk):
            ...
# 退出时自动重置状态
```

### 运行时监控

```python
vad = RmsVAD()
for chunk in mic_stream:
    vad.feed(chunk)
    print("电平:", vad.current_level, "阈值:", vad.threshold)

stats = vad.get_stats()
print("语音占比:", stats["speech_ratio"])
print("语音段数:", stats["speech_segments"])
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `sample_rate` | 16000 | 音频采样率（Hz） |
| `sample_width` | 2 | 每个采样的字节数（2 = 16-bit PCM） |
| `channels` | 1 | 音频声道数（自动转换为单声道） |
| `chunk_size` | 1024 | 每个音频块的采样数 |
| `max_level` | 25000 | RMS 归一化基准值 |
| `threshold` | 0.5 | 初始 VAD 阈值（0.0~1.0） |
| `attack` | 0.2 | 确认语音起始前的等待时间（秒） |
| `release` | 1.5 | 静音多久后判定语音结束（秒） |
| `history_size` | 500 | RMS 历史缓冲区大小 |
| `avg_window` | 30 | 移动平均的帧窗口大小 |
| `pre_buffer_size` | 10 | 预激活缓冲帧数 |
| `adapt_up_rate` | 0.0025 | 阈值缓慢上升速率 |
| `adapt_down_rate` | 1.0 | 阈值快速下降速率 |
| `hysteresis_multiply` | 1.05 | 历史百分比乘数 |
| `hysteresis_offset` | 0.02 | 历史百分比偏移量 |
| `max_speech_duration` | 0 | 超过 N 秒自动结束语音（0 = 不限制） |
| `min_speech_duration` | 0 | 丢弃短于 N 秒的语音段（0 = 不限制） |

所有参数在构造时自动校验，无效值会抛出 `ValueError` 并附带清晰的错误信息。

## 事件类型

| 事件 | 说明 | 关键字段 |
|------|------|----------|
| `SPEECH_START` | 检测到语音起始 | `pre_buffer`、`timestamp`、`rms_level` |
| `AUDIO` | 活跃语音期间的音频块 | `chunk`、`timestamp`、`rms_level` |
| `SPEECH_END` | 静音超时后语音结束 | `timestamp`、`rms_level`、`duration` |
| `SPEECH_TIMEOUT` | 语音超时自动结束（超过最大时长） | `timestamp`、`rms_level`、`duration` |

## 音频工具集

```python
from rms_vad import pcm_to_wav, wav_to_pcm, save_wav, compute_energy_db, compute_snr

# PCM <-> WAV 互转
wav_bytes = pcm_to_wav(pcm_data, sample_rate=16000)
pcm_data, sr, sw, ch = wav_to_pcm(wav_bytes)
save_wav("output.wav", pcm_data)

# 音频分析
db = compute_energy_db(pcm_data)           # 能量（dB）
snr = compute_snr(speech_pcm, noise_pcm)   # 信噪比（dB）
```

## 算法原理

1. **RMS 计算** - 对每个音频块执行 `audioop.rms(buffer, 2)`，归一化到 `[0, 1]`
2. **移动平均** - 30 帧窗口，带滞回处理（`*1.05 + 0.02`）
3. **自适应阈值** - 非对称调整：缓慢上升（每帧 0.25%），快速下降（每帧 100%）
4. **状态机** - `静音 <-> 说话`，带 attack/release 去抖
5. **预缓冲** - 捕获语音起始前的音频帧，避免截断起始音素
6. **时长保护** - 可选的最大/最小语音时长限制

## 环境要求

- Python >= 3.8
- 无必需依赖（使用标准库 `audioop`）
- 可选：`numpy`，用于高效的多声道转单声道转换

## 参与贡献

欢迎贡献代码！请随时提交 Pull Request。

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 发起 Pull Request

### 开发环境搭建

```bash
git clone https://github.com/akkoaya/rms-vad.git
cd rms-vad
pip install -e ".[dev]"
pytest
```

## 许可证

本项目基于 MIT 许可证开源 - 详见 [LICENSE](LICENSE) 文件。
