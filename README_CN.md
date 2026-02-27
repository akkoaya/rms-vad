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

## 算法原理

1. **RMS 计算** - 对每个音频块执行 `audioop.rms(buffer, 2)`，归一化到 `[0, 1]`
2. **移动平均** - 30 帧窗口，带滞回处理（`*1.05 + 0.02`）
3. **自适应阈值** - 非对称调整：缓慢上升（每帧 0.25%），快速下降（每帧 100%）
4. **状态机** - `静音 <-> 说话`，带 attack/release 去抖
5. **预缓冲** - 捕获语音起始前的音频帧，避免截断起始音素

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
