"""rms-vad: Lightweight RMS-based Voice Activity Detection for ASR pipelines."""

from .config import VADConfig
from .core import RmsVAD
from .events import VADEvent, VADEventType
from .audio_utils import (
    AudioCollector,
    SpeechSegment,
    pcm_to_wav,
    wav_to_pcm,
    save_wav,
    compute_energy_db,
    compute_snr,
)

__version__ = "0.2.0"
__all__ = [
    "RmsVAD",
    "VADConfig",
    "VADEvent",
    "VADEventType",
    "AudioCollector",
    "SpeechSegment",
    "pcm_to_wav",
    "wav_to_pcm",
    "save_wav",
    "compute_energy_db",
    "compute_snr",
]
