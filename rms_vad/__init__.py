"""rms-vad: Lightweight RMS-based Voice Activity Detection for ASR pipelines."""

from .config import VADConfig
from .core import RmsVAD
from .events import VADEvent, VADEventType

__version__ = "0.1.0"
__all__ = ["RmsVAD", "VADConfig", "VADEvent", "VADEventType"]
