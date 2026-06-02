"""
bci — real-time EEG streaming and CSA pipeline for Emotiv headsets.

Submodules
----------
csa      : Compressed Spectral Array (CSA/qEEG) computation
stream   : Emotiv Cortex WebSocket streaming (EmotivStream, PowStream)
"""

from .csa import compute_csa, CSAResult
from .stream import EmotivStream, stream_csa, PowStream, stream_pow, PowFrame

__all__ = [
    "compute_csa",
    "CSAResult",
    "EmotivStream",
    "stream_csa",
    "PowStream",
    "stream_pow",
    "PowFrame",
]
