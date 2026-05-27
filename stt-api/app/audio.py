"""
오디오 디코딩 & 길이 측정. /whisperx 와 동일하게 16kHz mono float32.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def decode_audio(path: str | Path) -> np.ndarray:
    """faster-whisper 의 decode_audio 를 우선 사용 (ffmpeg 백엔드)."""
    from faster_whisper.audio import decode_audio as _decode

    return _decode(str(path), sampling_rate=SAMPLE_RATE)


def audio_duration_seconds(path: str | Path, audio: np.ndarray | None = None) -> float:
    """가능하면 WAV 헤더로 빠르게, 안 되면 디코딩된 샘플 길이로 계산."""
    p = Path(path)
    if audio is not None:
        return float(len(audio)) / SAMPLE_RATE
    try:
        with wave.open(str(p)) as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        # WAV 가 아니면 디코딩 후 길이 계산
        return float(len(decode_audio(p))) / SAMPLE_RATE
