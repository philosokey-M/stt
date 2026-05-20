from abc import ABC, abstractmethod

import numpy as np

SAMPLE_RATE = 16000


class DiarizationBase(ABC):
    @abstractmethod
    def diarize(self, audio: np.ndarray, segments: list[dict]) -> list[dict]:
        """각 세그먼트에 'speaker' 키를 추가해 반환."""


class PyannotesDiarizer(DiarizationBase):
    """
    pyannote 기반 화자분리. model_id로 HF의 어떤 pyannote 화자분리 모델이든 지정 가능.
    예) "pyannote/speaker-diarization-3.1"
        "pyannote/speaker-diarization"
    HF_TOKEN 환경변수 필요.
    """

    def __init__(self, cfg: dict):
        import os
        from pyannote.audio import Pipeline

        model_id = cfg.get("model_id", "pyannote/speaker-diarization-3.1")
        token = cfg.get("hf_token") or os.environ.get("HF_TOKEN")
        self._pipeline = Pipeline.from_pretrained(model_id, use_auth_token=token)
        self._min_speakers = cfg.get("min_speakers")
        self._max_speakers = cfg.get("max_speakers")

    def diarize(self, audio: np.ndarray, segments: list[dict]) -> list[dict]:
        import torch

        waveform = torch.tensor(audio).unsqueeze(0)
        kwargs = {}
        if self._min_speakers:
            kwargs["min_speakers"] = self._min_speakers
        if self._max_speakers:
            kwargs["max_speakers"] = self._max_speakers

        diarization = self._pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs
        )

        for seg in segments:
            mid = (seg["start"] + seg["end"]) / 2
            seg["speaker"] = "UNKNOWN"
            for turn, _, spk in diarization.itertracks(yield_label=True):
                if turn.start <= mid <= turn.end:
                    seg["speaker"] = spk
                    break
        return segments


DIARIZER_REGISTRY: dict[str, type[DiarizationBase]] = {
    "pyannote": PyannotesDiarizer,
}
