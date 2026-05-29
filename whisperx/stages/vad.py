from abc import ABC, abstractmethod

import numpy as np

SAMPLE_RATE = 16000


class VADBase(ABC):
    @abstractmethod
    def detect(self, audio: np.ndarray) -> list[dict]:
        """음성 구간 검출. Returns: [{"start": float, "end": float}, ...]  (seconds)"""


class SileroVAD(VADBase):
    """faster-whisper 내장 Silero VAD. HF_TOKEN 불필요."""

    def __init__(self, cfg: dict):
        from faster_whisper.vad import SileroVADModel, VadOptions

        self._model = SileroVADModel()
        self._options = VadOptions(
            onset=cfg.get("onset", 0.500),
            offset=cfg.get("offset", 0.363),
            min_speech_duration_ms=cfg.get("min_speech_duration_ms", 250),
            min_silence_duration_ms=cfg.get("min_silence_duration_ms", 2000),
        )

    def detect(self, audio: np.ndarray) -> list[dict]:
        import torch

        timestamps = self._model(torch.from_numpy(audio), SAMPLE_RATE)
        return [
            {"start": t["start"] / SAMPLE_RATE, "end": t["end"] / SAMPLE_RATE}
            for t in timestamps
        ]


class PyannoteVAD(VADBase):
    """
    pyannote 기반 VAD. model_id로 HF의 어떤 pyannote VAD 모델이든 지정 가능.
    예) "pyannote/voice-activity-detection"
        "pyannote/segmentation-3.0"
    HF_TOKEN 환경변수 필요.
    """

    def __init__(self, cfg: dict):
        import os
        from pyannote.audio import Pipeline

        model_id = cfg.get("model_id", "pyannote/voice-activity-detection")
        token = cfg.get("hf_token") or os.environ.get("HF_TOKEN")
        self._pipeline = Pipeline.from_pretrained(model_id, token=token)

        # pyannote.audio.Pipeline.from_pretrained() 은 device 인자를 받지 않으므로
        # 로드 후 명시적으로 옮긴다. cfg.device 가 없으면 cuda 가 기본.
        device = cfg.get("device", "cuda")
        if device == "cuda":
            import torch
            if torch.cuda.is_available():
                self._pipeline.to(torch.device("cuda"))

    def detect(self, audio: np.ndarray) -> list[dict]:
        import torch

        waveform = torch.tensor(audio).unsqueeze(0)
        output = self._pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        return [
            {"start": seg.start, "end": seg.end}
            for seg in output.get_timeline().support()
        ]


VAD_REGISTRY: dict[str, type[VADBase]] = {
    "silero": SileroVAD,
    "pyannote": PyannoteVAD,
}
