"""
Pipeline: VAD → STT → Diarization 조립 및 실행.
config.yaml의 pipeline 섹션으로 각 단계 모델을 교체한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stages.diarizer import DIARIZER_REGISTRY, DiarizationBase
from stages.stt import STT_REGISTRY, FasterWhisperSTT, STTBase
from stages.vad import VAD_REGISTRY, VADBase


def load_config(config_path: str | Path) -> dict:
    """config.yaml을 로드하고, 같은 디렉토리의 config.local.yaml이 있으면 덮어씀."""
    import yaml

    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    local_path = config_path.with_name(config_path.stem + ".local" + config_path.suffix)
    if local_path.exists():
        with open(local_path) as f:
            local = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, local)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_audio(audio_path: str) -> np.ndarray:
    from faster_whisper.audio import decode_audio
    return decode_audio(audio_path, sampling_rate=16000)


class Pipeline:
    def __init__(
        self,
        vad: VADBase | None,
        stt: STTBase,
        diarizer: DiarizationBase | None,
    ):
        self.vad = vad
        self.stt = stt
        self.diarizer = diarizer

    def run(self, audio_path: str) -> dict:
        audio = _load_audio(audio_path)

        # 1. VAD
        vad_segments = self.vad.detect(audio) if self.vad is not None else None

        # 2. STT
        segments = self.stt.transcribe(audio, vad_segments)

        # 3. Diarization
        if self.diarizer is not None:
            segments = self.diarizer.diarize(audio, segments)

        return {"segments": segments}

    @property
    def config_tag(self) -> str:
        """결과 파일명에 쓸 모델 조합 태그."""
        vad = type(self.vad).__name__ if self.vad else "NoVAD"
        stt = type(self.stt).__name__
        diar = type(self.diarizer).__name__ if self.diarizer else "NoDiar"
        return f"{vad}__{stt}__{diar}"


def build_from_config(cfg: dict) -> Pipeline:
    p = cfg.get("pipeline", {})
    hf_token = cfg.get("hf_token") or None  # 빈 문자열은 None으로 처리

    # VAD
    vad_cfg = {**p.get("vad", {}), "hf_token": hf_token}
    vad_name = vad_cfg.get("model", "silero")
    vad = None if vad_name == "disabled" else VAD_REGISTRY[vad_name](vad_cfg)

    # STT
    stt_cfg = p.get("stt", {})
    from stages.stt import WhisperXSTT
    stt_cls = STT_REGISTRY.get(stt_cfg.get("backend", "whisperx"), WhisperXSTT)
    stt = stt_cls(stt_cfg)

    # Diarization
    diar_cfg = {**p.get("diarization", {}), "hf_token": hf_token}
    diarizer = None
    if diar_cfg.get("enabled", False):
        diarizer = DIARIZER_REGISTRY[diar_cfg.get("model", "pyannote")](diar_cfg)

    return Pipeline(vad, stt, diarizer)
