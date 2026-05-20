from abc import ABC, abstractmethod

import numpy as np

SAMPLE_RATE = 16000


class STTBase(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, segments: list[dict] | None = None) -> list[dict]:
        """
        audio   : float32 numpy array @ 16kHz
        segments: VAD로 사전 검출한 구간 [{"start": float, "end": float}].
                  None이면 모델 내부 VAD 사용.
        Returns : [{"start": float, "end": float, "text": str}, ...]
        """


class FasterWhisperSTT(STTBase):
    """
    CTranslate2 포맷 모델 전용. model_id에 HF ID 또는 로컬 경로 지정.
    예) "Systran/faster-whisper-large-v3", "large-v3", "./my-model"
    """

    def __init__(self, cfg: dict):
        from faster_whisper import WhisperModel

        self._language = cfg.get("language", "ko")
        self._beam_size = cfg.get("beam_size", 5)
        self._temperature = cfg.get("temperature", 0)
        self._model = WhisperModel(
            cfg["model_id"],
            device=cfg.get("device", "cuda"),
            compute_type=cfg.get("compute_type", "float16"),
        )

    def transcribe(self, audio: np.ndarray, segments: list[dict] | None = None) -> list[dict]:
        if segments is None:
            return self._transcribe_full(audio)
        return self._transcribe_segments(audio, segments)

    def _transcribe_full(self, audio: np.ndarray) -> list[dict]:
        raw, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            temperature=self._temperature,
            vad_filter=True,
        )
        return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in raw]

    def _transcribe_segments(self, audio: np.ndarray, segments: list[dict]) -> list[dict]:
        results = []
        for seg in segments:
            start_s, end_s = seg["start"], seg["end"]
            chunk = audio[int(start_s * SAMPLE_RATE): int(end_s * SAMPLE_RATE)]
            if len(chunk) == 0:
                continue
            raw, _ = self._model.transcribe(
                chunk,
                language=self._language,
                beam_size=self._beam_size,
                temperature=self._temperature,
                vad_filter=False,
            )
            for s in raw:
                results.append({
                    "start": round(start_s + s.start, 3),
                    "end": round(start_s + s.end, 3),
                    "text": s.text.strip(),
                })
        return results


class TransformersSTT(STTBase):
    """
    HuggingFace Transformers ASR 파이프라인. model_id로 어떤 HF 모델이든 사용 가능.
    예) "openai/whisper-large-v3", "SungBeom/whisper-small-ko", "kresnik/wav2vec2-large-xlsr-korean"
    """

    def __init__(self, cfg: dict):
        import torch
        from transformers import pipeline as hf_pipeline

        device = cfg.get("device", "cuda")
        dtype = torch.float16 if cfg.get("compute_type", "float16") == "float16" else torch.float32

        self._language = cfg.get("language", "ko")
        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=cfg["model_id"],
            device=0 if device == "cuda" and torch.cuda.is_available() else -1,
            torch_dtype=dtype,
            generate_kwargs={
                "language": cfg.get("language", "ko"),
                "task": "transcribe",
            },
        )

    def transcribe(self, audio: np.ndarray, segments: list[dict] | None = None) -> list[dict]:
        if segments is None:
            return self._transcribe_full(audio)
        return self._transcribe_segments(audio, segments)

    def _transcribe_full(self, audio: np.ndarray) -> list[dict]:
        result = self._pipe(audio.copy(), return_timestamps=True)
        chunks = result.get("chunks") or []
        if not chunks:
            return [{"start": 0.0, "end": 0.0, "text": result["text"].strip()}]
        return [
            {
                "start": c["timestamp"][0] or 0.0,
                "end": c["timestamp"][1] or 0.0,
                "text": c["text"].strip(),
            }
            for c in chunks
        ]

    def _transcribe_segments(self, audio: np.ndarray, segments: list[dict]) -> list[dict]:
        results = []
        for seg in segments:
            start_s, end_s = seg["start"], seg["end"]
            chunk = audio[int(start_s * SAMPLE_RATE): int(end_s * SAMPLE_RATE)]
            if len(chunk) == 0:
                continue
            result = self._pipe(chunk.copy(), return_timestamps=True)
            chunks = result.get("chunks") or []
            if chunks:
                for c in chunks:
                    results.append({
                        "start": round(start_s + (c["timestamp"][0] or 0.0), 3),
                        "end": round(start_s + (c["timestamp"][1] or 0.0), 3),
                        "text": c["text"].strip(),
                    })
            else:
                results.append({
                    "start": start_s,
                    "end": end_s,
                    "text": result["text"].strip(),
                })
        return results


class WhisperXSTT(STTBase):
    """
    whisperx 전체 파이프라인 (transcribe + align).
    외부 VAD 세그먼트는 무시하고 whisperx 내부 VAD를 사용한다.
    config에서 vad.model: "disabled" 로 설정 권장.
    """

    def __init__(self, cfg: dict):
        import whisperx

        self._device = cfg.get("device", "cuda")
        self._language = cfg.get("language", "ko")
        self._batch_size = cfg.get("batch_size", 16)
        self._do_align = cfg.get("align", True)
        # beam_size / temperature 는 transcribe() 인자가 아닌 load_model() asr_options 로 전달
        asr_options = {
            "beam_size": cfg.get("beam_size", 5),
            "temperatures": [cfg.get("temperature", 0)],
        }
        self._model = whisperx.load_model(
            cfg["model_id"],
            device=self._device,
            compute_type=cfg.get("compute_type", "float16"),
            language=self._language,
            asr_options=asr_options,
        )
        # 정렬 모델은 파일마다 로드하면 매우 느리므로 초기화 시 한 번만 로드
        self._align_model = None
        self._align_metadata = None
        if self._do_align:
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=self._language, device=self._device
            )

    def transcribe(self, audio: np.ndarray, segments: list[dict] | None = None) -> list[dict]:
        # segments 인자는 무시 — whisperx 내부 VAD 사용
        import whisperx

        result = self._model.transcribe(
            audio,
            batch_size=self._batch_size,
            language=self._language,
        )
        if self._do_align:
            result = whisperx.align(
                result["segments"], self._align_model, self._align_metadata,
                audio, self._device, return_char_alignments=False,
            )
        return [
            {"start": s["start"], "end": s["end"], "text": s.get("text", "").strip()}
            for s in result["segments"]
        ]


STT_REGISTRY: dict[str, type[STTBase]] = {
    "whisperx": WhisperXSTT,
    "faster-whisper": FasterWhisperSTT,
    "transformers": TransformersSTT,
}
