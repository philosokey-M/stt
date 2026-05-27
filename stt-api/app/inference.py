"""
ModelManager — 모델은 startup 시 1회 로드되고 프로세스가 사는 동안 재사용.

동시성 제어:
- Semaphore(MAX_CONCURRENT_INFERENCES) 로 GPU/모델에 동시에 진입할 수 있는 추론 수 제한.
- 추론 자체는 asyncio.to_thread 로 워커 스레드에서 실행 → 이벤트 루프 블로킹 없음.
- 다른 요청들은 큐에서 비동기로 대기 → 업로드, /health, /jobs 폴링 등은 그대로 응답.

확장 가능성:
- 동적 배치(micro-batching) 가 필요해지면 본 클래스에 request 큐 + 1초 윈도우 모으는
  배치 워커 스레드를 추가하면 된다. 외부 인터페이스(transcribe()) 는 그대로.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from .audio import audio_duration_seconds, decode_audio
from .language_router import LanguageRouter
from .pipeline_adapter import Pipeline, build_from_config
from .settings import APISettings

log = logging.getLogger(__name__)


@dataclass
class TranscribeResult:
    segments: list[dict[str, Any]]
    elapsed_s: float
    audio_duration_s: float
    language: str = ""

    @property
    def rtf(self) -> float | None:
        if self.audio_duration_s <= 0:
            return None
        return self.elapsed_s / self.audio_duration_s


class ModelManager:
    """싱글톤 모델 보관소. startup 에서 load() 후 transcribe() 호출."""

    def __init__(self, settings: APISettings):
        self._settings = settings
        self._pipeline: Pipeline | None = None
        self._router: LanguageRouter | None = None
        self._semaphore = asyncio.Semaphore(settings.server.max_concurrent_inferences)
        self._stats = {"requests": 0, "errors": 0, "total_elapsed_s": 0.0}

    # ---------------------------------------------------------- lifecycle

    def load(self) -> None:
        if self._pipeline is not None:
            return
        log.info(
            "STT 파이프라인 로딩… backend=%s model=%s device=%s diarization=%s",
            self._settings.pipeline_config.get("pipeline", {}).get("stt", {}).get("backend"),
            self._settings.pipeline_config.get("pipeline", {}).get("stt", {}).get("model_id"),
            self._settings.pipeline_config.get("pipeline", {}).get("stt", {}).get("device"),
            self._settings.pipeline_config.get("pipeline", {}).get("diarization", {}).get("enabled"),
        )
        t0 = time.perf_counter()
        self._pipeline = build_from_config(self._settings.pipeline_config)

        stt_cfg = self._settings.pipeline_config.get("pipeline", {}).get("stt", {})
        self._router = LanguageRouter(
            pipeline=self._pipeline,
            device=stt_cfg.get("device", "cuda"),
            default_language=stt_cfg.get("language", "ko"),
        )
        log.info("STT 파이프라인 로드 완료 (%.2fs)", time.perf_counter() - t0)

    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def info(self) -> dict[str, Any]:
        p = self._settings.pipeline_config.get("pipeline", {})
        return {
            "ready": self.is_ready(),
            "max_concurrent_inferences": self._settings.server.max_concurrent_inferences,
            "in_flight": self._settings.server.max_concurrent_inferences - self._semaphore._value,  # type: ignore[attr-defined]
            "backend": p.get("stt", {}).get("backend"),
            "model_id": p.get("stt", {}).get("model_id"),
            "device": p.get("stt", {}).get("device"),
            "compute_type": p.get("stt", {}).get("compute_type"),
            "default_language": p.get("stt", {}).get("language"),
            "cached_align_languages": self._router.cached_languages() if self._router else [],
            "diarization_enabled": p.get("diarization", {}).get("enabled", False),
            "vad_model": p.get("vad", {}).get("model"),
            "stats": dict(self._stats),
        }

    # ---------------------------------------------------------- inference

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscribeResult:
        """
        요청별 language 를 받아 LanguageRouter 가 stt 객체의 언어/align 모델을
        일시적으로 바꾼다. /whisperx 코드는 수정하지 않는다.

        first-call-of-language: 해당 언어의 align 모델 (~300MB) 다운로드/로드 비용.
        cached: 즉시 사용.
        """
        if self._pipeline is None or self._router is None:
            raise RuntimeError("모델이 아직 로드되지 않았습니다.")

        self._stats["requests"] += 1

        # CPU 디코딩 + duration 측정 (스레드 풀)
        duration = await asyncio.to_thread(audio_duration_seconds, audio_path, None)

        # GPU 진입은 세마포어로 제한
        async with self._semaphore:
            t0 = time.perf_counter()
            try:
                result, used_language = await asyncio.to_thread(
                    self._transcribe_with_language, audio_path, language
                )
            except Exception:
                self._stats["errors"] += 1
                raise
            finally:
                elapsed = time.perf_counter() - t0
                self._stats["total_elapsed_s"] += elapsed

        segments = self._coerce_segments(result.get("segments", []))
        return TranscribeResult(
            segments=segments,
            elapsed_s=round(elapsed, 3),
            audio_duration_s=round(duration, 3),
            language=used_language,
        )

    def _transcribe_with_language(
        self,
        audio_path: str,
        language: str | None,
    ) -> tuple[dict, str]:
        """워커 스레드에서 실행되는 동기 경로 — 락 + monkey-patch + run."""
        assert self._router is not None
        with self._router.use(language) as used_lang:
            result = self._pipeline.run(audio_path)  # type: ignore[union-attr]
        return result, used_lang

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _coerce_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """API 응답 스키마(start/end/text/speaker)로 강제 정렬."""
        out = []
        for s in segments:
            out.append({
                "start": float(s.get("start", 0.0) or 0.0),
                "end": float(s.get("end", 0.0) or 0.0),
                "text": (s.get("text") or "").strip(),
                "speaker": s.get("speaker") or None,
            })
        return out


_manager: ModelManager | None = None


def get_manager() -> ModelManager:
    global _manager
    if _manager is None:
        raise RuntimeError("ModelManager 가 초기화되지 않았습니다. main.py 의 lifespan 확인.")
    return _manager


def init_manager(settings: APISettings) -> ModelManager:
    global _manager
    _manager = ModelManager(settings)
    _manager.load()
    return _manager
