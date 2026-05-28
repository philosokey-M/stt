"""
요청별 파이프라인 오버라이드 — /whisperx 코드를 수정하지 않고 단일 파이프라인
인스턴스에서 요청마다 다른 언어 / 화자분리 설정을 적용하기 위한 어댑터.

핵심 아이디어:
- Whisper 모델 본체(large-v3 등)는 원래 다국어다. transcribe() 가 언어 파라미터를
  받기 때문에 model._language 만 바꿔치기하면 같은 모델이 다른 언어로 동작한다.
- WhisperX 의 align 모델만 언어 전용(Wav2Vec2 계열, 약 300MB). 요청된 언어별로
  lazy load + 캐시한다 → 첫 요청은 약간 느리고, 이후는 즉시.
- 화자분리는 pipeline.diarizer 자체를 일시적으로 None 으로 만들거나, 화자 수
  min/max 를 diarizer 내부 속성으로 swap 한다.
- 같은 모델 객체의 속성을 일시적으로 바꿔치기하므로 swap-transcribe-restore
  과정을 threading.Lock 으로 원자화한다 → 동시 요청은 자연히 직렬화된다.

이 라우터는 다음 백엔드를 지원한다:
- WhisperXSTT       : 언어 swap + align 모델 캐시
- FasterWhisperSTT  : 언어 swap (모델은 어차피 다국어, align 없음)
- TransformersSTT   : 베스트-에포트. 파이프라인의 generate_kwargs 까지 patch 시도.
                      실패 시 init 언어로만 동작 — 경고 로그.
- PyannotesDiarizer : 요청별 on/off + min/max speakers 오버라이드.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)


class LanguageRouter:
    def __init__(self, pipeline: Any, device: str, default_language: str):
        self._pipeline = pipeline
        self._stt = pipeline.stt
        self._device = device
        self._default_language = default_language

        # 모든 monkey-patch 와 추론을 직렬화 — 객체 속성을 바꿔치기하기 때문에 필수.
        # 외부 Semaphore 와 별개로 모델 호출 자체의 원자성을 보장한다.
        self._lock = threading.Lock()

        # WhisperX 백엔드 판별: stages/stt.py 의 WhisperXSTT 만 _align_model 보유
        self._is_whisperx = (
            hasattr(self._stt, "_align_model")
            and hasattr(self._stt, "_do_align")
        )

        # align 모델 캐시 {language -> (model, metadata)}
        self._align_cache: dict[str, tuple[Any, Any]] = {}
        if self._is_whisperx and getattr(self._stt, "_do_align", False):
            # init 시 로드된 align 모델을 캐시에 미리 등록
            self._align_cache[default_language] = (
                self._stt._align_model,
                self._stt._align_metadata,
            )

        # 알 수 없는 백엔드는 _language 만 swap 시도 — 경고 1회.
        self._warned_unsupported = False

    # ------------------------------------------------------------------ public

    @contextmanager
    def use(
        self,
        language: str | None,
        diarize: bool | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ):
        """
        with router.use("en", diarize=True, min_speakers=2, max_speakers=2):
            result = pipeline.run(audio_path)

        진입 직전에 stt/diarizer 객체의 설정을 바꾸고, finally 에서 원복.
        lock 보유 중에는 다른 요청이 같은 모델을 호출하지 못한다.

        파라미터:
            language     : None → 서버 default 사용.
            diarize      : None → pipeline 의 현재 diarizer 유무 그대로 사용.
                           True → diarizer 사용 (없으면 RuntimeError).
                           False → 이 요청만 diarization skip.
            min_speakers : None → diarizer 기본값 / 자동 추정.
                           int  → 이 요청만 해당 값으로 강제.
            max_speakers : 동일.
        """
        if not language:
            language = self._default_language

        with self._lock:
            saved = self._swap_in(language)
            diar_saved = self._swap_diarization_in(diarize, min_speakers, max_speakers)
            try:
                yield language
            finally:
                self._swap_diarization_out(diar_saved)
                self._swap_out(saved)

    def cached_languages(self) -> list[str]:
        return sorted(self._align_cache.keys())

    def has_diarization(self) -> bool:
        return self._pipeline.diarizer is not None

    # ------------------------------------------------------------------ internal

    def _swap_in(self, language: str) -> dict[str, Any]:
        """현재 stt 상태를 저장하고 새 언어로 교체."""
        saved: dict[str, Any] = {}

        if hasattr(self._stt, "_language"):
            saved["_language"] = self._stt._language
            self._stt._language = language

        if self._is_whisperx and getattr(self._stt, "_do_align", False):
            saved["_align_model"] = self._stt._align_model
            saved["_align_metadata"] = self._stt._align_metadata
            try:
                m, meta = self._ensure_align_model(language)
                self._stt._align_model = m
                self._stt._align_metadata = meta
            except Exception as e:
                # align 모델 로드 실패 → 해당 요청만 align 끄고 진행
                log.warning(
                    "언어 %s 의 align 모델을 로드하지 못함 (%s). 이 요청은 align 없이 진행.",
                    language, e,
                )
                saved["_do_align"] = self._stt._do_align
                self._stt._do_align = False

        # transformers 파이프라인은 generate_kwargs 에 언어가 박혀있음 → 추가 패치 시도
        pipe = getattr(self._stt, "_pipe", None)
        if pipe is not None:
            patched = self._patch_transformers_language(pipe, language, saved)
            if not patched and not self._warned_unsupported:
                log.warning(
                    "transformers 파이프라인의 언어 패치에 실패. 베스트-에포트로 진행."
                )
                self._warned_unsupported = True

        return saved

    def _swap_out(self, saved: dict[str, Any]) -> None:
        """저장된 원상태로 복원."""
        if "_language" in saved:
            self._stt._language = saved["_language"]
        if "_align_model" in saved:
            self._stt._align_model = saved["_align_model"]
        if "_align_metadata" in saved:
            self._stt._align_metadata = saved["_align_metadata"]
        if "_do_align" in saved:
            self._stt._do_align = saved["_do_align"]
        # transformers 패치 복원
        if "_pipe_generate_kwargs" in saved:
            self._stt._pipe._forward_params = saved["_pipe_generate_kwargs"]  # type: ignore[attr-defined]

    def _ensure_align_model(self, language: str) -> tuple[Any, Any]:
        if language in self._align_cache:
            return self._align_cache[language]

        import whisperx

        log.info("[language] align 모델 로드: %s", language)
        m, meta = whisperx.load_align_model(language_code=language, device=self._device)
        self._align_cache[language] = (m, meta)
        return m, meta

    # ------------------------------------------------------------------ diarization

    def _swap_diarization_in(
        self,
        diarize: bool | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> dict[str, Any]:
        """요청 단위로 pipeline.diarizer 와 그 내부 화자 수 한계를 바꿔둔다."""
        saved: dict[str, Any] = {}
        diarizer = self._pipeline.diarizer

        # diarize=True 인데 서버에 diarizer 가 없으면 명시적 에러
        if diarize is True and diarizer is None:
            raise RuntimeError(
                "서버에서 화자분리가 비활성화되어 있습니다. "
                ".env 의 DIARIZATION_MODE 를 auto/on 으로 두고 HF_TOKEN 을 설정한 뒤 "
                "서버를 재시작하세요."
            )

        # diarize=False → 이 요청만 diarizer 끄기
        if diarize is False and diarizer is not None:
            saved["pipeline.diarizer"] = diarizer
            self._pipeline.diarizer = None
            return saved  # diarizer 가 꺼졌으니 min/max 적용 의미 없음

        # 화자 수 오버라이드 — diarizer 가 살아있을 때만 의미 있음
        if diarizer is not None:
            if min_speakers is not None and hasattr(diarizer, "_min_speakers"):
                saved["diarizer._min_speakers"] = diarizer._min_speakers
                diarizer._min_speakers = min_speakers
            if max_speakers is not None and hasattr(diarizer, "_max_speakers"):
                saved["diarizer._max_speakers"] = diarizer._max_speakers
                diarizer._max_speakers = max_speakers

        return saved

    def _swap_diarization_out(self, saved: dict[str, Any]) -> None:
        if "pipeline.diarizer" in saved:
            self._pipeline.diarizer = saved["pipeline.diarizer"]
        diarizer = self._pipeline.diarizer
        if diarizer is not None:
            if "diarizer._min_speakers" in saved:
                diarizer._min_speakers = saved["diarizer._min_speakers"]
            if "diarizer._max_speakers" in saved:
                diarizer._max_speakers = saved["diarizer._max_speakers"]

    # ------------------------------------------------------------------ transformers helper

    def _patch_transformers_language(
        self,
        pipe: Any,
        language: str,
        saved: dict[str, Any],
    ) -> bool:
        """
        transformers Whisper 파이프라인의 default generate_kwargs 를 동적으로 변경.

        ASR pipeline 내부 구조 (transformers 4.30+):
          - pipe._forward_params  : dict, generate() 호출 시 사용
          - pipe.generate_kwargs  : pipe 객체에 따라 존재
        둘 다 best-effort 로 시도.
        """
        try:
            # 1) _forward_params 패치 (가장 안정적인 경로)
            params = getattr(pipe, "_forward_params", None)
            if isinstance(params, dict) and "language" in params:
                saved["_pipe_generate_kwargs"] = dict(params)
                params["language"] = language
                return True
            # 2) generate_kwargs 패치
            gk = getattr(pipe, "generate_kwargs", None)
            if isinstance(gk, dict) and "language" in gk:
                saved["_pipe_generate_kwargs"] = dict(gk)
                gk["language"] = language
                return True
        except Exception as e:
            log.debug("transformers 언어 패치 예외: %s", e)
        return False
