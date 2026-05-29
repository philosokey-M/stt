"""
.env 단일 소스 설정 로더.

설정 우선순위: 환경변수 (.env) > 코드 기본값.
모든 파이프라인 옵션은 .env 안에서 직접 표현된다 — config.yaml 은 사용하지 않는다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

API_DIR = Path(__file__).resolve().parent.parent
WHISPERX_DIR = API_DIR.parent / "whisperx"


# .env 로딩 (한 번만)
load_dotenv(API_DIR / ".env", override=False)


def _env(name: str, default: str | None = None) -> str | None:
    """
    환경변수 한 개를 읽어 양쪽 공백을 strip. 빈 문자열은 default 로 폴백.

    python-dotenv 는 `KEY=value   # 코멘트` 의 `# 코멘트` 만 잘라내고 값 뒤 공백은
    남겨두기 때문에 여기서 한 번 더 정리해야 한다.
    """
    v = os.environ.get(name)
    if v is not None:
        v = v.strip()
    return v if v not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = _env(name)
    return int(v) if v is not None else default


def _env_int_optional(name: str) -> int | None:
    v = _env(name)
    return int(v) if v is not None else None


def _env_float(name: str, default: float) -> float:
    v = _env(name)
    return float(v) if v is not None else default


class ServerSettings(BaseModel):
    host: str
    port: int
    log_level: str
    max_upload_mb: int
    min_audio_duration_s: int
    job_ttl_seconds: int
    max_concurrent_inferences: int
    io_worker_threads: int


class APISettings(BaseModel):
    server: ServerSettings
    pipeline_config: dict = Field(default_factory=dict)
    hf_token: str | None = None
    diarization_mode: Literal["auto", "on", "off"] = "auto"


def _build_pipeline_config() -> dict[str, Any]:
    """
    /whisperx/pipeline.py 의 build_from_config() 가 받는 dict 를 .env 만으로 구성.
    /whisperx 의 스키마를 그대로 따라 만든다.

    device 는 STT 뿐 아니라 vad / diarization 섹션에도 넣어 전 단계 일관성 유지
    (pyannote 모델들이 자기 cfg 의 device 를 읽어 GPU 로 이동).
    """
    device = _env("DEVICE", "cuda")
    return {
        "pipeline": {
            "vad": {
                "model": _env("VAD_MODEL", "disabled"),
                "model_id": _env("VAD_MODEL_ID", "pyannote/voice-activity-detection"),
                "device": device,
                # silero 전용 — 기본값은 stages/vad.py 와 동일
                "onset": _env_float("VAD_ONSET", 0.500),
                "offset": _env_float("VAD_OFFSET", 0.363),
                "min_speech_duration_ms": _env_int("VAD_MIN_SPEECH_MS", 250),
                "min_silence_duration_ms": _env_int("VAD_MIN_SILENCE_MS", 2000),
            },
            "stt": {
                "backend": _env("STT_BACKEND", "whisperx"),
                "model_id": _env("STT_MODEL_ID", "large-v3"),
                "language": _env("DEFAULT_LANGUAGE", "ko"),
                "device": device,
                "compute_type": _env("COMPUTE_TYPE", "float16"),
                "batch_size": _env_int("BATCH_SIZE", 16),
                "align": _env_bool("ALIGN", True),
                "beam_size": _env_int("BEAM_SIZE", 5),
                "temperature": _env_float("TEMPERATURE", 0.0),
            },
            "diarization": {
                # 최종 enabled 값은 _resolve_diarization() 에서 모드에 따라 결정
                "enabled": False,
                "model": _env("DIARIZATION_MODEL", "pyannote"),
                "model_id": _env(
                    "DIARIZATION_MODEL_ID",
                    "pyannote/speaker-diarization-3.1",
                ),
                "device": device,
                "min_speakers": _env_int_optional("DIAR_MIN_SPEAKERS"),
                "max_speakers": _env_int_optional("DIAR_MAX_SPEAKERS"),
            },
        },
        "hf_token": _env("HF_TOKEN"),
    }


def _resolve_diarization(cfg: dict, mode: str) -> None:
    """DIARIZATION_MODE 와 HF_TOKEN 상태에 따라 enabled 최종 결정."""
    diar = cfg["pipeline"]["diarization"]
    has_token = bool(cfg.get("hf_token"))

    if mode == "on":
        if not has_token:
            raise RuntimeError(
                "DIARIZATION_MODE=on 인데 HF_TOKEN 이 설정되지 않았습니다. "
                ".env 의 HF_TOKEN 을 채우거나 DIARIZATION_MODE=auto/off 로 바꾸세요."
            )
        diar["enabled"] = True
    elif mode == "off":
        diar["enabled"] = False
    else:  # auto
        diar["enabled"] = has_token


def _apply_hf_offline() -> bool:
    """
    HF_OFFLINE=true (기본) 일 때 HuggingFace 캐시 전용 모드로 강제.

    중요: 이 함수는 whisperx / pyannote / transformers 가 임포트되기 *전에*
    호출돼야 효과가 있다. 본 모듈(settings.py)은 main.py 에서 가장 먼저 SETTINGS 를
    참조하기 때문에 자연스럽게 보장된다 (HF 라이브러리는 model load 시점에 lazy
    import).

    효과:
      HF_HUB_OFFLINE=1       — huggingface_hub 가 네트워크 호출 안 함, 캐시만 사용.
      TRANSFORMERS_OFFLINE=1 — transformers 도 동일.
    """
    offline = _env_bool("HF_OFFLINE", True)
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        # 명시적으로 false 면 혹시 외부에서 세팅돼 있어도 제거 (덮어쓰기 방지 X)
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
    return offline


def load_settings() -> APISettings:
    # HF 오프라인 모드를 *가장 먼저* 적용 — HF 라이브러리 임포트 전에 환경변수가 박혀야 함.
    _apply_hf_offline()

    cfg = _build_pipeline_config()

    diarization_mode = (_env("DIARIZATION_MODE", "auto") or "auto").lower()
    if diarization_mode not in ("auto", "on", "off"):
        raise RuntimeError(f"DIARIZATION_MODE 는 auto/on/off 중 하나여야 합니다: {diarization_mode!r}")
    _resolve_diarization(cfg, diarization_mode)

    server = ServerSettings(
        host=_env("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8000),
        log_level=(_env("LOG_LEVEL", "info") or "info").lower(),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 200),
        min_audio_duration_s=_env_int("MIN_AUDIO_DURATION_S", 1),
        job_ttl_seconds=_env_int("JOB_TTL_SECONDS", 3600),
        max_concurrent_inferences=max(1, _env_int("MAX_CONCURRENT_INFERENCES", 1)),
        io_worker_threads=max(1, _env_int("IO_WORKER_THREADS", 4)),
    )

    return APISettings(
        server=server,
        pipeline_config=cfg,
        hf_token=cfg.get("hf_token"),
        diarization_mode=diarization_mode,
    )


# 모듈 임포트 시 1회 로딩 — startup 에서 재사용
SETTINGS = load_settings()
