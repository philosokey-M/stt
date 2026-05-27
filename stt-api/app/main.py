"""
FastAPI 엔트리포인트.

엔드포인트:
  POST /transcribe          — 동기 전사. 결과 JSON 즉시 반환.
  POST /transcribe/async    — 비동기 전사. job_id 반환.
  GET  /jobs/{job_id}       — 비동기 job 상태/결과 조회.
  GET  /health              — 모델 로드 상태/통계.
  GET  /                    — API 정보.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import mutagen
import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import JSONResponse

from .inference import ModelManager, TranscribeResult, get_manager, init_manager
from .jobs import Job, JobStatus, JobStore, sweeper_loop
from .metrics import cer, normalize_hypothesis, normalize_label, wer
from .schemas import (
    HealthResponse,
    JobCreateResponse,
    JobStatusResponse,
    TranscribeResponse,
)
from .settings import SETTINGS

log = logging.getLogger("stt-api")
logging.basicConfig(
    level=SETTINGS.server.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

job_store = JobStore(ttl_seconds=SETTINGS.server.job_ttl_seconds)


# --------------------------------------------------------------------------- lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("startup — 모델 로드 시작")
    init_manager(SETTINGS)
    sweeper_task = asyncio.create_task(sweeper_loop(job_store))
    try:
        yield
    finally:
        sweeper_task.cancel()
        try:
            await sweeper_task
        except asyncio.CancelledError:
            pass
        log.info("shutdown 완료")


app = FastAPI(
    title="STT API",
    version="0.1.0",
    description=(
        "WhisperX 기반 STT API. /whisperx 파이프라인을 그대로 사용하며 "
        "VAD / STT / 화자분리 모델을 .env 와 config.yaml 로 교체할 수 있다."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- helpers


def _build_response(
    result: TranscribeResult,
    requested_language: str | None,
    reference: str | None,
) -> TranscribeResponse:
    full_text = " ".join(s["text"] for s in result.segments).strip()

    payload: dict = {
        "segments": result.segments,
        "text": full_text,
        # router 가 실제로 사용한 언어를 그대로 노출 (요청 미지정 시 default)
        "language": result.language or requested_language or _default_language(),
        "elapsed_s": result.elapsed_s,
        "audio_duration_s": result.audio_duration_s,
        "rtf": round(result.rtf, 4) if result.rtf is not None else None,
    }

    if reference and reference.strip():
        ref_norm = normalize_label(reference)
        hyp_norm = normalize_hypothesis(full_text)
        payload.update({
            "ref": ref_norm,
            "hyp": hyp_norm,
            "wer": round(wer(ref_norm, hyp_norm), 4),
            "cer": round(cer(ref_norm, hyp_norm), 4),
        })

    return TranscribeResponse(**payload)


async def _save_upload(file: UploadFile) -> Path:
    max_bytes = SETTINGS.server.max_upload_mb * 1024 * 1024
    suffix = Path(file.filename or "audio").suffix or ".bin"
    fd, tmp_path = tempfile.mkstemp(prefix="stt_", suffix=suffix)
    os.close(fd)

    total = 0
    try:
        with open(tmp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"파일이 너무 큽니다. 최대 {SETTINGS.server.max_upload_mb}MB",
                    )
                out.write(chunk)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return Path(tmp_path)


def _cleanup(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception as e:
        log.warning("임시 파일 삭제 실패 %s: %s", path, e)


async def _check_audio_file(file: UploadFile) -> None:
    file_bytes = await file.read()
    try:
        audio_file = io.BytesIO(file_bytes)
        audio = mutagen.File(audio_file)
        if audio is None or audio.info is None:
            raise HTTPException(status_code=400, detail="지원하지 않거나 손상된 오디오 파일입니다.")
        duration = audio.info.length
        if duration < SETTINGS.server.min_audio_duration_s:
            raise HTTPException(status_code=400, detail=f"오디오 파일 길이가 너무 짧습니다. 최소 {SETTINGS.server.min_audio_duration_s}초")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"오류 발생: {str(e)}")

# --------------------------------------------------------------------------- endpoints


@app.get("/", tags=["meta"])
def root():
    return {
        "name": "STT API",
        "version": app.version,
        "endpoints": [
            "POST /transcribe",
            "POST /transcribe/async",
            "GET  /jobs/{job_id}",
            "GET  /health",
            "GET  /docs",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    mgr = get_manager()
    return HealthResponse(
        status="ok" if mgr.is_ready() else "loading",
        model=mgr.info,
    )


@app.post("/transcribe", response_model=TranscribeResponse, tags=["transcribe"])
async def transcribe(
    file: UploadFile = File(..., description="오디오 파일 (wav/mp3/m4a/flac/ogg 등)"),
    language: str | None = Form(
        None,
        description=(
            "언어 코드 (ko, en, ja, zh, fr, de, ...). 미지정 시 서버 기본값 사용. "
            "Whisper 가 지원하는 언어면 동일 모델에서 처리. 해당 언어의 align 모델은 "
            "첫 요청에서 자동 로드 + 캐시 (약 300MB)."
        ),
    ),
    reference: str | None = Form(None, description="정답 텍스트. 제공되면 wer/cer 계산."),
):
    """동기 전사. 짧은 오디오에 권장."""
    # TODO : 화자 최대, 최소 수 파라미터로 제어 하도록 수정해야함 (없으면 자동 추정)
    await _check_audio_file(file)

    
    mgr = get_manager()
    if not mgr.is_ready():
        raise HTTPException(status_code=503, detail="모델 로딩 중")

    tmp_path = await _save_upload(file)
    try:
        # language 가 None 이면 ModelManager → LanguageRouter 가 default 로 채움
        result = await mgr.transcribe(str(tmp_path), language=language)
    except Exception as e:
        log.exception("transcribe 실패")
        raise HTTPException(status_code=500, detail=f"전사 실패: {e}")
    finally:
        _cleanup(tmp_path)

    return _build_response(result, language, reference)


@app.post("/transcribe/async", response_model=JobCreateResponse, tags=["transcribe"])
async def transcribe_async(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    language: str | None = Form(None),
    reference: str | None = Form(None),
):
    """비동기 전사. 긴 파일에 권장. job_id 를 반환하며 결과는 /jobs/{id} 로 폴링."""
    # TODO : 화자 최대, 최소 수 파라미터로 제어 하도록 수정해야함 (없으면 자동 추정)
    await _check_audio_file(file)

    mgr = get_manager()
    if not mgr.is_ready():
        raise HTTPException(status_code=503, detail="모델 로딩 중")

    tmp_path = await _save_upload(file)

    job = await job_store.create()
    job.audio_path = str(tmp_path)
    await job_store.update(job)

    background.add_task(_run_job, job.id, str(tmp_path), language, reference)

    return JobCreateResponse(job_id=job.id, status=job.status.value)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["transcribe"])
async def get_job(job_id: str):
    job = await job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job 을 찾을 수 없습니다")
    d = job.to_dict()
    return JSONResponse(d)


# --------------------------------------------------------------------------- job worker


async def _run_job(job_id: str, audio_path: str, language: str | None, reference: str | None) -> None:
    job = await job_store.get(job_id)
    if job is None:
        return
    import time as _t

    job.status = JobStatus.PROCESSING
    job.started_at = _t.time()
    await job_store.update(job)

    try:
        mgr = get_manager()
        result = await mgr.transcribe(audio_path, language=language)
        response = _build_response(result, language, reference)
        job.result = response.model_dump()
        job.status = JobStatus.DONE
    except Exception as e:
        log.exception("job %s 실패", job_id)
        job.error = str(e)
        job.status = JobStatus.ERROR
    finally:
        job.finished_at = _t.time()
        _cleanup(audio_path)
        await job_store.update(job)


def _default_language() -> str:
    return (
        SETTINGS.pipeline_config.get("pipeline", {})
        .get("stt", {})
        .get("language", "ko")
    )
