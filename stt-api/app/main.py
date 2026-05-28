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
import shutil
import tempfile
import time
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
    total_elapsed_s: float | None = None,
) -> TranscribeResponse:
    full_text = " ".join(s["text"] for s in result.segments).strip()

    payload: dict = {
        "segments": result.segments,
        "text": full_text,
        # router 가 실제로 사용한 언어를 그대로 노출 (요청 미지정 시 default)
        "language": result.language or requested_language or _default_language(),
        "elapsed_s": result.elapsed_s,
        "total_elapsed_s": round(total_elapsed_s, 3) if total_elapsed_s is not None else None,
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
    """
    업로드된 파일을 /tmp 에 저장. UploadFile.file (SpooledTemporaryFile) 의 위치를
    명시적으로 0 으로 되돌리고, shutil.copyfileobj 로 한 번에 복사 — 청크 루프보다
    덜 미묘하고 더 안정적이다.

    빈 파일(0 bytes)이면 400 으로 즉시 반환한다.
    """
    max_bytes = SETTINGS.server.max_upload_mb * 1024 * 1024
    suffix = Path(file.filename or "audio").suffix or ".bin"
    fd, tmp_path = tempfile.mkstemp(prefix="stt_", suffix=suffix)
    os.close(fd)

    def _copy_sync() -> int:
        # 어떤 이유로든 파일 포인터가 끝에 가있는 경우를 방어
        try:
            file.file.seek(0)
        except Exception:
            pass
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(file.file, out, length=1024 * 1024)
        return os.path.getsize(tmp_path)

    try:
        size = await asyncio.to_thread(_copy_sync)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    log.info(
        "업로드 저장: name=%r content_type=%s size=%d → %s",
        file.filename, file.content_type, size, tmp_path,
    )

    if size == 0:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                f"업로드된 파일이 비어있습니다 (filename={file.filename!r}). "
                "multipart 폼 필드 이름이 'file' 인지, 파일 경로가 올바른지 확인하세요."
            ),
        )
    if size > max_bytes:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=413,
            detail=f"파일이 너무 큽니다 ({size} bytes). 최대 {SETTINGS.server.max_upload_mb}MB",
        )

    return Path(tmp_path)


def _cleanup(path: str | Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except Exception as e:
        log.warning("임시 파일 삭제 실패 %s: %s", path, e)


async def _check_audio_file(file: UploadFile) -> None:
    """
    업로드 직후 mutagen 으로 헤더만 검사. 길이가 너무 짧으면 400.

    SpooledTemporaryFile 객체를 mutagen 에 직접 넘겨 헤더 부근만 seek/read 한다.
    전체 본문을 RAM 에 올리지 않으므로 200MB 업로드여도 실제 RAM 사용은 수 KB.
    검사 전후로 파일 포인터를 0 으로 맞춰 이후 _save_upload() 가 정상 동작하게 한다.

    실제 I/O 는 sync 라 asyncio.to_thread 로 워커 스레드 위임 → 이벤트 루프 미블로킹.
    """
    import mutagen

    def _validate_sync(spooled):
        spooled.seek(0)
        try:
            return mutagen.File(spooled)
        finally:
            spooled.seek(0)

    try:
        audio = await asyncio.to_thread(_validate_sync, file.file)
    except Exception as e:
        # 안전을 위해 포인터도 복구 시도
        try:
            await file.seek(0)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail=f"오디오 헤더 파싱 실패: {e}")

    if audio is None or audio.info is None:
        raise HTTPException(status_code=400, detail="지원하지 않거나 손상된 오디오 파일입니다.")

    duration = audio.info.length
    if duration < SETTINGS.server.min_audio_duration_s:
        raise HTTPException(
            status_code=400,
            detail=f"오디오 파일 길이가 너무 짧습니다. 최소 {SETTINGS.server.min_audio_duration_s}초",
        )

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
    """
    동기 전사. 짧은 오디오에 권장.
    Returns:
        dict: 아래와 같은 구조의 STT 결과 딕셔너리를 반환합니다.
            - segments (list): 오디오를 문장/의미 단위로 분할한 상세 분석 결과 목록
                - start (float): 해당 구간의 시작 시간 (초 단위)
                - end (float): 해당 구간의 종료 시간 (초 단위)
                - text (str): 해당 구간에서 인식된 텍스트 내용
                - speaker (str): 화자 식별 ID (예: SPEAKER_00)
            - text (str): 오디오 전체를 텍스트로 변환한 통합 결과
            - language (str): 오디오에서 감지되거나 지정된 언어 코드 (예: 'ko')
            - elapsed_s (float): 순수 STT 모델 추론(Inference)에 걸린 시간 (초 단위)
            - total_elapsed_s (float): 전/후처리를 포함하여 API 요청부터 완료까지의 총 소요 시간 (초 단위)
            - audio_duration_s (float): 입력된 전체 오디오 파일의 총 길이 (초 단위)
            - rtf (float): Real-Time Factor (실시간 처리 지수, elapsed_s / audio_duration_s)
                        (1보다 작을수록 실시간보다 빠르게 처리됨을 의미)
            - ref (str, optional): 정답 텍스트 (성능 검증/벤치마크 모드가 아닐 경우 null)
            - hyp (str, optional): 모델 예측 텍스트 (성능 검증/벤치마크 모드가 아닐 경우 null)
            - wer (float, optional): Word Error Rate (단어 오류율, 성능 검증용 지표)
            - cer (float, optional): Character Error Rate (문자 오류율, 한국어 STT 주요 평가지표)
    """
    # TODO : 화자 최대, 최소 수 파라미터로 제어 하도록 수정해야함 (없으면 자동 추정)
    t_start = time.perf_counter()
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

    total_elapsed = time.perf_counter() - t_start
    return _build_response(result, language, reference, total_elapsed_s=total_elapsed)


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

    job.status = JobStatus.PROCESSING
    job.started_at = time.time()
    await job_store.update(job)

    t_start = time.perf_counter()
    try:
        mgr = get_manager()
        result = await mgr.transcribe(audio_path, language=language)
        total_elapsed = time.perf_counter() - t_start
        response = _build_response(result, language, reference, total_elapsed_s=total_elapsed)
        job.result = response.model_dump()
        job.status = JobStatus.DONE
    except Exception as e:
        log.exception("job %s 실패", job_id)
        job.error = str(e)
        job.status = JobStatus.ERROR
    finally:
        job.finished_at = time.time()
        _cleanup(audio_path)
        await job_store.update(job)


def _default_language() -> str:
    return (
        SETTINGS.pipeline_config.get("pipeline", {})
        .get("stt", {})
        .get("language", "ko")
    )
