"""
FastAPI 엔트리포인트.

엔드포인트:
  POST /transcribe          — 동기 전사. 결과 JSON 즉시 반환.
  POST /transcribe/async    — 비동기 전사. job_id 반환.
  GET  /progress/{task_id}  — 진행률 조회. (동기/비동기 모두 사용 가능)
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
import threading
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

from .audio import audio_duration_seconds
from .inference import ModelManager, TranscribeResult, get_manager, init_manager
from .jobs import Job, JobStatus, JobStore, sweeper_loop
from .metrics import cer, normalize_hypothesis, normalize_label, wer
from .progress import get_progress, mark_done, simulate_progress
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

    # 레거시 호환 chunks 포맷: segments 와 같은 데이터, 키 이름만 다름.
    chunks = [
        {"text": s["text"], "start_time": s["start"], "end_time": s["end"]}
        for s in result.segments
    ]

    payload: dict = {
        # 레거시 호환 필드 (외부 클라이언트가 의존)
        "chunks": chunks,
        "duration": result.audio_duration_s,
        # 확장 필드
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


def _validate_speaker_range(min_speakers: int | None, max_speakers: int | None) -> None:
    """min_speakers > max_speakers 등 명백히 잘못된 조합은 400 으로 거른다."""
    if (
        min_speakers is not None
        and max_speakers is not None
        and min_speakers > max_speakers
    ):
        raise HTTPException(
            status_code=400,
            detail=f"min_speakers({min_speakers}) > max_speakers({max_speakers}) 입니다.",
        )


async def _start_progress_simulator(
    task_id: str | None,
    tmp_path: Path,
) -> tuple[threading.Thread | None, threading.Event]:
    """
    task_id 가 주어진 경우 진행률 시뮬레이터 스레드를 시작.
    반환: (스레드, stop_event). 호출자는 finally 에서 _stop_progress_simulator 로 정리.

    실제 진행률은 알 수 없어서 audio_duration * EXPECTED_RTF 기반으로 시간 추정.
    """
    stop_event = threading.Event()
    if not task_id:
        return None, stop_event

    try:
        audio_dur = await asyncio.to_thread(audio_duration_seconds, str(tmp_path), None)
    except Exception:
        audio_dur = 0.0

    thread = threading.Thread(
        target=simulate_progress,
        args=(task_id, audio_dur, SETTINGS.server.expected_rtf, stop_event),
        daemon=True,
    )
    thread.start()
    return thread, stop_event


def _stop_progress_simulator(
    task_id: str | None,
    thread: threading.Thread | None,
    stop_event: threading.Event,
) -> None:
    """시뮬레이터 종료 + 진행률 1.0 표시 (TTL 후 자동 정리)."""
    stop_event.set()
    if thread is not None:
        thread.join(timeout=5)
    if task_id:
        mark_done(task_id)


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
            "GET  /progress/{task_id}",
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
    audio: UploadFile = File(..., description="오디오 파일 (wav/mp3/m4a/flac/ogg 등). 폼 필드명은 'audio'."),
    language: str | None = Form(
        None,
        description=(
            "언어 코드 (ko, en, ja, zh, fr, de, ...). 미지정 시 서버 기본값 사용. "
            "Whisper 가 지원하는 언어면 동일 모델에서 처리. 해당 언어의 align 모델은 "
            "첫 요청에서 자동 로드 + 캐시 (약 300MB)."
        ),
    ),
    task_id: str | None = Form(
        None,
        description=(
            "외부 큐(Celery 등)가 발행한 task ID. 주면 GET /progress/{task_id} 로 "
            "진행률 폴링 가능. 진행률은 오디오 길이 × EXPECTED_RTF 로 추정한 값."
        ),
    ),
    reference: str | None = Form(None, description="정답 텍스트. 제공되면 wer/cer 계산."),
    diarize: bool | None = Form(
        None,
        description=(
            "화자분리 사용 여부. 미지정 시 서버 기본값(.env DIARIZATION_MODE) 그대로. "
            "true 로 강제하려면 서버가 화자분리 모델을 로드한 상태여야 함."
        ),
    ),
    min_speakers: int | None = Form(
        None, ge=1, description="최소 화자 수. 미지정 시 모델 자동 추정."
    ),
    max_speakers: int | None = Form(
        None, ge=1, description="최대 화자 수. 미지정 시 모델 자동 추정."
    ),
):
    """
    동기 전사. 짧은 오디오에 권장.

    응답은 레거시 Whisper STT 서버와 호환되도록 `chunks` / `duration` 을 포함하고,
    동시에 우리 확장 필드(`segments`, `language`, `elapsed_s`, `total_elapsed_s`,
    `rtf`, `wer`/`cer`) 도 함께 포함한다. 레거시 클라이언트는 `chunks` + `duration`
    만 읽으면 되고, 새 클라이언트는 `segments` 사용 권장.
    """
    t_start = time.perf_counter()
    await _check_audio_file(audio)
    _validate_speaker_range(min_speakers, max_speakers)

    mgr = get_manager()
    if not mgr.is_ready():
        raise HTTPException(status_code=503, detail="모델 로딩 중")
    if diarize is True and not mgr.has_diarization():
        raise HTTPException(
            status_code=400,
            detail=(
                "서버에서 화자분리가 비활성화되어 있어 diarize=true 요청을 처리할 수 없습니다. "
                ".env 의 DIARIZATION_MODE 와 HF_TOKEN 을 설정한 뒤 서버를 재시작하세요."
            ),
        )

    tmp_path = await _save_upload(audio)

    sim_thread, sim_stop = await _start_progress_simulator(task_id, tmp_path)
    try:
        result = await mgr.transcribe(
            str(tmp_path),
            language=language,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except Exception as e:
        log.exception("transcribe 실패")
        raise HTTPException(status_code=500, detail=f"전사 실패: {e}")
    finally:
        _stop_progress_simulator(task_id, sim_thread, sim_stop)
        _cleanup(tmp_path)

    total_elapsed = time.perf_counter() - t_start
    return _build_response(result, language, reference, total_elapsed_s=total_elapsed)


@app.post("/transcribe/async", response_model=JobCreateResponse, tags=["transcribe"])
async def transcribe_async(
    background: BackgroundTasks,
    audio: UploadFile = File(..., description="오디오 파일. 폼 필드명은 'audio'."),
    language: str | None = Form(None),
    task_id: str | None = Form(None, description="외부 큐의 task ID. /progress/{task_id} 폴링용."),
    reference: str | None = Form(None),
    diarize: bool | None = Form(None),
    min_speakers: int | None = Form(None, ge=1),
    max_speakers: int | None = Form(None, ge=1),
):
    """비동기 전사. 긴 파일에 권장. job_id 를 반환하며 결과는 /jobs/{id} 로 폴링."""
    await _check_audio_file(audio)
    _validate_speaker_range(min_speakers, max_speakers)

    mgr = get_manager()
    if not mgr.is_ready():
        raise HTTPException(status_code=503, detail="모델 로딩 중")
    if diarize is True and not mgr.has_diarization():
        raise HTTPException(
            status_code=400,
            detail=(
                "서버에서 화자분리가 비활성화되어 있어 diarize=true 요청을 처리할 수 없습니다. "
                ".env 의 DIARIZATION_MODE 와 HF_TOKEN 을 설정한 뒤 서버를 재시작하세요."
            ),
        )

    tmp_path = await _save_upload(audio)

    job = await job_store.create()
    job.audio_path = str(tmp_path)
    await job_store.update(job)

    background.add_task(
        _run_job,
        job.id, str(tmp_path), language, reference,
        diarize, min_speakers, max_speakers, task_id,
    )

    return JobCreateResponse(job_id=job.id, status=job.status.value)


@app.get("/progress/{task_id}", tags=["transcribe"])
def get_progress_endpoint(task_id: str):
    """
    레거시 호환 진행률 조회. /transcribe 또는 /transcribe/async 호출 시 함께 보낸
    task_id 의 progress 를 반환. 미등록 task_id 거나 시작 전이면 0.0.

    응답: `{"task_id": "...", "progress": 0.0~1.0}`
    """
    return {"task_id": task_id, "progress": get_progress(task_id)}


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["transcribe"])
async def get_job(job_id: str):
    job = await job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job 을 찾을 수 없습니다")
    d = job.to_dict()
    return JSONResponse(d)


# --------------------------------------------------------------------------- job worker


async def _run_job(
    job_id: str,
    audio_path: str,
    language: str | None,
    reference: str | None,
    diarize: bool | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    task_id: str | None = None,
) -> None:
    job = await job_store.get(job_id)
    if job is None:
        return

    job.status = JobStatus.PROCESSING
    job.started_at = time.time()
    await job_store.update(job)

    sim_thread, sim_stop = await _start_progress_simulator(task_id, Path(audio_path))

    t_start = time.perf_counter()
    try:
        mgr = get_manager()
        result = await mgr.transcribe(
            audio_path,
            language=language,
            diarize=diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        total_elapsed = time.perf_counter() - t_start
        response = _build_response(result, language, reference, total_elapsed_s=total_elapsed)
        job.result = response.model_dump()
        job.status = JobStatus.DONE
    except Exception as e:
        log.exception("job %s 실패", job_id)
        job.error = str(e)
        job.status = JobStatus.ERROR
    finally:
        _stop_progress_simulator(task_id, sim_thread, sim_stop)
        job.finished_at = time.time()
        _cleanup(audio_path)
        await job_store.update(job)


def _default_language() -> str:
    return (
        SETTINGS.pipeline_config.get("pipeline", {})
        .get("stt", {})
        .get("language", "ko")
    )
