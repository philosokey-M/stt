"""
레거시 Whisper STT 서버 호환용 진행률 추적.

외부 큐(예: Celery) 가 발행한 `task_id` 를 받아, 모델의 불투명한 처리 시간을
"오디오 길이 × 예상 RTF" 로 시뮬레이션해서 0.0 ~ 1.0 사이 progress 를 게시한다.

**실제 모델 진행률이 아니라 추정치다.** HuggingFace / faster-whisper / whisperx 모두
generate() 가 콜백을 노출하지 않아 정확한 진행 측정이 불가능하므로, UX 용 시간 기반
추정으로 채워주는 패턴 (레거시 서버가 쓰던 방식과 동일).
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

# 기본 TTL: 완료 후 이만큼 지나면 자동 정리
DEFAULT_TTL_SECONDS = 30.0

_lock = threading.Lock()
# {task_id: (progress 0~1, finished_at | None)}
_progress: dict[str, tuple[float, float | None]] = {}


def get_progress(task_id: str, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> float:
    """task_id 의 현재 진행률. 미등록이거나 완료 후 TTL 초과면 0.0 (게으른 정리)."""
    with _lock:
        entry = _progress.get(task_id)
        if entry is None:
            return 0.0
        progress, finished_at = entry
        if finished_at is not None and (time.time() - finished_at) > ttl_seconds:
            del _progress[task_id]
            return 0.0
        return progress


def set_progress(task_id: str, value: float) -> None:
    """진행 중인 상태로 progress 업데이트. 완료 표시는 mark_done() 사용."""
    v = max(0.0, min(1.0, value))
    with _lock:
        _progress[task_id] = (v, None)


def mark_done(task_id: str) -> None:
    """완료 표시. 1.0 으로 고정되고 일정 TTL 후 자동 삭제."""
    with _lock:
        _progress[task_id] = (1.0, time.time())


def clear(task_id: str) -> None:
    with _lock:
        _progress.pop(task_id, None)


def simulate_progress(
    task_id: str,
    audio_duration_s: float,
    expected_rtf: float,
    stop_event: threading.Event,
    interval: float = 0.5,
) -> None:
    """
    경과시간/예상시간 비율로 progress 를 갱신하는 워커 스레드 본체.

    expected_rtf : 우리 파이프라인 평균 RTF 추정치 (예: whisperx large-v3 GPU = 0.05~0.15).
                   너무 낮게 잡으면 progress 가 0.99 에서 오래 정체. 약간 보수적이 안전.
    """
    estimated_total = max(0.1, audio_duration_s * expected_rtf)
    start = time.time()
    set_progress(task_id, 0.0)
    while not stop_event.is_set():
        elapsed = time.time() - start
        # 0.99 cap — 완료 표시는 호출자가 mark_done() 으로 한다.
        ratio = min(0.99, elapsed / estimated_total)
        set_progress(task_id, ratio)
        if stop_event.wait(interval):
            break
