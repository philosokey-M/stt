"""
비동기 job 큐. 긴 오디오 파일을 fire-and-forget 으로 처리.

- 클라이언트는 POST /transcribe?async=true 로 job_id 만 받음
- GET /jobs/{job_id} 로 상태/결과 폴링
- 만료된 job 은 백그라운드 sweeper 가 주기적으로 정리
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    # 처리 중에 임시 파일 경로를 보관해 두면 finalize 단계에서 삭제 가능
    audio_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status.value,
            "created_at": round(self.created_at, 3),
            "started_at": round(self.started_at, 3) if self.started_at else None,
            "finished_at": round(self.finished_at, 3) if self.finished_at else None,
            "result": self.result,
            "error": self.error,
        }


class JobStore:
    def __init__(self, ttl_seconds: int):
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds

    async def create(self) -> Job:
        async with self._lock:
            jid = uuid.uuid4().hex
            job = Job(id=jid)
            self._jobs[jid] = job
            return job

    async def get(self, jid: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(jid)

    async def update(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.id] = job

    async def sweep(self) -> int:
        """TTL 지난 완료/오류 job 제거. 반환: 제거된 개수."""
        now = time.time()
        removed = 0
        async with self._lock:
            for jid in list(self._jobs.keys()):
                j = self._jobs[jid]
                if j.status in (JobStatus.DONE, JobStatus.ERROR):
                    ref = j.finished_at or j.created_at
                    if now - ref > self._ttl:
                        del self._jobs[jid]
                        removed += 1
        return removed


async def sweeper_loop(store: JobStore, interval_seconds: int = 60) -> None:
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            removed = await store.sweep()
            if removed:
                log.info("job sweeper: %d 개 만료 job 제거", removed)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("job sweeper 오류: %s", e)
