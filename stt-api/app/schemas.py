"""API 응답/요청 Pydantic 스키마."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Segment(BaseModel):
    start: float = Field(..., description="발화 시작 시각 (초)")
    end: float = Field(..., description="발화 종료 시각 (초)")
    text: str = Field(..., description="전사된 텍스트")
    speaker: str | None = Field(None, description="화자 라벨 (예: SPEAKER_00). 비활성화 시 null.")


class TranscribeResponse(BaseModel):
    """동기 전사 응답. /results/benchmark_*.json 의 record 와 동일 포맷."""

    segments: list[Segment]
    text: str = Field(..., description="모든 세그먼트의 text 를 이어붙인 전체 발화")
    language: str = Field(..., description="사용된 언어 코드")
    elapsed_s: float = Field(..., description="추론 소요 시간 (초)")
    audio_duration_s: float = Field(..., description="입력 오디오 길이 (초)")
    rtf: float | None = Field(None, description="elapsed_s / audio_duration_s")
    # reference 가 함께 들어온 경우만 채워짐
    ref: str | None = Field(None, description="정규화된 reference (제공된 경우)")
    hyp: str | None = Field(None, description="정규화된 hypothesis (reference 제공 시)")
    wer: float | None = Field(None, description="Word Error Rate")
    cer: float | None = Field(None, description="Character Error Rate")


class JobCreateResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    id: str
    status: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    result: TranscribeResponse | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    model: dict[str, Any]
