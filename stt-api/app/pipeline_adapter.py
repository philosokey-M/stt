"""
/whisperx 의 stages/pipeline 을 sys.path 로 끌어와 그대로 재사용.

/whisperx 폴더는 수정하지 않는다. 새 모델을 추가하려면 /whisperx/stages 에
클래스만 추가하면 본 API 도 즉시 그 모델을 쓸 수 있다.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from .settings import WHISPERX_DIR

# /whisperx 를 모듈 검색 경로에 등록 (idempotent)
_wx = str(WHISPERX_DIR)
if _wx not in sys.path:
    sys.path.insert(0, _wx)

# noqa — sys.path 조작 이후 임포트해야 함
from pipeline import Pipeline, build_from_config  # type: ignore  # noqa: E402

if TYPE_CHECKING:
    from pipeline import Pipeline as PipelineType  # noqa: F401


__all__ = ["Pipeline", "build_from_config"]
