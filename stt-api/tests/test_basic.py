"""
모델 로딩 없이 빠르게 돌릴 수 있는 단위 테스트.
실제 STT 추론 테스트는 별도의 통합 테스트로 분리해야 한다 (GPU 필요).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_metrics_basic():
    from app.metrics import cer, normalize_hypothesis, normalize_label, wer

    ref = normalize_label("n/ 네 안녕하세요 (이거요)/(저거요).")
    assert "네 안녕하세요 이거요" in ref

    h = normalize_hypothesis("네 안녕하세요 이거요.")
    assert wer(ref, h) == 0.0
    assert cer(ref, h) == 0.0


def test_settings_loads():
    from app.settings import SETTINGS

    assert SETTINGS.server.max_concurrent_inferences >= 1
    assert "pipeline" in SETTINGS.pipeline_config
    assert SETTINGS.diarization_mode in ("auto", "on", "off")
