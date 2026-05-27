"""
WER / CER 계산. /whisperx/benchmark.py 의 로직을 그대로 가져옴 — 동일한 점수 보장.
"""

from __future__ import annotations

import re


def normalize_label(raw: str) -> str:
    """KtelSpeech 라벨 정규화. 일반 텍스트에도 안전하게 동작."""
    text = raw.strip()
    text = re.sub(r"\(([^)]+)\)/\([^)]+\)", r"\1", text)   # (A)/(B) → A
    text = re.sub(r"\b[a-zA-Z가-힣]{1,2}/", "", text)       # n/, 하/ 등 태그 제거
    text = re.sub(r"[^\w\s]", "", text)                     # 구두점 제거
    return " ".join(text.split())


def normalize_hypothesis(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r = [c for c in ref if c != " "]
    h = [c for c in hyp if c != " "]
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)
