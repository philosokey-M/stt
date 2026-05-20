#!/usr/bin/env python3
"""
WhisperX benchmark against KtelSpeech dataset.

Usage:
  python benchmark.py --data-dir test-data/sample_01 [--limit 50] [--output-dir results/]

Directory layout expected:
  <data-dir>/raw_data/<dataset>/<domain>/<speaker>/<utt>.wav
  <data-dir>/label_data/<dataset_label>/<domain>/<speaker>/<utt>.txt
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Label normalization for KtelSpeech
# ---------------------------------------------------------------------------

def normalize_label(raw: str) -> str:
    """
    KtelSpeech 라벨 정규화:
    - 앞/중간의 태그 제거: 'n/', 'u/', 'b/', '하/', 'o/' 등
    - 대안 표기 (A)/(B) → A 선택
    - 구두점 제거, 공백 정규화
    """
    text = raw.strip()

    # (A)/(B) 형태 → A 선택
    text = re.sub(r'\(([^)]+)\)/\([^)]+\)', r'\1', text)

    # 태그 제거: 문장 앞 'x/' 혹은 중간 'x/' (한두 글자 + 슬래시)
    text = re.sub(r'\b[a-zA-Z가-힣]{1,2}/', '', text)

    # 구두점 제거
    text = re.sub(r'[^\w\s]', '', text)

    # 공백 정규화
    text = ' '.join(text.split())
    return text


def normalize_hypothesis(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    return ' '.join(text.split())


# ---------------------------------------------------------------------------
# WER / CER
# ---------------------------------------------------------------------------

def edit_distance(a: list, b: list) -> int:
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
    return edit_distance(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    # 공백 제거 후 문자 단위 비교 (한국어는 어절 분리 없이 CER이 더 의미있음)
    r = [c for c in ref if c != ' ']
    h = [c for c in hyp if c != ' ']
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


# ---------------------------------------------------------------------------
# Data pairing
# ---------------------------------------------------------------------------

def find_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    """raw_data 아래 wav와 label_data 아래 txt를 1:1 매칭."""
    raw_root = data_dir / "raw_data"
    label_root = data_dir / "label_data"

    wav_dataset = next(raw_root.iterdir())    # KtelSpeech_train_D60_wav_0
    label_dataset = next(label_root.iterdir()) # KtelSpeech_train_D60_label_0

    pairs = []
    for wav_path in sorted(wav_dataset.rglob("*.wav")):
        if wav_path.suffix == ".Identifier":
            continue
        # wav 경로에서 domain/speaker/utt 추출 후 label 경로 구성
        rel = wav_path.relative_to(wav_dataset)   # J91/S0000xxxx/0001.wav
        txt_path = label_dataset / rel.with_suffix(".txt")
        if txt_path.exists():
            pairs.append((wav_path, txt_path))

    return pairs


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def load_model(cfg: dict):
    import whisperx
    m = cfg["model"]
    print(f"모델 로드: {m['name']} ({m['device']}, {m['compute_type']})")
    return whisperx.load_model(
        m["name"], m["device"],
        compute_type=m["compute_type"],
        language=m["language"],
    ), m["device"], m["language"]


def transcribe_one(model, audio_path: str, language: str, t_cfg: dict) -> str:
    import whisperx
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(
        audio,
        batch_size=t_cfg.get("batch_size", 16),
        language=language,
        beam_size=t_cfg.get("beam_size", 5),
        temperature=t_cfg.get("temperature", 0),
    )
    return " ".join(s["text"].strip() for s in result.get("segments", []))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="WhisperX benchmark (KtelSpeech)")
    parser.add_argument("--data-dir", required=True, help="sample_01 같은 데이터 루트 디렉토리")
    parser.add_argument("--config", default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--limit", type=int, default=None, help="테스트할 최대 파일 수 (미지정 시 전체)")
    parser.add_argument("--output-dir", default="results", help="결과 저장 디렉토리")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[오류] 디렉토리 없음: {data_dir}", file=sys.stderr)
        sys.exit(1)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    pairs = find_pairs(data_dir)
    if not pairs:
        print("[오류] WAV/TXT 쌍을 찾지 못했습니다.", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"평가 파일 수: {len(pairs)}")

    model, device, language = load_model(cfg)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_wer, total_cer = 0.0, 0.0
    total_rtf_num, total_rtf_den = 0.0, 0.0
    errors = 0

    for i, (wav_path, txt_path) in enumerate(pairs, 1):
        ref_raw = txt_path.read_text(encoding="utf-8").strip()
        ref = normalize_label(ref_raw)

        t0 = time.perf_counter()
        try:
            hyp_raw = transcribe_one(model, str(wav_path), language, cfg["transcription"])
        except Exception as e:
            print(f"  [오류] {wav_path.name}: {e}", file=sys.stderr)
            errors += 1
            continue
        elapsed = time.perf_counter() - t0

        hyp = normalize_hypothesis(hyp_raw)

        w = wer(ref, hyp)
        c = cer(ref, hyp)

        # RTF: elapsed / audio_duration (근사값 — wav 헤더에서 추출)
        try:
            import wave
            with wave.open(str(wav_path)) as wf:
                duration = wf.getnframes() / wf.getframerate()
            rtf = elapsed / duration if duration > 0 else 0.0
            total_rtf_num += elapsed
            total_rtf_den += duration
        except Exception:
            rtf = 0.0

        total_wer += w
        total_cer += c

        record = {
            "file": str(wav_path.relative_to(data_dir)),
            "ref": ref,
            "hyp": hyp,
            "wer": round(w, 4),
            "cer": round(c, 4),
            "rtf": round(rtf, 4),
            "elapsed_s": round(elapsed, 3),
        }
        records.append(record)

        print(f"[{i:4d}/{len(pairs)}] WER={w:.3f} CER={c:.3f} RTF={rtf:.3f}  {wav_path.name}")

    n = len(records)
    summary = {
        "model": cfg["model"]["name"],
        "device": cfg["model"]["device"],
        "compute_type": cfg["model"]["compute_type"],
        "language": cfg["model"]["language"],
        "total_files": len(pairs),
        "evaluated": n,
        "errors": errors,
        "avg_wer": round(total_wer / n, 4) if n else None,
        "avg_cer": round(total_cer / n, 4) if n else None,
        "overall_rtf": round(total_rtf_num / total_rtf_den, 4) if total_rtf_den > 0 else None,
    }

    # 결과 저장
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_tag = cfg["model"]["name"].replace("/", "-")
    result_path = output_dir / f"benchmark_{model_tag}_{ts}.json"
    result_path.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 텍스트 요약 출력
    print("\n" + "=" * 60)
    print("[ 벤치마크 결과 ]")
    print(f"  모델       : {summary['model']}")
    print(f"  평가 파일  : {n} / {len(pairs)}  (오류: {errors})")
    print(f"  평균 WER   : {summary['avg_wer']:.4f}  ({summary['avg_wer']*100:.2f}%)")
    print(f"  평균 CER   : {summary['avg_cer']:.4f}  ({summary['avg_cer']*100:.2f}%)")
    print(f"  전체 RTF   : {summary['overall_rtf']:.4f}")
    print(f"  결과 파일  : {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
