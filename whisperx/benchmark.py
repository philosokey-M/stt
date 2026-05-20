#!/usr/bin/env python3
"""
KtelSpeech 벤치마크. VAD / STT / Diarization 조합을 config.yaml로 제어한다.

Usage:
  python benchmark.py --data-dir test-data/sample_01 [--limit 50] [--output-dir results/]
"""

import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path


# ---------------------------------------------------------------------------
# KtelSpeech 라벨 정규화
# ---------------------------------------------------------------------------

def normalize_label(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r'\(([^)]+)\)/\([^)]+\)', r'\1', text)   # (A)/(B) → A
    text = re.sub(r'\b[a-zA-Z가-힣]{1,2}/', '', text)       # n/, 하/ 등 태그 제거
    text = re.sub(r'[^\w\s]', '', text)                      # 구두점 제거
    return ' '.join(text.split())


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
    r = [c for c in ref if c != ' ']
    h = [c for c in hyp if c != ' ']
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


# ---------------------------------------------------------------------------
# 데이터 페어링
# ---------------------------------------------------------------------------

def find_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    raw_root = data_dir / "raw_data"
    label_root = data_dir / "label_data"
    wav_dataset = next(raw_root.iterdir())
    label_dataset = next(label_root.iterdir())

    pairs = []
    for wav_path in sorted(wav_dataset.rglob("*.wav")):
        if ".Zone.Identifier" in wav_path.name:
            continue
        txt_path = label_dataset / wav_path.relative_to(wav_dataset).with_suffix(".txt")
        if txt_path.exists():
            pairs.append((wav_path, txt_path))
    return pairs


def audio_duration(wav_path: Path) -> float:
    try:
        with wave.open(str(wav_path)) as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="STT 벤치마크 (KtelSpeech)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[오류] 디렉토리 없음: {data_dir}", file=sys.stderr)
        sys.exit(1)

    from pipeline import build_from_config, load_config
    cfg = load_config(args.config)
    pipeline = build_from_config(cfg)

    p = cfg.get("pipeline", {})
    print(f"파이프라인 | VAD: {p.get('vad',{}).get('model','?')}  "
          f"STT: {p.get('stt',{}).get('model_id','?')} ({p.get('stt',{}).get('backend','?')})  "
          f"Diar: {'on' if p.get('diarization',{}).get('enabled') else 'off'}")

    pairs = find_pairs(data_dir)
    if not pairs:
        print("[오류] WAV/TXT 쌍을 찾지 못했습니다.", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        pairs = pairs[:args.limit]
    print(f"평가 파일 수: {len(pairs)}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    total_wer = total_cer = 0.0
    total_elapsed = total_audio_dur = 0.0
    errors = 0

    for i, (wav_path, txt_path) in enumerate(pairs, 1):
        ref = normalize_label(txt_path.read_text(encoding="utf-8"))
        dur = audio_duration(wav_path)

        t0 = time.perf_counter()
        try:
            result = pipeline.run(str(wav_path))
        except Exception as e:
            print(f"  [오류] {wav_path.name}: {e}", file=sys.stderr)
            errors += 1
            continue
        elapsed = time.perf_counter() - t0

        hyp = normalize_hypothesis(
            " ".join(s["text"] for s in result.get("segments", []))
        )

        w = wer(ref, hyp)
        c = cer(ref, hyp)
        rtf = elapsed / dur if dur > 0 else 0.0

        total_wer += w
        total_cer += c
        total_elapsed += elapsed
        total_audio_dur += dur

        records.append({
            "file": str(wav_path.relative_to(data_dir)),
            "ref": ref,
            "hyp": hyp,
            "wer": round(w, 4),
            "cer": round(c, 4),
            "rtf": round(rtf, 4),
            "elapsed_s": round(elapsed, 3),
        })

        print(f"[{i:4d}/{len(pairs)}] WER={w:.3f} CER={c:.3f} RTF={rtf:.3f}  {wav_path.name}")

    n = len(records)
    summary = {
        "pipeline": {
            "vad": p.get("vad", {}).get("model"),
            "stt_model": p.get("stt", {}).get("model_id"),
            "stt_backend": p.get("stt", {}).get("backend"),
            "diarization": p.get("diarization", {}).get("model") if p.get("diarization", {}).get("enabled") else "disabled",
        },
        "total_files": len(pairs),
        "evaluated": n,
        "errors": errors,
        "avg_wer": round(total_wer / n, 4) if n else None,
        "avg_cer": round(total_cer / n, 4) if n else None,
        "overall_rtf": round(total_elapsed / total_audio_dur, 4) if total_audio_dur > 0 else None,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    stt_tag = p.get("stt", {}).get("model", "unknown").replace("/", "-")
    vad_tag = p.get("vad", {}).get("model", "unknown")
    result_path = output_dir / f"benchmark_{stt_tag}_{vad_tag}_{ts}.json"
    result_path.write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("[ 벤치마크 결과 ]")
    print(f"  VAD        : {summary['pipeline']['vad']}")
    print(f"  STT        : {summary['pipeline']['stt_model']} ({summary['pipeline']['stt_backend']})")
    print(f"  화자분리   : {summary['pipeline']['diarization']}")
    print(f"  평가 파일  : {n} / {len(pairs)}  (오류: {errors})")
    if n > 0:
        print(f"  평균 WER   : {summary['avg_wer']:.4f}  ({summary['avg_wer']*100:.2f}%)")
        print(f"  평균 CER   : {summary['avg_cer']:.4f}  ({summary['avg_cer']*100:.2f}%)")
        print(f"  전체 RTF   : {summary['overall_rtf']:.4f}")
    else:
        print("  평가된 파일이 없습니다. 에러 로그를 확인하세요.")
    print(f"  결과 파일  : {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
