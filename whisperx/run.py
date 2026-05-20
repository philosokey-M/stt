#!/usr/bin/env python3
"""
Single-file WhisperX transcription entry point.
Usage: python run.py --audio <file> [--config config.yaml] [--output <file>]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def transcribe(audio_path: str, cfg: dict) -> dict:
    import whisperx

    model_cfg = cfg["model"]
    device = model_cfg["device"]
    compute_type = model_cfg["compute_type"]

    print(f"[1/3] 모델 로드: {model_cfg['name']} ({device}, {compute_type})")
    model = whisperx.load_model(
        model_cfg["name"],
        device,
        compute_type=compute_type,
        language=model_cfg["language"],
    )

    print(f"[2/3] 전사 중: {audio_path}")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(
        audio,
        batch_size=cfg["transcription"]["batch_size"],
        chunk_size=cfg["transcription"]["chunk_size"],
        language=model_cfg["language"],
        beam_size=cfg["transcription"].get("beam_size", 5),
        temperature=cfg["transcription"].get("temperature", 0),
    )

    if cfg["alignment"]["enabled"]:
        print("[2/3] 정렬 중 (단어 타임스탬프)...")
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, device,
            return_char_alignments=False,
        )

    if cfg["diarization"]["enabled"]:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("[경고] HF_TOKEN 환경변수가 없어 화자 분리를 건너뜁니다.", file=sys.stderr)
        else:
            print("[3/3] 화자 분리 중...")
            diarize_cfg = cfg["diarization"]
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token, device=device
            )
            diarize_segments = diarize_model(
                audio,
                min_speakers=diarize_cfg["min_speakers"],
                max_speakers=diarize_cfg["max_speakers"],
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
    else:
        print("[3/3] 화자 분리 건너뜀 (config: diarization.enabled=false)")

    return result


def format_output(result: dict) -> str:
    lines = []
    for seg in result.get("segments", []):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")
        ts = f"[{start:06.2f} --> {end:06.2f}]"
        prefix = f"  {speaker}" if speaker else ""
        lines.append(f"{ts}{prefix}  {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="WhisperX STT")
    parser.add_argument("--audio", required=True, help="입력 음성 파일 경로")
    parser.add_argument(
        "--config",
        default=Path(__file__).parent / "config.yaml",
        help="설정 파일 경로 (기본: config.yaml)",
    )
    parser.add_argument("--output", default=None, help="결과 저장 경로 (.txt 또는 .json)")
    args = parser.parse_args()

    if not Path(args.audio).exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {args.audio}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    result = transcribe(args.audio, cfg)
    formatted = format_output(result)

    print("\n" + "=" * 60)
    print(formatted)
    print("=" * 60)

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == ".json":
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            out_path.write_text(formatted, encoding="utf-8")
        print(f"\n결과 저장됨: {out_path}")


if __name__ == "__main__":
    main()
