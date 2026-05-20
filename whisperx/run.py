#!/usr/bin/env python3
"""
단일 음성 파일 전사.
Usage: python run.py --audio <file> [--config config.yaml] [--output <file>]
"""

import argparse
import json
import sys
from pathlib import Path


def format_output(result: dict) -> str:
    lines = []
    for seg in result.get("segments", []):
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "")
        ts = f"[{start:07.3f} --> {end:07.3f}]"
        prefix = f"  {speaker}" if speaker else ""
        lines.append(f"{ts}{prefix}  {text}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="STT 단일 파일 전사")
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

    from pipeline import build_from_config, load_config
    cfg = load_config(args.config)
    p = cfg.get("pipeline", {})
    print(f"VAD: {p.get('vad', {}).get('model', '?')}  "
          f"STT: {p.get('stt', {}).get('model_id', '?')} ({p.get('stt', {}).get('backend', '?')})  "
          f"Diar: {'on' if p.get('diarization', {}).get('enabled') else 'off'}")

    pipeline = build_from_config(cfg)
    result = pipeline.run(args.audio)
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
