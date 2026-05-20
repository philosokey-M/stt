# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

This repository is an STT (Speech-to-Text) research and experimentation workspace. The goal is to benchmark and optimize open-source STT pipelines for Korean speech, with a modular architecture that allows swapping individual components (VAD / STT / Speaker Diarization).

## Directory Structure

```
/test-data/      # Audio files for STT testing (wav, mp3, etc.)
/whisperx/       # WhisperX-based pipeline scripts and benchmarks
/<module>/       # Future: other STT modules added as experiments grow
```

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install WhisperX (requires CUDA for GPU acceleration)
pip install whisperx

# For speaker diarization (requires HuggingFace token)
# Set HF_TOKEN environment variable
export HF_TOKEN=<your_huggingface_token>
```

## Test Dataset

`test-data/sample_01/` — KtelSpeech (한국어 전화 상담 음성, 5,260 쌍)

```
sample_01/
  raw_data/KtelSpeech_train_D60_wav_0/J91/<speaker_id>/<utt>.wav
  label_data/KtelSpeech_train_D60_label_0/J91/<speaker_id>/<utt>.txt
```

라벨 포맷: `n/ 텍스트 하/ 내용 (이거요)/(이거요)?`
- `n/`, `하/` 등 태그는 정규화 시 제거
- `(A)/(B)` 는 A 선택

## Commands

```bash
# 단일 파일 전사
python whisperx/run.py --audio test-data/sample_01/raw_data/.../0001.wav

# 빠른 테스트 (50개 파일)
python whisperx/benchmark.py --data-dir test-data/sample_01 --limit 50

# 전체 벤치마크
python whisperx/benchmark.py --data-dir test-data/sample_01

# 결과는 results/ 디렉토리에 benchmark_<model>_<timestamp>.json 으로 저장
```

## Architecture

### Pipeline Stages

Each pipeline has three independently swappable stages:

1. **VAD (Voice Activity Detection)** — detects speech segments before passing to STT
2. **STT (Speech-to-Text)** — transcribes segmented audio
3. **Diarization** — identifies and labels individual speakers

The configuration file (`config.yaml` per module) controls which model is used at each stage, enabling automated combinatorial testing without code changes.

### Benchmark Output

Benchmark runs produce objective artifacts:
- WER (Word Error Rate) per audio file
- CER (Character Error Rate) — especially relevant for Korean
- RTF (Real-Time Factor) for speed measurement
- Per-stage latency breakdown
- Summary report in `results/` directory

### Korean-Specific Considerations

- Korean requires character-level evaluation (CER) in addition to WER
- KtelSpeech labels contain noise/filler tags (`n/`, `하/`) and alternative spellings `(A)/(B)` — `normalize_label()` in `benchmark.py` handles these before scoring
- WER and CER are both computed; CER is more meaningful for Korean due to spacing variability

### Adding a New Module

Create a new top-level directory (e.g., `/faster-whisper/`) following the same interface:
- `run.py` — single-file transcription entry point
- `benchmark.py` — evaluation against test-data
- `config.yaml` — model selection per pipeline stage
