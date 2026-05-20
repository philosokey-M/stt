# STT 연구 및 실험 공간

한국어 STT 파이프라인을 연구하고 최적화하는 실험 워크스페이스.  
WhisperX를 시작점으로 VAD / STT / 화자분리 각 단계를 독립적으로 교체하며 최적 조합을 찾는 것이 목표.

---

## 환경 설정

```bash
conda activate lxp_ai

pip install -r requirements.txt
```

GPU 없는 경우 `config.yaml` 수정:

```yaml
pipeline:
  stt:
    device: "cpu"
    compute_type: "int8"
```

---

## 데이터셋

`test-data/sample_01/` — KtelSpeech (한국어 전화 상담 음성, 5,260 쌍)

```
sample_01/
├── raw_data/KtelSpeech_train_D60_wav_0/J91/<speaker_id>/<utt>.wav
└── label_data/KtelSpeech_train_D60_label_0/J91/<speaker_id>/<utt>.txt
```

라벨 포맷 예시:

```
n/ 네 해지 처리해드리겠습니다. 고객님 정말 죄송합니다만 카드번호를.
n/ 하/ 뭘 확인하기 어려워요. 그냥 오늘 결제된 부분에 있어서 빨리 일처리 해주세요.
n/ 아닙니다. 무통장입금으로 하면 좋았을 것을. 그건 또 할인 혜택 적용 (안될 거)/(않될 거)?
```

- `n/`, `하/` 등 태그 → 벤치마크 시 자동 제거
- `(A)/(B)` 대안 표기 → A 선택

---

## 작업 계획

### Phase 1 — 기본 STT 스크립트 ✅

음성 파일 하나를 인자로 받아 전사 결과를 출력.

```bash
# whisperx/ 디렉토리 안에서 실행
python run.py --audio ../test-data/sample_01/raw_data/KtelSpeech_train_D60_wav_0/J91/S00007750/0001.wav

# 결과 저장
python run.py --audio <파일경로> --output result.txt
python run.py --audio <파일경로> --output result.json
```

출력 포맷:

```
[000.000 --> 003.200]  네 안녕하세요 제 번호로 조회해서 정보 한번 보시겠어요
```

### Phase 2 — 한국어 성능 벤치마크 ✅

WER / CER / RTF 지표 산출 후 `results/`에 JSON 저장.

```bash
# whisperx/ 디렉토리 안에서 실행
python benchmark.py --data-dir ../test-data/sample_01 --limit 50   # 빠른 검증
python benchmark.py --data-dir ../test-data/sample_01               # 전체 (5,260개)
```

결과 파일: `results/benchmark_<stt_model>_<vad>_<timestamp>.json`

```
평균 WER   : 0.1234  (12.34%)
평균 CER   : 0.0876  ( 8.76%)
전체 RTF   : 0.0421
```

### Phase 3 — 모듈형 파이프라인 ✅

VAD / STT / Diarization을 독립 클래스로 분리. `config.yaml`만 바꿔서 조합 교체 가능.

```
stages/
  vad.py       # VADBase, SileroVAD, PyannoteVAD
  stt.py       # STTBase, FasterWhisperSTT
  diarizer.py  # DiarizationBase, PyannotesDiarizer
pipeline.py    # Pipeline + build_from_config()
```

새 모델 추가 방법:
1. 해당 `stages/*.py`에 클래스 구현 (`VADBase` 등 상속)
2. `*_REGISTRY`에 등록
3. `config.yaml`에서 `model:` 값 변경

### Phase 4 — 조합 탐색 자동화 (예정)

오픈소스 모델 조합을 자동으로 실행하고 결과를 비교 테이블로 정리.

```bash
# 예정된 사용법
python sweep.py --data-dir ../test-data/sample_01 --limit 100
```

---

## 설정 파일 (`config.yaml`)

```yaml
pipeline:
  vad:
    model: "silero"           # silero | pyannote | disabled
    onset: 0.500
    offset: 0.363
    min_speech_duration_ms: 250
    min_silence_duration_ms: 2000

  stt:
    backend: "faster-whisper"
    model: "large-v3"         # tiny | base | small | medium | large-v1 | large-v2 | large-v3
    language: "ko"
    device: "cuda"            # cuda | cpu
    compute_type: "float16"   # float16 (GPU) | int8 (CPU)
    beam_size: 5
    temperature: 0

  diarization:
    model: "pyannote"
    enabled: false            # true 시 HF_TOKEN 환경변수 필요
```

---

## 파일 구조

```
stt/
├── CLAUDE.md
├── test-data/
│   └── sample_01/               # KtelSpeech 샘플 데이터
└── whisperx/
    ├── README.md
    ├── run.py                   # 단일 파일 전사
    ├── benchmark.py             # 성능 벤치마크
    ├── pipeline.py              # Pipeline + build_from_config()
    ├── config.yaml              # 파이프라인 설정
    ├── requirements.txt
    └── stages/
        ├── vad.py               # SileroVAD, PyannoteVAD
        ├── stt.py               # FasterWhisperSTT
        └── diarizer.py          # PyannotesDiarizer
```

