# STT 연구 및 실험 공간

한국어 STT 파이프라인을 연구하고 최적화하는 실험 워크스페이스.  
WhisperX를 시작점으로 VAD / STT / 화자분리 각 단계를 독립적으로 교체하며 최적 조합을 찾는 것이 목표.

---

## 환경 설정

```bash
conda activate stt

# whisperx 설치
pip install -r requirements.txt
```

GPU 없는 경우 `whisperx/config.yaml` 수정:

```yaml
model:
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

### Phase 1 — WhisperX 기본 STT ✅

음성 파일 하나를 인자로 받아 전사 결과를 출력하는 스크립트.

```bash
python whisperx/run.py --audio test-data/sample_01/raw_data/KtelSpeech_train_D60_wav_0/J91/S00007750/0001.wav

# 결과를 파일로 저장
python whisperx/run.py --audio <파일경로> --output result.txt    # 텍스트
python whisperx/run.py --audio <파일경로> --output result.json   # 타임스탬프 포함
```

출력 포맷:

```
[00:00.00 --> 00:03.20]  네 안녕하세요 제 번호로 조회해서 정보 한번 보시겠어요
```

### Phase 2 — 한국어 성능 벤치마크 ✅

WER / CER / RTF 지표를 산출하고 `results/` 디렉토리에 결과를 저장.

```bash
# 빠른 검증 (N개 파일)
python whisperx/benchmark.py --data-dir test-data/sample_01 --limit 50

# 전체 평가 (5,260개)
python whisperx/benchmark.py --data-dir test-data/sample_01

# 다른 config 사용
python whisperx/benchmark.py --data-dir test-data/sample_01 --config whisperx/config.yaml
```

결과 파일: `results/benchmark_<model>_<timestamp>.json`

```
평균 WER   : 0.1234  (12.34%)
평균 CER   : 0.0876  ( 8.76%)
전체 RTF   : 0.0421
```

### Phase 3 — 모듈형 파이프라인 (예정)

VAD / STT / Diarization 각 단계를 `config.yaml`만 수정해 교체 가능하도록 구조 변경.

계획 중인 config 구조:

```yaml
vad:
  model: "silero"        # silero | pyannote | webrtc

stt:
  model: "large-v3"

diarization:
  model: "pyannote"      # pyannote | nemo | speechbrain
  enabled: false
```

### Phase 4 — 조합 탐색 자동화 (예정)

오픈소스 모델 조합을 자동으로 실행하고 결과를 비교 테이블로 정리.

```bash
# 예정된 사용법
python whisperx/sweep.py --data-dir test-data/sample_01 --limit 100
```

---

## 설정 파일 (`whisperx/config.yaml`)

```yaml
model:
  name: "large-v3"       # tiny | base | small | medium | large-v1 | large-v2 | large-v3
  language: "ko"
  compute_type: "float16" # float16 (GPU) | int8 (CPU)
  device: "cuda"          # cuda | cpu

transcription:
  batch_size: 16
  beam_size: 5            # 1=greedy, 5=기본값. 높을수록 정확도↑ 속도↓
  temperature: 0          # 0=beam search 고정

alignment:
  enabled: true           # 단어 단위 타임스탬프

diarization:
  enabled: false          # true 시 HF_TOKEN 환경변수 필요
```

---

## 파일 구조

```
stt/
├── README.md
├── CLAUDE.md
├── test-data/
│   └── sample_01/           # KtelSpeech 샘플 데이터
└── whisperx/
    ├── run.py               # 단일 파일 전사
    ├── benchmark.py         # 성능 벤치마크
    ├── config.yaml          # 모델 설정
    └── requirements.txt
```

