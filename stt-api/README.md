# /stt-api — WhisperX 기반 STT REST API

`/whisperx` 의 모듈형 파이프라인(VAD / STT / 화자분리)을 그대로 재사용하는 FastAPI 서버.
음성 파일을 업로드하면 발화 구간 + 텍스트 + 화자 정보를 JSON 으로 반환한다.

```
stt-api/
├── README.md
├── Makefile              # make help 로 사용법 확인
├── .env.example          # 모든 설정의 단일 소스 (서버 + 파이프라인)
├── requirements.txt
├── app/
│   ├── main.py             # FastAPI 엔트리포인트
│   ├── settings.py         # .env 로딩 + 파이프라인 dict 빌드
│   ├── pipeline_adapter.py # /whisperx 의 pipeline.py 를 sys.path 로 임포트
│   ├── inference.py        # ModelManager + Semaphore 동시성 제어
│   ├── language_router.py  # 런타임 언어 스위칭 (align 모델 캐시)
│   ├── jobs.py             # 비동기 job 큐
│   ├── audio.py            # 디코딩 / 길이 계산
│   ├── metrics.py          # WER / CER (벤치마크와 동일 로직)
│   └── schemas.py          # Pydantic 응답 스키마
└── tests/
    └── test_basic.py
```

## 빠른 시작

```bash
# 1. conda env 활성화 (이 프로젝트는 stt 환경 사용)
conda activate stt

# 2. 환경변수 설정
cd stt-api
make env                    # .env.example → .env 복사
# .env 를 열어 HF_TOKEN, DEVICE 등을 채운다

# 3. 의존성 설치
make install                # FastAPI 등 API 전용 패키지만 추가 설치

# 4. 서버 실행
make run                    # 또는 make dev (--reload)

# 5. 동작 확인
make health
make curl-transcribe AUDIO=../test-data/sample_01/raw_data/KtelSpeech_train_D60_wav_0/J91/S00007727/0001.wav
```

서버가 뜨면 자동 생성된 OpenAPI 문서를 브라우저에서 볼 수 있다:
- Swagger UI : http://localhost:8000/docs
- Redoc      : http://localhost:8000/redoc

---

## 환경변수

전체 목록은 [.env.example](./.env.example) 참고. 자주 쓰는 것만 추림:

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MAX_CONCURRENT_INFERENCES` | `1` | 모델에 동시에 진입할 수 있는 추론 수. **GPU 1장에선 1 권장**. 2 이상은 메모리 여유와 `int8_float16` 같은 경량 설정에서만. |
| `IO_WORKER_THREADS` | `4` | 업로드/디코딩 등 비-GPU 작업 워커 수. |
| `MAX_UPLOAD_MB` | `200` | 업로드 허용 최대 크기. |
| `STT_BACKEND` | `whisperx` | `whisperx` / `faster-whisper` / `transformers` |
| `STT_MODEL_ID` | `large-v3` | Whisper 사이즈 또는 HF 모델 ID |
| `DEFAULT_LANGUAGE` | `ko` | 요청에서 `language` 미지정 시 사용 |
| `DEVICE` | `cuda` | `cuda` / `cpu` |
| `COMPUTE_TYPE` | `float16` | `float16` / `int8` / `int8_float16` |
| `DIARIZATION_MODE` | `auto` | `auto` (HF_TOKEN 있으면 ON) / `on` / `off` |
| `HF_TOKEN` | — | pyannote VAD / 화자분리에 필요 |

설정은 `.env` 한 곳에서만 관리한다. 서버 설정(HOST/PORT/동시성)과 파이프라인 설정(STT/VAD/화자분리)이 모두 여기에 있고, 변경 후 서버 재시작만 하면 반영된다.

---

## 엔드포인트

### `POST /transcribe` — 동기 전사 (짧은 오디오)

```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@sample.wav" \
  -F "language=ko" \
  -F "reference=네 안녕하세요"   # 옵션: 주면 wer/cer 계산
```

응답:

```json
{
  "segments": [
    {"start": 0.09, "end": 1.27, "text": "안녕하십니까 고객님.", "speaker": "SPEAKER_00"}
  ],
  "text": "안녕하십니까 고객님.",
  "language": "ko",
  "elapsed_s": 1.267,
  "audio_duration_s": 1.412,
  "rtf": 0.897,
  "ref": "네 안녕하세요",
  "hyp": "안녕하십니까 고객님",
  "wer": 1.0,
  "cer": 0.71
}
```

### `POST /transcribe/async` — 비동기 전사 (긴 오디오)

```bash
# 1) job 생성
curl -X POST http://localhost:8000/transcribe/async \
  -F "file=@long_call.wav" \
  -F "language=ko"
# → {"job_id":"abc123","status":"pending"}

# 2) 결과 폴링
curl http://localhost:8000/jobs/abc123
# → status: pending → processing → done
```

### `GET /health`

```json
{
  "status": "ok",
  "model": {
    "ready": true,
    "max_concurrent_inferences": 1,
    "in_flight": 0,
    "backend": "whisperx",
    "model_id": "large-v3",
    "device": "cuda",
    "diarization_enabled": true,
    "cached_align_languages": ["ko", "en", "ja"],
    "stats": {"requests": 12, "errors": 0, "total_elapsed_s": 8.4}
  }
}
```

---

## 동시성 모델

- **모델 로드**: 서버 startup 시 **1회만** (singleton). 요청마다 다시 로드하지 않는다.
- **요청 수락**: FastAPI 가 비동기로 N 개 동시 수락. 업로드/디코딩은 스레드 풀에서 병렬 처리.
- **GPU 진입 제어**: `asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)` 로 모델 호출을 직렬화.
  - 기본값 `1` 이면 한 번에 한 요청씩 GPU 사용. 다른 요청은 큐에 비동기 대기.
  - 큐에 대기 중에도 다른 엔드포인트(`/health`, `/jobs/{id}`) 는 정상 응답.
- **스레드 풀 위임**: 실제 `pipeline.run()` 은 `asyncio.to_thread` 로 호출 → PyTorch 가 GIL 을 풀어 이벤트 루프가 막히지 않는다.

GPU 사양이 정해지면 `.env` 만 바꿔 정책을 조정한다. 멀티 GPU / 더 큰 처리량이 필요해지면
`app/inference.py` 의 `ModelManager` 에 micro-batching 워커를 추가하면 된다.

---

## 다국어 (런타임 언어 스위칭)

요청마다 `language` 폼 필드로 다른 언어를 보낼 수 있다. **모델 1개만 GPU 에 띄우고** 요청 단위로 언어를 바꿔가며 처리한다 (`app/language_router.py`).

```bash
# 같은 서버, 같은 모델 인스턴스에 한국어 / 영어 / 일본어를 섞어 요청
curl -F "file=@ko.wav" -F "language=ko" http://localhost:8000/transcribe
curl -F "file=@en.wav" -F "language=en" http://localhost:8000/transcribe
curl -F "file=@ja.wav" -F "language=ja" http://localhost:8000/transcribe
```

### 동작 방식

- **STT 모델** (`large-v3` 등) 은 원래 다국어 모델 — 그대로 공유.
- **align 모델** 은 언어 전용 (Wav2Vec2 계열, 언어당 ~300MB). 해당 언어의 첫 요청에 lazy load 되고 이후 캐시됨 → 두 번째 요청부터 즉시.
- 요청 직전에 `stt._language` 와 `_align_model` 을 swap, 추론 종료 후 복원. `threading.Lock` 으로 원자성 보장.
- `/health` 의 `cached_align_languages` 에 현재 캐시된 언어 목록이 보임.

### 메모리 산정

- Whisper large-v3 ≈ 3GB
- 사용된 언어당 align 모델 ≈ 300MB
- 한·영·일·중 4개 언어 동시 운영 = `3GB + 4×300MB ≈ 4.2GB`

### 지원 / 제약

- **지원 언어**: Whisper 가 지원하는 모든 언어 (en, ko, ja, zh, fr, de, es, ru, it, pt, hi, ar, ...). 약 100개.
- **align 모델 없는 언어**: align 단계만 자동 skip + 경고 로그. STT 자체는 계속 동작.
- **`.en` 모델** (e.g. `tiny.en`) 은 영어 전용 — 다른 언어 요청 시 whisperx 가 오류. multilingual 모델(`large-v3` 등) 사용 권장.
- **동시 추론**: 같은 모델 객체의 속성을 swap 하므로 락에 의해 사실상 직렬화됨. `MAX_CONCURRENT_INFERENCES > 1` 로 올려도 언어가 다르면 직렬화. GPU 1장 환경에선 어차피 직렬이 안전하므로 문제 없음.
- **transformers 백엔드**: best-effort. 파이프라인의 `generate_kwargs` 패치를 시도하고, 실패하면 init 언어로 동작 + 경고.

---

## /whisperx 와의 관계

- `/stt-api` 는 `/whisperx/stages` 와 `/whisperx/pipeline.py` 를 **sys.path 로 임포트만** 한다.
- 새 STT / VAD / 화자분리 모델을 추가하려면 **/whisperx 쪽 클래스만 늘리면** 되고, `/stt-api` 는 그대로 인식한다.
- 단일 파일 전사·벤치마크는 계속 `/whisperx/run.py`, `/whisperx/benchmark.py` 를 사용.

---

## 다음 단계 (현재 범위 밖)

- 인증 (API key / JWT)
- Rate limiting
- Prometheus 메트릭 노출
- 클라우드 스토리지 업로드 / 결과 webhook
- micro-batching 스케줄러
