# CLAUDE.md — /stt-api

`/whisperx` 의 STT 파이프라인을 그대로 재사용하는 FastAPI 서버.
음성 파일을 업로드하면 `/results/benchmark_*.json` 의 record 와 동일한 포맷
(`segments[*].{start,end,text,speaker}` + `text/language/elapsed_s/rtf` +
선택적 `wer/cer`) 으로 JSON 응답을 돌려준다.

이 문서는 본 디렉토리에서 작업할 때의 규칙과 설계 결정을 정리한다.
프로젝트 전반 가이드는 `/home/hong/workspace/test/stt/CLAUDE.md` 참고.

---

## 핵심 원칙 (꼭 지킬 것)

1. **/whisperx 는 수정 금지**. `app/pipeline_adapter.py` 에서 `sys.path` 로
   끌어와서 `from pipeline import build_from_config` 만 임포트한다.
   새 STT/VAD/화자분리 모델이 필요하면 `/whisperx/stages/*.py` 에 클래스를
   추가하고 `/stt-api` 는 그대로 인식되게 한다. /whisperx 수정이 정말로 필요하면
   사용자에게 먼저 확인.
2. **Python 환경은 conda env `stt`**.
   인터프리터 절대경로: `/home/hong/miniconda3/envs/stt/bin/python`.
   Makefile 도 이걸 가리킨다. `conda activate stt` 가 안 되는 비대화형 셸에서도
   Makefile 명령은 그대로 동작한다.
3. **설정 단일 소스는 `.env`**. `config.yaml` 은 사용하지 않는다 (의도적으로 삭제됨).
   서버/인프라(HOST, PORT, MAX_CONCURRENT_INFERENCES, HF_TOKEN)와 파이프라인
   (STT_BACKEND, STT_MODEL_ID, DEVICE, ...)이 모두 `.env` 에 있다.
4. **`.env` 에는 인라인 코멘트 금지**. python-dotenv 가 `# 코멘트` 만 잘라내고
   값 뒤 공백은 남기는 함정이 있다. 코멘트는 변수 위쪽 줄에 적는다.
   `app/settings.py::_env()` 가 방어적으로 `.strip()` 하지만, 원칙은 위.

---

## 디렉토리 구조

```
stt-api/
├── CLAUDE.md
├── README.md
├── Makefile              # make help 로 사용법 확인
├── .env.example          # 모든 설정의 단일 소스 (서버 + 파이프라인)
├── requirements.txt      # FastAPI 계열만. whisperx/faster-whisper 는 stt env 에 이미 있음
├── app/
│   ├── main.py             # FastAPI 엔트리포인트 + lifespan
│   ├── settings.py         # .env → pipeline_config dict 구성
│   ├── pipeline_adapter.py # /whisperx 의 pipeline.py 를 sys.path 로 임포트
│   ├── inference.py        # ModelManager + Semaphore + 동기/비동기 transcribe
│   ├── language_router.py  # 런타임 언어 스위칭 (align 모델 캐시)
│   ├── jobs.py             # 비동기 job 큐 + TTL sweeper
│   ├── audio.py            # 오디오 디코딩 / 길이
│   ├── metrics.py          # WER / CER (/whisperx/benchmark.py 와 동일 로직)
│   └── schemas.py          # Pydantic 응답 스키마
└── tests/
    └── test_basic.py
```

---

## 실행

```bash
# 처음 한 번
cp .env.example .env
# .env 의 HF_TOKEN, DEVICE 등을 채운다
make install         # FastAPI 등 API 전용 패키지만 stt env 에 설치

# 개발 모드 (--reload, 디버그 로그)
make dev

# 프로덕션 모드 (단일 워커, 모델 1회 로드)
make run

# 헬스체크 / 빠른 전사 테스트
make health
make curl-transcribe AUDIO=../test-data/sample_01/raw_data/.../0001.wav
```

OpenAPI: `http://localhost:$(PORT)/docs`

---

## 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| POST | `/transcribe` | 동기 전사. 짧은 오디오 권장. `file`(필수), `language`(옵션), `reference`(옵션, 주면 wer/cer 계산). |
| POST | `/transcribe/async` | 비동기 전사. `job_id` 즉시 반환. |
| GET  | `/jobs/{job_id}` | job 상태/결과 폴링. |
| GET  | `/health` | 모델 로드 상태, in-flight 카운트, 캐시된 align 언어, stats. |
| GET  | `/` | API 메타 정보. |

응답 스키마는 `app/schemas.py::TranscribeResponse` 참고. `/results/benchmark_*.json`
의 record 와 같은 키 (`segments[*].{start,end,text,speaker}`, `wer/cer/rtf/elapsed_s`)
가 보장된다.

---

## 설계 결정

### 동시성

- **모델 로드는 startup 1회** (`app/main.py::lifespan` → `init_manager`).
  요청마다 재로드하지 않는다.
- **GPU 진입 제어**: `asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)` (기본 1).
  큐 대기 중에도 다른 엔드포인트(/health, /jobs, 새 업로드)는 정상 응답.
- **동기 모델 호출은 워커 스레드로**: `asyncio.to_thread(self._pipeline.run, ...)`
  → PyTorch 가 GIL 을 풀어 이벤트 루프가 막히지 않는다.
- **GPU 사양이 정해지지 않은 상태**이므로 `MAX_CONCURRENT_INFERENCES` 환경변수로
  병렬/직렬을 제어. 기본은 직렬(1). 2+ 는 `compute_type=int8_float16` 처럼 메모리
  가벼운 설정에서만 안전.

확장이 필요해지면 `app/inference.py::ModelManager` 에 micro-batching 워커 추가가
가장 깔끔한 진입점. 외부 인터페이스(`transcribe()`) 는 유지 가능.

### LanguageRouter — 런타임 언어 스위칭 (단일 모델)

`app/language_router.py`.

**제약**: 사용자 요청 — 메모리 부족, 그래서 언어당 모델 하나씩 띄울 수 없음.
**제약**: `/whisperx` 수정 금지.

**해결**:
- Whisper 모델 본체(`large-v3` 등)는 원래 다국어 → 그대로 공유.
- 각 요청 직전에 `stt._language` 를 monkey-patch 하고, 끝나면 복원.
- WhisperX 의 `_align_model`/`_align_metadata` 는 언어 전용 (Wav2Vec2, 약 300MB).
  → 사용된 언어별로 lazy load 해서 `_align_cache` 에 캐시.
- swap-run-restore 는 `threading.Lock` 으로 원자화. 동시 요청은 자연히 직렬화.
- 백엔드 판별은 duck typing (`hasattr(stt, "_align_model")`).
- align 모델 로드 실패하는 언어는 align 만 자동 skip + 경고 로그.
- transformers 백엔드는 `_pipe._forward_params["language"]` / `generate_kwargs`
  패치를 best-effort 로 시도.

`/health` 응답의 `cached_align_languages` 로 현재 캐시 상태 확인 가능.

### 설정

- `.env` 가 단일 소스. 코드 기본값(`_env(name, default)`) 위에 .env 가 얹어진다.
- `app/settings.py::_build_pipeline_config()` 가 `/whisperx/build_from_config()`
  가 기대하는 dict 스키마(`pipeline.vad/stt/diarization` + 최상위 `hf_token`)를
  .env 값만으로 직접 구성한다.
- 화자분리는 `DIARIZATION_MODE`(`auto`/`on`/`off`) + `HF_TOKEN` 유무로 결정:
  - `auto` (기본): 토큰 있으면 ON, 없으면 OFF — 시작 실패 없음.
  - `on`: 토큰 없으면 startup 시 RuntimeError.
  - `off`: 강제 OFF, `segments[*].speaker == null`.

### 응답 포맷

`/results/benchmark_*.json` 의 record 와 호환:
```json
{
  "segments": [{"start": 0.09, "end": 1.27, "text": "...", "speaker": "SPEAKER_00"}],
  "text": "...",
  "language": "ko",
  "elapsed_s": 1.267,
  "audio_duration_s": 1.412,
  "rtf": 0.897,
  "ref": "...",      // reference 가 같이 온 경우만
  "hyp": "...",
  "wer": 0.0,
  "cer": 0.0
}
```

`metrics.py::normalize_label/normalize_hypothesis` 는 `/whisperx/benchmark.py`
와 동일 로직 — 점수가 어긋나지 않도록.

---

## 알려진 함정

1. **Makefile `include .env` 가 위쪽 `PORT ?= 8000` 을 덮어쓴다.**
   .env 의 `KEY=value` 는 unconditional 할당이라 `?=` 보다 우선. 포트 바꾸려면
   .env 를 수정하거나 `make dev PORT=8002` 처럼 CLI 로.
2. **python-dotenv 는 인라인 코멘트의 값 뒤 공백을 strip 하지 않는다.**
   `KEY=value   # comment` → 값이 `"value   "` 가 됨. `_env()` 가 방어로
   `.strip()` 하지만, .env 자체에 인라인 코멘트를 두지 않는 게 원칙.
3. **`tiny.en` 같은 영어 전용 모델로는 다국어 요청이 깨진다.** multilingual
   모델 (`large-v3` 등) 사용 필요.
4. **`--reload` 좀비 프로세스**: uvicorn `--reload` 가 parent+worker 구조라
   비정상 종료 시 워커가 남아 포트를 점유할 수 있다. EADDRINUSE 인데 `lsof` 가
   비면 `ps aux | grep uvicorn` → `pkill -f uvicorn`.
5. **WSL2 환경**: 포트 점유가 Windows 쪽에 있을 수 있음. `lsof` 는 Linux 측만
   본다. `netstat.exe -ano | grep <port>` 로 Windows 측 확인.

---

## 의존성

`requirements.txt` 는 **API 전용 패키지만** 포함:
`fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `python-dotenv`,
`numpy`, `soundfile`.

STT 자체(`whisperx`, `faster-whisper`, `pyannote`, `torch` …)는 conda env `stt`
에 이미 설치되어 있으므로 명시하지 않는다 — 환경 정합성은 env 가 책임.

---

## 변경 이력 (요약)

- 초기 구축: FastAPI + WhisperX 파이프라인 sys.path 임포트, /transcribe(동기/비동기), job 큐.
- 동시성: `MAX_CONCURRENT_INFERENCES` 환경변수 도입 (기본 1, GPU 미확정 대응).
- 다국어: `LanguageRouter` 로 런타임 언어 스위칭. 단일 Whisper 모델 + 언어별
  align 모델 캐시. /whisperx 무수정.
- 설정 단일화: `config.yaml` 삭제 → `.env` 만으로 모든 설정 표현.
- 방어: `_env()` strip, .env 인라인 코멘트 금지 가이드라인.
