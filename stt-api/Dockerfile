# syntax=docker/dockerfile:1.7
# =====================================================================
# stt-api Docker 이미지
# - 베이스: PyTorch + CUDA 12.4 + cuDNN9 runtime (~5GB)
# - 빌드 컨텍스트는 프로젝트 루트(stt-api 의 부모) — /whisperx 도 함께 포함되어야 함.
#   docker-compose.yml 에서 context: .. 로 설정.
# =====================================================================

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # HF 캐시 위치 — 볼륨으로 마운트해 영속 유지
    HF_HOME=/root/.cache/huggingface \
    # 우리 .env 와 동일한 안전 기본값
    HF_OFFLINE=true

# 시스템 의존성
#   ffmpeg      — faster_whisper.audio.decode_audio() 가 libav 사용
#   libsndfile1 — soundfile 백엔드
#   git         — pip 가 git+https 형식 의존성을 처리할 수도 있음 (whisperx 일부 버전)
#   build-essential — 일부 휠 없는 패키지 컴파일용
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------
# Python 의존성을 먼저 설치 (layer 캐시 최적화)
# requirements 만 바뀌지 않으면 이 layer 는 캐시 사용 → 빌드 시간 단축
# ---------------------------------------------------------------------
COPY stt-api/requirements.txt /app/stt-api/requirements.txt
RUN pip install --no-cache-dir -r /app/stt-api/requirements.txt

# ---------------------------------------------------------------------
# 애플리케이션 코드
# /stt-api 와 /whisperx 가 sibling 으로 있어야 settings.py 의
# WHISPERX_DIR = API_DIR.parent / "whisperx" 가 동작한다.
# ---------------------------------------------------------------------
COPY stt-api/ /app/stt-api/
COPY whisperx/ /app/whisperx/

WORKDIR /app/stt-api

EXPOSE 8000

# /health 가 status=="ok" 일 때만 정상으로 판정.
# start-period 를 길게(300s) 잡는 이유: 첫 컨테이너 기동 시 모델 로드 + (선택) 다운로드 시간.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request,json,sys; \
r=urllib.request.urlopen('http://localhost:8000/health', timeout=3); \
d=json.loads(r.read()); \
sys.exit(0 if d.get('status')=='ok' else 1)" || exit 1

# 워커는 항상 1 — 모델을 워커마다 중복 로드하지 않기 위함.
# 동시성은 .env 의 MAX_CONCURRENT_INFERENCES 로 제어.
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
