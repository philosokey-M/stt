# =====================================================================
# /stt-api 실행 명령어 모음
# 사용:   make help
# =====================================================================

# conda env "stt" 의 인터프리터를 직접 가리킨다. (activate 안 해도 동작)
# 다른 env 를 쓰려면: make run PYTHON=/path/to/python
PYTHON ?= /home/hong/miniconda3/envs/stt/bin/python
PIP    ?= $(PYTHON) -m pip
HOST   ?= 0.0.0.0
PORT   ?= 8001

# .env 로딩 — 모든 타겟에서 사용 가능 (있을 때만)
ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: help install env run dev test health curl-transcribe lint clean \
        docker-build docker-up docker-down docker-restart docker-logs docker-shell docker-warmup

help:
	@echo ""
	@echo "  STT API — 사용 가능 명령어"
	@echo "  ---------------------------------------------"
	@echo "  make install         의존성 설치 (requirements.txt)"
	@echo "  make env             .env 파일 생성 (.env.example 복사)"
	@echo ""
	@echo "  make run             프로덕션 모드 실행 (단일 워커, 모델 1회 로드)"
	@echo "  make dev             개발 모드 실행 (--reload, 디버그 로그)"
	@echo ""
	@echo "  make health          /health 호출"
	@echo "  make curl-transcribe AUDIO=path/to/file.wav   파일 전사 테스트"
	@echo ""
	@echo "  make test            pytest 실행"
	@echo "  make clean           __pycache__ 정리"
	@echo ""
	@echo "  --- Docker (운영) ---"
	@echo "  make docker-build    이미지 빌드 (캐시 활용)"
	@echo "  make docker-up       백그라운드 기동 (모델은 hf_cache 볼륨에 영속)"
	@echo "  make docker-down     컨테이너 정지/제거 (볼륨은 유지)"
	@echo "  make docker-restart  컨테이너 재시작 (.env 변경 반영)"
	@echo "  make docker-logs     stt-api 로그 추적"
	@echo "  make docker-shell    컨테이너 안에 bash"
	@echo "  make docker-warmup   첫 셋업용: HF_OFFLINE=false 로 모델 다운로드 후 종료"
	@echo ""

install:
	@echo "stt env 에 의존성 설치 — $(PYTHON)"
	$(PIP) install -r requirements.txt

env:
	@if [ -f .env ]; then \
		echo ".env 가 이미 존재합니다. 덮어쓰려면 먼저 삭제하세요."; \
	else \
		cp .env.example .env; \
		echo ".env 생성됨. 값을 채워주세요."; \
	fi

# 프로덕션: 워커는 항상 1 (GPU/모델 중복 로드 방지).
#           동시성은 MAX_CONCURRENT_INFERENCES 로 제어.
run:
	$(PYTHON) -m uvicorn app.main:app \
		--host $(HOST) --port $(PORT) \
		--workers 1 \
		--log-level $(or $(LOG_LEVEL),info)

dev:
	$(PYTHON) -m uvicorn app.main:app \
		--host $(HOST) --port $(PORT) \
		--reload \
		--log-level debug

health:
	@curl -sf http://$(HOST):$(PORT)/health | python -m json.tool

# 사용 예: make curl-transcribe AUDIO=../test-data/sample_01/raw_data/.../0001.wav
curl-transcribe:
	@if [ -z "$(AUDIO)" ]; then \
		echo "AUDIO=<파일경로> 를 지정하세요. 예: make curl-transcribe AUDIO=sample.wav"; \
		exit 1; \
	fi
	curl -sf -X POST http://$(HOST):$(PORT)/transcribe \
		-F "audio=@$(AUDIO)" \
		-F "language=$(or $(LANG_CODE),ko)" | python -m json.tool

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true


# ============================================================
#   Docker 운영
# ============================================================
DC ?= docker compose

docker-build:
	$(DC) build

docker-up:
	$(DC) up -d
	@echo ""
	@echo "기동 중. 모델 로드 완료될 때까지 'make docker-logs' 로 확인하세요."
	@echo "준비 완료 후:  curl http://localhost:$(PORT)/health"

docker-down:
	$(DC) down

docker-restart:
	$(DC) restart stt-api

docker-logs:
	$(DC) logs -f stt-api

docker-shell:
	$(DC) exec stt-api /bin/bash

# 최초 셋업: HF_OFFLINE=false 로 일회성 실행해 모델 캐시 채우고 빠져나옴.
# 이후엔 .env 의 HF_OFFLINE=true (기본) 그대로 두고 docker-up 만 하면 됨.
docker-warmup:
	@echo "[warmup] HF_OFFLINE=false 로 모델 다운로드 시작 — HF_TOKEN 이 필요할 수 있음"
	$(DC) run --rm -e HF_OFFLINE=false stt-api \
		python -c "from app.settings import SETTINGS; from app.inference import init_manager; init_manager(SETTINGS); print('[warmup] 모델 캐시 준비 완료')"
