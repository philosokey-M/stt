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

.PHONY: help install env run dev test health curl-transcribe lint clean

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
		-F "file=@$(AUDIO)" \
		-F "language=$(or $(LANG_CODE),ko)" | python -m json.tool

test:
	$(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
