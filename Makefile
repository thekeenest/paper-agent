# =============================================================================
# Conference Paper Agent - Makefile
# =============================================================================

.PHONY: help install install-backend install-frontend run backend frontend stop \
        docker-up docker-down docker-build docker-logs \
        setup test lint typecheck repro repro-judges v1-cli v2-cli clean \
        kg-ingest kg-query bench-run bench-agreement final-report leaderboard

# Variables
PYTHON := python3
PIP := pip3
NPM := npm
VENV_DIR := ../.venv
BACKEND_PORT := 8000
FRONTEND_PORT := 5173

# =============================================================================
# Help
# =============================================================================

help:
	@echo ""
	@echo "Paper-Agent — Makefile Commands"
	@echo "================================"
	@echo ""
	@echo "Setup:"
	@echo "  make setup             - Install deps + verify .env (first-time)"
	@echo "  make install           - Install all dependencies (backend + frontend)"
	@echo "  make install-backend   - Install Python dependencies"
	@echo "  make install-frontend  - Install Node.js dependencies"
	@echo ""
	@echo "Run locally:"
	@echo "  make v1-cli            - Run v1 CLI (python -m src.v1.main)"
	@echo "  make v2-cli            - Run v2 CLI (placeholder)"
	@echo "  make run               - Run backend and frontend"
	@echo "  make backend           - Run backend only (FastAPI)"
	@echo "  make frontend          - Run frontend only (Vite)"
	@echo "  make stop              - Stop all processes"
	@echo ""
	@echo "Code quality:"
	@echo "  make test              - Run pytest test suite"
	@echo "  make lint              - ruff check src/"
	@echo "  make typecheck         - mypy src/v2/orchestration src/v2/agents"
	@echo ""
	@echo "Research / Evaluation:"
	@echo "  make repro             - Reproduce report from cache in <30 min (no API calls)"
	@echo "  make repro-full        - Full harness run (7 systems × 3 judges, needs API keys)"
	@echo "  make repro-judges      - Run LLM judge protocol on dev split"
	@echo "  make bench-run         - Run one system  (SYSTEM=full_v2 SPLIT=test)"
	@echo "  make bench-agreement   - Compute Cohen κ + Krippendorff α (SPLIT=dev)"
	@echo "  make final-report      - Regenerate experiments/final/REPORT.md from cache"
	@echo "  make leaderboard       - Regenerate benchmark/leaderboard.md from cache"
	@echo ""
	@echo "KG Layer:"
	@echo "  make kg-ingest INPUT=output/v2/work_items.jsonl  - Ingest JSONL into KG"
	@echo "  make kg-query Q='industry_share_by_venue --venue NeurIPS --year 2024'"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up         - Start Docker Compose"
	@echo "  make docker-down       - Stop Docker containers"
	@echo "  make docker-build      - Rebuild Docker images"
	@echo "  make docker-logs       - Show container logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             - Remove temp files"
	@echo ""

# =============================================================================
# Installation
# =============================================================================

install: install-backend install-frontend
	@echo "All dependencies installed"

install-backend:
	@echo "Installing Python dependencies..."
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && $(PIP) install -r requirements.txt; \
	else \
		$(PIP) install -r requirements.txt; \
	fi
	@echo "Backend dependencies installed"

install-frontend:
	@echo "Installing Node.js dependencies..."
	@cd frontend && $(NPM) install
	@echo "Frontend dependencies installed"

# =============================================================================
# Local run
# =============================================================================

run: check-env
	@echo "Starting backend and frontend..."
	@echo "Backend:  http://localhost:$(BACKEND_PORT)"
	@echo "Frontend: http://localhost:$(FRONTEND_PORT)"
	@echo "API Docs: http://localhost:$(BACKEND_PORT)/docs"
	@echo ""
	@echo "Press Ctrl+C to stop"
	@$(MAKE) -j2 backend-bg frontend-bg

backend: check-env
	@echo "Starting FastAPI backend on port $(BACKEND_PORT)..."
	@echo "API:  http://localhost:$(BACKEND_PORT)"
	@echo "Docs: http://localhost:$(BACKEND_PORT)/docs"
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && $(PYTHON) run_server.py; \
	else \
		$(PYTHON) run_server.py; \
	fi

backend-bg: check-env
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && $(PYTHON) run_server.py; \
	else \
		$(PYTHON) run_server.py; \
	fi

frontend:
	@echo "Starting React frontend on port $(FRONTEND_PORT)..."
	@echo "URL: http://localhost:$(FRONTEND_PORT)"
	@cd frontend && $(NPM) run dev

frontend-bg:
	@cd frontend && $(NPM) run dev

stop:
	@echo "Stopping processes..."
	@-pkill -f "uvicorn src.api.app:app" 2>/dev/null || true
	@-pkill -f "run_server.py" 2>/dev/null || true
	@-pkill -f "vite" 2>/dev/null || true
	@echo "Processes stopped"

# =============================================================================
# Docker
# =============================================================================

docker-up: check-env
	@echo "Starting Docker Compose..."
	docker-compose up -d
	@echo "Containers started"
	@echo "Backend:  http://localhost:8000"
	@echo "Frontend: http://localhost:3000"

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down
	@echo "Containers stopped"

docker-build: check-env
	@echo "Rebuilding Docker images..."
	docker-compose build --no-cache
	@echo "Images built"

docker-logs:
	docker-compose logs -f

docker-restart: docker-down docker-up

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "Running tests..."
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && $(PYTHON) -m pytest -q; \
	else \
		$(PYTHON) -m pytest -q; \
	fi

test-data-sources:
	@echo "Testing data sources..."
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && $(PYTHON) test_data_sources.py; \
	else \
		$(PYTHON) test_data_sources.py; \
	fi

evaluate:
	@echo "Running evaluation..."
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && $(PYTHON) evaluate.py; \
	else \
		$(PYTHON) evaluate.py; \
	fi

lint:
	@echo "Running ruff check..."
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && $(PYTHON) -m ruff check src/; \
	else \
		$(PYTHON) -m ruff check src/; \
	fi
	@echo "Lint complete"

typecheck:
	@echo "Running mypy on v2 packages..."
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && $(PYTHON) -m mypy src/v2/orchestration src/v2/agents; \
	else \
		$(PYTHON) -m mypy src/v2/orchestration src/v2/agents; \
	fi
	@echo "Type-check complete"

repro:
	@echo "Reproducing benchmark experiments from cached predictions..."
	@echo "  Split: $(if $(SPLIT),$(SPLIT),test)  Budget: $(if $(MAX_USD),$(MAX_USD),80) USD"
	@$(PYTHON) -m src.v2.eval.harness \
		--split "$(if $(SPLIT),$(SPLIT),test)" \
		--max-usd "$(if $(MAX_USD),$(MAX_USD),80)" \
		--from-cache \
		--skip-judges \
		$(if $(QUIET),--quiet,)
	@echo ""
	@echo "Report: experiments/final/REPORT.md"
	@echo "Leaderboard: benchmark/leaderboard.md"

repro-full:
	@echo "Running full Stage-8 harness (requires API keys + network)..."
	@$(PYTHON) -m src.v2.eval.harness \
		--split "$(if $(SPLIT),$(SPLIT),test)" \
		--max-usd "$(if $(MAX_USD),$(MAX_USD),80)" \
		$(if $(FROM_SCRATCH),,--from-cache) \
		$(if $(SKIP_JUDGES),--skip-judges,) \
		$(if $(QUIET),--quiet,)

repro-judges:
	@echo "Running LLM judge protocol on dev split..."
	@$(PYTHON) -m src.v2.eval.llm_judge \
		--split "$(if $(SPLIT),$(SPLIT),dev)" \
		--judges claude gpt4o gemini

bench-run:
	@echo "Running benchmark for system: $(if $(SYSTEM),$(SYSTEM),full_v2) on split: $(if $(SPLIT),$(SPLIT),test)"
	@$(PYTHON) -m src.v2.eval.runner \
		--system "$(if $(SYSTEM),$(SYSTEM),full_v2)" \
		--split "$(if $(SPLIT),$(SPLIT),test)" \
		$(if $(VERBOSE),--verbose,)

bench-agreement:
	@echo "Computing inter-annotator and inter-judge agreement..."
	@$(PYTHON) -m src.v2.eval.agreement \
		--split "$(if $(SPLIT),$(SPLIT),dev)"

final-report:
	@echo "Generating experiments/final/REPORT.md..."
	@$(PYTHON) -m src.v2.eval.harness \
		--split "$(if $(SPLIT),$(SPLIT),test)" \
		--skip-judges \
		--from-cache \
		--quiet
	@echo "Done: experiments/final/REPORT.md"

leaderboard:
	@echo "Updating benchmark/leaderboard.md..."
	@$(PYTHON) -m src.v2.eval.harness \
		--split "$(if $(SPLIT),$(SPLIT),test)" \
		--skip-judges \
		--from-cache \
		--quiet
	@echo "Done: benchmark/leaderboard.md"

# =============================================================================
# KG Layer
# =============================================================================

kg-ingest:
	@echo "Ingesting pipeline output into KuzuDB KG..."
	@$(PYTHON) -m src.v2.kg.cli ingest \
		--input "$(if $(INPUT),$(INPUT),output/v2/work_items.jsonl)" \
		--db "$(if $(DB),$(DB),output/v2/kg)"

kg-query:
	@echo "Running KG query: $(Q)"
	@$(PYTHON) -m src.v2.kg.cli query \
		--db "$(if $(DB),$(DB),output/v2/kg)" \
		$(Q)

v1-cli: check-env
	@echo "Running v1 CLI..."
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && $(PYTHON) -m src.v1.main $(ARGS); \
	else \
		$(PYTHON) -m src.v1.main $(ARGS); \
	fi

v2-cli: check-env
	@echo "Running v2 pipeline..."
	@$(PYTHON) -m src.v2.cli \
		--query "$(if $(QUERY),$(QUERY),cat:cs.AI)" \
		--n "$(if $(N),$(N),1)" \
		--output "$(if $(OUTPUT),$(OUTPUT),output/v2)" \
		$(if $(RESUME),--resume,) \
		$(if $(VERBOSE),--verbose,)

# =============================================================================
# Utilities
# =============================================================================

check-env:
	@if [ ! -f ".env" ]; then \
		echo "Error: .env file not found"; \
		echo "Create .env file based on .env.example:"; \
		echo "  cp .env.example .env"; \
		exit 1; \
	fi
	@if ! grep -q "OPENAI_API_KEY" .env 2>/dev/null; then \
		echo "Error: OPENAI_API_KEY not found in .env"; \
		exit 1; \
	fi
	@echo "Config .env verified"

clean:
	@echo "Cleaning temp files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf frontend/node_modules 2>/dev/null || true
	@rm -rf frontend/dist 2>/dev/null || true
	@echo "Temp files removed"

clean-output:
	@echo "Cleaning output directory..."
	@rm -rf output/*.json output/*.csv 2>/dev/null || true
	@echo "Output cleaned"

# =============================================================================
# Development
# =============================================================================

dev-backend: check-env
	@echo "Starting backend in dev mode (hot-reload)..."
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		uvicorn src.api.app:app --reload --host 0.0.0.0 --port $(BACKEND_PORT); \
	else \
		uvicorn src.api.app:app --reload --host 0.0.0.0 --port $(BACKEND_PORT); \
	fi

shell:
	@echo "Starting Python shell..."
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && $(PYTHON) -i -c "from src.graph import build_graph; from src.models import *; print('Modules loaded')"; \
	else \
		$(PYTHON) -i -c "from src.graph import build_graph; from src.models import *; print('Modules loaded')"; \
	fi

# =============================================================================
# Quick start
# =============================================================================

setup: install check-env
	@echo ""
	@echo "Setup complete!"
	@echo ""
	@echo "Run the v1 pipeline:  make v1-cli ARGS='--query cat:cs.AI --max-papers 5'"
	@echo "Run the test suite:   make test"
	@echo "Run the full stack:   make run"
	@echo "Or via Docker:        make docker-up"
	@echo ""

# Aliases
start: run
up: docker-up
down: docker-down
logs: docker-logs
