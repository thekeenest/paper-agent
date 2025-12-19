# =============================================================================
# Conference Paper Agent - Makefile
# =============================================================================

.PHONY: help install install-backend install-frontend run backend frontend stop \
        docker-up docker-down docker-build docker-logs test lint clean

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
	@echo "Conference Paper Agent - Makefile Commands"
	@echo "==========================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install           - Install all dependencies"
	@echo "  make install-backend   - Install Python dependencies"
	@echo "  make install-frontend  - Install Node.js dependencies"
	@echo ""
	@echo "Run locally:"
	@echo "  make run               - Run backend and frontend"
	@echo "  make backend           - Run backend only (FastAPI)"
	@echo "  make frontend          - Run frontend only (Vite)"
	@echo "  make stop              - Stop all processes"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up         - Start Docker Compose"
	@echo "  make docker-down       - Stop Docker containers"
	@echo "  make docker-build      - Rebuild Docker images"
	@echo "  make docker-logs       - Show container logs"
	@echo ""
	@echo "Testing:"
	@echo "  make test              - Run tests"
	@echo "  make lint              - Check code"
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
		. $(VENV_DIR)/bin/activate && $(PYTHON) -m pytest tests/ -v; \
	else \
		$(PYTHON) -m pytest tests/ -v; \
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
	@echo "Checking code..."
	@if [ -d "$(VENV_DIR)" ]; then \
		. $(VENV_DIR)/bin/activate && \
		$(PYTHON) -m flake8 src/ --max-line-length=100 --ignore=E501,W503 || true; \
	else \
		$(PYTHON) -m flake8 src/ --max-line-length=100 --ignore=E501,W503 || true; \
	fi
	@echo "Check complete"

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
	@echo "To start the system run:"
	@echo "  make run"
	@echo ""
	@echo "Or via Docker:"
	@echo "  make docker-up"
	@echo ""

# Aliases
start: run
up: docker-up
down: docker-down
logs: docker-logs
