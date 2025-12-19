# Backend Dockerfile for Conference Paper Agent
# Multi-stage build for optimized image size

FROM python:3.11-slim

WORKDIR /app

# ВАЖНО: Отключаем буферизацию вывода python, чтобы видеть логи в реальном времени
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create logs directory
RUN mkdir -p logs

# Expose the port (информативно, Railway сам прокидывает $PORT)
EXPOSE 8000

# ИСПРАВЛЕНИЕ:
# 1. Убрали явный вызов sh -c (Docker делает это сам).
# 2. Убрали одинарные кавычки, чтобы переменная $PORT корректно раскрылась.
CMD uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8000}