"""
FastAPI Backend для Conference Paper Agent.

Модуль предоставляет REST API и WebSocket endpoints для:
- Запуска анализа статей
- Отслеживания прогресса в реальном времени
- Получения результатов и метрик
- Визуализаций и аналитики
"""

from .app import create_app
from .models import (
    AnalysisRequest,
    AnalysisResponse,
    TaskStatus,
    TaskProgress,
    AnalyticsData,
)
from .task_manager import TaskManager

__all__ = [
    "create_app",
    "AnalysisRequest",
    "AnalysisResponse", 
    "TaskStatus",
    "TaskProgress",
    "AnalyticsData",
    "TaskManager",
]
