"""
Состояние графа LangGraph для агентной системы.

Определяет структуру данных, передаваемых между узлами графа.
"""

from typing import List, Optional, Annotated, Dict, Any
from typing_extensions import TypedDict
import operator

from .models import PaperMetadata, AnalyticsReport


def merge_papers(left: List[PaperMetadata], right: List[PaperMetadata]) -> List[PaperMetadata]:
    """
    Функция слияния списков статей.
    Используется для аннотации Annotated в TypedDict.
    
    При обновлении состояния заменяет весь список.
    """
    if right:
        return right
    return left


class AgentState(TypedDict):
    """
    Состояние агентного графа.
    
    Attributes:
        # Входные параметры
        query: Поисковый запрос для ArXiv API
        max_papers: Максимальное количество статей для обработки
        categories: Фильтр по категориям ArXiv (опционально)
        date_from: Начальная дата фильтра
        date_to: Конечная дата фильтра
        data_source: Источник данных (arxiv, semantic_scholar, openalex)
        
        # Рабочие данные
        papers: Список статей для обработки
        current_index: Индекс текущей обрабатываемой статьи
        
        # Статистика выполнения
        processed_count: Количество успешно обработанных статей
        error_count: Количество ошибок
        start_time: Время начала выполнения (timestamp)
        
        # Флаги управления
        should_stop: Флаг остановки обработки
        retry_count: Счётчик повторных попыток для текущей статьи
        
        # Результаты
        final_report: Итоговый аналитический отчёт
        output_path: Путь к сохранённым результатам
        
        # Логирование
        logs: Список сообщений логов
        errors: Список ошибок с деталями
    """
    
    # Входные параметры
    query: str
    max_papers: int
    categories: Optional[List[str]]
    date_from: Optional[str]
    date_to: Optional[str]
    data_source: str
    
    # Рабочие данные
    papers: List[PaperMetadata]
    current_index: int
    
    # Статистика
    processed_count: int
    error_count: int
    start_time: Optional[float]
    
    # Флаги
    should_stop: bool
    retry_count: int
    max_retries: int
    
    # Результаты
    final_report: Optional[AnalyticsReport]
    output_path: Optional[str]
    
    # Логирование
    logs: Annotated[List[str], operator.add]
    errors: Annotated[List[Dict[str, Any]], operator.add]


def create_initial_state(
    query: str,
    max_papers: int = 100,
    categories: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    max_retries: int = 3,
    data_source: str = "arxiv"
) -> AgentState:
    """
    Создать начальное состояние для графа.
    
    Args:
        query: Поисковый запрос ArXiv
        max_papers: Максимум статей
        categories: Фильтр категорий
        date_from: Дата начала (YYYYMMDD)
        date_to: Дата конца (YYYYMMDD)
        max_retries: Максимум повторных попыток
        data_source: Источник данных (arxiv, semantic_scholar, openalex)
    
    Returns:
        Инициализированное состояние AgentState
    """
    import time
    
    return AgentState(
        query=query,
        max_papers=max_papers,
        categories=categories,
        date_from=date_from,
        date_to=date_to,
        data_source=data_source,
        papers=[],
        current_index=0,
        processed_count=0,
        error_count=0,
        start_time=time.time(),
        should_stop=False,
        retry_count=0,
        max_retries=max_retries,
        final_report=None,
        output_path=None,
        logs=[],
        errors=[]
    )


class NodeOutput(TypedDict, total=False):
    """
    Базовый тип для выходных данных узла.
    
    Узлы возвращают частичное обновление состояния.
    """
    papers: List[PaperMetadata]
    current_index: int
    processed_count: int
    error_count: int
    should_stop: bool
    retry_count: int
    final_report: Optional[AnalyticsReport]
    output_path: Optional[str]
    logs: List[str]
    errors: List[Dict[str, Any]]
