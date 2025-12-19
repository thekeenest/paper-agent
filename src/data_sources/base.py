"""
Базовый класс для источников данных о публикациях.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from ..models import PaperMetadata


class DataSourceType(str, Enum):
    """Типы поддерживаемых источников данных"""
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"


@dataclass
class SearchParams:
    """Параметры поиска публикаций"""
    query: str
    max_results: int = 100
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    categories: Optional[List[str]] = None
    fields: Optional[List[str]] = None


class DataSourceBase(ABC):
    """
    Абстрактный базовый класс для источников данных.
    
    Все источники должны реализовывать методы:
    - search() - поиск публикаций
    - get_paper() - получение данных по ID
    - get_author() - получение информации об авторе (опционально)
    """
    
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        self._request_count = 0
    
    @abstractmethod
    def search(self, params: SearchParams) -> List[PaperMetadata]:
        """
        Поиск публикаций по параметрам.
        
        Args:
            params: Параметры поиска
            
        Returns:
            Список найденных публикаций
        """
        pass
    
    @abstractmethod
    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Получить данные о публикации по ID.
        
        Args:
            paper_id: Идентификатор публикации
            
        Returns:
            Данные публикации или None
        """
        pass
    
    def get_author(self, author_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию об авторе (опционально).
        
        Args:
            author_id: Идентификатор автора
            
        Returns:
            Данные автора или None
        """
        return None
    
    def supports_affiliations(self) -> bool:
        """Возвращает ли источник информацию об аффилиациях напрямую"""
        return False
    
    def supports_citations(self) -> bool:
        """Возвращает ли источник данные о цитированиях"""
        return False
    
    def get_request_count(self) -> int:
        """Количество выполненных запросов"""
        return self._request_count
    
    def reset_request_count(self) -> None:
        """Сбросить счётчик запросов"""
        self._request_count = 0
