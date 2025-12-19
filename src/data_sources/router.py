"""
Data Source Router

Унифицированный интерфейс для всех источников данных.
Позволяет переключаться между источниками и комбинировать их.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum

from .base import DataSourceBase, DataSourceType, SearchParams
from .arxiv_client import ArxivClient
from .semantic_scholar import SemanticScholarClient
from .openalex import OpenAlexClient
from .ror import RORLookup, get_ror_lookup
from ..models import PaperMetadata, AuthorAffiliation


class DataSourceRouter:
    """
    Маршрутизатор источников данных.
    
    Обеспечивает:
    - Унифицированный интерфейс для всех источников
    - Автоматический выбор источника по типу запроса
    - Обогащение данных из нескольких источников
    - Fallback при ошибках
    
    Примеры использования:
        router = DataSourceRouter()
        
        # Поиск в конкретном источнике
        papers = router.search("machine learning", source="openalex", max_results=50)
        
        # Поиск с автовыбором источника
        papers = router.search("arxiv:2401.12345")
        
        # Обогащение данных из ArXiv через Semantic Scholar
        enriched = router.enrich_paper(paper, sources=["semantic_scholar"])
    """
    
    def __init__(
        self,
        default_source: DataSourceType = DataSourceType.ARXIV,
        enable_ror: bool = True
    ):
        """
        Args:
            default_source: Источник по умолчанию
            enable_ror: Включить ROR для нормализации организаций
        """
        self.default_source = default_source
        self.enable_ror = enable_ror
        
        # Инициализация клиентов (ленивая)
        self._clients: Dict[DataSourceType, DataSourceBase] = {}
        self._ror: Optional[RORLookup] = None
    
    def _get_client(self, source: DataSourceType) -> DataSourceBase:
        """Получить или создать клиент для источника"""
        if source not in self._clients:
            if source == DataSourceType.ARXIV:
                self._clients[source] = ArxivClient()
            elif source == DataSourceType.SEMANTIC_SCHOLAR:
                self._clients[source] = SemanticScholarClient()
            elif source == DataSourceType.OPENALEX:
                self._clients[source] = OpenAlexClient()
            else:
                raise ValueError(f"Unknown data source: {source}")
        
        return self._clients[source]
    
    def _get_ror(self) -> RORLookup:
        """Получить ROR клиент"""
        if self._ror is None:
            self._ror = get_ror_lookup()
        return self._ror
    
    def search(
        self,
        query: str,
        source: Optional[Union[str, DataSourceType]] = None,
        max_results: int = 100,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        categories: Optional[List[str]] = None,
        enrich_affiliations: bool = False
    ) -> List[PaperMetadata]:
        """
        Поиск публикаций.
        
        Args:
            query: Поисковый запрос
            source: Источник данных (или None для автоопределения)
            max_results: Максимум результатов
            date_from: Начальная дата (YYYYMMDD)
            date_to: Конечная дата (YYYYMMDD)
            categories: Фильтр по категориям
            enrich_affiliations: Обогащать аффилиации через другие источники
            
        Returns:
            Список найденных публикаций
        """
        # Определяем источник
        if source is None:
            source = self._detect_source(query)
        elif isinstance(source, str):
            source = DataSourceType(source)
        
        # Получаем клиент
        client = self._get_client(source)
        
        # Параметры поиска
        params = SearchParams(
            query=query,
            max_results=max_results,
            date_from=date_from,
            date_to=date_to,
            categories=categories
        )
        
        # Выполняем поиск
        print(f"[DataSourceRouter] Searching in {client.name}...")
        papers = client.search(params)
        
        # Обогащаем аффилиации если нужно
        if enrich_affiliations and not client.supports_affiliations():
            papers = self._enrich_affiliations(papers)
        
        # Нормализуем организации через ROR если включено
        if self.enable_ror:
            papers = self._normalize_with_ror(papers)
        
        print(f"[DataSourceRouter] Found {len(papers)} papers from {client.name}")
        return papers
    
    def get_paper(
        self,
        paper_id: str,
        source: Optional[Union[str, DataSourceType]] = None
    ) -> Optional[PaperMetadata]:
        """
        Получить публикацию по ID.
        
        Args:
            paper_id: ID публикации
            source: Источник (или автоопределение по формату ID)
            
        Returns:
            Данные публикации или None
        """
        if source is None:
            source = self._detect_source_by_id(paper_id)
        elif isinstance(source, str):
            source = DataSourceType(source)
        
        client = self._get_client(source)
        return client.get_paper(paper_id)
    
    def enrich_paper(
        self,
        paper: PaperMetadata,
        sources: Optional[List[DataSourceType]] = None
    ) -> PaperMetadata:
        """
        Обогатить данные о публикации из дополнительных источников.
        
        Args:
            paper: Исходные данные публикации
            sources: Источники для обогащения (по умолчанию Semantic Scholar + OpenAlex)
            
        Returns:
            Обогащённые данные публикации
        """
        if sources is None:
            sources = [DataSourceType.SEMANTIC_SCHOLAR, DataSourceType.OPENALEX]
        
        # Очищаем ArXiv ID от версии для поиска в других источниках
        arxiv_id = paper.arxiv_id
        if arxiv_id and "v" in arxiv_id:
            # "2512.16917v1" -> "2512.16917"
            arxiv_id = arxiv_id.split("v")[0]
        
        for source in sources:
            try:
                client = self._get_client(source)
                
                # Пытаемся найти статью по ArXiv ID
                enriched = client.get_paper(arxiv_id)
                
                if enriched and enriched.authors:
                    # Обогащаем авторов если у исходных данных нет аффилиаций
                    if not any(a.raw_affiliation for a in paper.authors):
                        paper.authors = enriched.authors
                        print(f"[DataSourceRouter] Enriched affiliations from {client.name}")
                        break
                    else:
                        # Дополняем недостающие аффилиации
                        enriched_map = {a.name.lower(): a for a in enriched.authors}
                        for author in paper.authors:
                            if not author.raw_affiliation:
                                enriched_author = enriched_map.get(author.name.lower())
                                if enriched_author and enriched_author.raw_affiliation:
                                    author.raw_affiliation = enriched_author.raw_affiliation
                                    author.normalized_affiliation = enriched_author.normalized_affiliation
                                    author.country = enriched_author.country
                                    author.country_code = enriched_author.country_code
            except Exception as e:
                print(f"[DataSourceRouter] Enrichment from {source} failed: {e}")
                continue
        
        return paper
    
    def _detect_source(self, query: str) -> DataSourceType:
        """Определить источник по формату запроса"""
        query_lower = query.lower()
        
        # ArXiv-специфичные запросы
        if "cat:" in query_lower or "arxiv" in query_lower:
            return DataSourceType.ARXIV
        
        # Semantic Scholar лучше для поиска по авторам
        if "author:" in query_lower or "au:" in query_lower:
            return DataSourceType.SEMANTIC_SCHOLAR
        
        # OpenAlex для общего поиска (лучше покрытие)
        return self.default_source
    
    def _detect_source_by_id(self, paper_id: str) -> DataSourceType:
        """Определить источник по формату ID"""
        if paper_id.startswith("10."):
            # DOI - лучше через OpenAlex
            return DataSourceType.OPENALEX
        elif "." in paper_id and paper_id.split(".")[0].isdigit():
            # ArXiv ID (YYMM.NNNNN)
            return DataSourceType.ARXIV
        elif paper_id.startswith("W"):
            # OpenAlex work ID
            return DataSourceType.OPENALEX
        else:
            # По умолчанию Semantic Scholar
            return DataSourceType.SEMANTIC_SCHOLAR
    
    def _enrich_affiliations(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Обогатить аффилиации для списка статей"""
        # Используем Semantic Scholar для обогащения
        ss_client = self._get_client(DataSourceType.SEMANTIC_SCHOLAR)
        
        for paper in papers:
            if not any(a.raw_affiliation for a in paper.authors):
                # Очищаем ArXiv ID от версии
                arxiv_id = paper.arxiv_id
                if arxiv_id and "v" in arxiv_id:
                    arxiv_id = arxiv_id.split("v")[0]
                
                try:
                    enriched = ss_client.get_paper(arxiv_id)
                    if enriched and enriched.authors:
                        # Сопоставляем по именам
                        enriched_map = {a.name.lower(): a for a in enriched.authors}
                        for author in paper.authors:
                            enriched_author = enriched_map.get(author.name.lower())
                            if enriched_author and enriched_author.raw_affiliation:
                                author.raw_affiliation = enriched_author.raw_affiliation
                except Exception:
                    continue
        
        return papers
    
    def _normalize_with_ror(self, papers: List[PaperMetadata]) -> List[PaperMetadata]:
        """Нормализовать организации через ROR"""
        ror = self._get_ror()
        
        for paper in papers:
            for author in paper.authors:
                if author.raw_affiliation and not author.normalized_affiliation:
                    try:
                        result = ror.lookup(author.raw_affiliation)
                        if result:
                            author.normalized_affiliation = result["name"]
                            author.country = result["country"]
                            author.country_code = result["country_code"]
                            author.org_type = result["type"]
                            author.confidence = max(author.confidence, result["confidence"])
                    except Exception:
                        continue
        
        return papers
    
    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику по запросам"""
        stats = {
            "total_requests": 0,
            "by_source": {}
        }
        
        for source, client in self._clients.items():
            count = client.get_request_count()
            stats["by_source"][source.value] = count
            stats["total_requests"] += count
        
        if self._ror:
            stats["ror_requests"] = self._ror.get_request_count()
        
        return stats
    
    def close(self):
        """Закрыть все клиенты"""
        for client in self._clients.values():
            if hasattr(client, "close"):
                client.close()
        
        if self._ror:
            self._ror.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Глобальный роутер для удобного использования
_router_instance: Optional[DataSourceRouter] = None


def get_data_router() -> DataSourceRouter:
    """Получить глобальный экземпляр роутера"""
    global _router_instance
    if _router_instance is None:
        _router_instance = DataSourceRouter()
    return _router_instance
