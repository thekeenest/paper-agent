"""
ArXiv API Client

Клиент для работы с ArXiv API.
Оборачивает существующую логику в унифицированный интерфейс.
"""

import time
from typing import List, Optional
import arxiv

from .base import DataSourceBase, SearchParams
from ..models import PaperMetadata, ProcessingStatus, AuthorAffiliation


class ArxivClient(DataSourceBase):
    """
    Клиент для ArXiv API.
    
    Ограничения API:
    - Не более 1 запроса в 3 секунды
    - Максимум ~2000 результатов за запрос
    - Аффилиации НЕ предоставляются через API (только в PDF)
    """
    
    def __init__(self, delay_seconds: float = 3.0, num_retries: int = 3):
        """
        Args:
            delay_seconds: Задержка между запросами
            num_retries: Количество повторных попыток
        """
        super().__init__(
            name="ArXiv",
            base_url="http://export.arxiv.org/api/"
        )
        self.delay_seconds = delay_seconds
        self.num_retries = num_retries
        self._client = arxiv.Client(
            page_size=100,
            delay_seconds=delay_seconds,
            num_retries=num_retries
        )
        self._last_request_time = 0
    
    def _rate_limit(self):
        """Соблюдение rate limit"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self._last_request_time = time.time()
    
    def search(self, params: SearchParams) -> List[PaperMetadata]:
        """
        Поиск публикаций в ArXiv.
        
        Args:
            params: Параметры поиска
            
        Returns:
            Список найденных публикаций
        """
        self._rate_limit()
        self._request_count += 1
        
        # Формирование запроса
        query = params.query
        
        # Добавляем фильтр по датам если указан
        if params.date_from and params.date_to:
            query = f"{query} AND submittedDate:[{params.date_from} TO {params.date_to}]"
        
        # Добавляем категории если указаны
        if params.categories:
            cat_query = " OR ".join([f"cat:{cat}" for cat in params.categories])
            query = f"({query}) AND ({cat_query})"
        
        search = arxiv.Search(
            query=query,
            max_results=params.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers: List[PaperMetadata] = []
        
        for result in self._client.results(search):
            # Извлекаем авторов (ArXiv не предоставляет аффилиации через API)
            authors = [
                AuthorAffiliation(
                    name=author.name,
                    raw_affiliation="",  # ArXiv API не предоставляет аффилиации
                    confidence=0.5
                )
                for author in result.authors
            ]
            
            paper = PaperMetadata(
                arxiv_id=result.get_short_id(),
                title=result.title,
                abstract=result.summary[:500] if result.summary else None,
                categories=result.categories,
                published_date=str(result.published.date()) if result.published else None,
                pdf_url=result.pdf_url,
                authors=authors,
                processing_status=ProcessingStatus.PENDING
            )
            papers.append(paper)
        
        return papers
    
    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Получить данные о публикации по ArXiv ID.
        
        Args:
            paper_id: ArXiv ID (например, "2401.12345")
            
        Returns:
            Данные публикации или None
        """
        self._rate_limit()
        self._request_count += 1
        
        try:
            search = arxiv.Search(id_list=[paper_id])
            result = next(self._client.results(search))
            
            # Извлекаем авторов
            authors = [
                AuthorAffiliation(
                    name=author.name,
                    raw_affiliation="",  # ArXiv API не предоставляет аффилиации
                    confidence=0.5
                )
                for author in result.authors
            ]
            
            return PaperMetadata(
                arxiv_id=result.get_short_id(),
                title=result.title,
                abstract=result.summary[:500] if result.summary else None,
                categories=result.categories,
                published_date=str(result.published.date()) if result.published else None,
                pdf_url=result.pdf_url,
                authors=authors,
                processing_status=ProcessingStatus.PENDING
            )
        except StopIteration:
            return None
        except Exception as e:
            print(f"[ArxivClient] Error fetching paper {paper_id}: {e}")
            return None
    
    def supports_affiliations(self) -> bool:
        """ArXiv не предоставляет аффилиации через API"""
        return False
    
    def supports_citations(self) -> bool:
        """ArXiv не предоставляет данные о цитированиях"""
        return False
