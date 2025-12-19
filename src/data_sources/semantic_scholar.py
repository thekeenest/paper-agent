"""
Semantic Scholar API Client

Клиент для работы с Semantic Scholar Academic Graph API.
https://api.semanticscholar.org/api-docs/
"""

import os
import time
from typing import List, Optional, Dict, Any
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import DataSourceBase, SearchParams
from ..models import (
    PaperMetadata, 
    AuthorAffiliation, 
    ProcessingStatus,
    OrganizationType
)


class SemanticScholarClient(DataSourceBase):
    """
    Клиент для Semantic Scholar API.
    
    API Documentation: https://api.semanticscholar.org/api-docs/
    
    Ограничения:
    - Без API ключа: ~100 запросов/5 минут
    - С API ключом: ~1 запрос/секунду (1 RPS)
    - Аффилиации доступны частично (не для всех авторов)
    
    Environment variable: SEMANTIC_SCHOLAR_API_KEY
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    # Доступные поля для запроса
    # Документация: https://api.semanticscholar.org/api-docs/graph
    PAPER_FIELDS = [
        "paperId", "externalIds", "title", "abstract", "year",
        "authors", "authors.name", "authors.affiliations",
        "citationCount", "influentialCitationCount",
        "openAccessPdf", "fieldsOfStudy",
        "venue", "publicationVenue", "publicationTypes"  # Информация о месте публикации
    ]
    
    AUTHOR_FIELDS = [
        "authorId", "name", "affiliations", "paperCount",
        "citationCount", "hIndex"
    ]
    
    def __init__(self, api_key: Optional[str] = None, requests_per_second: float = 1.0):
        """
        Args:
            api_key: API ключ Semantic Scholar (или из env SEMANTIC_SCHOLAR_API_KEY)
            requests_per_second: Ограничение запросов в секунду
        """
        super().__init__(
            name="Semantic Scholar",
            base_url=self.BASE_URL
        )
        
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.requests_per_second = requests_per_second
        self._min_request_interval = 1.0 / requests_per_second
        self._last_request_time = 0
        
        # HTTP клиент
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=30.0
        )
    
    def _rate_limit(self):
        """Соблюдение rate limit"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Выполнить запрос к API с retry логикой"""
        self._rate_limit()
        self._request_count += 1
        
        response = self._client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def search(self, params: SearchParams) -> List[PaperMetadata]:
        """
        Поиск публикаций в Semantic Scholar.
        
        Args:
            params: Параметры поиска
            
        Returns:
            Список найденных публикаций
        """
        # Формируем параметры запроса
        request_params = {
            "query": params.query,
            "limit": min(params.max_results, 100),  # API лимит 100 за запрос
            "fields": ",".join(self.PAPER_FIELDS)
        }
        
        # Добавляем фильтр по годам если указаны даты
        if params.date_from:
            year_from = params.date_from[:4]
            request_params["year"] = f"{year_from}-"
        
        papers: List[PaperMetadata] = []
        offset = 0
        
        while len(papers) < params.max_results:
            request_params["offset"] = offset
            
            try:
                data = self._make_request("/paper/search", request_params)
            except Exception as e:
                print(f"[SemanticScholar] Search error: {e}")
                break
            
            if not data.get("data"):
                break
            
            for item in data["data"]:
                paper = self._convert_to_paper(item)
                if paper:
                    papers.append(paper)
                
                if len(papers) >= params.max_results:
                    break
            
            # Пагинация
            offset += len(data["data"])
            if offset >= data.get("total", 0):
                break
        
        return papers
    
    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Получить данные о публикации по Semantic Scholar ID или ArXiv ID.
        
        Args:
            paper_id: Semantic Scholar paperId или ArXiv:XXXX.XXXXX
            
        Returns:
            Данные публикации или None
        """
        # Если это ArXiv ID, добавляем префикс
        if not paper_id.startswith("ARXIV:") and "." in paper_id:
            paper_id = f"ARXIV:{paper_id}"
        
        endpoint = f"/paper/{paper_id}"
        params = {"fields": ",".join(self.PAPER_FIELDS)}
        
        try:
            data = self._make_request(endpoint, params)
            return self._convert_to_paper(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            print(f"[SemanticScholar] Error fetching paper {paper_id}: {e}")
            return None
    
    def get_author(self, author_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию об авторе.
        
        Args:
            author_id: Semantic Scholar authorId
            
        Returns:
            Данные автора с аффилиациями
        """
        endpoint = f"/author/{author_id}"
        params = {"fields": ",".join(self.AUTHOR_FIELDS)}
        
        try:
            data = self._make_request(endpoint, params)
            return {
                "id": data.get("authorId"),
                "name": data.get("name"),
                "affiliations": data.get("affiliations", []),
                "paper_count": data.get("paperCount", 0),
                "citation_count": data.get("citationCount", 0),
                "h_index": data.get("hIndex", 0)
            }
        except Exception as e:
            print(f"[SemanticScholar] Error fetching author {author_id}: {e}")
            return None
    
    def get_paper_authors_with_affiliations(self, paper_id: str) -> List[AuthorAffiliation]:
        """
        Получить авторов статьи с аффилиациями.
        
        Semantic Scholar предоставляет аффилиации в поле authors.affiliations.
        
        Args:
            paper_id: ID публикации
            
        Returns:
            Список авторов с аффилиациями
        """
        paper = self.get_paper(paper_id)
        if not paper:
            return []
        
        return paper.authors
    
    def _convert_to_paper(self, data: Dict) -> Optional[PaperMetadata]:
        """Конвертация ответа API в PaperMetadata"""
        if not data:
            return None
        
        # Извлекаем ArXiv ID если есть
        external_ids = data.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv", "")
        
        # Если нет ArXiv ID, используем Semantic Scholar ID
        paper_id = arxiv_id or data.get("paperId", "")
        
        # Извлекаем авторов с аффилиациями
        authors = []
        for author_data in data.get("authors") or []:
            affiliations = author_data.get("affiliations") or []
            raw_affiliation = affiliations[0] if affiliations else ""
            
            author = AuthorAffiliation(
                name=author_data.get("name", "Unknown"),
                raw_affiliation=raw_affiliation,
                confidence=0.9 if raw_affiliation else 0.5
            )
            authors.append(author)
        
        # URL для PDF
        pdf_url = None
        open_access = data.get("openAccessPdf")
        if open_access and isinstance(open_access, dict):
            pdf_url = open_access.get("url")
        elif arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # fieldsOfStudy может быть None, поэтому проверяем явно
        categories = data.get("fieldsOfStudy")
        if categories is None:
            categories = []
        
        # Извлекаем venue (место публикации: конференция/журнал)
        venue = data.get("venue") or ""
        publication_venue = data.get("publicationVenue")
        if publication_venue and isinstance(publication_venue, dict):
            # Используем полное название из publicationVenue если есть
            venue = publication_venue.get("name") or venue
        
        # Извлекаем тип публикации
        publication_types = data.get("publicationTypes") or []
        publication_type = publication_types[0] if publication_types else None
        
        # Количество цитирований
        citation_count = data.get("citationCount")
        
        return PaperMetadata(
            arxiv_id=paper_id,
            title=data.get("title", "Unknown Title"),
            abstract=data.get("abstract", "")[:500] if data.get("abstract") else None,
            categories=categories,
            published_date=str(data.get("year", "")),
            venue=venue if venue else None,
            publication_type=publication_type,
            citation_count=citation_count,
            pdf_url=pdf_url,
            authors=authors,
            processing_status=ProcessingStatus.PENDING if not authors else ProcessingStatus.EXTRACTED
        )
    
    def supports_affiliations(self) -> bool:
        """Semantic Scholar предоставляет частичные данные об аффилиациях"""
        return True
    
    def supports_citations(self) -> bool:
        """Semantic Scholar предоставляет данные о цитированиях"""
        return True
    
    def close(self):
        """Закрыть HTTP клиент"""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
