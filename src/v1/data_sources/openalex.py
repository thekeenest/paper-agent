"""
OpenAlex API Client

Клиент для работы с OpenAlex API.
https://docs.openalex.org/

OpenAlex - открытая замена Microsoft Academic Graph с ~92-93% точности
распознавания организаций через интеграцию с ROR.
"""

import os
import re
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


class OpenAlexClient(DataSourceBase):
    """
    Клиент для OpenAlex API.
    
    API Documentation: https://docs.openalex.org/
    
    Особенности:
    - Полностью бесплатный без ограничений
    - Рекомендуется указывать email (mailto) для приоритетного обслуживания
    - Организации привязаны к ROR идентификаторам
    - Высокая точность нормализации (~92-93%)
    
    Environment variable: OPENALEX_EMAIL (опционально)
    """
    
    BASE_URL = "https://api.openalex.org"
    
    def __init__(self, email: Optional[str] = None, polite_pool: bool = True):
        """
        Args:
            email: Email для polite pool (рекомендуется)
            polite_pool: Использовать polite pool (быстрее при указании email)
        """
        super().__init__(
            name="OpenAlex",
            base_url=self.BASE_URL
        )
        
        self.email = email or os.getenv("OPENALEX_EMAIL")
        self.polite_pool = polite_pool and bool(self.email)
        
        # HTTP клиент
        headers = {"Accept": "application/json"}
        
        self._client = httpx.Client(
            base_url=self.BASE_URL,
            headers=headers,
            timeout=30.0
        )
        
        self._last_request_time = 0
        # OpenAlex рекомендует 10 RPS для polite pool, но лучше быть консервативнее
        self._min_request_interval = 0.1 if self.polite_pool else 0.5
    
    def _rate_limit(self):
        """Соблюдение rate limit"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _build_params(self, base_params: Dict) -> Dict:
        """Добавить email для polite pool"""
        if self.email:
            base_params["mailto"] = self.email
        return base_params
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Выполнить запрос к API с retry логикой"""
        self._rate_limit()
        self._request_count += 1
        
        params = self._build_params(params or {})
        response = self._client.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    
    def search(self, params: SearchParams) -> List[PaperMetadata]:
        """
        Поиск публикаций в OpenAlex.
        
        Args:
            params: Параметры поиска
            
        Returns:
            Список найденных публикаций
        """
        # Формируем фильтр
        filters = []
        
        # Поиск по тексту - обрабатываем ArXiv-style синтаксис
        search_query = params.query
        
        # ArXiv категории -> OpenAlex concepts mapping
        concept_map = {
            "cs.AI": "artificial intelligence",
            "cs.LG": "machine learning",
            "cs.CV": "computer vision",
            "cs.CL": "natural language processing",
            "cs.NE": "neural network",
            "cs.RO": "robotics",
            "stat.ML": "machine learning",
            "cs.IR": "information retrieval",
            "cs.SE": "software engineering",
            "cs.DB": "database",
            "cs.DC": "distributed computing",
            "cs.CR": "cryptography",
            "cs.PL": "programming languages",
        }
        
        # Обрабатываем ArXiv-style запросы вида "cat:cs.AI" или "cs.AI"
        arxiv_cat_pattern = r'\bcat:([a-z]+\.[A-Z]+)\b'
        matches = re.findall(arxiv_cat_pattern, search_query)
        if matches:
            # Удаляем cat:X.XX из запроса и добавляем концепт
            for match in matches:
                search_query = re.sub(f'cat:{match}', '', search_query)
                if match in concept_map:
                    search_query = f"{search_query} {concept_map[match]}"
            search_query = search_query.strip()
            if not search_query:
                # Если после удаления cat: остался пустой запрос, используем концепт
                search_query = " ".join([concept_map.get(m, m) for m in matches])
        
        # Также проверяем простые ArXiv категории без "cat:" префикса
        simple_cat_pattern = r'\b([a-z]+\.[A-Z]+)\b'
        simple_matches = re.findall(simple_cat_pattern, search_query)
        if simple_matches:
            for match in simple_matches:
                if match in concept_map:
                    search_query = search_query.replace(match, concept_map[match])
        
        # Фильтр по датам
        if params.date_from:
            year_from = params.date_from[:4]
            filters.append(f"publication_year:>{int(year_from)-1}")
        if params.date_to:
            year_to = params.date_to[:4]
            filters.append(f"publication_year:<{int(year_to)+1}")
        
        # Фильтр по категориям (concepts в OpenAlex)
        if params.categories:
            # Преобразуем ArXiv категории в OpenAlex concepts
            concepts = [concept_map.get(cat, cat) for cat in params.categories]
            if concepts:
                search_query = f"{search_query} {' '.join(concepts)}"
        
        request_params = {
            "search": search_query,
            "per_page": min(params.max_results, 200),  # API лимит 200
            "select": "id,doi,title,display_name,publication_year,authorships,open_access,primary_location,concepts,type,cited_by_count"
        }
        
        if filters:
            request_params["filter"] = ",".join(filters)
        
        papers: List[PaperMetadata] = []
        page = 1
        
        while len(papers) < params.max_results:
            request_params["page"] = page
            
            try:
                data = self._make_request("/works", request_params)
            except Exception as e:
                print(f"[OpenAlex] Search error: {e}")
                break
            
            results = data.get("results", [])
            if not results:
                break
            
            for item in results:
                paper = self._convert_to_paper(item)
                if paper:
                    papers.append(paper)
                
                if len(papers) >= params.max_results:
                    break
            
            # Пагинация
            page += 1
            if page > data.get("meta", {}).get("count", 0) // 200 + 1:
                break
        
        return papers
    
    def get_paper(self, paper_id: str) -> Optional[PaperMetadata]:
        """
        Получить данные о публикации по OpenAlex ID или DOI.
        
        Args:
            paper_id: OpenAlex work ID (W...) или DOI
            
        Returns:
            Данные публикации или None
        """
        # Если это DOI, добавляем префикс
        if paper_id.startswith("10."):
            paper_id = f"https://doi.org/{paper_id}"
        
        endpoint = f"/works/{paper_id}"
        
        try:
            data = self._make_request(endpoint)
            return self._convert_to_paper(data)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            print(f"[OpenAlex] Error fetching paper {paper_id}: {e}")
            return None
    
    def get_author(self, author_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию об авторе.
        
        Args:
            author_id: OpenAlex author ID (A...)
            
        Returns:
            Данные автора с аффилиациями
        """
        endpoint = f"/authors/{author_id}"
        
        try:
            data = self._make_request(endpoint)
            
            # Извлекаем текущую аффилиацию
            affiliations = []
            last_known = data.get("last_known_institution")
            if last_known:
                affiliations.append({
                    "name": last_known.get("display_name", ""),
                    "ror": last_known.get("ror", ""),
                    "country": last_known.get("country_code", ""),
                    "type": last_known.get("type", "")
                })
            
            return {
                "id": data.get("id"),
                "name": data.get("display_name"),
                "affiliations": affiliations,
                "works_count": data.get("works_count", 0),
                "cited_by_count": data.get("cited_by_count", 0),
                "h_index": data.get("summary_stats", {}).get("h_index", 0),
                "orcid": data.get("orcid")
            }
        except Exception as e:
            print(f"[OpenAlex] Error fetching author {author_id}: {e}")
            return None
    
    def get_institution(self, institution_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить информацию об организации.
        
        Args:
            institution_id: OpenAlex institution ID (I...) или ROR ID
            
        Returns:
            Данные организации
        """
        # Если это ROR ID, преобразуем
        if institution_id.startswith("https://ror.org/"):
            pass  # Используем как есть
        
        endpoint = f"/institutions/{institution_id}"
        
        try:
            data = self._make_request(endpoint)
            
            return {
                "id": data.get("id"),
                "ror": data.get("ror"),
                "name": data.get("display_name"),
                "country": data.get("country_code"),
                "type": data.get("type"),
                "works_count": data.get("works_count", 0),
                "cited_by_count": data.get("cited_by_count", 0)
            }
        except Exception as e:
            print(f"[OpenAlex] Error fetching institution {institution_id}: {e}")
            return None
    
    def _convert_to_paper(self, data: Dict) -> Optional[PaperMetadata]:
        """Конвертация ответа API в PaperMetadata"""
        if not data:
            return None
        
        # Извлекаем ID (DOI или OpenAlex ID)
        doi_raw = data.get("doi")
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else ""
        openalex_id_raw = data.get("id")
        openalex_id = openalex_id_raw.replace("https://openalex.org/", "") if openalex_id_raw else ""
        paper_id = doi or openalex_id
        
        # Извлекаем авторов с аффилиациями
        authors = []
        for authorship in data.get("authorships", []):
            author_info = authorship.get("author", {})
            institutions = authorship.get("institutions", [])
            
            # Берём первую аффилиацию
            raw_affiliation = ""
            country = None
            country_code = None
            org_type = OrganizationType.UNKNOWN
            
            if institutions:
                inst = institutions[0]
                raw_affiliation = inst.get("display_name", "")
                country_code = inst.get("country_code")
                
                # Маппинг типа организации OpenAlex -> наш тип
                oa_type = inst.get("type", "").lower()
                type_map = {
                    "education": OrganizationType.UNIVERSITY,
                    "company": OrganizationType.COMPANY,
                    "government": OrganizationType.GOVERNMENT,
                    "nonprofit": OrganizationType.NONPROFIT,
                    "healthcare": OrganizationType.HOSPITAL,
                    "facility": OrganizationType.RESEARCH_INSTITUTE
                }
                org_type = type_map.get(oa_type, OrganizationType.UNKNOWN)
            
            author = AuthorAffiliation(
                name=author_info.get("display_name", "Unknown"),
                raw_affiliation=raw_affiliation,
                normalized_affiliation=raw_affiliation,  # OpenAlex уже нормализован
                country_code=country_code,
                org_type=org_type,
                confidence=0.92 if raw_affiliation else 0.5  # OpenAlex имеет ~92% точности
            )
            authors.append(author)
        
        # URL для PDF
        pdf_url = None
        open_access = data.get("open_access", {})
        if open_access.get("is_oa"):
            pdf_url = open_access.get("oa_url")
        
        if not pdf_url:
            primary_location = data.get("primary_location", {})
            if primary_location:
                pdf_url = primary_location.get("pdf_url")
        
        # Категории из concepts
        concepts = data.get("concepts") or []
        categories = [c.get("display_name", "") for c in concepts[:5] if c.get("display_name")]
        
        # Venue (журнал/конференция)
        venue = None
        primary_location = data.get("primary_location") or {}
        source = primary_location.get("source") or {}
        if source:
            venue = source.get("display_name")
        
        # Тип публикации
        publication_type = data.get("type")
        
        # Количество цитирований
        citation_count = data.get("cited_by_count")
        
        return PaperMetadata(
            arxiv_id=paper_id,
            title=data.get("title") or data.get("display_name", "Unknown Title"),
            abstract=None,  # OpenAlex не возвращает abstract в базовом запросе
            categories=categories,
            published_date=str(data.get("publication_year", "")),
            venue=venue,
            publication_type=publication_type,
            citation_count=citation_count,
            pdf_url=pdf_url,
            authors=authors,
            processing_status=ProcessingStatus.EXTRACTED if authors else ProcessingStatus.PENDING
        )
    
    def supports_affiliations(self) -> bool:
        """OpenAlex предоставляет аффилиации через ROR"""
        return True
    
    def supports_citations(self) -> bool:
        """OpenAlex предоставляет данные о цитированиях"""
        return True
    
    def close(self):
        """Закрыть HTTP клиент"""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
