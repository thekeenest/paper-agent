"""
Модели данных для системы анализа публикаций.

Использует Pydantic для валидации и сериализации данных.
"""

from typing import List, Optional, Dict, Any
from datetime import date
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class OrganizationType(str, Enum):
    """Тип организации"""
    UNIVERSITY = "university"
    COMPANY = "company"
    RESEARCH_INSTITUTE = "research_institute"
    GOVERNMENT = "government"
    HOSPITAL = "hospital"
    NONPROFIT = "nonprofit"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Статус обработки статьи"""
    PENDING = "pending"
    SEARCHING = "searching"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    PARSING = "parsing"
    PARSED = "parsed"
    EXTRACTING = "extracting"
    EXTRACTED = "extracted"
    NORMALIZING = "normalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class AuthorAffiliation(BaseModel):
    """
    Информация об авторе и его аффилиации.
    
    Attributes:
        name: Полное имя автора
        raw_affiliation: Аффилиация как указана в статье
        normalized_affiliation: Нормализованное название организации
        country: Страна организации (ISO 3166-1 alpha-2 или полное название)
        org_type: Тип организации
        email: Email автора (если доступен)
        confidence: Уверенность в корректности извлечения (0.0-1.0)
    """
    name: str = Field(..., description="Полное имя автора")
    raw_affiliation: str = Field(
        default="",
        description="Аффилиация как указана в статье"
    )
    normalized_affiliation: Optional[str] = Field(
        default=None,
        description="Нормализованное название организации"
    )
    country: Optional[str] = Field(
        default=None,
        description="Страна организации"
    )
    country_code: Optional[str] = Field(
        default=None,
        description="Код страны ISO 3166-1 alpha-2"
    )
    org_type: OrganizationType = Field(
        default=OrganizationType.UNKNOWN,
        description="Тип организации"
    )
    email: Optional[str] = Field(
        default=None,
        description="Email автора"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Уверенность в извлечении"
    )
    
    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class PaperMetadata(BaseModel):
    """
    Метаданные одной научной публикации.
    
    Attributes:
        arxiv_id: Идентификатор ArXiv (например, "2401.12345")
        title: Название статьи
        abstract: Аннотация
        categories: Категории ArXiv
        published_date: Дата публикации
        venue: Место публикации (конференция/журнал)
        publication_type: Тип публикации (Conference, JournalArticle, Review и т.д.)
        citation_count: Количество цитирований
        pdf_url: URL для скачивания PDF
        pdf_path: Локальный путь к PDF
        raw_text: Извлечённый текст (первые страницы)
        authors: Список авторов с аффилиациями
        processing_status: Текущий статус обработки
        error_message: Сообщение об ошибке (если есть)
        processing_time_ms: Время обработки в миллисекундах
    """
    arxiv_id: str = Field(..., description="Идентификатор ArXiv")
    title: str = Field(..., description="Название статьи")
    abstract: Optional[str] = Field(default=None, description="Аннотация")
    categories: List[str] = Field(default_factory=list, description="Категории ArXiv")
    published_date: Optional[str] = Field(default=None, description="Дата публикации")
    venue: Optional[str] = Field(default=None, description="Место публикации (конференция/журнал)")
    publication_type: Optional[str] = Field(default=None, description="Тип публикации")
    citation_count: Optional[int] = Field(default=None, description="Количество цитирований")
    pdf_url: Optional[str] = Field(default=None, description="URL PDF")
    pdf_path: Optional[str] = Field(default=None, description="Локальный путь к PDF")
    raw_text: Optional[str] = Field(default=None, description="Извлечённый текст")
    authors: List[AuthorAffiliation] = Field(
        default_factory=list,
        description="Список авторов с аффилиациями"
    )
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Статус обработки"
    )
    error_message: Optional[str] = Field(default=None, description="Сообщение об ошибке")
    processing_time_ms: Optional[int] = Field(default=None, description="Время обработки")
    
    def mark_failed(self, error: str) -> None:
        """Пометить статью как неудачно обработанную"""
        self.processing_status = ProcessingStatus.FAILED
        self.error_message = error
    
    def is_completed(self) -> bool:
        """Проверить, завершена ли обработка"""
        return self.processing_status == ProcessingStatus.COMPLETED
    
    def is_failed(self) -> bool:
        """Проверить, произошла ли ошибка"""
        return self.processing_status == ProcessingStatus.FAILED


class ExtractionResult(BaseModel):
    """
    Результат извлечения аффилиаций из статьи.
    
    Используется как structured output для LLM.
    """
    authors: List[AuthorAffiliation] = Field(
        ...,
        description="Список авторов с их аффилиациями"
    )
    extraction_notes: Optional[str] = Field(
        default=None,
        description="Примечания к извлечению (сложные случаи)"
    )


class NormalizationResult(BaseModel):
    """Результат нормализации названия организации"""
    original: str
    normalized: str
    country: str
    country_code: str
    org_type: OrganizationType
    confidence: float
    source: str = Field(
        default="llm",
        description="Источник нормализации: kb, fuzzy, llm"
    )


class AnalyticsReport(BaseModel):
    """Аналитический отчёт по обработанным данным"""
    total_papers: int
    total_authors: int
    successful_extractions: int
    failed_extractions: int
    
    top_organizations: List[Dict[str, Any]]
    top_countries: List[Dict[str, Any]]
    org_type_distribution: Dict[str, int]
    
    processing_time_total_ms: int
    average_authors_per_paper: float
    
    generated_at: str


# Модели для LLM structured output
class LLMAuthorExtraction(BaseModel):
    """Модель для извлечения одного автора через LLM"""
    name: str = Field(..., description="Full name of the author")
    affiliation: str = Field(..., description="Organization/university name as written")
    country: Optional[str] = Field(None, description="Country if mentioned or inferable")
    is_industry: bool = Field(False, description="True if company, False if academic")
    email: Optional[str] = Field(None, description="Email if visible")


class LLMExtractionResponse(BaseModel):
    """Ответ LLM при извлечении авторов"""
    authors: List[LLMAuthorExtraction]
    notes: Optional[str] = Field(
        None,
        description="Any notes about extraction difficulties"
    )
