"""
Pydantic модели для API endpoints.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class TaskStatusEnum(str, Enum):
    """Статус задачи анализа"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingStage(str, Enum):
    """Этап обработки"""
    IDLE = "idle"
    SEARCHING = "searching"
    DOWNLOADING = "downloading"
    PARSING = "parsing"
    EXTRACTING = "extracting"
    NORMALIZING = "normalizing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================
# REQUEST MODELS
# ============================================================

class AnalysisRequest(BaseModel):
    """Запрос на запуск анализа"""
    query: str = Field(
        default="cat:cs.AI",
        description="ArXiv search query",
        examples=["cat:cs.AI", "cat:cs.LG", "ti:transformer"]
    )
    max_papers: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Maximum number of papers to analyze"
    )
    data_source: str = Field(
        default="arxiv",
        description="Data source (arxiv, semantic_scholar, openalex)"
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Start date filter (YYYYMMDD)"
    )
    date_to: Optional[str] = Field(
        default=None,
        description="End date filter (YYYYMMDD)"
    )


class EvaluationRequest(BaseModel):
    """Запрос на оценку качества"""
    task_id: str = Field(..., description="Task ID to evaluate")
    gold_standard_path: Optional[str] = Field(
        default=None,
        description="Path to gold standard dataset"
    )


# ============================================================
# RESPONSE MODELS
# ============================================================

class AnalysisResponse(BaseModel):
    """Ответ на запуск анализа"""
    task_id: str
    status: TaskStatusEnum
    message: str
    created_at: datetime


class TaskStatus(BaseModel):
    """Полный статус задачи"""
    task_id: str
    status: TaskStatusEnum
    stage: ProcessingStage
    progress: float = Field(ge=0, le=100, description="Progress percentage")
    
    # Paper processing stats
    total_papers: int = 0
    processed_papers: int = 0
    failed_papers: int = 0
    current_paper_id: Optional[str] = None
    current_paper_title: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_seconds: float = 0
    estimated_remaining: Optional[float] = None
    
    # Errors
    errors: List[Dict[str, Any]] = []
    
    # Query info
    query: str = ""
    data_source: str = "arxiv"
    max_papers: int = 0


class TaskProgress(BaseModel):
    """Обновление прогресса для WebSocket"""
    task_id: str
    stage: ProcessingStage
    progress: float
    message: str
    current_paper: Optional[str] = None
    processed: int = 0
    total: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


class AuthorData(BaseModel):
    """Данные об авторе"""
    name: str
    raw_affiliation: str
    normalized_affiliation: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    org_type: str = "unknown"
    confidence: float = 1.0


class PaperData(BaseModel):
    """Данные о статье"""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    published_date: Optional[str] = None
    categories: List[str] = []
    authors: List[AuthorData] = []
    pdf_url: Optional[str] = None
    processing_status: str = "pending"


class OrganizationStats(BaseModel):
    """Статистика по организации"""
    name: str
    author_count: int
    country: Optional[str] = None
    org_type: str = "unknown"
    percentage: float = 0


class CountryStats(BaseModel):
    """Статистика по стране"""
    country: str
    country_code: Optional[str] = None
    author_count: int
    org_count: int
    percentage: float = 0


class OrgTypeStats(BaseModel):
    """Статистика по типам организаций"""
    org_type: str
    count: int
    percentage: float = 0


class AnalyticsData(BaseModel):
    """Аналитические данные для визуализаций"""
    # Summary stats
    total_papers: int = 0
    total_authors: int = 0
    unique_authors: int = 0
    unique_organizations: int = 0
    unique_countries: int = 0
    avg_authors_per_paper: float = 0
    avg_confidence: float = 0
    
    # Distributions
    top_organizations: List[OrganizationStats] = []
    country_distribution: List[CountryStats] = []
    org_type_distribution: List[OrgTypeStats] = []
    
    # Time series (if applicable)
    papers_by_date: Dict[str, int] = {}
    
    # Processing metrics
    processing_time_seconds: float = 0
    data_source: str = "arxiv"


class ExtractionMetricsResponse(BaseModel):
    """Метрики качества извлечения"""
    author_precision: float = 0
    author_recall: float = 0
    author_f1: float = 0
    affiliation_precision: float = 0
    affiliation_recall: float = 0
    affiliation_f1: float = 0
    org_normalization_accuracy: float = 0
    country_accuracy: float = 0
    hierarchical_accuracy: float = 0
    hallucination_rate: float = 0


class EvaluationResponse(BaseModel):
    """Полный отчёт об оценке"""
    task_id: str
    timestamp: datetime
    extraction_metrics: ExtractionMetricsResponse
    overall_score: float = 0
    gold_standard_papers: int = 0
    evaluated_papers: int = 0


class TaskResult(BaseModel):
    """Полный результат задачи"""
    task_id: str
    status: TaskStatusEnum
    analytics: Optional[AnalyticsData] = None
    papers: List[PaperData] = []
    evaluation: Optional[EvaluationResponse] = None
    output_files: Dict[str, str] = {}  # filename -> path
    errors: List[Dict[str, Any]] = []


class HealthResponse(BaseModel):
    """Ответ healthcheck"""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.now)
    services: Dict[str, bool] = {}


class ErrorResponse(BaseModel):
    """Ответ об ошибке"""
    detail: str
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
