"""
Узлы (Nodes) агентного графа.

Каждая функция — отдельный этап обработки.
"""

import os
import time
from typing import Dict, Any, List
from pathlib import Path

import arxiv
import fitz  # PyMuPDF
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential

from .state import AgentState, NodeOutput
from .models import (
    PaperMetadata, 
    AuthorAffiliation, 
    ProcessingStatus,
    OrganizationType,
    LLMExtractionResponse
)
from .normalizer import get_normalizer
from .data_sources import DataSourceRouter, DataSourceType, SearchParams


# ============================================================
# КОНФИГУРАЦИЯ
# ============================================================

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./data/pdf_cache"))
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Создаём директории
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ПРОМПТЫ
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert bibliometric analyst specializing in extracting author information from scientific papers.

Your task is to extract ALL authors and their affiliations from the provided text (usually from the first pages of a paper).

Guidelines:
1. Extract the EXACT name as written (preserve original format)
2. Extract the affiliation as written in the paper
3. If an author has multiple affiliations, use the PRIMARY one (usually listed first)
4. If affiliation is not explicitly stated but email domain is visible, infer from domain:
   - @google.com → Google
   - @stanford.edu → Stanford University
   - @fb.com or @meta.com → Meta
5. Identify the country if mentioned or clearly inferable
6. Mark is_industry=True for companies (Google, Meta, Microsoft, etc.), False for universities

Common patterns to recognize:
- Superscript indices linking authors to affiliations (^1, ^2, *, †)
- Affiliations listed below author names
- Affiliations in footnotes
- Equal contribution markers

Return a list of ALL authors found, even if their affiliation is unclear."""

EXTRACTION_USER_PROMPT = """Extract all authors and their affiliations from this paper header:

---
{text}
---

Return the structured list of authors."""


# ============================================================
# NODE 1: ПОИСК СТАТЕЙ
# ============================================================

# Глобальный роутер для источников данных
_data_router = None

def _get_data_router() -> DataSourceRouter:
    """Получить или создать роутер источников данных"""
    global _data_router
    if _data_router is None:
        _data_router = DataSourceRouter()
    return _data_router


def search_papers(state: AgentState) -> NodeOutput:
    """
    Поиск статей по запросу.
    
    Поддерживает несколько источников данных:
    - arxiv (по умолчанию)
    - semantic_scholar
    - openalex
    """
    data_source = state.get("data_source", "arxiv")
    log_msg = f"[SearchAgent] Searching {data_source} for: {state['query']}"
    print(log_msg)
    
    try:
        router = _get_data_router()
        
        # Определяем тип источника
        source_type = DataSourceType(data_source)
        
        # Параметры поиска
        search_params = SearchParams(
            query=state["query"],
            max_results=state["max_papers"],
            date_from=state.get("date_from"),
            date_to=state.get("date_to"),
            categories=state.get("categories")
        )
        
        # Получаем клиент и выполняем поиск
        client = router._get_client(source_type)
        papers = client.search(search_params)
        
        log_msg = f"[SearchAgent] Found {len(papers)} papers from {data_source}"
        print(log_msg)
        
        return {
            "papers": papers,
            "current_index": 0,
            "processed_count": 0,
            "error_count": 0,
            "logs": [log_msg]
        }
        
    except Exception as e:
        error_msg = f"[SearchAgent] Error: {str(e)}"
        print(error_msg)
        return {
            "papers": [],
            "should_stop": True,
            "logs": [error_msg],
            "errors": [{"node": "search", "error": str(e)}]
        }


# ============================================================
# NODE 2: СКАЧИВАНИЕ PDF
# ============================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def _download_pdf_with_retry(arxiv_id: str, save_path: Path) -> bool:
    """Скачивание PDF с ArXiv с retry логикой"""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))
    paper.download_pdf(dirpath=str(save_path.parent), filename=save_path.name)
    return True


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def _download_pdf_from_url(url: str, save_path: Path) -> bool:
    """Скачивание PDF по прямой ссылке с retry логикой"""
    import httpx
    
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ConferencePaperAgent/1.0; mailto:research@example.com)"
    }
    
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()
        
        # Проверяем что это PDF
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not response.content[:4] == b"%PDF":
            raise ValueError(f"Not a PDF file: {content_type}")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(response.content)
    
    return True


def _is_arxiv_id(paper_id: str) -> bool:
    """Проверка, является ли ID ArXiv идентификатором"""
    import re
    # ArXiv ID форматы: 2401.12345, hep-th/9901001, cs.AI/0001001
    arxiv_patterns = [
        r'^\d{4}\.\d{4,5}(v\d+)?$',  # Новый формат: 2401.12345
        r'^[a-z-]+/\d{7}(v\d+)?$',    # Старый формат: hep-th/9901001
    ]
    return any(re.match(pattern, paper_id) for pattern in arxiv_patterns)


def download_paper(state: AgentState) -> NodeOutput:
    """
    Скачивание PDF текущей статьи.
    
    Поддерживает несколько методов скачивания:
    1. Если есть pdf_url - скачивает напрямую
    2. Если ArXiv ID - использует ArXiv API
    3. Иначе - пропускает (для OpenAlex/Semantic Scholar без PDF)
    
    Реализует кэширование: если PDF уже скачан, пропускает загрузку.
    """
    idx = state["current_index"]
    papers = state["papers"]
    
    if idx >= len(papers):
        return {}
    
    paper = papers[idx]
    log_msg = f"[FetcherAgent] Processing {idx + 1}/{len(papers)}: {paper.arxiv_id}"
    print(log_msg)
    
    # Путь для сохранения
    # Заменяем / на _ и . для безопасности в имени файла
    safe_id = paper.arxiv_id.replace("/", "_").replace(".", "_")
    cache_path = CACHE_DIR / f"{safe_id}.pdf"
    
    # Проверка кэша
    if cache_path.exists():
        paper.pdf_path = str(cache_path)
        paper.processing_status = ProcessingStatus.DOWNLOADED
        log_msg = f"[FetcherAgent] Using cached PDF: {cache_path}"
        print(log_msg)
        return {"papers": papers, "logs": [log_msg]}
    
    # Скачивание - выбираем метод в зависимости от источника
    try:
        paper.processing_status = ProcessingStatus.DOWNLOADING
        downloaded = False
        
        # Метод 1: Прямая ссылка на PDF (приоритетный для OpenAlex/Semantic Scholar)
        if paper.pdf_url:
            try:
                log_msg = f"[FetcherAgent] Downloading from URL: {paper.pdf_url[:60]}..."
                print(log_msg)
                _download_pdf_from_url(paper.pdf_url, cache_path)
                downloaded = True
            except Exception as url_error:
                log_msg = f"[FetcherAgent] URL download failed: {str(url_error)[:100]}"
                print(log_msg)
        
        # Метод 2: ArXiv API (если это ArXiv ID)
        if not downloaded and _is_arxiv_id(paper.arxiv_id):
            try:
                log_msg = f"[FetcherAgent] Downloading from ArXiv: {paper.arxiv_id}"
                print(log_msg)
                _download_pdf_with_retry(paper.arxiv_id, cache_path)
                downloaded = True
            except Exception as arxiv_error:
                log_msg = f"[FetcherAgent] ArXiv download failed: {str(arxiv_error)[:100]}"
                print(log_msg)
        
        if downloaded:
            paper.pdf_path = str(cache_path)
            paper.processing_status = ProcessingStatus.DOWNLOADED
            log_msg = f"[FetcherAgent] Downloaded: {cache_path}"
            print(log_msg)
            return {"papers": papers, "logs": [log_msg]}
        else:
            # Нет способа скачать PDF - помечаем статью, но продолжаем
            # Используем аффилиации из метаданных API (OpenAlex уже предоставляет их)
            if paper.authors and any(a.raw_affiliation for a in paper.authors):
                paper.processing_status = ProcessingStatus.EXTRACTED
                log_msg = f"[FetcherAgent] No PDF available, using API affiliations for: {paper.arxiv_id}"
                print(log_msg)
                return {"papers": papers, "logs": [log_msg]}
            else:
                raise ValueError("No PDF URL and not an ArXiv paper")
        
    except Exception as e:
        error_msg = f"[FetcherAgent] Download failed for {paper.arxiv_id}: {str(e)}"
        print(error_msg)
        paper.mark_failed(str(e))
        return {
            "papers": papers,
            "error_count": state["error_count"] + 1,
            "logs": [error_msg],
            "errors": [{"node": "download", "paper_id": paper.arxiv_id, "error": str(e)}]
        }


# ============================================================
# NODE 3: ПАРСИНГ PDF
# ============================================================

def parse_pdf(state: AgentState) -> NodeOutput:
    """
    Извлечение текста из PDF.
    
    Использует PyMuPDF для быстрого парсинга первых страниц.
    """
    idx = state["current_index"]
    papers = state["papers"]
    paper = papers[idx]
    
    # Пропускаем если уже failed
    if paper.is_failed():
        return {"papers": papers}
    
    log_msg = f"[ParserAgent] Parsing PDF: {paper.arxiv_id}"
    print(log_msg)
    
    try:
        paper.processing_status = ProcessingStatus.PARSING
        
        doc = fitz.open(paper.pdf_path)
        
        # Извлекаем текст первых 2 страниц (там обычно аффилиации)
        text_parts = []
        for page_num in range(min(2, len(doc))):
            page = doc[page_num]
            text_parts.append(page.get_text("text"))
        
        doc.close()
        
        # Объединяем и ограничиваем длину
        full_text = "\n\n".join(text_parts)
        paper.raw_text = full_text[:8000]  # Лимит для контекста LLM
        paper.processing_status = ProcessingStatus.PARSED
        
        log_msg = f"[ParserAgent] Extracted {len(paper.raw_text)} chars from {paper.arxiv_id}"
        print(log_msg)
        
        return {"papers": papers, "logs": [log_msg]}
        
    except Exception as e:
        error_msg = f"[ParserAgent] Parse failed for {paper.arxiv_id}: {str(e)}"
        print(error_msg)
        paper.mark_failed(str(e))
        return {
            "papers": papers,
            "error_count": state["error_count"] + 1,
            "logs": [error_msg],
            "errors": [{"node": "parse", "paper_id": paper.arxiv_id, "error": str(e)}]
        }


# ============================================================
# NODE 4: ИЗВЛЕЧЕНИЕ АФФИЛИАЦИЙ (LLM)
# ============================================================

def extract_affiliations(state: AgentState) -> NodeOutput:
    """
    LLM-based извлечение авторов и аффилиаций.
    
    Использует structured output для получения типизированных данных.
    """
    idx = state["current_index"]
    papers = state["papers"]
    paper = papers[idx]
    
    # Пропускаем если уже failed
    if paper.is_failed():
        return {
            "papers": papers,
            "current_index": idx + 1,
            "retry_count": 0
        }
    
    log_msg = f"[ExtractorAgent] Extracting from: {paper.title[:50]}..."
    print(log_msg)
    
    start_time = time.time()
    
    try:
        paper.processing_status = ProcessingStatus.EXTRACTING
        
        # Инициализация LLM с structured output
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(LLMExtractionResponse)
        
        # Формирование промпта
        prompt = ChatPromptTemplate.from_messages([
            ("system", EXTRACTION_SYSTEM_PROMPT),
            ("user", EXTRACTION_USER_PROMPT)
        ])
        
        chain = prompt | structured_llm
        
        # Вызов LLM
        result: LLMExtractionResponse = chain.invoke({"text": paper.raw_text})
        
        # Конвертация результата в AuthorAffiliation
        authors = []
        for llm_author in result.authors:
            author = AuthorAffiliation(
                name=llm_author.name,
                raw_affiliation=llm_author.affiliation,
                country=llm_author.country,
                org_type=OrganizationType.COMPANY if llm_author.is_industry else OrganizationType.UNIVERSITY,
                email=llm_author.email,
                confidence=0.85
            )
            authors.append(author)
        
        paper.authors = authors
        paper.processing_status = ProcessingStatus.EXTRACTED
        paper.processing_time_ms = int((time.time() - start_time) * 1000)
        
        log_msg = f"[ExtractorAgent] Extracted {len(authors)} authors from {paper.arxiv_id}"
        print(log_msg)
        
        return {
            "papers": papers,
            "current_index": idx + 1,
            "processed_count": state["processed_count"] + 1,
            "retry_count": 0,
            "logs": [log_msg]
        }
        
    except Exception as e:
        error_msg = f"[ExtractorAgent] Extraction failed for {paper.arxiv_id}: {str(e)}"
        print(error_msg)
        
        # Retry logic
        if state["retry_count"] < state["max_retries"]:
            return {
                "papers": papers,
                "retry_count": state["retry_count"] + 1,
                "logs": [f"{error_msg} - retrying ({state['retry_count'] + 1}/{state['max_retries']})"]
            }
        
        paper.mark_failed(str(e))
        return {
            "papers": papers,
            "current_index": idx + 1,
            "error_count": state["error_count"] + 1,
            "retry_count": 0,
            "logs": [error_msg],
            "errors": [{"node": "extract", "paper_id": paper.arxiv_id, "error": str(e)}]
        }


# ============================================================
# NODE 5: НОРМАЛИЗАЦИЯ
# ============================================================

def normalize_affiliations(state: AgentState) -> NodeOutput:
    """
    Нормализация названий организаций.
    
    Использует Knowledge Base + fuzzy matching + LLM fallback.
    """
    idx = state["current_index"] - 1  # Индекс уже инкрементирован
    papers = state["papers"]
    
    if idx < 0 or idx >= len(papers):
        return {"papers": papers}
    
    paper = papers[idx]
    
    # Пропускаем если статья failed или нет авторов
    if paper.is_failed() or not paper.authors:
        return {"papers": papers}
    
    log_msg = f"[NormalizerAgent] Normalizing affiliations for {paper.arxiv_id}"
    print(log_msg)
    
    normalizer = get_normalizer()
    
    for author in paper.authors:
        if author.raw_affiliation:
            result = normalizer.normalize(author.raw_affiliation)
            author.normalized_affiliation = result.normalized
            author.country = result.country
            author.country_code = result.country_code
            author.org_type = result.org_type
            # Корректируем confidence с учётом нормализации
            author.confidence = min(author.confidence, result.confidence)
    
    paper.processing_status = ProcessingStatus.COMPLETED
    
    log_msg = f"[NormalizerAgent] Normalized {len(paper.authors)} affiliations"
    print(log_msg)
    
    return {"papers": papers, "logs": [log_msg]}


# ============================================================
# NODE 6: АГРЕГАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================

def aggregate_results(state: AgentState) -> NodeOutput:
    """
    Агрегация и сохранение результатов.
    
    Формирует итоговый датасет и аналитический отчёт.
    """
    import json
    import pandas as pd
    from datetime import datetime
    from .models import AnalyticsReport
    
    papers = state["papers"]
    
    log_msg = "[AggregateAgent] Building final report..."
    print(log_msg)
    
    # Подсчёт статистики
    total_papers = len(papers)
    successful = sum(1 for p in papers if p.is_completed())
    failed = sum(1 for p in papers if p.is_failed())
    total_authors = sum(len(p.authors) for p in papers)
    
    # Собираем все аффилиации для анализа
    all_affiliations = []
    for paper in papers:
        for author in paper.authors:
            all_affiliations.append({
                "paper_id": paper.arxiv_id,
                "paper_title": paper.title,
                "author_name": author.name,
                "raw_affiliation": author.raw_affiliation,
                "normalized_affiliation": author.normalized_affiliation,
                "country": author.country,
                "country_code": author.country_code,
                "org_type": author.org_type.value if author.org_type else "unknown",
                "confidence": author.confidence
            })
    
    df = pd.DataFrame(all_affiliations)
    
    # Топ организаций
    top_orgs = []
    if not df.empty and "normalized_affiliation" in df.columns:
        org_counts = df["normalized_affiliation"].value_counts().head(20)
        for org, count in org_counts.items():
            top_orgs.append({"organization": org, "count": int(count)})
    
    # Топ стран
    top_countries = []
    if not df.empty and "country" in df.columns:
        country_counts = df["country"].value_counts().head(15)
        for country, count in country_counts.items():
            top_countries.append({"country": country, "count": int(count)})
    
    # Распределение по типам
    org_type_dist = {}
    if not df.empty and "org_type" in df.columns:
        for otype, count in df["org_type"].value_counts().items():
            org_type_dist[otype] = int(count)
    
    # Время обработки
    total_time = sum(p.processing_time_ms or 0 for p in papers)
    
    # Формирование отчёта
    report = AnalyticsReport(
        total_papers=total_papers,
        total_authors=total_authors,
        successful_extractions=successful,
        failed_extractions=failed,
        top_organizations=top_orgs,
        top_countries=top_countries,
        org_type_distribution=org_type_dist,
        processing_time_total_ms=total_time,
        average_authors_per_paper=total_authors / total_papers if total_papers > 0 else 0,
        generated_at=datetime.now().isoformat()
    )
    
    # Сохранение результатов
    output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV с аффилиациями
    csv_path = output_dir / f"affiliations_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    
    # JSON отчёт
    report_path = output_dir / f"report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
    
    log_msg = f"[AggregateAgent] Results saved to {output_dir}"
    print(log_msg)
    print(f"\n{'='*50}")
    print(f"SUMMARY:")
    print(f"  Papers processed: {successful}/{total_papers}")
    print(f"  Total authors: {total_authors}")
    print(f"  Errors: {failed}")
    print(f"  Output: {csv_path}")
    print(f"{'='*50}\n")
    
    return {
        "final_report": report,
        "output_path": str(output_dir),
        "logs": [log_msg]
    }


# ============================================================
# УСЛОВНЫЕ ПЕРЕХОДЫ
# ============================================================

def should_continue_processing(state: AgentState) -> str:
    """
    Определяет, продолжать ли обработку следующей статьи.
    
    Returns:
        "download" - продолжить со следующей статьёй
        "aggregate" - завершить и агрегировать результаты
        "extract" - повторить извлечение (retry)
    """
    # Проверка на retry
    if state["retry_count"] > 0 and state["retry_count"] <= state["max_retries"]:
        return "extract"
    
    # Проверка флага остановки
    if state["should_stop"]:
        return "aggregate"
    
    # Есть ещё статьи для обработки?
    if state["current_index"] < len(state["papers"]):
        return "download"
    
    return "aggregate"


def should_retry_extraction(state: AgentState) -> str:
    """
    Определяет, нужен ли retry для extraction.
    """
    if state["retry_count"] > 0 and state["retry_count"] <= state["max_retries"]:
        return "extract"
    return "normalize"
