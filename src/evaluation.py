"""
Модуль оценки качества извлечения и работы агента.

Трёхуровневая система метрик:
1. Extraction Quality - качество извлечения данных (NER/NEN)
2. Agent Performance - надёжность работы агента
3. Engineering Metrics - эффективность системы

Методология основана на:
- Классические метрики NER: Precision, Recall, F1-Score
- Fuzzy matching для обработки вариаций написания
- Иерархическая оценка нормализации (Organization → Country)
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

from rapidfuzz import fuzz
import pandas as pd
import numpy as np

from .models import PaperMetadata, AuthorAffiliation, OrganizationType


# ============================================================
# DATA CLASSES FOR GOLD STANDARD
# ============================================================

@dataclass
class GoldAuthor:
    """Эталонные данные об авторе"""
    name: str
    raw_affiliation: str
    normalized_affiliation: str
    country: str
    country_code: str
    org_type: str  # university, company, research_institute, etc.


@dataclass
class GoldPaper:
    """Эталонные данные о статье"""
    paper_id: str
    title: str
    authors: List[GoldAuthor]
    source: str = "manual"  # manual, grobid, crossref
    annotator: str = ""
    annotation_date: str = ""
    notes: str = ""


@dataclass
class ExtractionMetrics:
    """Метрики качества извлечения"""
    # Author extraction
    author_precision: float = 0.0
    author_recall: float = 0.0
    author_f1: float = 0.0
    
    # Affiliation extraction
    affiliation_precision: float = 0.0
    affiliation_recall: float = 0.0
    affiliation_f1: float = 0.0
    
    # Normalization accuracy
    org_normalization_accuracy: float = 0.0
    country_accuracy: float = 0.0
    org_type_accuracy: float = 0.0
    
    # Hierarchical accuracy (org + country both correct)
    hierarchical_accuracy: float = 0.0
    
    # Hallucination metrics
    author_hallucination_rate: float = 0.0
    affiliation_hallucination_rate: float = 0.0
    
    # Counts for transparency
    total_gold_authors: int = 0
    total_pred_authors: int = 0
    matched_authors: int = 0


@dataclass
class AgentMetrics:
    """Метрики работы агента"""
    # Tool usage
    tool_success_rate: float = 0.0
    arxiv_api_success: float = 0.0
    pdf_download_success: float = 0.0
    pdf_parse_success: float = 0.0
    llm_extraction_success: float = 0.0
    
    # End-to-end
    e2e_success_rate: float = 0.0
    papers_fully_processed: int = 0
    papers_partial: int = 0
    papers_failed: int = 0
    
    # Error breakdown
    errors_by_stage: Dict[str, int] = field(default_factory=dict)


@dataclass
class EngineeringMetrics:
    """Инженерные метрики эффективности"""
    # Timing
    total_time_seconds: float = 0.0
    avg_time_per_paper: float = 0.0
    time_by_stage: Dict[str, float] = field(default_factory=dict)
    
    # Cost (OpenAI tokens)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    
    # API calls
    total_api_calls: int = 0
    api_calls_by_source: Dict[str, int] = field(default_factory=dict)
    
    # Cache efficiency
    cache_hit_rate: float = 0.0
    cached_pdfs: int = 0
    downloaded_pdfs: int = 0


@dataclass
class EvaluationReport:
    """Полный отчёт об оценке качества"""
    timestamp: str = ""
    extraction: ExtractionMetrics = field(default_factory=ExtractionMetrics)
    agent: AgentMetrics = field(default_factory=AgentMetrics)
    engineering: EngineeringMetrics = field(default_factory=EngineeringMetrics)
    
    # Summary
    overall_quality_score: float = 0.0  # Weighted composite score
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ============================================================
# GOLD STANDARD DATASET MANAGEMENT
# ============================================================

class GoldStandardDataset:
    """
    Управление эталонным датасетом для оценки качества.
    
    Формат хранения: JSON файл с массивом GoldPaper.
    """
    
    def __init__(self, path: str = "./data/gold_standard.json"):
        self.path = Path(path)
        self.papers: Dict[str, GoldPaper] = {}
        
        if self.path.exists():
            self.load()
    
    def load(self) -> None:
        """Загрузка датасета из файла"""
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for paper_data in data.get("papers", []):
            authors = [
                GoldAuthor(**author) 
                for author in paper_data.get("authors", [])
            ]
            paper = GoldPaper(
                paper_id=paper_data["paper_id"],
                title=paper_data.get("title", ""),
                authors=authors,
                source=paper_data.get("source", "manual"),
                annotator=paper_data.get("annotator", ""),
                annotation_date=paper_data.get("annotation_date", ""),
                notes=paper_data.get("notes", "")
            )
            self.papers[paper.paper_id] = paper
    
    def save(self) -> None:
        """Сохранение датасета в файл"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "total_papers": len(self.papers),
            "total_authors": sum(len(p.authors) for p in self.papers.values()),
            "papers": [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "authors": [asdict(a) for a in p.authors],
                    "source": p.source,
                    "annotator": p.annotator,
                    "annotation_date": p.annotation_date,
                    "notes": p.notes
                }
                for p in self.papers.values()
            ]
        }
        
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_paper(self, paper: GoldPaper) -> None:
        """Добавить статью в датасет"""
        self.papers[paper.paper_id] = paper
    
    def get_paper(self, paper_id: str) -> Optional[GoldPaper]:
        """Получить статью по ID"""
        return self.papers.get(paper_id)
    
    def get_all_papers(self) -> List[GoldPaper]:
        """Получить все статьи"""
        return list(self.papers.values())
    
    def stats(self) -> Dict[str, Any]:
        """Статистика датасета"""
        all_authors = [a for p in self.papers.values() for a in p.authors]
        countries = set(a.country for a in all_authors if a.country)
        orgs = set(a.normalized_affiliation for a in all_authors if a.normalized_affiliation)
        
        return {
            "total_papers": len(self.papers),
            "total_authors": len(all_authors),
            "unique_organizations": len(orgs),
            "unique_countries": len(countries),
            "avg_authors_per_paper": len(all_authors) / len(self.papers) if self.papers else 0,
            "sources": dict(pd.Series([p.source for p in self.papers.values()]).value_counts())
        }
    
    def create_template(self, paper_id: str, title: str) -> Dict:
        """Создать шаблон для ручной разметки"""
        return {
            "paper_id": paper_id,
            "title": title,
            "authors": [
                {
                    "name": "FILL: Full Author Name",
                    "raw_affiliation": "FILL: Affiliation as written in paper",
                    "normalized_affiliation": "FILL: Canonical organization name",
                    "country": "FILL: Country name",
                    "country_code": "FILL: ISO 3166-1 alpha-2 code (e.g., US, CN, GB)",
                    "org_type": "FILL: university|company|research_institute|government|hospital|nonprofit"
                }
            ],
            "source": "manual",
            "annotator": "FILL: Your name",
            "annotation_date": datetime.now().strftime("%Y-%m-%d"),
            "notes": ""
        }


# ============================================================
# EVALUATION ENGINE
# ============================================================

class EvaluationEngine:
    """
    Движок оценки качества системы.
    
    Реализует трёхуровневую систему метрик:
    1. Extraction Quality (NER/NEN)
    2. Agent Performance
    3. Engineering Metrics
    """
    
    def __init__(
        self,
        gold_standard_path: str = "./data/gold_standard.json",
        fuzzy_threshold: int = 85
    ):
        """
        Args:
            gold_standard_path: Путь к эталонному датасету
            fuzzy_threshold: Порог fuzzy matching (0-100)
        """
        self.gold_dataset = GoldStandardDataset(gold_standard_path)
        self.fuzzy_threshold = fuzzy_threshold
    
    def _fuzzy_match(self, str1: str, str2: str) -> bool:
        """
        Нечёткое сравнение строк.
        
        Использует token_sort_ratio для устойчивости к:
        - Разному порядку слов ("John Smith" vs "Smith, John")
        - Сокращениям ("Univ." vs "University")
        - Регистру
        """
        if not str1 or not str2:
            return False
        
        score = fuzz.token_sort_ratio(str1.lower().strip(), str2.lower().strip())
        return score >= self.fuzzy_threshold
    
    def _exact_match_normalized(self, str1: str, str2: str) -> bool:
        """Точное сравнение после нормализации"""
        if not str1 or not str2:
            return False
        return str1.lower().strip() == str2.lower().strip()
    
    # ============================================================
    # LEVEL 1: EXTRACTION QUALITY METRICS
    # ============================================================
    
    def evaluate_extraction(
        self,
        predictions: List[PaperMetadata],
        verbose: bool = False
    ) -> ExtractionMetrics:
        """
        Оценка качества извлечения данных.
        
        Метрики:
        - Author Precision/Recall/F1 (fuzzy matching)
        - Affiliation Precision/Recall/F1
        - Normalization Accuracy (Org, Country, Type)
        - Hierarchical Accuracy (Org + Country)
        - Hallucination Rate
        
        Args:
            predictions: Список результатов работы агента
            verbose: Выводить детали по каждой статье
            
        Returns:
            ExtractionMetrics с рассчитанными значениями
        """
        metrics = ExtractionMetrics()
        
        # Агрегированные счётчики
        author_tp = 0  # True Positives
        author_fp = 0  # False Positives (hallucinations)
        author_fn = 0  # False Negatives (missed)
        
        aff_tp = 0
        aff_fp = 0
        aff_fn = 0
        
        org_correct = 0
        country_correct = 0
        org_type_correct = 0
        hierarchical_correct = 0
        total_matched = 0
        
        for pred_paper in predictions:
            gold_paper = self.gold_dataset.get_paper(pred_paper.arxiv_id)
            
            if gold_paper is None:
                if verbose:
                    print(f"[WARN] No gold standard for {pred_paper.arxiv_id}")
                continue
            
            # Извлечение авторов
            pred_authors = {a.name for a in pred_paper.authors}
            gold_authors = {a.name for a in gold_paper.authors}
            
            # Fuzzy matching авторов
            matched_gold = set()
            matched_pred = set()
            
            for pred_name in pred_authors:
                for gold_name in gold_authors:
                    if gold_name in matched_gold:
                        continue
                    if self._fuzzy_match(pred_name, gold_name):
                        author_tp += 1
                        matched_gold.add(gold_name)
                        matched_pred.add(pred_name)
                        break
            
            author_fp += len(pred_authors - matched_pred)
            author_fn += len(gold_authors - matched_gold)
            
            # Оценка аффилиаций для совпавших авторов
            gold_by_name = {a.name.lower(): a for a in gold_paper.authors}
            
            for pred_author in pred_paper.authors:
                # Ищем соответствующего gold автора
                gold_author = None
                for gold_name, gold_a in gold_by_name.items():
                    if self._fuzzy_match(pred_author.name, gold_name):
                        gold_author = gold_a
                        break
                
                if gold_author is None:
                    continue
                
                total_matched += 1
                
                # Аффилиация извлечена?
                pred_has_aff = bool(pred_author.raw_affiliation)
                gold_has_aff = bool(gold_author.raw_affiliation)
                
                if pred_has_aff and gold_has_aff:
                    if self._fuzzy_match(pred_author.raw_affiliation, gold_author.raw_affiliation):
                        aff_tp += 1
                    else:
                        aff_fp += 1
                elif pred_has_aff and not gold_has_aff:
                    aff_fp += 1  # Hallucination
                elif not pred_has_aff and gold_has_aff:
                    aff_fn += 1  # Missed
                
                # Нормализация организации
                if pred_author.normalized_affiliation and gold_author.normalized_affiliation:
                    if self._fuzzy_match(
                        pred_author.normalized_affiliation,
                        gold_author.normalized_affiliation
                    ):
                        org_correct += 1
                
                # Страна
                if pred_author.country and gold_author.country:
                    if self._exact_match_normalized(pred_author.country, gold_author.country) or \
                       self._exact_match_normalized(pred_author.country_code or "", gold_author.country_code):
                        country_correct += 1
                
                # Тип организации
                pred_type = pred_author.org_type.value if pred_author.org_type else ""
                if pred_type and gold_author.org_type:
                    if pred_type == gold_author.org_type:
                        org_type_correct += 1
                
                # Иерархическая точность (и орг, и страна верны)
                org_match = self._fuzzy_match(
                    pred_author.normalized_affiliation or "",
                    gold_author.normalized_affiliation
                )
                country_match = self._exact_match_normalized(
                    pred_author.country or "",
                    gold_author.country
                ) or self._exact_match_normalized(
                    pred_author.country_code or "",
                    gold_author.country_code
                )
                
                if org_match and country_match:
                    hierarchical_correct += 1
            
            if verbose:
                print(f"[{pred_paper.arxiv_id}] Matched: {len(matched_gold)}/{len(gold_authors)} authors")
        
        # Расчёт метрик
        metrics.total_gold_authors = sum(
            len(p.authors) for p in self.gold_dataset.get_all_papers()
            if p.paper_id in {pp.arxiv_id for pp in predictions}
        )
        metrics.total_pred_authors = sum(len(p.authors) for p in predictions)
        metrics.matched_authors = author_tp
        
        # Author metrics
        if (author_tp + author_fp) > 0:
            metrics.author_precision = author_tp / (author_tp + author_fp)
        if (author_tp + author_fn) > 0:
            metrics.author_recall = author_tp / (author_tp + author_fn)
        if (metrics.author_precision + metrics.author_recall) > 0:
            metrics.author_f1 = 2 * metrics.author_precision * metrics.author_recall / \
                               (metrics.author_precision + metrics.author_recall)
        
        # Affiliation metrics
        if (aff_tp + aff_fp) > 0:
            metrics.affiliation_precision = aff_tp / (aff_tp + aff_fp)
        if (aff_tp + aff_fn) > 0:
            metrics.affiliation_recall = aff_tp / (aff_tp + aff_fn)
        if (metrics.affiliation_precision + metrics.affiliation_recall) > 0:
            metrics.affiliation_f1 = 2 * metrics.affiliation_precision * metrics.affiliation_recall / \
                                     (metrics.affiliation_precision + metrics.affiliation_recall)
        
        # Normalization metrics (only for matched authors)
        if total_matched > 0:
            metrics.org_normalization_accuracy = org_correct / total_matched
            metrics.country_accuracy = country_correct / total_matched
            metrics.org_type_accuracy = org_type_correct / total_matched
            metrics.hierarchical_accuracy = hierarchical_correct / total_matched
        
        # Hallucination rates
        if metrics.total_pred_authors > 0:
            metrics.author_hallucination_rate = author_fp / metrics.total_pred_authors
        if (aff_tp + aff_fp) > 0:
            metrics.affiliation_hallucination_rate = aff_fp / (aff_tp + aff_fp)
        
        return metrics
    
    # ============================================================
    # LEVEL 2: AGENT PERFORMANCE METRICS
    # ============================================================
    
    def evaluate_agent(
        self,
        run_logs: List[Dict[str, Any]],
        papers: List[PaperMetadata]
    ) -> AgentMetrics:
        """
        Оценка надёжности работы агента.
        
        Метрики:
        - Tool Success Rate (по каждому инструменту)
        - End-to-End Success Rate
        - Error breakdown by stage
        
        Args:
            run_logs: Логи выполнения агента
            papers: Результирующий список статей
            
        Returns:
            AgentMetrics
        """
        metrics = AgentMetrics()
        
        # Подсчёт успешности по этапам
        stage_attempts = defaultdict(int)
        stage_successes = defaultdict(int)
        
        for log in run_logs:
            stage = log.get("stage", "unknown")
            success = log.get("success", False)
            
            stage_attempts[stage] += 1
            if success:
                stage_successes[stage] += 1
        
        # Tool success rates
        if stage_attempts.get("arxiv_search", 0) > 0:
            metrics.arxiv_api_success = stage_successes["arxiv_search"] / stage_attempts["arxiv_search"]
        
        if stage_attempts.get("pdf_download", 0) > 0:
            metrics.pdf_download_success = stage_successes["pdf_download"] / stage_attempts["pdf_download"]
        
        if stage_attempts.get("pdf_parse", 0) > 0:
            metrics.pdf_parse_success = stage_successes["pdf_parse"] / stage_attempts["pdf_parse"]
        
        if stage_attempts.get("llm_extract", 0) > 0:
            metrics.llm_extraction_success = stage_successes["llm_extract"] / stage_attempts["llm_extract"]
        
        # Overall tool success
        total_attempts = sum(stage_attempts.values())
        total_successes = sum(stage_successes.values())
        if total_attempts > 0:
            metrics.tool_success_rate = total_successes / total_attempts
        
        # End-to-end success
        for paper in papers:
            if paper.processing_status.value == "completed":
                if paper.authors and any(a.normalized_affiliation for a in paper.authors):
                    metrics.papers_fully_processed += 1
                else:
                    metrics.papers_partial += 1
            else:
                metrics.papers_failed += 1
        
        total_papers = len(papers)
        if total_papers > 0:
            metrics.e2e_success_rate = metrics.papers_fully_processed / total_papers
        
        # Error breakdown
        metrics.errors_by_stage = {
            stage: attempts - stage_successes.get(stage, 0)
            for stage, attempts in stage_attempts.items()
        }
        
        return metrics
    
    # ============================================================
    # LEVEL 3: ENGINEERING METRICS
    # ============================================================
    
    def evaluate_engineering(
        self,
        start_time: float,
        end_time: float,
        papers_count: int,
        token_usage: Dict[str, int] = None,
        api_calls: Dict[str, int] = None,
        cache_stats: Dict[str, int] = None
    ) -> EngineeringMetrics:
        """
        Оценка инженерной эффективности.
        
        Args:
            start_time: Время начала (timestamp)
            end_time: Время окончания (timestamp)
            papers_count: Количество обработанных статей
            token_usage: {"input": N, "output": M}
            api_calls: {"arxiv": N, "semantic_scholar": M, ...}
            cache_stats: {"hits": N, "misses": M}
            
        Returns:
            EngineeringMetrics
        """
        metrics = EngineeringMetrics()
        
        # Timing
        metrics.total_time_seconds = end_time - start_time
        if papers_count > 0:
            metrics.avg_time_per_paper = metrics.total_time_seconds / papers_count
        
        # Token usage & cost (gpt-4o-mini pricing)
        if token_usage:
            metrics.total_input_tokens = token_usage.get("input", 0)
            metrics.total_output_tokens = token_usage.get("output", 0)
            
            # gpt-4o-mini: $0.15/1M input, $0.60/1M output
            input_cost = metrics.total_input_tokens * 0.15 / 1_000_000
            output_cost = metrics.total_output_tokens * 0.60 / 1_000_000
            metrics.estimated_cost_usd = input_cost + output_cost
        
        # API calls
        if api_calls:
            metrics.api_calls_by_source = api_calls
            metrics.total_api_calls = sum(api_calls.values())
        
        # Cache efficiency
        if cache_stats:
            hits = cache_stats.get("hits", 0)
            misses = cache_stats.get("misses", 0)
            total = hits + misses
            if total > 0:
                metrics.cache_hit_rate = hits / total
            metrics.cached_pdfs = hits
            metrics.downloaded_pdfs = misses
        
        return metrics
    
    # ============================================================
    # FULL EVALUATION
    # ============================================================
    
    def evaluate_full(
        self,
        predictions: List[PaperMetadata],
        run_logs: List[Dict[str, Any]] = None,
        start_time: float = None,
        end_time: float = None,
        token_usage: Dict[str, int] = None,
        api_calls: Dict[str, int] = None,
        cache_stats: Dict[str, int] = None,
        verbose: bool = False
    ) -> EvaluationReport:
        """
        Полная оценка системы по всем уровням метрик.
        
        Args:
            predictions: Результаты работы агента
            run_logs: Логи выполнения
            start_time: Время начала
            end_time: Время окончания
            token_usage: Использование токенов
            api_calls: Вызовы API
            cache_stats: Статистика кэша
            verbose: Подробный вывод
            
        Returns:
            EvaluationReport с полным набором метрик
        """
        report = EvaluationReport()
        report.timestamp = datetime.now().isoformat()
        
        # Level 1: Extraction Quality
        if self.gold_dataset.papers:
            report.extraction = self.evaluate_extraction(predictions, verbose)
        
        # Level 2: Agent Performance
        if run_logs:
            report.agent = self.evaluate_agent(run_logs, predictions)
        
        # Level 3: Engineering Metrics
        if start_time and end_time:
            report.engineering = self.evaluate_engineering(
                start_time, end_time, len(predictions),
                token_usage, api_calls, cache_stats
            )
        
        # Overall Quality Score (weighted composite)
        # Weights: Extraction 60%, Agent 25%, Engineering 15%
        extraction_score = (
            report.extraction.author_f1 * 0.3 +
            report.extraction.affiliation_f1 * 0.3 +
            report.extraction.hierarchical_accuracy * 0.4
        ) if report.extraction.author_f1 > 0 else 0
        
        agent_score = report.agent.e2e_success_rate
        
        # Engineering: normalize time (assuming 60s/paper is baseline)
        baseline_time = 60.0
        time_score = min(1.0, baseline_time / max(report.engineering.avg_time_per_paper, 1))
        
        report.overall_quality_score = (
            extraction_score * 0.60 +
            agent_score * 0.25 +
            time_score * 0.15
        )
        
        return report
    
    def print_report(self, report: EvaluationReport) -> None:
        """Красивый вывод отчёта в консоль"""
        print("\n" + "="*70)
        print("  EVALUATION REPORT")
        print("="*70)
        print(f"  Timestamp: {report.timestamp}")
        print(f"  Overall Quality Score: {report.overall_quality_score:.2%}")
        
        print("\n" + "-"*70)
        print("  LEVEL 1: EXTRACTION QUALITY")
        print("-"*70)
        e = report.extraction
        print(f"  Authors:")
        print(f"    Precision: {e.author_precision:.2%}")
        print(f"    Recall:    {e.author_recall:.2%}")
        print(f"    F1-Score:  {e.author_f1:.2%}")
        print(f"  Affiliations:")
        print(f"    Precision: {e.affiliation_precision:.2%}")
        print(f"    Recall:    {e.affiliation_recall:.2%}")
        print(f"    F1-Score:  {e.affiliation_f1:.2%}")
        print(f"  Normalization:")
        print(f"    Organization: {e.org_normalization_accuracy:.2%}")
        print(f"    Country:      {e.country_accuracy:.2%}")
        print(f"    Org Type:     {e.org_type_accuracy:.2%}")
        print(f"    Hierarchical: {e.hierarchical_accuracy:.2%}")
        print(f"  Hallucinations:")
        print(f"    Author rate:      {e.author_hallucination_rate:.2%}")
        print(f"    Affiliation rate: {e.affiliation_hallucination_rate:.2%}")
        
        print("\n" + "-"*70)
        print("  LEVEL 2: AGENT PERFORMANCE")
        print("-"*70)
        a = report.agent
        print(f"  Tool Success Rate: {a.tool_success_rate:.2%}")
        print(f"    ArXiv API:      {a.arxiv_api_success:.2%}")
        print(f"    PDF Download:   {a.pdf_download_success:.2%}")
        print(f"    PDF Parse:      {a.pdf_parse_success:.2%}")
        print(f"    LLM Extract:    {a.llm_extraction_success:.2%}")
        print(f"  End-to-End Success: {a.e2e_success_rate:.2%}")
        print(f"    Fully processed: {a.papers_fully_processed}")
        print(f"    Partial:         {a.papers_partial}")
        print(f"    Failed:          {a.papers_failed}")
        
        print("\n" + "-"*70)
        print("  LEVEL 3: ENGINEERING METRICS")
        print("-"*70)
        eng = report.engineering
        print(f"  Timing:")
        print(f"    Total time:     {eng.total_time_seconds:.1f}s")
        print(f"    Per paper:      {eng.avg_time_per_paper:.1f}s")
        print(f"  Cost:")
        print(f"    Input tokens:   {eng.total_input_tokens:,}")
        print(f"    Output tokens:  {eng.total_output_tokens:,}")
        print(f"    Estimated cost: ${eng.estimated_cost_usd:.4f}")
        print(f"  Cache:")
        print(f"    Hit rate:       {eng.cache_hit_rate:.2%}")
        print(f"    Cached PDFs:    {eng.cached_pdfs}")
        print(f"    Downloaded:     {eng.downloaded_pdfs}")
        
        print("\n" + "="*70)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_gold_standard_template(
    paper_ids: List[str],
    output_path: str = "./data/gold_standard_template.json"
) -> None:
    """
    Создать шаблон для ручной разметки gold standard датасета.
    
    Args:
        paper_ids: Список ArXiv ID статей для разметки
        output_path: Путь для сохранения шаблона
    """
    dataset = GoldStandardDataset(output_path)
    
    for paper_id in paper_ids:
        template = dataset.create_template(paper_id, f"FILL: Title for {paper_id}")
        paper = GoldPaper(
            paper_id=paper_id,
            title=template["title"],
            authors=[GoldAuthor(**template["authors"][0])],
            source="manual",
            annotator=template["annotator"],
            annotation_date=template["annotation_date"]
        )
        dataset.add_paper(paper)
    
    dataset.save()
    print(f"Template saved to {output_path}")
    print(f"Papers to annotate: {len(paper_ids)}")
    print("\nInstructions:")
    print("1. Open the JSON file")
    print("2. For each paper, fill in the author information")
    print("3. Add more authors by copying the author template")
    print("4. Save the file when done")


def load_predictions_from_csv(csv_path: str) -> List[PaperMetadata]:
    """
    Загрузить результаты агента из CSV файла.
    
    Args:
        csv_path: Путь к CSV с результатами
        
    Returns:
        Список PaperMetadata
    """
    df = pd.read_csv(csv_path)
    
    papers_dict: Dict[str, PaperMetadata] = {}
    
    for _, row in df.iterrows():
        paper_id = row["paper_id"]
        
        if paper_id not in papers_dict:
            papers_dict[paper_id] = PaperMetadata(
                arxiv_id=paper_id,
                title=row.get("paper_title", ""),
                authors=[]
            )
        
        author = AuthorAffiliation(
            name=row["author_name"],
            raw_affiliation=row.get("raw_affiliation", ""),
            normalized_affiliation=row.get("normalized_affiliation"),
            country=row.get("country"),
            country_code=row.get("country_code"),
            org_type=OrganizationType(row.get("org_type", "unknown")),
            confidence=float(row.get("confidence", 1.0))
        )
        papers_dict[paper_id].authors.append(author)
    
    return list(papers_dict.values())
