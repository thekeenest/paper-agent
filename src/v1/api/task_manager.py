"""
Task Manager для управления фоновыми задачами анализа.

Обеспечивает:
- Асинхронное выполнение задач
- Отслеживание прогресса
- WebSocket уведомления
- Хранение результатов
"""

import asyncio
import uuid
import time
import queue
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback

from .models import (
    TaskStatusEnum,
    ProcessingStage,
    TaskStatus,
    TaskProgress,
    AnalyticsData,
    PaperData,
    AuthorData,
    OrganizationStats,
    CountryStats,
    OrgTypeStats,
)


@dataclass
class TaskData:
    """Внутренние данные задачи"""
    task_id: str
    status: TaskStatusEnum = TaskStatusEnum.PENDING
    stage: ProcessingStage = ProcessingStage.IDLE
    progress: float = 0.0
    
    # Query params
    query: str = ""
    max_papers: int = 10
    data_source: str = "arxiv"
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    
    # Processing
    total_papers: int = 0
    processed_papers: int = 0
    failed_papers: int = 0
    current_paper_id: Optional[str] = None
    current_paper_title: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    papers: List[Any] = field(default_factory=list)
    analytics: Optional[AnalyticsData] = None
    output_path: Optional[str] = None
    
    # Errors
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_status(self) -> TaskStatus:
        """Конвертация в TaskStatus для API"""
        elapsed = 0.0
        estimated_remaining = None
        
        if self.started_at:
            elapsed = (datetime.now() - self.started_at).total_seconds()
            if self.processed_papers > 0 and self.total_papers > 0:
                avg_per_paper = elapsed / self.processed_papers
                remaining = self.total_papers - self.processed_papers
                estimated_remaining = avg_per_paper * remaining
        
        return TaskStatus(
            task_id=self.task_id,
            status=self.status,
            stage=self.stage,
            progress=self.progress,
            total_papers=self.total_papers,
            processed_papers=self.processed_papers,
            failed_papers=self.failed_papers,
            current_paper_id=self.current_paper_id,
            current_paper_title=self.current_paper_title,
            started_at=self.started_at,
            updated_at=self.updated_at,
            completed_at=self.completed_at,
            elapsed_seconds=elapsed,
            estimated_remaining=estimated_remaining,
            errors=self.errors,
            query=self.query,
            data_source=self.data_source,
            max_papers=self.max_papers,
        )


class TaskManager:
    """
    Менеджер асинхронных задач анализа.
    
    Singleton-паттерн для глобального доступа.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.tasks: Dict[str, TaskData] = {}
        self.websocket_subscribers: Dict[str, Set[Callable]] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._active_task_id: Optional[str] = None  # Single active task for all users
        self._progress_queues: Dict[str, queue.Queue] = {}  # Thread-safe queues for progress
        self._initialized = True
    
    def get_progress_queue(self, task_id: str) -> queue.Queue:
        """Get or create a thread-safe queue for task progress updates"""
        if task_id not in self._progress_queues:
            self._progress_queues[task_id] = queue.Queue()
        return self._progress_queues[task_id]
    
    def get_active_task(self) -> Optional[TaskData]:
        """Get currently running task (if any)"""
        if self._active_task_id:
            task = self.tasks.get(self._active_task_id)
            if task and task.status == TaskStatusEnum.RUNNING:
                return task
            # Clear stale reference
            self._active_task_id = None
        return None
    
    def get_active_task_id(self) -> Optional[str]:
        """Get ID of currently running task"""
        active = self.get_active_task()
        return active.task_id if active else None
    
    def can_start_new_task(self) -> bool:
        """Check if a new task can be started (no active task running)"""
        return self.get_active_task() is None
    
    def create_task(
        self,
        query: str,
        max_papers: int = 10,
        data_source: str = "arxiv",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> str:
        """Создание новой задачи"""
        task_id = str(uuid.uuid4())[:8]
        
        task = TaskData(
            task_id=task_id,
            query=query,
            max_papers=max_papers,
            data_source=data_source,
            date_from=date_from,
            date_to=date_to,
            updated_at=datetime.now(),
        )
        
        self.tasks[task_id] = task
        self.websocket_subscribers[task_id] = set()
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[TaskData]:
        """Получить данные задачи"""
        return self.tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Получить статус задачи для API"""
        task = self.tasks.get(task_id)
        if task:
            return task.to_status()
        return None
    
    def get_all_tasks(self) -> List[TaskStatus]:
        """Получить статусы всех задач"""
        return [task.to_status() for task in self.tasks.values()]
    
    def update_progress(
        self,
        task_id: str,
        stage: ProcessingStage,
        progress: float,
        message: str = "",
        current_paper: Optional[str] = None,
        processed: Optional[int] = None,
        total: Optional[int] = None,
    ):
        """Обновить прогресс задачи"""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        task.stage = stage
        task.progress = progress
        task.updated_at = datetime.now()
        
        if current_paper:
            task.current_paper_title = current_paper
        if processed is not None:
            task.processed_papers = processed
        if total is not None:
            task.total_papers = total
        
        # Put progress in queue for WebSocket consumers (thread-safe)
        progress_data = TaskProgress(
            task_id=task_id,
            stage=stage,
            progress=progress,
            message=message,
            current_paper=current_paper,
            processed=task.processed_papers,
            total=task.total_papers,
        )
        
        # Add to queue for any waiting WebSocket connections
        q = self.get_progress_queue(task_id)
        try:
            q.put_nowait(progress_data)
        except queue.Full:
            pass  # Queue full, skip this update
    
    def subscribe(self, task_id: str, callback: Callable):
        """Подписаться на обновления задачи"""
        if task_id not in self.websocket_subscribers:
            self.websocket_subscribers[task_id] = set()
        self.websocket_subscribers[task_id].add(callback)
    
    def unsubscribe(self, task_id: str, callback: Callable):
        """Отписаться от обновлений задачи"""
        if task_id in self.websocket_subscribers:
            self.websocket_subscribers[task_id].discard(callback)
    
    async def run_analysis(self, task_id: str):
        """
        Запуск анализа в фоне.
        
        Этот метод выполняет агентный граф и обновляет прогресс.
        """
        task = self.tasks.get(task_id)
        if not task:
            return
        
        # Set as active task
        self._active_task_id = task_id
        
        task.status = TaskStatusEnum.RUNNING
        task.started_at = datetime.now()
        
        try:
            # Импорт здесь чтобы избежать циклических зависимостей
            from ..graph import create_app
            from ..state import create_initial_state
            from ..analytics import AnalyticsEngine
            
            self.update_progress(
                task_id,
                ProcessingStage.SEARCHING,
                5,
                f"Searching papers: {task.query}"
            )
            
            # Создание приложения
            app = create_app()
            
            # Начальное состояние
            initial_state = create_initial_state(
                query=task.query,
                max_papers=task.max_papers,
                date_from=task.date_from,
                date_to=task.date_to,
                max_retries=3,
                data_source=task.data_source
            )
            
            # Запускаем в отдельном потоке чтобы не блокировать event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_agent_sync,
                app,
                initial_state,
                task_id,
                task.max_papers,
            )
            
            # Обработка результатов
            if result:
                task.papers = result.get("papers", [])
                task.processed_papers = result.get("processed_count", 0)
                task.failed_papers = result.get("error_count", 0)
                task.output_path = result.get("output_path")
                
                # Генерация аналитики
                if task.papers:
                    self.update_progress(
                        task_id,
                        ProcessingStage.AGGREGATING,
                        95,
                        "Generating analytics..."
                    )
                    
                    engine = AnalyticsEngine("./output")
                    engine.load_from_papers(task.papers)
                    
                    stats = engine.get_summary_stats()
                    top_orgs = engine.get_top_organizations(20)
                    countries = engine.get_country_distribution()
                    org_types = engine.get_org_type_distribution()
                    
                    task.analytics = AnalyticsData(
                        total_papers=stats["total_papers"],
                        total_authors=stats["total_authors"],
                        unique_authors=stats["unique_authors"],
                        unique_organizations=stats["unique_organizations"],
                        unique_countries=stats["unique_countries"],
                        avg_authors_per_paper=stats["avg_authors_per_paper"],
                        avg_confidence=stats.get("avg_confidence", 0),
                        top_organizations=[
                            OrganizationStats(
                                name=row["normalized_affiliation"],
                                author_count=row["author_count"],
                                country=row.get("country"),
                                org_type=row.get("org_type", "unknown"),
                                percentage=row["author_count"] / stats["total_authors"] * 100
                            )
                            for _, row in top_orgs.iterrows()
                        ],
                        country_distribution=[
                            CountryStats(
                                country=row["country"],
                                author_count=row["author_count"],
                                org_count=row["org_count"],
                                percentage=row["author_count"] / stats["total_authors"] * 100
                            )
                            for _, row in countries.iterrows()
                        ],
                        org_type_distribution=[
                            OrgTypeStats(
                                org_type=row["org_type"],
                                count=row["count"],
                                percentage=row["count"] / stats["total_authors"] * 100
                            )
                            for _, row in org_types.iterrows()
                        ],
                        processing_time_seconds=(datetime.now() - task.started_at).total_seconds(),
                        data_source=task.data_source,
                    )
                
                task.status = TaskStatusEnum.COMPLETED
                task.completed_at = datetime.now()
                
                # Clear active task
                if self._active_task_id == task_id:
                    self._active_task_id = None
                
                self.update_progress(
                    task_id,
                    ProcessingStage.COMPLETED,
                    100,
                    f"Analysis complete! Processed {task.processed_papers} papers."
                )
            else:
                raise Exception("Agent returned no results")
                
        except Exception as e:
            task.status = TaskStatusEnum.FAILED
            task.errors.append({
                "stage": task.stage.value,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "timestamp": datetime.now().isoformat(),
            })
            
            # Clear active task on failure too
            if self._active_task_id == task_id:
                self._active_task_id = None
            
            self.update_progress(
                task_id,
                ProcessingStage.FAILED,
                task.progress,
                f"Error: {str(e)}"
            )
    
    def _run_agent_sync(
        self,
        app,
        initial_state,
        task_id: str,
        max_papers: int,
    ) -> Dict[str, Any]:
        """
        Синхронный запуск агента с обновлениями прогресса.
        
        Эта функция выполняется в отдельном потоке.
        """
        task = self.tasks.get(task_id)
        
        # Добавляем callback для отслеживания прогресса
        # Используем streaming итератор LangGraph
        recursion_limit = max_papers * 6 + 20
        
        try:
            # Запуск с потоковой передачей состояний
            # Аккумулируем состояние из всех узлов
            accumulated_state = dict(initial_state)
            
            for i, state in enumerate(app.stream(
                initial_state,
                config={"recursion_limit": recursion_limit}
            )):
                # Получаем обновления из каждого узла
                if isinstance(state, dict):
                    for node_name, node_state in state.items():
                        # Обновляем накопленное состояние
                        if isinstance(node_state, dict):
                            accumulated_state.update(node_state)
                        
                        # Определяем стадию по имени узла
                        stage_map = {
                            "search": ProcessingStage.SEARCHING,
                            "download": ProcessingStage.DOWNLOADING,
                            "parse": ProcessingStage.PARSING,
                            "extract": ProcessingStage.EXTRACTING,
                            "normalize": ProcessingStage.NORMALIZING,
                            "aggregate": ProcessingStage.AGGREGATING,
                        }
                        
                        stage = stage_map.get(node_name, ProcessingStage.IDLE)
                        
                        # Обновляем прогресс из накопленного состояния
                        processed = accumulated_state.get("processed_count", 0)
                        papers = accumulated_state.get("papers", [])
                        total = len(papers) or max_papers
                        progress = min(90, (processed / total) * 90) if total > 0 else 0
                        
                        current_paper = None
                        current_idx = accumulated_state.get("current_index", 0)
                        if papers and current_idx < len(papers):
                            current_paper = papers[current_idx].title[:50] + "..." if len(papers[current_idx].title) > 50 else papers[current_idx].title
                        
                        if task:
                            task.stage = stage
                            task.progress = progress
                            task.processed_papers = processed
                            task.total_papers = total
                            task.current_paper_title = current_paper
                            task.updated_at = datetime.now()
                            
                            # Put progress to the correct per-task queue for WebSocket broadcast
                            self.get_progress_queue(task_id).put(TaskProgress(
                                task_id=task_id,
                                stage=stage,
                                progress=progress,
                                message=f"Stage: {node_name}",
                                current_paper=current_paper,
                                processed=processed,
                                total=total,
                            ))
            
            return accumulated_state
            
        except Exception as e:
            if task:
                task.errors.append({
                    "stage": "execution",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
            raise
    
    def cancel_task(self, task_id: str) -> bool:
        """Отмена задачи"""
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatusEnum.RUNNING:
            task.status = TaskStatusEnum.CANCELLED
            task.completed_at = datetime.now()
            return True
        return False
    
    def delete_task(self, task_id: str) -> bool:
        """Удаление задачи"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            if task_id in self.websocket_subscribers:
                del self.websocket_subscribers[task_id]
            return True
        return False


# Глобальный экземпляр
task_manager = TaskManager()
