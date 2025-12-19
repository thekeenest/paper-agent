"""
FastAPI Application - главное приложение API.

Endpoints:
- POST /api/analyze - запуск анализа
- GET /api/tasks - список задач
- GET /api/tasks/{task_id} - статус задачи
- GET /api/tasks/{task_id}/results - результаты
- GET /api/tasks/{task_id}/analytics - аналитика
- DELETE /api/tasks/{task_id} - удаление задачи
- WebSocket /ws/{task_id} - real-time updates
- GET /health - healthcheck
"""

import os
import asyncio
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager

from .models import (
    AnalysisRequest,
    AnalysisResponse,
    TaskStatus,
    TaskStatusEnum,
    TaskProgress,
    TaskResult,
    AnalyticsData,
    PaperData,
    AuthorData,
    EvaluationRequest,
    EvaluationResponse,
    ExtractionMetricsResponse,
    HealthResponse,
    ErrorResponse,
)
from .task_manager import task_manager


# ============================================================
# APPLICATION FACTORY
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager для FastAPI"""
    # Startup
    print("🚀 Conference Paper Agent API starting...")
    yield
    # Shutdown
    print("👋 Conference Paper Agent API shutting down...")


def create_app() -> FastAPI:
    """Создание FastAPI приложения"""
    
    app = FastAPI(
        title="Conference Paper Agent API",
        description="""
        ## Мульти-агентная система для анализа публикаций научных конференций
        
        ### Возможности:
        - 🔍 Поиск статей в ArXiv, Semantic Scholar, OpenAlex
        - 📄 Извлечение авторов и аффилиаций из PDF
        - 🏢 Нормализация организаций через KB и ROR
        - 📊 Аналитика и визуализации
        - 📈 Метрики качества извлечения
        
        ### WebSocket
        Подключитесь к `/ws/{task_id}` для получения real-time обновлений прогресса.
        """,
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS для фронтенда
    frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            frontend_url,
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register routes
    register_routes(app)
    
    return app


def register_routes(app: FastAPI):
    """Регистрация всех маршрутов"""
    
    # ============================================================
    # HEALTH CHECK
    # ============================================================
    
    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health_check():
        """Проверка работоспособности API"""
        # Проверяем доступность сервисов
        services = {
            "api": True,
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "semantic_scholar": bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY")),
        }
        
        return HealthResponse(
            status="healthy" if all(services.values()) else "degraded",
            services=services,
        )
    
    # ============================================================
    # ANALYSIS ENDPOINTS
    # ============================================================
    
    @app.get(
        "/api/active-task",
        response_model=Optional[TaskStatus],
        tags=["Tasks"],
        summary="Get active running task"
    )
    async def get_active_task():
        """
        Get the currently running task if any.
        Returns null if no task is running.
        """
        active_task = task_manager.get_active_task()
        if active_task:
            return active_task.to_status()
        return None
    
    @app.post(
        "/api/analyze",
        response_model=AnalysisResponse,
        tags=["Analysis"],
        summary="Start new analysis"
    )
    async def start_analysis(
        request: AnalysisRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Запуск нового анализа статей.
        
        - **query**: Поисковый запрос (ArXiv syntax: cat:cs.AI, ti:transformer)
        - **max_papers**: Максимум статей для анализа (1-500)
        - **data_source**: Источник данных (arxiv, semantic_scholar, openalex)
        
        Note: Only one task can run at a time. If a task is already running,
        this will return an error with the active task ID.
        """
        # Check if there's already an active task
        active_task = task_manager.get_active_task()
        if active_task:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "A task is already running. Please wait for it to complete.",
                    "active_task_id": active_task.task_id,
                    "active_task_query": active_task.query,
                    "active_task_progress": active_task.progress,
                    "active_task_stage": active_task.stage.value,
                }
            )
        
        # Проверка API ключа
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(
                status_code=500,
                detail="OPENAI_API_KEY not configured"
            )
        
        # Создание задачи
        task_id = task_manager.create_task(
            query=request.query,
            max_papers=request.max_papers,
            data_source=request.data_source,
            date_from=request.date_from,
            date_to=request.date_to,
        )
        
        # Запуск в фоне
        background_tasks.add_task(task_manager.run_analysis, task_id)
        
        return AnalysisResponse(
            task_id=task_id,
            status=TaskStatusEnum.PENDING,
            message=f"Analysis started for query: {request.query}",
            created_at=datetime.now(),
        )
    
    @app.get(
        "/api/tasks",
        response_model=List[TaskStatus],
        tags=["Tasks"],
        summary="List all tasks"
    )
    async def list_tasks(
        status: Optional[TaskStatusEnum] = Query(None, description="Filter by status"),
        limit: int = Query(50, ge=1, le=200),
    ):
        """Список всех задач с опциональной фильтрацией"""
        tasks = task_manager.get_all_tasks()
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        # Сортировка по дате создания (новые первыми)
        tasks.sort(key=lambda t: t.started_at or datetime.min, reverse=True)
        
        return tasks[:limit]
    
    @app.get(
        "/api/tasks/{task_id}",
        response_model=TaskStatus,
        tags=["Tasks"],
        summary="Get task status"
    )
    async def get_task_status(task_id: str):
        """Получить детальный статус задачи"""
        status = task_manager.get_task_status(task_id)
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        return status
    
    @app.get(
        "/api/tasks/{task_id}/results",
        response_model=TaskResult,
        tags=["Tasks"],
        summary="Get task results"
    )
    async def get_task_results(task_id: str):
        """Получить полные результаты задачи"""
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task.status not in [TaskStatusEnum.COMPLETED, TaskStatusEnum.FAILED]:
            raise HTTPException(
                status_code=400,
                detail=f"Task is still {task.status.value}"
            )
        
        # Конвертация papers в PaperData
        papers_data = []
        for paper in task.papers:
            authors_data = [
                AuthorData(
                    name=author.name,
                    raw_affiliation=author.raw_affiliation,
                    normalized_affiliation=author.normalized_affiliation,
                    country=author.country,
                    country_code=author.country_code,
                    org_type=author.org_type.value if author.org_type else "unknown",
                    confidence=author.confidence,
                )
                for author in paper.authors
            ]
            
            papers_data.append(PaperData(
                paper_id=paper.arxiv_id,
                title=paper.title,
                abstract=paper.abstract[:500] + "..." if paper.abstract and len(paper.abstract) > 500 else paper.abstract,
                published_date=paper.published_date,
                categories=paper.categories or [],
                authors=authors_data,
                pdf_url=paper.pdf_url,
                processing_status=paper.processing_status.value if paper.processing_status else "unknown",
            ))
        
        # Output files
        output_files = {}
        if task.output_path:
            output_dir = Path(task.output_path).parent
            for f in output_dir.glob("*"):
                if f.is_file() and task_id[:8] in f.name:
                    output_files[f.name] = str(f)
        
        return TaskResult(
            task_id=task_id,
            status=task.status,
            analytics=task.analytics,
            papers=papers_data,
            output_files=output_files,
            errors=task.errors,
        )
    
    @app.get(
        "/api/tasks/{task_id}/analytics",
        response_model=AnalyticsData,
        tags=["Analytics"],
        summary="Get analytics data"
    )
    async def get_task_analytics(task_id: str):
        """Получить аналитические данные для визуализаций"""
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Return analytics if available, otherwise return empty data structure
        if task.analytics:
            return task.analytics
        
        # Return empty analytics for tasks without data yet
        return AnalyticsData(
            total_papers=task.processed_papers or 0,
            total_authors=0,
            unique_authors=0,
            unique_organizations=0,
            unique_countries=0,
            avg_authors_per_paper=0,
            avg_confidence=0,
            top_organizations=[],
            country_distribution=[],
            org_type_distribution=[],
            processing_time_seconds=0,
            data_source=task.data_source,
        )
    
    @app.delete(
        "/api/tasks/{task_id}",
        tags=["Tasks"],
        summary="Delete task"
    )
    async def delete_task(task_id: str):
        """Удалить задачу и её данные"""
        if not task_manager.delete_task(task_id):
            raise HTTPException(status_code=404, detail="Task not found")
        return {"message": "Task deleted", "task_id": task_id}
    
    @app.post(
        "/api/tasks/{task_id}/cancel",
        tags=["Tasks"],
        summary="Cancel running task"
    )
    async def cancel_task(task_id: str):
        """Отменить выполняющуюся задачу"""
        if not task_manager.cancel_task(task_id):
            raise HTTPException(
                status_code=400,
                detail="Task cannot be cancelled (not running or not found)"
            )
        return {"message": "Task cancelled", "task_id": task_id}
    
    # ============================================================
    # FILE DOWNLOADS
    # ============================================================
    
    @app.get(
        "/api/tasks/{task_id}/download/{filename}",
        tags=["Downloads"],
        summary="Download output file"
    )
    async def download_file(task_id: str, filename: str):
        """Скачать выходной файл задачи"""
        task = task_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if not task.output_path:
            raise HTTPException(status_code=404, detail="No output files")
        
        output_dir = Path(task.output_path).parent
        file_path = output_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            file_path,
            filename=filename,
            media_type="application/octet-stream"
        )
    
    # ============================================================
    # EVALUATION ENDPOINTS
    # ============================================================
    
    @app.post(
        "/api/evaluate",
        response_model=EvaluationResponse,
        tags=["Evaluation"],
        summary="Evaluate extraction quality"
    )
    async def evaluate_task(request: EvaluationRequest):
        """Оценка качества извлечения против gold standard"""
        task = task_manager.get_task(request.task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        if task.status != TaskStatusEnum.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail="Task must be completed for evaluation"
            )
        
        try:
            from ..evaluation import EvaluationEngine, GoldStandardDataset
            
            gold_path = request.gold_standard_path or "./data/gold_standard.json"
            gold = GoldStandardDataset(gold_path)
            
            if not gold.papers:
                raise HTTPException(
                    status_code=404,
                    detail="Gold standard dataset not found or empty"
                )
            
            engine = EvaluationEngine(gold)
            
            # Загружаем predictions из output
            if task.output_path:
                csv_path = task.output_path.replace(".json", ".csv").replace("report_", "affiliations_")
                report = engine.generate_report(predictions_csv=csv_path)
                
                return EvaluationResponse(
                    task_id=request.task_id,
                    timestamp=datetime.now(),
                    extraction_metrics=ExtractionMetricsResponse(
                        author_precision=report.extraction.author_precision,
                        author_recall=report.extraction.author_recall,
                        author_f1=report.extraction.author_f1,
                        affiliation_precision=report.extraction.affiliation_precision,
                        affiliation_recall=report.extraction.affiliation_recall,
                        affiliation_f1=report.extraction.affiliation_f1,
                        org_normalization_accuracy=report.extraction.org_normalization_accuracy,
                        country_accuracy=report.extraction.country_accuracy,
                        hierarchical_accuracy=report.extraction.hierarchical_accuracy,
                        hallucination_rate=report.extraction.author_hallucination_rate,
                    ),
                    overall_score=report.overall_quality_score,
                    gold_standard_papers=len(gold.papers),
                    evaluated_papers=report.extraction.total_gold_authors,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No output file available for evaluation"
                )
                
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Evaluation module not available"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============================================================
    # WEBSOCKET ENDPOINT
    # ============================================================
    
    @app.websocket("/ws/{task_id}")
    async def websocket_endpoint(websocket: WebSocket, task_id: str):
        """WebSocket для real-time обновлений прогресса"""
        await websocket.accept()
        
        task = task_manager.get_task(task_id)
        if not task:
            await websocket.send_json({"error": "Task not found"})
            await websocket.close()
            return
        
        # Отправляем текущий статус
        status = task.to_status()
        await websocket.send_json({
            "type": "status",
            "data": status.model_dump(mode="json"),
        })
        
        # Get the progress queue for this task
        progress_queue = task_manager.get_progress_queue(task_id)
        
        try:
            while True:
                # Check for progress updates from queue (non-blocking)
                try:
                    while True:
                        progress = progress_queue.get_nowait()
                        await websocket.send_json({
                            "type": "progress",
                            "data": progress.model_dump(mode="json"),
                        })
                except Exception:
                    pass  # Queue empty or other error
                
                # Check for task completion
                task = task_manager.get_task(task_id)
                if task and task.status in [
                    TaskStatusEnum.COMPLETED,
                    TaskStatusEnum.FAILED,
                    TaskStatusEnum.CANCELLED
                ]:
                    await websocket.send_json({
                        "type": "completed",
                        "data": task.to_status().model_dump(mode="json"),
                    })
                    break
                
                # Wait for client messages with short timeout to allow progress checks
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.5  # Short timeout to check queue frequently
                    )
                    
                    if data == "ping":
                        await websocket.send_text("pong")
                    elif data == "status":
                        # Resend current status
                        task = task_manager.get_task(task_id)
                        if task:
                            await websocket.send_json({
                                "type": "status",
                                "data": task.to_status().model_dump(mode="json"),
                            })
                            
                except asyncio.TimeoutError:
                    # No message from client, continue loop to check queue
                    pass
                    
        except WebSocketDisconnect:
            pass
    
    # ============================================================
    # STATIC DATA ENDPOINTS
    # ============================================================
    
    @app.get(
        "/api/data-sources",
        tags=["System"],
        summary="List available data sources"
    )
    async def list_data_sources():
        """Список доступных источников данных"""
        return [
            {
                "id": "arxiv",
                "name": "ArXiv",
                "description": "Open-access preprint repository",
                "requires_key": False,
                "query_syntax": "cat:cs.AI, ti:transformer, au:bengio"
            },
            {
                "id": "semantic_scholar",
                "name": "Semantic Scholar",
                "description": "Academic search engine with citation data",
                "requires_key": True,
                "key_env": "SEMANTIC_SCHOLAR_API_KEY"
            },
            {
                "id": "openalex",
                "name": "OpenAlex",
                "description": "Open catalog of scholarly works",
                "requires_key": False,
                "query_syntax": "Free text search"
            }
        ]
    
    @app.get(
        "/api/query-examples",
        tags=["System"],
        summary="Get query examples"
    )
    async def get_query_examples():
        """Примеры запросов для разных источников"""
        return {
            "arxiv": [
                {"query": "cat:cs.AI", "description": "Artificial Intelligence"},
                {"query": "cat:cs.LG", "description": "Machine Learning"},
                {"query": "cat:cs.CV", "description": "Computer Vision"},
                {"query": "cat:cs.CL", "description": "NLP/Computational Linguistics"},
                {"query": "ti:transformer", "description": "Papers with 'transformer' in title"},
                {"query": "au:hinton", "description": "Papers by author Hinton"},
            ],
            "semantic_scholar": [
                {"query": "machine learning", "description": "ML papers"},
                {"query": "large language models", "description": "LLM research"},
            ],
            "openalex": [
                {"query": "artificial intelligence", "description": "AI research"},
                {"query": "neural networks", "description": "Neural network papers"},
            ]
        }


# Создание приложения
app = create_app()
