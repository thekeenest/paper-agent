"""
Сборка и компиляция агентного графа LangGraph.
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import (
    search_papers,
    download_paper,
    parse_pdf,
    extract_affiliations,
    normalize_affiliations,
    aggregate_results,
    should_continue_processing,
)


def build_agent_graph() -> StateGraph:
    """
    Построение графа состояний для агентной системы.
    
    Архитектура:
    
    [START] 
       ↓
    [search_papers] - Поиск статей в ArXiv
       ↓
    [download_paper] - Скачивание PDF ←──────┐
       ↓                                     │
    [parse_pdf] - Извлечение текста          │
       ↓                                     │
    [extract_affiliations] - LLM extraction  │
       ↓                                     │
    [normalize_affiliations] - Нормализация  │
       ↓                                     │
    [should_continue?] ──→ yes ──────────────┘
       ↓ no
    [aggregate_results] - Формирование отчёта
       ↓
    [END]
    
    Returns:
        Скомпилированный граф StateGraph
    """
    
    # Создаём граф с определённым состоянием
    workflow = StateGraph(AgentState)
    
    # ============================================================
    # ДОБАВЛЕНИЕ УЗЛОВ
    # ============================================================
    
    workflow.add_node("search", search_papers)
    workflow.add_node("download", download_paper)
    workflow.add_node("parse", parse_pdf)
    workflow.add_node("extract", extract_affiliations)
    workflow.add_node("normalize", normalize_affiliations)
    workflow.add_node("aggregate", aggregate_results)
    
    # ============================================================
    # ОПРЕДЕЛЕНИЕ ПЕРЕХОДОВ
    # ============================================================
    
    # Точка входа
    workflow.set_entry_point("search")
    
    # Последовательные переходы
    workflow.add_edge("search", "download")
    workflow.add_edge("download", "parse")
    workflow.add_edge("parse", "extract")
    workflow.add_edge("extract", "normalize")
    
    # Условный переход после нормализации:
    # - Если есть ещё статьи → download
    # - Если нужен retry → extract  
    # - Иначе → aggregate
    workflow.add_conditional_edges(
        "normalize",
        should_continue_processing,
        {
            "download": "download",
            "extract": "extract",
            "aggregate": "aggregate"
        }
    )
    
    # Завершение
    workflow.add_edge("aggregate", END)
    
    return workflow


def compile_graph():
    """
    Компиляция графа в исполняемый объект.
    
    Returns:
        Compiled graph ready for invocation
    """
    workflow = build_agent_graph()
    return workflow.compile()


def create_app():
    """
    Создание готового к использованию приложения.
    
    Returns:
        Compiled LangGraph application
    """
    return compile_graph()


# Визуализация графа (для отладки)
def visualize_graph():
    """
    Генерация визуализации графа в формате Mermaid.
    
    Полезно для документации и отладки.
    """
    workflow = build_agent_graph()
    compiled = workflow.compile()
    
    try:
        # LangGraph поддерживает экспорт в Mermaid
        mermaid = compiled.get_graph().draw_mermaid()
        return mermaid
    except Exception as e:
        print(f"Visualization not available: {e}")
        return None


# Простое ASCII представление
GRAPH_ASCII = """
Conference Paper Analysis Agent Graph
=====================================

    ┌─────────────────┐
    │     START       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  search_papers  │  ← ArXiv API query
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
┌──►│ download_paper  │  ← PDF download + cache
│   └────────┬────────┘
│            │
│            ▼
│   ┌─────────────────┐
│   │   parse_pdf     │  ← PyMuPDF text extraction
│   └────────┬────────┘
│            │
│            ▼
│   ┌─────────────────┐
│   │extract_affil.   │  ← LLM structured output
│   └────────┬────────┘
│            │
│            ▼
│   ┌─────────────────┐
│   │normalize_affil. │  ← KB + fuzzy + LLM
│   └────────┬────────┘
│            │
│            ▼
│   ┌─────────────────┐
│   │ more papers?    │
│   └────────┬────────┘
│      yes   │   no
│   ◄────────┘    │
│                 ▼
│        ┌─────────────────┐
│        │aggregate_results│  ← CSV + JSON report
│        └────────┬────────┘
│                 │
│                 ▼
│        ┌─────────────────┐
│        │      END        │
│        └─────────────────┘
│
└──── Loop until all papers processed
"""


def print_graph():
    """Вывод ASCII диаграммы графа"""
    print(GRAPH_ASCII)
