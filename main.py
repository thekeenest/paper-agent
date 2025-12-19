#!/usr/bin/env python3
"""
Conference Paper Agent - Main Entry Point

Мульти-агентная система для анализа публикаций научных конференций.

Использование:
    python main.py --query "cat:cs.AI" --max-papers 100
    python main.py --query "cat:cs.LG" --date-from 20240101 --date-to 20240131

Примеры запросов ArXiv:
    - cat:cs.AI                     # Artificial Intelligence
    - cat:cs.LG                     # Machine Learning
    - cat:cs.CV                     # Computer Vision
    - cat:cs.CL                     # Computation and Language (NLP)
    - cat:cs.AI AND au:bengio       # AI papers by Bengio
    - ti:transformer                # Papers with "transformer" in title
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Проверка API ключа
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in environment variables")
    print("Create a .env file with OPENAI_API_KEY=sk-...")
    sys.exit(1)

from src.graph import create_app, print_graph
from src.state import create_initial_state
from src.analytics import AnalyticsEngine


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Conference Paper Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        default="cat:cs.AI",
        help="ArXiv search query (default: cat:cs.AI)"
    )
    
    parser.add_argument(
        "--max-papers", "-n",
        type=int,
        default=10,
        help="Maximum number of papers to process (default: 10)"
    )
    
    parser.add_argument(
        "--date-from",
        type=str,
        help="Start date filter (YYYYMMDD)"
    )
    
    parser.add_argument(
        "--date-to",
        type=str,
        help="End date filter (YYYYMMDD)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    parser.add_argument(
        "--show-graph",
        action="store_true",
        help="Print the agent graph structure"
    )
    
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating visualization plots"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        choices=["arxiv", "semantic_scholar", "openalex"],
        default="arxiv",
        help="Data source to use (default: arxiv)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def main():
    """Главная функция запуска агента"""
    args = parse_args()
    
    # Показать структуру графа
    if args.show_graph:
        print_graph()
        return
    
    print("\n" + "="*60)
    print("  CONFERENCE PAPER ANALYSIS AGENT")
    print("="*60)
    print(f"  Query: {args.query}")
    print(f"  Data source: {args.source}")
    print(f"  Max papers: {args.max_papers}")
    if args.date_from:
        print(f"  Date from: {args.date_from}")
    if args.date_to:
        print(f"  Date to: {args.date_to}")
    print(f"  Output: {args.output_dir}")
    print("="*60 + "\n")
    
    # Создание приложения
    app = create_app()
    
    # Начальное состояние
    initial_state = create_initial_state(
        query=args.query,
        max_papers=args.max_papers,
        date_from=args.date_from,
        date_to=args.date_to,
        max_retries=3,
        data_source=args.source
    )
    
    # Запуск агента
    print("Starting agent processing...\n")
    start_time = time.time()
    
    # Рассчитываем recursion_limit: ~5 шагов на статью + запас
    # (search -> download -> parse -> extract -> normalize) * N papers + aggregate
    recursion_limit = args.max_papers * 6 + 20
    
    try:
        result = app.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit}
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    elapsed_time = time.time() - start_time
    
    # Вывод результатов
    print("\n" + "="*60)
    print("  PROCESSING COMPLETE")
    print("="*60)
    print(f"  Time elapsed: {elapsed_time:.1f} seconds")
    print(f"  Papers processed: {result.get('processed_count', 0)}")
    print(f"  Errors: {result.get('error_count', 0)}")
    
    if result.get("output_path"):
        print(f"  Results saved to: {result['output_path']}")
    
    # Генерация графиков
    if not args.no_plots and result.get("papers"):
        print("\nGenerating visualizations...")
        
        try:
            engine = AnalyticsEngine(args.output_dir)
            engine.load_from_papers(result["papers"])
            
            # Сводная статистика
            stats = engine.get_summary_stats()
            print(f"\n  Total authors: {stats['total_authors']}")
            print(f"  Unique organizations: {stats['unique_organizations']}")
            print(f"  Unique countries: {stats['unique_countries']}")
            
            # Графики
            paths = engine.generate_all_plots()
            print(f"\n  Generated {len(paths)} plots")
            
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Вывод топ организаций
    if result.get("final_report"):
        report = result["final_report"]
        if hasattr(report, "top_organizations") and report.top_organizations:
            print("TOP 10 ORGANIZATIONS:")
            print("-"*40)
            for i, org in enumerate(report.top_organizations[:10], 1):
                print(f"  {i:2d}. {org['organization']}: {org['count']} authors")
            print()


if __name__ == "__main__":
    main()
