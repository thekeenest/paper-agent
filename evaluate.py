#!/usr/bin/env python3
"""
Скрипт оценки качества системы.

Использование:
    # Создать шаблон для ручной разметки
    python evaluate.py --create-template --papers "2401.12345,2401.12346"
    
    # Оценить качество на gold standard
    python evaluate.py --evaluate --csv output/affiliations_*.csv
    
    # Показать статистику gold standard
    python evaluate.py --stats
"""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation import (
    EvaluationEngine,
    GoldStandardDataset,
    create_gold_standard_template,
    load_predictions_from_csv
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation tool for Conference Paper Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create gold standard template for annotation"
    )
    
    parser.add_argument(
        "--papers",
        type=str,
        help="Comma-separated paper IDs for template"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on predictions"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to predictions CSV (supports glob patterns)"
    )
    
    parser.add_argument(
        "--gold-standard",
        type=str,
        default="./data/gold_standard.json",
        help="Path to gold standard dataset"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show gold standard statistics"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save evaluation report to JSON"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.create_template:
        if not args.papers:
            print("ERROR: --papers required for template creation")
            print("Example: --papers '2401.12345,2401.12346'")
            sys.exit(1)
        
        paper_ids = [p.strip() for p in args.papers.split(",")]
        template_path = args.gold_standard.replace(".json", "_template.json")
        create_gold_standard_template(paper_ids, template_path)
        return
    
    if args.stats:
        dataset = GoldStandardDataset(args.gold_standard)
        
        if not dataset.papers:
            print("No gold standard dataset found.")
            print(f"Create one first: python evaluate.py --create-template --papers 'id1,id2'")
            return
        
        stats = dataset.stats()
        print("\n" + "="*50)
        print("  GOLD STANDARD DATASET STATISTICS")
        print("="*50)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("="*50)
        return
    
    if args.evaluate:
        if not args.csv:
            print("ERROR: --csv required for evaluation")
            sys.exit(1)
        
        # Support glob patterns
        csv_files = glob.glob(args.csv)
        if not csv_files:
            print(f"ERROR: No files found matching {args.csv}")
            sys.exit(1)
        
        # Use the most recent file
        csv_path = max(csv_files, key=lambda p: Path(p).stat().st_mtime)
        print(f"Evaluating: {csv_path}")
        
        # Load predictions
        predictions = load_predictions_from_csv(csv_path)
        print(f"Loaded {len(predictions)} papers with {sum(len(p.authors) for p in predictions)} authors")
        
        # Initialize evaluator
        engine = EvaluationEngine(args.gold_standard)
        
        if not engine.gold_dataset.papers:
            print("\nWARNING: No gold standard dataset found!")
            print("Running evaluation without extraction metrics...")
            print("Create gold standard: python evaluate.py --create-template --papers '...'")
        
        # Run evaluation
        report = engine.evaluate_full(
            predictions=predictions,
            verbose=args.verbose
        )
        
        # Print report
        engine.print_report(report)
        
        # Save if requested
        if args.output:
            report.to_json(args.output)
            print(f"\nReport saved to: {args.output}")
        
        return
    
    # Default: show help
    print("Use --help to see available options")
    print("\nQuick start:")
    print("  1. Create template: python evaluate.py --create-template --papers 'id1,id2'")
    print("  2. Fill in gold_standard.json manually")
    print("  3. Evaluate: python evaluate.py --evaluate --csv output/affiliations_*.csv")


if __name__ == "__main__":
    main()
