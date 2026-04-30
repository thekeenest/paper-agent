#!/usr/bin/env python3
"""
Paper-Agent v1 CLI — explicit src.v1 entry point.

Usage (as module):
    python -m src.v1.main --query "cat:cs.AI" --max-papers 10

Delegates to the root main.py logic but imports from src.v1.* directly so
this module can be invoked even when the src-level shims are being replaced.
"""

import os
import sys
import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in environment variables")
    print("Create a .env file with OPENAI_API_KEY=sk-...")
    sys.exit(1)

from src.v1.graph import create_app, print_graph  # type: ignore[attr-defined]
from src.v1.state import create_initial_state
from src.v1.analytics import AnalyticsEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conference Paper Analysis Agent (v1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--query", "-q", type=str, default="cat:cs.AI")
    parser.add_argument("--max-papers", "-n", type=int, default=10)
    parser.add_argument("--date-from", type=str)
    parser.add_argument("--date-to", type=str)
    parser.add_argument("--output-dir", "-o", type=str, default="./output")
    parser.add_argument("--show-graph", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["arxiv", "semantic_scholar", "openalex"],
        default="arxiv",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.show_graph:
        print_graph()
        return

    print("\n" + "=" * 60)
    print("  CONFERENCE PAPER ANALYSIS AGENT  [v1]")
    print("=" * 60)
    print(f"  Query: {args.query}")
    print(f"  Data source: {args.source}")
    print(f"  Max papers: {args.max_papers}")
    if args.date_from:
        print(f"  Date from: {args.date_from}")
    if args.date_to:
        print(f"  Date to: {args.date_to}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60 + "\n")

    app = create_app()
    initial_state = create_initial_state(
        query=args.query,
        max_papers=args.max_papers,
        date_from=args.date_from,
        date_to=args.date_to,
        max_retries=3,
        data_source=args.source,
    )

    print("Starting agent processing...\n")
    start_time = time.time()
    recursion_limit = args.max_papers * 6 + 20

    try:
        result = app.invoke(
            initial_state,
            config={"recursion_limit": recursion_limit},
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

    print("\n" + "=" * 60)
    print("  PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Time elapsed: {elapsed_time:.1f} seconds")
    print(f"  Papers processed: {result.get('processed_count', 0)}")
    print(f"  Errors: {result.get('error_count', 0)}")

    if result.get("output_path"):
        print(f"  Results saved to: {result['output_path']}")

    if not args.no_plots and result.get("papers"):
        print("\nGenerating visualizations...")
        try:
            engine = AnalyticsEngine(args.output_dir)
            engine.load_from_papers(result["papers"])
            stats = engine.get_summary_stats()
            print(f"\n  Total authors: {stats['total_authors']}")
            print(f"  Unique organizations: {stats['unique_organizations']}")
            print(f"  Unique countries: {stats['unique_countries']}")
            paths = engine.generate_all_plots()
            print(f"\n  Generated {len(paths)} plots")
        except Exception as e:
            print(f"  Warning: Could not generate plots: {e}")

    print("\n" + "=" * 60 + "\n")

    if result.get("final_report"):
        report = result["final_report"]
        if hasattr(report, "top_organizations") and report.top_organizations:
            print("TOP 10 ORGANIZATIONS:")
            print("-" * 40)
            for i, org in enumerate(report.top_organizations[:10], 1):
                print(f"  {i:2d}. {org['organization']}: {org['count']} authors")
            print()


if __name__ == "__main__":
    main()
