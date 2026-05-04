"""
CLI entry-point for KG layer operations.

Sub-commands
------------
  ingest  — Ingest a JSONL file of WorkItem dicts into the KuzuDB KG.
  query   — Run one of the versioned graph queries.

Usage (via Makefile)
--------------------
::

    make kg-ingest INPUT=output/v2/work_items.jsonl
    make kg-query Q="industry_share_by_venue --venue NeurIPS --year 2024"

Usage (direct)
--------------
::

    python -m src.v2.kg.cli ingest --input output/v2/work_items.jsonl --db output/v2/kg
    python -m src.v2.kg.cli query --db output/v2/kg industry_share_by_venue --venue NeurIPS --year 2024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_ingest(args: argparse.Namespace) -> None:
    from src.v2.kg.ingest import KGIngestor
    from src.v2.kg.schema import KGSchema, open_db

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    db, conn = open_db(args.db)
    KGSchema.create_all(conn)
    ingestor = KGIngestor(conn)

    n_ingested = 0
    n_skipped = 0
    with open(input_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                work_item = json.loads(line)
                ingestor.ingest_verdict(work_item)
                n_ingested += 1
            except json.JSONDecodeError as exc:
                print(f"Warning: line {line_no} is not valid JSON: {exc}", file=sys.stderr)
                n_skipped += 1
            except Exception as exc:
                print(f"Warning: line {line_no} ingest failed: {exc}", file=sys.stderr)
                n_skipped += 1

    print(f"Ingested {n_ingested} work items ({n_skipped} skipped) into KG at {args.db}")
    print(f"  Paper nodes : {ingestor.node_count('Paper')}")
    print(f"  Author nodes: {ingestor.node_count('Author')}")
    print(f"  Institution : {ingestor.node_count('Institution')}")
    print(f"  Venue nodes : {ingestor.node_count('Venue')}")


def _cmd_query(args: argparse.Namespace) -> None:
    import re
    from src.v2.kg.schema import KGSchema, open_db

    db, conn = open_db(args.db)
    KGSchema.create_all(conn)

    query_name = args.query_name
    extra = args.extra_args  # list of remaining args

    # Parse common flags: --venue, --year, --top-k, --hops, --author-id
    extra_parser = argparse.ArgumentParser(prog=query_name, add_help=False)
    extra_parser.add_argument("--venue", default=None)
    extra_parser.add_argument("--year", type=int, default=None)
    extra_parser.add_argument("--top-k", type=int, default=20)
    extra_parser.add_argument("--hops", type=int, default=1)
    extra_parser.add_argument("--author-id", default=None)
    extra_parser.add_argument("--author-name", default=None)
    extra_parser.add_argument("--min-papers", type=int, default=1)
    qargs = extra_parser.parse_args(extra)

    if query_name == "industry_share_by_venue":
        from src.v2.kg.queries.industry_share_by_venue import industry_share_by_venue
        result = industry_share_by_venue(conn, venue=qargs.venue, year=qargs.year)
        print(json.dumps(result.to_dict(), indent=2))

    elif query_name == "country_strategy_heatmap":
        from src.v2.kg.queries.country_strategy_heatmap import country_strategy_heatmap
        result = country_strategy_heatmap(conn, year=qargs.year, min_papers=qargs.min_papers)
        print(json.dumps(result.to_dict(), indent=2))

    elif query_name == "top_institutions_by_venue":
        if not qargs.venue:
            print("Error: --venue is required for top_institutions_by_venue", file=sys.stderr)
            sys.exit(1)
        from src.v2.kg.queries.top_institutions_by_venue import top_institutions_by_venue
        result = top_institutions_by_venue(conn, venue=qargs.venue, year=qargs.year, top_k=qargs.top_k)
        print(json.dumps(result.to_dict(), indent=2))

    elif query_name == "coauthor_neighborhood":
        from src.v2.kg.queries.coauthor_neighborhood import coauthor_neighborhood
        result = coauthor_neighborhood(
            conn,
            author_id=qargs.author_id,
            author_name=qargs.author_name,
            hops=qargs.hops,
        )
        print(json.dumps(result.to_dict(), indent=2))

    else:
        print(f"Unknown query: {query_name!r}", file=sys.stderr)
        print("Available: industry_share_by_venue, country_strategy_heatmap, "
              "coauthor_neighborhood, top_institutions_by_venue", file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.v2.kg.cli",
        description="KG layer CLI for Paper-Agent v2.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── ingest ────────────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest WorkItem JSONL into KuzuDB KG")
    p_ingest.add_argument(
        "--input", required=True, help="Path to JSONL file of WorkItem dicts"
    )
    p_ingest.add_argument(
        "--db", default="output/v2/kg", help="Path to KuzuDB directory"
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    # ── query ─────────────────────────────────────────────────────────────────
    p_query = sub.add_parser("query", help="Run a versioned KG graph query")
    p_query.add_argument(
        "--db", default="output/v2/kg", help="Path to KuzuDB directory"
    )
    p_query.add_argument(
        "query_name",
        choices=[
            "industry_share_by_venue",
            "country_strategy_heatmap",
            "top_institutions_by_venue",
            "coauthor_neighborhood",
        ],
        help="Query to run",
    )
    p_query.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Query-specific flags (--venue, --year, --top-k, etc.)",
    )
    p_query.set_defaults(func=_cmd_query)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
