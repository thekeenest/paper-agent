"""
v2 CLI entry point.

Usage
-----
::

    python -m src.v2.cli --query "cat:cs.AI" --n 1
    python -m src.v2.cli --query "cat:cs.AI" --n 1 --resume

    # via Makefile:
    make v2-cli QUERY="cat:cs.AI" N=1
    make v2-cli QUERY="cat:cs.AI" N=1 RESUME=1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import structlog

_LOG = structlog.get_logger(__name__)


def _configure_logging(verbose: bool = False) -> None:
    import logging
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if verbose else logging.INFO
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


async def _run(query: str, n: int, output_dir: Path, resume: bool) -> list[dict[str, object]]:
    from src.v2.orchestration.contracts import WorkItem
    from src.v2.orchestration.coordinator import build_graph

    output_dir.mkdir(parents=True, exist_ok=True)
    graph = await build_graph(use_checkpointer=True)

    results = []
    for i in range(n):
        work_id = f"run_{uuid.uuid4().hex[:8]}"
        thread_id = f"{query.replace(' ', '_')}_{i}" if resume else work_id

        config = {"configurable": {"thread_id": thread_id}}

        # If resuming, try to fetch existing state
        if resume:
            state = await graph.aget_state(config)
            if state and state.values:
                existing = state.values
                if isinstance(existing, dict) and existing.get("status") == "complete":
                    _LOG.info("cli.resume_skip", thread_id=thread_id, reason="already complete")
                    results.append(existing)
                    continue
                _LOG.info("cli.resume_continue", thread_id=thread_id)

        work_item = WorkItem(work_id=work_id, query=query)
        _LOG.info("cli.start", work_id=work_id, query=query)

        try:
            final_state = await graph.ainvoke(work_item, config=config)
        except Exception as exc:
            _LOG.error("cli.pipeline_error", error=str(exc))
            final_state = work_item.model_dump(mode="json")
            final_state["status"] = "failed"
            final_state["error"] = str(exc)

        if isinstance(final_state, dict):
            out_data = final_state
        else:
            out_data = final_state.model_dump(mode="json")

        # Write per-paper JSON
        out_file = output_dir / f"{out_data.get('work_id', work_id)}.json"
        out_file.write_text(json.dumps(out_data, indent=2, default=str))
        _LOG.info("cli.wrote", path=str(out_file))
        results.append(out_data)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-Agent v2 CLI")
    parser.add_argument("--query", required=True, help="ArXiv query or paper ID")
    parser.add_argument("--n", type=int, default=1, help="Number of papers")
    parser.add_argument(
        "--output", default="output/v2", help="Output directory"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume prior run (skip completed nodes)"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)

    results = asyncio.run(
        _run(
            query=args.query,
            n=args.n,
            output_dir=Path(args.output),
            resume=args.resume,
        )
    )

    for r in results:
        print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
