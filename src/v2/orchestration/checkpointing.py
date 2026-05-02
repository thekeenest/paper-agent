"""
Async SQLite-backed LangGraph checkpointer for resumable pipeline runs.

Usage
-----
::

    from src.v2.orchestration.checkpointing import make_checkpointer

    checkpointer = await make_checkpointer()          # uses default path
    # or
    checkpointer = await make_checkpointer("/tmp/my.db")

    # Pass to StateGraph.compile():
    graph = coordinator.compile(checkpointer=checkpointer)

    # Resume a prior run:
    config = {"configurable": {"thread_id": work_id}}
    result = await graph.ainvoke(work_item, config=config)

Checkpoint path
---------------
Default: ``output/v2/checkpoints.db`` (created if absent).
Override via ``LANGGRAPH_CHECKPOINT_DB`` env var.
"""
from __future__ import annotations

import os
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

_DEFAULT_DB = Path("output/v2/checkpoints.db")


async def make_checkpointer(db_path: str | Path | None = None) -> AsyncSqliteSaver:
    """Create and return an async SQLite-backed LangGraph checkpointer.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created if it does not exist.
        Falls back to ``LANGGRAPH_CHECKPOINT_DB`` env var, then
        ``output/v2/checkpoints.db``.
    """
    raw = db_path or os.getenv("LANGGRAPH_CHECKPOINT_DB", str(_DEFAULT_DB))
    resolved = Path(str(raw))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(resolved))
    return AsyncSqliteSaver(conn)
