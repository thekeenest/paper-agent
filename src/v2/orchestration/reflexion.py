"""
ReflexionStore — per-venue verbal memory for the Planner.

Architecture
------------
A single SQLite table keyed on (venue_key, year, layout_hash) stores free-text
summaries (≤ 500 tokens) of common extraction errors observed at that key.
Entries are injected into the Planner's prompt before each run.

After every paper, the Reflector sub-prompt compares the Critic's accepted
Candidates against:
  * gold annotations (if the paper is a benchmark paper), or
  * the no-Critic baseline otherwise.

The resulting diff summary is upserted into the store for the matching key.

Write idempotency
-----------------
Multiple runs over the same (paper_id, venue_key, year) are deduplicated via a
``seen_paper_ids`` JSON column — a paper's contribution is only counted once.

CLI
---
::

    python -m src.v2.orchestration.reflexion show --venue NeurIPS --year 2023
    python -m src.v2.orchestration.reflexion list
    python -m src.v2.orchestration.reflexion reset --venue NeurIPS --year 2023

Usage
-----
::

    store = ReflexionStore()
    await store.reflect(paper, accepted_candidates, baseline_candidates)
    memory = await store.fetch(venue="NeurIPS", year=2023)
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import structlog

_LOG = structlog.get_logger(__name__)

_DEFAULT_DB = Path(os.getenv("REFLEXION_DB", "output/v2/reflexion.db"))
_MAX_TOKENS = 500  # hard cap per entry (approximate: ~4 chars/token)
_MAX_CHARS = _MAX_TOKENS * 4

_SUMMARISER_MODEL = os.getenv("REFLEXION_MODEL", "claude-haiku-4-5-20251001")


# ─────────────────────────── store ──────────────────────────────────────────


class ReflexionStore:
    """SQLite-backed per-venue verbal memory store.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.  Created on first use.
    """

    _CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS reflexion_entries (
        venue_key    TEXT    NOT NULL,
        year         INTEGER,
        layout_hash  TEXT,
        summary      TEXT    NOT NULL DEFAULT '',
        updated_at   TEXT    NOT NULL,
        seen_paper_ids TEXT  NOT NULL DEFAULT '[]',
        PRIMARY KEY (venue_key, year, layout_hash)
    );
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        raw = db_path or _DEFAULT_DB
        self._path = Path(str(raw))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._CREATE_SQL)
        self._conn.commit()

    # ── public read ──────────────────────────────────────────────────────────

    async def fetch(
        self,
        venue: str,
        year: int | None = None,
        layout_hash: str | None = None,
    ) -> str | None:
        """Return the stored summary for (venue, year, layout_hash), or None."""
        t0 = time.perf_counter()
        vk = _venue_key(venue)
        lh = layout_hash or ""
        row = self._conn.execute(
            "SELECT summary FROM reflexion_entries "
            "WHERE venue_key=? AND (year IS NULL OR year=?) "
            "AND layout_hash=? "
            "ORDER BY updated_at DESC LIMIT 1",
            (vk, year, lh),
        ).fetchone()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if row and row["summary"]:
            _LOG.debug("reflexion.fetch_hit", venue=vk, year=year, elapsed_ms=round(elapsed_ms, 1))
            return str(row["summary"])
        return None

    def fetch_sync(self, venue: str, year: int | None = None, layout_hash: str | None = None) -> str | None:
        """Synchronous variant for CLI use."""
        vk = _venue_key(venue)
        lh = layout_hash or ""
        row = self._conn.execute(
            "SELECT summary FROM reflexion_entries "
            "WHERE venue_key=? AND (year IS NULL OR year=?) "
            "AND layout_hash=? "
            "ORDER BY updated_at DESC LIMIT 1",
            (vk, year, lh),
        ).fetchone()
        return str(row["summary"]) if row and row["summary"] else None

    def list_entries(self) -> list[dict[str, Any]]:
        """Return all stored entries as dicts (for CLI / debugging)."""
        rows = self._conn.execute(
            "SELECT venue_key, year, layout_hash, summary, updated_at FROM reflexion_entries "
            "ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_entry(self, venue: str, year: int | None = None) -> int:
        """Delete entry for (venue, year). Returns number of rows deleted."""
        vk = _venue_key(venue)
        cursor = self._conn.execute(
            "DELETE FROM reflexion_entries WHERE venue_key=? AND (year IS NULL OR year=?)",
            (vk, year),
        )
        self._conn.commit()
        return cursor.rowcount

    # ── public write ─────────────────────────────────────────────────────────

    async def reflect(
        self,
        paper_id: str,
        venue: str,
        year: int | None,
        layout_hash: str | None,
        accepted_candidates: list[dict[str, Any]],
        baseline_candidates: list[dict[str, Any]],
        gold_candidates: list[dict[str, Any]] | None = None,
    ) -> None:
        """Generate and upsert a reflexion summary for (venue, year, layout_hash).

        Parameters
        ----------
        paper_id:
            Unique paper identifier; used for idempotency (won't double-count).
        accepted_candidates:
            Candidates accepted by the Critic.
        baseline_candidates:
            Candidates from the no-Critic merge (all merged_candidates).
        gold_candidates:
            Gold annotations, if available (benchmark papers only).
        """
        vk = _venue_key(venue)

        lh = layout_hash or ""

        # Idempotency: check if this paper was already reflected
        row = self._conn.execute(
            "SELECT seen_paper_ids, summary FROM reflexion_entries "
            "WHERE venue_key=? AND year IS ? AND layout_hash=?",
            (vk, year, lh),
        ).fetchone()

        seen: list[str] = json.loads(row["seen_paper_ids"]) if row else []
        if paper_id in seen:
            _LOG.debug("reflexion.skip_duplicate", paper_id=paper_id, venue=vk)
            return

        # Summarise the diff
        existing_summary = row["summary"] if row else ""
        new_summary = await _summarise(
            venue=venue,
            year=year,
            accepted_names=[c.get("author_name", "") for c in accepted_candidates],
            baseline_names=[c.get("author_name", "") for c in baseline_candidates],
            gold_names=[c.get("author_name", "") for c in gold_candidates] if gold_candidates else None,
            existing_summary=existing_summary,
        )

        seen.append(paper_id)
        from datetime import UTC, datetime
        now = datetime.now(UTC).isoformat()

        self._conn.execute(
            """INSERT INTO reflexion_entries
               (venue_key, year, layout_hash, summary, updated_at, seen_paper_ids)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(venue_key, year, layout_hash) DO UPDATE SET
               summary=excluded.summary,
               updated_at=excluded.updated_at,
               seen_paper_ids=excluded.seen_paper_ids
            """,
            (vk, year, lh, new_summary[:_MAX_CHARS], now, json.dumps(seen)),
        )
        self._conn.commit()
        _LOG.info("reflexion.upserted", venue=vk, year=year, summary_len=len(new_summary))


# ─────────────────────────── reflector LLM ──────────────────────────────────


async def _summarise(
    venue: str,
    year: int | None,
    accepted_names: list[str],
    baseline_names: list[str],
    gold_names: list[str] | None,
    existing_summary: str,
) -> str:
    """Call a cheap LLM to generate a ≤500-token reflexion summary."""
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatAnthropic(model_name=_SUMMARISER_MODEL, temperature=0.0, max_tokens=600)  # type: ignore[call-arg]

        rejected = [n for n in baseline_names if n not in accepted_names]
        added_by_critic = [n for n in accepted_names if n not in baseline_names]

        if gold_names is not None:
            missed = [g for g in gold_names if not any(
                g.lower() in a.lower() or a.lower() in g.lower() for a in accepted_names
            )]
            false_accepts = [a for a in accepted_names if not any(
                a.lower() in g.lower() or g.lower() in a.lower() for g in gold_names
            )]
            diff_note = (
                f"Missed gold authors: {missed or 'none'}. "
                f"False accepts: {false_accepts or 'none'}."
            )
        else:
            diff_note = (
                f"Candidates rejected by Critic vs baseline: {rejected or 'none'}. "
                f"Candidates added by Critic: {added_by_critic or 'none'}."
            )

        system = (
            "You are summarising extraction errors for a scholarly author-affiliation pipeline. "
            "Produce a concise ≤150-word note about recurring patterns for the given venue/year. "
            "Focus on actionable guidance (e.g., 'check footnote 3', 'prefer footnote over header'). "
            "If the existing summary is non-empty, UPDATE it rather than replacing it."
        )
        user = (
            f"Venue: {venue}, Year: {year or 'unknown'}\n"
            f"Observation: {diff_note}\n"
            f"Existing memory: {existing_summary or '(none)'}\n\n"
            "Update or create the memory note (≤150 words)."
        )
        resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
        content = resp.content if hasattr(resp, "content") else str(resp)
        return str(content)[:_MAX_CHARS]
    except Exception as exc:
        _LOG.warning("reflexion.summarise_failed", error=str(exc))
        # Fallback: rule-based summary without LLM
        return _rule_based_summary(venue, year, accepted_names, baseline_names)


def _rule_based_summary(
    venue: str,
    year: int | None,
    accepted: list[str],
    baseline: list[str],
) -> str:
    rejected = [n for n in baseline if n not in accepted]
    summary = f"[{venue} {year or ''}] "
    if rejected:
        summary += f"Critic rejected {len(rejected)} candidate(s): {', '.join(rejected[:5])}. "
    if len(accepted) < len(baseline):
        summary += "Prefer multi-specialist agreement for confidence."
    return summary[:_MAX_CHARS]


# ─────────────────────────── helpers ────────────────────────────────────────


def _venue_key(venue: str) -> str:
    """Normalise venue name to a stable lowercase key."""
    return re.sub(r"\s+", "_", venue.strip().lower()) or "unknown"


# ─────────────────────────── CLI ────────────────────────────────────────────


def _cli_show(args: argparse.Namespace) -> None:
    store = ReflexionStore()
    entry = store.fetch_sync(venue=args.venue, year=args.year)
    if entry:
        print(f"[{args.venue} {args.year}]\n{entry}")
    else:
        print(f"No reflexion entry for venue={args.venue!r} year={args.year}")


def _cli_list(_args: argparse.Namespace) -> None:
    store = ReflexionStore()
    entries = store.list_entries()
    if not entries:
        print("(empty store)")
        return
    for e in entries:
        print(f"{e['venue_key']} / {e['year']} / {e['layout_hash'] or '*'}")
        print(f"  updated: {e['updated_at']}")
        print(f"  {e['summary'][:120]}...")
        print()


def _cli_reset(args: argparse.Namespace) -> None:
    store = ReflexionStore()
    n = store.delete_entry(venue=args.venue, year=args.year)
    print(f"Deleted {n} entry/entries for venue={args.venue!r} year={args.year}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.v2.orchestration.reflexion",
        description="Manage the Reflexion per-venue memory store.",
    )
    sub = parser.add_subparsers(dest="command")

    show_p = sub.add_parser("show", help="Print memory entry for a venue/year")
    show_p.add_argument("--venue", required=True)
    show_p.add_argument("--year", type=int, default=None)
    show_p.set_defaults(func=_cli_show)

    list_p = sub.add_parser("list", help="List all stored entries")
    list_p.set_defaults(func=_cli_list)

    reset_p = sub.add_parser("reset", help="Delete entry for a venue/year")
    reset_p.add_argument("--venue", required=True)
    reset_p.add_argument("--year", type=int, default=None)
    reset_p.set_defaults(func=_cli_reset)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
