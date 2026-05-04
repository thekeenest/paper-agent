"""
KuzuDB schema for the Paper-Agent v2 knowledge graph.

Defined in DEV_PLAN.md §3.3. This module:
  1. Exposes Python-level type constants for all node/rel tables.
  2. Provides ``KGSchema.create_all(conn)`` which executes the DDL.
  3. Exports ``open_db(path)`` returning ``(Database, Connection)``.

Nodes
-----
  Author, Paper, Institution, Venue, Topic, Evidence

Relationships
-------------
  AUTHORED, AFFILIATED_AT, PUBLISHED_AT, ABOUT,
  CHILD_OF, COAUTHORED_WITH, COLLABORATED_WITH

Usage
-----
::

    from src.v2.kg.schema import KGSchema, open_db

    db, conn = open_db("output/v2/kg")
    KGSchema.create_all(conn)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog

_LOG = structlog.get_logger(__name__)

_DEFAULT_DB_PATH = Path(os.getenv("KG_DB_PATH", "output/v2/kg"))

# ─────────────────────────── DDL statements ──────────────────────────────────

_DDL: list[str] = [
    # Nodes
    """CREATE NODE TABLE IF NOT EXISTS Author (
        canonical_id  STRING,
        s2_id         STRING,
        orcid         STRING,
        name_variants STRING[],
        PRIMARY KEY (canonical_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Paper (
        doi           STRING,
        arxiv_id      STRING,
        openalex_id   STRING,
        s2_id         STRING,
        title         STRING,
        year          INT32,
        venue_key     STRING,
        abstract      STRING,
        primary_topic STRING,
        PRIMARY KEY (doi)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Institution (
        ror_id         STRING,
        openalex_id    STRING,
        canonical_name STRING,
        country_code   STRING,
        org_type       STRING,
        parent_ror_id  STRING,
        PRIMARY KEY (ror_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Venue (
        key       STRING,
        full_name STRING,
        kind      STRING,
        PRIMARY KEY (key)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Topic (
        topic_id STRING,
        label    STRING,
        source   STRING,
        PRIMARY KEY (topic_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Evidence (
        evidence_id  STRING,
        source       STRING,
        raw_payload  STRING,
        retrieved_at STRING,
        PRIMARY KEY (evidence_id)
    )""",
    # Relationships
    """CREATE REL TABLE IF NOT EXISTS AUTHORED (
        FROM Author TO Paper,
        position INT32
    )""",
    """CREATE REL TABLE IF NOT EXISTS AFFILIATED_AT (
        FROM Author TO Institution,
        paper_id    STRING,
        evidence_id STRING,
        year        INT32
    )""",
    """CREATE REL TABLE IF NOT EXISTS PUBLISHED_AT (
        FROM Paper TO Venue,
        year INT32
    )""",
    """CREATE REL TABLE IF NOT EXISTS ABOUT (
        FROM Paper TO Topic,
        weight FLOAT
    )""",
    """CREATE REL TABLE IF NOT EXISTS CHILD_OF (
        FROM Institution TO Institution
    )""",
    """CREATE REL TABLE IF NOT EXISTS COAUTHORED_WITH (
        FROM Author TO Author,
        paper_id STRING,
        year     INT32
    )""",
    """CREATE REL TABLE IF NOT EXISTS COLLABORATED_WITH (
        FROM Institution TO Institution,
        year         INT32,
        papers_count INT32
    )""",
]


# ─────────────────────────── schema manager ──────────────────────────────────


class KGSchema:
    """Execute DDL and provide table-name constants."""

    # Node table names
    AUTHOR = "Author"
    PAPER = "Paper"
    INSTITUTION = "Institution"
    VENUE = "Venue"
    TOPIC = "Topic"
    EVIDENCE = "Evidence"

    # Relationship table names
    AUTHORED = "AUTHORED"
    AFFILIATED_AT = "AFFILIATED_AT"
    PUBLISHED_AT = "PUBLISHED_AT"
    ABOUT = "ABOUT"
    CHILD_OF = "CHILD_OF"
    COAUTHORED_WITH = "COAUTHORED_WITH"
    COLLABORATED_WITH = "COLLABORATED_WITH"

    @staticmethod
    def create_all(conn: Any) -> None:
        """Execute all DDL statements against *conn*."""
        for stmt in _DDL:
            try:
                conn.execute(stmt)
            except Exception as exc:
                # Log but don't raise — some versions warn on IF NOT EXISTS
                _LOG.debug("kg.ddl_skip", stmt=stmt[:40], error=str(exc))
        _LOG.info("kg.schema_ready")


# ─────────────────────────── db factory ──────────────────────────────────────


def open_db(
    path: str | Path | None = None,
) -> tuple[Any, Any]:  # (kuzu.Database, kuzu.Connection)
    """Open (or create) a KuzuDB at *path* and return (db, conn).

    The schema is NOT created automatically — call ``KGSchema.create_all(conn)``
    after receiving the connection.
    """
    import kuzu

    resolved = Path(str(path or _DEFAULT_DB_PATH))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    db = kuzu.Database(str(resolved))
    conn = kuzu.Connection(db)
    _LOG.info("kg.db_opened", path=str(resolved))
    return db, conn
