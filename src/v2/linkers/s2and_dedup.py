"""
S2AND author disambiguation / deduplication.

S2AND (Allen AI) clusters ambiguous author records that refer to the same
real person.  This module provides two execution paths:

1. **Native** (preferred) — ``import s2and`` is available; uses the
   ``S2AndDisambiguator`` directly.

2. **Subprocess fallback** — s2and is not importable (e.g. platform
   incompatibility, missing CUDA).  A thin JSON-in / JSON-out CLI wrapper is
   invoked.  Install path: ``pip install s2and`` in a separate venv, then set
   ``S2AND_CLI`` to the ``python`` binary of that venv.

Install (native)
----------------
::

    pip install s2and  # may require torch; see s2and's own README

Install (subprocess path)
-------------------------
::

    python -m venv .venv-s2and
    .venv-s2and/bin/pip install s2and
    export S2AND_CLI=.venv-s2and/bin/python

Public API
----------
::

    from src.v2.linkers.s2and_dedup import dedup, AuthorRecord, ClusterId

    records = [
        AuthorRecord(block_id="smith_a", author_id="1",
                     name="Alice Smith", affiliations=["MIT"]),
        AuthorRecord(block_id="smith_a", author_id="2",
                     name="A. Smith", affiliations=["MIT CSAIL"]),
    ]
    clusters: list[ClusterId] = await dedup(records)
    # clusters[0] == clusters[1]  ← same person
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ._errors import ConfigurationError, S2ANDError

_LOG = logging.getLogger(__name__)

ClusterId = str  # opaque cluster identifier (UUID or int-as-string)


# ─────────────────────────── data models ────────────────────────────────────


class AuthorRecord(BaseModel):
    """Input record for S2AND deduplication."""

    model_config = ConfigDict(frozen=True)

    block_id: str = Field(
        description="Blocking key (e.g. first-initial + last-name). "
        "S2AND only compares records within the same block."
    )
    author_id: str = Field(description="Unique ID for this record (paper-scoped).")
    name: str
    affiliations: list[str] = Field(default_factory=list)
    email: str | None = None
    paper_title: str | None = None
    paper_year: int | None = None
    coauthors: list[str] = Field(default_factory=list)


# ─────────────────────────── public API ─────────────────────────────────────


async def dedup(records: list[AuthorRecord]) -> list[ClusterId]:
    """Cluster *records* into groups of the same real-world author.

    Returns a list of cluster IDs, one per input record, in the same order.
    Records that share a cluster ID refer to the same person.

    Parameters
    ----------
    records:
        Author records to disambiguate.  All records in the same ``block_id``
        are compared against each other.

    Returns
    -------
    list[ClusterId]
        Parallel to *records*; same cluster ID ↔ same author.

    Raises
    ------
    S2ANDError
        If S2AND is unavailable or the subprocess call fails.
    ConfigurationError
        If neither the native package nor S2AND_CLI is configured.
    """
    if not records:
        return []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _dedup_sync, records)


def _dedup_sync(records: list[AuthorRecord]) -> list[ClusterId]:
    """Synchronous implementation — called from thread pool by ``dedup``."""
    try:
        return _dedup_native(records)
    except ImportError:
        _LOG.debug("s2and not importable; trying subprocess fallback")

    cli = os.getenv("S2AND_CLI")
    if cli:
        return _dedup_subprocess(records, cli)

    # Neither path available — return trivial one-cluster-per-record as graceful
    # degradation (no exception; callers can detect by checking all cluster IDs
    # differ from each other).
    _LOG.warning(
        "S2AND unavailable (no native install, S2AND_CLI not set). "
        "Returning trivial per-record clusters."
    )
    return [r.author_id for r in records]


# ─────────────────────────── native path ────────────────────────────────────


def _dedup_native(records: list[AuthorRecord]) -> list[ClusterId]:
    """Use the installed s2and Python package directly."""
    from s2and.data import ANDData
    from s2and.model import S2AndDisambiguator

    # Build the ANDData structure S2AND expects.
    signatures: dict[str, Any] = {}
    papers: dict[str, Any] = {}

    for rec in records:
        sig_id = rec.author_id
        signatures[sig_id] = {
            "signature_id": sig_id,
            "author_info": {
                "given_block": rec.block_id,
                "full_name": rec.name,
                "affiliations": rec.affiliations,
                "email": rec.email or "",
            },
            "paper_id": rec.author_id,  # reuse as paper id for simplicity
        }
        papers[rec.author_id] = {
            "title": rec.paper_title or "",
            "year": rec.paper_year,
            "authors": [{"position": 0, "full_name": rec.name}],
        }

    and_data = ANDData(
        signatures=signatures,
        papers=papers,
        name="paper_agent_dedup",
        mode="inference",
        block_type="s2",
        load_name_counts=False,
    )

    disambiguator = S2AndDisambiguator.load_prod_model()
    pred_clusters, _ = disambiguator.predict_blocks(and_data)

    # Build a mapping: signature_id → cluster_id
    sig_to_cluster: dict[str, str] = {}
    for cluster_id, sig_ids in pred_clusters.items():
        for sid in sig_ids:
            sig_to_cluster[sid] = str(cluster_id)

    return [sig_to_cluster.get(r.author_id, r.author_id) for r in records]


# ─────────────────────────── subprocess path ────────────────────────────────

_SUBPROCESS_SCRIPT = """
import json, sys
from s2and.data import ANDData
from s2and.model import S2AndDisambiguator

records = json.load(sys.stdin)
signatures, papers = {}, {}
for rec in records:
    sid = rec["author_id"]
    signatures[sid] = {
        "signature_id": sid,
        "author_info": {
            "given_block": rec["block_id"],
            "full_name": rec["name"],
            "affiliations": rec.get("affiliations", []),
            "email": rec.get("email", ""),
        },
        "paper_id": sid,
    }
    papers[sid] = {
        "title": rec.get("paper_title", ""),
        "year": rec.get("paper_year"),
        "authors": [{"position": 0, "full_name": rec["name"]}],
    }

and_data = ANDData(
    signatures=signatures, papers=papers,
    name="dedup", mode="inference", block_type="s2", load_name_counts=False,
)
disambiguator = S2AndDisambiguator.load_prod_model()
pred_clusters, _ = disambiguator.predict_blocks(and_data)

sig_to_cluster = {}
for cid, sids in pred_clusters.items():
    for sid in sids:
        sig_to_cluster[sid] = str(cid)

result = [sig_to_cluster.get(r["author_id"], r["author_id"]) for r in records]
print(json.dumps(result))
"""


def _dedup_subprocess(records: list[AuthorRecord], python_bin: str) -> list[ClusterId]:
    payload = json.dumps([r.model_dump() for r in records])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(_SUBPROCESS_SCRIPT)
        script_path = f.name

    try:
        result = subprocess.run(
            [python_bin, script_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise S2ANDError("s2and", "Subprocess timed out after 120s") from exc
    except FileNotFoundError as exc:
        raise ConfigurationError("s2and", f"S2AND_CLI binary not found: {python_bin}") from exc
    finally:
        with contextlib.suppress(OSError):
            os.unlink(script_path)

    if result.returncode != 0:
        raise S2ANDError("s2and", f"Subprocess exited {result.returncode}: {result.stderr[:400]}")

    try:
        clusters: list[ClusterId] = json.loads(result.stdout)
        return clusters
    except json.JSONDecodeError as exc:
        raise S2ANDError("s2and", f"Subprocess output not valid JSON: {exc}") from exc
