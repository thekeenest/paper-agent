"""
Idempotent KG ingestor for Paper-Agent v2.

``KGIngestor.ingest_verdict()`` converts a pipeline WorkItem + its Verdicts into
node/edge upserts.  Re-ingesting the same paper produces identical graph state.

Canonical keys
--------------
  Author      → S2AND cluster ID (``s2_cluster:<id>``) or ``name:<norm_name>`` fallback
  Institution → ROR ID (``ror:<id>``) or ``name:<norm_name>`` fallback
  Paper       → DOI if present, else ``arxiv:<id>``, else ``openalex:<id>``
  Venue       → venue_key (lowercased name)
  Evidence    → evidence_id from ToolEvidence

Usage
-----
::

    from src.v2.kg.schema import KGSchema, open_db
    from src.v2.kg.ingest import KGIngestor

    db, conn = open_db("output/v2/kg")
    KGSchema.create_all(conn)
    ingestor = KGIngestor(conn)
    ingestor.ingest_verdict(work_item_dict)
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

import structlog

_LOG = structlog.get_logger(__name__)


# ─────────────────────────── ingestor ────────────────────────────────────────


class KGIngestor:
    """Idempotent upsert of pipeline outputs into the KuzuDB KG.

    Parameters
    ----------
    conn:
        An open ``kuzu.Connection`` on a database where
        ``KGSchema.create_all`` has already been called.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # ── public entry point ────────────────────────────────────────────────────

    def ingest_verdict(self, work_item: dict[str, Any]) -> None:
        """Ingest all facts from a completed WorkItem dict.

        Idempotent: calling twice with the same paper_id produces identical
        graph state.

        Parameters
        ----------
        work_item:
            The JSON-serialised WorkItem (as from ``WorkItem.model_dump(mode='json')``).
        """
        paper_data = work_item.get("canonical_paper") or {}
        verdicts: list[dict[str, Any]] = work_item.get("verdicts") or []
        merged: list[dict[str, Any]] = work_item.get("merged_candidates") or []

        if not paper_data:
            _LOG.debug("kg.ingest_skip", reason="no canonical_paper")
            return

        paper_key = _paper_key(paper_data)
        year: int | None = paper_data.get("year")
        venue_key = _venue_key(paper_data.get("venue") or "")

        # 1. Upsert Paper node
        self._upsert_paper(paper_key, paper_data)

        # 2. Upsert Venue node + PUBLISHED_AT edge
        if venue_key:
            self._upsert_venue(venue_key, paper_data.get("venue") or "")
            self._upsert_edge_published_at(paper_key, venue_key, year)

        # 3. Build candidate lookup
        cand_map: dict[str, dict[str, Any]] = {
            c["candidate_id"]: c for c in merged if "candidate_id" in c
        }

        # 4. Process each accepted verdict
        accepted_author_ids: list[str] = []
        for verdict in verdicts:
            if verdict.get("decision") != "accept":
                continue
            cand = cand_map.get(verdict.get("candidate_id", ""))
            if not cand:
                continue

            author_id = self._ingest_author_affiliation(
                cand=cand,
                verdict=verdict,
                paper_key=paper_key,
                year=year,
            )
            if author_id:
                accepted_author_ids.append(author_id)

        # 5. COAUTHORED_WITH edges for all accepted author pairs
        for i, aid_i in enumerate(accepted_author_ids):
            for aid_j in accepted_author_ids[i + 1:]:
                self._upsert_edge_coauthored_with(aid_i, aid_j, paper_key, year)

        # 6. COLLABORATED_WITH edges between institutions
        self._update_collaborated_with(cand_map, verdicts, year)

        _LOG.info(
            "kg.ingested",
            paper=paper_key,
            n_accepted=len(accepted_author_ids),
        )

    def node_count(self, table: str) -> int:
        """Return the count of nodes in *table* (for testing)."""
        res = self._conn.execute(f"MATCH (n:{table}) RETURN count(*)")
        if res.has_next():
            return int(res.get_next()[0])
        return 0

    def edge_count(self, rel: str) -> int:
        """Return the count of edges in rel table *rel* (for testing)."""
        res = self._conn.execute(f"MATCH ()-[r:{rel}]->() RETURN count(*)")
        if res.has_next():
            return int(res.get_next()[0])
        return 0

    # ── private helpers ───────────────────────────────────────────────────────

    def _upsert_paper(self, doi: str, data: dict[str, Any]) -> None:
        self._conn.execute(
            """MERGE (p:Paper {doi: $doi})
               SET p.arxiv_id      = $arxiv_id,
                   p.openalex_id   = $openalex_id,
                   p.title         = $title,
                   p.year          = $year,
                   p.venue_key     = $venue_key,
                   p.abstract      = $abstract,
                   p.primary_topic = $primary_topic
            """,
            parameters={
                "doi": doi,
                "arxiv_id": _str(data.get("paper_id") if _is_arxiv(data) else ""),
                "openalex_id": _str(data.get("paper_id") if _is_openalex(data) else ""),
                "title": _str(data.get("title")),
                "year": int(data["year"]) if data.get("year") else 0,
                "venue_key": _venue_key(data.get("venue") or ""),
                "abstract": _str(data.get("abstract"))[:2000],
                "primary_topic": "",
            },
        )

    def _upsert_venue(self, key: str, full_name: str) -> None:
        self._conn.execute(
            """MERGE (v:Venue {key: $key})
               SET v.full_name = $full_name,
                   v.kind      = $kind
            """,
            parameters={"key": key, "full_name": full_name, "kind": "conference"},
        )

    def _upsert_edge_published_at(
        self, paper_doi: str, venue_key: str, year: int | None
    ) -> None:
        self._conn.execute(
            """MATCH (p:Paper {doi: $doi}), (v:Venue {key: $vkey})
               MERGE (p)-[:PUBLISHED_AT {year: $year}]->(v)
            """,
            parameters={"doi": paper_doi, "vkey": venue_key, "year": year or 0},
        )

    def _ingest_author_affiliation(
        self,
        cand: dict[str, Any],
        verdict: dict[str, Any],
        paper_key: str,
        year: int | None,
    ) -> str | None:
        """Upsert Author + Institution + AUTHORED + AFFILIATED_AT edges."""
        author_id = _author_canonical_id(cand)
        name = _str(cand.get("author_name"))

        # Author node
        self._conn.execute(
            """MERGE (a:Author {canonical_id: $cid})
               SET a.s2_id         = $s2_id,
                   a.orcid         = $orcid,
                   a.name_variants = $name_variants
            """,
            parameters={
                "cid": author_id,
                "s2_id": _str(cand.get("s2_id")),
                "orcid": "",
                "name_variants": [name] if name else [],
            },
        )

        # AUTHORED edge
        self._conn.execute(
            """MATCH (a:Author {canonical_id: $cid}), (p:Paper {doi: $doi})
               MERGE (a)-[:AUTHORED {position: $pos}]->(p)
            """,
            parameters={"cid": author_id, "doi": paper_key, "pos": 0},
        )

        # Evidence nodes and AFFILIATED_AT edges
        evidence_ids: list[str] = verdict.get("evidence_ids") or []
        affiliations: list[str] = cand.get("affiliations") or []

        for aff in affiliations:
            inst_id = _institution_id(aff, cand)
            ror_id = _extract_ror(cand) or inst_id

            # Institution node
            self._conn.execute(
                """MERGE (i:Institution {ror_id: $ror_id})
                   SET i.canonical_name = $name,
                       i.org_type       = $org_type,
                       i.country_code   = $country
                """,
                parameters={
                    "ror_id": ror_id,
                    "name": aff,
                    "org_type": _infer_org_type(aff),
                    "country": "",
                },
            )

            # Persist Evidence nodes
            for ev_id in evidence_ids[:1]:  # one evidence per affiliation (primary)
                self._upsert_evidence(ev_id, verdict)

            ev_id = evidence_ids[0] if evidence_ids else f"auto:{author_id}:{ror_id}"

            # AFFILIATED_AT edge (per-paper, carries evidence_id)
            self._conn.execute(
                """MATCH (a:Author {canonical_id: $cid}),
                         (i:Institution {ror_id: $ror_id})
                   MERGE (a)-[:AFFILIATED_AT {
                       paper_id:    $paper_id,
                       evidence_id: $ev_id,
                       year:        $year
                   }]->(i)
                """,
                parameters={
                    "cid": author_id,
                    "ror_id": ror_id,
                    "paper_id": paper_key,
                    "ev_id": ev_id,
                    "year": year or 0,
                },
            )

        return author_id

    def _upsert_evidence(self, evidence_id: str, verdict: dict[str, Any]) -> None:
        self._conn.execute(
            """MERGE (e:Evidence {evidence_id: $eid})
               SET e.source       = $source,
                   e.raw_payload  = $payload,
                   e.retrieved_at = $ts
            """,
            parameters={
                "eid": evidence_id,
                "source": "critic_verdict",
                "payload": json.dumps(verdict)[:1000],
                "ts": datetime.now(UTC).isoformat(),
            },
        )

    def _upsert_edge_coauthored_with(
        self, aid_i: str, aid_j: str, paper_key: str, year: int | None
    ) -> None:
        for src, dst in [(aid_i, aid_j), (aid_j, aid_i)]:
            self._conn.execute(
                """MATCH (a:Author {canonical_id: $src}),
                         (b:Author {canonical_id: $dst})
                   MERGE (a)-[:COAUTHORED_WITH {paper_id: $pid, year: $year}]->(b)
                """,
                parameters={"src": src, "dst": dst, "pid": paper_key, "year": year or 0},
            )

    def _update_collaborated_with(
        self,
        cand_map: dict[str, dict[str, Any]],
        verdicts: list[dict[str, Any]],
        year: int | None,
    ) -> None:
        """Materialise COLLABORATED_WITH between all institution pairs in this paper."""
        inst_ids: list[str] = []
        for v in verdicts:
            if v.get("decision") != "accept":
                continue
            cand = cand_map.get(v.get("candidate_id", ""))
            if not cand:
                continue
            for aff in cand.get("affiliations") or []:
                ror_id = _extract_ror(cand) or _institution_id(aff, cand)
                if ror_id not in inst_ids:
                    inst_ids.append(ror_id)

        for i, iid in enumerate(inst_ids):
            for jid in inst_ids[i + 1:]:
                # Check existing and increment papers_count
                res = self._conn.execute(
                    """MATCH (a:Institution {ror_id: $a})-[r:COLLABORATED_WITH {year: $y}]
                             ->(b:Institution {ror_id: $b})
                       RETURN r.papers_count
                    """,
                    parameters={"a": iid, "b": jid, "y": year or 0},
                )
                if res.has_next():
                    count = int(res.get_next()[0]) + 1
                    self._conn.execute(
                        """MATCH (a:Institution {ror_id: $a})-[r:COLLABORATED_WITH {year: $y}]
                                 ->(b:Institution {ror_id: $b})
                           SET r.papers_count = $count
                        """,
                        parameters={"a": iid, "b": jid, "y": year or 0, "count": count},
                    )
                else:
                    for src, dst in [(iid, jid), (jid, iid)]:
                        self._conn.execute(
                            """MATCH (a:Institution {ror_id: $src}), (b:Institution {ror_id: $dst})
                               MERGE (a)-[:COLLABORATED_WITH {year: $year, papers_count: 1}]->(b)
                            """,
                            parameters={"src": src, "dst": dst, "year": year or 0},
                        )


# ─────────────────────────── helpers ─────────────────────────────────────────


def _str(v: Any) -> str:
    return str(v) if v is not None else ""


def _paper_key(data: dict[str, Any]) -> str:
    pid = _str(data.get("paper_id"))
    if pid.startswith("doi:"):
        return pid.removeprefix("doi:")
    # Synthesise a stable DOI-like key from the source ID
    return f"synthetic:{hashlib.sha1(pid.encode()).hexdigest()[:12]}"


def _is_arxiv(data: dict[str, Any]) -> bool:
    return _str(data.get("paper_id")).startswith("arxiv:")


def _is_openalex(data: dict[str, Any]) -> bool:
    return _str(data.get("paper_id")).startswith("openalex:")


def _venue_key(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().lower()) or "unknown"


def _author_canonical_id(cand: dict[str, Any]) -> str:
    # Prefer S2AND cluster id; fall back to normalised name
    s2_cluster = _str(cand.get("s2_cluster_id") or cand.get("s2_id"))
    if s2_cluster:
        return f"s2:{s2_cluster}"
    name = re.sub(r"\s+", "_", _str(cand.get("author_name")).lower().strip())
    return f"name:{name}"


def _institution_id(aff: str, cand: dict[str, Any]) -> str:
    ror = _extract_ror(cand)
    if ror:
        return ror
    return f"name:{re.sub(r'\\s+', '_', aff.lower().strip())[:80]}"


def _extract_ror(cand: dict[str, Any]) -> str:
    """Try to find a ROR ID in the candidate's evidence_trail."""
    trail = cand.get("evidence_trail") or {}
    for ev in (trail.get("items") or []):
        payload = ev.get("raw_response") or {}
        ror = _str(payload.get("ror_id"))
        if ror and ror.startswith("https://ror.org/"):
            return ror
    return ""


def _infer_org_type(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("university", "college", "institute", "school", "academia", "eth", "mit", "caltech")):
        return "education"
    if any(k in n for k in ("google", "microsoft", "amazon", "meta", "apple", "deepmind", "openai", "anthropic")):
        return "company"
    if any(k in n for k in ("lab", "research", "centre", "center")):
        return "facility"
    return "other"
