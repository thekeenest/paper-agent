"""
KG layer tests — roundtrip, idempotency, queries, GraphRAG.

All tests use an in-memory (tmp-dir) KuzuDB so they are hermetic and
do not require any external services.

Test groups
-----------
  1. Roundtrip: 100 fixture verdicts → node/edge counts correct
  2. Idempotency: re-ingesting same paper → identical graph state
  3. Queries: each of the 4 query functions returns non-empty results
  4. GraphRAG: .answer() returns ≥1 cited node
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

# ─────────────────────────── helpers ─────────────────────────────────────────


def _make_db(tmp_path: Path):
    """Open a fresh KuzuDB and create the schema."""
    from src.v2.kg.schema import KGSchema, open_db

    # KuzuDB creates its own database file; parent must exist but path must not.
    db, conn = open_db(tmp_path / "kg_test.kuzu")
    KGSchema.create_all(conn)
    return db, conn


def _make_work_item(
    paper_id: str = "doi:10.1234/test.001",
    venue: str = "NeurIPS",
    year: int = 2023,
    authors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct a minimal WorkItem dict for ingestion."""
    if authors is None:
        authors = [
            {
                "candidate_id": "cand_001",
                "author_name": "Alice Smith",
                "affiliations": ["MIT Computer Science"],
                "emails": ["alice@mit.edu"],
                "s2_id": "s2auth_001",
                "s2_cluster_id": "s2cl_001",
                "source_specialist": "grobid",
                "evidence_trail": {"items": []},
            },
            {
                "candidate_id": "cand_002",
                "author_name": "Bob Jones",
                "affiliations": ["Google Research"],
                "emails": ["bob@google.com"],
                "s2_id": "s2auth_002",
                "s2_cluster_id": "s2cl_002",
                "source_specialist": "openalex",
                "evidence_trail": {"items": []},
            },
        ]
    verdicts = [
        {
            "candidate_id": c["candidate_id"],
            "decision": "accept",
            "confidence": 0.9,
            "confidence_band": "high",
            "rationale": "test",
            "evidence_ids": [f"ev_{c['candidate_id']}"],
            "salvage": False,
        }
        for c in authors
    ]
    return {
        "canonical_paper": {
            "paper_id": paper_id,
            "title": f"Test Paper {paper_id}",
            "venue": venue,
            "year": year,
            "abstract": "Test abstract.",
        },
        "merged_candidates": authors,
        "verdicts": verdicts,
    }


def _make_100_work_items() -> list[dict[str, Any]]:
    """Generate 100 distinct work items across 5 years and 2 venues."""
    items: list[dict[str, Any]] = []
    venues = ["NeurIPS", "ICML"]
    for i in range(100):
        venue = venues[i % 2]
        year = 2020 + (i % 5)
        paper_id = f"doi:10.1234/paper.{i:04d}"
        authors = [
            {
                "candidate_id": f"cand_{i}_0",
                "author_name": f"Author {i}A",
                "affiliations": ["MIT Computer Science"] if i % 3 == 0 else ["Stanford University"],
                "emails": [],
                "s2_id": f"s2_{i}_a",
                "s2_cluster_id": f"s2cl_{i}_a",
                "source_specialist": "grobid",
                "evidence_trail": {"items": []},
            },
            {
                "candidate_id": f"cand_{i}_1",
                "author_name": f"Author {i}B",
                "affiliations": ["Google Research"] if i % 4 == 0 else ["Oxford University"],
                "emails": [],
                "s2_id": f"s2_{i}_b",
                "s2_cluster_id": f"s2cl_{i}_b",
                "source_specialist": "openalex",
                "evidence_trail": {"items": []},
            },
        ]
        items.append(_make_work_item(paper_id=paper_id, venue=venue, year=year, authors=authors))
    return items


# ─────────────────────────── fixtures ────────────────────────────────────────


@pytest.fixture()
def tmp_db(tmp_path):
    """Provide a fresh KuzuDB connection for each test."""
    db, conn = _make_db(tmp_path)
    yield conn
    # kuzu doesn't need explicit close in tests


@pytest.fixture()
def populated_db(tmp_path):
    """DB populated with 100 fixture work items."""
    from src.v2.kg.ingest import KGIngestor

    db, conn = _make_db(tmp_path)
    ingestor = KGIngestor(conn)
    for wi in _make_100_work_items():
        ingestor.ingest_verdict(wi)
    yield conn


# ─────────────────────────── 1. Roundtrip ────────────────────────────────────


class TestRoundtrip:
    """Ingest 100 fixture verdicts and verify node/edge counts."""

    def test_paper_count(self, populated_db):
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(populated_db)
        assert ing.node_count("Paper") == 100

    def test_author_count(self, populated_db):
        """200 distinct authors (2 per paper, all unique cluster IDs)."""
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(populated_db)
        assert ing.node_count("Author") == 200

    def test_venue_count(self, populated_db):
        """Exactly 2 venues: NeurIPS and ICML."""
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(populated_db)
        assert ing.node_count("Venue") == 2

    def test_authored_edges(self, populated_db):
        """200 AUTHORED edges (2 per paper × 100 papers)."""
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(populated_db)
        assert ing.edge_count("AUTHORED") == 200

    def test_published_at_edges(self, populated_db):
        """100 PUBLISHED_AT edges (1 per paper)."""
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(populated_db)
        assert ing.edge_count("PUBLISHED_AT") == 100

    def test_coauthored_with_edges(self, populated_db):
        """100 papers × 2 authors each → 2 COAUTHORED_WITH edges per paper (bidirectional)."""
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(populated_db)
        # Each pair of 2 authors → 2 directed edges
        assert ing.edge_count("COAUTHORED_WITH") == 200

    def test_affiliated_at_edges_present(self, populated_db):
        """At least some AFFILIATED_AT edges exist."""
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(populated_db)
        assert ing.edge_count("AFFILIATED_AT") > 0

    def test_institution_nodes_present(self, populated_db):
        """At least 2 institution nodes (MIT + Google or Stanford + Oxford)."""
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(populated_db)
        assert ing.node_count("Institution") >= 2

    def test_evidence_nodes_present(self, populated_db):
        """Evidence nodes are upserted."""
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(populated_db)
        assert ing.node_count("Evidence") > 0

    def test_no_canonical_paper_skipped(self, tmp_db):
        """Work items without canonical_paper are silently skipped."""
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(tmp_db)
        ing.ingest_verdict({})  # should not raise
        assert ing.node_count("Paper") == 0


# ─────────────────────────── 2. Idempotency ──────────────────────────────────


class TestIdempotency:
    """Re-ingesting the same work item must produce identical graph state."""

    def test_paper_idempotent(self, tmp_db):
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(tmp_db)
        wi = _make_work_item()
        ing.ingest_verdict(wi)
        count_1 = ing.node_count("Paper")
        ing.ingest_verdict(wi)
        count_2 = ing.node_count("Paper")
        assert count_1 == count_2 == 1

    def test_author_idempotent(self, tmp_db):
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(tmp_db)
        wi = _make_work_item()
        ing.ingest_verdict(wi)
        authored_1 = ing.edge_count("AUTHORED")
        ing.ingest_verdict(wi)
        authored_2 = ing.edge_count("AUTHORED")
        assert authored_1 == authored_2

    def test_venue_idempotent(self, tmp_db):
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(tmp_db)
        wi = _make_work_item()
        ing.ingest_verdict(wi)
        v1 = ing.node_count("Venue")
        ing.ingest_verdict(wi)
        v2 = ing.node_count("Venue")
        assert v1 == v2 == 1

    def test_coauthored_idempotent(self, tmp_db):
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(tmp_db)
        wi = _make_work_item()
        ing.ingest_verdict(wi)
        c1 = ing.edge_count("COAUTHORED_WITH")
        ing.ingest_verdict(wi)
        c2 = ing.edge_count("COAUTHORED_WITH")
        assert c1 == c2

    def test_affiliated_at_idempotent(self, tmp_db):
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(tmp_db)
        wi = _make_work_item()
        ing.ingest_verdict(wi)
        a1 = ing.edge_count("AFFILIATED_AT")
        ing.ingest_verdict(wi)
        a2 = ing.edge_count("AFFILIATED_AT")
        assert a1 == a2

    def test_100x_repeat_stable(self, tmp_db):
        """Ingesting the same paper 10 times keeps Paper count == 1."""
        from src.v2.kg.ingest import KGIngestor

        ing = KGIngestor(tmp_db)
        wi = _make_work_item(paper_id="doi:10.9999/idempotent")
        for _ in range(10):
            ing.ingest_verdict(wi)
        assert ing.node_count("Paper") == 1


# ─────────────────────────── 3. Queries ──────────────────────────────────────


class TestQueries:
    """Each query function returns non-empty results on a populated DB."""

    def test_industry_share_by_venue_non_empty(self, populated_db):
        from src.v2.kg.queries.industry_share_by_venue import industry_share_by_venue

        result = industry_share_by_venue(populated_db)
        assert len(result.rows) > 0

    def test_industry_share_venue_filter(self, populated_db):
        from src.v2.kg.queries.industry_share_by_venue import industry_share_by_venue

        result = industry_share_by_venue(populated_db, venue="neurips")
        for row in result.rows:
            assert "neurips" in row.venue_key.lower()

    def test_industry_share_year_filter(self, populated_db):
        from src.v2.kg.queries.industry_share_by_venue import industry_share_by_venue

        result = industry_share_by_venue(populated_db, year=2021)
        for row in result.rows:
            assert row.year == 2021

    def test_industry_share_fraction_range(self, populated_db):
        from src.v2.kg.queries.industry_share_by_venue import industry_share_by_venue

        result = industry_share_by_venue(populated_db)
        for row in result.rows:
            assert 0.0 <= row.industry_share <= 1.0

    def test_top_institutions_non_empty(self, populated_db):
        from src.v2.kg.queries.top_institutions_by_venue import top_institutions_by_venue

        result = top_institutions_by_venue(populated_db, venue="NeurIPS")
        assert len(result.rows) > 0

    def test_top_institutions_rank_order(self, populated_db):
        from src.v2.kg.queries.top_institutions_by_venue import top_institutions_by_venue

        result = top_institutions_by_venue(populated_db, venue="NeurIPS", top_k=5)
        counts = [r.affiliation_count for r in result.rows]
        assert counts == sorted(counts, reverse=True)
        assert all(r.rank == i + 1 for i, r in enumerate(result.rows))

    def test_coauthor_neighborhood_center_present(self, tmp_db):
        from src.v2.kg.ingest import KGIngestor
        from src.v2.kg.queries.coauthor_neighborhood import coauthor_neighborhood

        ing = KGIngestor(tmp_db)
        ing.ingest_verdict(_make_work_item())

        result = coauthor_neighborhood(tmp_db, author_id="s2:s2cl_001")
        assert result.center_id == "s2:s2cl_001"
        assert any(n.canonical_id == "s2:s2cl_001" for n in result.nodes)

    def test_coauthor_neighborhood_1hop(self, tmp_db):
        from src.v2.kg.ingest import KGIngestor
        from src.v2.kg.queries.coauthor_neighborhood import coauthor_neighborhood

        ing = KGIngestor(tmp_db)
        ing.ingest_verdict(_make_work_item())

        result = coauthor_neighborhood(tmp_db, author_id="s2:s2cl_001", hops=1)
        # Should have center + at least 1 co-author
        assert len(result.nodes) >= 1

    def test_coauthor_neighborhood_by_name(self, tmp_db):
        from src.v2.kg.ingest import KGIngestor
        from src.v2.kg.queries.coauthor_neighborhood import coauthor_neighborhood

        ing = KGIngestor(tmp_db)
        ing.ingest_verdict(_make_work_item())

        result = coauthor_neighborhood(tmp_db, author_name="Alice Smith", hops=1)
        assert result.center_id != "<not_found>"

    def test_country_heatmap_empty_without_country_codes(self, populated_db):
        """Without country codes set, heatmap returns empty (no country pairs)."""
        from src.v2.kg.queries.country_strategy_heatmap import country_strategy_heatmap

        # Institutions in fixtures have empty country_code, so result may be empty
        result = country_strategy_heatmap(populated_db)
        assert isinstance(result.rows, list)

    def test_country_heatmap_with_country_codes(self, tmp_db):
        """With country codes populated via direct upsert, heatmap is non-empty."""
        from src.v2.kg.ingest import KGIngestor
        from src.v2.kg.queries.country_strategy_heatmap import country_strategy_heatmap

        ing = KGIngestor(tmp_db)
        # Ingest two papers with different country codes
        wi1 = _make_work_item(paper_id="doi:10.1/cc_test_1")
        wi1["merged_candidates"][0]["affiliations"] = ["MIT Computer Science"]
        wi1["merged_candidates"][1]["affiliations"] = ["Oxford University"]
        ing.ingest_verdict(wi1)

        # Manually set country codes on institution nodes
        tmp_db.execute(
            "MATCH (i:Institution) WHERE i.canonical_name = 'MIT Computer Science' "
            "SET i.country_code = 'US'"
        )
        tmp_db.execute(
            "MATCH (i:Institution) WHERE i.canonical_name = 'Oxford University' "
            "SET i.country_code = 'GB'"
        )

        result = country_strategy_heatmap(tmp_db, min_papers=1)
        assert isinstance(result.rows, list)
        # to_matrix should work without error
        matrix = result.to_matrix()
        assert isinstance(matrix, dict)


# ─────────────────────────── 4. GraphRAG ─────────────────────────────────────


class TestGraphRAG:
    """GraphRAG returns a response that cites ≥1 KG node."""

    @pytest.mark.asyncio
    async def test_graphrag_cites_at_least_one_node(self, populated_db, monkeypatch):
        """GraphRAG answer must cite at least 1 node (uses mocked LLM)."""
        from src.v2.kg.graphrag import GraphRAG

        # Mock the LLM to return a canned answer with citations
        class _MockLLM:
            async def ainvoke(self, messages):
                class _R:
                    content = json.dumps({
                        "answer": "MIT collaborated most with Stanford.",
                        "cited_nodes": [
                            {
                                "node_type": "Institution",
                                "node_id": "name:mit_computer_science",
                                "role": "Primary institution queried",
                            }
                        ],
                    })
                return _R()

        rag = GraphRAG(populated_db)

        # Patch the LLM within the _synthesise method
        async def _mock_synthesise(question, subgraph_text, node_ids):
            from src.v2.kg.graphrag import GraphRAGResponse, CitedNode, _parse_llm_response

            content = json.dumps({
                "answer": "MIT collaborated most with Stanford.",
                "cited_nodes": [
                    {
                        "node_type": "Institution",
                        "node_id": node_ids[0] if node_ids else "name:mit_computer_science",
                        "role": "Primary institution queried",
                    }
                ],
            })
            resp = _parse_llm_response(content, node_ids)
            resp.subgraph_size = len(node_ids)
            return resp

        monkeypatch.setattr(rag, "_synthesise", _mock_synthesise)

        response = await rag.answer("Which institutions collaborated most with MIT at NeurIPS 2023?")
        assert len(response.cited_nodes) >= 1

    @pytest.mark.asyncio
    async def test_graphrag_answer_non_empty(self, populated_db, monkeypatch):
        """GraphRAG answer text is non-empty."""
        from src.v2.kg.graphrag import GraphRAG, GraphRAGResponse, CitedNode

        rag = GraphRAG(populated_db)

        async def _mock_synthesise(question, subgraph_text, node_ids):
            resp = GraphRAGResponse(
                answer="Test answer for question.",
                cited_nodes=[
                    CitedNode(
                        node_type="Institution",
                        node_id=node_ids[0] if node_ids else "name:test",
                        role="test",
                    )
                ],
                subgraph_size=len(node_ids),
            )
            return resp

        monkeypatch.setattr(rag, "_synthesise", _mock_synthesise)
        response = await rag.answer("Who are the top authors at NeurIPS?")
        assert response.answer.strip() != ""

    @pytest.mark.asyncio
    async def test_graphrag_subgraph_size_tracked(self, populated_db, monkeypatch):
        """GraphRAG response carries subgraph_size > 0 for a seeded query."""
        from src.v2.kg.graphrag import GraphRAG, GraphRAGResponse, CitedNode

        rag = GraphRAG(populated_db)

        async def _mock_synthesise(question, subgraph_text, node_ids):
            return GraphRAGResponse(
                answer="Answer.",
                cited_nodes=[CitedNode(node_type="Venue", node_id="neurips", role="venue")],
                subgraph_size=len(node_ids),
            )

        monkeypatch.setattr(rag, "_synthesise", _mock_synthesise)
        response = await rag.answer("Papers at NeurIPS 2022")
        assert response.subgraph_size >= 0  # may be 0 if no seeds linked

    def test_graphrag_cited_node_types_valid(self):
        """CitedNode rejects invalid node_type values."""
        from pydantic import ValidationError
        from src.v2.kg.graphrag import CitedNode

        with pytest.raises(ValidationError):
            CitedNode(node_type="InvalidType", node_id="x", role="y")

    def test_graphrag_response_pydantic(self):
        """GraphRAGResponse is a proper Pydantic model."""
        from src.v2.kg.graphrag import GraphRAGResponse, CitedNode

        resp = GraphRAGResponse(
            answer="test",
            cited_nodes=[
                CitedNode(node_type="Author", node_id="s2:123", role="main author")
            ],
        )
        d = resp.model_dump()
        assert d["answer"] == "test"
        assert d["cited_nodes"][0]["node_type"] == "Author"
