"""
GraphRAG for Paper-Agent v2.
-----------------------------
Minimal retrieval-augmented generation pipeline grounded in the KuzuDB KG.

Pipeline
--------
1. **Entity linking** — extract entity mentions from the user question via an
   LLM call (or rule-based fallback), resolve them to canonical KG node IDs.
2. **Subgraph extraction** — from each seed node, traverse up to ``hops``
   (1 or 2) in the graph and collect a concise text representation of every
   reachable node.
3. **LLM synthesis** — inject the subgraph context into a grounded prompt and
   ask Claude to answer the question, citing node IDs in its response.

Pydantic output
---------------
``GraphRAGResponse`` — answer text + list of ``CitedNode`` objects each
carrying the node type, node id, and how it was used.

Usage
-----
::

    from src.v2.kg.schema import open_db, KGSchema
    from src.v2.kg.graphrag import GraphRAG

    db, conn = open_db()
    KGSchema.create_all(conn)
    rag = GraphRAG(conn)
    resp = await rag.answer("Which institutions collaborated most with MIT at NeurIPS 2023?")
    print(resp.answer)
    for c in resp.cited_nodes:
        print(c.node_type, c.node_id, c.role)
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

_LOG = structlog.get_logger(__name__)

_DEFAULT_MODEL = os.getenv("GRAPHRAG_MODEL", "claude-sonnet-4-6")

# ─────────────────────────── output types ────────────────────────────────────


class CitedNode(BaseModel):
    """A KG node cited in the GraphRAG answer."""

    node_type: Literal["Author", "Paper", "Institution", "Venue", "Topic", "Evidence"]
    node_id: str = Field(description="Canonical primary key of the node")
    role: str = Field(description="How this node was used in the answer (1-2 sentences)")


class GraphRAGResponse(BaseModel):
    """Typed response returned by :meth:`GraphRAG.answer`."""

    answer: str
    cited_nodes: list[CitedNode] = Field(default_factory=list)
    subgraph_size: int = Field(
        default=0, description="Number of KG nodes included in the context"
    )


# ─────────────────────────── GraphRAG ────────────────────────────────────────


class GraphRAG:
    """KG-grounded question answering with cited node IDs.

    Parameters
    ----------
    conn:
        Open ``kuzu.Connection`` on a database where
        ``KGSchema.create_all`` has already been called.
    model:
        Anthropic model name.  Defaults to ``GRAPHRAG_MODEL`` env var or
        ``claude-sonnet-4-6``.
    hops:
        Subgraph traversal depth: 1 = direct neighbours only, 2 = 2nd-order.
    max_nodes:
        Hard cap on nodes included in the LLM context window.
    """

    def __init__(
        self,
        conn: Any,
        model: str | None = None,
        hops: int = 2,
        max_nodes: int = 80,
    ) -> None:
        self._conn = conn
        self._model = model or os.getenv("GRAPHRAG_MODEL") or _DEFAULT_MODEL
        self._hops = hops
        self._max_nodes = max_nodes

    async def answer(self, question: str) -> GraphRAGResponse:
        """Answer *question* using the KG as context.

        Parameters
        ----------
        question:
            Natural-language question about authors, affiliations, venues, etc.

        Returns
        -------
        GraphRAGResponse
            Answer text with Pydantic-typed cited nodes.
        """
        # 1. Entity linking
        seed_nodes = self._link_entities(question)
        _LOG.debug("graphrag.seed_nodes", n=len(seed_nodes), seeds=seed_nodes[:5])

        # 2. Subgraph extraction
        subgraph_text, node_ids = self._extract_subgraph(seed_nodes)
        _LOG.debug("graphrag.subgraph", n_nodes=len(node_ids))

        # 3. LLM synthesis
        response = await self._synthesise(question, subgraph_text, node_ids)
        response.subgraph_size = len(node_ids)
        return response

    # ── entity linking ────────────────────────────────────────────────────────

    def _link_entities(self, question: str) -> list[tuple[str, str]]:
        """Return list of (node_type, node_id) seed nodes from the question.

        Uses rule-based heuristics + KG look-ups (no extra LLM call).
        """
        seeds: list[tuple[str, str]] = []
        q_lower = question.lower()

        # ── Authors by name ──────────────────────────────────────────────────
        # Extract quoted names or capitalized word-pairs
        name_candidates = re.findall(r'"([^"]+)"', question)
        name_candidates += re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", question)
        for name in name_candidates:
            res = self._conn.execute(
                "MATCH (a:Author) WHERE "
                "any(v IN a.name_variants WHERE LOWER(v) CONTAINS LOWER($n)) "
                "RETURN a.canonical_id LIMIT 3",
                parameters={"n": name},
            )
            while res.has_next():
                seeds.append(("Author", str(res.get_next()[0])))

        # ── Venues by name ───────────────────────────────────────────────────
        venue_candidates = re.findall(
            r"\b(NeurIPS|ICML|ICLR|CVPR|ECCV|ICCV|ACL|EMNLP|NAACL|AAAI|IJCAI|KDD|WWW|SIGIR)\b",
            question,
            re.IGNORECASE,
        )
        for vname in venue_candidates:
            vkey = re.sub(r"\s+", "_", vname.strip().lower())
            res = self._conn.execute(
                "MATCH (v:Venue) WHERE LOWER(v.key) CONTAINS $vk "
                "RETURN v.key LIMIT 3",
                parameters={"vk": vkey},
            )
            while res.has_next():
                seeds.append(("Venue", str(res.get_next()[0])))

        # ── Institutions by keyword ───────────────────────────────────────────
        inst_pattern = re.compile(
            r"\b(MIT|Stanford|Harvard|Berkeley|CMU|Oxford|Cambridge|Google|Microsoft|"
            r"Amazon|Meta|DeepMind|OpenAI|Anthropic|ETH|Tsinghua|Peking|Toronto|NYU)\b",
            re.IGNORECASE,
        )
        for m in inst_pattern.finditer(question):
            kw = m.group(0).lower()
            res = self._conn.execute(
                "MATCH (i:Institution) WHERE LOWER(i.canonical_name) CONTAINS $kw "
                "RETURN i.ror_id LIMIT 3",
                parameters={"kw": kw},
            )
            while res.has_next():
                seeds.append(("Institution", str(res.get_next()[0])))

        # ── Year-based Paper seeding ─────────────────────────────────────────
        year_m = re.search(r"\b(20\d{2})\b", question)
        if year_m and venue_candidates:
            year = int(year_m.group(1))
            vkey = re.sub(r"\s+", "_", venue_candidates[0].strip().lower())
            res = self._conn.execute(
                "MATCH (p:Paper)-[pub:PUBLISHED_AT]->(v:Venue) "
                "WHERE LOWER(v.key) CONTAINS $vk AND pub.year = $yr "
                "RETURN p.doi LIMIT 10",
                parameters={"vk": vkey, "yr": year},
            )
            while res.has_next():
                seeds.append(("Paper", str(res.get_next()[0])))

        # Deduplicate preserving order
        seen: set[tuple[str, str]] = set()
        unique: list[tuple[str, str]] = []
        for s in seeds:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    # ── subgraph extraction ───────────────────────────────────────────────────

    def _extract_subgraph(
        self, seeds: list[tuple[str, str]]
    ) -> tuple[str, list[str]]:
        """Fetch a concise text description of the subgraph around seed nodes.

        Returns
        -------
        subgraph_text:
            Human-readable block listing node properties and edges.
        node_ids:
            Flat list of canonical node IDs included.
        """
        collected: dict[str, str] = {}  # node_id → text description

        for node_type, node_id in seeds[: self._max_nodes]:
            self._collect_node(node_type, node_id, collected, hop=0)
            if len(collected) >= self._max_nodes:
                break

        if not collected:
            return "(no relevant KG context found)", []

        lines = ["=== KG Subgraph Context ==="]
        for nid, desc in list(collected.items())[: self._max_nodes]:
            lines.append(f"[{nid}] {desc}")
        return "\n".join(lines), list(collected.keys())

    def _collect_node(
        self,
        node_type: str,
        node_id: str,
        collected: dict[str, str],
        hop: int,
    ) -> None:
        """Recursively collect node text and its neighbours up to self._hops."""
        if node_id in collected or len(collected) >= self._max_nodes:
            return

        desc = self._node_description(node_type, node_id)
        if desc is None:
            return
        collected[node_id] = desc

        if hop >= self._hops:
            return

        # Traverse neighbours
        for neighbour_type, neighbour_id in self._get_neighbours(node_type, node_id):
            self._collect_node(neighbour_type, neighbour_id, collected, hop + 1)
            if len(collected) >= self._max_nodes:
                return

    def _node_description(self, node_type: str, node_id: str) -> str | None:
        """Fetch a one-line text description of a KG node."""
        try:
            if node_type == "Author":
                res = self._conn.execute(
                    "MATCH (a:Author {canonical_id: $id}) "
                    "RETURN a.name_variants, a.s2_id",
                    parameters={"id": node_id},
                )
                if res.has_next():
                    row = res.get_next()
                    names = ", ".join(row[0] or [])
                    return f"Author | names={names!r} s2_id={row[1]!r}"

            elif node_type == "Institution":
                res = self._conn.execute(
                    "MATCH (i:Institution {ror_id: $id}) "
                    "RETURN i.canonical_name, i.org_type, i.country_code",
                    parameters={"id": node_id},
                )
                if res.has_next():
                    row = res.get_next()
                    return (
                        f"Institution | name={row[0]!r} type={row[1]!r} "
                        f"country={row[2]!r}"
                    )

            elif node_type == "Paper":
                res = self._conn.execute(
                    "MATCH (p:Paper {doi: $id}) "
                    "RETURN p.title, p.year, p.venue_key",
                    parameters={"id": node_id},
                )
                if res.has_next():
                    row = res.get_next()
                    return f"Paper | title={row[0]!r} year={row[1]} venue={row[2]!r}"

            elif node_type == "Venue":
                res = self._conn.execute(
                    "MATCH (v:Venue {key: $id}) RETURN v.full_name, v.kind",
                    parameters={"id": node_id},
                )
                if res.has_next():
                    row = res.get_next()
                    return f"Venue | full_name={row[0]!r} kind={row[1]!r}"

        except Exception as exc:
            _LOG.debug("graphrag.node_desc_failed", node_type=node_type, id=node_id, error=str(exc))
        return None

    def _get_neighbours(
        self, node_type: str, node_id: str
    ) -> list[tuple[str, str]]:
        """Return (node_type, node_id) pairs reachable from node in 1 hop."""
        neighbours: list[tuple[str, str]] = []
        try:
            if node_type == "Author":
                # AUTHORED → Paper
                res = self._conn.execute(
                    "MATCH (a:Author {canonical_id: $id})-[:AUTHORED]->(p:Paper) "
                    "RETURN p.doi LIMIT 5",
                    parameters={"id": node_id},
                )
                while res.has_next():
                    neighbours.append(("Paper", str(res.get_next()[0])))
                # AFFILIATED_AT → Institution
                res = self._conn.execute(
                    "MATCH (a:Author {canonical_id: $id})-[:AFFILIATED_AT]->(i:Institution) "
                    "RETURN i.ror_id LIMIT 5",
                    parameters={"id": node_id},
                )
                while res.has_next():
                    neighbours.append(("Institution", str(res.get_next()[0])))

            elif node_type == "Paper":
                # PUBLISHED_AT → Venue
                res = self._conn.execute(
                    "MATCH (p:Paper {doi: $id})-[:PUBLISHED_AT]->(v:Venue) "
                    "RETURN v.key LIMIT 3",
                    parameters={"id": node_id},
                )
                while res.has_next():
                    neighbours.append(("Venue", str(res.get_next()[0])))
                # ← AUTHORED by Author
                res = self._conn.execute(
                    "MATCH (a:Author)-[:AUTHORED]->(p:Paper {doi: $id}) "
                    "RETURN a.canonical_id LIMIT 10",
                    parameters={"id": node_id},
                )
                while res.has_next():
                    neighbours.append(("Author", str(res.get_next()[0])))

            elif node_type == "Institution":
                # COLLABORATED_WITH → Institution
                res = self._conn.execute(
                    "MATCH (i:Institution {ror_id: $id})-[:COLLABORATED_WITH]->(j:Institution) "
                    "RETURN j.ror_id LIMIT 5",
                    parameters={"id": node_id},
                )
                while res.has_next():
                    neighbours.append(("Institution", str(res.get_next()[0])))

            elif node_type == "Venue":
                # ← PUBLISHED_AT by Paper (recent)
                res = self._conn.execute(
                    "MATCH (p:Paper)-[:PUBLISHED_AT]->(v:Venue {key: $id}) "
                    "RETURN p.doi LIMIT 5",
                    parameters={"id": node_id},
                )
                while res.has_next():
                    neighbours.append(("Paper", str(res.get_next()[0])))

        except Exception as exc:
            _LOG.debug("graphrag.neighbours_failed", node_type=node_type, id=node_id, error=str(exc))
        return neighbours

    # ── LLM synthesis ─────────────────────────────────────────────────────────

    async def _synthesise(
        self,
        question: str,
        subgraph_text: str,
        node_ids: list[str],
    ) -> GraphRAGResponse:
        """Call the LLM with subgraph context and return a structured answer."""
        system = (
            "You are a scholarly knowledge-graph assistant. "
            "Answer the question using ONLY the KG context provided. "
            "Cite each node you use by its bracketed ID, e.g. [s2:12345] or [10.1234/abc]. "
            "Be concise and factual."
        )
        user = (
            f"Context:\n{subgraph_text}\n\n"
            f"Question: {question}\n\n"
            "Respond with a JSON object: "
            '{"answer": "...", "cited_nodes": [{"node_type": "...", "node_id": "...", "role": "..."}]}'
        )

        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = ChatAnthropic(model_name=self._model, temperature=0.0, max_tokens=1024)  # type: ignore[call-arg]
            resp = await llm.ainvoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            content = resp.content if hasattr(resp, "content") else str(resp)
            return _parse_llm_response(str(content), node_ids)

        except Exception as exc:
            _LOG.error("graphrag.llm_failed", error=str(exc))
            # Fallback: rule-based answer listing node IDs
            return GraphRAGResponse(
                answer=f"Could not generate LLM answer ({exc}). "
                f"Found {len(node_ids)} relevant KG nodes.",
                cited_nodes=[
                    CitedNode(node_type="Paper", node_id=nid, role="retrieved context")
                    for nid in node_ids[:3]
                    if "doi" in nid or "synthetic" in nid
                ],
            )


# ─────────────────────────── helpers ─────────────────────────────────────────


def _parse_llm_response(content: str, known_ids: list[str]) -> GraphRAGResponse:
    """Parse LLM JSON output into a GraphRAGResponse, with graceful fallback."""
    # Try to extract JSON block
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            answer = str(data.get("answer", content))
            cited_raw = data.get("cited_nodes", [])
            cited: list[CitedNode] = []
            valid_types = {"Author", "Paper", "Institution", "Venue", "Topic", "Evidence"}
            for c in cited_raw:
                nt = str(c.get("node_type", "Paper"))
                if nt not in valid_types:
                    nt = "Paper"
                cited.append(
                    CitedNode(
                        node_type=nt,  # type: ignore[arg-type]
                        node_id=str(c.get("node_id", "")),
                        role=str(c.get("role", "")),
                    )
                )
            # If LLM returned no citations but we have known nodes, add them
            if not cited and known_ids:
                cited = [
                    CitedNode(
                        node_type="Paper",  # type: ignore[arg-type]
                        node_id=nid,
                        role="retrieved from KG context",
                    )
                    for nid in known_ids[:3]
                ]
            return GraphRAGResponse(answer=answer, cited_nodes=cited)
        except json.JSONDecodeError:
            pass

    # Fallback: use the raw content as the answer, extract bracket citations
    bracket_ids = re.findall(r"\[([^\]]+)\]", content)
    cited = [
        CitedNode(
            node_type=_guess_node_type(bid),
            node_id=bid,
            role="cited in answer",
        )
        for bid in bracket_ids
        if bid in known_ids
    ]
    if not cited and known_ids:
        cited = [
            CitedNode(
                node_type=_guess_node_type(known_ids[0]),
                node_id=known_ids[0],
                role="retrieved from KG context",
            )
        ]
    return GraphRAGResponse(answer=content, cited_nodes=cited)


def _guess_node_type(node_id: str) -> str:
    """Heuristically determine node type from its canonical ID."""
    if node_id.startswith("s2:") or node_id.startswith("name:"):
        return "Author"
    if node_id.startswith("https://ror.org/") or node_id.startswith("ror:"):
        return "Institution"
    if "/" in node_id or node_id.startswith("10.") or node_id.startswith("synthetic:"):
        return "Paper"
    return "Venue"
