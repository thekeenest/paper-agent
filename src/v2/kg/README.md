# src/v2/kg/

**Purpose:** Temporal heterogeneous knowledge graph — KuzuDB schema, ingestion, and GraphRAG.

## Graph schema (planned)

**Node tables:** `Author`, `Paper`, `Institution`, `Venue`, `Topic`, `Evidence`

**Relationship tables:**
| Relationship | From → To | Key property |
|---|---|---|
| `AUTHORED` | Author → Paper | `position` (first/last/middle) |
| `AFFILIATED_AT` | Author → Institution | `as_of_year`, `ror_id` |
| `PUBLISHED_AT` | Paper → Venue | `year`, `track` |
| `ABOUT` | Paper → Topic | `weight` |
| `CHILD_OF` | Institution → Institution | for sub-units |
| `COAUTHORED_WITH` | Author → Author | derived; `weight`, `year` |

## Planned modules

| Module | Purpose |
|---|---|
| `schema.py` | Kuzu DDL — `CREATE NODE TABLE`, `CREATE REL TABLE` |
| `ingest.py` | Idempotent upsert from `ExtractionResult` → KG |
| `queries.py` | Cypher query library |
| `graphrag.py` | Natural-language Q&A via LangChain GraphRAG |

See [DEV_PLAN.md §3.5](../../../coursework_v2/DEV_PLAN.md) and
[READING_LIST.md §B (SemOpenAlex, ORKG, StructRAG)](../../../coursework_v2/READING_LIST.md).
