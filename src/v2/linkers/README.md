# src/v2/linkers/

**Purpose:** External entity resolution — institutions (ROR, OpenAlex) and authors (S2AND).

| Planned module | Source | What it resolves |
|---|---|---|
| `ror_linker.py` | ROR API | Organization string → ROR ID + canonical name + country |
| `openalex_linker.py` | OpenAlex API | Paper ID → author affiliations + institution IDs |
| `s2and_linker.py` | Semantic Scholar | Author name + context → disambiguated author ID |
| `dns_resolver.py` | DNS | Email domain → institution (MX/A lookup) |

Replaces the v1 local knowledge-base (~50 entries) with live, grounded lookups.

See [DEV_PLAN.md §3.3](../../../coursework_v2/DEV_PLAN.md) and
[READING_LIST.md §D (S2AND, ROR, S2AFF)](../../../coursework_v2/READING_LIST.md).
