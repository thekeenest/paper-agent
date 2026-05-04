// Paper-Agent v2 — KuzuDB DDL
// Schema from DEV_PLAN.md §3.3
// Execute via KGSchema.create_all(conn)

// ─────────────────────────── Node tables ─────────────────────────────────────

CREATE NODE TABLE IF NOT EXISTS Author (
    canonical_id  STRING,
    s2_id         STRING,
    orcid         STRING,
    name_variants STRING[],
    PRIMARY KEY (canonical_id)
);

CREATE NODE TABLE IF NOT EXISTS Paper (
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
);

CREATE NODE TABLE IF NOT EXISTS Institution (
    ror_id         STRING,
    openalex_id    STRING,
    canonical_name STRING,
    country_code   STRING,
    org_type       STRING,
    parent_ror_id  STRING,
    PRIMARY KEY (ror_id)
);

CREATE NODE TABLE IF NOT EXISTS Venue (
    key       STRING,
    full_name STRING,
    kind      STRING,
    PRIMARY KEY (key)
);

CREATE NODE TABLE IF NOT EXISTS Topic (
    topic_id STRING,
    label    STRING,
    source   STRING,
    PRIMARY KEY (topic_id)
);

CREATE NODE TABLE IF NOT EXISTS Evidence (
    evidence_id  STRING,
    source       STRING,
    raw_payload  STRING,
    retrieved_at STRING,
    PRIMARY KEY (evidence_id)
);

// ─────────────────────────── Relationship tables ──────────────────────────────

// (:Author)-[:AUTHORED {position}]->(:Paper)
CREATE REL TABLE IF NOT EXISTS AUTHORED (
    FROM Author TO Paper,
    position INT32
);

// (:Author)-[:AFFILIATED_AT {paper_id, evidence_id, year}]->(:Institution)
CREATE REL TABLE IF NOT EXISTS AFFILIATED_AT (
    FROM Author TO Institution,
    paper_id    STRING,
    evidence_id STRING,
    year        INT32
);

// (:Paper)-[:PUBLISHED_AT {year}]->(:Venue)
CREATE REL TABLE IF NOT EXISTS PUBLISHED_AT (
    FROM Paper TO Venue,
    year INT32
);

// (:Paper)-[:ABOUT {weight}]->(:Topic)
CREATE REL TABLE IF NOT EXISTS ABOUT (
    FROM Paper TO Topic,
    weight FLOAT
);

// (:Institution)-[:CHILD_OF]->(:Institution)
CREATE REL TABLE IF NOT EXISTS CHILD_OF (
    FROM Institution TO Institution
);

// (:Author)-[:COAUTHORED_WITH {paper_id, year}]->(:Author)
CREATE REL TABLE IF NOT EXISTS COAUTHORED_WITH (
    FROM Author TO Author,
    paper_id STRING,
    year     INT32
);

// (:Institution)-[:COLLABORATED_WITH {year, papers_count}]->(:Institution)
CREATE REL TABLE IF NOT EXISTS COLLABORATED_WITH (
    FROM Institution TO Institution,
    year        INT32,
    papers_count INT32
);

// (:Evidence) linked from Verdict via raw payload — no separate edge needed
// (evidence_id on AFFILIATED_AT is the FK)
