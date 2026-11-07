# AI Developments Digest — Implementation Status & System Capabilities

**Date:** 2026-11-03  
**Project:** AI Developments Digest (`ai-developments-digest`)  
**Repository State:** Production-ready local-first RSS/Atom ingestion, SQLite source of truth, Qdrant vector retrieval, Multi-Source Story Clustering, and Ollama-based consolidated summarization.

---

## 1. Executive Summary

AI Developments Digest is a local-first AI intelligence pipeline that collects AI research, official announcements, industry news, and release notes from RSS/Atom feeds, normalizes them, clusters multi-source reporting of the same development, and produces consolidated daily digests in structured JSON and Markdown using a local LLM (Ollama).

The system has transitioned from a destructive semantic deduplication model (which previously discarded duplicate reporting) into a **Multi-Source Story Clustering Architecture**. All reporting publications are retained in SQLite, mapped into unified story clusters, prioritized by source authority, and synthesized by the LLM with 100% database-verified source attribution.

---

## 2. What Can It Do Right Now?

### 2.1 Multi-Source Story Clustering
- **Cross-Source Grouping:** Detects when multiple publications (e.g. an official OpenAI announcement, a TechCrunch news piece, and a Hugging Face technical analysis) cover the same underlying event and groups them into a single story cluster.
- **Zero Article Loss:** Retains all ingested article records, URLs, publication dates, and source metadata in SQLite.
- **Event vs. Topic Differentiation:** Distinguishes between shared technology topics and actual events. For example, an initial model launch and a performance optimization released weeks later are kept as separate clusters even if they discuss the same model.
- **Entity & Version Conflict Detection:** Automatically extracts model names/versions (e.g., `GPT-5`, `Claude 3.7`, `Llama 4`) and prevents false merges if model versions conflict.
- **Configurable Ambiguity Margin:** If candidate matches from competing clusters are too close in confidence ($\Delta < 0.03$), the system avoids guessing and safely creates an independent cluster.
- **Temporal Window Guard:** Filters out candidate articles published outside the configured temporal window (`STORY_CLUSTER_MAX_AGE_DAYS`, default 30 days).

### 2.2 Source Authority & Representative Selection
- **Source Prioritization:** Automatically identifies the primary/representative article within each cluster using category heuristics:
  1. `official_release` (score: 100)
  2. `official_research` (score: 90)
  3. `open_source` (score: 80)
  4. `research_paper` (score: 75)
  5. `ai_infrastructure` (score: 70)
  6. `developer_tools` (score: 60)
  7. `industry_analysis` (score: 55)
  8. `industry_news` (score: 50)
  9. `newsletter` (score: 40)
- **Timeliness & Content Richness:** Rewards earlier reporting timestamps and substantive article content over single-sentence excerpts.
- **Full Source Traceability:** Retains all secondary sources; the primary source is prominently badge-attributed, but all reporting publications remain included in both batch inputs and final digests.

### 2.3 Automated Database Migrations & Historical Backfill
- **Transactional Schema Migrations:** Auto-detects schema versions via `schema_migrations` table without breaking existing tables.
  - **Version 1:** Base schema (`feeds`, `items`, `item_sources`, `runs`).
  - **Version 2:** Story clustering tables (`story_clusters`, `story_cluster_articles`) with cascading foreign keys and performance indexes.
- **Historical Backfill Tooling:** Allows any existing unclustered articles in SQLite to be grouped and assigned into story clusters retroactively (`ai-digest clusters backfill`).
- **Cluster Validation:** Detects empty clusters, missing article foreign keys, duplicate memberships, or inconsistent counts (`ai-digest clusters validate`).

### 2.4 Multi-Source Consolidated Summarization (Ollama)
- **Cluster-Level Synthesis:** The LLM receives pre-grouped story clusters with clear delimiters for each reporting publication.
- **Synthesized Story Output:** Generates a unified development covering:
  - `canonical_title`: Database-backed representative story title.
  - `primary_source`: Authoritative publisher with link and metadata.
  - `consolidated_summary`: High-level synthesis across all reporting sources.
  - `what_happened`: Core announcement or breakthrough facts.
  - `technical_details`: Architecture, metrics, benchmarks, methodologies.
  - `why_it_matters`: Industry and research implications.
  - `maturity`: Classification (`research | prototype | preview | production | unclear`).
  - `importance`: 1 to 5 rating.
  - `sources`: Complete list of all reporting articles with titles, URLs, and source names.
  - `source_urls`: Exact array of URLs attached strictly from the database.
- **Anti-Hallucination Guardrails:** Source URLs and publisher names are attached strictly from verified database records, preventing LLM URL invention.
- **Prompt Size Guardrails & Chunking:** Bounded context windows (`STORY_CLUSTER_MAX_ARTICLES=10`, `STORY_CLUSTER_MAX_CONTEXT_CHARS=30000`, chunked LLM calls).

### 2.5 Ingestion & Deduplication
- **Conditional HTTP Caching:** Leverages HTTP `ETag` and `Last-Modified` headers; receives `304 Not Modified` to prevent re-downloading unchanged feeds.
- **Exact URL Deduplication:** Canonicalizes URLs (strips `utm_*`, `fbclid`, tracking parameters, fragments, trailing slashes) and records cross-feed provenance in `item_sources`.
- **Local Vectors (Qdrant):** Embeds article text into normalized 384-dimensional vectors on CPU using `sentence-transformers/all-MiniLM-L6-v2`.
- **Qdrant Error Resilience:** If Qdrant is temporarily unavailable, items are not marked as indexed, allowing safe retry on subsequent runs.
- **Auto Service Startup:** Automatically starts Qdrant (Docker) and Ollama daemons if they are not already running when `ai-digest run` is executed.

---

## 3. CLI Commands Available Right Now

| Command | Description |
|---|---|
| `ai-digest run` | Full pipeline: fetch feeds, deduplicate, cluster stories, summarize with Ollama, output batch and digest files. |
| `ai-digest feeds` | Outputs all enabled configured RSS/Atom feeds from `config/feeds.json`. |
| `ai-digest stats` | Displays SQLite statistics: total items, feeds, Qdrant indexed items, story clusters, and clustered articles. |
| `ai-digest clusters stats` | Displays story cluster analytics: total clusters, average articles per cluster, multi-source clusters, unassigned articles, and recent clusters. |
| `ai-digest clusters show <story_id>` | Inspects a story cluster, displaying canonical title, primary source, timestamps, and all member articles with similarity scores and URLs. |
| `ai-digest clusters backfill` | Backfills unassigned or historical articles into story clusters idempotently. |
| `ai-digest clusters validate` | Performs integrity checks on story clusters (checks for empty clusters, missing references, duplicate memberships, and count mismatches). |

---

## 4. Current Architecture & File Structure

```text
ai_developments_digest/
├── config/
│   └── feeds.json                          # Feed catalog (20 pre-configured AI feeds)
├── data/
│   ├── digest.db                           # SQLite source of truth (WAL mode)
│   └── items/YYYY-MM-DD/<id>.[json|md]     # Individual normalized item artifacts
├── output/
│   ├── batches/batch_<timestamp>.[json|md] # Normalized story-cluster input batches (v2.0)
│   └── digests/digest_<timestamp>.[json|md] # Consolidated multi-source daily digests
├── src/ai_digest/
│   ├── __init__.py
│   ├── config.py                           # Typed settings & environment variables
│   ├── storage.py                          # SQLite store & migration hook
│   ├── migrations.py                       # Transactional schema migrations (v1 base, v2 clustering)
│   ├── normalize.py                        # URL canonicalization & text cleaning
│   ├── rss_collector.py                    # Conditional HTTP feed fetcher
│   ├── embeddings.py                       # Sentence Transformers (MiniLM-L6-v2)
│   ├── qdrant_store.py                     # Qdrant vector database client
│   ├── summarizer.py                       # Ollama multi-source cluster summarizer
│   ├── pipeline.py                         # End-to-end pipeline orchestrator
│   ├── start_services.py                   # Automatic Docker & Ollama lifecycle management
│   ├── cli.py                              # CLI interface (run, feeds, stats, clusters)
│   └── story_clustering/                   # Multi-Source Story Clustering Subsystem
│       ├── __init__.py
│       ├── cluster_repository.py           # SQLite cluster & membership CRUD
│       ├── candidate_retriever.py          # Qdrant neighbor discovery & age filtering
│       ├── similarity_matcher.py           # Multi-factor matching heuristics & ambiguity margin
│       ├── source_selector.py              # Authority-based representative source selector
│       └── cluster_manager.py              # Public workflow coordinator & backfill
└── tests/
    ├── test_normalize.py                   # URL canonicalization & HTML parsing tests
    ├── test_storage.py                     # SQLite exact dedupe & provenance tests
    ├── test_migrations.py                  # Schema migration & idempotency tests
    ├── test_clustering_scenarios.py        # Scenarios 1–8 (multi-source, ambiguity, temporal, etc.)
    ├── test_qdrant_and_pipeline.py         # Qdrant failure resilience & E2E pipeline tests
    ├── test_cli_clusters.py                # CLI clusters subcommands tests
    └── test_summarizer_clustering.py       # Ollama cluster summarizer & source integrity tests
```

---

## 5. Verification & Test Suite Status

The automated test suite contains **26 automated tests** with **100% passing status**:

```bash
pytest -v
```

```text
tests/test_cli_clusters.py::test_cli_clusters_stats PASSED
tests/test_cli_clusters.py::test_cli_clusters_show PASSED
tests/test_cli_clusters.py::test_cli_clusters_validate PASSED
tests/test_cli_clusters.py::test_cli_clusters_backfill PASSED
tests/test_clustering_scenarios.py::test_same_development_different_publishers PASSED
tests/test_clustering_scenarios.py::test_same_subject_different_development PASSED
tests/test_clustering_scenarios.py::test_unrelated_articles PASSED
tests/test_clustering_scenarios.py::test_exact_url_duplication PASSED
tests/test_clustering_scenarios.py::test_repeated_ingestion_idempotency PASSED
tests/test_clustering_scenarios.py::test_ambiguity_margin_creates_separate_cluster PASSED
tests/test_clustering_scenarios.py::test_time_window_restriction PASSED
tests/test_clustering_scenarios.py::test_source_selection_priority PASSED
tests/test_clustering_scenarios.py::test_edge_cases_empty_and_missing_data PASSED
tests/test_migrations.py::test_migration_from_scratch PASSED
tests/test_migrations.py::test_migration_from_legacy_v1_database PASSED
tests/test_migrations.py::test_migrations_are_idempotent PASSED
tests/test_normalize.py::test_canonicalize_url_drops_tracking_and_fragment PASSED
tests/test_normalize.py::test_html_to_text_removes_tags PASSED
tests/test_normalize.py::test_normalize_entry_outputs_portable_record PASSED
tests/test_qdrant_and_pipeline.py::test_qdrant_failure_reports_and_recovers PASSED
tests/test_qdrant_and_pipeline.py::test_pipeline_story_clustering_end_to_end PASSED
tests/test_storage.py::test_sqlite_exact_url_dedup_and_source_provenance PASSED
tests/test_summarizer_clustering.py::test_summarizer_empty_input PASSED
tests/test_summarizer_clustering.py::test_summarizer_malformed_json_raises PASSED
tests/test_summarizer_clustering.py::test_summarizer_http_error_raises PASSED
tests/test_summarizer_clustering.py::test_summarizer_cluster_source_integrity PASSED

======================= 26 passed, 2 warnings in 13.60s ========================
```

---

## 6. Next Roadmap & Remaining Opportunities

1. **Full-Text Article Scraping:** Integrate an optional reader mode (e.g. `trafilatura`) to fetch full webpage text when RSS feeds provide only brief excerpts.
2. **Concurrent Polling:** Transition the feed collector from sequential fetching to parallel requests with per-domain rate limiting.
3. **Automated Publishing & Delivery:** Implement email (HTML newsletter), Discord/Slack webhooks, or an RSS output feed for generated digests.
4. **Cloud / Alternative LLM Provider:** Abstract the LLM interface to allow OpenAI/Anthropic/Gemini API endpoints alongside Ollama.
