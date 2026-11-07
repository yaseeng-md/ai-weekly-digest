# AI Developments Digest — RSS/Atom MVP

A local-first pipeline for collecting AI developments from RSS/Atom feeds, normalizing them to JSON/Markdown, storing metadata/content in SQLite, using Qdrant for semantic duplicate detection, and summarizing non-duplicate items with a local Ollama model.

## Architecture

```text
RSS / Atom feeds
      |
      v
HTTP collector (ETag / Last-Modified support)
      |
      v
feedparser parses the feed in memory (raw XML is not saved)
      |
      v
Canonical article record (Python dict -> JSON + Markdown)
      |
      +--> per-item JSON/MD files
      +--> SQLite source of truth, exact-URL dedupe, run state
      +--> SentenceTransformer embedding -> Qdrant similarity check
                                        |
                                        v
                       semantic duplicates flagged / skipped
      |
      v
Normalized JSON + Markdown batch files (created before the LLM call)
      |
      v
Ollama local LLM -> validated structured JSON -> digest JSON + Markdown
```

### Storage responsibilities

- **SQLite** stores feed state, normalized article text/metadata, source provenance, exact URL deduplication, Qdrant indexing state, LLM processing state and run metrics.
- **Qdrant** stores the article embeddings and minimal searchable payload. It is used for semantic similarity, not as the canonical database.
- **`data/items/YYYY-MM-DD/`** stores an individual normalized `.json` and `.md` file for each newly inserted item.
- **`output/batches/`** stores the complete normalized `.json` and `.md` input batch before any content is sent to the LLM.
- **`output/digests/`** stores the generated digest as `.json` and `.md`.

RSS/Atom is XML at the source. `feedparser` parses that response in memory; this project does **not** save raw XML files or pass XML to the LLM.

## Requirements

- Python 3.11 or newer (Python 3.12 is a good choice)
- Docker Engine/Desktop, for Qdrant
- Ollama installed and running for local summarization
- Internet access to fetch feeds and the embedding model on first run

## Quick start

### 1. Create an environment and install the package

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate             # Linux/macOS
# .venv\\Scripts\\Activate.ps1       # Windows PowerShell
python -m pip install --upgrade pip
pip install -e .
cp .env.example .env                  # Linux/macOS
# Copy-Item .env.example .env         # Windows PowerShell
```

### 2. Start Qdrant

```bash
docker compose up -d qdrant
```

- REST API: http://localhost:6333
- Dashboard: http://localhost:6333/dashboard

The Qdrant data is persisted in the Docker named volume `ai_digest_qdrant_storage`.

### 3. Start Ollama and pull the configured model

Install Ollama from its official download page, then run:

```bash
ollama pull qwen2.5:3b
```

Ollama should be accessible at `http://localhost:11434`. To use another local model, set `OLLAMA_MODEL` in `.env` to a model already available in Ollama.

The `sentence-transformers/all-MiniLM-L6-v2` embedding model is separate from the LLM. It is downloaded on first use and creates 384-dimensional vectors.

### 4. Inspect the feed list

```bash
ai-digest feeds
```

Edit `config/feeds.json` to enable/disable feeds or add another feed. Each feed needs `name`, `url`, and optionally `category` and `enabled`.

### 5. Run the pipeline

```bash
ai-digest run
```

Equivalent command if the console script isn't available yet:

```bash
python -m ai_digest.cli run
```

To inspect collection stats:

```bash
ai-digest stats
```

## First-run behavior

- `LOOKBACK_HOURS=168` limits initial ingestion to entries published in the last 7 days. Change it in `.env` if you need a longer initial catch-up period.
- Items without parseable publication dates are retained rather than silently discarded.
- Exact duplicate URLs are not inserted twice. Tracking parameters such as `utm_source` are removed for canonical URL matching.
- Semantic duplicates are detected using cosine similarity in Qdrant. The default threshold is `0.93`; tune this based on observed false positives/negatives. A high threshold is conservative, but the ideal value depends on source content and embedding model.
- Similar stories are retained in SQLite and Qdrant with `duplicate_of` metadata, but semantic duplicates are omitted from the LLM batch. The original article and provenance are not deleted.
- Articles are marked as summarized only after successful LLM summarization. Pending articles are retried on a later run if Ollama failed.
- HTTP `ETag` / `Last-Modified` values are used where supported to reduce repeat downloading. A `304 Not Modified` response is treated as a successful no-change check.

## LLM input and output contracts

The normalized batch has this top-level JSON shape:

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "created_at": "...",
  "source_format": "normalized-json",
  "article_count": 1,
  "articles": [
    {
      "id": "...",
      "title": "...",
      "url": "...",
      "source_name": "...",
      "category": "research_paper",
      "published_at": "...",
      "authors": [],
      "tags": [],
      "description": "...",
      "content": "..."
    }
  ]
}
```

The `.md` batch is a human-readable counterpart. It is written **before** the model request, so you can inspect exactly what was sent. The LLM is asked for structured JSON with per-development summary, technical details, why it matters, maturity, importance, and source URLs. Output source URLs are checked against URLs supplied in the input batch.

## Main configuration

| Variable | Default | Purpose |
|---|---|---|
| `SQLITE_PATH` | `data/digest.db` | SQLite database location |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `ai_developments` | Vector collection |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model for similarity search |
| `EMBEDDING_DEVICE` | `cpu` | Keep embeddings off a small GPU by default |
| `SEMANTIC_DUPLICATE_THRESHOLD` | `0.93` | Minimum cosine similarity to flag a possible duplicate |
| `LOOKBACK_HOURS` | `168` | Initial/recovery feed date window |
| `MAX_ITEMS_PER_FEED` | `60` | Max recent entries processed per feed per poll |
| `LLM_PROVIDER` | `ollama` | Current MVP supports Ollama |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Local summarization model |
| `MAX_ARTICLE_CHARS` | `5000` | Max article-content characters sent per item |
| `MAX_ARTICLES_PER_LLM_BATCH` | `10` | Bound the model prompt size; larger batches are summarized in chunks |
| `STORY_CLUSTERING_ENABLED` | `true` | Enable multi-source story clustering |
| `STORY_CLUSTER_TOP_K` | `10` | Number of candidate articles retrieved from Qdrant |
| `STORY_CLUSTER_SIMILARITY_THRESHOLD` | `0.85` | Multi-factor confidence threshold for cluster matching |
| `STORY_CLUSTER_MAX_AGE_DAYS` | `30` | Maximum publication age difference to consider for merging |
| `STORY_CLUSTER_MAX_ARTICLES` | `10` | Maximum reporting articles included per story cluster |
| `STORY_CLUSTER_MAX_CONTEXT_CHARS` | `30000` | Context character budget per story cluster for LLM |
| `STORY_CLUSTER_AMBIGUITY_MARGIN` | `0.03` | Safety margin between competing clusters to prevent false merges |

## Story Clustering CLI Commands

```bash
# View cluster collection statistics
ai-digest clusters stats

# Inspect a specific story cluster and its articles
ai-digest clusters show <story_id>

# Backfill unassigned/historical articles into story clusters
ai-digest clusters backfill

# Validate story cluster integrity and detect inconsistencies
ai-digest clusters validate
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Known limitations / next steps

1. **Semantic threshold calibration.** While multi-factor matching combines vector similarity, title token overlap, entity extraction, and temporal windows, continuous monitoring against specific domain feeds is recommended.
2. **Full article extraction is not guaranteed.** Feeds that expose only excerpts provide only excerpts to the LLM in this version. A later source-fetching stage can extract full pages where permitted.
3. **Cross-source story clustering is active.** Multi-source reporting across official announcements, news reporting, and technical analyses is consolidated into single developments with full attribution to all sources.
4. Gmail, newsletters that arrive only by email, LinkedIn, Brave Search, scheduled collection, and email delivery are deliberately not implemented in this RSS-first version.
5. Add request rate limiting, feed-level allowlists, and richer observability before deploying as a long-running service.

## Privacy and security

Keep `.env` out of version control. Do not expose Qdrant or Ollama ports publicly without authentication and network controls. Add a `data/` and `.env` entry to `.gitignore` before committing the project.
