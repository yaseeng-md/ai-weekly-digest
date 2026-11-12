from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_digest.config import Settings
from ai_digest.embeddings import Embedder
from ai_digest.normalize import to_json_text, to_markdown
from ai_digest.notion_publisher import NotionPublisher
from ai_digest.qdrant_store import QdrantVectorStore
from ai_digest.rss_collector import RSSCollector
from ai_digest.storage import SQLiteStore
from ai_digest.story_clustering import ClusterManager
from ai_digest.summarizer import OllamaSummarizer

logger = logging.getLogger(__name__)


def date_slug(value: str | None) -> str:
    if value:
        match = re.match(r"^(\d{4}-\d{2}-\d{2})", value)
        if match:
            return match.group(1)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def article_for_llm(record: dict[str, Any], max_chars: int) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "title": record.get("title", ""),
        "url": record.get("url", ""),
        "source_name": record.get("source_name", ""),
        "category": record.get("category", "uncategorized"),
        "published_at": record.get("published_at"),
        "authors": record.get("authors", []),
        "tags": record.get("tags", []),
        "description": record.get("description", "")[:1500],
        "content": (record.get("content") or record.get("description") or "")[:max_chars],
    }


def batch_markdown(batch: dict[str, Any]) -> str:
    created_at = batch.get("created_at", "")
    clusters = batch.get("clusters")

    if clusters:
        lines = [
            f"# Normalized Story Clusters Batch — {created_at}",
            "",
            f"- **Story Clusters:** {len(clusters)}",
            f"- **Total Articles:** {batch.get('article_count', len(batch.get('articles', [])))}",
            "- **Purpose:** Pre-LLM normalized story-cluster input synthesizing multiple sources.",
            "",
        ]
        for idx, cl in enumerate(clusters, start=1):
            primary = cl.get("primary_source", {})
            lines.extend([
                f"## Cluster {idx}: {cl.get('canonical_title') or 'Untitled development'}",
                "",
                f"- **Story ID:** `{cl.get('story_id')}`",
                f"- **Category:** {cl.get('category', 'AI')}",
                f"- **Articles Count:** {cl.get('article_count', len(cl.get('articles', [])))}",
                f"- **Primary Source:** {primary.get('source_name', 'Unknown')} ({primary.get('url', 'No URL')})",
                "",
                "### Clustered Sources",
                "",
            ])
            for art in cl.get("articles", []):
                lines.extend([
                    f"- **[{art.get('source_name', 'Source')}] {art.get('title', 'Untitled')}**",
                    f"  - URL: {art.get('url', 'None')}",
                    f"  - Published: {art.get('published_at', 'Unknown')}",
                ])
                desc = art.get("description", "")
                if desc:
                    lines.append(f"  - Summary: {desc[:300]}")
            lines.extend(["", "---", ""])
        return "\n".join(lines).rstrip() + "\n"

    # Legacy article-based batch
    lines = [
        f"# Normalized Article Batch — {created_at}",
        "",
        f"- **Articles:** {len(batch.get('articles', []))}",
        "- **Purpose:** Pre-LLM normalized input; this file is JSON/Markdown, not raw RSS XML.",
        "",
    ]
    for index, article in enumerate(batch.get("articles", []), start=1):
        lines.extend([
            f"## {index}. {article.get('title') or 'Untitled'}",
            "",
            f"- **Source:** {article.get('source_name', '')}",
            f"- **Category:** {article.get('category', '')}",
            f"- **Published:** {article.get('published_at') or 'Unknown'}",
            f"- **URL:** {article.get('url') or 'Not provided'}",
            "",
            str(article.get("description") or ""),
            "",
        ])
        content = str(article.get("content") or "")
        if content and content != article.get("description"):
            lines.extend(["### Available article content", "", content, ""])
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def digest_markdown(digest: dict[str, Any], batch_path: str | None = None) -> str:
    lines = [
        f"# {digest.get('digest_title') or 'AI Developments Digest'}",
        "",
        f"> Generated: {digest.get('generated_at', '')}",
        "",
    ]
    exec_summary = str(digest.get("executive_summary") or "").strip()
    if exec_summary and exec_summary != "No summary available.":
        lines.extend([
            "## Executive summary",
            "",
            exec_summary,
            "",
        ])
    if digest.get("developments"):
        lines.extend(["## Developments", ""])
        for index, item in enumerate(digest["developments"], start=1):
            title = item.get("title", item.get("canonical_title", "Untitled development"))
            primary = item.get("primary_source", {})
            primary_name = (
                primary.get("source_name")
                or primary.get("name")
                if isinstance(primary, dict)
                else ""
            )
            primary_url = primary.get("url") if isinstance(primary, dict) else ""
            importance = item.get("importance", 3)

            lines.extend([
                f"### {index}. {title}",
                "",
                f"**Importance:** {importance}/5  ",
            ])

            if primary_name and primary_url:
                lines.append(f"**Primary Source:** [{primary_name}]({primary_url})  ")
            elif primary_name:
                lines.append(f"**Primary Source:** {primary_name}  ")
            lines.append("")

            # Consolidated summary
            consolidated = (
                item.get("consolidated_summary")
                or item.get("what_happened")
                or ""
            ).strip()
            if consolidated:
                lines.extend([
                    "**Consolidated Summary**",
                    consolidated,
                    "",
                ])

            # Technical details
            tech_details = str(item.get("technical_details", "")).strip()
            if tech_details:
                lines.extend([
                    "**Technical details**",
                    tech_details,
                    "",
                ])

            # Multi-source attribution
            sources = item.get("sources", [])
            urls = item.get("source_urls", [])

            if sources:
                lines.append("**Sources**")
                for src in sources:
                    s_name = src.get("source_name", "Source")
                    s_title = src.get("title")
                    s_url = src.get("url")
                    is_prim = src.get("is_primary", False)
                    label = f"{s_name}: {s_title}" if s_title else s_name
                    badge = " *(Primary Source)*" if is_prim else ""
                    if s_url:
                        lines.append(f"- [{label}]({s_url}){badge}")
                    else:
                        lines.append(f"- {label}{badge}")
                lines.append("")
            elif urls:
                lines.append("**Sources**")
                lines.extend([f"- [{url}]({url})" for url in urls])
                lines.append("")

    for key, heading in (("themes", "Emerging themes"), ("watchlist", "What to watch"), ("caveats", "Caveats")):
        values = digest.get(key, [])
        if values:
            lines.extend([f"## {heading}", ""])
            lines.extend([f"- {value}" for value in values])
            lines.append("")
    if batch_path:
        lines.extend(["---", "", f"Normalized input batch: `{batch_path}`", ""])
    return "\n".join(lines).rstrip() + "\n"


class DigestGraphNodes:
    """Encapsulates LangGraph node implementations and shared service state."""

    def __init__(
        self,
        settings: Settings,
        store: SQLiteStore,
        collector: RSSCollector | None = None,
        embedder: Embedder | None = None,
        vector_store: QdrantVectorStore | None = None,
        summarizer: OllamaSummarizer | None = None,
        notion_publisher: NotionPublisher | None = None,
    ):
        self.settings = settings
        self.store = store
        self.collector = collector or RSSCollector(settings, store)
        self.embedder = embedder
        self.vector_store = vector_store
        self.summarizer = summarizer or OllamaSummarizer(settings)
        self.notion_publisher = notion_publisher or NotionPublisher(
            api_key=settings.notion_api_key,
            root_page_id=settings.notion_root_page_id,
        )

    def _get_embedder(self) -> Embedder:
        if self.embedder is None:
            self.embedder = Embedder(
                self.settings.embedding_model,
                device=self.settings.embedding_device,
            )
        return self.embedder

    def _get_vector_store(self, dimension: int) -> QdrantVectorStore:
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                collection_name=self.settings.qdrant_collection,
                dimension=dimension,
            )
        return self.vector_store

    def fetch_feeds_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Node 1: Fetch entries from configured RSS/Atom feeds and persist in SQLite."""
        feeds = state.get("feeds", [])
        run_id = state.get("run_id") or str(uuid.uuid4())
        self.store.start_run(run_id, len(feeds))

        for feed in feeds:
            self.store.register_feed(feed)

        fetched_entries = 0
        inserted_items = 0
        exact_duplicates = 0
        new_ids: list[str] = []
        feed_errors: list[str] = []

        for feed in feeds:
            try:
                records, feed_status = self.collector.fetch_feed(feed)
                if feed_status == "not_modified":
                    logger.info("Not modified: %s", feed["name"])
                    continue
                fetched_entries += len(records)
                for record in records:
                    day = date_slug(record.get("published_at"))
                    item_dir = self.settings.items_dir / day
                    item_dir.mkdir(parents=True, exist_ok=True)
                    json_path = item_dir / f"{record['id']}.json"
                    markdown_path = item_dir / f"{record['id']}.md"
                    inserted, item_id = self.store.add_item(
                        record, str(json_path), str(markdown_path)
                    )
                    if not inserted:
                        exact_duplicates += 1
                        continue
                    json_path.write_text(to_json_text(record), encoding="utf-8")
                    markdown_path.write_text(to_markdown(record), encoding="utf-8")
                    inserted_items += 1
                    new_ids.append(item_id)
            except Exception as exc:
                message = f"{feed.get('name', feed.get('url'))}: {exc}"
                feed_errors.append(message)
                logger.exception("Could not collect feed %s", feed.get("url"))

        return {
            "run_id": run_id,
            "fetched_entries": fetched_entries,
            "inserted_items": inserted_items,
            "exact_duplicates": exact_duplicates,
            "new_ids": new_ids,
            "feed_errors": feed_errors,
        }

    def embed_and_index_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Node 2: Generate SentenceTransformers embeddings and upsert vectors into Qdrant."""
        pending_ids = self.store.pending_qdrant_ids()
        if not pending_ids:
            return {"semantic_duplicates": 0}

        embedder = self._get_embedder()
        vectors = self._get_vector_store(embedder.dimension)
        cluster_manager = ClusterManager(
            self.settings, self.store, vector_store=vectors, embedder=embedder
        )

        pending_records = self.store.list_items(pending_ids, include_semantic_duplicates=True)
        new_ids = set(state.get("new_ids", []))
        semantic_duplicates = 0

        for record in pending_records:
            normalized = json.loads(record.get("normalized_json") or "{}")
            embedding_text = normalized.get("embedding_text") or "\n".join([
                record.get("title", ""),
                record.get("description", ""),
                record.get("content", "")[:2000],
            ])
            vector = embedder.encode(embedding_text)

            if self.settings.story_clustering_enabled:
                story_id, is_new, match = cluster_manager.assign_article(record, vector=vector)
                payload = {
                    "item_id": record["id"],
                    "story_id": story_id,
                    "title": record.get("title", ""),
                    "url": record.get("url", ""),
                    "canonical_url": record.get("canonical_url", ""),
                    "source_name": record.get("source_name", ""),
                    "category": record.get("category", ""),
                    "published_at": record.get("published_at"),
                    "semantic_duplicate": not is_new,
                    "duplicate_of": None,
                }
                vectors.upsert(record["id"], vector, payload)
                self.store.mark_qdrant_indexed(record["id"])
                if not is_new and record["id"] in new_ids:
                    semantic_duplicates += 1
            else:
                matches = vectors.find_similar(vector, limit=self.settings.semantic_search_limit)
                duplicate_of = None
                for match in matches:
                    if match["score"] < self.settings.semantic_duplicate_threshold:
                        continue
                    candidate_id = str(match["payload"].get("item_id") or match["id"])
                    if candidate_id == record["id"]:
                        continue
                    if self.store.get_item(candidate_id) is not None:
                        duplicate_of = candidate_id
                        break
                self.store.mark_semantic_duplicate(record["id"], duplicate_of)
                payload = {
                    "item_id": record["id"],
                    "title": record.get("title", ""),
                    "url": record.get("url", ""),
                    "canonical_url": record.get("canonical_url", ""),
                    "source_name": record.get("source_name", ""),
                    "category": record.get("category", ""),
                    "published_at": record.get("published_at"),
                    "semantic_duplicate": bool(duplicate_of),
                    "duplicate_of": duplicate_of,
                }
                vectors.upsert(record["id"], vector, payload)
                self.store.mark_qdrant_indexed(record["id"])
                if duplicate_of and record["id"] in new_ids:
                    semantic_duplicates += 1

        return {"semantic_duplicates": semantic_duplicates}

    def cluster_stories_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Node 3: Retrieve pending story clusters or legacy articles for LLM batching."""
        if self.settings.story_clustering_enabled:
            embedder = self._get_embedder()
            vectors = self._get_vector_store(embedder.dimension)
            cluster_manager = ClusterManager(
                self.settings, self.store, vector_store=vectors, embedder=embedder
            )
            pending_clusters = cluster_manager.get_pending_clusters_for_summary()
            return {"pending_clusters": pending_clusters, "pending_articles": []}
        else:
            pending_ids = self.store.pending_llm_ids()
            eligible_records = self.store.list_items(pending_ids, include_semantic_duplicates=False)
            pending_articles = [
                article_for_llm(json.loads(record["normalized_json"]), self.settings.max_article_chars)
                for record in eligible_records
            ]
            return {"pending_clusters": [], "pending_articles": pending_articles}

    def batch_and_summarize_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Node 4: Token-aware batch preparation and Ollama LLM synthesis."""
        run_id = state["run_id"]
        batch_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        batch_json_path = self.settings.output_dir / "batches" / f"batch_{batch_stamp}_{run_id[:8]}.json"
        batch_md_path = self.settings.output_dir / "batches" / f"batch_{batch_stamp}_{run_id[:8]}.md"

        clusters = state.get("pending_clusters", [])
        articles = state.get("pending_articles", [])

        if self.settings.story_clustering_enabled and clusters:
            all_cluster_articles = [
                article_for_llm(art, self.settings.max_article_chars)
                for cl in clusters
                for art in cl.get("articles", [])
            ]
            batch_payload = {
                "schema_version": "2.0",
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "source_format": "story-clusters",
                "cluster_count": len(clusters),
                "article_count": len(all_cluster_articles),
                "clusters": clusters,
                "articles": all_cluster_articles,
            }
            batch_json_path.write_text(json.dumps(batch_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            batch_md_path.write_text(batch_markdown(batch_payload), encoding="utf-8")

            llm_batch = json.loads(batch_json_path.read_text(encoding="utf-8"))
            llm_clusters = llm_batch.get("clusters", [])
            digest = self.summarizer.summarize(llm_clusters)
            digest["story_cluster_count"] = len(clusters)
            digest["source_article_count"] = len(all_cluster_articles)

            summarized_article_ids = [
                art["id"] for cl in clusters for art in cl.get("articles", []) if "id" in art
            ]
            self.store.set_llm_status(summarized_article_ids, "summarized")
            llm_input_count = len(llm_clusters)
        else:
            batch_payload = {
                "schema_version": "1.0",
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "source_format": "normalized-json",
                "article_count": len(articles),
                "articles": articles,
            }
            batch_json_path.write_text(json.dumps(batch_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            batch_md_path.write_text(batch_markdown(batch_payload), encoding="utf-8")

            llm_batch = json.loads(batch_json_path.read_text(encoding="utf-8"))
            llm_articles = llm_batch.get("articles", [])
            digest = self.summarizer.summarize(llm_articles)
            digest["source_article_count"] = len(articles)

            self.store.set_llm_status([a["id"] for a in articles], "summarized")
            llm_input_count = len(llm_articles)

        return {
            "batch_json_path": str(batch_json_path),
            "batch_md_path": str(batch_md_path),
            "llm_input_count": llm_input_count,
            "digest": digest,
        }

    def generate_digest_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Node 5: Emit formatted Markdown and JSON digest files."""
        run_id = state["run_id"]
        digest = state.get("digest") or {
            "digest_title": "AI Developments Digest",
            "executive_summary": "No developments to summarize.",
            "developments": [],
            "themes": [],
            "watchlist": [],
            "caveats": [],
        }
        batch_json_path = state.get("batch_json_path")
        batch_md_path = state.get("batch_md_path")

        digest["run_id"] = run_id
        digest["input_batch_json"] = batch_json_path
        digest["input_batch_markdown"] = batch_md_path
        digest["semantic_duplicates_skipped"] = state.get("semantic_duplicates", 0)
        digest["feeds_with_errors"] = state.get("feed_errors", [])

        digest_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
        digest_json_path = self.settings.output_dir / "digests" / f"digest_{digest_stamp}_{run_id[:8]}.json"
        digest_md_path = self.settings.output_dir / "digests" / f"digest_{digest_stamp}_{run_id[:8]}.md"

        digest_json_path.write_text(json.dumps(digest, ensure_ascii=False, indent=2), encoding="utf-8")
        digest_md_path.write_text(digest_markdown(digest, batch_json_path), encoding="utf-8")

        return {
            "digest": digest,
            "digest_json_path": str(digest_json_path),
            "digest_md_path": str(digest_md_path),
        }

    def publish_notion_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Node 6: Publish Markdown digest to Notion under ai-weekly-digest."""
        if not self.settings.notion_enabled or not self.notion_publisher.is_configured():
            logger.info("Notion publishing skipped (disabled or key missing).")
            return {"notion_result": {"status": "skipped", "reason": "Notion not configured or disabled"}}

        digest_md_path = state.get("digest_md_path")
        if not digest_md_path or not Path(digest_md_path).exists():
            return {"notion_result": {"status": "skipped", "reason": "No digest markdown file available"}}

        markdown_content = Path(digest_md_path).read_text(encoding="utf-8")
        digest_type = state.get("digest_type") or self.settings.digest_type or "daily"

        try:
            result = self.notion_publisher.publish_digest(
                markdown_content=markdown_content,
                digest_type=digest_type,
            )
            return {"notion_result": result}
        except Exception as exc:
            logger.exception("Failed to publish digest to Notion")
            return {"notion_result": {"status": "error", "error": str(exc)}}

    def finalize_run_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Node 7: Finalize SQLite run status, capture errors, and format output dict."""
        run_id = state["run_id"]
        feed_errors = state.get("feed_errors", [])
        status = "partial" if feed_errors else "ok"
        digest_path = state.get("digest_md_path")

        self.store.finish_run(
            run_id,
            fetched_entries=state.get("fetched_entries", 0),
            inserted_items=state.get("inserted_items", 0),
            exact_duplicates=state.get("exact_duplicates", 0),
            semantic_duplicates=state.get("semantic_duplicates", 0),
            digest_path=digest_path,
            status=status,
            error="\n".join(feed_errors)[:5000] if feed_errors else None,
        )

        final_result = {
            "run_id": run_id,
            "status": status,
            "feeds": len(state.get("feeds", [])),
            "feeds_with_errors": len(feed_errors),
            "fetched_entries": state.get("fetched_entries", 0),
            "inserted_items": state.get("inserted_items", 0),
            "exact_duplicates": state.get("exact_duplicates", 0),
            "semantic_duplicates": state.get("semantic_duplicates", 0),
            "llm_input_items": state.get("llm_input_count", 0),
            "batch_json": state.get("batch_json_path"),
            "batch_markdown": state.get("batch_md_path"),
            "digest_markdown": digest_path,
            "digest_json": state.get("digest_json_path"),
            "feed_errors": feed_errors,
            "notion": state.get("notion_result", {}),
            "storage_stats": self.store.stats(),
        }

        return {"final_result": final_result, "status": status}

    def close(self) -> None:
        if self.collector:
            self.collector.close()
        if self.summarizer:
            self.summarizer.close()
        if self.vector_store:
            self.vector_store.close()
