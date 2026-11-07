from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_digest.config import Settings, load_feeds
from ai_digest.embeddings import Embedder
from ai_digest.normalize import to_json_text, to_markdown
from ai_digest.qdrant_store import QdrantVectorStore
from ai_digest.rss_collector import RSSCollector
from ai_digest.storage import SQLiteStore
from ai_digest.story_clustering import ClusterManager
from ai_digest.summarizer import OllamaSummarizer

logger = logging.getLogger(__name__)


def _date_slug(value: str | None) -> str:
    if value:
        match = re.match(r"^(\d{4}-\d{2}-\d{2})", value)
        if match:
            return match.group(1)
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _article_for_llm(record: dict[str, Any], max_chars: int) -> dict[str, Any]:
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


def _batch_markdown(batch: dict[str, Any]) -> str:
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


def _digest_markdown(digest: dict[str, Any], batch_path: str | None = None) -> str:
    lines = [
        f"# {digest.get('digest_title') or 'AI Developments Digest'}",
        "",
        f"> Generated: {digest.get('generated_at', '')}",
        "",
        "## Executive summary",
        "",
        digest.get("executive_summary", "No summary available."),
        "",
    ]
    if digest.get("developments"):
        lines.extend(["## Developments", ""])
        for index, item in enumerate(digest["developments"], start=1):
            title = item.get("title", item.get("canonical_title", "Untitled development"))
            article_count = item.get("article_count", len(item.get("sources", [])))
            primary = item.get("primary_source", {})
            primary_name = primary.get("source_name") or primary.get("name") if isinstance(primary, dict) else ""
            primary_url = primary.get("url") if isinstance(primary, dict) else ""

            meta_parts = [
                f"**Category:** {item.get('category', 'AI')}",
                f"**Importance:** {item.get('importance', 3)}/5",
                f"**Maturity:** {item.get('maturity', 'unclear')}",
            ]
            if article_count > 1:
                meta_parts.append(f"**Sources:** {article_count} reporting publications")

            lines.extend([
                f"### {index}. {title}",
                "",
                " | ".join(meta_parts) + "  ",
            ])

            if primary_name and primary_url:
                lines.append(f"**Primary Source:** [{primary_name}]({primary_url})  ")
            lines.append("")

            # Consolidated summary if available
            consolidated = item.get("consolidated_summary")
            if consolidated and consolidated != item.get("what_happened"):
                lines.extend([
                    "**Consolidated Summary**",
                    consolidated,
                    "",
                ])

            lines.extend([
                "**What happened**",
                item.get("what_happened", ""),
                "",
                "**Technical details**",
                item.get("technical_details", ""),
                "",
                "**Why it matters**",
                item.get("why_it_matters", ""),
                "",
            ])

            # Rich multi-source attribution
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


class DigestPipeline:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.store = SQLiteStore(self.settings.sqlite_path)

    def run(self) -> dict[str, Any]:
        feeds = load_feeds(self.settings.feeds_path)
        run_id = str(uuid.uuid4())
        self.store.start_run(run_id, len(feeds))
        fetched_entries = 0
        inserted_items = 0
        exact_duplicates = 0
        semantic_duplicates = 0
        feed_errors: list[str] = []
        new_ids: list[str] = []
        digest_path: str | None = None
        status = "ok"
        collector = RSSCollector(self.settings, self.store)
        embedder: Embedder | None = None
        vectors: QdrantVectorStore | None = None
        summarizer = OllamaSummarizer(self.settings)

        try:
            # 1. Register all configured feeds
            for feed in feeds:
                self.store.register_feed(feed)

            # 2. Collect from feeds
            for feed in feeds:
                try:
                    records, feed_status = collector.fetch_feed(feed)
                    if feed_status == "not_modified":
                        logger.info("Not modified: %s", feed["name"])
                        continue
                    fetched_entries += len(records)
                    for record in records:
                        day = _date_slug(record.get("published_at"))
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
                    message = f"{feed.get('name', feed['url'])}: {exc}"
                    feed_errors.append(message)
                    logger.exception("Could not collect feed %s", feed["url"])

            # 3. Vector Store & Embedding Model Initialization
            embedder = Embedder(self.settings.embedding_model, device=self.settings.embedding_device)
            vectors = QdrantVectorStore(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                collection_name=self.settings.qdrant_collection,
                dimension=embedder.dimension,
            )

            # 4. Multi-Source Story Clustering or Legacy Deduplication
            cluster_manager = ClusterManager(
                self.settings, self.store, vector_store=vectors, embedder=embedder
            )

            pending_ids = self.store.pending_qdrant_ids()
            pending_records = self.store.list_items(pending_ids, include_semantic_duplicates=True)

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
                    # Legacy deduplication path
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

            # 5. LLM Batch Preparation
            batch_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            batch_json_path = self.settings.output_dir / "batches" / f"batch_{batch_stamp}_{run_id[:8]}.json"
            batch_md_path = self.settings.output_dir / "batches" / f"batch_{batch_stamp}_{run_id[:8]}.md"

            if self.settings.story_clustering_enabled:
                clusters = cluster_manager.get_pending_clusters_for_summary()
                all_cluster_articles = [
                    _article_for_llm(art, self.settings.max_article_chars)
                    for cl in clusters
                    for art in cl.get("articles", [])
                ]
                total_articles = len(all_cluster_articles)

                batch_payload = {
                    "schema_version": "2.0",
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "source_format": "story-clusters",
                    "cluster_count": len(clusters),
                    "article_count": total_articles,
                    "clusters": clusters,
                    "articles": all_cluster_articles,
                }
                batch_json_path.write_text(json.dumps(batch_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                batch_md_path.write_text(_batch_markdown(batch_payload), encoding="utf-8")

                llm_batch = json.loads(batch_json_path.read_text(encoding="utf-8"))
                llm_clusters = llm_batch.get("clusters", [])
                digest = self._summarize_in_chunks(summarizer, llm_clusters)
                digest["story_cluster_count"] = len(clusters)
                digest["source_article_count"] = total_articles

                # Mark all articles in summarized clusters as summarized
                summarized_article_ids = [
                    art["id"] for cl in clusters for art in cl.get("articles", []) if "id" in art
                ]
                self.store.set_llm_status(summarized_article_ids, "summarized")
            else:
                # Legacy batch
                llm_input_ids = self.store.pending_llm_ids()
                eligible_records = self.store.list_items(llm_input_ids, include_semantic_duplicates=False)
                articles = [
                    _article_for_llm(json.loads(record["normalized_json"]), self.settings.max_article_chars)
                    for record in eligible_records
                ]
                batch_payload = {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "source_format": "normalized-json",
                    "article_count": len(articles),
                    "articles": articles,
                }
                batch_json_path.write_text(json.dumps(batch_payload, ensure_ascii=False, indent=2), encoding="utf-8")
                batch_md_path.write_text(_batch_markdown(batch_payload), encoding="utf-8")

                llm_batch = json.loads(batch_json_path.read_text(encoding="utf-8"))
                llm_articles = llm_batch["articles"]
                digest = self._summarize_in_chunks(summarizer, llm_articles)
                digest["source_article_count"] = len(articles)
                self.store.set_llm_status([a["id"] for a in articles], "summarized")

            # 6. Digest Generation
            digest["run_id"] = run_id
            digest["input_batch_json"] = str(batch_json_path)
            digest["input_batch_markdown"] = str(batch_md_path)
            digest["semantic_duplicates_skipped"] = semantic_duplicates
            digest["feeds_with_errors"] = feed_errors

            digest_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_UTC")
            digest_json_path = self.settings.output_dir / "digests" / f"digest_{digest_stamp}_{run_id[:8]}.json"
            digest_md_path = self.settings.output_dir / "digests" / f"digest_{digest_stamp}_{run_id[:8]}.md"
            digest_json_path.write_text(json.dumps(digest, ensure_ascii=False, indent=2), encoding="utf-8")
            digest_md_path.write_text(_digest_markdown(digest, str(batch_json_path)), encoding="utf-8")
            digest_path = str(digest_md_path)

            if feed_errors:
                status = "partial"

            self.store.finish_run(
                run_id,
                fetched_entries=fetched_entries,
                inserted_items=inserted_items,
                exact_duplicates=exact_duplicates,
                semantic_duplicates=semantic_duplicates,
                digest_path=digest_path,
                status=status,
                error="\n".join(feed_errors)[:5000] if feed_errors else None,
            )

            return {
                "run_id": run_id,
                "status": status,
                "feeds": len(feeds),
                "feeds_with_errors": len(feed_errors),
                "fetched_entries": fetched_entries,
                "inserted_items": inserted_items,
                "exact_duplicates": exact_duplicates,
                "semantic_duplicates": semantic_duplicates,
                "llm_input_items": len(llm_batch.get("clusters", llm_batch.get("articles", []))),
                "batch_json": str(batch_json_path),
                "batch_markdown": str(batch_md_path),
                "digest_markdown": digest_path,
                "digest_json": str(digest_json_path),
                "feed_errors": feed_errors,
                "storage_stats": self.store.stats(),
            }
        except Exception as exc:
            self.store.finish_run(
                run_id,
                fetched_entries=fetched_entries,
                inserted_items=inserted_items,
                exact_duplicates=exact_duplicates,
                semantic_duplicates=semantic_duplicates,
                digest_path=digest_path,
                status="error",
                error=str(exc)[:5000],
            )
            raise
        finally:
            collector.close()
            summarizer.close()
            if vectors:
                vectors.close()

    def _summarize_in_chunks(
        self, summarizer: OllamaSummarizer, items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not items:
            return summarizer.summarize([])
        chunk_size = max(1, self.settings.max_articles_per_llm_batch)
        if len(items) <= chunk_size:
            return summarizer.summarize(items)

        # Chunk items to keep prompt size bounded
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        outputs = [summarizer.summarize(chunk) for chunk in chunks]
        developments: list[dict[str, Any]] = []
        seen: set[str] = set()

        for output in outputs:
            for development in output.get("developments", []):
                key = re.sub(r"\W+", " ", development.get("title", "").lower()).strip()
                if key and key not in seen:
                    seen.add(key)
                    developments.append(development)

        return {
            "digest_title": "AI Developments Digest",
            "executive_summary": "\n\n".join(
                output.get("executive_summary", "") for output in outputs if output.get("executive_summary")
            ),
            "developments": sorted(developments, key=lambda item: item.get("importance", 3), reverse=True),
            "themes": list(dict.fromkeys(theme for output in outputs for theme in output.get("themes", []))),
            "watchlist": list(dict.fromkeys(item for output in outputs for item in output.get("watchlist", []))),
            "caveats": list(dict.fromkeys(item for output in outputs for item in output.get("caveats", []))),
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "llm_model": self.settings.ollama_model,
            "llm_chunks": len(chunks),
        }
