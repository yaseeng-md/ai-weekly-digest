from __future__ import annotations

import logging
import uuid
from typing import Any

from ai_digest.config import Settings, load_feeds
from ai_digest.embeddings import Embedder
from ai_digest.graph.nodes import (
    DigestGraphNodes,
    article_for_llm,
    batch_markdown,
    date_slug,
    digest_markdown,
)
from ai_digest.graph.workflow import build_digest_graph
from ai_digest.notion_publisher import NotionPublisher
from ai_digest.qdrant_store import QdrantVectorStore
from ai_digest.rss_collector import RSSCollector
from ai_digest.storage import SQLiteStore
from ai_digest.story_clustering import ClusterManager
from ai_digest.summarizer import OllamaSummarizer

logger = logging.getLogger(__name__)

# Preserved aliases for internal and test backwards compatibility
_date_slug = date_slug
_article_for_llm = article_for_llm
_batch_markdown = batch_markdown
_digest_markdown = digest_markdown


class DigestPipeline:
    """
    LangGraph-based AI Developments Digest Pipeline.
    Coordinates collection, vector indexing, multi-source clustering,
    dynamic batching, LLM summarization, and Notion publishing as a state graph.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.settings.ensure_directories()
        self.store = SQLiteStore(self.settings.sqlite_path)

    def run(self, digest_type: str | None = None) -> dict[str, Any]:
        """
        Execute the full LangGraph digest pipeline.

        Args:
            digest_type: 'daily', 'weekly', or 'both' (defaults to settings.digest_type).

        Returns:
            Dictionary containing run metrics, file outputs, and Notion status.
        """
        feeds = load_feeds(self.settings.feeds_path)
        run_id = str(uuid.uuid4())
        chosen_digest_type = digest_type or self.settings.digest_type or "daily"

        collector = RSSCollector(self.settings, self.store)
        summarizer = OllamaSummarizer(self.settings)
        notion_pub = NotionPublisher(
            api_key=self.settings.notion_api_key,
            root_page_id=self.settings.notion_root_page_id,
        )

        node_handler = DigestGraphNodes(
            settings=self.settings,
            store=self.store,
            collector=collector,
            summarizer=summarizer,
            notion_publisher=notion_pub,
        )

        graph = build_digest_graph(node_handler)

        initial_state: dict[str, Any] = {
            "run_id": run_id,
            "digest_type": chosen_digest_type,
            "feeds": feeds,
            "fetched_entries": 0,
            "inserted_items": 0,
            "exact_duplicates": 0,
            "feed_errors": [],
            "new_ids": [],
            "semantic_duplicates": 0,
            "pending_clusters": [],
            "pending_articles": [],
        }

        try:
            final_state = graph.invoke(initial_state)
            return final_state.get("final_result", {})
        except Exception as exc:
            self.store.finish_run(
                run_id,
                fetched_entries=0,
                inserted_items=0,
                exact_duplicates=0,
                semantic_duplicates=0,
                digest_path=None,
                status="error",
                error=str(exc)[:5000],
            )
            raise
        finally:
            node_handler.close()

    def _summarize_in_chunks(
        self, summarizer: OllamaSummarizer, items: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Summarize items using token-aware dynamic batching handled by OllamaSummarizer."""
        return summarizer.summarize(items)
