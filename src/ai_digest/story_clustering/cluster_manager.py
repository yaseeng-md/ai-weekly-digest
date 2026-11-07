from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from ai_digest.config import Settings
from ai_digest.embeddings import Embedder
from ai_digest.qdrant_store import QdrantVectorStore
from ai_digest.story_clustering.candidate_retriever import CandidateRetriever
from ai_digest.story_clustering.cluster_repository import ClusterRepository
from ai_digest.story_clustering.similarity_matcher import MatchResult, SimilarityMatcher
from ai_digest.story_clustering.source_selector import SourceSelector

logger = logging.getLogger(__name__)


class ClusterManager:
    """Orchestrates story clustering: candidate retrieval, matching, assignment, and source selection."""

    def __init__(
        self,
        settings: Settings,
        store: Any,
        vector_store: QdrantVectorStore | None = None,
        embedder: Embedder | None = None,
    ):
        self.settings = settings
        self.store = store
        self.vector_store = vector_store
        self.embedder = embedder
        self.repository = ClusterRepository(store)
        self.source_selector = SourceSelector()
        self.matcher = SimilarityMatcher(
            similarity_threshold=settings.story_cluster_similarity_threshold,
            max_age_days=settings.story_cluster_max_age_days,
            ambiguity_margin=settings.story_cluster_ambiguity_margin,
        )

    def _get_retriever(self) -> CandidateRetriever:
        if not self.vector_store:
            raise RuntimeError("QdrantVectorStore is required for candidate retrieval")
        return CandidateRetriever(self.vector_store, self.repository, self.store)

    def assign_article(
        self,
        article_record: dict[str, Any],
        vector: list[float] | None = None,
    ) -> tuple[str, bool, MatchResult]:
        """Assign an article to an existing cluster or create a new cluster.
        
        Returns: (story_id, is_new_cluster, match_result)
        """
        article_id = str(article_record["id"])

        # 1. Idempotency check: see if article already belongs to a cluster
        existing_cluster = self.repository.get_cluster_by_article(article_id)
        if existing_cluster:
            story_id = str(existing_cluster["story_id"])
            match = MatchResult(
                matched=True,
                story_id=story_id,
                similarity_score=float(existing_cluster.get("similarity_score") or 1.0),
                assignment_method="idempotent_existing_assignment",
                reason=f"Article {article_id} already assigned to cluster {story_id}",
            )
            return story_id, False, match

        # 2. Check for historical semantic duplicate provenance
        # If this item was previously marked as a duplicate of another item in SQLite
        dup_parent_id = article_record.get("duplicate_of")
        if dup_parent_id:
            parent_cluster = self.repository.get_cluster_by_article(str(dup_parent_id))
            if parent_cluster:
                story_id = str(parent_cluster["story_id"])
                self.repository.assign_article(
                    story_id=story_id,
                    article_id=article_id,
                    similarity_score=1.0,
                    assignment_method="historical_duplicate_linkage",
                )
                self._refresh_primary_source(story_id)
                match = MatchResult(
                    matched=True,
                    story_id=story_id,
                    similarity_score=1.0,
                    assignment_method="historical_duplicate_linkage",
                    reason=f"Assigned via historical duplicate_of link to parent item {dup_parent_id}",
                )
                return story_id, False, match

        # 3. Vector candidate retrieval
        if vector is None:
            if not self.embedder:
                raise RuntimeError("Embedder is required to encode article vector")
            embedding_text = article_record.get("embedding_text") or "\n".join([
                article_record.get("title", ""),
                article_record.get("description", ""),
                article_record.get("content", "")[:2000],
            ])
            vector = self.embedder.encode(embedding_text)

        candidates = []
        if self.vector_store:
            retriever = self._get_retriever()
            candidates = retriever.retrieve_candidates(
                article_id=article_id,
                vector=vector,
                published_at=article_record.get("published_at"),
                limit=self.settings.story_cluster_top_k,
                max_age_days=self.settings.story_cluster_max_age_days,
            )

        # 4. Similarity Matching
        match = self.matcher.evaluate(article_record, candidates)

        if match.matched and match.story_id:
            # Assign to existing matching cluster
            story_id = match.story_id
            self.repository.assign_article(
                story_id=story_id,
                article_id=article_id,
                similarity_score=match.similarity_score,
                assignment_method=match.assignment_method,
            )
            self._refresh_primary_source(story_id)
            logger.info(
                "Assigned article '%s' to cluster %s (score: %.3f, method: %s)",
                article_record.get("title", "")[:50],
                story_id,
                match.similarity_score,
                match.assignment_method,
            )
            return story_id, False, match

        # 5. Create new story cluster
        story_id = f"story_{uuid.uuid4().hex[:12]}"
        title = article_record.get("title") or "Untitled AI development"
        pub_date = article_record.get("published_at")

        self.repository.create_cluster(
            story_id=story_id,
            canonical_title=title,
            representative_article_id=article_id,
            primary_source_name=article_record.get("source_name", ""),
            first_published_at=pub_date,
            last_published_at=pub_date,
            status="active",
        )
        self.repository.assign_article(
            story_id=story_id,
            article_id=article_id,
            similarity_score=1.0,
            assignment_method="initial_cluster_creation",
        )
        logger.info(
            "Created new cluster %s for article '%s'",
            story_id,
            title[:50],
        )
        return story_id, True, match

    def _refresh_primary_source(self, story_id: str) -> None:
        """Re-evaluate primary source for the cluster and update metadata if changed."""
        articles = self.repository.get_cluster_articles(story_id)
        if not articles:
            return

        primary_article, method, rationale = self.source_selector.select_primary(articles)
        self.repository.update_cluster_representative(
            story_id=story_id,
            representative_article_id=str(primary_article["id"]),
            primary_source_name=primary_article.get("source_name", ""),
            canonical_title=primary_article.get("title"),
        )
        logger.debug(
            "Cluster %s primary source set to '%s' via %s (%s)",
            story_id,
            primary_article.get("source_name"),
            method,
            rationale,
        )

    def get_pending_clusters_for_summary(self) -> list[dict[str, Any]]:
        """Prepare story clusters containing articles pending summarization."""
        pending_story_ids = self.repository.get_pending_cluster_ids()
        prepared_clusters: list[dict[str, Any]] = []

        for story_id in pending_story_ids:
            cluster = self.repository.get_cluster(story_id)
            if not cluster:
                continue

            articles = self.repository.get_cluster_articles(story_id)
            if not articles:
                continue

            # Pick or confirm representative source
            primary_article, _, rationale = self.source_selector.select_primary(articles)

            # Order articles: primary first, then others by publication date descending
            other_articles = [a for a in articles if a["id"] != primary_article["id"]]
            other_articles.sort(
                key=lambda x: str(x.get("published_at") or x.get("created_at") or ""),
                reverse=True,
            )
            sorted_articles = [primary_article] + other_articles

            # Cap total articles per cluster
            max_articles = self.settings.story_cluster_max_articles
            capped_articles = sorted_articles[:max_articles]

            # Collect source metadata without hallucination
            sources = []
            allowed_urls = []
            for art in capped_articles:
                url = art.get("url") or art.get("canonical_url", "")
                if url:
                    allowed_urls.append(url)
                sources.append({
                    "id": art["id"],
                    "source_name": art.get("source_name", "Unknown Source"),
                    "title": art.get("title", ""),
                    "url": url,
                    "category": art.get("category", "uncategorized"),
                    "published_at": art.get("published_at"),
                    "is_primary": (art["id"] == primary_article["id"]),
                })

            # Build consolidated context budget
            total_context_budget = self.settings.story_cluster_max_context_chars
            chars_per_article = max(
                1000, total_context_budget // max(1, len(capped_articles))
            )

            prepared_cluster = {
                "story_id": story_id,
                "canonical_title": cluster.get("canonical_title") or primary_article.get("title", ""),
                "category": primary_article.get("category", "AI"),
                "representative_article_id": primary_article["id"],
                "primary_source": {
                    "source_name": primary_article.get("source_name", ""),
                    "url": primary_article.get("url") or primary_article.get("canonical_url", ""),
                    "title": primary_article.get("title", ""),
                    "category": primary_article.get("category", ""),
                    "published_at": primary_article.get("published_at"),
                },
                "sources": sources,
                "source_urls": allowed_urls,
                "article_count": len(articles),
                "capped_article_count": len(capped_articles),
                "first_published_at": cluster.get("first_published_at"),
                "last_published_at": cluster.get("last_published_at"),
                "articles": [
                    {
                        "id": art["id"],
                        "source_name": art.get("source_name", ""),
                        "title": art.get("title", ""),
                        "category": art.get("category", ""),
                        "url": art.get("url") or art.get("canonical_url", ""),
                        "published_at": art.get("published_at"),
                        "description": str(art.get("description") or "")[:1000],
                        "content": (
                            str(art.get("content") or art.get("description") or "")
                        )[:chars_per_article],
                    }
                    for art in capped_articles
                ],
            }
            prepared_clusters.append(prepared_cluster)

        return prepared_clusters

    def backfill_historical_articles(
        self,
        embedder: Embedder | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> dict[str, Any]:
        """Backfill story clusters for historical or unassigned articles."""
        unassigned_ids = self.repository.get_unassigned_article_ids()
        if not unassigned_ids:
            return {
                "total_unassigned": 0,
                "processed": 0,
                "new_clusters_created": 0,
                "assigned_to_existing": 0,
            }

        active_embedder = embedder or self.embedder
        active_vector_store = vector_store or self.vector_store

        if not active_embedder:
            active_embedder = Embedder(
                self.settings.embedding_model, device=self.settings.embedding_device
            )
            self.embedder = active_embedder

        if not active_vector_store:
            active_vector_store = QdrantVectorStore(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                collection_name=self.settings.qdrant_collection,
                dimension=active_embedder.dimension,
            )
            self.vector_store = active_vector_store

        processed = 0
        new_clusters = 0
        assigned_existing = 0

        for article_id in unassigned_ids:
            record = self.store.get_item(article_id)
            if not record:
                continue

            normalized = json.loads(record.get("normalized_json") or "{}")
            embedding_text = normalized.get("embedding_text") or "\n".join([
                record.get("title", ""),
                record.get("description", ""),
                record.get("content", "")[:2000],
            ])
            vector = active_embedder.encode(embedding_text)

            story_id, is_new, match = self.assign_article(record, vector=vector)
            if is_new:
                new_clusters += 1
            else:
                assigned_existing += 1

            # Ensure Qdrant point contains story_id
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
            }
            active_vector_store.upsert(record["id"], vector, payload)
            self.store.mark_qdrant_indexed(record["id"])
            processed += 1

        return {
            "total_unassigned": len(unassigned_ids),
            "processed": processed,
            "new_clusters_created": new_clusters,
            "assigned_to_existing": assigned_existing,
        }

    def validate_clusters(self) -> dict[str, Any]:
        return self.repository.validate_clusters()

    def get_cluster_stats(self) -> dict[str, Any]:
        return self.repository.get_cluster_stats()
