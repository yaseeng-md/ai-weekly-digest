from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any
from datetime import datetime, timedelta, timezone
logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ClusterRepository:
    """Manages persistence and queries for story clusters in SQLite."""

    def __init__(self, store: Any):
        # store can be SQLiteStore or any object providing a .connection() context manager
        self.store = store

    def create_cluster(
        self,
        story_id: str,
        canonical_title: str,
        representative_article_id: str,
        primary_source_name: str = "",
        first_published_at: str | None = None,
        last_published_at: str | None = None,
        status: str = "active",
    ) -> dict[str, Any]:
        now = utc_now()
        with self.store.connection() as conn:
            conn.execute(
                """
                INSERT INTO story_clusters (
                    story_id, canonical_title, representative_article_id,
                    primary_source_name, article_count, created_at, updated_at,
                    first_published_at, last_published_at, status
                ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(story_id) DO UPDATE SET
                    canonical_title = excluded.canonical_title,
                    representative_article_id = excluded.representative_article_id,
                    primary_source_name = excluded.primary_source_name,
                    updated_at = excluded.updated_at
                """,
                (
                    story_id,
                    canonical_title,
                    representative_article_id,
                    primary_source_name,
                    now,
                    now,
                    first_published_at or now,
                    last_published_at or first_published_at or now,
                    status,
                ),
            )
            # Fetch and return newly created/updated cluster
            row = conn.execute(
                "SELECT * FROM story_clusters WHERE story_id = ?", (story_id,)
            ).fetchone()
            return dict(row)

    def get_cluster(self, story_id: str) -> dict[str, Any] | None:
        with self.store.connection() as conn:
            row = conn.execute(
                "SELECT * FROM story_clusters WHERE story_id = ?", (story_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_cluster_by_article(self, article_id: str) -> dict[str, Any] | None:
        with self.store.connection() as conn:
            row = conn.execute(
                """
                SELECT sc.*, sca.similarity_score, sca.assignment_method, sca.assigned_at
                FROM story_clusters sc
                JOIN story_cluster_articles sca ON sc.story_id = sca.story_id
                WHERE sca.article_id = ?
                """,
                (article_id,),
            ).fetchone()
        return dict(row) if row else None

    def assign_article(
        self,
        story_id: str,
        article_id: str,
        similarity_score: float | None = None,
        assignment_method: str = "created_initial",
    ) -> bool:
        """Assign article to cluster idempotently. Returns True if newly assigned."""
        now = utc_now()
        with self.store.connection() as conn:
            existing = conn.execute(
                "SELECT story_id FROM story_cluster_articles WHERE article_id = ?",
                (article_id,),
            ).fetchone()
            if existing:
                if existing["story_id"] == story_id:
                    return False  # Already assigned to this cluster
                # Already assigned to another cluster
                logger.warning(
                    "Article %s already assigned to cluster %s, skipping assignment to %s",
                    article_id,
                    existing["story_id"],
                    story_id,
                )
                return False

            conn.execute(
                """
                INSERT INTO story_cluster_articles (
                    story_id, article_id, similarity_score, assignment_method, assigned_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(article_id) DO NOTHING
                """,
                (story_id, article_id, similarity_score, assignment_method, now),
            )

            # Update cluster metadata: article count, updated_at, last_published_at
            item_row = conn.execute(
                "SELECT published_at FROM items WHERE id = ?", (article_id,)
            ).fetchone()
            pub_date = item_row["published_at"] if item_row else None

            conn.execute(
                """
                UPDATE story_clusters
                SET article_count = (
                    SELECT COUNT(*) FROM story_cluster_articles WHERE story_id = ?
                ),
                updated_at = ?,
                last_published_at = CASE
                    WHEN ? IS NOT NULL AND (last_published_at IS NULL OR ? > last_published_at)
                    THEN ?
                    ELSE last_published_at
                END,
                first_published_at = CASE
                    WHEN ? IS NOT NULL AND (first_published_at IS NULL OR ? < first_published_at)
                    THEN ?
                    ELSE first_published_at
                END
                WHERE story_id = ?
                """,
                (story_id, now, pub_date, pub_date, pub_date, pub_date, pub_date, pub_date, story_id),
            )
            return True

    def get_cluster_articles(self, story_id: str) -> list[dict[str, Any]]:
        with self.store.connection() as conn:
            rows = conn.execute(
                """
                SELECT i.*, sca.similarity_score, sca.assignment_method, sca.assigned_at
                FROM items i
                JOIN story_cluster_articles sca ON i.id = sca.article_id
                WHERE sca.story_id = ?
                ORDER BY sca.assigned_at ASC
                """,
                (story_id,),
            ).fetchall()
        articles = []
        for row in rows:
            data = dict(row)
            data["authors"] = json.loads(data.pop("authors_json", "[]"))
            data["tags"] = json.loads(data.pop("tags_json", "[]"))
            articles.append(data)
        return articles

    def update_cluster_representative(
        self,
        story_id: str,
        representative_article_id: str,
        primary_source_name: str,
        canonical_title: str | None = None,
    ) -> None:
        now = utc_now()
        with self.store.connection() as conn:
            if canonical_title:
                conn.execute(
                    """
                    UPDATE story_clusters
                    SET representative_article_id = ?,
                        primary_source_name = ?,
                        canonical_title = ?,
                        updated_at = ?
                    WHERE story_id = ?
                    """,
                    (representative_article_id, primary_source_name, canonical_title, now, story_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE story_clusters
                    SET representative_article_id = ?,
                        primary_source_name = ?,
                        updated_at = ?
                    WHERE story_id = ?
                    """,
                    (representative_article_id, primary_source_name, now, story_id),
                )

    def list_clusters(
        self, limit: int = 50, offset: int = 0, status: str | None = None
    ) -> list[dict[str, Any]]:
        with self.store.connection() as conn:
            query = "SELECT * FROM story_clusters"
            params: list[Any] = []
            if status:
                query += " WHERE status = ?"
                params.append(status)
            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get_unassigned_article_ids(self) -> list[str]:
        with self.store.connection() as conn:
            rows = conn.execute(
                """
                SELECT i.id
                FROM items i
                LEFT JOIN story_cluster_articles sca ON i.id = sca.article_id
                WHERE sca.article_id IS NULL
                ORDER BY COALESCE(i.published_at, i.created_at) ASC
                """
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def get_pending_cluster_ids(self, lookback_hours: int | None = None) -> list[str]:
        """Return cluster IDs that contain articles with pending or error LLM status."""
        with self.store.connection() as conn:
            if lookback_hours is not None:
                cutoff = (
                    datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
                ).isoformat().replace("+00:00", "Z")
                rows = conn.execute(
                    """
                    SELECT sca.story_id
                    FROM story_cluster_articles sca
                    JOIN items i ON sca.article_id = i.id
                    WHERE i.llm_status IN ('pending', 'error')
                    AND (i.published_at IS NULL OR julianday(i.published_at) >= julianday(?))
                    GROUP BY sca.story_id
                    ORDER BY MIN(sca.assigned_at) ASC
                    """,
                    (cutoff,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT sca.story_id
                    FROM story_cluster_articles sca
                    JOIN items i ON sca.article_id = i.id
                    WHERE i.llm_status IN ('pending', 'error')
                    GROUP BY sca.story_id
                    ORDER BY MIN(sca.assigned_at) ASC
                    """
                ).fetchall()
        return [str(row["story_id"]) for row in rows]

    
    def get_cluster_stats(self) -> dict[str, Any]:
        with self.store.connection() as conn:
            total_clusters = conn.execute(
                "SELECT COUNT(*) FROM story_clusters"
            ).fetchone()[0]
            total_clustered_articles = conn.execute(
                "SELECT COUNT(*) FROM story_cluster_articles"
            ).fetchone()[0]
            multi_source_clusters = conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT story_id FROM story_cluster_articles
                    GROUP BY story_id HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
            unassigned_articles = conn.execute(
                """
                SELECT COUNT(*) FROM items i
                LEFT JOIN story_cluster_articles sca ON i.id = sca.article_id
                WHERE sca.article_id IS NULL
                """
            ).fetchone()[0]
            recent_rows = conn.execute(
                """
                SELECT story_id, canonical_title, primary_source_name, article_count, updated_at
                FROM story_clusters
                ORDER BY updated_at DESC LIMIT 5
                """
            ).fetchall()

        avg_articles = (
            round(total_clustered_articles / total_clusters, 2)
            if total_clusters > 0
            else 0.0
        )
        return {
            "total_clusters": total_clusters,
            "total_clustered_articles": total_clustered_articles,
            "average_articles_per_cluster": avg_articles,
            "multi_source_clusters": multi_source_clusters,
            "unassigned_articles": unassigned_articles,
            "recently_updated_clusters": [dict(r) for r in recent_rows],
        }

    def validate_clusters(self) -> dict[str, Any]:
        """Validate story cluster consistency and identify issues."""
        with self.store.connection() as conn:
            empty_clusters = [
                row["story_id"]
                for row in conn.execute(
                    """
                    SELECT sc.story_id
                    FROM story_clusters sc
                    LEFT JOIN story_cluster_articles sca ON sc.story_id = sca.story_id
                    WHERE sca.article_id IS NULL
                    """
                ).fetchall()
            ]

            missing_articles = [
                row["article_id"]
                for row in conn.execute(
                    """
                    SELECT sca.article_id
                    FROM story_cluster_articles sca
                    LEFT JOIN items i ON sca.article_id = i.id
                    WHERE i.id IS NULL
                    """
                ).fetchall()
            ]

            duplicate_memberships = [
                row["article_id"]
                for row in conn.execute(
                    """
                    SELECT article_id, COUNT(*) as cnt
                    FROM story_cluster_articles
                    GROUP BY article_id HAVING cnt > 1
                    """
                ).fetchall()
            ]

            unassigned = [
                row["id"]
                for row in conn.execute(
                    """
                    SELECT i.id
                    FROM items i
                    LEFT JOIN story_cluster_articles sca ON i.id = sca.article_id
                    WHERE sca.article_id IS NULL
                    """
                ).fetchall()
            ]

            inconsistent_counts = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT sc.story_id, sc.article_count as stored_count, COUNT(sca.article_id) as actual_count
                    FROM story_clusters sc
                    LEFT JOIN story_cluster_articles sca ON sc.story_id = sca.story_id
                    GROUP BY sc.story_id
                    HAVING stored_count != actual_count
                    """
                ).fetchall()
            ]

        is_valid = not (
            empty_clusters
            or missing_articles
            or duplicate_memberships
            or unassigned
            or inconsistent_counts
        )
        return {
            "is_valid": is_valid,
            "empty_clusters_count": len(empty_clusters),
            "empty_clusters": empty_clusters,
            "missing_article_references_count": len(missing_articles),
            "missing_article_references": missing_articles,
            "duplicate_memberships_count": len(duplicate_memberships),
            "duplicate_memberships": duplicate_memberships,
            "unassigned_articles_count": len(unassigned),
            "unassigned_articles": unassigned[:20],
            "inconsistent_counts_count": len(inconsistent_counts),
            "inconsistent_counts": inconsistent_counts,
        }
