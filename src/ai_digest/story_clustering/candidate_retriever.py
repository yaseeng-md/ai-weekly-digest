from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ai_digest.qdrant_store import QdrantVectorStore
from ai_digest.story_clustering.cluster_repository import ClusterRepository

logger = logging.getLogger(__name__)


def _parse_iso_date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


@dataclass(frozen=True)
class CandidateMatch:
    article_id: str
    score: float
    story_id: str | None
    article_record: dict[str, Any]
    cluster_record: dict[str, Any] | None
    payload: dict[str, Any]


class CandidateRetriever:
    """Retrieves semantically similar article candidates from Qdrant and resolves SQLite cluster assignments."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        cluster_repo: ClusterRepository,
        sqlite_store: Any,
    ):
        self.vector_store = vector_store
        self.cluster_repo = cluster_repo
        self.sqlite_store = sqlite_store

    def retrieve_candidates(
        self,
        article_id: str,
        vector: list[float],
        published_at: str | None = None,
        limit: int = 10,
        max_age_days: int = 30,
    ) -> list[CandidateMatch]:
        """Search Qdrant for similar items and resolve their story-cluster assignments."""
        # Query more points than limit to account for self and age filtering
        search_limit = max(limit * 3, 20)
        try:
            raw_matches = self.vector_store.find_similar(vector, limit=search_limit)
        except Exception as exc:
            logger.error("Failed to query Qdrant for article %s: %s", article_id, exc)
            raise

        article_date = _parse_iso_date(published_at)
        candidate_matches: list[CandidateMatch] = []

        for match in raw_matches:
            candidate_id = str(match["payload"].get("item_id") or match["id"])

            # Never match against self
            if candidate_id == article_id:
                continue

            # Load full record from SQLite source of truth
            article_record = self.sqlite_store.get_item(candidate_id)
            if not article_record:
                continue

            # Temporal filtering: verify candidate falls within max_age_days if dates are known
            candidate_date = _parse_iso_date(article_record.get("published_at"))
            if article_date and candidate_date:
                age_diff = abs(article_date - candidate_date)
                if age_diff > timedelta(days=max_age_days):
                    logger.debug(
                        "Candidate %s rejected: age difference %s > %d days",
                        candidate_id,
                        age_diff,
                        max_age_days,
                    )
                    continue

            # Resolve story cluster assignment from SQLite
            cluster_row = self.cluster_repo.get_cluster_by_article(candidate_id)
            story_id = cluster_row["story_id"] if cluster_row else None

            candidate_matches.append(
                CandidateMatch(
                    article_id=candidate_id,
                    score=match["score"],
                    story_id=story_id,
                    article_record=article_record,
                    cluster_record=cluster_row,
                    payload=match.get("payload", {}),
                )
            )

            if len(candidate_matches) >= limit:
                break

        return candidate_matches
