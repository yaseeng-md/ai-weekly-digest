from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

DEFAULT_CATEGORY_PRIORITIES: dict[str, int] = {
    "official_release": 100,
    "official_research": 90,
    "open_source": 80,
    "research_paper": 75,
    "ai_infrastructure": 70,
    "developer_tools": 60,
    "industry_analysis": 55,
    "industry_news": 50,
    "newsletter": 40,
    "uncategorized": 30,
}


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


class SourceSelector:
    """Evaluates articles in a cluster to select the representative/primary source."""

    def __init__(self, category_priorities: dict[str, int] | None = None):
        self.category_priorities = category_priorities or DEFAULT_CATEGORY_PRIORITIES

    def score_article(self, article: dict[str, Any]) -> tuple[float, list[str]]:
        """Calculate source score and rationale components for an article."""
        score = 0.0
        rationale_parts: list[str] = []

        category = str(article.get("category", "uncategorized")).lower()
        cat_score = self.category_priorities.get(
            category, self.category_priorities.get("uncategorized", 30)
        )
        score += cat_score
        rationale_parts.append(f"category '{category}' (+{cat_score})")

        # Official publisher / primary source heuristics
        source_name = str(article.get("source_name", "")).lower()
        title = str(article.get("title", "")).lower()

        # Check for official release keywords in source name
        if any(term in source_name for term in ("news", "blog", "research", "official", "announcement")):
            score += 5.0
            rationale_parts.append("official channel indicator (+5)")

        # Content richness: substantive content preferred over bare excerpts
        content_len = len(str(article.get("content") or article.get("description") or ""))
        if content_len > 1500:
            score += 10.0
            rationale_parts.append("comprehensive content (+10)")
        elif content_len > 500:
            score += 5.0
            rationale_parts.append("moderate content (+5)")

        # Timeliness / first reporting bonus
        pub_date = _parse_iso_date(article.get("published_at"))
        if pub_date:
            score += 2.0
            rationale_parts.append("verified publication date (+2)")

        return score, rationale_parts

    def select_primary(
        self, articles: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], str, str]:
        """Select representative article from a list of cluster articles.
        
        Returns: (representative_article, selection_method, selection_rationale)
        """
        if not articles:
            raise ValueError("Cannot select primary source from empty articles list")

        if len(articles) == 1:
            article = articles[0]
            cat = article.get("category", "uncategorized")
            return (
                article,
                "sole_source",
                f"Initial and only source in cluster (category: {cat})",
            )

        scored: list[tuple[float, datetime | None, dict[str, Any], list[str]]] = []
        for article in articles:
            score, rationale = self.score_article(article)
            pub_date = _parse_iso_date(article.get("published_at"))
            scored.append((score, pub_date, article, rationale))

        # Sort criteria:
        # 1. Higher score first
        # 2. Earlier publication date first (if equal score)
        # 3. Stable sort by article ID
        def sort_key(item: tuple[float, datetime | None, dict[str, Any], list[str]]) -> tuple[float, float, str]:
            score, pub_date, article, _ = item
            # Invert score for descending
            date_timestamp = pub_date.timestamp() if pub_date else 9999999999.0
            return (-score, date_timestamp, str(article.get("id", "")))

        scored.sort(key=sort_key)
        best_score, best_date, best_article, best_rationale = scored[0]

        selection_method = (
            "source_priority_and_authority"
            if best_score >= 80
            else "temporal_and_content_quality"
        )
        rationale = (
            f"Selected as primary source with score {best_score:.1f}: "
            + "; ".join(best_rationale)
        )

        return best_article, selection_method, rationale
