from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ai_digest.story_clustering.candidate_retriever import CandidateMatch

logger = logging.getLogger(__name__)

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "it", "its", "this", "that", "these", "those", "how", "what",
    "why", "when", "where", "who", "which", "new", "ai", "model", "models",
}

MODEL_REGEX = re.compile(
    r"\b(gpt[-\s]?[0-9a-z\.]+|claude[-\s]?[0-9a-z\.]+|gemini[-\s]?[0-9a-z\.]+|"
    r"llama[-\s]?[0-9a-z\.]+|mistral[-\s]?[0-9a-z\.]+|qwen[-\s]?[0-9a-z\.]+|"
    r"deepseek[-\s]?[0-9a-z\.]+|flux[-\s]?[0-9a-z\.]+|stable[-\s]?diffusion[-\s]?[0-9a-z\.]+|"
    r"dall[-\s]?e[-\s]?[0-9a-z\.]+|grok[-\s]?[0-9a-z\.]+|alphafold[-\s]?[0-9a-z\.]+|v[0-9]+(?:\.[0-9]+)*)\b",
    re.IGNORECASE,
)

RELEASE_TERMS = {
    "release", "releases", "released", "launch", "launches", "launched",
    "announces", "announced", "announce", "announcement", "unveils", "unveiled",
    "unveil", "introduces", "introduced", "introduce", "debuts", "debut"
}

UPDATE_TERMS = {
    "improves", "improved", "improvement", "optimizes", "optimized",
    "optimization", "update", "updates", "updated", "speedup",
    "latency", "inference performance"
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


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 1}


def _extract_models(text: str) -> set[str]:
    matches = MODEL_REGEX.findall(text)
    clean = set()
    for m in matches:
        normalized = re.sub(r"[\s\-_]+", "", m.lower())
        clean.add(normalized)
    return clean


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    story_id: str | None
    similarity_score: float
    assignment_method: str
    reason: str


class SimilarityMatcher:
    """Evaluates candidate matches using semantic, temporal, entity, and title heuristics."""

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_age_days: int = 30,
        ambiguity_margin: float = 0.03,
    ):
        self.similarity_threshold = similarity_threshold
        self.max_age_days = max_age_days
        self.ambiguity_margin = ambiguity_margin

    def evaluate(
        self,
        article: dict[str, Any],
        candidates: list[CandidateMatch],
    ) -> MatchResult:
        if not candidates:
            return MatchResult(
                matched=False,
                story_id=None,
                similarity_score=0.0,
                assignment_method="no_candidates",
                reason="No candidate articles found in vector store",
            )

        # Consider candidates with meaningful candidate similarity or existing story_id
        min_candidate_score = max(0.60, self.similarity_threshold - 0.25)
        valid_candidates = [
            c for c in candidates
            if c.score >= min_candidate_score and c.story_id is not None
        ]

        if not valid_candidates:
            highest_score = max((c.score for c in candidates), default=0.0)
            return MatchResult(
                matched=False,
                story_id=None,
                similarity_score=highest_score,
                assignment_method="below_threshold",
                reason=f"Candidates found but highest score ({highest_score:.3f}) below candidate threshold ({min_candidate_score:.3f})",
            )

        # Group candidates by story_id
        cluster_groups: dict[str, list[CandidateMatch]] = {}
        for c in valid_candidates:
            cluster_groups.setdefault(c.story_id, []).append(c)  # type: ignore[arg-type]

        article_title = article.get("title", "")
        article_text = f"{article_title} {article.get('description', '')}"
        article_tokens = _tokenize(article_title)
        article_models = _extract_models(article_text)
        article_pub_date = _parse_iso_date(article.get("published_at"))

        evaluated_clusters: list[tuple[str, float, CandidateMatch, str]] = []

        for story_id, matches in cluster_groups.items():
            best_match = max(matches, key=lambda m: m.score)
            cand_article = best_match.article_record
            cand_title = cand_article.get("title", "")
            cand_text = f"{cand_title} {cand_article.get('description', '')}"
            cand_tokens = _tokenize(cand_title)
            cand_models = _extract_models(cand_text)
            cand_pub_date = _parse_iso_date(cand_article.get("published_at"))

            # 1. Temporal Check: verify within max_age_days
            days_diff = None
            if article_pub_date and cand_pub_date:
                age_gap = abs(article_pub_date - cand_pub_date)
                if age_gap > timedelta(days=self.max_age_days):
                    logger.debug(
                        "Cluster %s rejected due to age gap: %s > %d days",
                        story_id, age_gap, self.max_age_days
                    )
                    continue
                days_diff = age_gap.days

            # 2. Entity / Model Conflict Check:
            # If both mention model names but have zero common models, reject!
            if article_models and cand_models and not (article_models & cand_models):
                logger.debug(
                    "Cluster %s rejected: conflicting model entities %s vs %s",
                    story_id, article_models, cand_models
                )
                continue

            # 3. Action / Event Type Divergence Check:
            # Initial release vs later optimization weeks/months apart
            article_lower = article_text.lower()
            cand_lower = cand_text.lower()
            is_article_update = any(term in article_lower for term in UPDATE_TERMS)
            is_cand_release = any(term in cand_lower for term in RELEASE_TERMS)
            is_cand_update = any(term in cand_lower for term in UPDATE_TERMS)
            is_article_release = any(term in article_lower for term in RELEASE_TERMS)

            if days_diff is not None and days_diff > 7:
                if (
                    (is_article_update and is_cand_release and not is_article_release)
                    or (is_cand_update and is_article_release and not is_cand_update)
                ):
                    logger.debug(
                        "Cluster %s rejected: distinct event types (release vs update separated by %d days)",
                        story_id, days_diff
                    )
                    continue

            # 4. Multi-factor Confidence Score Calculation
            base_score = best_match.score
            confidence = base_score

            # Shared model name bonus
            has_shared_model = bool(article_models and cand_models and (article_models & cand_models))
            if has_shared_model:
                confidence += 0.10

            # Title token overlap
            overlap = len(article_tokens & cand_tokens)
            union = len(article_tokens | cand_tokens)
            jaccard = (overlap / union) if union > 0 else 0.0

            if jaccard >= 0.35:
                confidence += 0.08
            elif jaccard >= 0.20:
                confidence += 0.04

            # Tight temporal proximity bonus (< 48 hours)
            if days_diff is not None and days_diff <= 2:
                confidence += 0.04

            confidence = min(1.0, confidence)

            # Check if effective confidence reaches the required similarity threshold
            if confidence >= self.similarity_threshold:
                evaluated_clusters.append((
                    story_id,
                    confidence,
                    best_match,
                    (
                        f"Matched cluster {story_id} (confidence: {confidence:.3f}, base vector: {base_score:.3f}, "
                        f"title Jaccard: {jaccard:.2f}, shared models: {has_shared_model})"
                    ),
                ))

        if not evaluated_clusters:
            highest_score = max((c.score for c in valid_candidates), default=0.0)
            return MatchResult(
                matched=False,
                story_id=None,
                similarity_score=highest_score,
                assignment_method="below_threshold",
                reason=f"Candidates found but combined confidence did not meet threshold ({self.similarity_threshold})",
            )

        # Sort evaluated clusters by confidence descending
        evaluated_clusters.sort(key=lambda x: x[1], reverse=True)
        best_story_id, best_confidence, best_match, best_reason = evaluated_clusters[0]

        # 5. Ambiguity Check: compare top 2 candidate clusters
        if len(evaluated_clusters) > 1:
            second_story_id, second_confidence, _, _ = evaluated_clusters[1]
            if best_story_id != second_story_id:
                diff = best_confidence - second_confidence
                if diff < self.ambiguity_margin:
                    return MatchResult(
                        matched=False,
                        story_id=None,
                        similarity_score=best_confidence,
                        assignment_method="ambiguous_clusters",
                        reason=(
                            f"Ambiguous candidate clusters: {best_story_id} ({best_confidence:.3f}) and "
                            f"{second_story_id} ({second_confidence:.3f}) differ by {diff:.3f} "
                            f"(< margin {self.ambiguity_margin})"
                        ),
                    )

        return MatchResult(
            matched=True,
            story_id=best_story_id,
            similarity_score=best_confidence,
            assignment_method="multi_factor_semantic_match",
            reason=best_reason,
        )
