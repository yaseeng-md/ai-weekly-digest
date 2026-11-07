from __future__ import annotations

from ai_digest.story_clustering.candidate_retriever import CandidateMatch, CandidateRetriever
from ai_digest.story_clustering.cluster_manager import ClusterManager
from ai_digest.story_clustering.cluster_repository import ClusterRepository
from ai_digest.story_clustering.similarity_matcher import MatchResult, SimilarityMatcher
from ai_digest.story_clustering.source_selector import SourceSelector

__all__ = [
    "CandidateMatch",
    "CandidateRetriever",
    "ClusterManager",
    "ClusterRepository",
    "MatchResult",
    "SimilarityMatcher",
    "SourceSelector",
]
