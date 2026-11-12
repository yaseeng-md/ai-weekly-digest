from __future__ import annotations

from typing import Any, TypedDict


class DigestState(TypedDict, total=False):
    """Execution state across all LangGraph nodes in the AI Digest pipeline."""

    # Execution configuration & metadata
    run_id: str
    digest_type: str  # 'daily', 'weekly', or 'both'
    feeds: list[dict[str, Any]]

    # Ingestion phase
    fetched_entries: int
    inserted_items: int
    exact_duplicates: int
    feed_errors: list[str]
    new_ids: list[str]

    # Indexing & Clustering phase
    semantic_duplicates: int
    pending_clusters: list[dict[str, Any]]
    pending_articles: list[dict[str, Any]]

    # Batching & LLM phase
    batch_json_path: str
    batch_md_path: str
    llm_input_count: int
    digest: dict[str, Any]

    # Digest Output files
    digest_json_path: str
    digest_md_path: str

    # Notion Publishing
    notion_result: dict[str, Any]

    # Status & Final Summary
    status: str
    error: str | None
    final_result: dict[str, Any]
