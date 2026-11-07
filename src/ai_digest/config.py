from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _path_from_env(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    sqlite_path: Path = _path_from_env("SQLITE_PATH", "data/digest.db")
    output_dir: Path = _path_from_env("OUTPUT_DIR", "output")
    items_dir: Path = _path_from_env("ITEMS_DIR", "data/items")
    feeds_path: Path = PROJECT_ROOT / "config" / "feeds.json"
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "ai_developments")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    semantic_duplicate_threshold: float = float(
        os.getenv("SEMANTIC_DUPLICATE_THRESHOLD", "0.93")
    )
    semantic_search_limit: int = int(os.getenv("SEMANTIC_SEARCH_LIMIT", "5"))
    lookback_hours: int = int(os.getenv("LOOKBACK_HOURS", "168"))
    max_items_per_feed: int = int(os.getenv("MAX_ITEMS_PER_FEED", "60"))
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "25"))
    user_agent: str = os.getenv(
        "USER_AGENT", "AI-Developments-Digest/0.1 (+personal research project)"
    )
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "2048"))
    ollama_safety_margin: int = int(os.getenv("OLLAMA_SAFETY_MARGIN", "512"))
    token_estimator_type: str = os.getenv("TOKEN_ESTIMATOR_TYPE", "auto")
    tokenizer_name_or_path: str = os.getenv("TOKENIZER_NAME_OR_PATH", "Qwen/Qwen2.5-3B")
    max_articles_per_llm_batch: int = int(
        os.getenv("MAX_ARTICLES_PER_LLM_BATCH", "10")
    )
    max_article_chars: int = int(os.getenv("MAX_ARTICLE_CHARS", "5000"))
    story_clustering_enabled: bool = os.getenv(
        "STORY_CLUSTERING_ENABLED", "true"
    ).lower() in ("true", "1", "yes")
    story_cluster_top_k: int = int(os.getenv("STORY_CLUSTER_TOP_K", "10"))
    story_cluster_similarity_threshold: float = float(
        os.getenv("STORY_CLUSTER_SIMILARITY_THRESHOLD", "0.85")
    )
    story_cluster_max_age_days: int = int(
        os.getenv("STORY_CLUSTER_MAX_AGE_DAYS", "30")
    )
    story_cluster_max_articles: int = int(
        os.getenv("STORY_CLUSTER_MAX_ARTICLES", "10")
    )
    story_cluster_max_context_chars: int = int(
        os.getenv("STORY_CLUSTER_MAX_CONTEXT_CHARS", "30000")
    )
    story_cluster_ambiguity_margin: float = float(
        os.getenv("STORY_CLUSTER_AMBIGUITY_MARGIN", "0.03")
    )

    def ensure_directories(self) -> None:
        for path in (
            self.sqlite_path.parent,
            self.output_dir / "normalized",
            self.output_dir / "batches",
            self.output_dir / "digests",
            self.items_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_feeds(path: Path | None = None) -> list[dict[str, Any]]:
    feed_path = path or Settings().feeds_path
    with feed_path.open("r", encoding="utf-8") as handle:
        feeds = json.load(handle)
    if not isinstance(feeds, list):
        raise ValueError(f"Feed configuration must be a JSON list: {feed_path}")
    validated: list[dict[str, Any]] = []
    for feed in feeds:
        if not isinstance(feed, dict) or not feed.get("url") or not feed.get("name"):
            raise ValueError(f"Each feed needs non-empty 'name' and 'url': {feed!r}")
        if feed.get("enabled", True):
            validated.append(feed)
    return validated
