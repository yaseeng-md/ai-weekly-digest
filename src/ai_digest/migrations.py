from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _migration_v1_base(conn: sqlite3.Connection) -> None:
    """Base schema for feeds, items, item_sources, runs."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS feeds (
            feed_url TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'uncategorized',
            etag TEXT,
            last_modified TEXT,
            last_checked_at TEXT,
            last_status TEXT,
            last_error TEXT
        );

        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            canonical_url TEXT NOT NULL UNIQUE,
            guid TEXT,
            source_name TEXT NOT NULL,
            feed_url TEXT NOT NULL,
            category TEXT NOT NULL,
            published_at TEXT,
            fetched_at TEXT NOT NULL,
            authors_json TEXT NOT NULL DEFAULT '[]',
            tags_json TEXT NOT NULL DEFAULT '[]',
            description TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL DEFAULT '',
            embedding_text TEXT NOT NULL DEFAULT '',
            content_hash TEXT NOT NULL,
            normalized_json TEXT NOT NULL,
            json_path TEXT,
            markdown_path TEXT,
            semantic_duplicate INTEGER NOT NULL DEFAULT 0,
            duplicate_of TEXT,
            qdrant_indexed INTEGER NOT NULL DEFAULT 0,
            llm_status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY (duplicate_of) REFERENCES items(id)
        );

        CREATE INDEX IF NOT EXISTS idx_items_published_at ON items(published_at);
        CREATE INDEX IF NOT EXISTS idx_items_duplicate ON items(semantic_duplicate);
        CREATE INDEX IF NOT EXISTS idx_items_created_at ON items(created_at);

        CREATE TABLE IF NOT EXISTS item_sources (
            item_id TEXT NOT NULL,
            feed_url TEXT NOT NULL,
            source_name TEXT NOT NULL,
            discovered_at TEXT NOT NULL,
            PRIMARY KEY (item_id, feed_url),
            FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            feed_count INTEGER NOT NULL DEFAULT 0,
            fetched_entries INTEGER NOT NULL DEFAULT 0,
            inserted_items INTEGER NOT NULL DEFAULT 0,
            exact_duplicates INTEGER NOT NULL DEFAULT 0,
            semantic_duplicates INTEGER NOT NULL DEFAULT 0,
            digest_path TEXT,
            status TEXT NOT NULL DEFAULT 'running',
            error TEXT
        );
        """
    )


def _migration_v2_story_clusters(conn: sqlite3.Connection) -> None:
    """Multi-source story clusters and article cluster assignments."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS story_clusters (
            story_id TEXT PRIMARY KEY,
            canonical_title TEXT NOT NULL,
            representative_article_id TEXT NOT NULL,
            primary_source_name TEXT NOT NULL DEFAULT '',
            article_count INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            first_published_at TEXT,
            last_published_at TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            FOREIGN KEY (representative_article_id) REFERENCES items(id)
        );

        CREATE TABLE IF NOT EXISTS story_cluster_articles (
            story_id TEXT NOT NULL,
            article_id TEXT NOT NULL UNIQUE,
            similarity_score REAL,
            assignment_method TEXT NOT NULL,
            assigned_at TEXT NOT NULL,
            PRIMARY KEY (story_id, article_id),
            FOREIGN KEY (story_id) REFERENCES story_clusters(story_id) ON DELETE CASCADE,
            FOREIGN KEY (article_id) REFERENCES items(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_sca_story_id ON story_cluster_articles(story_id);
        CREATE INDEX IF NOT EXISTS idx_sca_article_id ON story_cluster_articles(article_id);
        CREATE INDEX IF NOT EXISTS idx_story_clusters_updated ON story_clusters(updated_at);
        CREATE INDEX IF NOT EXISTS idx_story_clusters_status ON story_clusters(status);
        """
    )


MIGRATIONS: list[tuple[int, str, Callable[[sqlite3.Connection], None]]] = [
    (1, "Base schema (feeds, items, item_sources, runs)", _migration_v1_base),
    (2, "Story clusters and cluster article memberships", _migration_v2_story_clusters),
]


def get_current_version(conn: sqlite3.Connection) -> int:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    )
    if not cursor.fetchone():
        # Check if legacy items table already exists
        items_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
        ).fetchone()
        if items_check:
            # Table exists without migration table
            return 1
        return 0
    row = conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def run_migrations(conn: sqlite3.Connection) -> int:
    """Run pending migrations transactionally. Returns number of migrations applied."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL,
            description TEXT NOT NULL
        )
        """
    )

    # If items exists but version 1 not recorded in schema_migrations, record it
    has_items = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
    ).fetchone()
    v1_recorded = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE version=1"
    ).fetchone()
    if has_items and not v1_recorded:
        conn.execute(
            "INSERT INTO schema_migrations(version, applied_at, description) VALUES (1, ?, ?)",
            (utc_now(), "Base schema (existing installation)"),
        )

    applied_count = 0
    applied_versions = {
        row[0] for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
    }

    for version, description, func in MIGRATIONS:
        if version not in applied_versions:
            logger.info("Applying migration %d: %s", version, description)
            func(conn)
            conn.execute(
                "INSERT INTO schema_migrations(version, applied_at, description) VALUES (?, ?, ?)",
                (version, utc_now(), description),
            )
            applied_count += 1

    return applied_count
