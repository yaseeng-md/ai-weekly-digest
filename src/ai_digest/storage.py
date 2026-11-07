from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ai_digest.migrations import run_migrations


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SQLiteStore:
    """SQLite is the source of truth; Qdrant is used for similarity search."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connection() as conn:
            run_migrations(conn)

    def register_feed(self, feed: dict[str, Any]) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT INTO feeds(feed_url, name, category) VALUES (?, ?, ?)
                   ON CONFLICT(feed_url) DO UPDATE SET name=excluded.name, category=excluded.category""",
                (feed["url"], feed["name"], feed.get("category", "uncategorized")),
            )

    def feed_http_state(self, feed_url: str) -> dict[str, str | None]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT etag, last_modified FROM feeds WHERE feed_url=?", (feed_url,)
            ).fetchone()
        if not row:
            return {"etag": None, "last_modified": None}
        return {"etag": row["etag"], "last_modified": row["last_modified"]}

    def update_feed_state(
        self,
        feed: dict[str, Any],
        *,
        etag: str | None = None,
        last_modified: str | None = None,
        status: str = "ok",
        error: str | None = None,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """UPDATE feeds SET etag=COALESCE(?, etag),
                   last_modified=COALESCE(?, last_modified), last_checked_at=?,
                   last_status=?, last_error=? WHERE feed_url=?""",
                (etag, last_modified, utc_now(), status, error, feed["url"]),
            )

    def add_item(self, record: dict[str, Any], json_path: str, markdown_path: str) -> tuple[bool, str]:
        """Return (inserted, existing_or_new_item_id); exact URL duplicates are not reinserted."""
        with self.connection() as conn:
            existing = conn.execute(
                "SELECT id FROM items WHERE canonical_url=?", (record["canonical_url"],)
            ).fetchone()
            if existing:
                conn.execute(
                    """INSERT OR IGNORE INTO item_sources(item_id, feed_url, source_name, discovered_at)
                       VALUES (?, ?, ?, ?)""",
                    (existing["id"], record["feed_url"], record["source_name"], utc_now()),
                )
                return False, str(existing["id"])

            conn.execute(
                """INSERT INTO items(
                    id, title, url, canonical_url, guid, source_name, feed_url, category,
                    published_at, fetched_at, authors_json, tags_json, description, content,
                    embedding_text, content_hash, normalized_json, json_path, markdown_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["id"], record["title"], record.get("url", ""), record["canonical_url"],
                    record.get("guid", ""), record["source_name"], record["feed_url"],
                    record.get("category", "uncategorized"), record.get("published_at"),
                    record["fetched_at"], json.dumps(record.get("authors", []), ensure_ascii=False),
                    json.dumps(record.get("tags", []), ensure_ascii=False), record.get("description", ""),
                    record.get("content", ""), record.get("embedding_text", ""), record["content_hash"],
                    json.dumps(record, ensure_ascii=False), json_path, markdown_path, utc_now(),
                ),
            )
            conn.execute(
                """INSERT OR IGNORE INTO item_sources(item_id, feed_url, source_name, discovered_at)
                   VALUES (?, ?, ?, ?)""",
                (record["id"], record["feed_url"], record["source_name"], utc_now()),
            )
            return True, record["id"]

    def mark_semantic_duplicate(self, item_id: str, duplicate_of: str | None) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE items SET semantic_duplicate=?, duplicate_of=?, llm_status=CASE WHEN ? IS NOT NULL THEN 'skipped_duplicate' ELSE llm_status END WHERE id=?",
                (1 if duplicate_of else 0, duplicate_of, duplicate_of, item_id),
            )

    def mark_qdrant_indexed(self, item_id: str) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE items SET qdrant_indexed=1 WHERE id=?", (item_id,))

    def pending_qdrant_ids(self) -> list[str]:
        with self.connection() as conn:
            rows = conn.execute("SELECT id FROM items WHERE qdrant_indexed=0 ORDER BY created_at").fetchall()
        return [str(row["id"]) for row in rows]

    def pending_llm_ids(self) -> list[str]:
        # Items remain pending after a failed LLM call and are retried on the next run.
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT id FROM items WHERE semantic_duplicate=0 AND llm_status IN ('pending', 'error') ORDER BY COALESCE(published_at, created_at) DESC"
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["authors"] = json.loads(data.pop("authors_json"))
        data["tags"] = json.loads(data.pop("tags_json"))
        return data

    def list_items(self, item_ids: list[str], include_semantic_duplicates: bool = False) -> list[dict[str, Any]]:
        if not item_ids:
            return []
        placeholders = ",".join("?" for _ in item_ids)
        where = f"id IN ({placeholders})"
        if not include_semantic_duplicates:
            where += " AND semantic_duplicate=0"
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM items WHERE {where} ORDER BY COALESCE(published_at, created_at) DESC",
                tuple(item_ids),
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["authors"] = json.loads(item.pop("authors_json"))
            item["tags"] = json.loads(item.pop("tags_json"))
            records.append(item)
        return records

    def set_llm_status(self, item_ids: list[str], status: str) -> None:
        if not item_ids:
            return
        with self.connection() as conn:
            conn.executemany("UPDATE items SET llm_status=? WHERE id=?", [(status, item_id) for item_id in item_ids])

    def start_run(self, run_id: str, feed_count: int) -> None:
        with self.connection() as conn:
            conn.execute(
                "INSERT INTO runs(run_id, started_at, feed_count, status) VALUES (?, ?, ?, 'running')",
                (run_id, utc_now(), feed_count),
            )

    def finish_run(self, run_id: str, **values: Any) -> None:
        allowed = {
            "fetched_entries", "inserted_items", "exact_duplicates", "semantic_duplicates",
            "digest_path", "status", "error",
        }
        updates = {key: value for key, value in values.items() if key in allowed}
        updates["finished_at"] = utc_now()
        assignments = ", ".join(f"{key}=?" for key in updates)
        with self.connection() as conn:
            conn.execute(
                f"UPDATE runs SET {assignments} WHERE run_id=?",
                (*updates.values(), run_id),
            )

    def stats(self) -> dict[str, int]:
        with self.connection() as conn:
            items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            duplicates = conn.execute("SELECT COUNT(*) FROM items WHERE semantic_duplicate=1").fetchone()[0]
            indexed = conn.execute("SELECT COUNT(*) FROM items WHERE qdrant_indexed=1").fetchone()[0]
            feeds = conn.execute("SELECT COUNT(*) FROM feeds").fetchone()[0]
            clusters = conn.execute("SELECT COUNT(*) FROM story_clusters").fetchone()[0]
            clustered_articles = conn.execute("SELECT COUNT(*) FROM story_cluster_articles").fetchone()[0]
        return {
            "items": items,
            "semantic_duplicates": duplicates,
            "qdrant_indexed": indexed,
            "feeds": feeds,
            "story_clusters": clusters,
            "clustered_articles": clustered_articles,
        }
