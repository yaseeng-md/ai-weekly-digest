from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import requests

from ai_digest.config import Settings
from ai_digest.normalize import normalize_entry
from ai_digest.storage import SQLiteStore


class RSSCollector:
    def __init__(self, settings: Settings, store: SQLiteStore):
        self.settings = settings
        self.store = store
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": settings.user_agent,
            "Accept": "application/atom+xml, application/rss+xml, application/xml, text/xml, */*;q=0.8",
        })

    @staticmethod
    def _is_recent(record: dict[str, Any], lookback_hours: int) -> bool:
        published = record.get("published_at")
        if not published:
            # Some feeds omit dates; keep those items rather than silently losing them.
            return True
        try:
            dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return True
        cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        return dt.astimezone(timezone.utc) >= cutoff

    def fetch_feed(self, feed: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
        self.store.register_feed(feed)
        state = self.store.feed_http_state(feed["url"])
        headers = {}
        if state.get("etag"):
            headers["If-None-Match"] = str(state["etag"])
        if state.get("last_modified"):
            headers["If-Modified-Since"] = str(state["last_modified"])

        try:
            response = self.session.get(
                feed["url"], headers=headers, timeout=self.settings.request_timeout_seconds
            )
            if response.status_code == 304:
                self.store.update_feed_state(feed, status="not_modified")
                return [], "not_modified"
            response.raise_for_status()
            parsed = feedparser.parse(response.content)
            entries = list(getattr(parsed, "entries", []) or [])
            if getattr(parsed, "bozo", False) and not entries:
                error = str(getattr(parsed, "bozo_exception", "Unknown feed parsing error"))
                raise ValueError(f"Feed parsing failed: {error}")

            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            records = [normalize_entry(entry, feed, fetched_at=now) for entry in entries]
            records = [r for r in records if self._is_recent(r, self.settings.lookback_hours)]
            records = records[: self.settings.max_items_per_feed]
            self.store.update_feed_state(
                feed,
                etag=response.headers.get("ETag"),
                last_modified=response.headers.get("Last-Modified"),
                status="ok",
                error=None,
            )
            return records, "ok"
        except Exception as exc:
            self.store.update_feed_state(feed, status="error", error=str(exc)[:1000])
            raise

    def close(self) -> None:
        self.session.close()
