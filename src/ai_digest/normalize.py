from __future__ import annotations

import hashlib
import html
import json
import re
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

NAMESPACE = uuid.UUID("d4c967a8-8a95-46c7-8afe-eab6ce4a03f7")
TRACKING_KEYS = {
    "fbclid", "gclid", "dclid", "mc_cid", "mc_eid", "ref", "ref_src",
    "igshid", "ocid", "cmpid", "source", "_hsenc", "_hsmi",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(value: str | None) -> str:
    if not value:
        return ""
    parser = _TextExtractor()
    try:
        parser.feed(value)
        text = " ".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def canonicalize_url(url: str | None) -> str:
    """Remove fragments and common tracking parameters without changing the destination."""
    if not url:
        return ""
    raw = html.unescape(str(url)).strip()
    parts = urlsplit(raw)
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        return raw
    hostname = parts.hostname.lower() if parts.hostname else parts.netloc.lower()
    if parts.port and not ((parts.scheme.lower() == "http" and parts.port == 80) or (parts.scheme.lower() == "https" and parts.port == 443)):
        hostname = f"{hostname}:{parts.port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = []
    for key, val in parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in TRACKING_KEYS:
            continue
        query.append((key, val))
    query.sort()
    return urlunsplit((parts.scheme.lower(), hostname, path, urlencode(query, doseq=True), ""))


def parse_date(value: Any = None, parsed_struct: Any = None) -> str | None:
    if parsed_struct:
        try:
            dt = datetime(*parsed_struct[:6], tzinfo=timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except (TypeError, ValueError):
            pass
    if not value:
        return None
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = parsedate_to_datetime(text)
        except (TypeError, ValueError, OverflowError):
            return text[:100]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _get_content(entry: Any) -> str:
    content_parts = getattr(entry, "content", None) or []
    if content_parts:
        best = content_parts[0].get("value", "")
        if best:
            return html_to_text(best)
    return html_to_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))


def normalize_entry(entry: Any, feed: dict[str, Any], fetched_at: str | None = None) -> dict[str, Any]:
    title = html_to_text(getattr(entry, "title", "")) or "Untitled entry"
    link = str(getattr(entry, "link", "") or "").strip()
    guid = str(getattr(entry, "id", "") or getattr(entry, "guid", "") or "").strip()
    canonical_url = canonicalize_url(link)
    if not canonical_url:
        fallback = guid or f"{feed.get('url', '')}|{title}|{getattr(entry, 'published', '')}"
        canonical_url = "urn:rss-item:" + hashlib.sha256(fallback.encode("utf-8")).hexdigest()
    item_id = str(uuid.uuid5(NAMESPACE, canonical_url))
    published_at = parse_date(
        getattr(entry, "published", None) or getattr(entry, "updated", None),
        getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None),
    )
    authors = []
    for author in (getattr(entry, "authors", None) or []):
        name = html_to_text(author.get("name", "")) if isinstance(author, dict) else str(author)
        if name:
            authors.append(name)
    tags = []
    for tag in (getattr(entry, "tags", None) or []):
        term = html_to_text(tag.get("term", "")) if isinstance(tag, dict) else str(tag)
        if term:
            tags.append(term)
    content = _get_content(entry)
    description = html_to_text(getattr(entry, "summary", "") or getattr(entry, "description", ""))
    embedding_text = "\n".join(part for part in (title, description, content[:2000]) if part).strip()
    content_hash = hashlib.sha256(embedding_text.casefold().encode("utf-8")).hexdigest()
    return {
        "id": item_id,
        "title": title,
        "url": link,
        "canonical_url": canonical_url,
        "guid": guid,
        "source_name": feed.get("name", "Unknown source"),
        "feed_url": feed.get("url", ""),
        "category": feed.get("category", "uncategorized"),
        "published_at": published_at,
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "authors": authors,
        "tags": tags,
        "description": description,
        "content": content,
        "content_hash": content_hash,
        "embedding_text": embedding_text,
    }


def to_json_text(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)


def to_markdown(record: dict[str, Any]) -> str:
    lines = [f"# {record.get('title') or 'Untitled entry'}", ""]
    lines.extend([
        f"- **Source:** {record.get('source_name', '')}",
        f"- **Category:** {record.get('category', '')}",
        f"- **Published:** {record.get('published_at') or 'Unknown'}",
        f"- **URL:** {record.get('url') or 'Not provided'}",
    ])
    if record.get("authors"):
        lines.append(f"- **Authors:** {', '.join(record['authors'])}")
    lines.append("")
    description = record.get("description", "")
    content = record.get("content", "")
    if description:
        lines.extend(["## Description", "", description, ""])
    if content and content != description:
        lines.extend(["## Available content", "", content, ""])
    if record.get("tags"):
        lines.extend(["## Tags", "", ", ".join(record["tags"]), ""])
    return "\n".join(lines).rstrip() + "\n"
