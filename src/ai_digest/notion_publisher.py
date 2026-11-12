from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

NOTION_API_VERSION = "2022-06-28"
NOTION_BASE_URL = "https://api.notion.com/v1"


def get_ordinal_suffix(day: int) -> str:
    """Return ordinal suffix (st, nd, rd, th) for a day of the month."""
    if 11 <= (day % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def format_daily_title(dt: datetime | None = None) -> str:
    """Format date for Daily Digest page title, e.g. 'Nov 1st, 2026' or 'Nov 11th, 2026'."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    month = dt.strftime("%b")
    suffix = get_ordinal_suffix(dt.day)
    return f"{month} {dt.day}{suffix}, {dt.year}"


def format_weekly_title(dt: datetime | None = None) -> str:
    """Format week for Weekly Digest page title, e.g. 'Week 46 2026'."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    iso_year, iso_week, _ = dt.isocalendar()
    return f"Week {iso_week} {iso_year}"


def markdown_to_rich_text(text: str) -> list[dict[str, Any]]:
    """
    Parse a text string into Notion rich_text objects, supporting:
    - Markdown links: [link text](http://...)
    - Bold text: **bold text**
    - Code spans: `code`
    Respects Notion's 2000 character limit per text chunk.
    """
    if not text:
        return []

    # Combined regex to match links, bold, or inline code
    token_pattern = re.compile(
        r"(\[(?P<link_text>[^\]]+)\]\((?P<link_url>[^\)]+)\))|"
        r"(\*\*(?P<bold_text>[^*]+)\*\*)|"
        r"(`(?P<code_text>[^`]+)`)"
    )

    rich_text: list[dict[str, Any]] = []
    last_idx = 0

    for match in token_pattern.finditer(text):
        start, end = match.span()
        if start > last_idx:
            plain = text[last_idx:start]
            _append_text_chunks(rich_text, plain)

        if match.group("link_text"):
            l_text = match.group("link_text")
            l_url = match.group("link_url")
            _append_text_chunks(rich_text, l_text, link_url=l_url)
        elif match.group("bold_text"):
            b_text = match.group("bold_text")
            _append_text_chunks(rich_text, b_text, bold=True)
        elif match.group("code_text"):
            c_text = match.group("code_text")
            _append_text_chunks(rich_text, c_text, code=True)

        last_idx = end

    if last_idx < len(text):
        _append_text_chunks(rich_text, text[last_idx:])

    return rich_text or [{"type": "text", "text": {"content": ""}}]


def _append_text_chunks(
    rich_text: list[dict[str, Any]],
    content: str,
    link_url: str | None = None,
    bold: bool = False,
    code: bool = False,
) -> None:
    # Notion has a strict 2000 character limit per text object
    max_len = 1900
    for i in range(0, max(1, len(content)), max_len):
        chunk = content[i : i + max_len]
        if not chunk:
            continue
        obj: dict[str, Any] = {
            "type": "text",
            "text": {"content": chunk},
        }
        if link_url:
            obj["text"]["link"] = {"url": link_url}
        annotations: dict[str, Any] = {}
        if bold:
            annotations["bold"] = True
        if code:
            annotations["code"] = True
        if annotations:
            obj["annotations"] = annotations
        rich_text.append(obj)


def markdown_to_notion_blocks(markdown_content: str) -> list[dict[str, Any]]:
    """Convert Markdown content into a list of Notion block objects."""
    blocks: list[dict[str, Any]] = []
    lines = markdown_content.splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("### "):
            text = line[4:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {"rich_text": markdown_to_rich_text(text)},
            })
        elif line.startswith("## "):
            text = line[3:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_2",
                "heading_2": {"rich_text": markdown_to_rich_text(text)},
            })
        elif line.startswith("# "):
            text = line[2:].strip()
            blocks.append({
                "object": "block",
                "type": "heading_1",
                "heading_1": {"rich_text": markdown_to_rich_text(text)},
            })
        elif line.startswith("> "):
            text = line[2:].strip()
            blocks.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": markdown_to_rich_text(text),
                    "icon": {"emoji": "💡"},
                },
            })
        elif line.startswith("- ") or line.startswith("* "):
            text = line[2:].strip()
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": markdown_to_rich_text(text)},
            })
        elif line == "---" or line == "***":
            blocks.append({"object": "block", "type": "divider", "divider": {}})
        else:
            blocks.append({
                "object": "block",
                "type": "paragraph",
                "paragraph": {"rich_text": markdown_to_rich_text(line)},
            })

    return blocks


class NotionPublisher:
    """Manages publishing digest markdown to Notion under ai-weekly-digest hierarchy."""

    def __init__(
        self,
        api_key: str | None = None,
        root_page_id: str | None = None,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key
        self.root_page_id = root_page_id
        self.session = session or requests.Session()

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": NOTION_API_VERSION,
            "Content-Type": "application/json",
        }

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def _clean_id(self, page_id: str) -> str:
        return page_id.replace("-", "")

    def _get_page_title(self, page_data: dict[str, Any]) -> str:
        props = page_data.get("properties", {})
        title_prop = props.get("title") or props.get("Name") or {}
        if isinstance(title_prop, dict) and "title" in title_prop:
            title_objs = title_prop.get("title", [])
            return "".join(t.get("plain_text", "") for t in title_objs)
        return ""

    def ensure_main_page(self) -> str:
        """
        Locate or create the main 'ai-weekly-digest' parent page.
        Returns the clean page ID of 'ai-weekly-digest'.
        """
        if not self.is_configured():
            raise ValueError("Notion API key is not configured.")

        # 1. If root_page_id is provided, check if it is already ai-weekly-digest or its parent
        if self.root_page_id:
            cleaned_root = self._clean_id(self.root_page_id)
            resp = self.session.get(
                f"{NOTION_BASE_URL}/pages/{cleaned_root}",
                headers=self.headers,
                timeout=20,
            )
            if resp.status_code == 200:
                page_info = resp.json()
                title = self._get_page_title(page_info).lower().strip()
                if "ai-weekly-digest" in title or "ai weekly digest" in title:
                    logger.info("Found root page matching 'ai-weekly-digest': %s", cleaned_root)
                    return cleaned_root

            # Check if root has a child page named ai-weekly-digest
            children = self._list_child_pages(cleaned_root)
            for child_id, child_title in children.items():
                if child_title.lower().strip() in ("ai-weekly-digest", "ai weekly digest"):
                    logger.info("Found child 'ai-weekly-digest' page: %s", child_id)
                    return child_id

            # Create 'ai-weekly-digest' under root_page_id
            logger.info("Creating 'ai-weekly-digest' child page under %s", cleaned_root)
            return self._create_page(
                parent_id=cleaned_root,
                title="ai-weekly-digest",
                icon_emoji="🤖",
            )

        # 2. Search Notion workspace for a page named 'ai-weekly-digest'
        search_payload = {
            "query": "ai-weekly-digest",
            "filter": {"value": "page", "property": "object"},
        }
        resp = self.session.post(
            f"{NOTION_BASE_URL}/search",
            headers=self.headers,
            json=search_payload,
            timeout=20,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            for r in results:
                title = self._get_page_title(r).lower().strip()
                if title in ("ai-weekly-digest", "ai weekly digest"):
                    page_id = self._clean_id(r["id"])
                    logger.info("Found existing 'ai-weekly-digest' page via search: %s", page_id)
                    return page_id

        raise RuntimeError(
            "Could not locate 'ai-weekly-digest' page and NOTION_ROOT_PAGE_ID is not configured. "
            "Please provide NOTION_ROOT_PAGE_ID to create the hierarchy."
        )

    def _list_child_pages(self, parent_id: str) -> dict[str, str]:
        """List all child page blocks under a parent block/page. Returns {page_id: title}."""
        child_pages: dict[str, str] = {}
        cursor = None

        while True:
            url = f"{NOTION_BASE_URL}/blocks/{self._clean_id(parent_id)}/children"
            params: dict[str, Any] = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor

            resp = self.session.get(url, headers=self.headers, params=params, timeout=20)
            if resp.status_code != 200:
                logger.warning("Could not list children for block %s: %s", parent_id, resp.text)
                break

            data = resp.json()
            for block in data.get("results", []):
                if block.get("type") == "child_page":
                    c_id = self._clean_id(block["id"])
                    c_title = block.get("child_page", {}).get("title", "")
                    child_pages[c_id] = c_title

            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")

        return child_pages

    def _create_page(
        self,
        parent_id: str,
        title: str,
        icon_emoji: str = "📄",
        children: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create a new page in Notion under parent_id."""
        url = f"{NOTION_BASE_URL}/pages"
        payload: dict[str, Any] = {
            "parent": {"page_id": self._clean_id(parent_id)},
            "icon": {"type": "emoji", "emoji": icon_emoji},
            "properties": {
                "title": {
                    "title": [
                        {
                            "type": "text",
                            "text": {"content": title},
                        }
                    ]
                }
            },
        }
        # Notion accepts up to 100 blocks at creation
        if children:
            payload["children"] = children[:100]

        resp = self.session.post(url, headers=self.headers, json=payload, timeout=30)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Failed to create page '{title}': {resp.status_code} - {resp.text}")

        page_data = resp.json()
        new_page_id = self._clean_id(page_data["id"])

        # If there are remaining blocks, append them in batches of 100
        if children and len(children) > 100:
            self._append_blocks(new_page_id, children[100:])

        return new_page_id

    def _append_blocks(self, page_id: str, blocks: list[dict[str, Any]]) -> None:
        """Append blocks to a page in chunks of 100."""
        url = f"{NOTION_BASE_URL}/blocks/{self._clean_id(page_id)}/children"
        chunk_size = 100
        for i in range(0, len(blocks), chunk_size):
            chunk = blocks[i : i + chunk_size]
            resp = self.session.patch(
                url, headers=self.headers, json={"children": chunk}, timeout=30
            )
            if resp.status_code not in (200, 201):
                logger.error("Failed to append blocks to %s: %s", page_id, resp.text)
                break

    def ensure_subsections(self, main_page_id: str) -> dict[str, str]:
        """
        Ensure 'Weekly Digest' and 'Daily Digest' child pages exist under ai-weekly-digest.
        Returns dict with keys 'daily' and 'weekly' pointing to their Notion page IDs.
        """
        existing = self._list_child_pages(main_page_id)
        sections: dict[str, str] = {}

        for c_id, c_title in existing.items():
            norm_title = c_title.lower().strip()
            if norm_title in ("daily digest", "daily"):
                sections["daily"] = c_id
            elif norm_title in ("weekly digest", "weekly"):
                sections["weekly"] = c_id

        if "daily" not in sections:
            logger.info("Creating 'Daily Digest' sub-section page")
            sections["daily"] = self._create_page(
                parent_id=main_page_id,
                title="Daily Digest",
                icon_emoji="📅",
            )

        if "weekly" not in sections:
            logger.info("Creating 'Weekly Digest' sub-section page")
            sections["weekly"] = self._create_page(
                parent_id=main_page_id,
                title="Weekly Digest",
                icon_emoji="📰",
            )

        return sections

    def publish_digest(
        self,
        markdown_content: str,
        digest_type: str = "daily",
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Publish digest markdown content to Notion.

        Args:
            markdown_content: The full markdown digest string.
            digest_type: 'daily', 'weekly', or 'both'.
            timestamp: Reference timestamp (default: now UTC).

        Returns:
            Dict summarizing created/updated Notion pages.
        """
        if not self.is_configured():
            logger.info("Notion API key not configured; skipping Notion publishing.")
            return {"status": "skipped", "reason": "Notion API key not configured."}

        ts = timestamp or datetime.now(timezone.utc)
        main_page_id = self.ensure_main_page()
        subsections = self.ensure_subsections(main_page_id)
        blocks = markdown_to_notion_blocks(markdown_content)

        created_pages: list[dict[str, Any]] = []

        # 1. Publish Daily Digest if requested
        if digest_type in ("daily", "both"):
            daily_title = format_daily_title(ts)
            daily_page_id = self._create_page(
                parent_id=subsections["daily"],
                title=daily_title,
                icon_emoji="⚡",
                children=blocks,
            )
            created_pages.append({
                "section": "Daily Digest",
                "title": daily_title,
                "page_id": daily_page_id,
                "url": f"https://www.notion.so/{daily_page_id}",
            })
            logger.info("Published Daily Digest to Notion: '%s' (%s)", daily_title, daily_page_id)

        # 2. Publish Weekly Digest if requested
        if digest_type in ("weekly", "both"):
            weekly_title = format_weekly_title(ts)
            weekly_page_id = self._create_page(
                parent_id=subsections["weekly"],
                title=weekly_title,
                icon_emoji="📚",
                children=blocks,
            )
            created_pages.append({
                "section": "Weekly Digest",
                "title": weekly_title,
                "page_id": weekly_page_id,
                "url": f"https://www.notion.so/{weekly_page_id}",
            })
            logger.info("Published Weekly Digest to Notion: '%s' (%s)", weekly_title, weekly_page_id)

        return {
            "status": "ok",
            "main_page_id": main_page_id,
            "created_pages": created_pages,
        }
