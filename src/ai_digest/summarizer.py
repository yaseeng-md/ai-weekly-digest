from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import requests

from ai_digest.config import Settings
from ai_digest.token_budget import (
    SYSTEM_PROMPT,
    USER_PROMPT_PREFIX,
    DynamicBatcher,
    OversizedClusterError,
    TokenEstimator,
)

logger = logging.getLogger(__name__)


class OllamaSummarizer:
    def __init__(
        self,
        settings: Settings,
        batcher: DynamicBatcher | None = None,
    ):
        if settings.llm_provider != "ollama":
            raise ValueError("This version supports LLM_PROVIDER=ollama only.")
        self.settings = settings
        self.session = requests.Session()
        self.batcher = batcher or DynamicBatcher(settings)

    def summarize(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize a list of items (either story clusters or legacy articles) using dynamic batching."""
        if not items:
            return {
                "digest_title": "No new AI developments",
                "executive_summary": "No new items were collected in this run.",
                "developments": [],
                "themes": [],
                "watchlist": [],
                "caveats": [],
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "llm_model": self.settings.ollama_model,
            }

        # Check if input is already story clusters (contains 'sources' or 'story_id')
        is_clusters = bool(items and ("story_id" in items[0] or "sources" in items[0]))
        if is_clusters:
            clusters = items
        else:
            # Legacy article list: convert to minimal single-article clusters
            clusters = []
            for art in items:
                clusters.append({
                    "story_id": art.get("id", ""),
                    "canonical_title": art.get("title", ""),
                    "category": art.get("category", "AI"),
                    "primary_source": {
                        "source_name": art.get("source_name", ""),
                        "url": art.get("url", ""),
                        "title": art.get("title", ""),
                    },
                    "sources": [
                        {
                            "source_name": art.get("source_name", ""),
                            "url": art.get("url", ""),
                            "title": art.get("title", ""),
                        }
                    ],
                    "source_urls": [art.get("url")] if art.get("url") else [],
                    "article_count": 1,
                    "articles": [art],
                })

        # Dynamically partition into batches fitting the context token budget
        batches = self.batcher.build_batches(clusters)
        if not batches:
            return self.summarize_single_batch([])

        if len(batches) == 1:
            res = self.summarize_single_batch(batches[0])
            res["batch_count"] = 1
            return res

        logger.info(
            "Summarizing %d clusters across %d dynamically sized token batches",
            len(clusters),
            len(batches),
        )
        batch_results: list[dict[str, Any]] = []
        for b_idx, batch in enumerate(batches, 1):
            logger.info(
                "Processing dynamic batch %d of %d (%d clusters)...",
                b_idx,
                len(batches),
                len(batch),
            )
            batch_res = self.summarize_single_batch(batch)
            batch_results.append(batch_res)

        return self._merge_batch_results(batch_results)

    def summarize_clusters(self, clusters: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize story clusters (delegates to summarize with dynamic batching)."""
        return self.summarize(clusters)

    def summarize_single_batch(self, clusters: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize a single batch of story clusters that has been verified to fit the token budget."""
        if not clusters:
            return {
                "digest_title": "No new AI developments",
                "executive_summary": "No new story clusters available for summarization.",
                "developments": [],
                "themes": [],
                "watchlist": [],
                "caveats": [],
                "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "llm_model": self.settings.ollama_model,
            }

        messages = self.batcher.build_messages_for_clusters(clusters, system_prompt=SYSTEM_PROMPT)
        estimated_tokens, estimator_method = self.batcher.estimator.estimate_prompt_tokens(messages)

        num_ctx = self.settings.ollama_num_ctx
        num_predict = self.settings.ollama_num_predict

        endpoint = self.settings.ollama_base_url.rstrip("/") + "/api/chat"
        req_payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
            "messages": messages,
        }

        logger.info(
            "Ollama request | model=%s | num_ctx=%d | num_predict=%d | estimated_tokens=%d (%s) | clusters=%d",
            self.settings.ollama_model,
            num_ctx,
            num_predict,
            estimated_tokens,
            estimator_method,
            len(clusters),
        )

        max_retries = 3
        response = None
        for attempt in range(1, max_retries + 1):
            try:
                response = self.session.post(
                    endpoint,
                    json=req_payload,
                    timeout=self.settings.ollama_timeout_seconds,
                )
                response.raise_for_status()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
                if attempt == max_retries:
                    raise
                logger.warning(
                    "Ollama connection error (attempt %d/%d): %s. Retrying in 3s...",
                    attempt,
                    max_retries,
                    exc,
                )
                import time
                time.sleep(3)
        payload = response.json()

        # Observability & token accounting
        prompt_eval_count = payload.get("prompt_eval_count", 0)
        eval_count = payload.get("eval_count", 0)
        done_reason = payload.get("done_reason", "unknown")
        eval_duration_ms = (payload.get("eval_duration") or 0) / 1e6
        total_duration_ms = (payload.get("total_duration") or 0) / 1e6

        if prompt_eval_count > 0:
            diff = abs(estimated_tokens - prompt_eval_count)
            pct = (diff / prompt_eval_count) * 100
            logger.info(
                "Ollama token evaluation | estimated=%d | actual_prompt_eval=%d | "
                "diff=%d (%.1f%%) | eval_count=%d | done_reason=%s | latency=%.1fms",
                estimated_tokens,
                prompt_eval_count,
                diff,
                pct,
                eval_count,
                done_reason,
                total_duration_ms,
            )
        else:
            logger.info(
                "Ollama response received | estimated_tokens=%d | eval_count=%d | done_reason=%s",
                estimated_tokens,
                eval_count,
                done_reason,
            )

        if done_reason == "length":
            logger.warning(
                "Ollama response stopped because it reached num_predict limit (done_reason=length). "
                "Output might be truncated."
            )

        message = payload.get("message", {}).get("content", "").strip()

        # Clean potential markdown wrapping
        if message.startswith("```"):
            message = re.sub(r"^```(?:json)?\s*", "", message)
            message = re.sub(r"\s*```$", "", message).strip()

        try:
            result = json.loads(message)
        except json.JSONDecodeError as exc:
            # If Ollama cut off near the end (e.g. done_reason == "length"), repair unclosed quotes/brackets
            repaired = False
            repair_suffixes = [
                '"\n}]}',
                '""\n}]}',
                '"}]}',
                '"\n}',
                '}]}',
                ']}',
                '}',
                ']',
            ]
            for suffix in repair_suffixes:
                try:
                    result = json.loads(message + suffix)
                    repaired = True
                    logger.warning("Repaired incomplete JSON from Ollama using suffix %r", suffix)
                    break
                except json.JSONDecodeError:
                    continue

            if not repaired:
                # If still failing, try rolling back to the last completed development object
                last_brace = message.rfind("},")
                if last_brace != -1:
                    truncated_msg = message[:last_brace + 1] + "\n]}"
                    try:
                        result = json.loads(truncated_msg)
                        repaired = True
                        logger.warning("Repaired incomplete JSON by rolling back to last complete development object")
                    except json.JSONDecodeError:
                        pass

            if not repaired:
                raise ValueError(f"Ollama did not return valid JSON: {message[:500]}") from exc

        validated = self._validate_cluster_summary(result, clusters)
        validated["llm_model"] = self.settings.ollama_model
        validated["prompt_tokens_estimated"] = estimated_tokens
        validated["prompt_eval_count"] = prompt_eval_count
        validated["eval_count"] = eval_count
        validated["done_reason"] = done_reason
        validated["eval_duration_ms"] = round(eval_duration_ms, 2)
        validated["total_duration_ms"] = round(total_duration_ms, 2)
        return validated

    def _merge_batch_results(
        self, batch_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Merge multiple dynamic batch outputs into a single consolidated digest."""
        all_developments: list[dict[str, Any]] = []
        exec_summaries: list[str] = []
        all_themes: list[str] = []
        all_watchlist: list[str] = []
        all_caveats: list[str] = []
        seen_keys: set[str] = set()

        total_est_prompt_tokens = 0
        total_eval_tokens = 0
        total_prompt_eval = 0

        for res in batch_results:
            total_est_prompt_tokens += res.get("prompt_tokens_estimated", 0)
            total_prompt_eval += res.get("prompt_eval_count", 0)
            total_eval_tokens += res.get("eval_count", 0)

            if res.get("executive_summary"):
                exec_summaries.append(res["executive_summary"])

            for dev in res.get("developments", []):
                key = (dev.get("story_id") or dev.get("canonical_title") or dev.get("title", "")).strip().lower()
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    all_developments.append(dev)

            all_themes.extend(res.get("themes", []))
            all_watchlist.extend(res.get("watchlist", []))
            all_caveats.extend(res.get("caveats", []))

        return {
            "digest_title": "AI Developments Digest",
            "executive_summary": "\n\n".join(exec_summaries),
            "developments": sorted(all_developments, key=lambda item: item.get("importance", 3), reverse=True),
            "themes": list(dict.fromkeys(all_themes)),
            "watchlist": list(dict.fromkeys(all_watchlist)),
            "caveats": list(dict.fromkeys(all_caveats)),
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "llm_model": self.settings.ollama_model,
            "batch_count": len(batch_results),
            "prompt_tokens_estimated": total_est_prompt_tokens,
            "prompt_eval_count": total_prompt_eval,
            "eval_count": total_eval_tokens,
        }

    @staticmethod
    def _validate_cluster_summary(
        result: dict[str, Any], clusters: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise ValueError("LLM output must be a JSON object")

        result.setdefault("digest_title", "AI Developments Digest")
        result.setdefault("executive_summary", "")
        result.setdefault("developments", [])
        result.setdefault("themes", [])
        result.setdefault("watchlist", [])
        result.setdefault("caveats", [])

        # Index clusters by story_id and by index for fallback alignment
        cluster_by_id = {cl.get("story_id"): cl for cl in clusters if cl.get("story_id")}
        raw_developments = result.get("developments", [])
        if not isinstance(raw_developments, list):
            raw_developments = []

        cleaned_developments: list[dict[str, Any]] = []

        for idx, dev in enumerate(raw_developments):
            if not isinstance(dev, dict):
                continue

            # Resolve which cluster this development corresponds to
            story_id = dev.get("story_id")
            cluster = cluster_by_id.get(story_id)
            if not cluster and idx < len(clusters):
                cluster = clusters[idx]
                story_id = cluster.get("story_id")

            if not cluster:
                # If cannot match to a known cluster, construct default metadata
                title = str(dev.get("title", "Untitled development"))
                cluster = {
                    "story_id": story_id or f"story_unassigned_{idx}",
                    "canonical_title": title,
                    "primary_source": {},
                    "sources": [],
                    "source_urls": [],
                    "article_count": 1,
                    "category": str(dev.get("category", "AI")),
                }

            # Canonical metadata attached strictly from verified database records
            canonical_title = cluster.get("canonical_title") or dev.get("title") or "Untitled development"
            primary_source = cluster.get("primary_source", {})
            sources = cluster.get("sources", [])
            source_urls = cluster.get("source_urls", [])

            try:
                importance = min(5, max(1, int(dev.get("importance", 3))))
            except (TypeError, ValueError):
                importance = 3

            technical_details = str(dev.get("technical_details", "")).strip()
            consolidated_summary = str(
                dev.get("consolidated_summary") or dev.get("what_happened", "")
            ).strip()

            cleaned_developments.append({
                "story_id": cluster.get("story_id"),
                "canonical_title": canonical_title,
                "title": str(dev.get("title") or canonical_title),
                "importance": importance,
                "consolidated_summary": consolidated_summary,
                "what_happened": consolidated_summary,  # backward compatibility alias
                "technical_details": technical_details,
                "primary_source": primary_source,
                "sources": sources,
                "source_urls": source_urls,
                "article_count": cluster.get("article_count", len(sources)),
            })

        result["developments"] = cleaned_developments
        for key in ("themes", "watchlist", "caveats"):
            value = result.get(key, [])
            result[key] = [str(item) for item in value] if isinstance(value, list) else []

        result["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        return result

    def close(self) -> None:
        self.session.close()
