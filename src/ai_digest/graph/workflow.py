from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from ai_digest.graph.nodes import DigestGraphNodes
from ai_digest.graph.state import DigestState

logger = logging.getLogger(__name__)


def should_summarize(state: dict[str, Any]) -> Literal["batch_and_summarize", "generate_digest"]:
    """Conditional routing: check if there are clusters or articles pending summarization."""
    clusters = state.get("pending_clusters", [])
    articles = state.get("pending_articles", [])
    if clusters or articles:
        return "batch_and_summarize"
    logger.info("No pending clusters or articles to summarize.")
    return "generate_digest"


def build_digest_graph(node_handler: DigestGraphNodes) -> Any:
    """Build and compile the LangGraph workflow for the AI Digest pipeline."""
    builder = StateGraph(DigestState)

    # 1. Add discrete workflow nodes
    builder.add_node("fetch_feeds", node_handler.fetch_feeds_node)
    builder.add_node("embed_and_index", node_handler.embed_and_index_node)
    builder.add_node("cluster_stories", node_handler.cluster_stories_node)
    builder.add_node("batch_and_summarize", node_handler.batch_and_summarize_node)
    builder.add_node("generate_digest", node_handler.generate_digest_node)
    builder.add_node("publish_notion", node_handler.publish_notion_node)
    builder.add_node("finalize_run", node_handler.finalize_run_node)

    # 2. Add sequential transitions and conditional routing
    builder.add_edge(START, "fetch_feeds")
    builder.add_edge("fetch_feeds", "embed_and_index")
    builder.add_edge("embed_and_index", "cluster_stories")

    builder.add_conditional_edges(
        "cluster_stories",
        should_summarize,
        {
            "batch_and_summarize": "batch_and_summarize",
            "generate_digest": "generate_digest",
        },
    )

    builder.add_edge("batch_and_summarize", "generate_digest")
    builder.add_edge("generate_digest", "publish_notion")
    builder.add_edge("publish_notion", "finalize_run")
    builder.add_edge("finalize_run", END)

    return builder.compile()
