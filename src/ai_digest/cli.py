from __future__ import annotations

import argparse
import json
import logging
import sys

from ai_digest.config import Settings, load_feeds
from ai_digest.embeddings import Embedder
from ai_digest.pipeline import DigestPipeline
from ai_digest.qdrant_store import QdrantVectorStore
from ai_digest.start_services import start_services
from ai_digest.storage import SQLiteStore
from ai_digest.story_clustering import ClusterManager, ClusterRepository


def _handle_clusters_command(args: argparse.Namespace, settings: Settings) -> None:
    store = SQLiteStore(settings.sqlite_path)
    repo = ClusterRepository(store)

    if args.cluster_subcommand == "stats":
        stats = repo.get_cluster_stats()
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    if args.cluster_subcommand == "show":
        story_id = args.story_id
        cluster = repo.get_cluster(story_id)
        if not cluster:
            print(f"Error: Story cluster '{story_id}' not found.", file=sys.stderr)
            sys.exit(1)
        articles = repo.get_cluster_articles(story_id)
        cluster_info = dict(cluster)
        cluster_info["articles"] = [
            {
                "id": a["id"],
                "title": a["title"],
                "source_name": a["source_name"],
                "url": a.get("url") or a.get("canonical_url"),
                "published_at": a.get("published_at"),
                "similarity_score": a.get("similarity_score"),
                "assignment_method": a.get("assignment_method"),
                "assigned_at": a.get("assigned_at"),
                "is_representative": (a["id"] == cluster.get("representative_article_id")),
            }
            for a in articles
        ]
        print(json.dumps(cluster_info, ensure_ascii=False, indent=2))
        return

    if args.cluster_subcommand == "validate":
        validation = repo.validate_clusters()
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        if not validation.get("is_valid"):
            sys.exit(1)
        return

    if args.cluster_subcommand == "backfill":
        if not start_services():
            raise RuntimeError("Failed to start required services for backfill")
        embedder = Embedder(settings.embedding_model, device=settings.embedding_device)
        vector_store = QdrantVectorStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection,
            dimension=embedder.dimension,
        )
        try:
            manager = ClusterManager(
                settings, store, vector_store=vector_store, embedder=embedder
            )
            result = manager.backfill_historical_articles(embedder=embedder, vector_store=vector_store)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            vector_store.close()
        return


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect and summarize AI developments from RSS/Atom feeds."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Collect feeds, normalize, deduplicate, and generate a digest")
    run_parser.add_argument(
        "--digest-type",
        choices=["daily", "weekly", "both"],
        default="daily",
        help="Type of digest to generate and log to Notion ('daily', 'weekly', or 'both')",
    )
    run_parser.add_argument(
        "--no-notion",
        action="store_true",
        help="Disable Notion synchronization for this run",
    )
    run_parser.add_argument(
        "--notion-page-id",
        type=str,
        default=None,
        help="Optional Notion root page ID override",
    )
    subparsers.add_parser("feeds", help="List enabled configured feeds")
    subparsers.add_parser("stats", help="Show SQLite collection statistics")

    # Story clustering subcommands
    clusters_parser = subparsers.add_parser(
        "clusters", help="Manage and inspect multi-source story clusters"
    )
    cluster_subparsers = clusters_parser.add_subparsers(
        dest="cluster_subcommand", required=True
    )
    cluster_subparsers.add_parser("stats", help="Show story cluster statistics")

    show_parser = cluster_subparsers.add_parser("show", help="Inspect a specific story cluster")
    show_parser.add_argument("story_id", help="The story ID to inspect")

    cluster_subparsers.add_parser("backfill", help="Backfill historical articles into story clusters")
    cluster_subparsers.add_parser("validate", help="Validate story cluster integrity")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    settings.ensure_directories()

    if args.command == "feeds":
        print(json.dumps(load_feeds(settings.feeds_path), ensure_ascii=False, indent=2))
        return

    if args.command == "stats":
        print(json.dumps(SQLiteStore(settings.sqlite_path).stats(), indent=2))
        return

    if args.command == "clusters":
        _handle_clusters_command(args, settings)
        return

    if args.command == "run":
        if not start_services():
            raise RuntimeError("Failed to start required services")
        print("All services are ready!")

        try:
            result = DigestPipeline(settings).run()
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            logging.exception("Digest run failed")
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
