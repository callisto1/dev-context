#!/usr/bin/env python3
"""CLI tool to ingest a git repository into the dev context engine."""
import argparse
import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


async def main():
    parser = argparse.ArgumentParser(description="Ingest a git repository")
    parser.add_argument("repo_url", help="Git repository URL or local path")
    parser.add_argument("--branch", "-b", help="Branch to ingest")
    parser.add_argument("--include", "-i", nargs="*", help="Include file patterns (e.g. '*.py')")
    parser.add_argument("--exclude", "-e", nargs="*", help="Exclude file patterns")
    args = parser.parse_args()

    from src.db.session import async_session, init_db
    from src.db.vector_store import VectorStore
    from src.ingestion.git_ingest import GitIngester
    from src.ingestion.processor import IngestionProcessor

    await init_db()

    async with async_session() as session:
        store = VectorStore(session)
        processor = IngestionProcessor(store)
        ingester = GitIngester()

        result = await processor.process(
            ingester,
            args.repo_url,
            branch=args.branch,
            include_patterns=args.include,
            exclude_patterns=args.exclude,
        )

        print(f"\nResult: {result['status']}")
        print(f"Chunks processed: {result['chunks_processed']}")

        if result.get("source_name"):
            stats = await store.get_stats()
            print(f"Total chunks in DB: {stats['total_chunks']}")


if __name__ == "__main__":
    asyncio.run(main())
