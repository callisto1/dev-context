#!/usr/bin/env python3
"""Bulk-ingest knowledge sources from the registry into pgvector.

Usage:
    # Ingest everything
    python scripts/ingest_knowledge.py

    # Ingest only CNCF graduated projects
    python scripts/ingest_knowledge.py --category cncf_graduated

    # Ingest specific sources
    python scripts/ingest_knowledge.py --source kubernetes --source istio --source google-eng-practices

    # Use a custom registry file
    python scripts/ingest_knowledge.py --registry /path/to/my-registry.yml

    # List available sources without ingesting
    python scripts/ingest_knowledge.py --list
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Ingest knowledge sources from registry")
    parser.add_argument("--registry", "-r", type=Path, help="Path to registry YAML")
    parser.add_argument("--category", "-c", action="append", dest="categories", help="Filter by category (repeatable)")
    parser.add_argument("--source", "-s", action="append", dest="sources", help="Filter by source name (repeatable)")
    parser.add_argument("--list", "-l", action="store_true", help="List available sources and exit")
    parser.add_argument("--force", "-f", action="store_true", help="Re-ingest even if source already exists in DB")
    args = parser.parse_args()

    from src.ingestion.knowledge_ingest import (
        _flatten_sources,
        ingest_registry,
        load_registry,
    )

    registry = load_registry(args.registry)
    all_sources = _flatten_sources(registry)

    if args.list:
        current_cat = None
        for src in all_sources:
            if src.get("category") != current_cat:
                current_cat = src.get("category", "")
                print(f"\n[{current_cat}]")
            kind = src["kind"]
            if kind == "git_docs":
                target = src.get("repo", "")
            else:
                target = src.get("seed_url", "")
            print(f"  {src['name']:40s} {kind:10s} {target}")
        print(f"\nTotal: {len(all_sources)} sources")
        return

    from src.db.session import async_session, init_db
    from src.db.vector_store import VectorStore

    await init_db()

    async with async_session() as session:
        store = VectorStore(session)
        results = await ingest_registry(
            store,
            registry_path=args.registry,
            categories=args.categories,
            source_names=args.sources,
            force=args.force,
        )

    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)

    for r in results:
        status = r.get("status", "?")
        chunks = r.get("chunks_processed", 0)
        name = r.get("name", "?")
        icon = "OK" if status == "success" else "FAIL" if status == "error" else "SKIP" if status == "skipped" else "?"
        line = f"  [{icon:4s}] {name:40s} {chunks:6d} chunks"
        if status == "error":
            line += f"  — {r.get('message', '')[:60]}"
        print(line)

    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    total_chunks = sum(r.get("chunks_processed", 0) for r in results if r.get("status") == "success")
    print(f"\n  {succeeded} succeeded, {failed} failed, {skipped} skipped, {total_chunks} total chunks")


if __name__ == "__main__":
    asyncio.run(main())
