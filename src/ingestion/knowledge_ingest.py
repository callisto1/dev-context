import logging
from pathlib import Path

import yaml

from src.db.vector_store import VectorStore
from src.ingestion.confluence_ingest import ConfluenceIngester
from src.ingestion.git_ingest import GitIngester
from src.ingestion.processor import IngestionProcessor
from src.ingestion.web_docs_ingest import WebDocsIngester

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent.parent / "knowledge" / "registry.yml"


def load_registry(path: Path | None = None) -> dict:
    registry_path = path or DEFAULT_REGISTRY
    with open(registry_path) as f:
        return yaml.safe_load(f)


def _flatten_sources(registry: dict) -> list[dict]:
    sources = []
    for category, items in registry.items():
        if not isinstance(items, list):
            continue
        for item in items:
            item["category"] = category
            sources.append(item)
    return sources


async def ingest_source(source: dict, processor: IngestionProcessor, force: bool = False) -> dict:
    name = source["name"]
    kind = source["kind"]

    logger.info(f"[knowledge] Ingesting {name} ({kind}) from category {source.get('category', '?')}")

    if not force:
        existing = await processor.vector_store.count_by_source("knowledge", name)
        if existing > 0:
            logger.info(f"[knowledge] Skipping {name} — already has {existing} chunks (use --force to re-ingest)")
            return {"status": "skipped", "name": name, "chunks_processed": existing}

    try:
        if kind in ("git_docs", "git_full"):
            ingester = GitIngester()
            include = [source["docs_path"]] if source.get("docs_path") else None
            exclude = source.get("exclude_patterns") if source.get("exclude_patterns") else None
            result = await processor.process(
                ingester,
                source["repo"],
                branch=source.get("branch"),
                include_patterns=include,
                exclude_patterns=exclude,
                source_type_override="knowledge",
                source_name_override=name,
            )
            return result

        elif kind == "confluence":
            ingester = ConfluenceIngester()
            try:
                result = await processor.process(
                    ingester,
                    source.get("space_key", name),
                    space_key=source.get("space_key"),
                    page_ids=source.get("page_ids"),
                    max_pages=source.get("max_pages", 500),
                    include_labels=source.get("include_labels"),
                    include_children=source.get("include_children", True),
                )
                return result
            finally:
                await ingester.close()

        elif kind == "web_docs":
            ingester = WebDocsIngester()
            try:
                result = await processor.process(
                    ingester,
                    name,
                    seed_url=source["seed_url"],
                    max_pages=source.get("max_pages", 100),
                    allowed_path_prefix=source.get("allowed_path_prefix"),
                )
                return result
            finally:
                await ingester.close()

        else:
            return {"status": "error", "name": name, "message": f"Unknown kind: {kind}"}

    except Exception as e:
        logger.exception(f"Failed to ingest knowledge source: {name}")
        return {"status": "error", "name": name, "message": str(e)[:500]}


async def ingest_registry(
    vector_store: VectorStore,
    registry_path: Path | None = None,
    categories: list[str] | None = None,
    source_names: list[str] | None = None,
    force: bool = False,
) -> list[dict]:
    registry = load_registry(registry_path)
    all_sources = _flatten_sources(registry)

    if categories:
        cat_set = set(categories)
        all_sources = [s for s in all_sources if s.get("category") in cat_set]

    if source_names:
        name_set = set(source_names)
        all_sources = [s for s in all_sources if s["name"] in name_set]

    logger.info(f"[knowledge] Processing {len(all_sources)} sources")

    processor = IngestionProcessor(vector_store)
    results = []

    for source in all_sources:
        result = await ingest_source(source, processor, force=force)
        result["name"] = source["name"]
        result["category"] = source.get("category", "")
        results.append(result)
        logger.info(f"[knowledge] {source['name']}: {result.get('status', '?')} — {result.get('chunks_processed', 0)} chunks")

    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    total_chunks = sum(r.get("chunks_processed", 0) for r in results)

    logger.info(f"[knowledge] Done: {succeeded} succeeded, {failed} failed, {total_chunks} total chunks")
    return results
