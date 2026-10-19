import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from sqlalchemy import select, update

from src.config import settings
from src.db.models import IndexedSource
from src.db.session import async_session
from src.db.vector_store import VectorStore
from src.ingestion.confluence_ingest import ConfluenceIngester
from src.ingestion.git_ingest import GitIngester
from src.ingestion.processor import IngestionProcessor

logger = logging.getLogger(__name__)


def _parse_cron(expression: str) -> dict:
    """Parse a cron expression into its 5 fields.

    Supports: *, */N, and literal integers for each field.
    Fields: minute, hour, day_of_month, month, day_of_week.
    """
    parts = expression.strip().split()
    if len(parts) != 5:
        raise ValueError(f"Invalid cron expression: {expression}")
    return {
        "minute": parts[0],
        "hour": parts[1],
        "day_of_month": parts[2],
        "month": parts[3],
        "day_of_week": parts[4],
    }


def _cron_field_matches(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return value % step == 0
    if "," in field:
        return value in [int(x) for x in field.split(",")]
    if "-" in field:
        lo, hi = field.split("-")
        return int(lo) <= value <= int(hi)
    return value == int(field)


def _should_run(cron_expr: str, now: datetime) -> bool:
    try:
        cron = _parse_cron(cron_expr)
    except ValueError:
        return False

    return (
        _cron_field_matches(cron["minute"], now.minute)
        and _cron_field_matches(cron["hour"], now.hour)
        and _cron_field_matches(cron["day_of_month"], now.day)
        and _cron_field_matches(cron["month"], now.month)
        and _cron_field_matches(cron["day_of_week"], now.isoweekday() % 7)
    )


def _get_ingester(source_type: str):
    if source_type == "git":
        return GitIngester()
    if source_type == "confluence":
        return ConfluenceIngester()
    if source_type == "knowledge":
        from src.ingestion.web_docs_ingest import WebDocsIngester
        return WebDocsIngester()
    raise ValueError(f"Unknown source type: {source_type}")


def _resolve_remote_head(repo_url: str, branch: str | None = None) -> str | None:
    """Get the HEAD commit hash of a remote git repo without cloning."""
    import subprocess

    ref = branch or "HEAD"
    try:
        result = subprocess.run(
            ["git", "ls-remote", repo_url, ref],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()[0]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _resolve_local_head(repo_path: str) -> str | None:
    """Get the HEAD commit hash of a local git repo."""
    try:
        from git import Repo

        repo = Repo(repo_path)
        return repo.head.commit.hexsha
    except Exception:
        return None


def _get_current_head(source: IndexedSource) -> str | None:
    """Resolve the current HEAD for a git source (remote URL or local path)."""
    uri = source.source_uri
    ingest_kwargs = json.loads(source.ingest_kwargs_json) if source.ingest_kwargs_json else {}
    branch = ingest_kwargs.get("branch")

    if uri.startswith(("http://", "https://", "git@")):
        return _resolve_remote_head(uri, branch)
    elif os.path.isdir(uri):
        return _resolve_local_head(uri)
    return None


async def _reindex_source(source: IndexedSource, force: bool = False) -> None:
    logger.info(f"Reindexing {source.source_type}:{source.source_name} ({source.source_uri})")

    # Change detection for git sources — skip if HEAD hasn't moved
    if source.source_type in ("git", "knowledge") and not force:
        current_head = _get_current_head(source)
        if current_head and current_head == source.last_commit_hash:
            logger.info(
                f"Skipping {source.source_name} — HEAD unchanged ({current_head[:10]})"
            )
            async with async_session() as session:
                await session.execute(
                    update(IndexedSource)
                    .where(IndexedSource.id == source.id)
                    .values(
                        last_indexed_at=datetime.now(timezone.utc),
                        last_status="skipped",
                        last_error=None,
                    )
                )
                await session.commit()
            return

    async with async_session() as session:
        await session.execute(
            update(IndexedSource)
            .where(IndexedSource.id == source.id)
            .values(last_status="running")
        )
        await session.commit()

    ingester = None
    try:
        ingester = _get_ingester(source.source_type)
        ingest_kwargs = json.loads(source.ingest_kwargs_json) if source.ingest_kwargs_json else {}

        async with async_session() as session:
            store = VectorStore(session)
            processor = IngestionProcessor(store)
            result = await processor.process(ingester, source.source_uri, **ingest_kwargs)

        # Capture the commit hash after successful indexing
        new_head = _get_current_head(source)

        async with async_session() as session:
            await session.execute(
                update(IndexedSource)
                .where(IndexedSource.id == source.id)
                .values(
                    last_indexed_at=datetime.now(timezone.utc),
                    last_status=result["status"],
                    last_error=None,
                    last_chunk_count=result["chunks_processed"],
                    last_commit_hash=new_head,
                )
            )
            await session.commit()

        logger.info(
            f"Reindex complete: {source.source_type}:{source.source_name} "
            f"— {result['chunks_processed']} chunks"
            f"{f' (commit {new_head[:10]})' if new_head else ''}"
        )

    except Exception as e:
        logger.exception(f"Reindex failed: {source.source_type}:{source.source_name}")
        async with async_session() as session:
            await session.execute(
                update(IndexedSource)
                .where(IndexedSource.id == source.id)
                .values(
                    last_indexed_at=datetime.now(timezone.utc),
                    last_status="error",
                    last_error=str(e)[:2000],
                )
            )
            await session.commit()

    finally:
        if ingester and hasattr(ingester, "close"):
            await ingester.close()


async def _tick() -> None:
    now = datetime.now(timezone.utc)

    async with async_session() as session:
        result = await session.execute(
            select(IndexedSource).where(IndexedSource.enabled.is_(True))
        )
        sources = result.scalars().all()

    for source in sources:
        if not _should_run(source.cron_expression, now):
            continue

        if source.last_status == "running":
            logger.debug(f"Skipping {source.source_name} — already running")
            continue

        asyncio.create_task(_reindex_source(source))


class ReindexScheduler:
    def __init__(self):
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self):
        if not settings.reindex_enabled:
            logger.info("Reindex scheduler disabled via config")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Reindex scheduler started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Reindex scheduler stopped")

    async def _loop(self):
        while self._running:
            try:
                await _tick()
            except Exception:
                logger.exception("Scheduler tick failed")
            await asyncio.sleep(60)

    async def trigger_now(self, source_id: str, force: bool = False) -> dict:
        async with async_session() as session:
            result = await session.execute(
                select(IndexedSource).where(IndexedSource.id == source_id)
            )
            source = result.scalar_one_or_none()

        if not source:
            return {"status": "error", "message": "Source not found"}

        asyncio.create_task(_reindex_source(source, force=force))
        return {"status": "triggered", "source_name": source.source_name, "force": force}


scheduler = ReindexScheduler()
