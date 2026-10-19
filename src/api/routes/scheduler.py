import json
import logging
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.core.scheduler import scheduler
from src.db.models import IndexedSource
from src.db.session import get_session

logger = logging.getLogger(__name__)
router = APIRouter()


class RegisterSourceRequest(BaseModel):
    source_type: str  # "git" or "confluence"
    source_uri: str  # repo URL or space key
    source_name: str | None = None  # auto-derived if not given
    cron_expression: str | None = None  # defaults to REINDEX_DEFAULT_CRON
    enabled: bool = True
    ingest_kwargs: dict | None = None  # branch, include_patterns, etc.


class SourceResponse(BaseModel):
    id: str
    source_type: str
    source_name: str
    source_uri: str
    enabled: bool
    cron_expression: str
    last_commit_hash: str | None
    last_indexed_at: str | None
    last_status: str | None
    last_error: str | None
    last_chunk_count: int | None


class UpdateSourceRequest(BaseModel):
    enabled: bool | None = None
    cron_expression: str | None = None
    ingest_kwargs: dict | None = None


def _source_to_response(src: IndexedSource) -> SourceResponse:
    return SourceResponse(
        id=str(src.id),
        source_type=src.source_type,
        source_name=src.source_name,
        source_uri=src.source_uri,
        enabled=src.enabled,
        cron_expression=src.cron_expression,
        last_commit_hash=src.last_commit_hash,
        last_indexed_at=src.last_indexed_at.isoformat() if src.last_indexed_at else None,
        last_status=src.last_status,
        last_error=src.last_error,
        last_chunk_count=src.last_chunk_count,
    )


@router.get("/sources", response_model=list[SourceResponse])
async def list_sources(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(IndexedSource).order_by(IndexedSource.created_at))
    sources = result.scalars().all()
    return [_source_to_response(s) for s in sources]


@router.post("/sources", response_model=SourceResponse)
async def register_source(
    request: RegisterSourceRequest,
    session: AsyncSession = Depends(get_session),
):
    source_name = request.source_name
    if not source_name:
        if request.source_type == "git":
            source_name = request.source_uri.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        else:
            source_name = request.source_uri

    source = IndexedSource(
        source_type=request.source_type,
        source_name=source_name,
        source_uri=request.source_uri,
        enabled=request.enabled,
        cron_expression=request.cron_expression or settings.reindex_default_cron,
        ingest_kwargs_json=json.dumps(request.ingest_kwargs) if request.ingest_kwargs else None,
    )
    session.add(source)
    await session.commit()
    await session.refresh(source)

    logger.info(f"Registered source: {source.source_type}:{source.source_name} [{source.cron_expression}]")
    return _source_to_response(source)


@router.patch("/sources/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: str,
    request: UpdateSourceRequest,
    session: AsyncSession = Depends(get_session),
):
    values = {}
    if request.enabled is not None:
        values["enabled"] = request.enabled
    if request.cron_expression is not None:
        values["cron_expression"] = request.cron_expression
    if request.ingest_kwargs is not None:
        values["ingest_kwargs_json"] = json.dumps(request.ingest_kwargs)

    if values:
        await session.execute(
            update(IndexedSource).where(IndexedSource.id == source_id).values(**values)
        )
        await session.commit()

    result = await session.execute(
        select(IndexedSource).where(IndexedSource.id == source_id)
    )
    source = result.scalar_one_or_none()
    if not source:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Source not found")

    return _source_to_response(source)


@router.delete("/sources/{source_id}")
async def delete_source(
    source_id: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        delete(IndexedSource).where(IndexedSource.id == source_id)
    )
    await session.commit()
    return {"status": "ok", "deleted": result.rowcount > 0}


@router.post("/sources/{source_id}/trigger")
async def trigger_reindex(source_id: str, force: bool = False):
    """Trigger an immediate reindex. Use ?force=true to skip change detection."""
    result = await scheduler.trigger_now(source_id, force=force)
    return result
