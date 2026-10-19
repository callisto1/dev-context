import logging

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session
from src.db.vector_store import VectorStore
from src.ingestion.confluence_ingest import ConfluenceIngester
from src.ingestion.git_ingest import GitIngester
from src.ingestion.processor import IngestionProcessor

logger = logging.getLogger(__name__)
router = APIRouter()


class GitIngestRequest(BaseModel):
    repo_url: str
    branch: str | None = None
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None


class ConfluenceIngestRequest(BaseModel):
    space_key: str | None = None
    page_ids: list[str] | None = None
    max_pages: int = 500
    include_labels: list[str] | None = None
    include_children: bool = True


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_processed: int = 0


@router.post("/git", response_model=IngestResponse)
async def ingest_git_repo(
    request: GitIngestRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    store = VectorStore(session)
    processor = IngestionProcessor(store)
    ingester = GitIngester()

    try:
        result = await processor.process(
            ingester,
            request.repo_url,
            branch=request.branch,
            include_patterns=request.include_patterns,
            exclude_patterns=request.exclude_patterns,
        )
        return IngestResponse(
            status=result["status"],
            message=f"Ingested {result['chunks_processed']} chunks from {result.get('source_name', request.repo_url)}",
            chunks_processed=result["chunks_processed"],
        )
    except Exception as e:
        logger.exception(f"Failed to ingest {request.repo_url}")
        return IngestResponse(status="error", message=str(e))


@router.post("/confluence", response_model=IngestResponse)
async def ingest_confluence(
    request: ConfluenceIngestRequest,
    session: AsyncSession = Depends(get_session),
):
    store = VectorStore(session)
    processor = IngestionProcessor(store)
    ingester = ConfluenceIngester()

    source = request.space_key or "confluence"

    try:
        result = await processor.process(
            ingester,
            source,
            space_key=request.space_key,
            page_ids=request.page_ids,
            max_pages=request.max_pages,
            include_labels=request.include_labels,
            include_children=request.include_children,
        )
        return IngestResponse(
            status=result["status"],
            message=f"Ingested {result['chunks_processed']} chunks from Confluence:{result.get('source_name', source)}",
            chunks_processed=result["chunks_processed"],
        )
    except Exception as e:
        logger.exception(f"Failed to ingest Confluence source: {source}")
        return IngestResponse(status="error", message=str(e))
    finally:
        await ingester.close()


@router.delete("/source/{source_type}/{source_name}")
async def delete_source(
    source_type: str,
    source_name: str,
    session: AsyncSession = Depends(get_session),
):
    store = VectorStore(session)
    deleted = await store.delete_by_source(source_type, source_name)
    return {"status": "ok", "deleted_chunks": deleted}
