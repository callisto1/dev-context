import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session
from src.db.vector_store import VectorStore
from src.ingestion.knowledge_ingest import (
    _flatten_sources,
    ingest_registry,
    load_registry,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class KnowledgeIngestRequest(BaseModel):
    categories: list[str] | None = None
    source_names: list[str] | None = None


class KnowledgeSourceInfo(BaseModel):
    name: str
    kind: str
    category: str
    target: str


class KnowledgeIngestResult(BaseModel):
    name: str
    category: str
    status: str
    chunks_processed: int = 0
    message: str | None = None


class KnowledgeIngestResponse(BaseModel):
    succeeded: int
    failed: int
    total_chunks: int
    results: list[KnowledgeIngestResult]


@router.get("/sources", response_model=list[KnowledgeSourceInfo])
async def list_knowledge_sources():
    registry = load_registry()
    sources = _flatten_sources(registry)
    return [
        KnowledgeSourceInfo(
            name=s["name"],
            kind=s["kind"],
            category=s.get("category", ""),
            target=s.get("repo", s.get("seed_url", "")),
        )
        for s in sources
    ]


@router.post("/ingest", response_model=KnowledgeIngestResponse)
async def ingest_knowledge(
    request: KnowledgeIngestRequest,
    session: AsyncSession = Depends(get_session),
):
    store = VectorStore(session)

    results = await ingest_registry(
        store,
        categories=request.categories,
        source_names=request.source_names,
    )

    succeeded = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    total_chunks = sum(r.get("chunks_processed", 0) for r in results)

    return KnowledgeIngestResponse(
        succeeded=succeeded,
        failed=failed,
        total_chunks=total_chunks,
        results=[
            KnowledgeIngestResult(
                name=r.get("name", "?"),
                category=r.get("category", ""),
                status=r.get("status", "unknown"),
                chunks_processed=r.get("chunks_processed", 0),
                message=r.get("message"),
            )
            for r in results
        ],
    )
