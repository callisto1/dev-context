from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.rag import rag_pipeline
from src.db.session import get_session
from src.db.vector_store import VectorStore

router = APIRouter()


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    conversation_history: list[ChatMessage] = []
    source_type: str | None = None
    source_name: str | list[str] | None = None
    stream: bool = False


class ContextChunk(BaseModel):
    source_type: str
    source_name: str
    file_path: str | None
    chunk_type: str
    language: str | None
    content: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    context: list[ContextChunk]
    context_count: int


@router.post("/query")
async def chat_query(
    request: ChatRequest,
    session: AsyncSession = Depends(get_session),
):
    store = VectorStore(session)

    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]

    if request.stream:
        async def generate():
            async for token in rag_pipeline.query_stream(
                question=request.question,
                vector_store=store,
                conversation_history=history or None,
                source_type=request.source_type,
                source_name=request.source_name,
            ):
                yield token

        return StreamingResponse(generate(), media_type="text/plain")

    result = await rag_pipeline.query(
        question=request.question,
        vector_store=store,
        conversation_history=history or None,
        source_type=request.source_type,
        source_name=request.source_name,
    )

    return ChatResponse(
        answer=result["answer"],
        context=[
            ContextChunk(
                source_type=c["source_type"],
                source_name=c["source_name"],
                file_path=c.get("file_path"),
                chunk_type=c["chunk_type"],
                language=c.get("language"),
                content=c["content"],
                similarity=c["similarity"],
            )
            for c in result["context"]
        ],
        context_count=result["context_count"],
    )
