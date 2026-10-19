"""OpenAI-compatible /v1/chat/completions endpoint.

Wraps the RAG pipeline so external UIs (Open WebUI, etc.) can use
this app as if it were an OpenAI API.
"""
import json
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.rag import rag_pipeline
from src.db.session import get_session
from src.db.vector_store import VectorStore

router = APIRouter()

MODEL_ID = "dev-context-engine"


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    # Ignored but accepted so clients don't error
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None
    n: int | None = None


def _build_completion_response(content: str, model: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _build_chunk(content: str, model: str, chunk_id: str, finish_reason: str | None = None) -> str:
    data = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(data)}\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    session: AsyncSession = Depends(get_session),
):
    user_message = ""
    conversation_history = []

    for msg in request.messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            user_message = msg.content
        else:
            conversation_history.append({"role": msg.role, "content": msg.content})

    if not user_message:
        return _build_completion_response("No question provided.", request.model)

    # Include earlier user messages as conversation history
    user_messages = [m for m in request.messages if m.role == "user"]
    if len(user_messages) > 1:
        for msg in request.messages[:-1]:
            if msg.role in ("user", "assistant"):
                conversation_history.append({"role": msg.role, "content": msg.content})
        conversation_history = [
            {"role": m.role, "content": m.content}
            for m in request.messages[:-1]
            if m.role in ("user", "assistant")
        ]

    store = VectorStore(session)

    if request.stream:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        async def generate():
            async for token in rag_pipeline.query_stream(
                question=user_message,
                vector_store=store,
                conversation_history=conversation_history or None,
            ):
                yield _build_chunk(token, request.model, chunk_id)
            yield _build_chunk("", request.model, chunk_id, finish_reason="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    result = await rag_pipeline.query(
        question=user_message,
        vector_store=store,
        conversation_history=conversation_history or None,
    )

    return _build_completion_response(result["answer"], request.model)


@router.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "dev-context-engine",
            }
        ],
    }
