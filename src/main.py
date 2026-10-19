from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastmcp.utilities.lifespan import combine_lifespans

from src.api.routes import chat, health, ingest, knowledge, openai_compat, scheduler as scheduler_routes
from src.core.scheduler import scheduler
from src.db.session import engine, init_db
from src.mcp import mcp


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    await init_db()
    await scheduler.start()
    yield
    await scheduler.stop()
    await engine.dispose()


mcp_app = mcp.http_app(path="/")

app = FastAPI(
    title="Dev Context Engine",
    description="RAG-powered dev context engine backed by Qwen3-Coder and pgvector",
    version="0.1.0",
    lifespan=combine_lifespans(app_lifespan, mcp_app.lifespan),
)

app.mount("/mcp", mcp_app)
app.include_router(openai_compat.router, tags=["openai-compat"])
app.include_router(health.router, tags=["health"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingestion"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(knowledge.router, prefix="/knowledge", tags=["knowledge"])
app.include_router(scheduler_routes.router, prefix="/scheduler", tags=["scheduler"])
