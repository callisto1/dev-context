"""MCP server exposing the dev context engine as agent-friendly tools.

Mounted into the FastAPI app at /mcp. Any MCP-compatible client
(Claude Desktop, Cursor, Droid, custom agents) can connect to
http://host:8080/mcp and use these tools.
"""

from fastmcp import FastMCP

from src.core.rag import rag_pipeline
from src.db.session import async_session
from src.db.vector_store import VectorStore
from src.ingestion.git_ingest import GitIngester
from src.ingestion.processor import IngestionProcessor

mcp = FastMCP(
    name="Dev Context Engine",
    instructions=(
        "You have access to a code-aware knowledge engine that indexes git repos, "
        "Confluence pages, and engineering documentation (CNCF projects, RFCs, style guides). "
        "Use search_code to find relevant code snippets. "
        "Use search_docs to find engineering docs and best practices. "
        "Use ask to get an AI-generated answer grounded in the indexed context. "
        "Use ingest_repo to index a new git repository on demand. "
        "Use list_sources to see what's currently indexed. "
        "After a PR is merged, always re-ingest the affected repo using ingest_repo "
        "to keep the knowledge base up to date with the latest changes."
    ),
)


@mcp.tool
async def search_code(
    query: str,
    repo: str | None = None,
) -> str:
    """Search indexed codebases for relevant code snippets.

    Args:
        query: Natural language description of what you're looking for.
               Example: "JWT validation middleware", "database connection pooling setup"
        repo: Optional repo name to scope the search. If omitted, searches all repos.
    """
    async with async_session() as session:
        store = VectorStore(session)
        results = await rag_pipeline.build_context(
            query=query,
            vector_store=store,
            source_type="git",
            source_name=repo,
        )

    if not results:
        return "No relevant code found."

    parts = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] {r['source_name']}"
        if r.get("file_path"):
            header += f" — {r['file_path']}"
        if r.get("language"):
            header += f" ({r['language']})"
        score = r.get("rerank_score")
        if score is not None:
            header += f" [relevance: {score:.3f}]"
        parts.append(f"{header}\n```\n{r['content']}\n```")

    return "\n\n".join(parts)


@mcp.tool
async def search_docs(
    query: str,
    source: str | None = None,
) -> str:
    """Search indexed documentation, engineering best practices, and knowledge base.

    Covers: CNCF project docs, Confluence pages, RFCs, style guides, SRE books, etc.

    Args:
        query: What you're looking for.
               Example: "Istio mTLS configuration", "OWASP SQL injection prevention"
        source: Optional source name to scope the search (e.g. "kubernetes", "owasp-cheatsheets").
                If omitted, searches all documentation.
    """
    source_type = None
    if source:
        source_type = "knowledge"

    async with async_session() as session:
        store = VectorStore(session)
        results = await rag_pipeline.build_context(
            query=query,
            vector_store=store,
            source_type=source_type,
            source_name=source,
        )

    # Include confluence results when no specific source is given
    if not source:
        async with async_session() as session:
            store = VectorStore(session)
            confluence_results = await rag_pipeline.build_context(
                query=query,
                vector_store=store,
                source_type="confluence",
            )
        results.extend(confluence_results)
        results.sort(key=lambda r: r.get("rerank_score", r.get("similarity", 0)), reverse=True)
        results = results[:5]

    if not results:
        return "No relevant documentation found."

    parts = []
    for i, r in enumerate(results, 1):
        header = f"[{i}] {r['source_type']}:{r['source_name']}"
        if r.get("file_path"):
            header += f" — {r['file_path']}"
        score = r.get("rerank_score")
        if score is not None:
            header += f" [relevance: {score:.3f}]"
        parts.append(f"{header}\n{r['content']}")

    return "\n\n---\n\n".join(parts)


@mcp.tool
async def ask(
    question: str,
    repo: str | None = None,
    source_type: str | None = None,
) -> str:
    """Ask a question and get an AI-generated answer grounded in indexed context.

    The answer will reference specific files, functions, and documentation from the
    indexed codebase and knowledge base. Use this for complex questions that need
    reasoning, not just code lookup.

    Args:
        question: Your question.
                  Example: "How should I implement rate limiting for our API?"
        repo: Optional repo name to focus the search on a specific repository.
        source_type: Optional filter: "git", "confluence", or "knowledge".
    """
    async with async_session() as session:
        store = VectorStore(session)
        result = await rag_pipeline.query(
            question=question,
            vector_store=store,
            source_type=source_type,
            source_name=repo,
        )

    answer = result["answer"]
    ctx_count = result["context_count"]
    sources = set()
    for c in result["context"]:
        label = f"{c['source_type']}:{c['source_name']}"
        if c.get("file_path"):
            label += f" ({c['file_path']})"
        sources.add(label)

    footer = f"\n\n---\nBased on {ctx_count} context chunks from: {', '.join(sorted(sources))}"
    return answer + footer


@mcp.tool
async def ingest_repo(
    repo_url: str,
    branch: str | None = None,
) -> str:
    """Index a git repository so it becomes searchable.

    Clones the repo, chunks the code, embeds it, and stores it in the vector database.
    If the repo was previously indexed, it will be re-indexed with fresh content.

    Args:
        repo_url: Git repository URL (HTTPS or SSH) or local path.
        branch: Optional branch to index. Defaults to the repo's default branch.
    """
    async with async_session() as session:
        store = VectorStore(session)
        processor = IngestionProcessor(store)
        ingester = GitIngester()

        result = await processor.process(
            ingester,
            repo_url,
            branch=branch,
        )

    status = result["status"]
    chunks = result["chunks_processed"]
    name = result.get("source_name", repo_url)
    return f"Ingested {name}: {status}, {chunks} chunks indexed."


@mcp.tool
async def list_sources() -> str:
    """List all indexed sources and their chunk counts.

    Shows what's currently in the knowledge base — repos, Confluence spaces,
    and documentation sources.
    """
    async with async_session() as session:
        store = VectorStore(session)
        stats = await store.get_stats()

    if not stats["sources"]:
        return "No sources indexed yet."

    lines = [f"Total chunks: {stats['total_chunks']}", ""]
    current_type = None
    for s in stats["sources"]:
        if s["source_type"] != current_type:
            current_type = s["source_type"]
            lines.append(f"[{current_type}]")
        lines.append(f"  {s['source_name']}: {s['chunk_count']} chunks")

    return "\n".join(lines)
