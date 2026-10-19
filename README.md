# Dev Context Engine

RAG-powered dev assistant backed by Qwen3-Coder-32B and pgvector. Indexes your repos, Confluence pages, and a curated knowledge base of 60+ engineering sources (CNCF projects, SRE books, RFCs, style guides) to answer questions with full codebase context.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  FastAPI     │────>│  RAG Pipeline│────>│  Qwen3-Coder-32B │
│  API Server  │     │  (retrieve → │     │  (vLLM on GPU    │
│  (this box)  │     │   rerank →   │     │   server)        │
│              │     │   prompt)    │     │                  │
└──────┬───────┘     └──────┬───────┘     └──────────────────┘
       │                    │
       │              ┌─────┴──────┐
       │              │  pgvector  │
       │              │  (Postgres)│
       └──────────────┴────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         Git repos   Confluence   Knowledge
                                  (CNCF, RFCs,
                                   SRE, etc.)
```

**Retrieval pipeline**: Voyage `voyage-code-3` embeddings → pgvector HNSW similarity search (top-30) → cross-encoder reranker (top-5) → Qwen3-Coder-32B.

---

## Prerequisites

- **Docker** and **Docker Compose** (that's it)
- **A GPU server** running vLLM with Qwen3-Coder-32B (see Step 1)
- **Voyage AI API key** — sign up at https://dash.voyageai.com

---

## Quick start (3 commands)

```bash
# 1. Configure
cp .env.example .env    # edit .env with your GPU server IP and Voyage API key

# 2. Launch (builds app + starts bundled Postgres with pgvector)
docker compose --profile db up -d

# 3. Verify
curl http://localhost:8080/health
```

That's it. Works on Linux, macOS, Windows — anything that runs Docker.

### Using your own PostgreSQL instance

If you already have a Postgres instance (RDS, Cloud SQL, Supabase, self-hosted), skip the bundled container entirely. Your instance needs PostgreSQL 15+ with the `pgvector` extension available (the app runs `CREATE EXTENSION IF NOT EXISTS vector` on startup).

```bash
# In .env, point DATABASE_URL to your instance:
DATABASE_URL=postgresql+asyncpg://user:password@your-db-host:5432/dev_context_engine

# Launch just the app (no --profile db)
docker compose up -d
```

### Migrating away from the bundled Postgres

```bash
# Export from the Docker container
docker compose exec postgres pg_dump -U devctx dev_context_engine > backup.sql

# Import into your target instance
psql -h your-new-host -U user -d dev_context_engine < backup.sql

# Update .env, restart the app
docker compose up -d
```

---

## Detailed setup

### Step 1: Set up Qwen3-Coder-32B on your GPU server

SSH into your GPU server and install vLLM:

```bash
pip install vllm
```

Serve the model:

```bash
vllm serve Qwen/Qwen3-Coder-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

> For multi-GPU setups, increase `--tensor-parallel-size` to match your GPU count.

Verify it's running:

```bash
curl http://your-gpu-server:8000/v1/models
```

### Step 2: Get your Voyage AI API key

1. Go to https://dash.voyageai.com
2. Create an account and generate an API key
3. Free tier includes 200M tokens — enough for initial setup

### Step 3: Configure the environment

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```bash
# Point to your GPU server
LLM_BASE_URL=http://your-gpu-server:8000/v1

# Your Voyage API key
EMBEDDING_API_KEY=pa-xxxxxxxxxxxxxxxx

# (Optional) Confluence credentials
CONFLUENCE_BASE_URL=https://yourcompany.atlassian.net
CONFLUENCE_USER_EMAIL=you@company.com
CONFLUENCE_API_TOKEN=your-token
```

### Step 4: Launch

```bash
docker compose up -d
```

This builds and starts two containers:
- **postgres** — PostgreSQL 16 with pgvector extension
- **app** — FastAPI server with the full RAG pipeline, reranker, and scheduler

Data persists across restarts via Docker volumes (`pgdata` for the database, `repos` for cloned git repos).

The server starts with:
- pgvector extension and tables auto-created
- Reindex scheduler running in the background
- API docs available at http://localhost:8080/docs

### Useful Docker commands

```bash
# View logs
docker compose logs -f app

# Restart after config changes
docker compose restart app

# Rebuild after code changes
docker compose up -d --build app

# Stop everything
docker compose down

# Stop and wipe all data (database + repos)
docker compose down -v
```

### Local development (without Docker)

If you prefer running the app directly for development:

```bash
# Start only Postgres in Docker
docker compose up -d postgres

# Run the app locally
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

## Step 5: Ingest your data

All commands below use the API (works from any machine that can reach the server). For CLI scripts, prefix with `docker compose exec app` when running in Docker.

### Ingest a git repo

```bash
# Via API
curl -X POST http://localhost:8080/ingest/git \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/your-org/your-repo.git"}'

# Via CLI (inside the container)
docker compose exec app python scripts/ingest_repo.py https://github.com/your-org/your-repo.git
docker compose exec app python scripts/ingest_repo.py https://github.com/your-org/repo.git --branch main --include "*.py" "*.go"
```

### Ingest Confluence

```bash
# Entire space
curl -X POST http://localhost:8080/ingest/confluence \
  -H "Content-Type: application/json" \
  -d '{"space_key": "ENG"}'

# Specific pages
curl -X POST http://localhost:8080/ingest/confluence \
  -H "Content-Type: application/json" \
  -d '{"page_ids": ["12345", "67890"]}'

# Only pages with certain labels
curl -X POST http://localhost:8080/ingest/confluence \
  -H "Content-Type: application/json" \
  -d '{"space_key": "ENG", "include_labels": ["architecture", "runbook"]}'
```

### Ingest knowledge base (CNCF, RFCs, style guides, etc.)

```bash
# List all available sources
docker compose exec app python scripts/ingest_knowledge.py --list

# Ingest everything (~60 sources, ~$30 in Voyage API costs)
docker compose exec app python scripts/ingest_knowledge.py

# Ingest by category
docker compose exec app python scripts/ingest_knowledge.py --category cncf_graduated
docker compose exec app python scripts/ingest_knowledge.py --category engineering_practices
docker compose exec app python scripts/ingest_knowledge.py --category security

# Ingest specific sources
docker compose exec app python scripts/ingest_knowledge.py --source kubernetes --source istio --source owasp-cheatsheets

# Via API
curl -X POST http://localhost:8080/knowledge/ingest \
  -H "Content-Type: application/json" \
  -d '{"categories": ["cncf_graduated", "engineering_practices"]}'
```

Available categories: `cncf_graduated`, `cncf_incubating`, `engineering_practices`, `rfcs_and_design`, `frameworks_and_tools`, `observability_sre`, `security`.

## Step 6: Ask questions

```bash
# Basic query
curl -X POST http://localhost:8080/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does the auth middleware work?"}'

# Scoped to a specific repo
curl -X POST http://localhost:8080/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I add a new API endpoint?",
    "source_type": "git",
    "source_name": "covalent-cloud-server"
  }'

# With conversation history
curl -X POST http://localhost:8080/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Can you show me the tests for that?",
    "conversation_history": [
      {"role": "user", "content": "How does the auth middleware work?"},
      {"role": "assistant", "content": "The auth middleware is in src/middleware/auth.py..."}
    ]
  }'

# Streaming response
curl -X POST http://localhost:8080/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the CI/CD pipeline", "stream": true}'
```

---

## MCP (Model Context Protocol) — Agent Integration

The engine exposes an MCP server at `http://localhost:8080/mcp` so any MCP-compatible agent can use your indexed codebase and knowledge as tools.

### Available tools

| Tool | Description |
|---|---|
| `search_code` | Search indexed repos for relevant code snippets. Optional `repo` filter. |
| `search_docs` | Search documentation, knowledge base, and Confluence. Optional `source` filter. |
| `ask` | Get an AI-generated answer grounded in indexed context. |
| `ingest_repo` | Index a new git repo on demand. |
| `list_sources` | Show all indexed sources and chunk counts. |

### Connect from Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "dev-context-engine": {
      "url": "http://localhost:8080/mcp/mcp"
    }
  }
}
```

### Connect from Cursor

In Cursor settings, add an MCP server with URL: `http://localhost:8080/mcp/mcp`

### Connect from Droid (Factory)

```bash
droid mcp add dev-context http://localhost:8080/mcp/mcp --type http
```

### Connect from a custom agent (Python)

```python
from fastmcp.client import Client

async with Client("http://your-server:8080/mcp/mcp") as client:
    # Search for code
    result = await client.call_tool("search_code", {
        "query": "how is authentication handled",
        "repo": "covalent-cloud-server"
    })

    # Ask a question with full RAG
    result = await client.call_tool("ask", {
        "question": "What's the best way to set up mTLS with Istio?",
    })

    # Index a new repo
    result = await client.call_tool("ingest_repo", {
        "repo_url": "https://github.com/your-org/new-service.git",
    })
```

### Connect from any HTTP client

The MCP endpoint also works with raw HTTP. Use the [MCP Streamable HTTP transport](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http) to interact from any language.

---

## Operations

### Scheduled re-indexing

The scheduler runs in-process with the API server. It checks every 60 seconds and runs jobs whose cron expression matches the current minute.

**Git change detection**: Before re-indexing a git source, the scheduler checks the remote HEAD via `git ls-remote`. If the commit hash hasn't changed since the last index, the source is skipped (status: `skipped`). This saves Voyage API costs — only repos with new commits get re-embedded.

#### Register a source for daily re-indexing

```bash
# Register a repo (defaults to daily at midnight UTC)
curl -X POST http://localhost:8080/scheduler/sources \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "git",
    "source_uri": "https://github.com/your-org/your-repo.git",
    "ingest_kwargs": {"branch": "main"}
  }'

# Custom cron schedule
curl -X POST http://localhost:8080/scheduler/sources \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "git",
    "source_uri": "https://github.com/your-org/active-repo.git",
    "cron_expression": "0 */12 * * *"
  }'

# Register a Confluence space
curl -X POST http://localhost:8080/scheduler/sources \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "confluence",
    "source_uri": "ENG",
    "cron_expression": "0 0 * * *"
  }'
```

#### Manage scheduled sources

```bash
# List all registered sources and their status
curl http://localhost:8080/scheduler/sources

# Trigger an immediate reindex
curl -X POST http://localhost:8080/scheduler/sources/<id>/trigger

# Force reindex (bypass change detection)
curl -X POST "http://localhost:8080/scheduler/sources/<id>/trigger?force=true"

# Disable a source
curl -X PATCH http://localhost:8080/scheduler/sources/<id> \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'

# Change schedule
curl -X PATCH http://localhost:8080/scheduler/sources/<id> \
  -H "Content-Type: application/json" \
  -d '{"cron_expression": "0 0 * * 1"}'

# Delete a source
curl -X DELETE http://localhost:8080/scheduler/sources/<id>
```

### Health and stats

```bash
# Health check (includes DB connectivity)
curl http://localhost:8080/health

# Index stats (chunk counts by source)
curl http://localhost:8080/stats

# List available knowledge sources
curl http://localhost:8080/knowledge/sources
```

### Delete indexed data

```bash
# Delete all chunks for a specific source
curl -X DELETE http://localhost:8080/ingest/source/git/your-repo-name
curl -X DELETE http://localhost:8080/ingest/source/confluence/ENG
curl -X DELETE http://localhost:8080/ingest/source/knowledge/kubernetes
```

---

## Cost estimate (Voyage API)

| What | Tokens | Cost | Frequency |
|---|---|---|---|
| Initial: your repos | ~150M | ~$9 | One-time |
| Initial: knowledge base | ~500M | ~$30 | One-time |
| Daily reindex (with change detection) | ~15M/day | ~$27/month | Daily |
| Knowledge refresh | ~500M | ~$30 | Manual (quarterly) |
| Queries (100/day) | ~0.1M/day | ~$0.20/month | Continuous |
| **Estimated monthly** | | **~$30/month** | |

> Change detection means only repos with new commits get re-embedded. Most of your repos won't change daily.

---

## Configuration reference

All settings are in `.env` (see `.env.example`). Key ones:

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `LLM_MODEL` | `Qwen/Qwen3-Coder-32B` | Model name |
| `EMBEDDING_MODE` | `voyage` | `voyage`, `local`, or `remote` |
| `EMBEDDING_API_KEY` | | Voyage AI API key |
| `RAG_RETRIEVAL_K` | `30` | Candidates retrieved before reranking |
| `RERANKER_ENABLED` | `true` | Enable cross-encoder reranking |
| `RERANKER_TOP_N` | `5` | Final chunks sent to LLM |
| `REINDEX_ENABLED` | `true` | Enable background scheduler |
| `REINDEX_DEFAULT_CRON` | `0 0 * * *` | Default: daily at midnight UTC |

---

## Project structure

```
dev-context-engine/
├── docker-compose.yml           # PostgreSQL 16 + pgvector
├── pyproject.toml               # Dependencies
├── .env.example                 # Config template
├── knowledge/
│   └── registry.yml             # 60+ curated knowledge sources
├── scripts/
│   ├── setup_db.sh              # Start Postgres
│   ├── ingest_repo.py           # CLI: ingest a single repo
│   └── ingest_knowledge.py      # CLI: bulk-ingest knowledge sources
├── src/
│   ├── main.py                  # FastAPI app + lifespan + MCP mount
│   ├── mcp.py                   # MCP server with agent-facing tools
│   ├── config.py                # Pydantic settings
│   ├── api/routes/
│   │   ├── chat.py              # POST /chat/query
│   │   ├── ingest.py            # POST /ingest/git, /ingest/confluence
│   │   ├── knowledge.py         # GET/POST /knowledge/...
│   │   ├── scheduler.py         # CRUD /scheduler/sources
│   │   └── health.py            # GET /health, /stats
│   ├── core/
│   │   ├── llm.py               # vLLM client (OpenAI-compatible)
│   │   ├── embeddings.py        # Voyage / local / remote embeddings
│   │   ├── reranker.py          # Cross-encoder reranker
│   │   ├── rag.py               # Retrieve → rerank → prompt pipeline
│   │   └── scheduler.py         # Background reindex with change detection
│   ├── ingestion/
│   │   ├── git_ingest.py        # Clone + chunk git repos
│   │   ├── confluence_ingest.py # Confluence Cloud REST API v2
│   │   ├── web_docs_ingest.py   # Async web doc crawler
│   │   ├── knowledge_ingest.py  # Registry-driven bulk ingestion
│   │   ├── chunker.py           # Code-aware chunking
│   │   ├── processor.py         # Ingest → embed → store orchestrator
│   │   └── base.py              # Base ingester interface
│   └── db/
│       ├── models.py            # Document + IndexedSource tables
│       ├── session.py           # Async SQLAlchemy engine
│       └── vector_store.py      # pgvector similarity search
├── alembic/                     # DB migrations
└── tests/                       # Unit tests
```

---

## Adding custom knowledge sources

Edit `knowledge/registry.yml` to add your own sources:

```yaml
my_custom_sources:
  - name: my-internal-standards
    kind: git_docs
    repo: https://github.com/my-org/engineering-standards.git
    docs_path: "**/*.md"
    branch: main

  - name: internal-wiki
    kind: web_docs
    seed_url: https://wiki.mycompany.com/engineering/
    max_pages: 100
    allowed_path_prefix: /engineering/
```

Then ingest:

```bash
python scripts/ingest_knowledge.py --source my-internal-standards
```
