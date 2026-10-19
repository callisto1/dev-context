#!/usr/bin/env bash
set -euo pipefail

echo "Starting PostgreSQL with pgvector..."
docker compose up -d postgres

echo "Waiting for PostgreSQL to be ready..."
until docker compose exec postgres pg_isready -U devctx 2>/dev/null; do
    sleep 1
done

echo "Database is ready."
echo ""
echo "Connection string: postgresql+asyncpg://devctx:devctx@localhost:5432/dev_context_engine"
echo ""
echo "To start the API server:"
echo "  uvicorn src.main:app --reload --host 0.0.0.0 --port 8080"
