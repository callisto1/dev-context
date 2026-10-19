import logging
from dataclasses import asdict

from src.core.embeddings import embedding_client
from src.db.vector_store import VectorStore
from src.ingestion.base import BaseIngester, ChunkedDocument

logger = logging.getLogger(__name__)

MAX_BATCH_TOKENS = 100_000
CHARS_PER_TOKEN_ESTIMATE = 4


class IngestionProcessor:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    async def process(
        self,
        ingester: BaseIngester,
        source: str,
        delete_existing: bool = True,
        **ingest_kwargs,
    ) -> dict:
        documents = await ingester.ingest(source, **ingest_kwargs)

        if not documents:
            return {"status": "no_documents", "chunks_processed": 0}

        logger.info(f"Collected {len(documents)} chunks, starting processing")

        if delete_existing:
            source_type = documents[0].source_type
            source_name = documents[0].source_name
            deleted = await self.vector_store.delete_by_source(source_type, source_name)
            logger.info(f"Deleted {deleted} existing chunks for {source_type}:{source_name}")

        count = await self._embed_and_upsert(documents)

        return {
            "status": "success",
            "chunks_processed": count,
            "source_type": documents[0].source_type,
            "source_name": documents[0].source_name,
        }

    def _build_batches(self, doc_dicts: list[dict]) -> list[list[dict]]:
        batches = []
        current_batch = []
        current_tokens = 0

        for doc in doc_dicts:
            est_tokens = len(doc["content"]) // CHARS_PER_TOKEN_ESTIMATE
            if current_batch and current_tokens + est_tokens > MAX_BATCH_TOKENS:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(doc)
            current_tokens += est_tokens

        if current_batch:
            batches.append(current_batch)
        return batches

    async def _embed_and_upsert(self, documents: list[ChunkedDocument]) -> int:
        doc_dicts = [asdict(doc) for doc in documents]
        batches = self._build_batches(doc_dicts)
        total_count = 0

        for batch_idx, batch in enumerate(batches, 1):
            texts = [d["content"] for d in batch]

            logger.info(f"Embedding batch {batch_idx}/{len(batches)} ({len(batch)} chunks)")
            embeddings = await embedding_client.embed(texts)

            for doc, emb in zip(batch, embeddings):
                doc["embedding"] = emb

            count = await self.vector_store.upsert_documents(batch)
            total_count += count
            logger.info(f"Batch {batch_idx} upserted ({total_count} total so far)")

        logger.info(f"Embedding + upsert complete: {total_count} chunks stored")
        return total_count
