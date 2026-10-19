import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        _local_model = SentenceTransformer(settings.embedding_model, trust_remote_code=True)
    return _local_model


VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_MAX_BATCH = 128


class EmbeddingClient:
    def __init__(self):
        self.mode = settings.embedding_mode
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self.mode == "local":
            return self._embed_local(texts)
        if self.mode == "voyage":
            return await self._embed_voyage(texts)
        return await self._embed_remote(texts)

    def _embed_local(self, texts: list[str]) -> list[list[float]]:
        model = _get_local_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    async def _embed_voyage(self, texts: list[str]) -> list[list[float]]:
        client = await self._get_http_client()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), VOYAGE_MAX_BATCH):
            batch = texts[i : i + VOYAGE_MAX_BATCH]
            response = await client.post(
                VOYAGE_API_URL,
                json={
                    "model": self.model_name,
                    "input": batch,
                    "input_type": "document",
                },
                headers={
                    "Authorization": f"Bearer {settings.embedding_api_key}",
                    "Content-Type": "application/json",
                },
            )
            if response.status_code != 200:
                body = response.text[:500]
                batch_idx = i // VOYAGE_MAX_BATCH + 1
                text_lens = [len(t) for t in batch]
                logger.error(
                    f"Voyage batch {batch_idx} failed ({response.status_code}): {body} | "
                    f"batch_size={len(batch)}, text_lengths={text_lens[:5]}..."
                )
                response.raise_for_status()

            data = response.json()

            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)

            usage = data.get("usage", {})
            if usage:
                logger.debug(f"Voyage batch {i // VOYAGE_MAX_BATCH + 1}: {usage.get('total_tokens', '?')} tokens")

        return all_embeddings

    async def _embed_remote(self, texts: list[str]) -> list[list[float]]:
        """Generic OpenAI-compatible embedding endpoint."""
        client = await self._get_http_client()
        response = await client.post(
            f"{settings.embedding_base_url}/embeddings",
            json={"model": self.model_name, "input": texts},
            headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query — Voyage uses input_type='query' for retrieval queries."""
        if self.mode == "voyage":
            client = await self._get_http_client()
            response = await client.post(
                VOYAGE_API_URL,
                json={
                    "model": self.model_name,
                    "input": [text],
                    "input_type": "query",
                },
                headers={
                    "Authorization": f"Bearer {settings.embedding_api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["data"][0]["embedding"]

        results = await self.embed([text])
        return results[0]

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text as a document."""
        results = await self.embed([text])
        return results[0]

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()


embedding_client = EmbeddingClient()
