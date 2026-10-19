"""Cross-encoder reranker for improving retrieval precision.

Retrieves a broad set of candidates via embedding similarity (top-30),
then reranks them with a cross-encoder that scores (query, document) pairs
jointly — much more accurate than bi-encoder similarity alone.

Supports two modes:
  local  — runs a cross-encoder model via sentence-transformers on this machine
  remote — calls a reranking API (Cohere-compatible or vLLM-served)
"""

import logging

import httpx

from src.config import settings

logger = logging.getLogger(__name__)

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from sentence_transformers import CrossEncoder

        _local_model = CrossEncoder(settings.reranker_model, max_length=512)
        logger.info(f"Loaded reranker model: {settings.reranker_model}")
    return _local_model


class Reranker:
    def __init__(self):
        self.enabled = settings.reranker_enabled
        self.mode = settings.reranker_mode
        self.top_n = settings.reranker_top_n
        self._http_client: httpx.AsyncClient | None = None

    async def rerank(self, query: str, results: list[dict]) -> list[dict]:
        if not self.enabled or not results:
            return results[: self.top_n]

        if self.mode == "local":
            return self._rerank_local(query, results)
        return await self._rerank_remote(query, results)

    def _rerank_local(self, query: str, results: list[dict]) -> list[dict]:
        model = _get_local_model()

        pairs = [[query, r["content"]] for r in results]
        scores = model.predict(pairs)

        for r, score in zip(results, scores):
            r["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda r: r["rerank_score"], reverse=True)
        return reranked[: self.top_n]

    async def _rerank_remote(self, query: str, results: list[dict]) -> list[dict]:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)

        documents = [r["content"] for r in results]

        try:
            resp = await self._http_client.post(
                f"{settings.reranker_base_url}/rerank",
                json={
                    "model": settings.reranker_model,
                    "query": query,
                    "documents": documents,
                    "top_n": self.top_n,
                },
                headers={"Authorization": f"Bearer {settings.reranker_api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()

            scored = []
            for item in data["results"]:
                idx = item["index"]
                result = results[idx].copy()
                result["rerank_score"] = item["relevance_score"]
                scored.append(result)

            scored.sort(key=lambda r: r["rerank_score"], reverse=True)
            return scored

        except (httpx.HTTPStatusError, httpx.RequestError, KeyError) as e:
            logger.warning(f"Remote reranker failed, falling back to similarity order: {e}")
            return results[: self.top_n]

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()


reranker = Reranker()
