import asyncio

from src.config import settings
from src.core.embeddings import embedding_client
from src.core.llm import llm_client
from src.core.reranker import reranker
from src.db.vector_store import VectorStore

SYSTEM_PROMPT = """You are an expert software engineer assistant with deep knowledge of the user's codebase and development context. You have access to relevant code snippets, documentation, and project context retrieved from the user's repositories.

When answering:
- Reference specific files, functions, and code patterns from the provided context
- Be precise about file paths and code locations
- If the context doesn't contain enough information to answer, say so clearly
- Provide actionable, implementation-ready responses
- When suggesting code changes, show the exact file and location"""

CODE_SOURCE_TYPES = {"git"}
DOCS_SOURCE_TYPES = {"confluence", "knowledge"}

SYNTHESIS_PROMPT = """You are an expert software engineer assistant. You have been given two specialist analyses of a question — one focused on code context and one focused on documentation/best practices. Synthesize them into a single, coherent answer.

When synthesizing:
- Combine insights from both analyses without repeating information
- Reference specific files, functions, and documentation sources mentioned
- If the analyses conflict, note the discrepancy and recommend the best approach
- Provide a unified, actionable response"""


class RAGPipeline:
    async def build_context(
        self,
        query: str,
        vector_store: VectorStore,
        source_type: str | None = None,
        source_name: str | list[str] | None = None,
    ) -> list[dict]:
        query_embedding = await embedding_client.embed_query(query)

        retrieval_k = settings.rag_retrieval_k if reranker.enabled else settings.rag_top_k

        if isinstance(source_name, list) and len(source_name) > 1:
            per_source_k = max(retrieval_k // len(source_name), 5)
            candidates = []
            for name in source_name:
                source_candidates = await vector_store.similarity_search(
                    query_embedding=query_embedding,
                    top_k=per_source_k,
                    similarity_threshold=settings.rag_similarity_threshold,
                    source_type=source_type,
                    source_name=name,
                )
                candidates.extend(source_candidates)
        elif source_name is None:
            all_sources = await vector_store.get_distinct_sources(source_type)
            if len(all_sources) > 1:
                per_source_k = max(retrieval_k // len(all_sources), 3)
                candidates = []
                for src in all_sources:
                    source_candidates = await vector_store.similarity_search(
                        query_embedding=query_embedding,
                        top_k=per_source_k,
                        similarity_threshold=settings.rag_similarity_threshold,
                        source_type=source_type,
                        source_name=src,
                    )
                    candidates.extend(source_candidates)
            else:
                candidates = await vector_store.similarity_search(
                    query_embedding=query_embedding,
                    top_k=retrieval_k,
                    similarity_threshold=settings.rag_similarity_threshold,
                    source_type=source_type,
                    source_name=source_name,
                )
        else:
            candidates = await vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=retrieval_k,
                similarity_threshold=settings.rag_similarity_threshold,
                source_type=source_type,
                source_name=source_name,
            )

        results = await reranker.rerank(query, candidates)
        return results

    def format_context(self, results: list[dict]) -> str:
        if not results:
            return "No relevant context found."

        parts = []
        for i, r in enumerate(results, 1):
            header = f"[{i}] {r['source_type']}:{r['source_name']}"
            if r.get("file_path"):
                header += f" — {r['file_path']}"
            if r.get("language"):
                header += f" ({r['language']})"
            if r.get("rerank_score") is not None:
                header += f" (rerank: {r['rerank_score']:.3f}, sim: {r['similarity']:.3f})"
            else:
                header += f" (similarity: {r['similarity']:.3f})"

            parts.append(f"{header}\n```\n{r['content']}\n```")

        return "\n\n".join(parts)

    def _split_by_source_type(self, results: list[dict]) -> tuple[list[dict], list[dict]]:
        code_results = [r for r in results if r.get("source_type") in CODE_SOURCE_TYPES]
        docs_results = [r for r in results if r.get("source_type") in DOCS_SOURCE_TYPES]
        return code_results, docs_results

    async def _single_pass(
        self,
        question: str,
        context_results: list[dict],
        model: str,
        conversation_history: list[dict] | None = None,
    ) -> str:
        context_text = self.format_context(context_results)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": f"## Retrieved Context\n{context_text}\n\n## Question\n{question}"})
        return await llm_client.chat(messages, model=model)

    async def query(
        self,
        question: str,
        vector_store: VectorStore,
        conversation_history: list[dict] | None = None,
        source_type: str | None = None,
        source_name: str | list[str] | None = None,
    ) -> dict:
        context_results = await self.build_context(
            question, vector_store, source_type, source_name
        )

        code_results, docs_results = self._split_by_source_type(context_results)

        if code_results and docs_results:
            code_answer, docs_answer = await asyncio.gather(
                self._single_pass(question, code_results, settings.llm_code_model, conversation_history),
                self._single_pass(question, docs_results, settings.llm_docs_model, conversation_history),
            )

            synthesis_messages = [
                {"role": "system", "content": SYNTHESIS_PROMPT},
            ]
            if conversation_history:
                synthesis_messages.extend(conversation_history)
            synthesis_messages.append({
                "role": "user",
                "content": (
                    f"## Question\n{question}\n\n"
                    f"## Code Analysis\n{code_answer}\n\n"
                    f"## Documentation Analysis\n{docs_answer}"
                ),
            })
            answer = await llm_client.chat(synthesis_messages, model=settings.llm_docs_model)
        elif code_results:
            answer = await self._single_pass(question, code_results, settings.llm_code_model, conversation_history)
        elif docs_results:
            answer = await self._single_pass(question, docs_results, settings.llm_docs_model, conversation_history)
        else:
            answer = await self._single_pass(question, context_results, settings.llm_model, conversation_history)

        return {
            "answer": answer,
            "context": context_results,
            "context_count": len(context_results),
        }

    def _select_model(self, context_results: list[dict]) -> str:
        code_results, docs_results = self._split_by_source_type(context_results)
        if code_results and not docs_results:
            return settings.llm_code_model
        if docs_results and not code_results:
            return settings.llm_docs_model
        return settings.llm_docs_model

    async def query_stream(
        self,
        question: str,
        vector_store: VectorStore,
        conversation_history: list[dict] | None = None,
        source_type: str | None = None,
        source_name: str | list[str] | None = None,
    ):
        context_results = await self.build_context(
            question, vector_store, source_type, source_name
        )
        context_text = self.format_context(context_results)
        model = self._select_model(context_results)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if conversation_history:
            messages.extend(conversation_history)

        user_message = f"""## Retrieved Context
{context_text}

## Question
{question}"""

        messages.append({"role": "user", "content": user_message})

        async for token in llm_client.chat_stream(messages, model=model):
            yield token


rag_pipeline = RAGPipeline()
