import json
import uuid

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Document


class VectorStore:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_documents(self, docs: list[dict]) -> int:
        count = 0
        for doc in docs:
            record = Document(
                id=doc.get("id", uuid.uuid4()),
                source_type=doc["source_type"],
                source_name=doc["source_name"],
                file_path=doc.get("file_path"),
                chunk_index=doc.get("chunk_index", 0),
                chunk_type=doc.get("chunk_type", "code"),
                language=doc.get("language"),
                content=doc["content"],
                embedding=doc.get("embedding"),
                metadata_json=json.dumps(doc.get("metadata")) if doc.get("metadata") else None,
            )
            self.session.add(record)
            count += 1
        await self.session.commit()
        return count

    async def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        source_type: str | None = None,
        source_name: str | list[str] | None = None,
    ) -> list[dict]:
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        filters = []
        params: dict = {
            "embedding": embedding_str,
            "threshold": similarity_threshold,
            "top_k": top_k,
        }

        if source_type:
            filters.append("source_type = :source_type")
            params["source_type"] = source_type
        if source_name:
            if isinstance(source_name, list):
                placeholders = ", ".join(f":sn_{i}" for i in range(len(source_name)))
                filters.append(f"source_name IN ({placeholders})")
                for i, name in enumerate(source_name):
                    params[f"sn_{i}"] = name
            else:
                filters.append("source_name = :source_name")
                params["source_name"] = source_name

        where_clause = ""
        if filters:
            where_clause = "AND " + " AND ".join(filters)

        query = text(f"""
            SELECT
                id, source_type, source_name, file_path,
                chunk_index, chunk_type, language, content,
                metadata_json,
                1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM documents
            WHERE 1 - (embedding <=> CAST(:embedding AS vector)) >= :threshold
            {where_clause}
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)

        result = await self.session.execute(query, params)
        rows = result.fetchall()

        return [
            {
                "id": str(row.id),
                "source_type": row.source_type,
                "source_name": row.source_name,
                "file_path": row.file_path,
                "chunk_index": row.chunk_index,
                "chunk_type": row.chunk_type,
                "language": row.language,
                "content": row.content,
                "metadata": json.loads(row.metadata_json) if row.metadata_json else None,
                "similarity": float(row.similarity),
            }
            for row in rows
        ]

    async def get_distinct_sources(self, source_type: str | None = None) -> list[str]:
        if source_type:
            result = await self.session.execute(
                text("SELECT DISTINCT source_name FROM documents WHERE source_type = :st"),
                {"st": source_type},
            )
        else:
            result = await self.session.execute(
                text("SELECT DISTINCT source_name FROM documents")
            )
        return [row[0] for row in result.fetchall()]

    async def count_by_source(self, source_type: str, source_name: str) -> int:
        result = await self.session.execute(
            text("SELECT COUNT(*) FROM documents WHERE source_type = :st AND source_name = :sn"),
            {"st": source_type, "sn": source_name},
        )
        return result.scalar() or 0

    async def delete_by_source(self, source_type: str, source_name: str) -> int:
        result = await self.session.execute(
            delete(Document).where(
                Document.source_type == source_type,
                Document.source_name == source_name,
            )
        )
        await self.session.commit()
        return result.rowcount

    async def get_stats(self) -> dict:
        result = await self.session.execute(
            text("""
                SELECT source_type, source_name, COUNT(*) as chunk_count
                FROM documents
                GROUP BY source_type, source_name
                ORDER BY source_type, source_name
            """)
        )
        rows = result.fetchall()
        return {
            "total_chunks": sum(r.chunk_count for r in rows),
            "sources": [
                {
                    "source_type": r.source_type,
                    "source_name": r.source_name,
                    "chunk_count": r.chunk_count,
                }
                for r in rows
            ],
        }
