import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.config import settings


class Base(DeclarativeBase):
    pass


class IndexedSource(Base):
    """Tracks sources registered for periodic reindexing."""

    __tablename__ = "indexed_sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # git, confluence
    source_name: Mapped[str] = mapped_column(String(500), nullable=False)
    source_uri: Mapped[str] = mapped_column(String(1000), nullable=False)  # repo URL, space key
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    cron_expression: Mapped[str] = mapped_column(
        String(100), default="0 0 * * *"
    )  # default: daily at midnight
    ingest_kwargs_json: Mapped[str | None] = mapped_column(Text)  # JSON: branch, patterns, etc.
    last_commit_hash: Mapped[str | None] = mapped_column(String(64))  # git HEAD at last index
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_status: Mapped[str | None] = mapped_column(String(50))  # success, error, skipped, running
    last_error: Mapped[str | None] = mapped_column(Text)
    last_chunk_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)  # git, confluence, jira
    source_name: Mapped[str] = mapped_column(String(500), nullable=False)  # repo name, page title
    file_path: Mapped[str | None] = mapped_column(String(1000))
    chunk_index: Mapped[int] = mapped_column(default=0)
    chunk_type: Mapped[str] = mapped_column(String(50), default="code")  # code, function, class, doc
    language: Mapped[str | None] = mapped_column(String(50))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(settings.embedding_dimension), nullable=True)
    metadata_json: Mapped[str | None] = mapped_column(Text)  # extra metadata as JSON
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index(
            "ix_documents_embedding_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_documents_source", "source_type", "source_name"),
    )
