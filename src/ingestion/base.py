from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChunkedDocument:
    source_type: str
    source_name: str
    file_path: str | None = None
    chunk_index: int = 0
    chunk_type: str = "code"
    language: str | None = None
    content: str = ""
    metadata: dict = field(default_factory=dict)


class BaseIngester(ABC):
    @abstractmethod
    async def ingest(self, source: str, **kwargs) -> list[ChunkedDocument]:
        ...

    @abstractmethod
    async def cleanup(self, source: str) -> None:
        ...
