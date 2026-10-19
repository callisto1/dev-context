import asyncio
import logging
import re
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from src.ingestion.base import BaseIngester, ChunkedDocument

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 3000
OVERLAP_CHARS = 200
REQUEST_DELAY = 0.5  # seconds between requests to be polite


class WebDocsIngester(BaseIngester):
    """Crawls a documentation website and ingests pages as text chunks."""

    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "DevContextEngine/0.1 (documentation indexer)"
            },
        )

    async def ingest(
        self,
        source: str,
        seed_url: str | None = None,
        max_pages: int = 100,
        allowed_path_prefix: str | None = None,
    ) -> list[ChunkedDocument]:
        start_url = seed_url or source
        parsed = urlparse(start_url)
        base_domain = parsed.netloc
        path_prefix = allowed_path_prefix or parsed.path

        visited: set[str] = set()
        queue: list[str] = [start_url]
        pages: list[dict] = []

        while queue and len(pages) < max_pages:
            url = queue.pop(0)
            normalized = self._normalize_url(url)

            if normalized in visited:
                continue
            visited.add(normalized)

            page = await self._fetch_page(url)
            if not page:
                continue

            pages.append(page)

            for link in page.get("links", []):
                link_parsed = urlparse(link)
                if link_parsed.netloc != base_domain:
                    continue
                if not link_parsed.path.startswith(path_prefix):
                    continue
                if self._normalize_url(link) not in visited:
                    queue.append(link)

            await asyncio.sleep(REQUEST_DELAY)

        documents = []
        for page in pages:
            page_docs = self._process_page(page, source)
            documents.extend(page_docs)

        logger.info(f"Crawled {len(pages)} pages, {len(documents)} chunks from {source}")
        return documents

    async def cleanup(self, source: str) -> None:
        pass

    async def close(self):
        await self.client.aclose()

    async def _fetch_page(self, url: str) -> dict | None:
        try:
            resp = await self.client.get(url)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "text/html" not in content_type:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            body = self._extract_content(soup)
            links = self._extract_links(soup, url)

            if not body.strip():
                return None

            return {
                "url": url,
                "title": title,
                "body": body,
                "links": links,
            }

        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return None

    def _extract_content(self, soup: BeautifulSoup) -> str:
        for tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style", "noscript"]):
            tag.decompose()

        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find(class_=re.compile(r"(content|docs|article|post)", re.I))
        )
        target = main or soup.body or soup

        lines = []
        for el in target.descendants:
            if el.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
                level = int(el.name[1])
                prefix = "#" * level
                text = el.get_text(strip=True)
                if text:
                    lines.append(f"\n{prefix} {text}\n")
            elif el.name == "li":
                text = el.get_text(strip=True)
                if text:
                    lines.append(f"- {text}")
            elif el.name == "pre":
                code = el.get_text()
                if code.strip():
                    lines.append(f"\n```\n{code.strip()}\n```\n")
            elif el.name == "code" and el.parent and el.parent.name != "pre":
                continue  # inline code, handled by parent text
            elif el.name == "p":
                text = el.get_text(strip=True)
                if text:
                    lines.append(f"{text}\n")
            elif el.name == "br":
                lines.append("")

        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
                continue
            absolute = urljoin(base_url, href)
            absolute = absolute.split("#")[0].split("?")[0]
            links.append(absolute)
        return links

    def _normalize_url(self, url: str) -> str:
        return url.split("#")[0].split("?")[0].rstrip("/")

    def _process_page(self, page: dict, source_name: str) -> list[ChunkedDocument]:
        text = page["body"]
        if not text.strip():
            return []

        title = page.get("title", "")
        if title:
            text = f"# {title}\n\n{text}"

        chunks = _chunk_text(text)
        documents = []

        for i, chunk_text in enumerate(chunks):
            documents.append(
                ChunkedDocument(
                    source_type="knowledge",
                    source_name=source_name,
                    file_path=page.get("url", ""),
                    chunk_index=i,
                    chunk_type="web_doc",
                    language=None,
                    content=chunk_text,
                    metadata={
                        "url": page.get("url", ""),
                        "title": page.get("title", ""),
                    },
                )
            )
        return documents


def _chunk_text(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    chunks = []
    sections = re.split(r"(^#{1,4} .+$)", text, flags=re.MULTILINE)

    current = ""
    for section in sections:
        if len(current) + len(section) > MAX_CHUNK_CHARS and current:
            chunks.append(current.strip())
            overlap = current[-OVERLAP_CHARS:] if len(current) > OVERLAP_CHARS else current
            current = overlap + section
        else:
            current += section

    if current.strip():
        chunks.append(current.strip())

    final = []
    for chunk in chunks:
        if len(chunk) > MAX_CHUNK_CHARS:
            final.extend(_split_by_paragraphs(chunk))
        else:
            final.append(chunk)

    return final


def _split_by_paragraphs(text: str) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 > MAX_CHUNK_CHARS and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    return chunks
