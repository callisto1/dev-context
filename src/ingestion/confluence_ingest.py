import base64
import html
import logging
import re

import httpx

from src.config import settings
from src.core.llm import llm_client
from src.ingestion.base import BaseIngester, ChunkedDocument

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 3000
OVERLAP_CHARS = 200
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


class ConfluenceIngester(BaseIngester):
    """Ingests pages from Confluence Cloud using the REST API v2.

    Supports ingesting:
      - A single page by ID
      - All pages in a space by space key
    """

    def __init__(self):
        self.base_url = settings.confluence_base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            timeout=60.0,
            auth=(settings.confluence_user_email, settings.confluence_api_token),
        )

    async def ingest(
        self,
        source: str,
        space_key: str | None = None,
        page_ids: list[str] | None = None,
        max_pages: int = 500,
        include_labels: list[str] | None = None,
        include_children: bool = True,
    ) -> list[ChunkedDocument]:
        """Ingest from Confluence.

        Args:
            source: Space key (used as source_name) or ignored if page_ids given.
            space_key: Confluence space key to crawl.
            page_ids: Explicit list of page IDs to ingest.
            max_pages: Max pages to fetch when crawling a space.
            include_labels: Only ingest pages with at least one of these labels.
            include_children: When using page_ids, also fetch all descendant subpages.
        """
        pages = []

        if page_ids:
            seen_ids = set()
            for pid in page_ids:
                page = await self._fetch_page(pid)
                if page and page["id"] not in seen_ids:
                    pages.append(page)
                    seen_ids.add(page["id"])
                if include_children:
                    children = await self._fetch_descendants(pid)
                    for child in children:
                        if child["id"] not in seen_ids:
                            pages.append(child)
                            seen_ids.add(child["id"])
        elif space_key:
            pages = await self._fetch_space_pages(space_key, max_pages)
        else:
            space_key = source
            pages = await self._fetch_space_pages(space_key, max_pages)

        if include_labels:
            label_set = set(include_labels)
            pages = [p for p in pages if label_set & set(p.get("labels", []))]

        source_name = space_key or source
        documents = []

        for page in pages:
            page_docs = await self._process_page(page, source_name)
            documents.extend(page_docs)

        logger.info(
            f"Ingested {len(pages)} pages, {len(documents)} chunks from Confluence:{source_name}"
        )
        return documents

    async def cleanup(self, source: str) -> None:
        pass

    async def close(self):
        await self.client.aclose()

    async def _fetch_page(self, page_id: str) -> dict | None:
        try:
            resp = await self.client.get(
                f"{self.base_url}/wiki/api/v2/pages/{page_id}",
                params={"body-format": "storage"},
            )
            resp.raise_for_status()
            data = resp.json()

            labels = await self._fetch_page_labels(page_id)

            return {
                "id": data["id"],
                "title": data["title"],
                "space_key": data.get("spaceId", ""),
                "body": data.get("body", {}).get("storage", {}).get("value", ""),
                "version": data.get("version", {}).get("number", 1),
                "labels": labels,
                "url": f"{self.base_url}/wiki/spaces/{data.get('spaceId', '')}/pages/{data['id']}",
            }
        except httpx.HTTPStatusError as e:
            logger.error(f"Failed to fetch page {page_id}: {e.response.status_code}")
            return None

    async def _fetch_page_labels(self, page_id: str) -> list[str]:
        try:
            resp = await self.client.get(
                f"{self.base_url}/wiki/api/v2/pages/{page_id}/labels",
            )
            resp.raise_for_status()
            data = resp.json()
            return [label["name"] for label in data.get("results", [])]
        except httpx.HTTPStatusError:
            return []

    async def _resolve_space_id(self, space_key: str) -> str:
        resp = await self.client.get(
            f"{self.base_url}/wiki/api/v2/spaces",
            params={"keys": space_key, "limit": 1},
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            raise ValueError(f"Confluence space not found: {space_key}")
        space_id = results[0]["id"]
        logger.info(f"Resolved space key '{space_key}' to space ID '{space_id}'")
        return space_id

    async def _fetch_space_pages(self, space_key: str, max_pages: int) -> list[dict]:
        space_id = await self._resolve_space_id(space_key)
        pages = []
        cursor = None

        while len(pages) < max_pages:
            params = {
                "space-id": space_id,
                "body-format": "storage",
                "limit": min(50, max_pages - len(pages)),
                "status": "current",
            }
            if cursor:
                params["cursor"] = cursor

            try:
                resp = await self.client.get(
                    f"{self.base_url}/wiki/api/v2/pages",
                    params=params,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"Failed to fetch pages for space {space_key}: {e.response.status_code}")
                break

            for item in data.get("results", []):
                labels = await self._fetch_page_labels(item["id"])
                pages.append({
                    "id": item["id"],
                    "title": item["title"],
                    "space_key": space_key,
                    "body": item.get("body", {}).get("storage", {}).get("value", ""),
                    "version": item.get("version", {}).get("number", 1),
                    "labels": labels,
                    "url": f"{self.base_url}/wiki/spaces/{space_key}/pages/{item['id']}",
                })

            next_link = data.get("_links", {}).get("next")
            if not next_link:
                break

            cursor = _extract_cursor(next_link)
            if not cursor:
                break

        logger.info(f"Fetched {len(pages)} pages from space {space_key}")
        return pages

    async def _fetch_descendants(self, page_id: str) -> list[dict]:
        descendants = []
        queue = [page_id]

        while queue:
            current_id = queue.pop(0)
            cursor = None

            while True:
                params = {
                    "body-format": "storage",
                    "limit": 50,
                }
                if cursor:
                    params["cursor"] = cursor

                try:
                    resp = await self.client.get(
                        f"{self.base_url}/wiki/api/v2/pages/{current_id}/children",
                        params=params,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except httpx.HTTPStatusError as e:
                    logger.warning(f"Failed to fetch children of page {current_id}: {e.response.status_code}")
                    break

                for item in data.get("results", []):
                    labels = await self._fetch_page_labels(item["id"])
                    child = {
                        "id": item["id"],
                        "title": item["title"],
                        "space_key": item.get("spaceId", ""),
                        "body": item.get("body", {}).get("storage", {}).get("value", ""),
                        "version": item.get("version", {}).get("number", 1),
                        "labels": labels,
                        "url": f"{self.base_url}/wiki/spaces/{item.get('spaceId', '')}/pages/{item['id']}",
                    }
                    descendants.append(child)
                    queue.append(item["id"])

                next_link = data.get("_links", {}).get("next")
                if not next_link:
                    break
                cursor = _extract_cursor(next_link)
                if not cursor:
                    break

        logger.info(f"Fetched {len(descendants)} descendant pages from page {page_id}")
        return descendants

    async def _fetch_page_attachments(self, page_id: str) -> list[dict]:
        attachments = []
        try:
            resp = await self.client.get(
                f"{self.base_url}/wiki/api/v2/pages/{page_id}/attachments",
                params={"limit": 100},
            )
            resp.raise_for_status()
            data = resp.json()
            for att in data.get("results", []):
                title = att.get("title", "")
                ext = title.rsplit(".", 1)[-1].lower() if "." in title else ""
                if f".{ext}" in IMAGE_EXTENSIONS:
                    download_url = att.get("downloadLink") or att.get("_links", {}).get("download")
                    if download_url:
                        if download_url.startswith("/"):
                            download_url = f"{self.base_url}/wiki{download_url}"
                        attachments.append({
                            "filename": title,
                            "download_url": download_url,
                            "size": att.get("fileSize", 0),
                        })
        except httpx.HTTPStatusError as e:
            logger.warning(f"Failed to fetch attachments for page {page_id}: {e.response.status_code}")
        return attachments

    async def _download_image(self, url: str) -> bytes | None:
        try:
            resp = await self.client.get(url, follow_redirects=True)
            resp.raise_for_status()
            if len(resp.content) > MAX_IMAGE_SIZE:
                logger.warning(f"Image too large ({len(resp.content)} bytes), skipping: {url}")
                return None
            return resp.content
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            logger.warning(f"Failed to download image {url}: {e}")
            return None

    async def _describe_image(self, image_bytes: bytes, page_title: str, filename: str) -> str | None:
        try:
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            description = await llm_client.describe_image(
                image_b64=image_b64,
                context=f"{page_title} — {filename}",
            )
            return description
        except Exception as e:
            logger.warning(f"Failed to describe image {filename}: {e}")
            return None

    async def _extract_image_descriptions(self, page: dict) -> list[dict]:
        page_id = page["id"]
        page_title = page.get("title", "Untitled")
        raw_html = page.get("body", "")

        referenced_filenames = set()
        for match in re.finditer(r'<ri:attachment\s+ri:filename="([^"]+)"', raw_html):
            referenced_filenames.add(match.group(1))
        for match in re.finditer(r'<img[^>]+src="([^"]*(/download/attachments/[^"]*|/rest/api/content/[^"]*))"', raw_html):
            referenced_filenames.add(match.group(1).rsplit("/", 1)[-1])

        attachments = await self._fetch_page_attachments(page_id)

        if referenced_filenames:
            attachments = [a for a in attachments if a["filename"] in referenced_filenames]

        if not attachments:
            return []

        descriptions = []
        for att in attachments:
            image_bytes = await self._download_image(att["download_url"])
            if not image_bytes:
                continue

            description = await self._describe_image(image_bytes, page_title, att["filename"])
            if description:
                descriptions.append({
                    "filename": att["filename"],
                    "description": description,
                })
                logger.info(f"Described image {att['filename']} from page '{page_title}'")

        return descriptions

    async def _process_page(self, page: dict, source_name: str) -> list[ChunkedDocument]:
        raw_html = page.get("body", "")
        text = _html_to_text(raw_html)
        title = page.get("title", "Untitled")

        image_descriptions = await self._extract_image_descriptions(page)

        if image_descriptions:
            image_section = "\n\n## Images\n\n"
            for img in image_descriptions:
                image_section += f"**{img['filename']}:**\n{img['description']}\n\n"
            text = text + image_section if text.strip() else image_section

        if not text.strip():
            return []

        header = f"# {title}\n\n"
        full_text = header + text

        chunks = _chunk_text(full_text)

        documents = []
        for i, chunk_text in enumerate(chunks):
            documents.append(
                ChunkedDocument(
                    source_type="confluence",
                    source_name=source_name,
                    file_path=title,
                    chunk_index=i,
                    chunk_type="wiki_page",
                    language=None,
                    content=chunk_text,
                    metadata={
                        "page_id": page["id"],
                        "title": title,
                        "version": page.get("version", 1),
                        "labels": page.get("labels", []),
                        "url": page.get("url", ""),
                        "images": [img["filename"] for img in image_descriptions],
                    },
                )
            )
        return documents


def _html_to_text(html_content: str) -> str:
    if not html_content:
        return ""

    text = re.sub(r"<ac:structured-macro[^>]*>.*?</ac:structured-macro>", "", html_content, flags=re.DOTALL)

    text = re.sub(r"<h[1-6][^>]*>(.*?)</h[1-6]>", r"\n## \1\n", text)
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text)
    text = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n", text)
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<ac:plain-text-body><!\[CDATA\[(.*?)\]\]></ac:plain-text-body>", r"\n```\n\1\n```\n", text, flags=re.DOTALL)

    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _chunk_text(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    chunks = []
    sections = re.split(r"(^## .+$)", text, flags=re.MULTILINE)

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

    final_chunks = []
    for chunk in chunks:
        if len(chunk) > MAX_CHUNK_CHARS:
            final_chunks.extend(_split_long_chunk(chunk))
        else:
            final_chunks.append(chunk)

    return final_chunks


def _split_long_chunk(text: str) -> list[str]:
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


def _extract_cursor(next_link: str) -> str | None:
    match = re.search(r"cursor=([^&]+)", next_link)
    return match.group(1) if match else None
