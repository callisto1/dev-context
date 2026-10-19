import logging
import os
import shutil
import time

from git import Repo, RemoteProgress

from src.config import settings
from src.ingestion.base import BaseIngester, ChunkedDocument
from src.ingestion.chunker import (
    chunk_code,
    detect_language,
    should_skip_dir,
    should_skip_file,
)

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 500_000  # 500KB


class _CloneProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=""):
        if max_count:
            pct = int(cur_count / max_count * 100)
            if pct % 20 == 0:
                logger.info(f"[clone] {pct}% ({cur_count}/{max_count}) {message.strip()}")


class GitIngester(BaseIngester):
    def __init__(self):
        self.clone_dir = settings.git_clone_dir
        os.makedirs(self.clone_dir, exist_ok=True)

    async def ingest(
        self,
        source: str,
        branch: str | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        source_type_override: str | None = None,
        source_name_override: str | None = None,
    ) -> list[ChunkedDocument]:
        repo_name = source_name_override or self._extract_repo_name(source)
        source_type = source_type_override or "git"
        repo_path = os.path.join(self.clone_dir, self._extract_repo_name(source))

        clone_url = self._inject_token(source)
        partial = bool(include_patterns)

        if os.path.exists(repo_path):
            try:
                logger.info(f"Updating existing clone: {repo_name}")
                repo = Repo(repo_path)
                repo.remotes.origin.set_url(clone_url)
                repo.remotes.origin.pull()
            except Exception:
                logger.warning(f"Pull failed for {repo_name}, removing stale clone and re-cloning")
                shutil.rmtree(repo_path)
                repo = self._fresh_clone(clone_url, repo_path, branch, partial=partial)
        else:
            repo = self._fresh_clone(clone_url, repo_path, branch, partial=partial)

        documents = []
        file_count = 0
        skipped_count = 0
        chunk_count = 0

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]
            rel_dir = os.path.relpath(root, repo_path)
            if len(files) > 100:
                logger.info(f"[{repo_name}] Scanning {rel_dir}/ ({len(files)} files)")

            for filename in files:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, repo_path)

                if should_skip_file(relative_path):
                    skipped_count += 1
                    continue

                if include_patterns and not any(
                    _match_pattern(relative_path, p) for p in include_patterns
                ):
                    skipped_count += 1
                    continue

                if exclude_patterns and any(
                    _match_pattern(relative_path, p) for p in exclude_patterns
                ):
                    skipped_count += 1
                    continue

                if os.path.getsize(file_path) > MAX_FILE_SIZE:
                    skipped_count += 1
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except (OSError, UnicodeDecodeError):
                    skipped_count += 1
                    continue

                if not content.strip():
                    skipped_count += 1
                    continue

                if "\x00" in content:
                    skipped_count += 1
                    continue

                language = detect_language(relative_path)
                t0 = time.monotonic()
                chunks = chunk_code(content, language)
                elapsed = time.monotonic() - t0
                if elapsed > 2.0:
                    logger.warning(f"[{repo_name}] Slow chunk: {relative_path} ({len(content)} chars, {elapsed:.1f}s)")

                for i, chunk in enumerate(chunks):
                    documents.append(
                        ChunkedDocument(
                            source_type=source_type,
                            source_name=repo_name,
                            file_path=relative_path,
                            chunk_index=i,
                            chunk_type=chunk.chunk_type,
                            language=language,
                            content=chunk.content,
                            metadata={
                                "start_line": chunk.start_line,
                                "end_line": chunk.end_line,
                                "repo_url": source,
                                "branch": branch or "default",
                            },
                        )
                    )

                file_count += 1
                chunk_count += len(chunks)

                if file_count % 50 == 0:
                    logger.info(f"[{repo_name}] Processed {file_count} files, {chunk_count} chunks so far (skipped {skipped_count})")

        logger.info(f"Ingested {file_count} files, {len(documents)} chunks from {repo_name} (skipped {skipped_count} files)")
        return documents

    async def cleanup(self, source: str) -> None:
        repo_name = self._extract_repo_name(source)
        repo_path = os.path.join(self.clone_dir, repo_name)
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

    @staticmethod
    def _fresh_clone(clone_url: str, repo_path: str, branch: str | None, partial: bool = True) -> Repo:
        logger.info(f"Cloning into {repo_path} (partial={partial})")
        clone_kwargs = {
            "depth": 1,
            "single_branch": True,
        }
        if partial:
            clone_kwargs["filter"] = "blob:none"
        if branch:
            clone_kwargs["branch"] = branch
        return Repo.clone_from(clone_url, repo_path, progress=_CloneProgress(), **clone_kwargs)

    @staticmethod
    def _inject_token(url: str) -> str:
        token = settings.git_token
        if token and url.startswith("https://"):
            user = settings.git_user or "x-access-token"
            return url.replace("https://", f"https://{user}:{token}@", 1)
        return url

    @staticmethod
    def _extract_repo_name(source: str) -> str:
        if source.startswith(("http://", "https://", "git@")):
            name = source.rstrip("/").rsplit("/", 1)[-1]
            return name.removesuffix(".git")
        return os.path.basename(os.path.abspath(source))


def _match_pattern(path: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(path, pattern)
