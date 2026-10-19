import logging
import re
from dataclasses import dataclass

from chonkie import CodeChunker as ChonkieCodeChunker
from chonkie import RecursiveChunker as ChonkieRecursiveChunker
from chonkie import RecursiveLevel, RecursiveRules

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".kt": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".php": "php",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
    ".rst": "markdown",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".tf": "terraform",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".dockerfile": "dockerfile",
}

CHONKIE_LANGUAGE_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "go": "go",
    "rust": "rust",
    "ruby": "ruby",
    "cpp": "cpp",
    "c": "c",
    "csharp": "c_sharp",
    "kotlin": "kotlin",
    "swift": "swift",
    "scala": "scala",
    "php": "php",
    "shell": "bash",
}

SKIP_EXTENSIONS = {
    ".min.js", ".min.css", ".map", ".lock", ".svg", ".png", ".jpg",
    ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".eot",
    ".pyc", ".pyo", ".class", ".o", ".so", ".dll", ".exe",
    ".zip", ".tar", ".gz", ".jar", ".war", ".bin", ".dat",
    ".ipynb", ".parquet", ".pickle", ".pkl", ".npy", ".npz",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".flv",
    ".db", ".sqlite", ".sqlite3",
    ".whl", ".egg", ".rpm", ".deb", ".msi", ".dmg", ".iso",
    ".7z", ".rar", ".bz2", ".xz", ".zst", ".tgz",
    ".pb", ".onnx", ".pt", ".h5", ".safetensors",
    ".img", ".raw", ".bmp", ".tiff", ".webp", ".heic",
}

SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Pipfile.lock", "composer.lock", "Gemfile.lock", "Cargo.lock",
    "go.sum", "flake.lock",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "target", "vendor", ".idea", ".vscode",
    "tests", "test", "__tests__", "spec", "specs",
    "test_fixtures", "fixtures", "testdata", "test_data",
    "wheels", ".wheels", "artifacts",
}

MAX_CHUNK_TOKENS = 512

_TEST_FILE_PATTERNS = re.compile(
    r"(^|/)test_[^/]+\.\w+$|_test\.\w+$|\.test\.\w+$|\.spec\.\w+$|conftest\.py$",
)

_code_chunkers: dict[str, ChonkieCodeChunker] = {}
_md_chunker: ChonkieRecursiveChunker | None = None


def _get_code_chunker(language: str) -> ChonkieCodeChunker:
    if language not in _code_chunkers:
        _code_chunkers[language] = ChonkieCodeChunker(
            chunk_size=MAX_CHUNK_TOKENS,
            language=language,
        )
    return _code_chunkers[language]


def _get_md_chunker() -> ChonkieRecursiveChunker:
    global _md_chunker
    if _md_chunker is None:
        md_rules = RecursiveRules(levels=[
            RecursiveLevel(delimiters=["\n# ", "\n## ", "\n### ", "\n#### "]),
            RecursiveLevel(delimiters=["\n\n", "\n---\n", "\n***\n"]),
            RecursiveLevel(delimiters=["\n- ", "\n* ", "\n1. "]),
            RecursiveLevel(delimiters=[". ", "! ", "? "]),
            RecursiveLevel(whitespace=True),
        ])
        _md_chunker = ChonkieRecursiveChunker(
            chunk_size=MAX_CHUNK_TOKENS,
            rules=md_rules,
        )
    return _md_chunker


@dataclass
class Chunk:
    content: str
    chunk_type: str
    start_line: int = 0
    end_line: int = 0


def detect_language(file_path: str) -> str | None:
    for ext, lang in LANGUAGE_MAP.items():
        if file_path.endswith(ext):
            return lang
    if file_path.endswith("Dockerfile") or file_path.endswith("dockerfile"):
        return "dockerfile"
    return None


def should_skip_file(file_path: str) -> bool:
    basename = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
    if basename in SKIP_FILES:
        return True
    for ext in SKIP_EXTENSIONS:
        if file_path.endswith(ext):
            return True
    if _TEST_FILE_PATTERNS.search(file_path):
        return True
    return False


def should_skip_dir(dir_name: str) -> bool:
    return dir_name in SKIP_DIRS


def chunk_code(content: str, language: str | None) -> list[Chunk]:
    if not content.strip():
        return []

    if language == "markdown":
        return _chunk_with_chonkie_md(content)

    chonkie_lang = CHONKIE_LANGUAGE_MAP.get(language or "") if language else None
    if chonkie_lang:
        return _chunk_with_chonkie_code(content, chonkie_lang)

    return _fallback_chunk(content)


def _chunk_with_chonkie_code(content: str, language: str) -> list[Chunk]:
    try:
        chunker = _get_code_chunker(language)
        chonkie_chunks = chunker.chunk(content)
        return _convert_chonkie_chunks(content, chonkie_chunks, "code")
    except Exception as e:
        logger.warning(f"Chonkie CodeChunker failed for {language}: {e}, falling back")
        return _fallback_chunk(content)


def _chunk_with_chonkie_md(content: str) -> list[Chunk]:
    try:
        chunker = _get_md_chunker()
        chonkie_chunks = chunker.chunk(content)
        return _convert_chonkie_chunks(content, chonkie_chunks, "section")
    except Exception as e:
        logger.warning(f"Chonkie MarkdownChunker failed: {e}, falling back")
        return _fallback_chunk(content)


def _convert_chonkie_chunks(content: str, chonkie_chunks, chunk_type: str) -> list[Chunk]:
    results = []
    for c in chonkie_chunks:
        text = c.text.strip()
        if not text:
            continue
        start_line = content[: c.start_index].count("\n") if c.start_index > 0 else 0
        end_line = content[: c.end_index].count("\n")
        results.append(Chunk(
            content=text,
            chunk_type=chunk_type,
            start_line=start_line,
            end_line=end_line,
        ))
    if not results:
        return _fallback_chunk(content)
    return results


def _fallback_chunk(content: str) -> list[Chunk]:
    max_chars = 3000
    lines = content.split("\n")
    chunks = []
    current_lines = []
    current_chars = 0
    start_line = 0

    for i, line in enumerate(lines):
        current_lines.append(line)
        current_chars += len(line) + 1

        if current_chars >= max_chars:
            chunks.append(Chunk(
                content="\n".join(current_lines),
                chunk_type="file_section",
                start_line=start_line,
                end_line=i,
            ))
            overlap_lines = current_lines[-3:]
            current_lines = overlap_lines
            current_chars = sum(len(l) + 1 for l in current_lines)
            start_line = max(0, i - 2)

    if current_lines:
        chunks.append(Chunk(
            content="\n".join(current_lines),
            chunk_type="file_section",
            start_line=start_line,
            end_line=len(lines) - 1,
        ))

    return chunks
