from src.ingestion.chunker import (
    chunk_code,
    detect_language,
    should_skip_dir,
    should_skip_file,
)


def test_detect_language():
    assert detect_language("main.py") == "python"
    assert detect_language("index.ts") == "typescript"
    assert detect_language("app.go") == "go"
    assert detect_language("README.md") == "markdown"
    assert detect_language("Dockerfile") == "dockerfile"
    assert detect_language("unknown.xyz") is None


def test_should_skip_file():
    assert should_skip_file("bundle.min.js") is True
    assert should_skip_file("package-lock.json") is False
    assert should_skip_file("image.png") is True
    assert should_skip_file("main.py") is False


def test_should_skip_dir():
    assert should_skip_dir(".git") is True
    assert should_skip_dir("node_modules") is True
    assert should_skip_dir("src") is False


def test_chunk_small_file():
    content = "def hello():\n    print('hello')\n"
    chunks = chunk_code(content, "python")
    assert len(chunks) == 1
    assert chunks[0].chunk_type == "file"
    assert chunks[0].content == content


def test_chunk_large_file_python():
    functions = []
    for i in range(20):
        functions.append(f"def func_{i}(x):\n" + "\n".join(f"    line_{j} = {j}" for j in range(20)) + "\n\n")
    content = "\n".join(functions)
    chunks = chunk_code(content, "python")
    assert len(chunks) > 1
    assert any(c.chunk_type == "function" for c in chunks)


def test_chunk_empty_content():
    assert chunk_code("", "python") == []
    assert chunk_code("   \n  \n  ", "python") == []


def test_fallback_chunking():
    content = "line\n" * 5000
    chunks = chunk_code(content, None)
    assert len(chunks) > 1
    assert all(c.chunk_type == "file_section" for c in chunks)
