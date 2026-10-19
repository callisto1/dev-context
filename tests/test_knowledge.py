from pathlib import Path

from src.ingestion.knowledge_ingest import _flatten_sources, load_registry
from src.ingestion.web_docs_ingest import _chunk_text, _split_by_paragraphs

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "registry.yml"


def test_load_registry():
    registry = load_registry(REGISTRY_PATH)
    assert isinstance(registry, dict)
    assert "cncf_graduated" in registry
    assert "engineering_practices" in registry
    assert "rfcs_and_design" in registry


def test_flatten_sources():
    registry = load_registry(REGISTRY_PATH)
    sources = _flatten_sources(registry)
    assert len(sources) > 40
    for s in sources:
        assert "name" in s
        assert "kind" in s
        assert s["kind"] in ("git_docs", "web_docs")
        assert "category" in s


def test_flatten_sources_all_have_targets():
    registry = load_registry(REGISTRY_PATH)
    sources = _flatten_sources(registry)
    for s in sources:
        if s["kind"] == "git_docs":
            assert "repo" in s, f"{s['name']} missing repo"
        elif s["kind"] == "web_docs":
            assert "seed_url" in s, f"{s['name']} missing seed_url"


def test_cncf_graduated_count():
    registry = load_registry(REGISTRY_PATH)
    graduated = registry["cncf_graduated"]
    assert len(graduated) >= 20


def test_web_chunk_text_small():
    text = "Short document."
    chunks = _chunk_text(text)
    assert len(chunks) == 1


def test_web_chunk_text_large():
    text = "# Title\n\n" + ("paragraph text " * 200 + "\n\n") * 5
    chunks = _chunk_text(text)
    assert len(chunks) > 1


def test_split_by_paragraphs():
    text = "\n\n".join(["paragraph " * 100] * 5)
    chunks = _split_by_paragraphs(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 3500
