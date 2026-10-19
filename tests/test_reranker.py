import pytest

from src.core.reranker import Reranker


def _make_results(n: int) -> list[dict]:
    return [
        {
            "id": str(i),
            "source_type": "git",
            "source_name": "test-repo",
            "file_path": f"file_{i}.py",
            "chunk_index": 0,
            "chunk_type": "function",
            "language": "python",
            "content": f"def function_{i}(): pass  # this is function number {i}",
            "similarity": 0.9 - (i * 0.02),
        }
        for i in range(n)
    ]


@pytest.mark.asyncio
async def test_reranker_disabled_returns_top_n():
    r = Reranker()
    r.enabled = False
    r.top_n = 3

    results = _make_results(10)
    reranked = await r.rerank("test query", results)

    assert len(reranked) == 3
    assert reranked[0]["id"] == "0"  # original order preserved


@pytest.mark.asyncio
async def test_reranker_disabled_empty_results():
    r = Reranker()
    r.enabled = False
    r.top_n = 5

    reranked = await r.rerank("query", [])
    assert reranked == []


@pytest.mark.asyncio
async def test_reranker_disabled_fewer_than_top_n():
    r = Reranker()
    r.enabled = False
    r.top_n = 10

    results = _make_results(3)
    reranked = await r.rerank("query", results)
    assert len(reranked) == 3


@pytest.mark.asyncio
async def test_reranker_local_adds_scores():
    """Integration test — only runs if the cross-encoder model can be loaded."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        pytest.skip("sentence-transformers not installed")

    r = Reranker()
    r.enabled = True
    r.mode = "local"
    r.top_n = 3

    results = _make_results(10)
    reranked = r._rerank_local("find function number 5", results)

    assert len(reranked) == 3
    assert all("rerank_score" in item for item in reranked)
    # The reranker should score the matching doc higher
    names = [item["content"] for item in reranked]
    assert any("function_5" in n for n in names)
