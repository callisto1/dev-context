from src.ingestion.confluence_ingest import _html_to_text, _chunk_text


def test_html_to_text_basic():
    html = "<p>Hello world</p>"
    assert _html_to_text(html).strip() == "Hello world"


def test_html_to_text_headings():
    html = "<h2>Section Title</h2><p>Some content here.</p>"
    result = _html_to_text(html)
    assert "## Section Title" in result
    assert "Some content here." in result


def test_html_to_text_lists():
    html = "<ul><li>Item one</li><li>Item two</li></ul>"
    result = _html_to_text(html)
    assert "- Item one" in result
    assert "- Item two" in result


def test_html_to_text_code_block():
    html = '<ac:plain-text-body><![CDATA[print("hello")]]></ac:plain-text-body>'
    result = _html_to_text(html)
    assert '```' in result
    assert 'print("hello")' in result


def test_html_to_text_strips_macros():
    html = '<ac:structured-macro ac:name="toc"><ac:parameter ac:name="maxLevel">3</ac:parameter></ac:structured-macro><p>Real content</p>'
    result = _html_to_text(html)
    assert "toc" not in result
    assert "Real content" in result


def test_html_to_text_unescapes():
    html = "<p>A &amp; B &lt; C</p>"
    result = _html_to_text(html)
    assert "A & B < C" in result


def test_html_to_text_empty():
    assert _html_to_text("") == ""
    assert _html_to_text(None) == ""


def test_chunk_text_small():
    text = "Short page content."
    chunks = _chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_large():
    text = "## Section\n\n" + ("word " * 600 + "\n\n") * 5
    chunks = _chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 3500  # allow some overhead from overlap


def test_chunk_text_respects_sections():
    sections = [f"## Section {i}\n\n{'Content ' * 200}\n\n" for i in range(5)]
    text = "".join(sections)
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    assert any("## Section" in c for c in chunks)
